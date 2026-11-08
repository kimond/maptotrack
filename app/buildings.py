"""Buildings from the aerial photo (and the Quebec LiDAR canopy-height model when available).

1. Roof colours are learned from the buildings already mapped (OSM / Overture) against the ground
   around them -> a roof-probability map of the whole photo.
2. Mapped footprints are snapped onto the photo: Overture/OSM outlines are often 1-3 m off. A common
   shift is found first, then each building is refined within +-1.5 m.
3. New buildings:
   * LiDAR: regions >= 2.2 m tall with a smooth (roof-plane) surface.
   * Photo: roof-coloured blobs that cast a shadow in the same direction as the mapped buildings
     (that rejects driveways and parking lots with roof-like colours). This also finds houses
     built after the LiDAR survey.
   Every candidate must look like a building: compact, nearly rectangular, not too elongated.
   Rectangular blobs are turned into clean rotated rectangles.
4. Heights: LiDAR (75th percentile of the roof) where available, otherwise by footprint size.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import photo as photo_mod
from .parallel import pmap
from .roadwidth import rgb_to_lab


class _BoundsGrid(photo_mod.Grid):
    """Analysis grid when there is no photo."""

    def __init__(self, bounds, res):
        x0, y0, x1, y1 = bounds
        self.k, self.res = 1, res
        self.x0, self.y1 = x0, y1
        self.W = int(np.ceil((x1 - x0) / res))
        self.H = int(np.ceil((y1 - y0) / res))


def _eave_ridge(hv):
    """LiDAR heights over a roof -> (eave, ridge): a pitched roof spans eave..ridge."""
    hv = hv[hv > 1.5]
    if len(hv) == 0:
        return None
    eave = float(np.percentile(hv, 15))
    ridge = float(np.percentile(hv, 95))
    return (max(2.2, eave), max(ridge, eave + 0.3))


def default_height(area):
    if area < 40:
        return 3.0
    if area < 250:
        return 6.5
    if area < 800:
        return 9.0
    return 11.0


def _looks_like_building(p, min_area, max_area, min_rect=0.62):
    if not (min_area <= p.area <= max_area):
        return False
    rect = p.minimum_rotated_rectangle
    hull = p.convex_hull
    if rect.area <= 0 or hull.area <= 0:
        return False
    xy = np.asarray(rect.exterior.coords)
    e = np.linalg.norm(np.diff(xy[:3], axis=0), axis=1)
    aspect = max(e) / max(min(e), 0.1)
    return p.area / rect.area >= min_rect and p.area / hull.area >= 0.82 and aspect <= 4.0 and min(e) >= 3.0


def _regularise(p):
    """Mostly-rectangular blobs -> a clean rotated rectangle of the same area; others simplified."""
    from shapely import affinity

    rect = p.minimum_rotated_rectangle
    if rect.area > 0 and p.area / rect.area >= 0.7:
        f = float(np.sqrt(p.area / rect.area))
        return affinity.scale(rect, f, f, origin=rect.centroid)
    return p.simplify(0.8, preserve_topology=True)


def _blobs(grid, mask, min_area, max_area, min_rect=0.62, labels=None):
    """Connected blobs (or a given label image) -> [(polygon, label)] that pass the shape test."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    if labels is not None:
        lab, n = labels, int(labels.max())
    else:
        lab, n = ndi.label(mask)
    out = []
    if n == 0:
        return out, lab
    for geom, val in shapes(lab.astype(np.int32), mask=lab > 0, transform=grid.transform):
        p = shape(geom).buffer(0)
        if p.is_empty:
            continue
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
        if _looks_like_building(p, min_area, max_area, min_rect):
            out.append((_regularise(p), int(val)))
    return out, lab


def _grid_heights(chm, proj, grid):
    xs = grid.x0 + (np.arange(grid.W) + 0.5) * grid.res
    ys = grid.y1 - (np.arange(grid.H) + 0.5) * grid.res
    X, Y = np.meshgrid(xs, ys)
    lat, lon = proj.to_latlon(X, Y)
    h = chm.sample(lat.ravel(), lon.ravel()).reshape(X.shape)
    return np.nan_to_num(h, nan=0.0)


def _roof_probability(lab, veg, valid, grid, known_polys, erode):
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    q = np.stack([np.clip(L / 100 * 12, 0, 11), np.clip((A + 30) / 60 * 12, 0, 11),
                  np.clip((B + 30) / 60 * 12, 0, 11)], -1).astype(int)
    code = q[..., 0] * 144 + q[..., 1] * 12 + q[..., 2]
    roof = grid.mask_from_polys(known_polys, buffer=-erode) & valid
    ground = grid.mask_from_polys(known_polys, buffer=8.0) & ~grid.mask_from_polys(known_polys, buffer=2.0) & valid
    if roof.sum() < 150 or ground.sum() < 150:
        return None
    hr = np.bincount(code[roof], minlength=1728).astype(float) + 0.5
    hg = np.bincount(code[ground & ~veg], minlength=1728).astype(float) + 0.5
    hr /= hr.sum()
    hg /= hg.sum()
    p = hr[code] / (hr[code] + hg[code])
    p[veg] *= 0.3
    return ndi.gaussian_filter(p, 1.0)


def _align(evidence, grid, known_polys, max_common=4.0, max_local=1.5):
    """Shift each footprint onto the evidence map (roof probability / LiDAR height)."""
    from shapely import affinity

    res = grid.res
    steps = np.arange(-max_common, max_common + 1e-6, 0.5)
    masks = []
    for p in known_polys:
        minx, miny, maxx, maxy = p.bounds
        pad = max_common + max_local + 1
        c0 = int((minx - pad - grid.x0) / res)
        c1 = int((maxx + pad - grid.x0) / res) + 1
        r0 = int((grid.y1 - maxy - pad) / res)
        r1 = int((grid.y1 - miny + pad) / res) + 1
        if c0 < 0 or r0 < 0 or c1 > grid.W or r1 > grid.H:
            masks.append(None)
            continue
        m = grid.mask_from_polys([p])[r0:r1, c0:c1]
        ring = ndi.binary_dilation(m, iterations=max(2, int(round(1.5 / res)))) & ~m
        masks.append((m, ring, r0, r1, c0, c1))

    def score(i, dx, dy):
        # roof evidence inside the outline minus evidence just outside it (edge contrast)
        m, ring, r0, r1, c0, c1 = masks[i]
        sh = (-dy / res, dx / res)
        sm = ndi.shift(m.astype(np.float32), sh, order=0) > 0.5
        sr = ndi.shift(ring.astype(np.float32), sh, order=0) > 0.5
        if sm.sum() == 0 or sr.sum() == 0:
            return -1.0
        ev = evidence[r0:r1, c0:c1]
        return float(ev[sm].mean() - ev[sr].mean())

    ok = [i for i, m in enumerate(masks) if m is not None]
    if not ok:
        return list(known_polys), (0.0, 0.0)
    # common shift (sum over buildings), coarse grid
    sample = ok[:: max(1, len(ok) // 40)]
    combos = [(float(dx), float(dy)) for dx in steps for dy in steps]
    totals = pmap(lambda c: sum(score(i, c[0], c[1]) for i in sample), combos)
    best, bxy = -1, (0.0, 0.0)
    for c, tot in zip(combos, totals):                   # same order as the old nested loops
        if tot > best:
            best, bxy = tot, c
    out = list(known_polys)
    loc = np.arange(-max_local, max_local + 1e-6, 0.5)

    def refine(i):
        base = score(i, *bxy)
        bs, bl = base, bxy
        for ddx in loc:
            for ddy in loc:
                s = score(i, bxy[0] + ddx, bxy[1] + ddy)
                if s > bs + 0.02:
                    bs, bl = s, (bxy[0] + ddx, bxy[1] + ddy)
        return bl if bs > score(i, 0.0, 0.0) + 0.05 else None

    for i, bl in zip(ok, pmap(refine, ok)):
        if bl is not None:
            out[i] = affinity.translate(known_polys[i], bl[0], bl[1])
    return out, bxy


def _shadow_direction(L, known_mask, res):
    km = known_mask.astype(np.float32)

    def contrast(ang):
        dr, dc = -np.sin(ang), np.cos(ang)
        sh = ndi.shift(km, (dr * 2.0 / res, dc * 2.0 / res), order=0) > 0.5
        opp = ndi.shift(km, (-dr * 2.0 / res, -dc * 2.0 / res), order=0) > 0.5
        band, oband = sh & ~known_mask, opp & ~known_mask
        if band.sum() < 50 or oband.sum() < 50:
            return None
        return float(L[oband].mean() - L[band].mean()), (dr, dc)

    best, best_d = None, 0.0
    for r in pmap(contrast, list(np.radians(np.arange(0, 360, 22.5)))):
        if r is not None and r[0] > best_d:
            best_d, best = r
    return best, best_d


def _has_shadow(m, L, direction, strength, res):
    if direction is None or strength < 4:
        return True
    dr, dc = direction
    sh = ndi.shift(m.astype(np.float32), (dr * 2.0 / res, dc * 2.0 / res), order=0) > 0.5
    opp = ndi.shift(m.astype(np.float32), (-dr * 2.0 / res, -dc * 2.0 / res), order=0) > 0.5
    band, oband = sh & ~m, opp & ~m
    if band.sum() < 8 or oband.sum() < 8:
        return False
    return L[oband].mean() - L[band].mean() >= max(4.0, strength * 0.35)


def _split_cores(raw, it, res, grow_m=2.0):
    """Erode to cores (cuts thin links: driveways, shadows, neighbours), then grow each core back
    without merging, up to grow_m inside the original mask. Returns a label image."""
    core = ndi.binary_opening(raw, iterations=3 * it)
    core_lab, _ = ndi.label(core)
    if core_lab.max() == 0:
        return core_lab
    dist, (ir, ic) = ndi.distance_transform_edt(core_lab == 0, return_indices=True)
    return np.where(raw & (dist * res <= grow_m), core_lab[ir, ic], 0)


def _bootstrap_roofs(lab, veg, valid, exclude, grid):
    """First guess without any mapped building: dark or reddish, smooth, rectangular blobs that
    cast a shadow in a direction shared by most of them."""
    res = grid.res
    L, A = lab[..., 0], lab[..., 1]
    win = max(3, int(round(1.5 / res)))
    m1 = ndi.uniform_filter(L, win)
    tex = np.sqrt(np.clip(ndi.uniform_filter(L * L, win) - m1 * m1, 0, None))
    it = max(1, int(round(0.8 / res)))
    raw = valid & ~veg & ~exclude & ((L < 42) | (A > 8)) & (tex < 9)
    raw = ndi.binary_fill_holes(ndi.binary_closing(raw, iterations=it))
    lab_img = _split_cores(raw, it, res)
    blobs, labimg = _blobs(grid, lab_img > 0, 40.0, 600.0, min_rect=0.75, labels=lab_img)
    if len(blobs) < 5:
        return []
    masks = [labimg == i for _, i in blobs]

    def kept(ang):
        dr, dc = -np.sin(ang), np.cos(ang)
        keep = []
        for (p, i), m in zip(blobs, masks):
            sh = ndi.shift(m.astype(np.float32), (dr * 2.0 / res, dc * 2.0 / res), order=0) > 0.5
            opp = ndi.shift(m.astype(np.float32), (-dr * 2.0 / res, -dc * 2.0 / res), order=0) > 0.5
            band, oband = sh & ~m, opp & ~m
            if band.sum() > 8 and oband.sum() > 8 and L[oband].mean() - L[band].mean() > 6:
                keep.append(p)
        return keep

    best, best_keep = 0, []
    for keep in pmap(kept, list(np.radians(np.arange(0, 360, 22.5)))):
        if len(keep) > best:
            best, best_keep = len(keep), keep
    return best_keep


def detect(pm, chm, proj, bounds, known_polys, corridor, street_polys, log=print,
           min_area=25.0, max_area=3000.0, train_hint=None, res=0.5):
    """Returns dict with
         new:      [(polygon, height|None, source)]
         known:    aligned footprints (same order as known_polys)
         known_h:  LiDAR height per known footprint (or None)
         shift:    common shift applied to the mapped footprints (m)"""
    grid = photo_mod.Grid(pm, res) if pm is not None else _BoundsGrid(bounds, res)
    exclude = grid.mask_from_polys([corridor] + list(street_polys))
    result = {"new": [], "known": list(known_polys), "known_h": [None] * len(known_polys), "shift": (0.0, 0.0)}

    lab = veg = valid = P = None
    if pm is not None:
        k = grid.k
        rgb = pm.rgb[:grid.H * k, :grid.W * k].reshape(grid.H, k, grid.W, k, 3).mean((1, 3))
        valid = pm.valid[:grid.H * k, :grid.W * k].reshape(grid.H, k, grid.W, k).all((1, 3))
        lab = rgb_to_lab(rgb)
        veg = (photo_mod.features(rgb)["exg"] > 0.045) & valid
    Hc = _grid_heights(chm, proj, grid) if chm is not None else None

    # ---- snap mapped footprints onto the photo / LiDAR
    if known_polys and (pm is not None or Hc is not None):
        if pm is not None:
            P = _roof_probability(lab, veg, valid, grid, known_polys, erode=1.5)
        ev = None
        if P is not None:
            ev = P.copy()
        if Hc is not None:
            tall = ((Hc > 2.2).astype(np.float32))
            ev = tall if ev is None else 0.5 * ev + 0.5 * tall
        if ev is not None:
            aligned, shift = _align(ev, grid, known_polys)
            result["known"], result["shift"] = aligned, shift
            if pm is not None:   # retrain on the aligned outlines
                P2 = _roof_probability(lab, veg, valid, grid, aligned, erode=0.6)
                P = P2 if P2 is not None else P

    known_now = result["known"]
    known_mask = grid.mask_from_polys(known_now, buffer=1.5) if known_now else np.zeros((grid.H, grid.W), bool)
    taken = exclude | known_mask
    res = grid.res
    it = max(1, int(round(0.8 / res)))

    # ---- LiDAR candidates
    if Hc is not None:
        win = max(3, int(round(1.5 / res)))
        m1 = ndi.uniform_filter(Hc, win)
        rough = np.sqrt(np.clip(ndi.uniform_filter(Hc * Hc, win) - m1 * m1, 0, None))
        cand = (Hc >= 2.2) & (rough < 0.9) & ~taken
        cand = ndi.binary_fill_holes(ndi.binary_closing(ndi.binary_opening(cand, iterations=it), iterations=it))
        # LiDAR roofs are crisp rectangles; round blobs are tree crowns
        blobs, labimg = _blobs(grid, cand, min_area, max_area, min_rect=0.72)

        def judge(blob, P=P):
            p, i = blob
            m = labimg == i
            if veg is not None and veg[m].mean() > 0.3:
                return None   # trees (or a lot cleared since the survey is now green/sand - checked below)
            if P is not None and P[m].mean() < 0.3:
                return None   # nothing roof-like on the photo any more (demolished / cleared)
            if p.area < 40 and P is not None and P[m].mean() < 0.5:
                return None   # small and not clearly a roof: most likely a tree crown
            h = _eave_ridge(Hc[m])
            if h and p.area < 80 and h[0] > 7.0:
                return None   # small and tall: a tree crown, not a house
            return (p, h, "lidar"), i

        accepted = [r for r in pmap(judge, blobs) if r is not None]
        for found, _ in accepted:
            result["new"].append(found)
        if accepted:
            taken |= ndi.binary_dilation(np.isin(labimg, [i for _, i in accepted]), iterations=it)

    # ---- no (or too few) mapped buildings: learn the roof colours from LiDAR finds or a first guess
    train = known_now
    if pm is not None and P is None:
        lidar_found = [p for p, _, src in result["new"] if src == "lidar"]
        if len(lidar_found) >= 5:
            train = lidar_found
            log(f"  learning roof colours from {len(train)} LiDAR buildings")
        elif train_hint and len(train_hint) >= 5:
            train = list(train_hint)
            log(f"  learning roof colours from {len(train)} Overture footprints")
        else:
            train = _bootstrap_roofs(lab, veg, valid, exclude, grid)
            if len(train) >= 5:
                log(f"  learning roof colours from {len(train)} obvious roofs (no mapped buildings here)")
        if len(train) >= 5:
            P = _roof_probability(lab, veg, valid, grid, train, erode=0.6)

    # ---- photo candidates (also finds houses built after the LiDAR survey)
    if P is not None:
        direction, strength = _shadow_direction(lab[..., 0], grid.mask_from_polys(train), res)
        raw = (P > 0.6) & valid & ~veg & ~taken
        raw = ndi.binary_fill_holes(ndi.binary_closing(raw, iterations=it))
        # split roofs from attached driveways / neighbours: erode to cores, then grow each core back
        # (without merging) up to 2 m inside the roof-coloured area
        grown = _split_cores(raw, it, res)
        cand = grown > 0
        blobs, labimg = _blobs(grid, cand, min_area, max_area, min_rect=0.66, labels=grown)
        # blobs that failed the shape test usually include a driveway or a neighbour's lot:
        # retry inside them with stricter roof-probability thresholds
        passed = {i for _, i in blobs}
        nxt = int(labimg.max()) + 1
        extra = np.zeros_like(labimg)

        def retry(i, P=P):
            region = grown == i
            if region.sum() * res * res < min_area:
                return None
            for thr in (0.7, 0.8):
                sub = region & (P > thr)
                sub = ndi.binary_fill_holes(ndi.binary_opening(sub, iterations=it))
                sl, sn = ndi.label(sub)
                if sn == 0:
                    return None
                sizes = ndi.sum(sub, sl, range(1, sn + 1))
                big = sl == (int(np.argmax(sizes)) + 1)
                b2, _ = _blobs(grid, big, min_area, max_area, min_rect=0.66)
                if b2:
                    return big, b2[0][0]
            return None

        failed = [i for i in range(1, int(grown.max()) + 1) if i not in passed]
        for r in pmap(retry, failed):                     # in label order, as before
            if r is not None:
                big, poly = r
                extra[big] = nxt
                blobs.append((poly, nxt))
                nxt += 1
        labimg = np.where(extra > 0, extra, labimg)
        def check(blob):
            p, i = blob
            m = labimg == i
            if not _has_shadow(m, lab[..., 0], direction, strength, res):
                return None
            h = None
            if Hc is not None:
                hv = Hc[m]
                if (hv > 2.2).mean() > 0.5:
                    h = _eave_ridge(hv)
                    # the survey is older than the photo: tall/ragged returns there are the trees that
                    # stood before the house was built, not the house
                    if h and (h[0] > 7.5 or h[1] - h[0] > 6.0):
                        h = None
            return p, h, "photo"

        n_photo = 0
        for r in pmap(check, blobs):
            if r is not None:
                result["new"].append(r)
                n_photo += 1
    elif pm is not None:
        log("  photo buildings: could not learn roof colours here (no mapped or obvious roofs)")

    # ---- heights of mapped buildings from LiDAR
    if Hc is not None:
        def known_height(kp):
            m = grid.mask_from_polys([kp], buffer=-0.5)
            if m.sum() >= 8:
                hv = Hc[m]
                if (hv > 1.5).mean() >= 0.4:
                    h = _eave_ridge(hv)
                    if h and not (kp.area < 300 and h[0] > 12.0):
                        return h
            return None

        result["known_h"] = pmap(known_height, known_now)
    return result

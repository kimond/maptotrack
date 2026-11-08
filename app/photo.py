"""Read the aerial photo: vegetation / tree crowns and paved areas (road width: roadwidth.py).

Everything works on a photo mosaic resampled into the local track frame (x east, y north),
row 0 = north edge. Classic colour/texture analysis - no GPU or AI model needed.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates

from . import imagery


class PhotoMap:
    def __init__(self, rgb: np.ndarray, valid: np.ndarray, x0: float, y1: float, res: float):
        self.rgb, self.valid, self.x0, self.y1, self.res = rgb, valid, x0, y1, res
        self.h, self.w = valid.shape

    # pixel centre of column i = x0 + (i + 0.5) * res
    def _rc(self, x, y):
        return (self.y1 - np.asarray(y)) / self.res - 0.5, (np.asarray(x) - self.x0) / self.res - 0.5

    def sample(self, x, y, order=1):
        r, c = self._rc(x, y)
        out = np.stack([map_coordinates(self.rgb[..., k], [r, c], order=order, mode="nearest").astype(np.float32)
                        for k in range(3)], axis=-1)
        ok = map_coordinates(self.valid.astype(np.uint8), [r, c], order=0, mode="constant", cval=0).astype(bool)
        return out, ok

    def grid(self, xs, ys):
        X, Y = np.meshgrid(xs, ys)
        rgb, ok = self.sample(X.ravel(), Y.ravel())
        return rgb.reshape(X.shape + (3,)), ok.reshape(X.shape)


def build_photomap(src, proj, bounds, res=0.25, chunk_m=900.0, log=print) -> PhotoMap:
    x0, y0, x1, y1 = bounds
    W = int(np.ceil((x1 - x0) / res))
    H = int(np.ceil((y1 - y0) / res))
    rgb = np.zeros((H, W, 3), np.uint8)
    valid = np.zeros((H, W), bool)
    from .parallel import io_map

    step = int(chunk_m / res)
    chunks = [(r0, c0) for r0 in range(0, H, step) for c0 in range(0, W, step)]
    log(f"  aerial photo: {len(chunks)} piece(s) ...")

    def fetch(rc):                         # pieces download in parallel, each writes its own window
        r0, c0 = rc
        r1, c1 = min(r0 + step, H), min(c0 + step, W)
        xs = x0 + (np.arange(c0, c1) + 0.5) * res
        ys = y1 - (np.arange(r0, r1) + 0.5) * res
        img, ok = imagery.ground_image(src, proj, xs, ys)
        rgb[r0:r1, c0:c1] = np.clip(img, 0, 255).astype(np.uint8)
        valid[r0:r1, c0:c1] = ok

    io_map(fetch, chunks, threads=4)
    return PhotoMap(rgb, valid, x0, y1, res)


CELL_DEG = (0.008, 0.012)        # (lat, lon) degrees: ~900 m photo pieces on a fixed grid (disk cache hits)


def corridor_photomap(src, proj, cl, reach=18.0, res=0.25, log=print) -> PhotoMap:
    """Aerial photo only along the track, within `reach` m of the centreline (what the road width needs):
    the road preview. Pieces are fixed lat/lon cells, so moving a waypoint re-uses the cached pieces."""
    from scipy.spatial import cKDTree

    from .parallel import io_map
    from .track import resample

    x0, y0 = cl.p.min(0) - reach - 2
    x1, y1 = cl.p.max(0) + reach + 2
    W, H = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    # which pixels are needed: a coarse distance test, then the exact one
    tree = cKDTree(resample(cl.p, 1.0, cl.loop))
    cs = 2.0
    cw, ch = int(np.ceil((x1 - x0) / cs)), int(np.ceil((y1 - y0) / cs))
    CX, CY = np.meshgrid(x0 + (np.arange(cw) + 0.5) * cs, y1 - (np.arange(ch) + 0.5) * cs)
    d, _ = tree.query(np.column_stack([CX.ravel(), CY.ravel()]))
    coarse = (d.reshape(ch, cw) < reach + 2.0)
    k = int(round(cs / res))
    need = np.repeat(np.repeat(coarse, k, 0), k, 1)[:H, :W]
    rr, cc = np.nonzero(need)
    xs = x0 + (cc + 0.5) * res
    ys = y1 - (rr + 0.5) * res
    lat, lon = proj.to_latlon(xs, ys)
    ci = np.floor(lat / CELL_DEG[0]).astype(np.int64)
    cj = np.floor(lon / CELL_DEG[1]).astype(np.int64)
    cells = sorted(set(zip(ci.tolist(), cj.tolist())))
    log(f"  aerial photo along the track: {len(cells)} piece(s) ...")
    rgb = np.zeros((H, W, 3), np.uint8)
    valid = np.zeros((H, W), bool)
    pad = 0.0003

    def fetch(cell):
        a, b = cell
        sel = (ci == a) & (cj == b)
        r = imagery.fetch_raster(src, a * CELL_DEG[0] - pad, b * CELL_DEG[1] - pad,
                                 (a + 1) * CELL_DEG[0] + pad, (b + 1) * CELL_DEG[1] + pad, res)
        px = r.sample(lat[sel], lon[sel])
        mx, mn = px.max(-1), px.min(-1)
        rgb[rr[sel], cc[sel]] = np.clip(px, 0, 255).astype(np.uint8)
        valid[rr[sel], cc[sel]] = ~((mn > 248) | (mx < 4))

    io_map(fetch, cells, threads=4)
    return PhotoMap(rgb, valid, float(x0), float(y1), res)


class TiledPhoto:
    """The aerial photo sampled on demand: fixed lat/lon pieces (CELL_DEG, ~900 m, the same as the road preview's,
    so they come from the disk cache) loaded when a point needs them and kept in a small cache. Measures the
    width and markings of every street of a big area at full resolution without one huge image in memory.
    Only `sample` (and `res`) of PhotoMap: what roadwidth and markings use."""

    def __init__(self, src, proj, res=0.25, max_cells=12):
        import threading
        from collections import OrderedDict

        self.src, self.proj, self.res, self.max_cells = src, proj, res, max_cells
        self._cells = OrderedDict()
        self._lock = threading.Lock()
        self._loading = {}
        self.loaded = 0

    def _cell(self, key):
        import threading

        with self._lock:
            if key in self._cells:
                self._cells.move_to_end(key)
                return self._cells[key]
            ev = self._loading.get(key)
            mine = ev is None
            if mine:
                ev = self._loading[key] = threading.Event()
        if not mine:
            ev.wait()
            with self._lock:
                if key in self._cells:
                    return self._cells[key]
            return self._cell(key)
        a, b = key
        pad = 0.0003
        try:
            r = imagery.fetch_raster(self.src, a * CELL_DEG[0] - pad, b * CELL_DEG[1] - pad,
                                     (a + 1) * CELL_DEG[0] + pad, (b + 1) * CELL_DEG[1] + pad, self.res)
            chans = [np.ascontiguousarray(r.img[..., c]) for c in range(3)]
            cell = (r, chans)
        except Exception:
            cell = None
        with self._lock:
            self._cells[key] = cell
            self.loaded += 1
            while len(self._cells) > self.max_cells:
                self._cells.popitem(last=False)
            self._loading.pop(key, None)
        ev.set()
        return cell

    def sample(self, x, y, order=1):
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        out = np.zeros((len(x), 3), np.float32)
        ok = np.zeros(len(x), bool)
        if not len(x):
            return out, ok
        lat, lon = self.proj.to_latlon(x, y)
        ci = np.floor(lat / CELL_DEG[0]).astype(np.int64)
        cj = np.floor(lon / CELL_DEG[1]).astype(np.int64)
        keys = np.stack([ci, cj], 1)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel()
        for u, (a, b) in enumerate(uniq):
            sel = inv == u
            cell = self._cell((int(a), int(b)))
            if cell is None:
                continue
            r, chans = cell
            mx, my = imagery._merc(lat[sel], lon[sel])
            col = (mx - r.mx0) / r.rx - 0.5
            row = (r.my1 - my) / r.ry - 0.5
            px = np.stack([map_coordinates(ch, [row, col], order=order, mode="nearest", output=np.float32)
                           for ch in chans], -1)
            out[sel] = px
            mxv, mnv = px.max(-1), px.min(-1)
            ok[sel] = ~((mnv > 248) | (mxv < 4))
        return out, ok


# ------------------------------------------------------------------ colour features
def features(rgb):
    """rgb float/uint8 (...,3) -> dict of value, saturation, hue(deg), excess-green, blue excess."""
    f = np.asarray(rgb, np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx, mn = f.max(-1), f.min(-1)
    v = mx
    s = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
    tot = r + g + b + 1e-6
    exg = (2 * g - r - b) / tot
    d = mx - mn + 1e-6
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return {"v": v, "s": s, "h": hue, "exg": exg, "r": r, "g": g, "b": b}


class Grid:
    """A coarser analysis grid aligned with a PhotoMap (default 0.5 m)."""

    def __init__(self, pm: PhotoMap, res=0.5):
        self.k = max(1, int(round(res / pm.res)))
        self.res = pm.res * self.k
        self.H, self.W = pm.h // self.k, pm.w // self.k
        self.x0, self.y1 = pm.x0, pm.y1

    @property
    def transform(self):
        from rasterio.transform import Affine

        return Affine(self.res, 0, self.x0, 0, -self.res, self.y1)

    def xy_of(self, rows, cols):
        return self.x0 + (np.asarray(cols) + 0.5) * self.res, self.y1 - (np.asarray(rows) + 0.5) * self.res

    def rc_of(self, x, y):
        c = np.clip(((np.asarray(x) - self.x0) / self.res).astype(int), 0, self.W - 1)
        r = np.clip(((self.y1 - np.asarray(y)) / self.res).astype(int), 0, self.H - 1)
        return r, c

    def mask_from_polys(self, polys, buffer=0.0):
        """Rasterise shapely polygons (local coords) onto the grid."""
        from rasterio.features import rasterize

        geoms = [p.buffer(buffer) if buffer else p for p in polys if not p.is_empty]
        if not geoms:
            return np.zeros((self.H, self.W), bool)
        return rasterize(geoms, out_shape=(self.H, self.W), transform=self.transform, fill=0,
                         default_value=1, dtype="uint8").astype(bool)

    def polygons(self, mask, min_area, max_area, simplify=0.3):
        from rasterio.features import shapes
        from shapely.geometry import shape

        out = []
        if not mask.any():
            return out
        for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=self.transform):
            p = shape(geom)
            if min_area <= p.area <= max_area:
                p = p.simplify(simplify, preserve_topology=True).buffer(0)
                if not p.is_empty and p.geom_type == "Polygon":
                    out.append(p)
        return out


# Tree-vs-ground logistic model on photo features (0.5 m grid), fitted against the Quebec LiDAR
# canopy-height model on the Saguenay test area (LiDAR > 2 m vs < 0.3 m; ~83 % per pixel after
# smoothing). Colours are taken relative to the scene's typical vegetation, so the model does not depend
# on the season or the photo's colour balance. Features: lightness L and yellowness b (CIE Lab) minus
# the scene's vegetation median, brightness std over 1.5 m and 3 m, and brightness relative to the
# surroundings (15 m). Trees are darker, bluer and more textured than grass.
TREE_MODEL = {
    "sd": [12.9785, 4.3411, 2.1994, 2.8634, 7.1534],
    "coef": [-2.2365, -0.3899, 0.017, 0.4325, 1.0849],
    "b": 0.0003,
    "fallback_median": [46.6, 6.3],     # L, b of vegetation when the scene has too little to measure
}


def _local_std(x, w):
    m1 = ndi.uniform_filter(x, w)
    m2 = ndi.uniform_filter(x * x, w)
    return np.sqrt(np.clip(m2 - m1 * m1, 0, None))


class Analysis(Grid):
    """Vegetation classes and paved areas on the analysis grid.

    canopy: trees, bushes and hedges (LiDAR height checked against the photo when available)
    lawn:   smooth grass (mowed)               -> short 3D grass tufts
    wild:   textured / dry grass, hay fields   -> tall grass clumps
    paved:  driveways and parking lots
    """

    def __init__(self, pm: PhotoMap, res=0.5, block_mask=None, street_mask=None, chm_grid=None):
        from .roadwidth import rgb_to_lab

        super().__init__(pm, res)
        k, H, W = self.k, self.H, self.W
        rgb = pm.rgb[:H * k, :W * k].reshape(H, k, W, k, 3).mean((1, 3)).astype(np.float32)
        self.rgb = rgb
        self.valid = pm.valid[:H * k, :W * k].reshape(H, k, W, k).all((1, 3))
        valid = self.valid
        F = features(rgb)
        exg = F["exg"]
        lab = rgb_to_lab(rgb)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        t15 = _local_std(L, max(3, int(round(1.5 / self.res))))
        t3 = _local_std(L, max(3, int(round(3.0 / self.res))))
        rel = L - ndi.uniform_filter(L, max(3, int(round(15.0 / self.res))) | 1)
        m = TREE_MODEL
        vegish = (exg > 0.0) & valid
        if vegish.sum() > 2000:
            medL, medb = float(np.median(L[vegish])), float(np.median(b[vegish]))
        else:
            medL, medb = m["fallback_median"]
        X = np.stack([L - medL, b - medb, t15, t3, rel], -1)
        z = (X / np.array(m["sd"])) @ np.array(m["coef"]) + m["b"]
        p = ndi.gaussian_filter(1.0 / (1.0 + np.exp(-z)), max(1.0, 1.0 / self.res))
        self.p_tree = p
        block = block_mask if block_mask is not None else np.zeros_like(valid)
        street = street_mask if street_mask is not None else np.zeros_like(valid)
        self.block, self.street = block, street
        pool = (b < -6) & (a < -4) & (L > 45)                      # turquoise water
        self.pool = pool
        self.chm = chm_grid
        if chm_grid is not None:
            hc = np.nan_to_num(chm_grid, nan=-1.0)
            has = np.isfinite(chm_grid)
            # LiDAR is older than the photo: a tall return only counts if the photo still looks like
            # a tree there (cut trees, new houses, power lines over the road are dropped)
            canopy = np.where(has, (hc > 1.5) & (p > 0.3), p > 0.6)
        else:
            hc, has = None, np.zeros_like(valid)
            canopy = p > 0.6
        canopy &= valid & ~block & ~street & ~pool
        canopy = ndi.binary_closing(ndi.binary_opening(canopy, iterations=1), iterations=1) & ~block & ~street
        self.canopy = canopy
        self.tree = canopy                                         # (older name)
        # grass: green, or dry (yellow-olive, not bright sand / reddish soil)
        veg = ((exg > 0.02) | (a < -2.5)) & ~pool
        dry = ~veg & (b > 3) & (a > -4) & (a < 5) & (L > 35) & (L < 66) & (exg > -0.015) & ~pool
        self.paved = self._driveways(rgb, valid, exg, block_mask, street_mask)
        grassy = (veg | dry) & valid & ~canopy & ~block & ~street & ~self.paved
        smooth = ndi.gaussian_filter(t15, 2.0 / self.res * 0.5) < 2.2
        lawn = grassy & smooth
        wild = grassy & ~smooth
        if hc is not None:
            wild |= grassy & has & (hc > 0.35) & (hc <= 1.5)      # knee-high vegetation in the LiDAR
        lawn &= ~wild
        # big smooth grass areas away from houses are hay / meadow, not lawn
        if lawn.any():
            lab_, n = ndi.label(lawn)
            ids = np.arange(1, n + 1)
            area = ndi.sum(lawn, lab_, ids) * self.res ** 2
            far = ndi.distance_transform_edt(~block) * self.res > 25 if block.any() else np.ones_like(lawn)
            farfrac = ndi.mean(far, lab_, ids)
            field = np.isin(lab_, ids[(area > 2500) & (farfrac > 0.6)])
            wild |= field
            lawn &= ~field
        self.lawn = ndi.binary_opening(lawn, iterations=1)
        self.wild = ndi.binary_opening(wild, iterations=1)
        self.dry = dry                                             # for picking dry vs green grass cards
        self.L, self.a, self.b = L, a, b
        # conifers: darker and bluer crowns than broadleaf trees
        if canopy.sum() > 50:
            mL, sL = np.median(L[canopy]), L[canopy].std() + 1e-3
            mb, sb = np.median(b[canopy]), b[canopy].std() + 1e-3
            self.conifer_like = (-(L - mL) / sL - (b - mb) / sb) > 0
        else:
            self.conifer_like = np.zeros_like(canopy)

    def _driveways(self, rgb, valid, exg, block_mask, street_mask):
        """Driveways / parking lots: neutral-grey (chroma < 4), very smooth surfaces that touch a
        street or a building. Sand (yellow), dry grass, tree shadows (textured) and gravel shoulders
        running along the road are rejected."""
        from rasterio.features import shapes
        from shapely.geometry import shape

        from .roadwidth import rgb_to_lab

        res = self.res
        lab = rgb_to_lab(rgb)
        L = lab[..., 0]
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        m1 = ndi.uniform_filter(L, 3)
        tex = np.sqrt(np.clip(ndi.uniform_filter(L * L, 3) - m1 * m1, 0, None))
        street = street_mask if street_mask is not None else np.zeros_like(valid)
        block = block_mask if block_mask is not None else np.zeros_like(valid)
        cand = valid & (exg < 0.03) & (chroma < 4.0) & (tex < 1.5) & (L > 35) & (L < 78) & ~street & ~block
        cand = ndi.binary_closing(ndi.binary_opening(cand, iterations=1), iterations=2) & ~street & ~block
        lab_, n = ndi.label(cand)
        if n == 0:
            return cand
        ids = np.arange(1, n + 1)
        area = ndi.sum(cand, lab_, ids) * res * res
        touch_s = ndi.maximum(ndi.binary_dilation(street, iterations=max(1, int(2.0 / res))), lab_, ids) > 0
        touch_b = ndi.maximum(ndi.binary_dilation(block, iterations=max(1, int(2.0 / res))), lab_, ids) > 0
        near3 = ndi.binary_dilation(street, iterations=max(1, int(3.0 / res)))
        frac_near = ndi.mean(near3.astype(float), lab_, ids)
        keep = set(ids[(area >= 12) & (area <= 6000) & (touch_s | touch_b)
                       & ~((frac_near > 0.75) & ~touch_b)].tolist())
        # long thin strips running along a street are gravel shoulders, not driveways
        dist = ndi.gaussian_filter(ndi.distance_transform_edt(~street), 2)
        gy, gx = np.gradient(dist)
        out = np.zeros_like(cand)
        for geom, val in shapes(lab_.astype(np.int32), mask=np.isin(lab_, list(keep)), transform=self.transform):
            p = shape(geom)
            rect = p.minimum_rotated_rectangle
            xy = np.asarray(rect.exterior.coords)
            e = np.diff(xy[:3], axis=0)
            ln = np.linalg.norm(e, axis=1)
            if ln.min() <= 0:
                continue
            long_v = e[np.argmax(ln)] / ln.max()
            r, c = self.rc_of(p.centroid.x, p.centroid.y)
            gv = np.array([gx[r, c], -gy[r, c]])
            gv /= np.linalg.norm(gv) + 1e-9
            if ln.max() / ln.min() > 2.0 and abs(float(long_v @ gv)) < 0.5 and ln.min() < 4.0:
                continue
            out |= lab_ == int(val)
        return out


# ------------------------------------------------------------------ tree crowns
def _thin(x, y, spacing):
    """Greedy thinning: keep points at least `spacing` apart."""
    from scipy.spatial import cKDTree

    if len(x) == 0:
        return np.zeros(0, int)
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    taken = np.zeros(len(pts), bool)
    keep = []
    for i in range(len(pts)):
        if taken[i]:
            continue
        keep.append(i)
        taken[tree.query_ball_point(pts[i], spacing)] = True
    return np.asarray(keep, int)


def hedges(an: Analysis, rng=None):
    """Long, narrow canopy strips (hedges, lines of shrubs) -> (bush dict, mask of those strips)."""
    rng = rng or np.random.default_rng(12)
    canopy = an.canopy
    empty = {k: np.zeros(0) for k in ("x", "y", "h", "w", "kind")}
    if not canopy.any():
        return empty, np.zeros_like(canopy)
    dist = ndi.distance_transform_edt(canopy) * an.res
    ridge = canopy & (dist >= ndi.maximum_filter(dist, size=3) - 1e-6)
    lab_, n = ndi.label(canopy)
    ids = np.arange(1, n + 1)
    area = ndi.sum(canopy, lab_, ids) * an.res ** 2
    width = 2 * ndi.mean(dist, lab_ * ridge, ids)              # mean width along the medial line
    width = np.nan_to_num(width, nan=0.0) + an.res
    length = area / np.maximum(width, 0.1)
    hmed = np.full(n, np.nan)
    if an.chm is not None:
        hmed = np.asarray(ndi.median(np.nan_to_num(an.chm, nan=0.0), lab_, ids))
    is_hedge = (width < 3.5) & (length > 6.0) & (length / np.maximum(width, 0.1) > 3.5) & ~(hmed > 5.0)
    hmask = np.isin(lab_, ids[is_hedge])
    if not is_hedge.any():
        return empty, hmask
    rr, cc = np.nonzero(ridge & hmask)
    x, y = an.xy_of(rr, cc)
    keep = _thin(x, y, 1.0)
    rr, cc, x, y = rr[keep], cc[keep], x[keep], y[keep]
    comp = lab_[rr, cc] - 1
    w = np.clip(width[comp] * 1.25, 1.0, 3.5) * rng.uniform(0.9, 1.15, len(x))
    h = np.where(np.isfinite(hmed[comp]) & (hmed[comp] > 0.5), np.clip(hmed[comp], 0.8, 4.0), 1.8)
    h = h * rng.uniform(0.9, 1.1, len(x))
    return {"x": x, "y": y, "h": h, "w": w, "kind": np.full(len(x), 2)}, hmask


def trees_from_canopy(an: Analysis, tops=None, rng=None):
    """Every tree, bush and hedge of the canopy mask.

    tops: LiDAR tree tops (x, y, h) -> exact positions/heights, kept only where the photo still shows
    canopy. Without LiDAR, crown centres come from the shape of the canopy patches."""
    rng = rng or np.random.default_rng(7)
    hed, hmask = hedges(an, rng)
    rest = an.canopy & ~hmask
    edge = ndi.distance_transform_edt(rest) * an.res                 # distance to the crown's edge
    if tops is not None and len(tops[0]):
        tx, ty, th = (np.asarray(v) for v in tops)
        r_, c_ = an.rc_of(tx, ty)
        inside = ndi.binary_dilation(rest, iterations=1)[r_, c_]
        tx, ty, th, r_, c_ = tx[inside], ty[inside], th[inside], r_[inside], c_[inside]
        t = trees_from_treetops(tx, ty, th, an.conifer_like[r_, c_], rng)
        t["w"] = np.minimum(t["w"], 2 * (edge[r_, c_] + 0.8))     # crowns stay inside the canopy patch
        t["w"] = np.maximum(t["w"], 0.8)
        if an.chm is not None:                       # canopy outside the LiDAR coverage: from the photo
            gap = rest & ~np.isfinite(an.chm)
            if gap.sum() > 20:
                sub = Analysis.__new__(Analysis)
                sub.__dict__.update(an.__dict__)
                sub.tree = gap
                t = merge_dicts([t, crowns_to_trees(crowns_from_photo(sub, rng), None, rng)])
    else:
        sub = Analysis.__new__(Analysis)
        sub.__dict__.update(an.__dict__)
        sub.tree = rest
        t = crowns_to_trees(crowns_from_photo(sub, rng), None, rng)
    return merge_dicts([t, hed]), {"trees": int(len(t["x"])), "hedge_bushes": int(len(hed["x"]))}


def merge_dicts(ds):
    keys = ("x", "y", "h", "w", "kind")
    return {k: np.concatenate([np.asarray(d[k], float if k != "kind" else int) for d in ds]) for k in keys}


# ------------------------------------------------------------------ tree crowns
def crowns_from_photo(an: Analysis, rng=None):
    """Tree/bush positions from the tree mask: peaks of the distance transform = crown centres."""
    rng = rng or np.random.default_rng(4)
    if not an.tree.any():
        return {"x": np.zeros(0), "y": np.zeros(0), "r": np.zeros(0), "conifer": np.zeros(0, bool)}
    dist = ndi.distance_transform_edt(an.tree) * an.res
    dist = ndi.gaussian_filter(dist, 1.0)
    size = max(3, int(round(2.6 / an.res)))
    peaks = (dist == ndi.maximum_filter(dist, size=size)) & (dist >= 0.7)
    rows, cols = np.nonzero(peaks)
    x, y = an.xy_of(rows, cols)
    r = dist[rows, cols] + 0.4
    # inside large crown blobs the distance transform is flat - add jittered fill so forests are dense
    big = dist > 4.0
    if big.any():
        fr, fc = np.nonzero(big[::6, ::6])
        fr, fc = fr * 6, fc * 6
        fx, fy = an.xy_of(fr, fc)
        fx = fx + rng.uniform(-1.2, 1.2, len(fx))
        fy = fy + rng.uniform(-1.2, 1.2, len(fy))
        x, y = np.concatenate([x, fx]), np.concatenate([y, fy])
        r = np.concatenate([r, rng.uniform(1.8, 3.2, len(fx))])
        rows, cols = np.concatenate([rows, fr]), np.concatenate([cols, fc])
    conifer = an.conifer_like[rows, cols]
    return {"x": x, "y": y, "r": np.clip(r, 0.6, 8.0), "conifer": conifer}


def crowns_to_trees(cr, chm=None, rng=None):
    """Convert crowns to the tree dict used by landcover.tree_meshes (heights from CHM if given)."""
    rng = rng or np.random.default_rng(6)
    n = len(cr["x"])
    r = cr["r"]
    kind = np.where(cr["conifer"], 0, 1)
    bush = r < 1.3
    kind[bush] = 2
    h = np.where(kind == 0, np.clip(r * 2 * 2.3, 5, 26), np.clip(r * 2 * 1.35, 4, 20))
    h[bush] = r[bush] * 1.6 + 0.4
    if chm is not None:
        hc = chm(cr["x"], cr["y"], np.maximum(r, 1.0))
        okc = np.isfinite(hc) & (hc > 0.4)
        h = np.where(okc, hc, h)
        kind = np.where(okc & (hc < 2.0), 2, np.where(kind == 2, 1, kind))
        kind = np.where(okc & (hc >= 2.0) & (cr["conifer"]), 0, kind)
    w = np.where(kind == 0, np.minimum(r * 2, h * 0.55), np.minimum(r * 2.1, h * 1.1))
    w = np.maximum(w, 0.8)
    return {"x": cr["x"], "y": cr["y"], "h": h * rng.uniform(0.95, 1.05, n), "w": w, "kind": kind}


def trees_from_treetops(x, y, h, conifer, rng=None):
    """LiDAR canopy-height tree tops -> tree dict (bushes below 2 m)."""
    rng = rng or np.random.default_rng(9)
    kind = np.where(conifer, 0, 1)
    kind[h < 2.0] = 2
    w = np.where(kind == 0, 0.9 + 0.22 * h, 1.2 + 0.45 * h)
    w = np.where(kind == 2, np.maximum(1.0, h * 1.5), np.minimum(w, h * 1.05))
    return {"x": x, "y": y, "h": np.maximum(h, 0.6) * rng.uniform(0.97, 1.03, len(h)), "w": w, "kind": kind}

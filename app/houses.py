"""House footprints made buildable, and where their doors go.

Footprints traced on the photo / LiDAR (and some mapped ones) have chamfered corners, notches and odd angles:
on a pitched-roof house that gives a flat roof on a strange polygon. Here each house footprint becomes
  * one rectangle when it mostly is one (corners cut, small notches filled),
  * or two / three rectangles (an L, T or U house: a main block and wings, each with its own gable roof),
  * or, for big buildings (shops, sheds, blocks: over BIG_M2) and shapes that really aren't made of rectangles,
    the same outline with straight, square corners and a flat roof.
All in the building's own direction (its longest walls), at 25 cm.

Doors: the front door goes in the middle bay of the wall that faces the nearest road (track or street), with
concrete steps in front (the ground floor is above the ground); a garage door goes where a driveway meets the
house.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

RES = 0.25           # m per cell
BIG_M2 = 500.0       # bigger buildings: pitched only when they are (made of) rectangles, else a flat roof
HUGE_M2 = 1500.0     # warehouses, shops, schools: flat roof
MIN_WING = 2.8       # m: narrowest wing
MAX_PARTS = 3


def _frame(poly):
    """Unit vectors (t, n) along the building's long walls (length-weighted edge direction, mod 90 deg)."""
    xy = np.asarray(poly.exterior.coords)
    e = np.diff(xy, axis=0)
    L = np.linalg.norm(e, axis=1)
    ang = np.arctan2(e[:, 1], e[:, 0]) % (np.pi / 2)
    # circular mean on the 90-degree circle
    z = (L * np.exp(4j * ang)).sum()
    a = (np.angle(z) / 4) % (np.pi / 2)
    t = np.array([np.cos(a), np.sin(a)])
    return t, np.array([-t[1], t[0]])


def _mask(poly, t, n, close=1.0, open_=1.0):
    """The footprint in the building's frame, squared: rasterised at RES, small notches closed, chamfers and
    spurs opened away, holes filled. -> (mask, origin (u0, v0) of cell (0, 0))."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely import affinity

    q = affinity.affine_transform(poly, [t[0], t[1], n[0], n[1], 0, 0])
    u0, v0, u1, v1 = q.bounds
    pad = 1.5
    W = int(np.ceil((u1 - u0 + 2 * pad) / RES))
    H = int(np.ceil((v1 - v0 + 2 * pad) / RES))
    tr = Affine(RES, 0, u0 - pad, 0, RES, v0 - pad)            # row r -> v increasing
    m = rasterize([q], out_shape=(H, W), transform=tr, fill=0, default_value=1, dtype="uint8").astype(bool)
    kc, ko = max(1, int(round(close / RES / 2))), max(1, int(round(open_ / RES / 2)))
    m = ndi.binary_closing(m, structure=np.ones((2 * kc + 1, 2 * kc + 1)))
    m = ndi.binary_opening(m, structure=np.ones((2 * ko + 1, 2 * ko + 1)))
    m = ndi.binary_fill_holes(m)
    lab, nl = ndi.label(m)
    if nl > 1:
        sizes = ndi.sum(m, lab, np.arange(1, nl + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m, (u0 - pad, v0 - pad)


def _max_rect(m):
    """Largest all-True axis-aligned rectangle of a boolean grid -> (r0, r1, c0, c1) (exclusive ends), area."""
    H, W = m.shape
    h = np.zeros(W, int)
    best, area = (0, 0, 0, 0), 0
    for r in range(H):
        h = np.where(m[r], h + 1, 0)
        stack = []
        for c in range(W + 1):
            cur = h[c] if c < W else 0
            start = c
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                a = hh * (c - s)
                if a > area:
                    area, best = a, (r - hh + 1, r + 1, s, c)
                start = s
            stack.append((start, cur))
    return best, area


def _corners(r0, r1, c0, c1, origin, t, n):
    u0, v0 = origin
    us = (u0 + c0 * RES, u0 + c1 * RES)
    vs = (v0 + r0 * RES, v0 + r1 * RES)
    pts = [(us[0], vs[0]), (us[1], vs[0]), (us[1], vs[1]), (us[0], vs[1])]
    return np.array([t * u + n * v for u, v in pts])


def shape(poly):
    """-> ("rects", [corners (4, 2) ...] main block first) or ("flat", polygon with square corners)."""
    from rasterio.features import shapes as _shapes
    from rasterio.transform import Affine
    from shapely import affinity
    from shapely.geometry import shape as _shape

    if poly.area < 12:
        r = poly.minimum_rotated_rectangle
        return "rects", [np.asarray(r.exterior.coords)[:4]]
    t, n = _frame(poly)
    m, origin = _mask(poly, t, n)
    if m.sum() * RES * RES < 8:
        return "rects", [np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4]]
    rows = np.nonzero(m.any(1))[0]
    cols = np.nonzero(m.any(0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    total = int(m.sum())
    big = poly.area > BIG_M2
    huge = poly.area > HUGE_M2
    if not huge and total / ((r1 - r0) * (c1 - c0)) >= 0.86:
        # (nearly) a rectangle: the bounding box, shrunk to the footprint's area (cut corners filled evenly)
        f = float(np.sqrt(total / ((r1 - r0) * (c1 - c0))))
        cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
        hr_, hc = (r1 - r0) / 2 * f, (c1 - c0) / 2 * f
        return "rects", [_corners(cr - hr_, cr + hr_, cc - hc, cc + hc, origin, t, n)]
    if not huge:
        rest = m.copy()
        rects = []
        covered = 0
        for _ in range(MAX_PARTS):
            (a0, a1, b0, b1), area = _max_rect(rest)
            if area == 0:
                break
            if rects and (area * RES * RES < max(10.0, 0.12 * total * RES * RES)
                          or min(a1 - a0, b1 - b0) * RES < MIN_WING):
                break
            rects.append((a0, a1, b0, b1))
            rest[a0:a1, b0:b1] = False
            covered += area
        if rects and covered / total >= (0.85 if big else 0.8):
            return "rects", [_corners(*r, origin, t, n) for r in rects]
    if not big:
        # a house is never left with an odd outline: the rectangle of its main extent, at its own area
        f = float(np.sqrt(total / ((r1 - r0) * (c1 - c0))))
        cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
        hr_, hc = (r1 - r0) / 2 * f, (c1 - c0) / 2 * f
        return "rects", [_corners(cr - hr_, cr + hr_, cc - hc, cc + hc, origin, t, n)]
    # flat roof on the squared outline
    tr = Affine(RES, 0, origin[0], 0, RES, origin[1])
    geoms = [_shape(g) for g, v in _shapes(m.astype(np.uint8), mask=m, transform=tr)]
    if not geoms:
        return "flat", poly
    g = max(geoms, key=lambda x: x.area).simplify(0.6)          # the 25 cm steps of a slanted wall: one line
    back = affinity.affine_transform(g, [t[0], n[0], t[1], n[1], 0, 0]).buffer(0)
    if back.geom_type != "Polygon":
        back = max(getattr(back, "geoms", [poly]), key=lambda x: x.area)
    from shapely.geometry.polygon import orient

    return "flat", orient(back, 1.0)


def door_specials(corners, road_tree, driveways=None, others=(), ground_at=None, door=True):
    """Doors for one rectangle's walls -> {wall index: {bay index: "door" | ("garage", ground z)}}.
    corners: (4, 2) counter-clockwise; road_tree: cKDTree of road centre points (or None); driveways: shapely
    polygons; others: the other rectangles of the same house (bays hidden behind them get no door)."""
    from shapely.geometry import Point, Polygon

    from .textures import BAY_M

    a = np.asarray(corners, float)
    b = np.roll(a, -1, axis=0)
    c = a.mean(0)
    out = {}
    walls = []
    for i in range(4):
        L = float(np.linalg.norm(b[i] - a[i]))
        nb = int(round(L / BAY_M)) if L >= 2.0 else 0
        d = (b[i] - a[i]) / max(L, 1e-9)
        nout = np.array([d[1], -d[0]])                       # outward for a counter-clockwise ring
        walls.append((i, L, nb, d, nout))
    others = [Polygon(o) for o in others]

    def free(i, k):
        _, L, nb, d, nout = walls[i]
        p = a[i] + d * (L * (k + 0.5) / nb) + nout * 0.4
        return not any(o.contains(Point(p)) for o in others)

    # garage: where a driveway meets the house
    if driveways:
        house = Polygon(a)
        best = None
        for dw in driveways:
            dist = dw.distance(house)
            if dist > 1.2:
                continue
            q = np.asarray(dw.representative_point().coords[0]) if dw.area > 0 else c
            from shapely.ops import nearest_points

            # the middle of the stretch of wall the driveway meets (not just any nearest point)
            touch = house.exterior.intersection(dw.buffer(dist + 0.3))
            if not touch.is_empty and touch.length > 0.3:
                pt = np.asarray(touch.interpolate(0.5, normalized=True).coords[0]) if touch.geom_type == "LineString" \
                    else np.asarray(touch.centroid.coords[0])
            else:
                pt = np.asarray(nearest_points(house.exterior, dw)[0].coords[0])
            for i, L, nb, d, nout in walls:
                if nb < 1:
                    continue
                s = float(np.dot(pt - a[i], d))
                off = float(np.dot(pt - a[i], nout))
                if -0.5 <= s <= L + 0.5 and abs(off) < 0.6 and np.dot(q - pt, nout) > 0:
                    k = int(np.clip(s / L * nb, 0, nb - 1))
                    if free(i, k) and (best is None or dist < best[0]):
                        best = (dist, i, k, pt)
        if best is not None:
            _, i, k, pt = best
            zg = float(ground_at(pt[0], pt[1])) if ground_at is not None else None
            out.setdefault(i, {})[k] = ("garage", zg) if zg is not None else "garage"
    # front door: the wall facing the nearest road, its middle bay (not the garage's)
    if door:
        if road_tree is not None:
            _, j = road_tree.query(c)
            to_road = road_tree.data[j] - c
            to_road = to_road / max(np.linalg.norm(to_road), 1e-9)
        else:
            to_road = walls[0][4]
        order = sorted([w for w in walls if w[2] >= 1], key=lambda w: -float(np.dot(w[4], to_road)))
        for i, L, nb, d, nout in order:
            bays = sorted(range(nb), key=lambda k: abs(k - (nb - 1) / 2))
            for k in bays:
                if k in out.get(i, {}) or not free(i, k):
                    continue
                if any(v != "door" and abs(kk - k) < 1 for kk, v in out.get(i, {}).items()):
                    continue
                out.setdefault(i, {})[k] = "door"
                return out
    return out


def steps(corners, specials, floor0, base, offsets=None):
    """Concrete steps in front of each front door: two treads from the ground up to the floor (offsets:
    {(wall, bay): m} -> in front of a porch deck that far out). -> list of (V, F, UV)."""
    from .textures import BAY_M

    a = np.asarray(corners, float)
    b = np.roll(a, -1, axis=0)
    pieces = []
    for i, bays in specials.items():
        for k, v in bays.items():
            if v != "door":
                continue
            L = float(np.linalg.norm(b[i] - a[i]))
            nb = max(1, int(round(L / BAY_M)))
            d = (b[i] - a[i]) / max(L, 1e-9)
            nout = np.array([d[1], -d[0]])
            mid = a[i] + d * (L * (k + 0.5) / nb) + nout * (offsets or {}).get((i, k), 0.0)
            rise = floor0 - base
            for depth, top in ((1.05, base + rise * 0.6 if rise > 0.35 else floor0 - 0.02), (0.6, floor0 - 0.02)):
                if top <= base + 0.05:
                    continue
                pieces.append(_box(mid, d, nout, 1.5, depth, base, top))
    return pieces


def _box(mid, d, nout, width, depth, z0, z1):
    """A box against the wall (mid on the wall line), width along d, depth out along nout: top + 3 sides."""
    w2 = width / 2
    p = [mid - d * w2, mid + d * w2, mid + d * w2 + nout * depth, mid - d * w2 + nout * depth]
    V, F, UV = [], [], []

    def quad(q0, q1, q2, q3, uv):
        n0 = len(V)
        V.extend([q0, q1, q2, q3])
        UV.extend(uv)
        F.extend([[n0, n0 + 1, n0 + 2], [n0, n0 + 2, n0 + 3]])

    top = [np.r_[q, z1] for q in p]
    quad(top[3], top[2], top[1], top[0], [(0, 0), (width / 2, 0), (width / 2, depth / 2), (0, depth / 2)])
    h = (z1 - z0) / 2
    for qa, qb in ((p[3], p[2]), (p[2], p[1]), (p[0], p[3])):        # front, right, left (outward faces)
        ln = float(np.linalg.norm(qb - qa)) / 2
        quad(np.r_[qa, z0], np.r_[qb, z0], np.r_[qb, z1], np.r_[qa, z1], [(0, h), (ln, h), (ln, 0), (0, 0)])
    return np.array(V, float), np.array(F, np.int64), np.array(UV, float)


# ---------------------------------------------------------------------------------------------- building types
GARAGE_TAGS = {"garage", "garages", "carport", "shed"}
INDUSTRIAL_TAGS = {"industrial", "warehouse", "manufacture", "hangar", "factory", "storage_tank", "barn",
                   "farm_auxiliary", "service", "transformer_tower"}
COMMERCIAL_TAGS = {"commercial", "retail", "office", "supermarket", "kiosk", "hotel", "civic", "public", "school",
                   "college", "university", "hospital", "government", "church", "chapel", "mosque", "synagogue",
                   "temple", "fire_station", "sports_hall", "train_station", "transportation", "bank", "restaurant"}
BLOCK_TAGS = {"apartments", "dormitory", "residential", "terrace"}
HOUSE_TAGS = {"house", "detached", "semidetached_house", "bungalow", "cabin", "farm", "static_caravan", "hut"}


def building_kind(b: dict, area: float) -> str:
    """What to build: "house" (pitched roof, siding, door, porch), "garage" (a detached garage / shed: garage
    doors, no windows), "commercial" (flat roof with a parapet and cornice, shop fronts on the street, office
    windows), "block" (apartments: flat roof, cornice, house windows), "industrial" (plain walls, roller doors,
    flat roof). From OSM building=* (and shop / amenity / office tags); untagged buildings by size."""
    tags = b.get("tags") or {}
    t = str(tags.get("building", "yes")).lower()
    levels = 0.0
    try:
        levels = float(str(tags.get("building:levels", "0")).split(";")[0])
    except ValueError:
        pass
    if t in GARAGE_TAGS:
        return "garage"
    if t in INDUSTRIAL_TAGS:
        return "industrial"
    if t in COMMERCIAL_TAGS or any(k in tags for k in ("shop", "office")) or \
            tags.get("amenity") in ("restaurant", "cafe", "bank", "pharmacy", "fast_food", "bar", "pub", "fuel"):
        return "commercial"
    if t in BLOCK_TAGS and (levels >= 3 or area > 400):
        return "block"
    if t in HOUSE_TAGS:
        return "house"
    if t in BLOCK_TAGS:
        return "house"
    if area < 30:
        return "garage"                      # untagged sheds / small garages behind houses
    if area > 900:
        return "commercial" if levels <= 2 else "block"
    return "house"


def garage_specials(corners, road_tree, driveways=None, n_doors=1, ground_at=None):
    """Garage doors of a detached garage: on the wall a driveway meets, else the wall facing the nearest road;
    n_doors bays next to each other in the middle of that wall (n_doors < 0: every bay). -> _facade specials."""
    a = np.asarray(corners, float)
    b = np.roll(a, -1, axis=0)
    c = a.mean(0)
    from .textures import BAY_M

    sp = door_specials(corners, road_tree, driveways, (), ground_at, door=False)
    wall = next(iter(sp), None)
    if wall is None:
        to_road = np.array([0.0, -1.0])
        if road_tree is not None:
            _, j = road_tree.query(c)
            to_road = road_tree.data[j] - c
        to_road = to_road / max(np.linalg.norm(to_road), 1e-9)
        best = None
        for i in range(4):
            L = float(np.linalg.norm(b[i] - a[i]))
            if L < 2.4:
                continue
            d = (b[i] - a[i]) / L
            nout = np.array([d[1], -d[0]])
            score = float(np.dot(nout, to_road)) + 0.05 * L
            if best is None or score > best[0]:
                best = (score, i)
        if best is None:
            return {}
        wall = best[1]
    L = float(np.linalg.norm(b[wall] - a[wall]))
    nb = max(1, int(round(L / BAY_M)))
    want = nb if n_doors < 0 else min(nb, max(1, n_doors))
    k0 = (nb - want) // 2
    mid = a[wall] + (b[wall] - a[wall]) * 0.5
    zg = float(ground_at(mid[0], mid[1])) if ground_at is not None else None
    return {wall: {k: (("garage", zg) if zg is not None else "garage") for k in range(k0, k0 + want)}}


# ---------------------------------------------------------------------------------------------- porches
PORCH_SHARE = 0.4             # houses with a covered porch at the front door (the others: a small door canopy)
PORCH_DEPTH, CANOPY_DEPTH = 1.8, 1.0


def _quad_piece(p0, p1, p2, p3, uv):
    return np.array([p0, p1, p2, p3], float), np.array([[0, 1, 2], [0, 2, 3]]), np.asarray(uv, float)


def door_extras(corners, specials, floor0, ground, porch=False, eave=None):
    """A shed roof over each front door: a small canopy (1.9 x 1.0 m) or a covered porch (a deck at the floor,
    a roof on posts, PORCH_DEPTH deep, up to two bays wide, never over a garage door).
    -> (roof pieces [shingle], trim pieces [(V, F, UV in metres)], {(wall, bay): steps offset out from the wall})"""
    from .textures import BAY_M

    a = np.asarray(corners, float)
    b = np.roll(a, -1, axis=0)
    roofs, trims, offsets = [], [], {}
    for i, bays in specials.items():
        for k, v in bays.items():
            if v != "door":
                continue
            L = float(np.linalg.norm(b[i] - a[i]))
            nb = max(1, int(round(L / BAY_M)))
            d = (b[i] - a[i]) / max(L, 1e-9)
            nout = np.array([d[1], -d[0]])
            bay = L / nb
            if porch:
                lo, hi = k, k                                  # grow over free neighbour bays, at most 2 wide
                for kk in (k + 1, k - 1):
                    if hi - lo + 1 >= 2:
                        break
                    if 0 <= kk < nb and kk not in bays:
                        lo, hi = min(lo, kk), max(hi, kk)
                s0, s1 = lo * bay + 0.15, (hi + 1) * bay - 0.15
                depth, z = PORCH_DEPTH, floor0 + 2.55
            else:
                c0 = (k + 0.5) * bay
                s0, s1 = c0 - 0.95, c0 + 0.95
                depth, z = CANOPY_DEPTH, floor0 + 2.4
            p0, p1 = a[i] + d * s0, a[i] + d * s1
            q0, q1 = p0 + nout * depth, p1 + nout * depth
            zt = z + 0.3 * depth                               # the roof falls 30 % towards the edge
            if eave is not None and zt > eave - 0.2:           # ... but stays under the house's roof overhang
                zt = eave - 0.2
                z = max(floor0 + 2.25, zt - 0.3 * depth)
                zt = max(zt, z)
            w = s1 - s0
            roofs.append(_quad_piece(np.r_[q0, z], np.r_[q1, z], np.r_[p1, zt], np.r_[p0, zt],
                                     [(0, 0), (w / 3, 0), (w / 3, -depth / 3), (0, -depth / 3)]))
            under = 0.12
            trims.append(_quad_piece(np.r_[p0, zt - under], np.r_[p1, zt - under], np.r_[q1, z - under],
                                     np.r_[q0, z - under], [(0, 0), (w, 0), (w, depth), (0, depth)]))     # soffit
            trims.append(_quad_piece(np.r_[q0, z - under], np.r_[q1, z - under], np.r_[q1, z], np.r_[q0, z],
                                     [(0, 0), (w, 0), (w, under), (0, under)]))                         # fascia
            # the two ends of the roof slab (wall end higher than the outer end)
            trims.append(_quad_piece(np.r_[p0, zt - under], np.r_[q0, z - under], np.r_[q0, z], np.r_[p0, zt],
                                     [(0, 0), (depth, 0), (depth, under), (0, under)]))
            trims.append(_quad_piece(np.r_[q1, z - under], np.r_[p1, zt - under], np.r_[p1, zt], np.r_[q1, z],
                                     [(0, 0), (depth, 0), (depth, under), (0, under)]))
            if porch:
                deck = floor0 - 0.02
                mid = a[i] + d * (s0 + s1) / 2
                if deck > ground + 0.05:
                    trims.append(_box(mid, d, nout, w, depth, ground - 0.05, deck))
                for s in (s0 + 0.1, s1 - 0.1) + (((s0 + s1) / 2,) if w > 4.0 else ()):
                    pm = a[i] + d * s + nout * (depth - 0.1)
                    trims.append(_box(pm - nout * 0.06, d, nout, 0.12, 0.12, deck, z - under))
                offsets[(i, k)] = depth
    return roofs, trims, offsets

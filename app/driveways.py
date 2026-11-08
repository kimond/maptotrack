"""Clean driveway shapes.

The photo finds driveways and parking lots as ragged pixel blobs: holes where a car, a shadow or a bush sits on
them, a frayed edge where the asphalt meets the lawn, and a gap of 1-5 m before the street (the strip next to
the street is excluded from the search, and the kerb / verge mixes colours). Real driveways are simple shapes
that run from the street to the house. Here each blob becomes:

  * a rectangle at right angles to its street, from inside the street to the end of the blob, as wide as the
    blob typically is (most residential driveways);
  * or, for parking lots and L shapes, the blob smoothed into straight edges parallel / perpendicular to the
    street, plus an apron joining it to the street;
  * blobs that only touch a building (behind a house) are smoothed in their own direction.

Holes are filled (no bushes, cars or shadows in the middle) and buildings are cut out.
"""
from __future__ import annotations

import numpy as np

MAX_GAP = 7.0          # m: a blob this close to a street edge is joined to it
MIN_W, MAX_W = 2.8, 9.0  # m: plausible width of a single driveway (wider -> parking lot shape)
INTO_STREET = 1.0      # m: driveways start this far inside the street (the street asphalt covers it)


class Axes:
    """Centrelines of the track and the streets: points every ~1 m with tangent and half widths. The track and
    the streets are kept apart (each its own nearest-point search) and the nearest *edge* wins: the measured
    track is often wider than the street mapped under it."""

    def __init__(self, groups):
        from scipy.spatial import cKDTree

        self.groups = [(np.asarray(p, float), np.asarray(t, float), np.asarray(hl, float), np.asarray(hr, float),
                        cKDTree(p)) for p, t, hl, hr in groups if len(p)]
        self.tree = self.groups[0][4] if self.groups else None       # (None = nothing to join to)

    @classmethod
    def build(cls, cl=None, hl=None, hr=None, lines=()):
        """cl: track centreline (p, t) with half widths hl / hr (left / right); lines: [(xy, half width)]."""
        from .track import resample

        groups = []
        if cl is not None:
            groups.append((cl.p, cl.t, np.broadcast_to(np.asarray(hl, float), (len(cl.p),)),
                           np.broadcast_to(np.asarray(hr, float), (len(cl.p),))))
        P, T, H = [], [], []
        for xy, h in lines:
            try:
                p = resample(np.asarray(xy, float), 1.0, False)
            except ValueError:
                continue
            if len(p) < 2:
                continue
            t = np.gradient(p, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            P.append(p), T.append(t), H.append(np.full(len(p), h))
        if P:
            groups.append((np.vstack(P), np.vstack(T), np.concatenate(H), np.concatenate(H)))
        return cls(groups)

    def edge_gap(self, xy):
        """-> (gap to the nearest street edge (m, negative = on the street), centre point, tangent, half width
        on that side, side)."""
        best = None
        for p, t, hl, hr, tree in self.groups:
            d, i = tree.query(xy)
            rel = xy - p[i]
            n = np.column_stack([-t[i, 1], t[i, 0]])                 # left normal
            side = np.sign((rel * n).sum(1))
            half = np.where(side >= 0, hl[i], hr[i])
            g = d - half
            if best is None:
                best = [g, p[i], t[i], half, side]
            else:
                b = g < best[0]
                best = [np.where(b, g, best[0]), np.where(b[:, None], p[i], best[1]),
                        np.where(b[:, None], t[i], best[2]), np.where(b, half, best[3]), np.where(b, side, best[4])]
        return tuple(best)


def _grid_points(poly, step=0.5):
    from shapely import contains_xy

    x0, y0, x1, y1 = poly.bounds
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2))
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return pts[contains_xy(poly, pts[:, 0], pts[:, 1])]


def _fill(poly):
    from shapely.geometry import MultiPolygon, Polygon

    if poly.is_empty:
        return poly
    if isinstance(poly, MultiPolygon):
        return MultiPolygon([Polygon(g.exterior) for g in poly.geoms]).buffer(0)
    return Polygon(poly.exterior)


def _largest(geom, min_area=4.0):
    if geom.is_empty:
        return []
    parts = list(getattr(geom, "geoms", [geom]))
    return [g for g in parts if g.geom_type == "Polygon" and g.area >= min_area]


def _frame_poly(origin, t, n, corners):
    from shapely.geometry import Polygon

    return Polygon([origin + t * u + n * v for u, v in corners])


def _rectilinear(poly, origin, t, n, close=1.5, open_=1.0, res=0.25):
    """Smooth a blob into straight edges along t / n (staircase-free at `res`): close, open, fill holes."""
    from rasterio.features import rasterize, shapes
    from rasterio.transform import Affine
    from scipy import ndimage as ndi
    from shapely import affinity
    from shapely.geometry import shape

    # into the frame (u along t, v along n)
    M = [t[0], t[1], n[0], n[1], -(origin @ t), -(origin @ n)]
    q = affinity.affine_transform(poly, M)
    x0, y0, x1, y1 = q.bounds
    pad = close + 2
    W = int(np.ceil((x1 - x0 + 2 * pad) / res))
    H = int(np.ceil((y1 - y0 + 2 * pad) / res))
    tr = Affine(res, 0, x0 - pad, 0, -res, y1 + pad)
    m = rasterize([q], out_shape=(H, W), transform=tr, fill=0, default_value=1, dtype="uint8").astype(bool)
    kc = max(1, int(round(close / res)))
    ko = max(1, int(round(open_ / res)))
    m = ndi.binary_closing(np.pad(m, kc), structure=np.ones((2 * kc + 1, 2 * kc + 1)))[kc:-kc, kc:-kc]
    m = ndi.binary_opening(m, structure=np.ones((2 * ko + 1, 2 * ko + 1)))
    m = ndi.binary_fill_holes(m)
    if not m.any():
        return None
    geoms = [shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=tr)]
    g = max(geoms, key=lambda s: s.area).simplify(res * 0.6)
    # back to world: inverse of the rotation (t, n orthonormal)
    Mi = [t[0], n[0], t[1], n[1], origin[0], origin[1]]
    return affinity.affine_transform(_fill(g), Mi)


def _fit_rect(q, pts, origin, t, n):
    """Rectangle in the (t, n) frame for blob q (interior points pts): -> (IoU with q, (u0, u1, v0, v1),
    single driveway?) or None. A single driveway is as wide as the blob typically is row by row (bulges for a
    parked car or a turning spot ignored); a lot spans the blob's 2-98 % extent."""
    u = (pts - origin) @ t
    v = (pts - origin) @ n
    v0, v1 = np.percentile(v, [1, 98])
    rows = np.floor((v - v.min()) / 0.5).astype(int)
    cnt = np.bincount(rows)
    usum = np.bincount(rows, u)
    ok = cnt >= 3
    if ok.sum() < 4:
        return None
    widths = cnt[ok] * 0.5
    centres = usum[ok] / cnt[ok]
    w = float(np.median(widths))
    uc = float(np.median(centres))
    straight = np.percentile(np.abs(centres - uc), 90) < max(1.0, 0.35 * w)
    single = MIN_W * 0.8 <= w <= MAX_W and straight
    if single:
        w = max(w, MIN_W)
        u0, u1 = uc - w / 2, uc + w / 2
    else:
        u0, u1 = (float(x) for x in np.percentile(u, [2, 98]))
    rect = _frame_poly(origin, t, n, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    if not rect.is_valid or rect.area <= 0:
        return None
    iou = rect.intersection(q).area / max(rect.union(q).area, 1e-6)
    return iou, (u0, u1, float(v0), float(v1)), single


def _reach_street(axes, origin, t, n, uc, v0, step=0.5):
    """Extend a driveway running along n back (towards -n) from v0 until it is INTO_STREET inside a street
    (at most MAX_GAP + 3 m)."""
    v = v0
    while v > v0 - (MAX_GAP + 3.0):
        p = origin + t * uc + n * v
        g = axes.edge_gap(p[None, :])[0]
        if g[0] <= -INTO_STREET:
            return v
        v -= step
    return v0


def regularize(polys, axes: Axes, buildings=None, log=print):
    """Ragged driveway / parking polygons -> clean shapes joined to their street. buildings: shapely geometry
    (union of footprints) cut out of the result."""
    from shapely.ops import unary_union

    if not polys:
        return []
    out = []
    n_rect = n_lot = n_joined = 0
    for p in polys:
        q = _fill(p.buffer(1.0, join_style=2).buffer(-1.0, join_style=2))
        q = _fill(max(_largest(q, 1.0) or [p], key=lambda g: g.area))
        pts = _grid_points(q)
        if len(pts) < 20 or axes.tree is None:
            out.append(q)
            continue
        gap, cen, tan, hside, side = axes.edge_gap(pts)
        k = int(np.argmin(gap))
        g0 = float(gap[k])
        a = cen[k]
        t = tan[k]
        n = np.array([-t[1], t[0]]) * (side[k] if side[k] != 0 else 1.0)
        half = float(hside[k])
        u = (pts - a) @ t
        v = (pts - a) @ n
        joined = g0 < MAX_GAP and (v.max() - v.min()) > 1.0
        if not joined:
            # behind a house: straight edges along its own long side
            rect = q.minimum_rotated_rectangle
            xy = np.asarray(rect.exterior.coords)
            e = xy[1] - xy[0]
            tt = e / max(np.linalg.norm(e), 1e-9)
            shp = _rectilinear(q, xy[0], tt, np.array([-tt[1], tt[0]]), close=2.0, open_=1.5)
            out += _largest(shp) if shp is not None else [q]
            continue
        n_joined += 1
        v_near = half - INTO_STREET
        # the best rectangle: square to the street, or along the blob's own long side (a driveway that
        # meets the street at an angle) - whichever covers the blob best
        mr = np.asarray(q.minimum_rotated_rectangle.exterior.coords)
        e1, e2 = mr[1] - mr[0], mr[2] - mr[1]
        ax_long = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        ax_long = ax_long / max(np.linalg.norm(ax_long), 1e-9)
        if ax_long @ n < 0:
            ax_long = -ax_long                         # pointing away from the street
        frames = [(t, n, True)]
        if ax_long @ n > 0.5:                          # heads for the street, at an angle
            frames.append((np.array([ax_long[1], -ax_long[0]]), ax_long, False))
        best = None
        for ft, fn, square in frames:
            fit = _fit_rect(q, pts, a, ft, fn)
            if fit is not None and (best is None or fit[0] > best[0]):
                best = fit + (ft, fn, square)
        if best is not None and best[0] >= 0.62:
            iou, (u0, u1, v0, v1), single, ft, fn, square = best
            if square:
                v0 = v_near
            else:
                v0 = _reach_street(axes, a, ft, fn, (u0 + u1) / 2, v0)
            rect = _frame_poly(a, ft, fn, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
            r = 0.5 if single else 0.6
            out.append(rect.buffer(-r, join_style=1).buffer(r, join_style=1))   # softly rounded corners
            if single:
                n_rect += 1
            else:
                n_lot += 1
            continue
        v = (pts - a) @ n
        u = (pts - a) @ t
        # L shapes, irregular lots: straight edges along / across the street, plus an apron to the street
        shp = _rectilinear(q, a, t, n, close=2.5, open_=2.0, res=0.5)
        if shp is None:
            out.append(q)
            continue
        vmin = v.min()
        front = v < vmin + 3.0
        ua, ub = np.percentile(u[front], [5, 95])
        if ub - ua < MIN_W:
            m = (ua + ub) / 2
            ua, ub = m - MIN_W / 2, m + MIN_W / 2
        if ub - ua > 14.0:
            m = float(np.median(u[front]))
            ua, ub = max(ua, m - 7.0), min(ub, m + 7.0)
        apron = _frame_poly(a, t, n, [(ua, v_near), (ub, v_near), (ub, vmin + 1.0), (ua, vmin + 1.0)])
        out += _largest(unary_union([shp, apron]).buffer(0))
        n_lot += 1
    if buildings is not None and not buildings.is_empty:
        cut = buildings.buffer(0.3)
        out = [g for o in out for g in _largest(o.difference(cut))]
    out = [g for o in out for g in _largest(o.buffer(0), 6.0)]
    log(f"  driveways: {n_rect} straight, {n_lot} lots / L shapes, {n_joined} joined to their street")
    return out

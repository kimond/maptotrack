"""Road integrated into the ground (one continuous surface, like Map2Lap / BeamNG.Earth).

Instead of a road mesh laid on a terrain mesh (two surfaces joined by shoulders), the ground itself is one
triangulated surface (TIN) that contains the road:

* the road's own cross-sections (7 points across, every 2 m, crown included) are vertices of the ground,
  so the asphalt keeps exactly its shape and height;
* around it, ground points every 2 m in a band along the track and the streets, coarser further out,
  with heights blended into the road (no step, no seam, nothing to get stuck on);
* everything is triangulated together; each triangle is a driving surface by what is under its centre:
  track asphalt, streets and driveways -> 1ROAD_GND_*, the rest -> 1GRASS_GND_*;
* the asphalt is painted into the ground texture (crisp, anti-aliased), and the markings are a separate
  visual-only mesh 1.5 cm above it so they stay sharp whatever the texture resolution.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .meshes import MeshBuilder, _strip_rows, tile_uv

FINE = 2.0                 # m: ground point spacing near the track and the streets
BAND = 14.0                # m beyond the road edge with fine points
EDGE_CLEAR = 0.6           # m: no free ground points this close to the asphalt edge (road edge row is there)
ROAD_COLS = 7              # points across the road (as the old road mesh: crown shape)


class Tin:
    """The ground surface: points (x, y, z) + Delaunay triangles, with exact height lookup."""

    def __init__(self, pts, z, fixed=None):
        from scipy.spatial import Delaunay

        self.xy, self.z = pts, z
        self.fixed = np.zeros(len(pts), bool) if fixed is None else np.asarray(fixed, bool)
        self.dl = Delaunay(pts)
        tri = self.dl.simplices.copy()
        a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        tri[cross < 0] = tri[cross < 0][:, [0, 2, 1]]          # counter-clockwise = normal up
        keep = np.abs(cross) > 1e-6
        self.tri = tri[keep]

    def with_points(self, xy, z, r=0.45):
        """A new surface with extra fixed points (e.g. outlines where one ground material meets another).
        Free grid points closer than r to a new point are removed; new points closer than r to a fixed point
        (road, street and shoulder edges) are dropped."""
        from scipy.spatial import cKDTree

        xy, z = np.asarray(xy, float).reshape(-1, 2), np.asarray(z, float).ravel()
        lo, hi = self.xy.min(0), self.xy.max(0)              # never grow the surface past its area
        inside = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
        xy, z = xy[inside], z[inside]
        if not len(xy):
            return self
        fx = self.xy[self.fixed]
        if len(fx):
            d, _ = cKDTree(fx).query(xy)
            xy, z = xy[d > r], z[d > r]
        if not len(xy):
            return self
        d, _ = cKDTree(xy).query(self.xy)
        keep = self.fixed | (d > r)
        return Tin(np.vstack([self.xy[keep], xy]), np.concatenate([self.z[keep], z]),
                   np.concatenate([self.fixed[keep], np.ones(len(xy), bool)]))

    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        q = np.column_stack([x.ravel(), y.ravel()])
        s = self.dl.find_simplex(q)
        out = np.full(len(q), np.nan)
        ok = s >= 0
        if ok.any():
            T = self.dl.transform[s[ok]]
            bary = np.einsum("nij,nj->ni", T[:, :2], q[ok] - T[:, 2])
            w = np.column_stack([bary, 1 - bary.sum(1)])
            out[ok] = (w * self.z[self.dl.simplices[s[ok]]]).sum(1)
        return out.reshape(x.shape)


EDGE_STEP = 0.8            # m: most spacing of points along the road / street / shoulder edges


def _fill_line(line, step=EDGE_STEP, loop=False):
    """Extra points on a polyline (N, 2 or 3) so that no two consecutive points are more than `step` apart
    (only the added points, not the originals). z, when given, is interpolated."""
    a = line
    b = np.roll(line, -1, axis=0) if loop else line[1:]
    a = a if loop else line[:-1]
    seg = np.linalg.norm(b[:, :2] - a[:, :2], axis=1)
    n = np.ceil(seg / step).astype(int) - 1
    out = [a[k] + (b[k] - a[k]) * (j / (n[k] + 1)) for k in np.nonzero(n > 0)[0] for j in range(1, n[k] + 1)]
    return np.array(out).reshape(-1, line.shape[1])


def road_points(cl, hl, hr, camber):
    """The road's cross-sections as ground vertices (same shape as the old road mesh), plus points along both
    edges every EDGE_STEP m: on the outside of tight corners the cross-sections fan out (4 m apart at a kart
    hairpin) and Delaunay then joined edge points to grass points beyond the edge - asphalt spikes in the grass."""
    t = np.linspace(0, 1, ROAD_COLS)
    offsets = [hl - tc * (hl + hr) for tc in t]
    zoffs = [camber * (1 - (2 * tc - 1) ** 2) for tc in t]
    rows = _strip_rows(cl, offsets, zoffs)
    if cl.loop:
        rows = rows[:-1]                                   # _strip_rows repeats the first row on loops
    fill = [_fill_line(rows[:, c], loop=cl.loop) for c in (0, ROAD_COLS - 1)]
    return np.vstack([rows.reshape(-1, 3)] + fill)


def _offset_rows(xy, half, step=2.0, loop=False):
    """Points along both sides of a polyline at +-half (its edges), every `step` m."""
    from .track import resample

    try:
        p = resample(np.asarray(xy, float), step, loop)
    except ValueError:
        p = np.asarray(xy, float)
    if len(p) < 2:
        return np.zeros((0, 2))
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    return np.vstack([p + n * half, p - n * half])


NEAR_FINE_M = 15.0         # m beyond the road edge with 1 m ground points (sharper surface boundaries near the car)


def off_track_pieces(xy, half, cl, tm, step=2.0, min_len=4.0):
    """The parts of a street that are not the track itself. The track usually follows mapped streets (or a
    mapped raceway): those ways run along it, with their own centreline and width, and their asphalt stuck
    out past the measured track edges in spikes. Points within the track + street half width + 2 m that run
    parallel to the track are dropped; streets crossing or joining it are kept (their end inside the track is
    covered by the track anyway)."""
    from scipy.spatial import cKDTree

    from .track import resample

    try:
        p = resample(np.asarray(xy, float), step, False)
    except ValueError:
        return [np.asarray(xy, float)]
    if len(p) < 2:
        return [p]
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    d, _, hw = tm.road_info3(p[:, 0], p[:, 1])
    _, i = cKDTree(cl.p).query(p)
    along = (d < hw + half + 2.0) & (np.abs((t * cl.t[i]).sum(1)) > 0.7)
    out, run = [], []
    for k in range(len(p)):
        if along[k]:
            if len(run) * step >= min_len:
                out.append(np.array(run))
            run = []
        else:
            run.append(p[k])
    if len(run) * step >= min_len:
        out.append(np.array(run))
    return out


def build_tin(tm, cl, hl, hr, camber, bounds, coarse, street_lines=(), band=BAND, far_points=True, shoulder=0.0,
              near=0.0, log=print):
    """Ground points + triangulation. street_lines: [(xy array, half width)] get fine points and exact edge
    rows (so the street asphalt material ends on the street's edge). shoulder: a row of points at the shoulder's
    outer edge (its own material). band: metres of fine points beyond the road edge. far_points=False: no ground
    beyond the band (terrain off: only a strip of ground along the road and streets)."""
    from scipy.spatial import cKDTree

    x0, y0, x1, y1 = bounds
    road = road_points(cl, hl, hr, camber)
    extra = []
    if shoulder > 0:
        rows = _strip_rows(cl, [hl + shoulder, -(hr + shoulder)], [0.0, 0.0])
        sp = rows[:-1] if cl.loop else rows
        fill = [_fill_line(sp[:, c, :2], loop=cl.loop) for c in (0, 1)]
        extra.append(np.vstack([sp.reshape(-1, 3)[:, :2]] + fill))
    # fine grid near the track / streets, coarse grid elsewhere
    def grid(step):
        xs = np.arange(x0, x1 + step * 0.5, step)
        ys = np.arange(y0, y1 + step * 0.5, step)
        X, Y = np.meshgrid(xs, ys)
        return np.column_stack([X.ravel(), Y.ravel()])

    band = max(band, near)
    fine = grid(FINE)
    if near > 0:
        # 1 m points close to the track (each 2 m point gets its three 1 m neighbours)
        d0, _, hw0 = tm.road_info3(fine[:, 0], fine[:, 1])
        close = fine[d0 < hw0 + min(near, NEAR_FINE_M) + FINE]
        fine = np.vstack([fine] + [close + o for o in ((FINE / 2, 0), (0, FINE / 2), (FINE / 2, FINE / 2))])
    d, _, hw = tm.road_info3(fine[:, 0], fine[:, 1])
    near_track = d < hw + band
    near_street = np.zeros(len(fine), bool)
    if street_lines:
        pts, half = [], []
        for xy, h in street_lines:
            from .track import resample

            try:
                p = resample(np.asarray(xy, float), 1.0, False)
            except ValueError:
                p = np.asarray(xy, float)
            pts.append(p)
            half.append(np.full(len(p), h))
        tree = cKDTree(np.vstack(pts))
        ds, idx = tree.query(fine)
        near_street = ds < np.concatenate(half)[idx] + 4.0
    clear = hw + EDGE_CLEAR + (shoulder if shoulder > 0 else 0.0)
    fine = fine[(near_track | near_street) & (d > clear)]
    if street_lines:
        # edge points every 0.8 m: with the 0.6 m clearance below, Delaunay then always joins them along the edge
        edges = [_offset_rows(xy, h, step=EDGE_STEP) for xy, h in street_lines]
        edges = np.vstack([e for e in edges if len(e)]) if any(len(e) for e in edges) else np.zeros((0, 2))
        if len(edges):
            de, _, hwe = tm.road_info3(edges[:, 0], edges[:, 1])
            edges = edges[de > hwe + EDGE_CLEAR + (shoulder if shoulder > 0 else 0.0)]
        if len(edges):
            # no free point right next to a street edge: Delaunay then joins the edge points along the edge
            de2, _ = cKDTree(edges).query(fine)
            fine = fine[de2 > EDGE_CLEAR]
            extra.append(edges)
    far = grid(max(coarse, FINE)) if far_points else np.zeros((0, 2))
    dfar, _, hwf = tm.road_info3(far[:, 0], far[:, 1])
    far = far[dfar >= hwf + band]
    # coarse points that would sit inside the fine band are dropped (the band has its own)
    if len(fine) and len(far):
        dd, _ = cKDTree(fine).query(far)
        far = far[dd > FINE * 0.75]
    # street edges and shoulder rows only inside the map area: streets run on past it, and ground beyond the
    # bounds has no ground texture tile (it stretched the edge tile's picture over and over)
    extra = [e[(e[:, 0] >= x0) & (e[:, 0] <= x1) & (e[:, 1] >= y0) & (e[:, 1] <= y1)] for e in extra]
    if far_points:
        # a frame of points around the map every grid step: no long thin triangles along the outer edge
        st = max(coarse, FINE)
        fx = np.linspace(x0, x1, max(2, int(np.ceil((x1 - x0) / st)) + 1))
        fy = np.linspace(y0, y1, max(2, int(np.ceil((y1 - y0) / st)) + 1))
        frame = np.vstack([np.column_stack([fx, np.full(len(fx), y0)]), np.column_stack([fx, np.full(len(fx), y1)]),
                           np.column_stack([np.full(len(fy), x0), fy]), np.column_stack([np.full(len(fy), x1), fy])])
        dfr, _, hwr = tm.road_info3(frame[:, 0], frame[:, 1])
        frame = frame[dfr > hwr + EDGE_CLEAR]              # not on the road where it crosses the map edge
        far = far[(far[:, 0] > x0 + 0.5) & (far[:, 0] < x1 - 0.5) & (far[:, 1] > y0 + 0.5) & (far[:, 1] < y1 - 0.5)]
        fine = fine[(fine[:, 0] > x0 + 0.5) & (fine[:, 0] < x1 - 0.5) & (fine[:, 1] > y0 + 0.5) & (fine[:, 1] < y1 - 0.5)]
        far = np.vstack([far, frame])
    if extra:
        ex = np.vstack(extra)
        if len(far):
            dd, _ = cKDTree(ex).query(far)
            far = far[dd > EDGE_CLEAR]
    free = np.vstack([fine, far] + extra)
    zf = tm.height(free[:, 0], free[:, 1])
    pts = np.vstack([road[:, :2], free])
    z = np.concatenate([road[:, 2], zf])
    fixed = np.concatenate([np.ones(len(road), bool), np.zeros(len(fine) + len(far), bool),
                            np.ones(sum(len(e) for e in extra), bool)])
    # merge duplicates (folded inner edges of tight corners give repeated road points)
    key = np.round(pts / 0.02).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    first.sort()
    tin = Tin(pts[first], z[first], fixed[first])
    log(f"  ground surface: {len(first):,} points ({len(road):,} on the road), {len(tin.tri):,} triangles")
    return tin


def _ring_points(geom, step=EDGE_STEP):
    """Points every `step` m along every ring (outline and holes) of a (multi)polygon."""
    out = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        for ring in [poly.exterior] + list(poly.interiors):
            c = np.asarray(ring.segmentize(step).coords)[:-1]
            if len(c):
                out.append(c[:, :2])
    return np.vstack(out) if out else np.zeros((0, 2))


def _inside(geom, xy):
    from shapely import contains_xy

    if geom is None or geom.is_empty or not len(xy):
        return np.zeros(len(xy), bool)
    return contains_xy(geom, xy[:, 0], xy[:, 1])


def build_tin_net(tm, net, bounds, coarse, band=BAND, near=0.0, far_points=True, log=print, keep=None):
    """The ground surface of an area: every street's cross-sections (crown) and the asphalt outline (kerb
    corners included) are fixed vertices, all at the network's road height; ground points every 2 m near the
    streets (1 m close to them), coarser elsewhere."""
    from scipy.spatial import cKDTree

    x0, y0, x1, y1 = bounds
    asph = net.asphalt
    # -- road vertices: the outline every 0.8 m, and the cross-sections of every street inside it
    edge = _ring_points(asph)
    edge = edge[(edge[:, 0] >= x0) & (edge[:, 0] <= x1) & (edge[:, 1] >= y0) & (edge[:, 1] <= y1)]
    rows = []
    t = np.linspace(0, 1, ROAD_COLS)[1:-1]
    for c in net.chains:
        cl = c.cl
        for tc in t:
            off = c.hl - tc * (c.hl + c.hr)
            rows.append(cl.p + cl.n * off[:, None])
    inner = np.vstack(rows) if rows else np.zeros((0, 2))
    inner = inner[_inside(asph.buffer(-0.35), inner)]
    if len(inner):                                     # junctions: the crossing streets' rows overlap
        key = np.round(inner / 0.5).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        inner = inner[np.sort(first)]
        if len(edge):
            dd, _ = cKDTree(edge).query(inner)
            inner = inner[dd > 0.3]
    road = np.vstack([edge, inner])

    def grid(step):
        xs = np.arange(x0, x1 + step * 0.5, step)
        ys = np.arange(y0, y1 + step * 0.5, step)
        X, Y = np.meshgrid(xs, ys)
        return np.column_stack([X.ravel(), Y.ravel()])

    band = max(band, near)
    fine = grid(FINE)
    _, e0 = net.query(fine[:, 0], fine[:, 1], k=4)
    if near > 0:
        close = fine[e0 < min(near, NEAR_FINE_M) + FINE]
        fine = np.vstack([fine] + [close + o for o in ((FINE / 2, 0), (0, FINE / 2), (FINE / 2, FINE / 2))])
        _, e0 = net.query(fine[:, 0], fine[:, 1], k=4)
    fine = fine[(e0 < band)]
    # no free point on the asphalt or right next to its edge (the edge row is there)
    fine = fine[~_inside(asph.buffer(EDGE_CLEAR), fine)]
    far = grid(max(coarse, FINE)) if far_points else np.zeros((0, 2))
    if len(far):
        _, ef = net.query(far[:, 0], far[:, 1], k=4)
        far = far[ef >= band]
        if len(fine):
            dd, _ = cKDTree(fine).query(far)
            far = far[dd > FINE * 0.75]
    if keep is not None:
        # an area polygon: points only inside it; its outline is a row of points (below), so the ground
        # ends exactly on it
        inner_k = keep.buffer(-0.5)
        fine = fine[_inside(inner_k, fine)]
        far = far[_inside(inner_k, far)]
    if far_points:
        st = max(coarse, FINE)
        if keep is not None:                            # a frame of points along the polygon's outline
            frame = _ring_points(keep.buffer(0.01), step=st)
        else:
            fx = np.linspace(x0, x1, max(2, int(np.ceil((x1 - x0) / st)) + 1))
            fy = np.linspace(y0, y1, max(2, int(np.ceil((y1 - y0) / st)) + 1))
            frame = np.vstack([np.column_stack([fx, np.full(len(fx), y0)]), np.column_stack([fx, np.full(len(fx), y1)]),
                               np.column_stack([np.full(len(fy), x0), fy]), np.column_stack([np.full(len(fy), x1), fy])])
        frame = frame[~_inside(asph.buffer(EDGE_CLEAR), frame)]
        far = far[(far[:, 0] > x0 + 0.5) & (far[:, 0] < x1 - 0.5) & (far[:, 1] > y0 + 0.5) & (far[:, 1] < y1 - 0.5)]
        fine = fine[(fine[:, 0] > x0 + 0.5) & (fine[:, 0] < x1 - 0.5) & (fine[:, 1] > y0 + 0.5) & (fine[:, 1] < y1 - 0.5)]
        far = np.vstack([far, frame])
    zr, _ = net.query(road[:, 0], road[:, 1])
    free = np.vstack([fine, far])
    zf = tm.height(free[:, 0], free[:, 1])
    pts = np.vstack([road, free])
    z = np.concatenate([zr, zf])
    fixed = np.concatenate([np.ones(len(road), bool), np.zeros(len(free), bool)])
    key = np.round(pts / 0.02).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    first.sort()
    tin = Tin(pts[first], z[first], fixed[first])
    log(f"  ground surface: {len(first):,} points ({len(road):,} on the streets), {len(tin.tri):,} triangles")
    return tin


ASPHALT_REPEAT = 2.5       # m per repeat of the tiled asphalt photo (track and streets)
GRAVEL_REPEAT = 2.0        # m per repeat of the tiled gravel photo (shoulders)


def ground_meshes(tin, tiles, road_geom, keep=None, chunk_tris=40000, asphalt_geom=None, shoulder_geom=None,
                  near=None):
    """Split the triangles by tile (ground texture) and by driving surface. road_geom: shapely geometry of
    everything driven on as asphalt (track, streets, driveways). keep: only triangles whose centre is inside
    this geometry are built (None = all).
    asphalt_geom (track + streets, whose edges are vertices of the surface): their own tiled photo asphalt
    material (1ROAD_ASPH, 'asphalt_tiled'); shoulder_geom: tiled gravel (1GRASS_SHLD, 'gravel_tiled'). Close to
    the car a 25 cm / pixel ground texture is a blur; these repeat a photo every few metres instead.
    near: (geometry, classify(x, y) -> land-cover class): the ground there gets tiled photo materials per
    surface (near_<class>, groundmat.NEAR_CLASSES) instead of the 25 cm ground tiles."""
    from shapely import contains_xy

    P = np.column_stack([tin.xy, tin.z])
    cen = P[tin.tri].mean(1)
    on_road = contains_xy(road_geom, cen[:, 0], cen[:, 1]) if road_geom is not None else np.zeros(len(cen), bool)
    on_asph = contains_xy(asphalt_geom, cen[:, 0], cen[:, 1]) if asphalt_geom is not None else np.zeros(len(cen), bool)
    on_shld = (contains_xy(shoulder_geom, cen[:, 0], cen[:, 1]) & ~on_asph & ~on_road) if shoulder_geom is not None \
        else np.zeros(len(cen), bool)
    built = contains_xy(keep, cen[:, 0], cen[:, 1]) if keep is not None else np.ones(len(cen), bool)
    parts = []
    in_near = np.zeros(len(cen), bool)
    if near is not None:
        ngeom, classify = near
        in_near = contains_xy(ngeom, cen[:, 0], cen[:, 1]) & ~on_asph & ~on_shld & built
        idx_n = np.nonzero(in_near)[0]
        if len(idx_n):
            from .groundmat import NEAR_CLASSES

            cls = classify(cen[idx_n, 0], cen[idx_n, 1]).astype(object)   # driveways are a class of their own
            kx = np.floor(cen[idx_n, 0] / 128.0).astype(int)
            ky = np.floor(cen[idx_n, 1] / 128.0).astype(int)
            groups = {}
            for j, i in enumerate(idx_n):
                groups.setdefault((cls[j], kx[j], ky[j], bool(on_road[i])), []).append(i)
            for (c, gx, gy, road), ids in sorted(groups.items(), key=lambda kv: str(kv[0])):
                rep_m = NEAR_CLASSES[c][1]
                prefix = "1ROAD_NEAR" if road else "1GRASS_NEAR"
                mb = MeshBuilder(f"{prefix}_{c.upper()}_{gx + 500}_{gy + 500}", f"near_{c}", smooth=True)
                tri = tin.tri[np.asarray(ids)]
                used, inv = np.unique(tri, return_inverse=True)
                V = P[used]
                mb.add(V, inv.reshape(-1, 3), np.column_stack([V[:, 0] / rep_m, -V[:, 1] / rep_m]))
                parts += mb.done()
    built = built & ~in_near
    nx = max(t["ti"] for t in tiles) + 1
    ny = max(t["tj"] for t in tiles) + 1
    size = tiles[0]["size"]
    X0, Y0 = tiles[0]["x0"] - tiles[0]["ti"] * size, tiles[0]["y0"] - tiles[0]["tj"] * size   # grid origin
    ti = np.clip(np.floor((cen[:, 0] - X0) / size).astype(int), 0, nx - 1)
    tj = np.clip(np.floor((cen[:, 1] - Y0) / size).astype(int), 0, ny - 1)
    for t in tiles:
        in_t = (ti == t["ti"]) & (tj == t["tj"])
        in_t &= built
        tag = t["material"][len("ground_"):]
        for surf, sel, mat, rep in (("1ROAD_ASPH", in_t & on_asph, "asphalt_tiled", ASPHALT_REPEAT),
                                    ("1ROAD_GND", in_t & on_road & ~on_asph, t["material"], None),
                                    ("1GRASS_SHLD", in_t & on_shld, "gravel_tiled", GRAVEL_REPEAT),
                                    ("1GRASS_GND", in_t & ~on_road & ~on_asph & ~on_shld, t["material"], None)):
            idx = np.nonzero(sel)[0]
            if not len(idx):
                continue
            mb = MeshBuilder(f"{surf}_{tag}", mat, smooth=True)
            for k in range(0, len(idx), chunk_tris):
                tri = tin.tri[idx[k:k + chunk_tris]]
                used, inv = np.unique(tri, return_inverse=True)
                V = P[used]
                uv = tile_uv(V[:, :2], t) if rep is None else np.column_stack([V[:, 0] / rep, -V[:, 1] / rep])
                mb.add(V, inv.reshape(-1, 3), uv)
                mb.flush()
            parts += mb.done()
    return parts


def ground_tiles(bounds, tile_m=512.0):
    """Ground texture tiles: equal squares (at most tile_m) covering the bounds, one material ground_jj_ii each.
    Square, so tiled close-up detail (ksMultilayer) has the same scale both ways; the last row / column may reach
    past the bounds (that part of the texture is never seen)."""
    x0, y0, x1, y1 = bounds
    W, H = x1 - x0, y1 - y0
    nx = max(1, int(np.ceil(W / tile_m - 1e-9)))
    ny = max(1, int(np.ceil(H / tile_m - 1e-9)))
    size = max(W / nx, H / ny)
    nx = max(1, int(np.ceil(W / size - 1e-9)))
    ny = max(1, int(np.ceil(H / size - 1e-9)))
    tiles = []
    for tj in range(ny):
        for ti in range(nx):
            tx0, ty0 = x0 + ti * size, y0 + tj * size
            tiles.append({"material": f"ground_{tj:02d}_{ti:02d}", "x0": tx0, "x1": tx0 + size,
                          "y0": ty0, "y1": ty0 + size, "ti": ti, "tj": tj, "size": size})
    return tiles


# ------------------------------------------------------------------ painting and markings
TRACK_ASPHALT = (74, 74, 77)
STREET_ASPHALT = (84, 84, 86)
SHOULDER_GRAVEL = (138, 130, 112)


def plain_tile(t, res, seed=11):
    """'Plain' ground: mown-grass green with soft noise (the asphalt is painted on top afterwards)."""
    from .landcover import GROUND_STYLE, _value_noise

    W = int(min(np.ceil((t["x1"] - t["x0"]) / res), 4096))
    H = int(min(np.ceil((t["y1"] - t["y0"]) / res), 4096))
    X, Y = np.meshgrid(np.linspace(t["x0"], t["x1"], W, dtype=np.float32),
                       np.linspace(t["y1"], t["y0"], H, dtype=np.float32))
    base, alt, var, grain = GROUND_STYLE["lawn"]
    mix = np.clip(_value_noise(X, Y, 9.0, seed) * 1.4 - 0.2, 0, 1)[..., None]
    rgb = np.asarray(base, np.float32) * (1 - mix) + np.asarray(alt, np.float32) * mix
    rgb *= (1 + var * (_value_noise(X, Y, 3.0, seed + 1) - 0.5) + grain * (_value_noise(X, Y, 0.4, seed + 2) - 0.5))[..., None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def paint_roads(textures, tiles, layers, seed=17):
    """Paint asphalt into the ground textures. layers: [(shapely geometry, rgb)], painted in order."""
    from scipy.ndimage import gaussian_filter
    from shapely.geometry import MultiPolygon, Polygon
    from PIL import ImageDraw

    from .landcover import _value_noise
    from .parallel import pmap

    def polys_of(g):
        if g is None or g.is_empty:
            return []
        if isinstance(g, Polygon):
            return [g]
        if isinstance(g, MultiPolygon):
            return list(g.geoms)
        return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]

    layers = [(polys_of(g), c) for g, c in layers]

    def paint(t):
        img = textures.get(t["material"])
        if img is None:
            return None
        W, H = img.size
        sx, sy = W / (t["x1"] - t["x0"]), H / (t["y1"] - t["y0"])
        ss = 2
        out = np.asarray(img, np.float32)
        X, Y = np.meshgrid(np.linspace(t["x0"], t["x1"], W), np.linspace(t["y1"], t["y0"], H))
        grain = (0.5 * _value_noise(X, Y, 0.1, seed) + 0.3 * _value_noise(X, Y, 0.5, seed + 1)
                 + 0.2 * _value_noise(X, Y, 6.0, seed + 2))
        hit = False
        for polys, rgb in layers:
            mask = Image.new("L", (W * ss, H * ss), 0)
            dm = ImageDraw.Draw(mask)
            any_p = False
            for p in polys:
                bx0, by0, bx1, by1 = p.bounds
                if bx1 < t["x0"] or bx0 > t["x1"] or by1 < t["y0"] or by0 > t["y1"]:
                    continue
                any_p = True
                ring = [((x - t["x0"]) * sx * ss, (t["y1"] - y) * sy * ss) for x, y in np.asarray(p.exterior.coords)]
                dm.polygon(ring, fill=255)
                for hole in p.interiors:
                    hp = [((x - t["x0"]) * sx * ss, (t["y1"] - y) * sy * ss) for x, y in np.asarray(hole.coords)]
                    dm.polygon(hp, fill=0)
            if not any_p:
                continue
            hit = True
            a = np.asarray(mask.resize((W, H), Image.BOX), np.float32) / 255.0
            col = np.array(rgb, np.float32)[None, None, :] * (0.88 + 0.24 * grain)[..., None]
            # slightly darker along the edges (dirt, wear), lighter wheel paths are left to the grain
            rim = np.clip(1.0 - gaussian_filter(a, 0.4 * sx), 0, 1) * a
            col = col * (1.0 - 0.12 * rim)[..., None]
            out = out * (1 - a[..., None]) + col * a[..., None]
        if not hit:
            return None
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    for t, img in zip(tiles, pmap(paint, tiles)):
        if img is not None:
            textures[t["material"]] = img


def paint_texture(size=16):
    rng = np.random.default_rng(3)
    a = np.clip(228 + rng.normal(0, 6, (size, size)), 0, 255)
    return Image.fromarray(np.stack([a, a, a * 0.97], -1).astype(np.uint8), "RGB")


def tiled_texture(photo, mean_rgb, size=1024, contrast=1.0, seed=0):
    """A seamless photo material (ambientCG) recoloured to mean_rgb, its big stains removed so the repeat does
    not show; procedural grain when the photo is missing."""
    from . import materials

    if photo is not None:
        src = np.asarray(photo.convert("RGB").resize((size, size), Image.LANCZOS), np.float32)
        from scipy.ndimage import gaussian_filter

        low = np.stack([gaussian_filter(src[..., c], size * 0.08, mode="wrap") for c in range(3)], -1)
        rel = src / np.maximum(low, 1.0)                         # the photo's structure around 1, stains removed
        rel = 1.0 + contrast * (rel - 1.0)
        chroma = rel / np.maximum(rel.mean(-1, keepdims=True), 1e-3)
        lum = rel.mean(-1, keepdims=True)
        img = np.asarray(mean_rgb, np.float32) * lum * (1.0 + 0.35 * (chroma - 1.0))
    else:
        d = materials.detail(None, size, size, fallback_seed=seed, highpass=0.05)
        img = np.asarray(mean_rgb, np.float32)[None, None, :] * (1.0 + 0.12 * contrast * d)[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

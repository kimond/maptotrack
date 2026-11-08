"""Mesh generation in the local track frame (x east, y north, z up, metres).

Naming follows Assetto Corsa conventions so that physics works after ksEditor:
    1ROAD_xxx   drivable asphalt         1GRASS_xxx  grass / terrain
    1WALL_xxx   collision walls          (no digit prefix = visual only)
Every mesh is kept below 65k vertices (KN5 limit).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mapbox_earcut as earcut
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .track import Centerline, resample

MAX_VERTS = 60000


@dataclass
class MeshPart:
    name: str
    material: str
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    smooth: bool = False

    @property
    def nverts(self):
        return len(self.vertices)


@dataclass
class MeshBuilder:
    """Accumulates geometry and splits it into <MAX_VERTS chunks."""
    prefix: str
    material: str
    smooth: bool = False
    parts: list = field(default_factory=list)
    _v: list = field(default_factory=list)
    _f: list = field(default_factory=list)
    _uv: list = field(default_factory=list)
    _count: int = 0

    def add(self, v, f, uv):
        if self._count + len(v) > MAX_VERTS:
            self.flush()
        self._v.append(v)
        self._f.append(f + self._count)
        self._uv.append(uv)
        self._count += len(v)

    def flush(self):
        if self._count:
            self.parts.append(MeshPart(
                f"{self.prefix}_{len(self.parts):03d}", self.material,
                np.vstack(self._v), np.vstack(self._f).astype(np.int64), np.vstack(self._uv), self.smooth))
        self._v, self._f, self._uv, self._count = [], [], [], 0

    def done(self):
        self.flush()
        return self.parts


def ribbon(rows: np.ndarray, uv: np.ndarray):
    """rows (N,C,3) / uv (N,C,2) -> vertices, faces. Normal = column_dir x row_dir."""
    N, C, _ = rows.shape
    idx = np.arange(N * C).reshape(N, C)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, :-1].ravel()
    d = idx[1:, 1:].ravel()
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([b, d, c])])
    return rows.reshape(-1, 3), faces, uv.reshape(-1, 2)


def collapse_folds(pts: np.ndarray, tang: np.ndarray, eps=0.02) -> np.ndarray:
    """Offset curves fold back on themselves on the inside of tight corners; collapse those points."""
    q = pts.copy()
    last = q[0].copy()
    for i in range(1, len(q)):
        if np.dot(q[i] - last, tang[i]) < eps:
            q[i] = last
        else:
            last = q[i].copy()
    return q


def _closed(cl: Centerline, arr):
    """For loops, repeat the first row at the end so UVs don't wrap."""
    return np.concatenate([arr, arr[:1]], axis=0) if cl.loop else arr


def _strip_rows(cl: Centerline, offsets, zoff, tang=None):
    """Build rows (N,C,3) at lateral offsets (left positive) with per-column z offsets.
    Offsets and z offsets may be scalars or per-centreline-point arrays (N,)."""
    cols = []
    for off, dz in zip(offsets, zoff):
        off_a = np.broadcast_to(np.asarray(off, float), (len(cl.p),))
        xy = cl.p + cl.n * off_a[:, None]
        xy = collapse_folds(xy, cl.t)
        z = cl.z + np.broadcast_to(np.asarray(dz, float), (len(cl.p),))
        cols.append(np.column_stack([xy, z]))
    rows = np.stack(cols, axis=1)
    return _closed(cl, rows)


def _chunks(nrows, per_chunk):
    a = 0
    while a < nrows - 1:
        b = min(a + per_chunk, nrows - 1)
        yield a, b + 1
        a = b


def half_widths(cl: Centerline, width):
    """width: scalar or (hl, hr) tuple of per-point arrays -> (hl, hr) arrays."""
    if isinstance(width, tuple):
        return np.asarray(width[0], float), np.asarray(width[1], float)
    return np.full(len(cl.p), width / 2.0), np.full(len(cl.p), width / 2.0)


def build_walls(cl: Centerline, width, offset: float, height=1.0, thick=0.6, chunk_m=250.0):
    hl, hr = half_widths(cl, width)
    s = np.concatenate([cl.s, [cl.length]]) if cl.loop else cl.s
    per = max(int(chunk_m / cl.spacing), 4)
    base, top = -0.4, height
    i_l, o_l = hl + offset, hl + offset + thick
    i_r, o_r = -(hr + offset), -(hr + offset + thick)
    faces = [
        # (offsets, z-offsets)  columns ordered so normals face outwards
        ([i_l, i_l], [top, base]),      # left inner face -> faces the road
        ([o_l, o_l], [base, top]),      # left outer face
        ([o_l, i_l], [top, top]),       # left top
        ([i_r, i_r], [base, top]),      # right inner face
        ([o_r, o_r], [top, base]),      # right outer face
        ([i_r, o_r], [top, top]),       # right top
    ]
    mb = MeshBuilder("1WALL", "concrete")
    built = [_strip_rows(cl, offs, zs) for offs, zs in faces]
    v = np.array([0.0, 1.0])
    uv = np.stack(np.broadcast_arrays(v[None, :], (s / 3.0)[:, None]), axis=-1)
    for a, b in _chunks(len(built[0]), per):
        for rows in built:
            mb.add(*ribbon(rows[a:b], uv[a:b]))
        mb.flush()
    return mb.done()


class TerrainModel:
    """DEM heights blended into the road corridor so the ground continues the road at its edge.

    Within half road width + `extra` metres of the centreline the ground takes the (smoothed) road height,
    then blends back to the DEM over `blend` metres."""

    def __init__(self, proj, dem, cl: Centerline, extra: float, half_width=None, blend: float = 25.0):
        from scipy.spatial import cKDTree

        self.proj, self.dem, self.cl = proj, dem, cl
        dense = resample(cl.p, 1.0, cl.loop)
        self.tree = cKDTree(dense)
        _, idx = cKDTree(cl.p).query(dense)
        self.zc = cl.z[idx]
        hw = np.full(len(cl.p), 5.0) if half_width is None else np.asarray(half_width, float)
        self.hwc = hw[idx]
        self.extra = extra
        self.blend = blend

    def road_info(self, x, y):
        d, idx = self.tree.query(np.column_stack([np.ravel(x), np.ravel(y)]))
        return d.reshape(np.shape(x)), self.zc[idx].reshape(np.shape(x))

    def road_info3(self, x, y):
        """distance to centreline, road height, road half width at the nearest point."""
        d, idx = self.tree.query(np.column_stack([np.ravel(x), np.ravel(y)]))
        sh = np.shape(x)
        return d.reshape(sh), self.zc[idx].reshape(sh), self.hwc[idx].reshape(sh)

    tin = None              # ground.Tin: the one ground surface (road included)

    def surface_height(self, x, y):
        """Height of the built ground surface (its triangles, exactly as built), or the model height."""
        if self.tin is None:
            return self.height(x, y)
        h = self.tin.height(x, y)
        bad = ~np.isfinite(h)
        if bad.any():
            h = np.where(bad, self.height(x, y), h)
        return h

    def raw(self, x, y):
        lat, lon = self.proj.to_latlon(x, y)
        return np.asarray(self.dem.sample(lat, lon), dtype=float)

    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = self.raw(x, y)
        d, rz, hw = self.road_info3(x, y)
        w = np.clip((d - (hw + self.extra)) / self.blend, 0, 1)
        w = w * w * (3 - 2 * w)
        return rz * (1 - w) + h * w


def corridor_polygon(cl: Centerline, half, half_right=None):
    """Area covered by the road (+ clearance). half / half_right: scalars or per-point arrays."""
    hl = np.broadcast_to(np.asarray(half, float), (len(cl.p),))
    hr = hl if half_right is None else np.broadcast_to(np.asarray(half_right, float), (len(cl.p),))
    if np.ptp(hl) < 1e-6 and np.ptp(hr) < 1e-6 and abs(hl[0] - hr[0]) < 1e-6:
        pts = np.vstack([cl.p, cl.p[:1]]) if cl.loop else cl.p
        return LineString(pts).buffer(float(hl[0]), cap_style=2 if not cl.loop else 1, join_style=1, resolution=8)
    L = cl.p + cl.n * hl[:, None]
    R = cl.p - cl.n * hr[:, None]
    n = len(cl.p)
    idx = list(range(n)) + ([0] if cl.loop else [])
    quads = []
    for a, b in zip(idx[:-1], idx[1:]):
        q = Polygon([L[a], L[b], R[b], R[a]])
        if not q.is_valid:
            q = q.buffer(0)
        quads.append(q)
    return unary_union(quads).buffer(0.3).buffer(-0.3)


def _polygons_for_building(b, proj):
    outers, inners = [], []
    for ring in b["outer"]:
        xy = proj.latlngs_to_xy(ring)
        if len(xy) >= 4:
            outers.append(Polygon(xy))
    for ring in b.get("inner", []):
        xy = proj.latlngs_to_xy(ring)
        if len(xy) >= 4:
            inners.append(Polygon(xy))
    if not outers:
        return []
    geom = unary_union([p.buffer(0) for p in outers])
    if inners:
        geom = geom.difference(unary_union([p.buffer(0) for p in inners]))
    geom = geom.simplify(0.15, preserve_topology=True)
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom] if isinstance(geom, Polygon) else []
    return [orient(p, 1.0) for p in polys if p.is_valid and p.area > 4.0]


def classify_buildings(buildings, proj, cl: Centerline, clearance_half, clearance_right=None):
    """Return ids of buildings that overlap the road corridor (auto-excluded)."""
    corr = corridor_polygon(cl, clearance_half, clearance_right)
    hits = []
    for b in buildings:
        for poly in _polygons_for_building(b, proj):
            if poly.intersects(corr):
                hits.append(b["id"])
                break
    return hits


def _earcut_polygon(poly):
    """Triangulate a shapely Polygon (with holes). Returns (xy (n,2), tris (m,3)) with CCW triangles."""
    rings = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return np.zeros((0, 2)), np.zeros((0, 3), np.int64)
    flat = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = np.asarray(earcut.triangulate_float64(flat, ends)).reshape(-1, 3).astype(np.int64)
    if len(tri):
        p = flat[tri]
        cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
        tri[cross < 0] = tri[cross < 0][:, [0, 2, 1]]
    return flat, tri


def tile_uv(xy, t):
    return np.column_stack([(xy[:, 0] - t["x0"]) / (t["x1"] - t["x0"]), (t["y1"] - xy[:, 1]) / (t["y1"] - t["y0"])])


ROOF_SLOPE = 0.5          # rise / run (about 27 degrees, a typical 6/12 roof)
OVERHANG = 0.35
STOREY = 2.8
# facade materials (textures.FACADES) and how common they are (Québec suburbs: mostly vinyl siding, a lot of brick)
SIDING_NAMES = ["siding_white", "siding_cream", "siding_beige", "siding_grey", "siding_blue", "siding_sage",
                "siding_yellow", "siding_charcoal", "siding_brick", "siding_brick_brown", "siding_brick_buff",
                "siding_stucco_white", "siding_stucco_sand"]
SIDING_WEIGHTS = [0.16, 0.08, 0.12, 0.12, 0.08, 0.05, 0.04, 0.05, 0.13, 0.08, 0.04, 0.03, 0.02]
SHUTTER_SHARE = 0.35        # houses with shutters beside their windows
FOUNDATION_SHOW = 0.3       # m of concrete foundation above the highest ground at the house
SHINGLE_RGB = {"shingle_charcoal": (62, 64, 68), "shingle_brown": (96, 74, 58), "shingle_grey": (122, 122, 120),
               "shingle_red": (122, 62, 48), "shingle_black": (44, 47, 54)}


def _pick(seed_text, names, weights=None):
    import zlib

    r = (zlib.crc32(str(seed_text).encode()) % 10000) / 10000.0
    w = np.cumsum(weights if weights else np.ones(len(names)) / len(names))
    return names[int(np.searchsorted(w / w[-1], r))]


def _roof_material(b):
    if b.get("roof_mat") in SHINGLE_RGB:
        return b["roof_mat"]
    rgb = b.get("roof_rgb")
    if rgb is None:
        return _pick(b["id"] + "r", list(SHINGLE_RGB))
    rgb = np.asarray(rgb, float)
    return min(SHINGLE_RGB, key=lambda k: float(np.sum((np.asarray(SHINGLE_RGB[k]) - rgb) ** 2)))


def _facade(a, b, base, floor0, eave, shutters=False, specials=None, ground=None, plain=False):
    """Walls along edges a[i] -> b[i]: concrete foundation from base to floor0, storeys from floor0 to eave.
    Each wall gets a whole number of window bays (textures.BAY_M wide, stretched to fit), so no window is ever
    cut at a corner; walls too short for a window are plain. Storeys use texture row A (or B, with shutters).
    specials: {wall index: {bay index: "door" | ("garage", ground z)}}: that ground-floor bay shows the front door
    (row D) or a garage door (row G, from the ground up: no foundation band under it)."""
    from .textures import BAY_M, PLAIN_M, ROW_A, ROW_B, ROW_D, ROW_F, ROW_G, ROW_P, STOREY_M, row_v

    n_st = max(1, int(round((eave - floor0) / STOREY_M)))
    st = (eave - floor0) / n_st
    row = ROW_B if shutters else ROW_A
    vt, vb = row_v(row)
    if plain:                                  # no windows (garages, sheds): the plain wall row, one storey of it
        vt, vb = row_v(ROW_P, 0.0, min(1.0, st / PLAIN_M))
    ft, fb = row_v(ROW_F)
    dt, db = row_v(ROW_D)
    gt, gb = row_v(ROW_G)
    specials = specials or {}
    g0 = base if ground is None else ground
    Vs, UVs = [], []

    def quad(pa, pb, z0, z1, u0, u1, v_top, v_bot):
        Vs.append(np.array([[pa[0], pa[1], z0], [pb[0], pb[1], z0], [pb[0], pb[1], z1], [pa[0], pa[1], z1]]))
        UVs.append(np.array([[u0, v_bot], [u1, v_bot], [u1, v_top], [u0, v_top]]))

    for wi, (pa, pb) in enumerate(zip(a, b)):
        L = float(np.linalg.norm(pb - pa))
        if L < 0.05:
            continue
        nb = int(round(L / BAY_M)) if L >= 2.0 else 0
        sp, zg = {}, {}
        for k, v in specials.get(wi, {}).items():
            if 0 <= k < nb:
                sp[k] = v if isinstance(v, str) else v[0]
                zg[k] = g0 if isinstance(v, str) else float(v[1])        # ground under a garage door
        # foundation, but not under a garage door (its floor is at the ground)
        if floor0 > base + 0.01:
            if nb == 0:
                quad(pa, pb, base, floor0, 0.0, L / BAY_M, ft, fb)
            else:
                u = 0
                for k in sorted(k for k, v in sp.items() if v == "garage") + [nb]:
                    if k > u:
                        t0, t1 = u / nb, k / nb
                        quad(pa + (pb - pa) * t0, pa + (pb - pa) * t1, base, floor0, L * t0 / BAY_M, L * t1 / BAY_M,
                             ft, fb)
                    u = k + 1
        if nb >= 1:
            for kst in range(n_st):
                z1 = eave - kst * st
                lowest = kst == n_st - 1
                if not lowest or not sp:
                    quad(pa, pb, z1 - st, z1, 0.0, float(nb), vt, vb)
                    continue
                # ground floor: runs of window bays, the special bays on their own
                k = 0
                while k < nb:
                    if k in sp:
                        q0, q1 = pa + (pb - pa) * k / nb, pa + (pb - pa) * (k + 1) / nb
                        if sp[k] == "door":
                            quad(q0, q1, z1 - st, z1, 0.0, 1.0, dt, db)
                        else:
                            quad(q0, q1, min(zg[k], z1 - st), z1, 0.0, 1.0, gt, gb)
                        k += 1
                        continue
                    k1 = k
                    while k1 < nb and k1 not in sp:
                        k1 += 1
                    quad(pa + (pb - pa) * k / nb, pa + (pb - pa) * k1 / nb, z1 - st, z1, float(k), float(k1), vt, vb)
                    k = k1
        else:
            pt, _ = row_v(ROW_P)
            _, pb_ = row_v(ROW_P, 0.0, min(1.0, (eave - floor0) / PLAIN_M))
            quad(pa, pb, floor0, eave, 0.0, L / BAY_M, pt, pb_)
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 2))
    m = len(Vs)
    k = np.arange(m) * 4
    F = np.vstack([np.column_stack([k, k + 1, k + 2]), np.column_stack([k, k + 2, k + 3])])
    return np.vstack(Vs), F, np.vstack(UVs)


PARAPET_T = 0.25


def _block_facade(ring, base, floor0, levels, top, kind, street=(), parapet=True, door_wall=None):
    """Walls of a flat-roofed block (ring counter-clockwise). levels: [(z0, z1, "shop" | "office" | "house" |
    "plain")] bottom up; walls in `street` get the shop front (commercial) or roller doors (industrial) on the
    ground floor and door_wall a door in its middle bay; a parapet with a cornice from top to top + CORNICE_M.
    -> (V, F, UV)."""
    from .textures import (BAY_M, BLOCK_M, CORNICE_M, GARAGE_M, PLAIN_M, ROW_A, ROW_C, ROW_D, ROW_F, ROW_G, ROW_K,
                           ROW_P, ROW_S, ROW_T, SHOP_M, STOREY_M, row_v)

    Vs, UVs = [], []

    def quad(pa, pb, z0, z1, u0, u1, v_top, v_bot, back=False):
        P = np.array([[pa[0], pa[1], z0], [pb[0], pb[1], z0], [pb[0], pb[1], z1], [pa[0], pa[1], z1]])
        T = np.array([[u0, v_bot], [u1, v_bot], [u1, v_top], [u0, v_top]])
        if back:
            P, T = P[[1, 0, 3, 2]], T[[1, 0, 3, 2]]
        Vs.append(P)
        UVs.append(T)

    def plain(pa, pb, z0, z1, u0, u1):
        """Plain wall from z1 down to z0 in storey-high pieces (the boards / bricks repeat every storey, so
        pieces and neighbouring quads meet without a seam, however tall the wall)."""
        z = z1
        while z > z0 + 1e-3:
            h = min(STOREY_M, z - z0)
            vt_, vb_ = row_v(ROW_P, 0.0, h / PLAIN_M)
            quad(pa, pb, z - h, z, u0, u1, vt_, vb_)
            z -= h

    rows = {"shop": (ROW_S, SHOP_M), "office": (ROW_C, BLOCK_M), "house": (ROW_A, STOREY_M)}
    a = np.asarray(ring, float)
    b = np.roll(a, -1, axis=0)
    for wi, (pa, pb) in enumerate(zip(a, b)):
        L = float(np.linalg.norm(pb - pa))
        if L < 0.05:
            continue
        nb = int(round(L / BAY_M)) if L >= 2.0 else 0
        ft, fb = row_v(ROW_F)
        if floor0 > base + 0.01:
            quad(pa, pb, base, floor0, 0.0, L / BAY_M, ft, fb)
        for li, (z0, z1, use) in enumerate(levels):
            ground_floor = li == 0
            if ground_floor and wi in street and kind == "commercial":
                use = "shop"
            if nb == 0 or use == "plain" or (ground_floor and kind == "industrial"):
                if ground_floor and kind == "industrial" and wi in street and nb >= 2:
                    # roller doors in up to two middle bays, plain wall around them
                    want = 1 if nb < 4 else 2
                    k0 = (nb - want) // 2
                    gt, gb = row_v(ROW_G)
                    for k in range(nb):
                        q0, q1 = pa + (pb - pa) * k / nb, pa + (pb - pa) * (k + 1) / nb
                        if k0 <= k < k0 + want:
                            zt = min(z0 + GARAGE_M, z1)
                            quad(q0, q1, z0, zt, 0.0, 1.0, gt, gb)
                            if z1 > zt + 0.01:
                                plain(q0, q1, zt, z1, k * L / nb / BAY_M, (k + 1) * L / nb / BAY_M)
                        else:
                            plain(q0, q1, z0, z1, k * L / nb / BAY_M, (k + 1) * L / nb / BAY_M)
                    continue
                plain(pa, pb, z0, z1, 0.0, L / BAY_M)
                continue
            row, hm = rows[use]
            vt, vb = row_v(row)
            if ground_floor and wi == door_wall and nb >= 1:
                kd = nb // 2
                dt, db = row_v(ROW_D)
                for k0, k1 in ((0, kd), (kd + 1, nb)):
                    if k1 > k0:
                        quad(pa + (pb - pa) * k0 / nb, pa + (pb - pa) * k1 / nb, z0, z1, k0, k1, vt, vb)
                q0, q1 = pa + (pb - pa) * kd / nb, pa + (pb - pa) * (kd + 1) / nb
                zt = min(z0 + STOREY_M, z1)
                quad(q0, q1, z0, zt, 0.0, 1.0, dt, db)
                if z1 > zt + 0.01:
                    plain(q0, q1, zt, z1, 0.0, 1.0)
                continue
            quad(pa, pb, z0, z1, 0.0, float(nb), vt, vb)
        z_top = levels[-1][1] if levels else floor0
        if top > z_top + 0.01:
            plain(pa, pb, z_top, top, 0.0, L / BAY_M)
        if parapet:                            # a 0.25 m thick parapet: cornice outside, plain inside, a cap
            kt, kb = row_v(ROW_K)
            quad(pa, pb, top, top + CORNICE_M, 0.0, L / BAY_M, kt, kb)
            d = (pb - pa) / L
            inn = np.array([-d[1], d[0]]) * PARAPET_T
            vt, vb = row_v(ROW_P, 0.0, min(1.0, CORNICE_M / PLAIN_M))
            quad(pa + inn, pb + inn, top, top + CORNICE_M, 0.0, L / BAY_M, vt, vb, back=True)
            tt_, tb_ = row_v(ROW_T)
            zc = top + CORNICE_M
            Vs.append(np.array([[pa[0], pa[1], zc], [pb[0], pb[1], zc], [pb[0] + inn[0], pb[1] + inn[1], zc],
                                [pa[0] + inn[0], pa[1] + inn[1], zc]]))
            UVs.append(np.array([[0.0, tb_], [L / BAY_M, tb_], [L / BAY_M, tt_], [0.0, tt_]]))
    if not Vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 2))
    m = len(Vs)
    k = np.arange(m) * 4
    F = np.vstack([np.column_stack([k, k + 1, k + 2]), np.column_stack([k, k + 2, k + 3])])
    return np.vstack(Vs), F, np.vstack(UVs)


def _block_levels(b, fp, kind, floor0):
    """[(z0, z1, use)] of a flat-roofed building and the roof height: storeys from building:levels / height,
    else typical ones (shops: one 4 m storey, or 4 m + 3.2 m on big footprints; apartments: 3 storeys of 2.8 m;
    industrial: one 6 m hall)."""
    from .textures import BLOCK_M, SHOP_M, STOREY_M

    tags = b.get("tags") or {}

    def num(k):
        try:
            return float(str(tags.get(k, "")).split(";")[0].replace("m", "").strip())
        except ValueError:
            return 0.0

    levels = num("building:levels")
    height = num("height") or num("building:height")
    area = fp.area
    if kind == "industrial":
        h = height or (levels * 4.5 if levels else (6.0 if area < 3000 else 8.0))
        return [(floor0, floor0 + h, "plain")], floor0 + h
    if kind == "commercial":
        g, up, use = SHOP_M, BLOCK_M, "office"
        n = int(levels) if levels >= 1 else (1 if area < 1000 else 2)
        if height and not levels:
            n = max(1, 1 + int(round((height - g) / up)))
    else:                                                  # apartments / big houses
        g, up, use = STOREY_M, STOREY_M, "house"
        n = int(levels) if levels >= 1 else (3 if kind == "block" else max(1, int(round((float(b.get("height", 6)) or 6) / 3.0))))
        if height and not levels:
            n = max(1, int(round(height / up)))
    n = int(np.clip(n, 1, 30))
    out = [(floor0, floor0 + g, use)]
    z = floor0 + g
    for _ in range(n - 1):
        out.append((z, z + up, use))
        z += up
    return out, z


def _street_walls(ring, road_tree, max_dist=60.0):
    """Indices of the walls (ring counter-clockwise) that face a road within max_dist."""
    if road_tree is None:
        return set()
    a = np.asarray(ring, float)
    b = np.roll(a, -1, axis=0)
    out = set()
    for i in range(len(a)):
        L = float(np.linalg.norm(b[i] - a[i]))
        if L < 3.0:
            continue
        d = (b[i] - a[i]) / L
        nout = np.array([d[1], -d[0]])
        mid = (a[i] + b[i]) / 2
        dist, j = road_tree.query(mid + nout * 1.0)
        to = road_tree.data[j] - mid
        if dist < max_dist and float(np.dot(to, nout)) > 0.5 * np.linalg.norm(to):
            out.add(i)
    return out


def _gable_uv(pa, pb, eave, zr):
    """UVs of a gable triangle (pa, pb at the eave, apex at zr) on the plain wall row."""
    from .textures import BAY_M, PLAIN_M, ROW_P, row_v

    L = float(np.linalg.norm(np.asarray(pb) - np.asarray(pa)))
    pt, _ = row_v(ROW_P)
    _, pe = row_v(ROW_P, 0.0, min(1.0, (zr - eave) / PLAIN_M))
    return [(0.0, pe), (L / BAY_M, pe), (L / BAY_M / 2, pt)]


def _quad(p0, p1, p2, p3, uv):
    V = np.array([p0, p1, p2, p3], float)
    return V, np.array([[0, 1, 2], [0, 2, 3]]), np.asarray(uv, float)


def _tri(p0, p1, p2, uv):
    return np.array([p0, p1, p2], float), np.array([[0, 1, 2]]), np.asarray(uv, float)


def _pitched_house(rect_xy, base, eave, ridge, floor0=None, shutters=False, specials=None, ground=None,
                   allow_hip=True, plain=False):
    """Gable roof (or hip roof for squarish houses) on a rectangle. Returns (walls, gables, roof) lists
    of (V, F, UV) pieces. Ridge runs along the long side. specials: callable(corners (4, 2) counter-clockwise)
    -> _facade specials for those walls (doors / garage doors), or None. allow_hip=False: always a gable
    (wings of an L / T house: their end meets the main block)."""
    c = rect_xy.mean(0)
    e1, e2 = rect_xy[1] - rect_xy[0], rect_xy[3] - rect_xy[0]
    if np.linalg.norm(e1) >= np.linalg.norm(e2):
        u, v = e1, e2
    else:
        u, v = e2, e1
    Lu, Lv = np.linalg.norm(u), np.linalg.norm(v)
    u, v = u / Lu, v / Lv
    hu, hv = Lu / 2, Lv / 2
    hip = allow_hip and Lu / Lv < 1.35
    walls, gables, roof = [], [], []

    def P(a, b):
        return c + u * a + v * b

    corners = [P(-hu, -hv), P(hu, -hv), P(hu, hv), P(-hu, hv)]          # counter-clockwise
    d1, d2 = corners[1] - corners[0], corners[2] - corners[1]
    if d1[0] * d2[1] - d1[1] * d2[0] < 0:
        corners = corners[::-1]
        v = -v
    a = np.array(corners)
    sp = specials(a) if callable(specials) else None
    walls.append(_facade(a, np.roll(a, -1, axis=0), base, base if floor0 is None else floor0, eave, shutters,
                         specials=sp, ground=ground, plain=plain))
    o = OVERHANG
    ze = eave - o * ROOF_SLOPE                                          # eave line drops with the overhang
    zr = ridge
    if hip:
        rl = max(hu - hv, 0.0)                                            # half ridge length
        R1, R2 = np.r_[P(-rl, 0), zr], np.r_[P(rl, 0), zr]
        E = [np.r_[P(-hu - o, -hv - o), ze], np.r_[P(hu + o, -hv - o), ze],
             np.r_[P(hu + o, hv + o), ze], np.r_[P(-hu - o, hv + o), ze]]
        sl = (hv + o) * np.sqrt(1 + ROOF_SLOPE ** 2) / 3.0
        roof.append(_quad(E[0], E[1], R2, R1, [(0, 0), ((Lu + 2 * o) / 3, 0), ((hu + rl + o) / 3, -sl), ((hu - rl + o) / 3, -sl)]))
        roof.append(_quad(E[2], E[3], R1, R2, [(0, 0), ((Lu + 2 * o) / 3, 0), ((hu + rl + o) / 3, -sl), ((hu - rl + o) / 3, -sl)]))
        roof.append(_tri(E[1], E[2], R2, [(0, 0), ((Lv + 2 * o) / 3, 0), ((hv + o) / 3, -sl)]))
        roof.append(_tri(E[3], E[0], R1, [(0, 0), ((Lv + 2 * o) / 3, 0), ((hv + o) / 3, -sl)]))
    else:
        R1, R2 = np.r_[P(-hu - o, 0), zr], np.r_[P(hu + o, 0), zr]
        E = [np.r_[P(-hu - o, -hv - o), ze], np.r_[P(hu + o, -hv - o), ze],
             np.r_[P(hu + o, hv + o), ze], np.r_[P(-hu - o, hv + o), ze]]
        sl = (hv + o) * np.sqrt(1 + ROOF_SLOPE ** 2) / 3.0
        L3 = (Lu + 2 * o) / 3
        roof.append(_quad(E[0], E[1], R2, R1, [(0, 0), (L3, 0), (L3, -sl), (0, -sl)]))
        roof.append(_quad(E[2], E[3], R1, R2, [(0, 0), (L3, 0), (L3, -sl), (0, -sl)]))
        # gable triangles on the short walls (plain siding band of the texture, no windows)
        for (pa, pb) in ((corners[1], corners[2]), (corners[3], corners[0])):
            mid = (pa + pb) / 2
            gables.append(_tri(np.r_[pa, eave], np.r_[pb, eave], np.r_[mid, zr], _gable_uv(pa, pb, eave, zr)))
        # roof underside at the gable overhang is skipped (not visible from the road)
    return walls, gables, roof


def _house_heights(b, poly):
    rect = poly.minimum_rotated_rectangle
    xy = np.asarray(rect.exterior.coords)
    short = min(np.linalg.norm(xy[1] - xy[0]), np.linalg.norm(xy[2] - xy[1]))
    rise = ROOF_SLOPE * (short / 2)
    if b.get("eave"):
        eave = float(b["eave"])
        ridge = max(float(b.get("height", eave + rise)), eave + 0.5)
        return eave, min(ridge, eave + rise * 1.6)
    h = float(b.get("height", 0) or 0)
    tags = b.get("tags", {})
    explicit = any(k in tags for k in ("height", "building:levels", "building:height"))
    if explicit and h > 0:
        return max(2.4, h - rise), max(h, 2.9)
    area = poly.area
    if area < 40:
        eave = 2.4
    elif area < 300:
        eave = 2.9            # bungalow (incl. attached garage)
    elif area < 450:
        eave = 5.6            # two storeys
    else:
        eave = max(6.0, min(h, 12.0))
    return eave, eave + rise


def plan_buildings(buildings, proj, skip_ids=()):
    """{building id: [(kind, geometry, footprint polygon)]}: houses.shape() of each footprint ("rects": list of
    rectangles, main block first; "flat": a squared outline)."""
    from shapely.geometry import Polygon as _Poly
    from shapely.ops import unary_union as _uu

    from . import houses

    plans = {}
    for b in buildings:
        if b["id"] in skip_ids:
            continue
        out = []
        for poly in _polygons_for_building(b, proj):
            try:
                kind, geo = houses.shape(poly)
            except Exception:
                kind, geo = "flat", poly
            fp = _uu([_Poly(r) for r in geo]).buffer(0) if kind == "rects" else geo
            out.append((kind, geo, fp))
        plans[b["id"]] = out
    return plans


def footprints(plans):
    """All footprint polygons of a plan (for masks, clearances, driveway cuts)."""
    out = []
    for parts in plans.values():
        for _, _, fp in parts:
            out += list(getattr(fp, "geoms", [fp]))
    return [p for p in out if not p.is_empty]


def build_buildings(buildings, proj, tm: TerrainModel, skip_ids: set, collide=True, plans=None, roads_xy=None,
                    driveways=None):
    """Houses: siding walls with windows, gable/hip shingle roofs on the rectangles of houses.shape() (an L / T
    house: a main block and gabled wings), a front door facing the nearest road (roads_xy: road centre points)
    with concrete steps, a garage door where a driveway (polygons) meets the house. Big / irregular buildings:
    squared outline, flat roof."""
    from scipy.spatial import cKDTree
    from shapely.geometry import Polygon as _Poly

    from . import houses

    plans = plans if plans is not None else plan_buildings(buildings, proj, skip_ids)
    road_tree = cKDTree(np.asarray(roads_xy)) if roads_xy is not None and len(roads_xy) else None
    dws = [d for d in (driveways or []) if d is not None and not d.is_empty]
    wall_b, roof_b = {}, {}
    prefix_w = "1WALL_BLD" if collide else "BLD_WALLS"
    steps_b = MeshBuilder("BLD_STEPS", "concrete")

    def mb(store, prefix, mat):
        if mat not in store:
            store[mat] = MeshBuilder(f"{prefix}_{mat.split('_', 1)[1].upper()}", mat)
        return store[mat]

    def ground_at(x, y):
        return float(tm.surface_height(np.array([x]), np.array([y]))[0])

    from shapely.geometry.polygon import orient as _orient

    from .textures import BAY_M, ROW_T, row_v

    tt, tb = row_v(ROW_T)

    def trim_uv(UV):                          # trim pieces: the plain trim band of the facade atlas
        UV = np.asarray(UV, float)
        return np.column_stack([UV[:, 0] / BAY_M, np.full(len(UV), (tt + tb) / 2)])

    count = 0
    n_doors = n_garages = n_porches = n_canopies = 0
    kinds = {}
    for b in buildings:
        if b["id"] in skip_ids or b["id"] not in plans:
            continue
        parts_b = plans[b["id"]]
        area = float(sum(fp.area for _, _, fp in parts_b))
        bk = houses.building_kind(b, area)
        kinds[bk] = kinds.get(bk, 0) + 1
        if bk == "commercial" or bk == "block":
            siding = _pick(b["id"], [n for n in SIDING_NAMES if "brick" in n or "stucco" in n])
        elif bk == "industrial":
            siding = _pick(b["id"], ["siding_grey", "siding_white", "siding_charcoal", "siding_beige"])
        else:
            siding = _pick(b["id"], SIDING_NAMES, SIDING_WEIGHTS)
        shutters = bk == "house" and _pick(b["id"] + "s", [True, False], [SHUTTER_SHARE, 1 - SHUTTER_SHARE])
        porch = bk == "house" and _pick(b["id"] + "p", [True, False], [houses.PORCH_SHARE, 1 - houses.PORCH_SHARE])
        shingle = _roof_material(b)
        tags = b.get("tags") or {}
        for kind, geo, fp in parts_b:
            ext = np.asarray(fp.exterior.coords)[:-1] if fp.geom_type == "Polygon" else \
                np.vstack([np.asarray(g.exterior.coords)[:-1] for g in fp.geoms])
            if len(ext) < 3:
                continue
            gz = tm.surface_height(ext[:, 0], ext[:, 1])
            ground = float(gz.max())
            base = float(gz.min()) - 0.6
            flat_kind = bk if bk in ("commercial", "block", "industrial") else ("block" if kind != "rects" else None)
            if flat_kind:
                floor0 = ground + (0.15 if flat_kind != "block" else FOUNDATION_SHOW)
                levels, top = _block_levels(b, fp, flat_kind, floor0)
                for poly in getattr(fp, "geoms", [fp]):
                    poly = _orient(poly, 1.0)
                    outer = np.asarray(poly.exterior.coords)[:-1]
                    street = _street_walls(outer, road_tree)
                    door_wall = None
                    if street and flat_kind != "industrial":
                        door_wall = max(street, key=lambda i: float(np.linalg.norm(outer[(i + 1) % len(outer)] - outer[i])))
                    V, F, UV = _block_facade(outer, base, floor0, levels, top, flat_kind, street,
                                             parapet=flat_kind != "industrial", door_wall=door_wall)
                    mb(wall_b, prefix_w, siding).add(V, F, UV)
                    for r in poly.interiors:
                        ring = np.asarray(r.coords)[:-1]
                        V, F, UV = _block_facade(ring, base, floor0, levels, top, flat_kind, (),
                                                 parapet=flat_kind != "industrial")
                        mb(wall_b, prefix_w, siding).add(V, F, UV)
                    n_doors += door_wall is not None
                    if flat_kind == "industrial":
                        n_garages += sum(1 for i in street
                                         if np.linalg.norm(outer[(i + 1) % len(outer)] - outer[i]) >= 2 * BAY_M)
                    flat, tri = _earcut_polygon(poly)
                    if len(tri):
                        RV = np.column_stack([flat, np.full(len(flat), top)])
                        mb(roof_b, "BLD_ROOF", "roof_flat").add(RV, tri, flat / 6.0)
                count += 1
                continue
            main = _Poly(geo[0])
            if bk == "garage":
                eave_h, ridge_h = 2.6, 2.6 + min(1.2, ROOF_SLOPE * 0.6 * min(
                    np.linalg.norm(np.asarray(geo[0][1]) - np.asarray(geo[0][0])),
                    np.linalg.norm(np.asarray(geo[0][2]) - np.asarray(geo[0][1]))) / 2)
                floor0 = ground + 0.05
            else:
                eave_h, ridge_h = _house_heights(b, main)
                floor0 = min(ground + FOUNDATION_SHOW, ground + eave_h - 2.2)
            for j, rect in enumerate(geo):
                others = [r for jj, r in enumerate(geo) if jj != j]
                rp = _Poly(rect)
                xy = np.asarray(rp.minimum_rotated_rectangle.exterior.coords)[:4]
                short = min(np.linalg.norm(xy[1] - xy[0]), np.linalg.norm(xy[2] - xy[1]))
                ridge = ground + min(ridge_h, eave_h + ROOF_SLOPE * short / 2) if j else ground + ridge_h
                got = {}

                if bk == "garage":
                    n_want = -1 if str(tags.get("building", "")) == "garages" else (1 if short < 5.5 else 2)

                    def choose(corners, got=got, n_want=n_want, j=j):
                        sp = houses.garage_specials(corners, road_tree, dws, n_want, ground_at) if j == 0 else {}
                        got["sp"], got["c"] = sp, corners
                        return sp
                else:
                    def choose(corners, j=j, others=others, got=got):
                        sp = houses.door_specials(corners, road_tree, dws, others, ground_at, door=(j == 0))
                        got["sp"], got["c"] = sp, corners
                        return sp

                walls, gables, roof = _pitched_house(np.asarray(rect), base, ground + eave_h, ridge,
                                                     floor0=floor0, shutters=shutters, specials=choose,
                                                     ground=ground, allow_hip=(j == 0 and bk != "garage"),
                                                     plain=bk == "garage")
                for V, F, UV in walls + gables:
                    mb(wall_b, prefix_w, siding).add(V, F, UV)
                for V, F, UV in roof:
                    mb(roof_b, "BLD_ROOF", shingle).add(V, F, UV)
                sp = got.get("sp") or {}
                offsets = {}
                if bk == "house" and any(v == "door" for w in sp.values() for v in w.values()):
                    roofs, trims, offsets = houses.door_extras(got.get("c", rect), sp, floor0, ground, porch,
                                                               eave=ground + eave_h)
                    for V, F, UV in roofs:
                        mb(roof_b, "BLD_ROOF", shingle).add(V, F, UV)
                    for V, F, UV in trims:
                        mb(wall_b, prefix_w, siding).add(V, F, trim_uv(UV))
                    n_porches += bool(offsets)
                    n_canopies += not offsets
                for V, F, UV in houses.steps(got.get("c", rect), sp, floor0, ground - 0.05, offsets):
                    steps_b.add(V, F, UV)
                n_doors += sum(1 for w in sp.values() for v in w.values() if v == "door")
                n_garages += sum(1 for w in sp.values() for v in w.values() if v != "door")
            count += 1
    out = []
    for store in (wall_b, roof_b):
        for m in store.values():
            out += m.done()
    out += steps_b.done()
    build_buildings.last_stats = {"doors": n_doors, "garages": n_garages, "porches": n_porches,
                                  "canopies": n_canopies, "kinds": kinds}
    return out, count


SIDE_ROAD_SKIP = {"footway", "path", "cycleway", "steps", "bridleway", "corridor", "via_ferrata"}



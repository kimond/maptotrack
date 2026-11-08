"""Area mode (like Map2Lap's areas): every drivable street inside a rectangle becomes road.

The streets of OpenStreetMap are turned into a network:

* ways are cut at the rectangle and split where they meet (shared nodes); pieces that simply continue each
  other (same class and width) are joined again into *chains*, so a street is one long line between
  junctions, as a track would be;
* every chain gets the full track treatment where the photo covers it: width measured on the photo (and
  smoothed), markings read on the photo (or the usual layout on main roads), crown;
* heights are smoothed over the whole network at once (a screened-Poisson smoothing on the street graph), so
  the streets meeting at a junction have exactly the same height there and no step appears;
* the asphalt is the union of the chains plus rounded kerb corners (fillets) at the junctions;
* one height function for the whole road surface (`Network.query`): near a junction it blends the streets that
  meet there and their crown fades out (flat crossings).

Nothing here depends on a track centreline: the pipeline uses `NetTerrain` (a TerrainModel over the network)
and `ground.build_tin_net` instead of the track versions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .track import Centerline, build_centerline, resample

MAX_SIDE_M = 6000.0          # largest side of the area's bounding box
MAX_AREA_KM2 = 10.0          # largest area (AC loads the whole track at once)
BIG_AREA_KM2 = 4.0           # above this (bounding box), photo / ground resolution scale down to keep memory and
                             # the .kn5 size as for a 2 x 2 km area
DRIVABLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "living_street", "service", "road", "raceway", "motorway_link", "trunk_link", "primary_link",
            "secondary_link", "tertiary_link"}
SKIP_SERVICE = {"driveway", "parking_aisle", "drive-through", "emergency_access"}   # the photo finds driveways / lots
MARKED = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link", "primary_link",
          "secondary_link", "tertiary_link"}           # standard markings when the photo can't tell
KEY = 20.0                   # 5 cm: vertices this close are the same node
FILLET_M = 7.0               # kerb radius at the street corners (Québec residential corners: 6-9 m)
JUNCTION_PAD = 1.5           # m beyond the widest street at a junction: flat (no crown), lines stop
CROWN_FADE_M = 10.0          # m over which the crown comes back after a junction
JUNCTION_BLEND_M = 10.0      # m beyond a junction over which its plane hands over to the streets' own surface
MIN_CHAIN_M = 4.2            # shorter pieces are only asphalt (no own centreline)
MIN_MEASURE_M = 25.0         # shorter chains take the width measured on their neighbours / class


@dataclass
class Chain:
    cl: Centerline
    hl: np.ndarray
    hr: np.ndarray
    conf: np.ndarray
    highway: str
    width: float              # nominal width (OSM tags / class)
    oneway: int               # 1 = along the chain, -1 = against, 0 = both ways
    ends: tuple               # (node key at the start, node key at the end); None on loops
    ids: list = field(default_factory=list)
    measured: bool = False
    mk: object = None          # markings.Markings


def _key(p):
    return (int(round(p[0] * KEY)), int(round(p[1] * KEY)))


def _clean(xy):
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.05]
    return xy[keep]


def ways_in(roads, proj, bounds, inset=1.0, clip=None):
    """Drivable OSM ways cut to the area (the rectangle `bounds`, or the polygon `clip`): [(xy, road record)]."""
    from shapely.geometry import LineString, box

    x0, y0, x1, y1 = bounds
    clip = box(x0 + inset, y0 + inset, x1 - inset, y1 - inset) if clip is None else clip.buffer(-inset)
    out = []
    for r in roads:
        if r["highway"] not in DRIVABLE or r.get("tunnel") or r.get("service") in SKIP_SERVICE:
            continue
        xy = _clean(proj.latlngs_to_xy(r["coords"]))
        if len(xy) < 2:
            continue
        ls = LineString(xy)
        if not ls.intersects(clip):
            continue
        if clip.contains(ls):
            out.append((xy, r))
            continue
        g = ls.intersection(clip)
        for gg in getattr(g, "geoms", [g]):
            if gg.geom_type == "LineString" and gg.length > 1.0:
                out.append((_clean(np.asarray(gg.coords)), r))
    return out


def _edges(ways):
    """Split the ways where they meet -> [(xy, road, start key, end key)] and the node degrees."""
    count = {}
    for xy, _ in ways:
        for p in xy:
            k = _key(p)
            count[k] = count.get(k, 0) + 1
    edges = []
    for xy, r in ways:
        keys = [_key(p) for p in xy]
        cut = [0] + [i for i in range(1, len(xy) - 1) if count[keys[i]] >= 2] + [len(xy) - 1]
        for a, b in zip(cut[:-1], cut[1:]):
            if b > a:
                edges.append((xy[a:b + 1], r, keys[a], keys[b]))
    degree = {}
    for _, _, ka, kb in edges:
        degree[ka] = degree.get(ka, 0) + 1
        degree[kb] = degree.get(kb, 0) + 1
    return edges, degree


def _same(ra, rb, flip_b):
    """Two ways that can be one chain: same class, about the same width, same one-way rule."""
    if ra["highway"] != rb["highway"] or abs(ra["width"] - rb["width"]) > 0.6:
        return False
    ob = -(rb.get("oneway") or 0) if flip_b else (rb.get("oneway") or 0)
    return (ra.get("oneway") or 0) == ob


def chains_of(edges, degree):
    """Join edges through nodes where exactly two of them meet (and they match) -> [(xy, road, oneway, ka, kb,
    ids, loop)]."""
    at = {}
    for i, (_, _, ka, kb) in enumerate(edges):
        at.setdefault(ka, []).append((i, 0))
        at.setdefault(kb, []).append((i, 1))
    used = np.zeros(len(edges), bool)
    out = []
    for i0 in range(len(edges)):
        if used[i0]:
            continue
        used[i0] = True
        xy0, r0, ka, kb = edges[i0]
        parts = [xy0]
        ids = [r0["id"]]
        oneway = r0.get("oneway") or 0
        loop = False

        def extend(end_key, forward):
            nonlocal loop
            k = end_key
            while degree.get(k, 0) == 2:
                nxt = [(j, e) for j, e in at[k] if not used[j]]
                if not nxt:
                    if any(j == i0 for j, _ in at[k]) and len(parts) > 1:
                        loop = True
                    break
                j, e = nxt[0]
                xy, r, a, b = edges[j]
                # orient edge j so that it continues from k
                flip = (e == 1) if forward else (e == 0)
                if not _same(r0, r, flip):
                    break
                used[j] = True
                seg = xy[::-1] if flip else xy
                if forward:
                    parts.append(seg[1:])
                    k = a if flip else b
                else:
                    parts.insert(0, seg[:-1])
                    k = b if flip else a
                ids.append(r["id"])
            return k

        kb2 = extend(kb, True)
        ka2 = ka if loop else extend(ka, False)
        xy = np.vstack(parts)
        # a closed way is a loop only when nothing else meets it at its closing node (else: a street from that
        # junction back to it, like a crescent)
        if np.linalg.norm(xy[0] - xy[-1]) < 0.06 and len(xy) > 3 and degree.get(kb2, 0) <= 2:
            loop = True
        if loop:
            xy = xy[:-1] if np.linalg.norm(xy[0] - xy[-1]) < 0.06 else xy
        out.append((xy, r0, oneway, None if loop else ka2, None if loop else kb2, ids, loop))
    return out


def _flip_cl(cl):
    return Centerline(p=cl.p[::-1].copy(), z=cl.z[::-1].copy(), t=-cl.t[::-1], n=-cl.n[::-1],
                      s=(cl.length - cl.s[::-1]) if not cl.loop else cl.s.copy(), loop=cl.loop, length=cl.length,
                      spacing=cl.spacing, radius=-cl.radius[::-1])


class Network:
    """All the streets of the area as chains, their junctions, the asphalt and one surface height function."""

    def __init__(self, chains, nodes, degree, stubs, camber=0.06, bounds=None, turning=(), clip=None):
        self.chains = chains
        self.nodes = nodes                    # key -> xy
        self.degree = degree
        self.stubs = stubs                    # short pieces: [(xy, half width)]
        self.camber = camber
        self.bounds = bounds
        self.clip = clip                      # the area's polygon (None = the rectangle `bounds`)
        self.turning = list(turning)          # cul-de-sac turning circles: [(xy, radius)]
        self.junctions = {}                   # key -> (xy, radius) where 3+ streets meet
        for k, deg in degree.items():
            if deg >= 3:
                hw = [max(float(c.hl[0] if c.ends[0] == k else c.hl[-1]), float(c.hr[0] if c.ends[0] == k else c.hr[-1]))
                      for c in chains if c.ends and k in c.ends]
                self.junctions[k] = (nodes[k], (max(hw) if hw else 4.0) + JUNCTION_PAD)
        self._dense()
        self._junction_planes()
        self.asphalt = self._asphalt()

    # ------------------------------------------------------------------ geometry
    def _dense(self):
        from scipy.spatial import cKDTree

        P, Z, HL, HR, N, CID, S, L = [], [], [], [], [], [], [], []
        for ci, c in enumerate(self.chains):
            cl = c.cl
            try:
                p = resample(cl.p, 1.0, cl.loop)
            except ValueError:
                p = cl.p
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            ss = np.concatenate([[0], np.cumsum(seg)])            # same arc length as the centreline's s
            S_ = np.concatenate([cl.s, [cl.length]]) if cl.loop else cl.s

            def at(v):
                vv = np.concatenate([v, v[:1]]) if cl.loop else v
                return np.interp(ss, S_, vv)

            n = np.column_stack([at(cl.n[:, 0]), at(cl.n[:, 1])])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            P.append(p), Z.append(at(cl.z)), HL.append(at(c.hl)), HR.append(at(c.hr)), N.append(n)
            CID.append(np.full(len(p), ci)), S.append(ss), L.append(np.full(len(p), cl.length))
        self.P = np.vstack(P) if P else np.zeros((0, 2))
        self.Z = np.concatenate(Z) if Z else np.zeros(0)
        self.HL = np.concatenate(HL) if HL else np.zeros(0)
        self.HR = np.concatenate(HR) if HR else np.zeros(0)
        self.N = np.vstack(N) if N else np.zeros((0, 2))
        self.CID = np.concatenate(CID) if CID else np.zeros(0, int)
        self.S = np.concatenate(S) if S else np.zeros(0)
        self.L = np.concatenate(L) if L else np.zeros(0)
        self.tree = cKDTree(self.P) if len(self.P) else None
        # crown fade near junctions
        self.F = np.ones(len(self.P))
        flat = list(self.junctions.values()) + list(self.turning)
        if flat and len(self.P):
            J = np.array([v[0] for v in flat])
            R = np.array([v[1] for v in flat])
            d, j = cKDTree(J).query(self.P)
            self.F = np.clip((d - R[j]) / CROWN_FADE_M, 0, 1)
            self.F = self.F * self.F * (3 - 2 * self.F)

    def _junction_planes(self):
        """Each junction's surface: a plane fitted to the heights of the streets that meet there. Streets meeting
        on a hillside disagree (one runs up the slope, the other across it): the plane takes the warp, and hands
        over to each street's own surface over JUNCTION_BLEND_M, so there is no ridge or step anywhere."""
        from scipy.spatial import cKDTree

        self.jc, self.jr, self.jplane = np.zeros((0, 2)), np.zeros(0), np.zeros((0, 3))
        if not self.junctions or not len(self.P):
            self.jtree = None
            return
        C, R, PL = [], [], []
        for xy, r in self.junctions.values():
            reach = r + JUNCTION_BLEND_M
            idx = self.tree.query_ball_point(xy, reach)
            if len(idx) < 3:
                continue
            q = self.P[idx] - xy
            d = np.linalg.norm(q, axis=1)
            w = np.clip(1.0 - d / (reach + 1e-6), 0.05, 1.0)
            A = np.column_stack([np.ones(len(q)), q]) * w[:, None]
            coef, *_ = np.linalg.lstsq(A, self.Z[idx] * w, rcond=None)
            C.append(xy), R.append(r), PL.append(coef)
        if C:
            self.jc, self.jr, self.jplane = np.array(C, float), np.array(R, float), np.array(PL, float)
            self.jtree = cKDTree(self.jc)
        else:
            self.jtree = None

    def _asphalt(self):
        from shapely.geometry import LineString, Point, box
        from shapely.ops import unary_union

        from .meshes import corridor_polygon

        geoms = [corridor_polygon(c.cl, c.hl, c.hr) for c in self.chains]
        for k, deg in self.degree.items():
            if deg >= 2 and k in self.nodes:
                hw = [max(float(c.hl[0] if c.ends[0] == k else c.hl[-1]), float(c.hr[0] if c.ends[0] == k else c.hr[-1]))
                      for c in self.chains if c.ends and k in c.ends]
                if hw:
                    geoms.append(Point(self.nodes[k]).buffer(max(hw), resolution=12))
        for xy, h in self.stubs:
            geoms.append(LineString(xy).buffer(h, cap_style=2))
        for xy, r in self.turning:
            geoms.append(Point(xy).buffer(r, resolution=16))
        if not geoms:
            return None
        g = unary_union(geoms)
        g = g.buffer(FILLET_M, join_style=1, resolution=8).buffer(-FILLET_M, join_style=1, resolution=8)
        if self.clip is not None:
            g = g.intersection(self.clip)
        elif self.bounds is not None:
            g = g.intersection(box(*self.bounds))
        return g.simplify(0.05).buffer(0)

    def corridor(self, extra=0.0):
        return self.asphalt.buffer(extra) if extra > 0 else self.asphalt

    @property
    def length(self):
        return float(sum(c.cl.length for c in self.chains))

    # ------------------------------------------------------------------ surface
    def query(self, x, y, k=10, chunk=200000):
        """Road surface height (crown included, flat at junctions) and the distance beyond the nearest asphalt
        edge (0 on the road), blended over the nearest streets so it is continuous everywhere."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        q = np.column_stack([x.ravel(), y.ravel()])
        zr = np.empty(len(q))
        em = np.empty(len(q))
        k = min(k, len(self.P))
        for a in range(0, len(q), chunk):
            qq = q[a:a + chunk]
            d, ii = self.tree.query(qq, k=k)
            if k == 1:
                d, ii = d[:, None], ii[:, None]
            dx = qq[:, None, :] - self.P[ii]
            lat = (dx * self.N[ii]).sum(-1)
            hl, hr = self.HL[ii], self.HR[ii]
            half = np.where(lat >= 0, hl, hr)
            e = np.maximum(d - half, 0.0)
            tc = np.clip((hl - lat) / np.maximum(hl + hr, 1e-6), 0, 1)
            zi = self.Z[ii] + self.camber * (1 - (2 * tc - 1) ** 2) * self.F[ii]
            dk = d[:, -1:] * 1.0001 + 1e-6
            w = (1.0 / (e + 1.0) ** 2) * (1 - d / dk) ** 2 + 1e-12
            z = (w * zi).sum(1) / w.sum(1)
            if self.jtree is not None:
                # junction planes (the two nearest junctions, so two close ones blend too)
                kj = min(2, len(self.jc))
                dj, jj = self.jtree.query(qq, k=kj)
                if kj == 1:
                    dj, jj = dj[:, None], jj[:, None]
                t = np.clip(1.0 - (dj - self.jr[jj]) / JUNCTION_BLEND_M, 0, 1)
                t = t * t * (3 - 2 * t)
                tot = t.sum(1, keepdims=True)
                t = t / np.maximum(tot, 1.0)
                pz = (self.jplane[jj, 0] + self.jplane[jj, 1] * (qq[:, None, 0] - self.jc[jj, 0])
                      + self.jplane[jj, 2] * (qq[:, None, 1] - self.jc[jj, 1]))
                z = z * (1 - t.sum(1)) + (t * pz).sum(1)
            zr[a:a + chunk] = z
            em[a:a + chunk] = e.min(1)
        return zr.reshape(x.shape), em.reshape(x.shape)

    def inside_other(self, xy, ci, margin=0.25):
        """Points on chain ci that lie on the asphalt of another street (not just at its end, where it meets
        this one)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        k = min(12, len(self.P))
        d, ii = self.tree.query(xy, k=k)
        if k == 1:
            d, ii = d[:, None], ii[:, None]
        other = self.CID[ii] != ci
        end = (self.S[ii] < 0.9) | (self.S[ii] > self.L[ii] - 0.9)
        lat = ((xy[:, None, :] - self.P[ii]) * self.N[ii]).sum(-1)
        half = np.where(lat >= 0, self.HL[ii], self.HR[ii])
        hit = other & ~end & (np.abs(lat) < half + margin) & (d < half + margin + 1.0)
        return hit.any(1)

    def near_junction(self, xy, ci):
        """Points of chain ci inside a junction where lines stop: 4+ streets, or a side street of a T."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = np.zeros(len(xy), bool)
        c = self.chains[ci]
        if not c.ends:
            return out
        for e_i, k in enumerate(c.ends):
            if k not in self.junctions:
                continue
            jxy, R = self.junctions[k]
            if self.degree[k] == 3 and self._through(k, ci, e_i):
                continue
            out |= np.linalg.norm(xy - jxy, axis=1) < R
        return out

    def _through(self, k, ci, e_i):
        """Is chain ci (its end e_i at node k) part of the straight-through street of a T junction?"""
        def away(c, end):
            p = c.cl.p
            j = min(3, len(p) - 1)
            v = (p[j] - p[0]) if end == 0 else (p[-1 - j] - p[-1])
            return v / max(np.linalg.norm(v), 1e-9)

        arms = []
        for cj, c in enumerate(self.chains):
            if c.ends:
                for e, kk in enumerate(c.ends):
                    if kk == k:
                        arms.append((cj, e, away(c, e)))
        if len(arms) != 3:
            return False
        best, pair = 1.0, None
        for a in range(3):
            for b in range(a + 1, 3):
                cos = float(np.dot(arms[a][2], arms[b][2]))
                if cos < best:
                    best, pair = cos, (a, b)
        if pair is None or best > -0.85:                       # no straight pair (Y junction): all stop
            return False
        return any(arms[i][0] == ci and arms[i][1] == e_i for i in pair)

    # ------------------------------------------------------------------ outputs
    def lines(self):
        """[(xy, width)] of every street (maps)."""
        return [(np.vstack([c.cl.p, c.cl.p[:1]]) if c.cl.loop else c.cl.p, float(np.median(c.hl + c.hr)))
                for c in self.chains]

    def overlay(self, proj, objs=()):
        """What the map shows: every street's edges, its centre line, width and edge confidence."""
        streets = []
        for c in self.chains:
            cl = c.cl
            k = max(1, len(cl.p) // 300)
            le = cl.p + cl.n * c.hl[:, None]
            re_ = cl.p - cl.n * c.hr[:, None]
            if cl.loop:
                le, re_, cp = np.vstack([le, le[:1]]), np.vstack([re_, re_[:1]]), np.vstack([cl.p, cl.p[:1]])
            else:
                cp = cl.p
            ll = lambda a: np.column_stack(proj.to_latlon(a[::k, 0], a[::k, 1])).round(7).tolist()
            streets.append({"centre": ll(cp), "edges": [ll(le), ll(re_)], "highway": c.highway,
                            "width": round(float(np.median(c.hl + c.hr)), 1), "measured": c.measured,
                            "conf": round(float(np.mean(c.conf)) if len(c.conf) else 0.0, 2),
                            "length": round(cl.length, 1)})
        obj_ll = []
        for o in objs:
            la, lo = proj.to_latlon(o["pos"][0], o["pos"][1])
            obj_ll.append({"name": o["name"], "lat": float(la), "lon": float(lo), "pos": o["pos"], "forward": o["forward"]})
        z = self.Z
        return {"streets": streets, "objects": obj_ll, "length_m": round(self.length, 1),
                "n_streets": len(self.chains), "n_junctions": len(self.junctions),
                "elevation": {"min": round(float(z.min()), 1) if len(z) else 0.0,
                              "max": round(float(z.max()), 1) if len(z) else 0.0}}


class NetTerrain:
    """TerrainModel over the network: the ground takes the road height near any street and blends back to the
    DEM further out (same interface as meshes.TerrainModel)."""

    tin = None

    def __init__(self, proj, dem, net, extra, blend=25.0):
        self.proj, self.dem, self.net, self.cl = proj, dem, net, None
        self.tree, self.zc, self.hwc = net.tree, net.Z, np.maximum(net.HL, net.HR)
        self.extra, self.blend = extra, blend

    def road_info(self, x, y):
        d, idx = self.tree.query(np.column_stack([np.ravel(x), np.ravel(y)]))
        return d.reshape(np.shape(x)), self.zc[idx].reshape(np.shape(x))

    def road_info3(self, x, y):
        d, idx = self.tree.query(np.column_stack([np.ravel(x), np.ravel(y)]))
        sh = np.shape(x)
        return d.reshape(sh), self.zc[idx].reshape(sh), self.hwc[idx].reshape(sh)

    def raw(self, x, y):
        lat, lon = self.proj.to_latlon(x, y)
        return np.asarray(self.dem.sample(lat, lon), dtype=float)

    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = self.raw(x, y)
        zr, e = self.net.query(x, y)
        w = np.clip((e - self.extra) / self.blend, 0, 1)
        w = w * w * (3 - 2 * w)
        h = np.where(np.isfinite(h), h, zr)
        return zr * (1 - w) + h * w

    def surface_height(self, x, y):
        if self.tin is None:
            return self.height(x, y)
        h = self.tin.height(x, y)
        bad = ~np.isfinite(h)
        if bad.any():
            h = np.where(bad, self.height(x, y), h)
        return h


# ---------------------------------------------------------------------- building the network
def _graph_heights(chains, node_xy, proj, dem, sigma_m):
    """Road heights over the whole network: median of 5 samples across each street every 2 m, then a
    screened-Poisson smoothing on the street graph (streets that meet share the junction's height)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    var_of_node = {}
    idx = []
    raw = []
    nvar = 0
    rows, cols = [], []
    for c in chains:
        cl = c.cl
        n = len(cl.p)
        zs = []
        for off in (-2.0, -1.0, 0.0, 1.0, 2.0):
            q = cl.p + cl.n * off
            la, lo = proj.to_latlon(q[:, 0], q[:, 1])
            zs.append(np.asarray(dem.sample(la, lo), float))
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            z = np.nanmedian(np.vstack(zs), axis=0)
        v = np.arange(nvar, nvar + n)
        nvar += n
        if c.ends:
            for end, k in ((0, c.ends[0]), (n - 1, c.ends[1])):
                if k is None:
                    continue
                if k in var_of_node:
                    v[end] = var_of_node[k]
                else:
                    var_of_node[k] = v[end]
        idx.append(v)
        raw.append(z)
        a, b = v[:-1], v[1:]
        if cl.loop:
            a, b = np.r_[a, v[-1]], np.r_[b, v[0]]
        rows.append(a), cols.append(b)
    if not idx:
        return
    zraw = np.full(nvar, np.nan)
    cnt = np.zeros(nvar)
    acc = np.zeros(nvar)
    for v, z in zip(idx, raw):
        ok = np.isfinite(z)
        np.add.at(acc, v[ok], z[ok])
        np.add.at(cnt, v[ok], 1)
    has = cnt > 0
    zraw[has] = acc[has] / cnt[has]
    fill = float(np.nanmedian(zraw)) if has.any() else 0.0
    zraw[~has] = fill
    used = np.zeros(nvar, bool)
    for v in idx:
        used[v] = True
    r = np.concatenate(rows)
    c_ = np.concatenate(cols)
    keep = r != c_
    r, c_ = r[keep], c_[keep]
    W = sp.coo_matrix((np.ones(len(r)), (r, c_)), shape=(nvar, nvar))
    W = (W + W.T).tocsr()
    W.data[:] = 1.0
    deg = np.asarray(W.sum(1)).ravel()
    Lap = sp.diags(deg) - W
    spacing = float(np.median([c.cl.spacing for c in chains])) or 2.0
    lam = (max(sigma_m, 4.0) / spacing) ** 2
    A = (sp.identity(nvar) + lam * Lap).tocsc()
    z = spsolve(A, zraw)
    for ch, v in zip(chains, idx):
        ch.cl.z = z[v]


def _junction_zone(c, node_xy, degree, pad=JUNCTION_PAD + 2.0):
    """Stations of chain c inside a junction (the cross street's asphalt widens the photo edges there)."""
    z = np.zeros(len(c.cl.p), bool)
    if not c.ends:
        return z
    for k in c.ends:
        if k is not None and degree.get(k, 0) >= 3:
            z |= np.linalg.norm(c.cl.p - node_xy[k], axis=1) < 0.5 * c.width + 6.0 + pad
    return z


def build(roads, proj, bounds, dem, P, pm=None, log=print, warnings=None, turning=(), clip=None):
    """-> Network of every drivable street in the area (bounds in local metres). turning: OSM turning circles
    [{lat, lon, diameter}] (cul-de-sac bulbs)."""
    from . import parallel, roadwidth
    from .elevation import FlatDEM

    warnings = [] if warnings is None else warnings
    ways = ways_in(roads, proj, bounds, clip=clip)
    edges, degree = _edges(ways)
    raw = chains_of(edges, degree)
    node_xy = {}
    for xy, _, ka, kb in edges:
        node_xy.setdefault(ka, xy[0])
        node_xy.setdefault(kb, xy[-1])
    chains, stubs = [], []
    cs = min(float(P.get("corner_smoothing", 8.0)), 4.0)
    flat = FlatDEM()
    for xy, r, oneway, ka, kb, ids, loop in raw:
        L = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        w = float(np.clip(r["width"], 3.0, 24.0))
        if L < MIN_CHAIN_M:
            stubs.append((xy, w / 2))
            continue
        latlng = np.column_stack(proj.to_latlon(xy[:, 0], xy[:, 1]))
        try:
            cl = build_centerline(latlng, proj, flat, loop, spacing=2.0, corner_smoothing=cs, elev_smoothing=0.0)
        except ValueError:
            stubs.append((xy, w / 2))
            continue
        if not loop:                                   # exact junction positions (smoothing keeps them, rounding)
            cl.p[0], cl.p[-1] = xy[0], xy[-1]
        n = len(cl.p)
        chains.append(Chain(cl, np.full(n, w / 2), np.full(n, w / 2), np.zeros(n), r["highway"], w, int(oneway),
                            None if loop else (ka, kb), list(ids)))
    if not chains:
        raise ValueError("No drivable street in this area.")
    # the node degree as the chains see it (pieces merged, stubs left out)
    deg2 = {}
    for c in chains:
        if c.ends:
            for k in c.ends:
                deg2[k] = deg2.get(k, 0) + 1
    log(f"  {len(chains)} streets ({sum(c.cl.length for c in chains) / 1000:.1f} km), "
        f"{sum(1 for d in deg2.values() if d >= 3)} junctions")

    # ---- heights over the whole network
    _graph_heights(chains, node_xy, proj, dem, float(P.get("elev_smoothing", 25.0)))

    # ---- widths: measured on the photo per street, smoothed; junctions take the width from either side
    if pm is not None and P.get("width_mode", "photo") == "photo":
        log("  measuring every street's width on the photo ...")

        def measure(c):
            if c.cl.length < MIN_MEASURE_M:
                return None
            try:
                hl, hr, info = roadwidth.measure_road_edges(pm, c.cl, c.width)
            except Exception:
                return None
            if not info.get("used"):
                return None
            conf = np.asarray(info.get("confidence"), float)
            zone = _junction_zone(c, node_xy, deg2)
            if zone.any() and (~zone).sum() >= 3:
                s = c.cl.s
                hl = np.where(zone, np.interp(s, s[~zone], hl[~zone]), hl)
                hr = np.where(zone, np.interp(s, s[~zone], hr[~zone]), hr)
                conf = np.where(zone, 0.0, conf)
            if P.get("width_smoothing", 30.0) > 0:
                hl, hr = roadwidth.smooth_widths(c.cl, hl, hr, conf, float(P["width_smoothing"]))
            return hl, hr, conf

        res = parallel.pmap(measure, chains)
        for c, m in zip(chains, res):
            if m is not None:
                c.hl, c.hr, c.conf = m
                c.measured = True
        # streets the photo can't read: the typical measured width of their class (Québec streets are wider
        # than OSM's defaults), else the nominal width
        by_cls = {}
        for c in chains:
            if c.measured:
                by_cls.setdefault(c.highway, []).append((float(np.median(c.hl + c.hr)), c.cl.length))
        typ = {k: float(np.average([a for a, _ in v], weights=[b for _, b in v])) for k, v in by_cls.items()}
        for c in chains:
            if not c.measured and c.highway in typ:
                c.hl = np.full(len(c.cl.p), typ[c.highway] / 2)
                c.hr = c.hl.copy()
        nm = sum(c.measured for c in chains)
        log(f"  width measured on {nm} of {len(chains)} streets; "
            + ", ".join(f"{k} {v:.1f} m" for k, v in sorted(typ.items())))
    # manual width settings: on the nearest street
    for ov in P.get("width_overrides") or []:
        try:
            ox, oy = proj.to_xy(float(ov["lat"]), float(ov["lon"]))
        except (KeyError, TypeError, ValueError):
            continue
        best = min(range(len(chains)), key=lambda i: np.min(np.hypot(chains[i].cl.p[:, 0] - ox,
                                                                        chains[i].cl.p[:, 1] - oy)))
        c = chains[best]
        i = int(np.argmin(np.hypot(c.cl.p[:, 0] - ox, c.cl.p[:, 1] - oy)))
        c.hl, c.hr = roadwidth.apply_overrides(c.cl, c.hl, c.hr, [{"s": float(c.cl.s[i]), "width": float(ov["width"]),
                                                                  "span": float(ov.get("span", 40))}])
    # cul-de-sacs: a round turning area at the end of the street (OSM turning_circle on a dead end)
    circles = []
    ends = [(c, e) for c in chains if c.ends for e in (0, 1) if deg2.get(c.ends[e], 0) == 1]
    for t in turning or []:
        q = np.array(proj.to_xy(float(t["lat"]), float(t["lon"])), float)
        best = None
        for c, e in ends:
            p = c.cl.p[0 if e == 0 else -1]
            d = float(np.linalg.norm(p - q))
            if d < 12.0 and (best is None or d < best[0]):
                best = (d, c, p)
        if best is None:
            continue
        _, c, p = best
        dia = t.get("diameter")
        r = float(np.clip(dia / 2, 6.0, 20.0)) if dia else (11.0 if c.highway != "service" else 8.0)
        if t.get("loop"):
            continue                                      # a loop around an island: the way is mapped itself
        circles.append((p, max(r, float(np.max([c.hl[0], c.hr[0], c.hl[-1], c.hr[-1]])) + 1.0)))
    if circles:
        log(f"  {len(circles)} cul-de-sac turning circle(s)")
    net = Network(chains, node_xy, deg2, stubs, camber=float(P.get("camber", 0.06)), bounds=bounds, clip=clip,
                  turning=circles)
    crossings = _crossings(chains)
    if crossings:
        warnings.append(f"{crossings} bridge / overpass crossing(s) are built at street level "
                        f"(the ground can't pass under a road).")
    return net


def _crossings(chains):
    """Streets that cross without a shared node (bridges, overpasses)."""
    from shapely.geometry import LineString
    from shapely.strtree import STRtree

    lines = [LineString(c.cl.p) for c in chains]
    tree = STRtree(lines)
    n = 0
    for i, ln in enumerate(lines):
        for j in tree.query(ln, predicate="crosses"):
            if j > i:
                n += 1
    return n


# ---------------------------------------------------------------------- markings
def markings(net, P, pm, log=print, warnings=None):
    """Lines on every street: read on the photo, or the usual layout on main roads only (residential streets
    and service roads usually have none). Lines stop inside junctions and across side streets."""
    from . import markings as mk_mod

    mode = P.get("markings", "photo")
    stats = {"photo": 0, "standard": 0, "none": 0}
    for ci, c in enumerate(net.chains):
        mk = None
        if mode == "none":
            mk = mk_mod.Markings([], {"source": "none"})
        elif mode == "photo" and pm is not None and c.cl.length >= MIN_MEASURE_M:
            try:
                m = mk_mod.detect(pm, c.cl, c.hl, c.hr, log=lambda *_: None)
                if m is not None and m.info.get("usable"):
                    m.info["source"] = "photo"
                    mk = m
            except Exception:
                mk = None
        if mk is None:
            if c.highway in MARKED and c.cl.length >= MIN_MEASURE_M:
                mk = mk_mod.standard(c.cl, c.hl, c.hr)
                if c.oneway:                                     # one-way: edge lines only
                    mk.lines = [ln for ln in mk.lines if ln.side != "centre"]
            else:
                mk = mk_mod.Markings([], {"source": "none"})
        c.mk = _cut(net, ci, mk)
        stats[mk.info.get("source", "none")] = stats.get(mk.info.get("source", "none"), 0) + 1
    log(f"  markings: {stats.get('photo', 0)} streets read on the photo, {stats.get('standard', 0)} standard, "
        f"{stats.get('none', 0)} without lines")
    return stats


def _cut(net, ci, mk, step=0.5, min_piece=1.0):
    """Remove the parts of every line that run across another street's asphalt or through a junction."""
    c = net.chains[ci]
    cl = c.cl
    S = np.concatenate([cl.s, [cl.length]]) if cl.loop else cl.s
    P_ = np.vstack([cl.p, cl.p[:1]]) if cl.loop else cl.p
    N_ = np.vstack([cl.n, cl.n[:1]]) if cl.loop else cl.n
    for ln in mk.lines:
        off = np.concatenate([ln.offset, ln.offset[:1]]) if cl.loop else np.asarray(ln.offset, float)
        pieces = []
        for a, b in ln.pieces:
            ss = np.arange(a, b + 1e-6, step)
            if len(ss) < 2:
                continue
            sq = ss % cl.length if cl.loop else ss
            p = np.column_stack([np.interp(sq, S, P_[:, 0]), np.interp(sq, S, P_[:, 1])])
            nn = np.column_stack([np.interp(sq, S, N_[:, 0]), np.interp(sq, S, N_[:, 1])])
            o = np.interp(sq, S, off)
            q = p + nn * o[:, None]
            bad = net.inside_other(q, ci) | net.near_junction(q, ci)
            run = None
            for k in range(len(ss)):
                if not bad[k]:
                    run = ss[k] if run is None else run
                if (bad[k] or k == len(ss) - 1) and run is not None:
                    end = ss[k] if not bad[k] else ss[k - 1]
                    if end - run >= min_piece:
                        pieces.append((float(run), float(end)))
                    run = None
        ln.pieces = pieces
    mk.lines = [ln for ln in mk.lines if ln.pieces]
    return mk


def marking_meshes(net, tm, lift=0.015):
    """Line meshes of every street, laid exactly on the built ground surface."""
    from . import markings as mk_mod

    parts_by = {}
    for c in net.chains:
        if c.mk is None or not c.mk.lines:
            continue
        for p in mk_mod.meshes(c.mk, c.cl, c.hl, c.hr, 0.0, lift=0.0):
            V = p.vertices.copy()
            V[:, 2] = tm.surface_height(V[:, 0], V[:, 1]) + lift
            parts_by.setdefault(p.material, []).append((V, p.faces, p.uv))
    from .meshes import MeshBuilder

    out = []
    for mat, items in parts_by.items():
        mb = MeshBuilder("ROAD_LINES" if mat == "road_paint" else "ROAD_LINES_Y", mat)
        for V, F, UV in items:
            mb.add(V, F, UV)
        out += mb.done()
    return out


# ---------------------------------------------------------------------- spawn
def spawn_street(net, proj, P, warnings=None):
    """The street the cars start on: (centreline, hl, hr, s0) facing the way traffic can go."""
    warnings = [] if warnings is None else warnings
    x0, y0, x1, y1 = net.bounds
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) if net.clip is None else \
        np.asarray(net.clip.representative_point().coords[0])
    sp = P.get("spawn")
    if sp:
        q = np.array(proj.to_xy(float(sp[0]), float(sp[1])), float)
        d, i = net.tree.query(q)
        ci = int(net.CID[i])
        c = net.chains[ci]
        s0 = float(c.cl.s[int(np.argmin(np.linalg.norm(c.cl.p - net.P[i], axis=1)))])
        if d > 25:
            warnings.append(f"The spawn point is {d:.0f} m from the nearest street - cars start on that street.")
    else:
        def score(c):
            dc = float(np.min(np.linalg.norm(c.cl.p - centre, axis=1)))
            return c.cl.length * float(np.median(c.hl + c.hr)) / (1 + dc / 300.0)

        ci = max(range(len(net.chains)), key=lambda i: score(net.chains[i]))
        c = net.chains[ci]
        s0 = c.cl.length / 2
    cl, hl, hr = c.cl, c.hl, c.hr
    i0 = int(np.clip(round(s0 / cl.spacing), 0, len(cl.p) - 1))
    flip = float(np.dot(cl.t[i0], centre - cl.p[i0])) < 0          # head into the area
    if P.get("reverse"):
        flip = not flip
    if c.oneway:
        flip = c.oneway < 0                                      # one-way street: the way traffic goes
    if flip:
        cl, hl, hr, s0 = _flip_cl(cl), hr[::-1].copy(), hl[::-1].copy(), cl.length - s0
    return cl, hl, hr, s0, ci

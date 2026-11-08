"""Tyre walls read from the aerial photo (kart tracks, club circuits): where they really stand beside the
track, built as stacks of tyres with collision.

On a 20 cm photo a tyre wall is a dark, beaded band 0.6-1 m wide, grey (not green like a hedge), running along
the track 0-5 m outside the asphalt. Method:

  1. Beside each asphalt edge, the photo is sampled along the track (a row every 0.25 m) and outwards (every
     5 cm, from 0.3 m inside the edge to REACH m outside; less on the inside of tight corners, where the rays
     would cross).
  2. Dark ridge: lightness of the surroundings (1 m) minus the band (0.2 m), dark on both sides, not green,
     not colourful. Tyres on grass or gravel both show.
  3. The ridges are followed along the track like the road markings (markings._chain_strokes): pieces of wall
     at least WALL_MIN m long, gaps under GAP_FILL m bridged (a car, a shadow, a tyre missing on the photo).
  4. Position as measured (smoothed): a wall 2 m outside the asphalt stays 2 m outside.

-> list of walls {side, s0, s1, rel (m outside the edge, per station between s0 and s1), xy polyline}.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .markings import DS, _at, _chain_strokes, _stations

REACH = 5.0          # m outside the asphalt edge to look for tyres
DL = 0.05            # m per column outwards
WALL_MIN = 6.0       # m: shortest wall kept
GAP_FILL = 4.0       # m: gaps bridged inside a wall
TYRE_D = 0.60        # m: tyre diameter
TYRE_H = 0.21        # m: one tyre's height
STACK = 3            # tyres per stack


def detect(pm, cl, hl, hr, log=print):
    from .roadwidth import rgb_to_lab

    s, px, nn, S = _stations(cl)
    hls, hrs = _at(s, S, hl, cl.loop), _at(s, S, hr, cl.loop)
    rad = _at(s, S, np.where(np.isfinite(cl.radius), cl.radius, 1e6), cl.loop)
    rel = np.arange(-0.3, REACH + DL / 2, DL)
    walls = []
    info = {"walls": 0, "length_m": 0.0}
    for side, sg, hs in (("left", 1.0, hls), ("right", -1.0, hrs)):
        off = sg * (hs[:, None] + rel[None])
        X = px[:, None, 0] + nn[:, None, 0] * off
        Y = px[:, None, 1] + nn[:, None, 1] * off
        rgb, ok = pm.sample(X.ravel(), Y.ravel(), order=1)
        rgb = rgb.reshape(len(s), len(rel), 3)
        ok = ok.reshape(len(s), len(rel))
        # inside of a corner: rays from the edge towards the corner's centre cross beyond the radius
        inner = np.sign(rad) == sg
        lim = np.where(inner, np.abs(rad) - hs - 0.5, REACH + 1)
        ok &= rel[None] < lim[:, None]
        lab = rgb_to_lab(rgb)
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        g1 = ndi.gaussian_filter(L, (2.0, 0.20 / DL))
        g2 = ndi.gaussian_filter(L, (2.0, 1.00 / DL))
        a1 = ndi.gaussian_filter(a, (2.0, 0.20 / DL))
        a2 = ndi.gaussian_filter(a, (2.0, 1.00 / DL))
        # darker than the surroundings, and on grass also greyer (a black tyre on grass is hardly darker on a
        # blurred photo, but it is not green)
        T = (g2 - g1) + 1.5 * np.clip(a1 - a2, 0, None)
        sw = int(round(0.7 / DL))
        sym = np.minimum(T - np.roll(T, sw, axis=1), T - np.roll(T, -sw, axis=1))
        green = (a1 < -6) & (a1 - a2 < 1.5)                  # green and not greyer than around it: a hedge
        b1 = ndi.gaussian_filter(b, (2.0, 0.20 / DL))
        b2 = ndi.gaussian_filter(b, (2.0, 1.00 / DL))
        c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
        grey = c1 < np.maximum(18.0, c2 - 1.0)                # rubber is grey (or greyer than its surroundings)
        valid = ok & ~green
        vis = valid & (rel[None] > 0.3)
        noise = 1.4826 * float(np.median(np.abs(T[vis] - np.median(T[vis])))) if vis.sum() > 200 else 2.0
        thr = max(2.5, 2.5 * noise)
        info.setdefault("thr", []).append(round(thr, 1))
        peak = (T >= ndi.maximum_filter1d(T, int(round(0.6 / DL)) | 1, axis=1))
        cand = peak & (T > thr) & (sym > 0.4 * thr) & valid & grey & (L < 62)
        for cov, pr in _chain_strokes(cand, rel, cl.loop, max_gap_m=GAP_FILL * 2):
            ext = np.nonzero(np.isfinite(pr))[0]
            if len(ext) < 2:
                continue
            # fill short gaps, split at long ones
            painted = ndi.binary_closing(cov, structure=np.ones(int(GAP_FILL / DS) | 1, bool)) | cov
            painted &= np.isfinite(pr)
            m = np.concatenate([[False], painted, [False]]).astype(np.int8)
            d = np.diff(m)
            for i0, i1 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
                if (i1 - i0) * DS < WALL_MIN or cov[i0:i1].sum() * DS < 0.5 * WALL_MIN:
                    continue
                r = pr[i0:i1].copy()
                r = ndi.median_filter(r, size=int(3.0 / DS) | 1, mode="nearest")
                r = ndi.gaussian_filter1d(r, 1.0 / DS, mode="nearest")
                r = np.clip(r, 0.05, REACH)
                o = sg * (hs[i0:i1] + np.maximum(r, TYRE_D / 2 + 0.05))   # tyre centres, never on the asphalt
                xy = px[i0:i1] + nn[i0:i1] * o[:, None]
                walls.append({"side": side, "s0": float(s[i0]), "s1": float(s[i1 - 1] + DS),
                              "rel": r, "xy": xy})
    walls = _dedupe(walls)
    info["walls"] = len(walls)
    info["length_m"] = round(float(sum(w["s1"] - w["s0"] for w in walls)))
    return walls, info


def is_race_track(roads, proj, cl, near=10.0, share=0.4):
    """Does OpenStreetMap map this track as a race / kart track (highway=raceway along most of it)?"""
    from scipy.spatial import cKDTree

    from .track import resample

    pts = []
    for r in roads:
        if r.get("highway") == "raceway" and len(r["coords"]) >= 2:
            try:
                pts.append(resample(proj.latlngs_to_xy(r["coords"]), 2.0, False))
            except ValueError:
                continue
    if not pts:
        return False
    d, _ = cKDTree(np.vstack(pts)).query(cl.p)
    return float((d < near).mean()) >= share


def _dedupe(walls):
    """The same wall seen from two stretches of track (a hairpin's infield): keep one."""
    from scipy.spatial import cKDTree

    walls = sorted(walls, key=lambda w: -(w["s1"] - w["s0"]))
    kept = []
    for w in walls:
        dup = False
        for k in kept:
            d, _ = cKDTree(k["xy"]).query(w["xy"])
            if (d < 0.6).mean() > 0.6:
                dup = True
                break
        if not dup:
            kept.append(w)
    return kept


def stacks(walls, spacing=TYRE_D * 0.97):
    """Tyre stack centres (x, y) every `spacing` m along each wall."""
    out = []
    for w in walls:
        xy = w["xy"]
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        sc = np.concatenate([[0], np.cumsum(seg)])
        if sc[-1] < spacing:
            continue
        t = np.arange(spacing / 2, sc[-1], spacing)
        out.append(np.column_stack([np.interp(t, sc, xy[:, 0]), np.interp(t, sc, xy[:, 1])]))
    return np.vstack(out) if out else np.zeros((0, 2))


# One tyre's cross-section (r, z) in metres, a closed loop: inner bead (bottom) -> bottom sidewall -> rounded
# shoulder -> tread -> shoulder -> top sidewall -> inner bead (top) -> back down the bead face. A car tyre
# (about 205/55 R16 worn): 0.60 m across, 0.20 m wide, 0.38 m hole.
PROFILE = np.array([
    (0.190, 0.030), (0.222, 0.010), (0.258, 0.004), (0.284, 0.018), (0.297, 0.052), (0.300, 0.100),
    (0.297, 0.148), (0.284, 0.182), (0.258, 0.196), (0.222, 0.190), (0.190, 0.170), (0.184, 0.100),
])
NEAR_M = 70.0        # m: detailed tyres up to here, the simple stacks (always drawn) beyond
SEGMENTS = 14        # around a tyre


def _profile_frame():
    """v coordinate (0..1 along the loop) and outward normals (nr, nz) of PROFILE."""
    p = np.vstack([PROFILE, PROFILE[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    v = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
    q = np.vstack([PROFILE[-1:], PROFILE, PROFILE[:1]])
    d = q[2:] - q[:-2]                                    # tangent (dr, dz)
    n = np.column_stack([d[:, 1], -d[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    centre = np.array([0.24, 0.10])
    if ((PROFILE - centre) * n).sum(1).mean() < 0:        # point away from the tyre's core
        n = -n
    return v, np.vstack([n, n[:1]])


def _tyre(x, y, z, rot, scale_r, scale_w, tilt, segments=SEGMENTS):
    """One tyre lying flat (a revolved cross-section) -> vertices, faces, uv, normals."""
    v_prof, n_prof = _profile_frame()
    prof = np.vstack([PROFILE, PROFILE[:1]]) * [scale_r, scale_w]
    K = len(prof)
    ang = np.linspace(0, 2 * np.pi, segments + 1) + rot
    ca, sa = np.cos(ang), np.sin(ang)
    R = prof[:, 0][:, None]
    Z = prof[:, 1][:, None]
    X = R * ca[None]
    Y = R * sa[None]
    NX = n_prof[:, 0][:, None] * ca[None]
    NY = n_prof[:, 0][:, None] * sa[None]
    NZ = np.repeat(n_prof[:, 1][:, None], segments + 1, axis=1)
    # a small tilt (tyres never lie perfectly flat on each other)
    tx, ty = tilt
    Zt = Z + X * tx + Y * ty
    V = np.stack([X + x, Y + y, Zt + z], -1).reshape(-1, 3)
    N = np.stack([NX, NY, NZ], -1).reshape(-1, 3)
    U = np.repeat((np.linspace(0, 1, segments + 1) * 0.5)[None], K, axis=0)       # left half of the atlas
    UV = np.stack([U, np.repeat(v_prof[:, None], segments + 1, axis=1)], -1).reshape(-1, 2)
    i, j = np.meshgrid(np.arange(K - 1), np.arange(segments), indexing="ij")
    a = (i * (segments + 1) + j).ravel()
    b = ((i + 1) * (segments + 1) + j).ravel()
    F = np.vstack([np.column_stack([a, b, b + 1]), np.column_stack([a, b + 1, a + 1])])
    # wind the faces so they face along the normals
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if (fn * N[F[:, 0]]).sum(1).mean() < 0:
        F = F[:, [0, 2, 1]]
    return V, F, UV, N


def meshes(walls, tm, seed=4):
    """Tyre walls: stacks of 3-4 real tyres (revolved cross-section: rounded shoulders, tread, sidewalls, the
    hole in the middle), each a little different in size, turned and tilted, drawn up to NEAR_M m
    ('TYRES', visual); plus a simple 8-sided stack inside them, always drawn: the collision ('1WALL_TYRES', karts
    bounce off it) and what is seen from further away."""
    from .meshes import MAX_VERTS, MeshPart

    pts = stacks(walls)
    if not len(pts):
        return []
    rng = np.random.default_rng(seed)
    z0 = tm.surface_height(pts[:, 0], pts[:, 1]) - 0.02
    parts = []
    acc = {"V": [], "F": [], "UV": [], "N": [], "n": 0}

    def flush():
        if acc["n"]:
            p = MeshPart(f"TYRES_{len(parts):03d}", "tyres", np.vstack(acc["V"]), np.vstack(acc["F"]),
                         np.vstack(acc["UV"]), True)
            p.normals = np.vstack(acc["N"])
            p.lod_out = NEAR_M
            parts.append(p)
        acc.update(V=[], F=[], UV=[], N=[], n=0)

    heights = np.zeros(len(pts))
    for k, ((x, y), z) in enumerate(zip(pts, z0)):
        n_t = 4 if rng.random() < 0.2 else STACK
        zz = z
        for t in range(n_t):
            sr = rng.uniform(0.93, 1.05)
            sw = rng.uniform(0.92, 1.12)
            V, F, UV, N = _tyre(x + rng.normal(0, 0.012), y + rng.normal(0, 0.012), zz, rng.uniform(0, 2 * np.pi),
                                sr, sw, rng.normal(0, 0.03, 2))
            if acc["n"] + len(V) > MAX_VERTS:
                flush()
            acc["V"].append(V)
            acc["F"].append(F + acc["n"])
            acc["UV"].append(UV)
            acc["N"].append(N)
            acc["n"] += len(V)
            zz += PROFILE[:, 1].max() * sw * 0.96         # sidewalls squash a little under the weight
        heights[k] = zz - z
    flush()
    # simple stacks: collision + far view (hidden inside the tyres up close)
    from .meshes import MeshBuilder

    mb = MeshBuilder("1WALL_TYRES", "tyres", smooth=True)
    sides = 8
    ang = np.linspace(0, 2 * np.pi, sides + 1)
    i = np.arange(sides)
    n = sides + 1
    Rc = 0.27
    for (x, y), z, h in zip(pts, z0, heights):
        a = ang + rng.uniform(0, 2 * np.pi)
        cx, cy = np.cos(a), np.sin(a)
        top_z = z + h - 0.30                                 # below the top tyre: seen through its hole as dark
        bot = np.column_stack([x + Rc * cx, y + Rc * cy, np.full(n, z)])
        top = np.column_stack([x + Rc * cx, y + Rc * cy, np.full(n, top_z)])
        u = 0.5 + np.linspace(0, 1, n) * 0.25                 # 3rd quarter of the atlas: a stack from afar
        V = np.vstack([bot, top, [[x, y, top_z]]])
        UV = np.vstack([np.column_stack([u, np.ones(n)]), np.column_stack([u, np.zeros(n)]), [[0.875, 0.5]]])
        F = np.vstack([np.column_stack([i, i + 1, n + i + 1]), np.column_stack([i, n + i + 1, n + i]),
                       np.column_stack([np.full(sides, 2 * n), n + i, n + i + 1])])
        mb.add(V, F, UV)
    return parts + mb.done()


def texture(photo=None, size=256):
    """Atlas 1024 x 256: left half, one tyre's surface (u around the tyre, twice; v along its cross-section:
    bead, sidewall, shoulder, tread with blocks and grooves, shoulder, sidewall, bead); third quarter, a whole
    stack seen from afar (three tyres, dark gaps); last quarter, black (the inside of the hole). photo: a rubber
    photo (materials 'rubber') whose grain is laid over it; None: procedural grain."""
    from PIL import Image

    from .materials import detail

    rng = np.random.default_rng(9)
    W, H = size * 4, size
    img = np.zeros((H, W, 3), np.float32)
    rub = np.array([36, 36, 38], np.float32)
    v_prof, _ = _profile_frame()
    vk = v_prof[:-1]                                       # v at each profile point (loop start = bead)
    v = (np.arange(H) + 0.5) / H
    # zones along the cross-section (by profile point index): 3-7 shoulder / tread, 1-2 and 8-9 sidewalls
    t0, t1 = vk[4], vk[6]                                  # tread
    s0, s1 = vk[3], vk[7]                                  # shoulders
    lum = np.full(H, 1.0)
    side = (v < s0) | (v > s1)
    lum[side] = 1.05                                       # sidewalls: smooth, a bit lighter (worn, dusty)
    bead = (v < vk[1]) | (v > vk[9])
    lum[bead] = 0.75
    sh = ((v >= s0) & (v < t0)) | ((v > t1) & (v <= s1))
    lum[sh] = 0.95
    # the surface around: ~140 tread blocks around the tyre (the left half is the whole circumference)
    u = (np.arange(size * 2) + 0.5) / (size * 2)
    blocks = ((u * 140) % 1.0) < 0.18                      # lateral grooves
    left = np.ones((H, size * 2), np.float32) * lum[:, None]
    tread = (v >= t0) & (v <= t1)
    left[np.ix_(tread, blocks)] *= 0.55
    for c in (0.33, 0.67):                                 # two circumferential grooves
        g = np.abs((v - t0) / max(t1 - t0, 1e-6) - c) < 0.05
        left[g & tread, :] *= 0.5
    # a raised ring on each sidewall (where the lettering would be), lighter
    for vv in ((vk[1] + vk[2]) / 2, (vk[8] + vk[9]) / 2):
        left[np.abs(v - vv) < 0.012, :] *= 1.15
    img[:, :size * 2] = rub * left[..., None]
    # far stack: three tyres, rounded, dark gaps between them
    k = (v * STACK) % 1.0
    shade = 0.55 + 0.45 * np.sin(np.pi * k) ** 0.6
    img[:, size * 2:size * 3] = (rub * 1.2)[None, None, :] * shade[:, None, None]
    img[:, size * 3:] = (8, 8, 9)
    g = detail(photo, W, H, fallback_seed=11, highpass=0.08)
    strength = 0.30 if photo is not None else 0.12
    img[:, :size * 3] *= (1.0 + strength * g[:, :size * 3])[..., None]
    img += rng.normal(0, 1.5 if photo is not None else 3, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def overlay(walls, proj):
    out = []
    for w in walls:
        xy = w["xy"][::4]
        la, lo = proj.to_latlon(xy[:, 0], xy[:, 1])
        out.append(np.column_stack([la, lo]).round(7).tolist())
    return out

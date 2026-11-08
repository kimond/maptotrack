"""3D grass: short tufts on lawns, tall grass clumps on wild / dry areas and hay fields.

Grass is dense next to the track (where the driver sees it) and thins out with distance, within a
vertex budget. Each clump is a few crossed, alpha-tested cards ('veg_grass' material, ksPerPixelAT);
chunks are grouped by area and get a draw distance (LOD out) so far-away grass is not drawn.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

from .meshes import MAX_VERTS, MeshPart

# atlas columns: 0 short green, 1 short dry, 2 tall green, 3 tall dry
GRASS_COLS = 4
LAWN, WILD = 0, 1

# clumps per m2 by distance from the track edge (m): (distance, density) points, linear in between
DENSITY = {
    LAWN: [(0, 0.9), (12, 0.9), (30, 0.3), (50, 0.06), (70, 0.0)],
    WILD: [(0, 0.75), (20, 0.75), (50, 0.25), (100, 0.05), (140, 0.0)],
}
LOD_OUT = {LAWN: 90.0, WILD: 200.0}


def _density(kind, d, scale):
    pts = np.array(DENSITY[kind])
    return np.interp(d, pts[:, 0], pts[:, 1], right=0.0) * scale


def scatter(an, tm, clear, streets, density=1.0, max_lawn=30000, max_wild=15000, wild=True, seed=21):
    """-> dict x, y, h, w, kind (LAWN/WILD), col (atlas column)."""
    rng = np.random.default_rng(seed)
    out = []
    masks = [(LAWN, an.lawn)] + ([(WILD, an.wild)] if wild else [])
    for kind, mask in masks:
        rr, cc = np.nonzero(mask)
        if len(rr) == 0:
            continue
        x, y = an.xy_of(rr, cc)
        d, _, hw = tm.road_info3(x, y)
        edge = d - hw
        far = DENSITY[kind][-1][0]
        near = edge < far
        rr, cc, x, y, edge = rr[near], cc[near], x[near], y[near], edge[near]
        lam = _density(kind, edge, density) * an.res ** 2           # expected clumps per pixel
        cnt = rng.poisson(lam)
        sel = cnt > 0
        rr, cc, x, y, edge, cnt = rr[sel], cc[sel], x[sel], y[sel], edge[sel], cnt[sel]
        rr, cc, x, y, edge = (np.repeat(v, cnt) for v in (rr, cc, x, y, edge))
        x = x + rng.uniform(-0.5, 0.5, len(x)) * an.res
        y = y + rng.uniform(-0.5, 0.5, len(y)) * an.res
        keep = edge > clear + (0.3 if kind == LAWN else 1.5)
        if streets is not None and len(streets[0]) and len(x):
            ds, idx = cKDTree(streets[0]).query(np.column_stack([x, y]))
            keep &= ds > streets[1][idx] + (0.4 if kind == LAWN else 1.2)
        rr, cc, x, y, edge = rr[keep], cc[keep], x[keep], y[keep], edge[keep]
        cap = max_lawn if kind == LAWN else max_wild
        if len(x) > cap:                                             # keep the ones nearest the track
            o = np.argsort(edge + rng.uniform(0, 15, len(edge)))[:cap]
            rr, cc, x, y, edge = rr[o], cc[o], x[o], y[o], edge[o]
        n = len(x)
        dry = an.dry[rr, cc] if hasattr(an, "dry") else np.zeros(n, bool)
        if kind == LAWN:
            h = rng.uniform(0.16, 0.32, n)
            w = rng.uniform(0.8, 1.4, n)
            col = np.where(dry, 1, 0)
        else:
            h = rng.uniform(0.5, 1.0, n)
            w = rng.uniform(1.2, 2.2, n)
            col = np.where(dry | (rng.random(n) < 0.25), 3, 2)
        out.append({"x": x, "y": y, "h": h, "w": w, "kind": np.full(n, kind), "col": col})
    if not out:
        return {k: np.zeros(0) for k in ("x", "y", "h", "w", "kind", "col")}
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def scatter_landcover(lc, tm, bounds, clear, streets, density=1.0, max_lawn=30000, max_wild=15000, wild=True):
    """No aerial photo: lawns on grass/urban land cover, tall grass on crops, shrubs and wetlands."""
    from . import landcover as lcm

    class _G:
        pass

    x0, y0, x1, y1 = bounds
    res = 1.0
    xs, ys = np.arange(x0 + res / 2, x1, res), np.arange(y1 - res / 2, y0, -res)
    X, Y = np.meshgrid(xs, ys)
    cls = lc.sample(X.ravel(), Y.ravel()).reshape(X.shape)
    blocked = lc.is_blocked(X.ravel(), Y.ravel()).reshape(X.shape)
    g = _G()
    g.res = res
    g.lawn = np.isin(cls, [lcm.GRASS, lcm.URBAN]) & ~blocked
    g.wild = np.isin(cls, [lcm.CROP, lcm.SHRUB, lcm.WETLAND]) & ~blocked
    g.dry = np.isin(cls, [lcm.CROP])
    g.xy_of = lambda r, c: (xs[c], ys[r])
    return scatter(g, tm, clear, streets, density, max_lawn, max_wild, wild)


def meshes(g, tm, prefix="VEG"):
    """Lawn tufts: 2 crossed cards (8 vertices). Tall grass: 3 cards at 60 degrees (12 vertices)."""
    from .landcover import spatial_order

    parts = []
    # lawn tufts have a material of their own (same texture): CSP Grass FX hides them and grows its own 3D
    # grass there (csp.py); the tall clumps stay
    for kind, name, ncards, mat in ((LAWN, "GRASS", 2, "veg_grass_short"), (WILD, "TALLGRASS", 3, "veg_grass")):
        sel = g["kind"] == kind
        if not sel.any():
            continue
        o = spatial_order(g["x"][sel], g["y"][sel], cell=32.0)
        x, y, h, w, col = (np.asarray(g[k])[sel][o] for k in ("x", "y", "h", "w", "col"))
        n = len(x)
        rng = np.random.default_rng(5 + kind)
        z = tm.surface_height(x, y) - 0.06
        ang = rng.uniform(0, np.pi, n)
        per = ncards * 4
        maxn = MAX_VERTS // per
        for start in range(0, n, maxn):
            sl = slice(start, min(start + maxn, n))
            m = sl.stop - sl.start
            V = np.zeros((m, per, 3))
            UV = np.zeros((m, per, 2))
            u0 = col[sl] / GRASS_COLS
            du = 1.0 / GRASS_COLS
            for k in range(ncards):
                a = ang[sl] + k * np.pi / ncards
                dx, dy = np.cos(a), np.sin(a)
                lean = rng.uniform(-0.12, 0.12, (m, 2)) * h[sl, None]       # a little wind / random lean
                for c, (sx, up) in enumerate(((-1, 0), (1, 0), (1, 1), (-1, 1))):
                    vi = k * 4 + c
                    V[:, vi, 0] = x[sl] + dx * w[sl] / 2 * sx + lean[:, 0] * up
                    V[:, vi, 1] = y[sl] + dy * w[sl] / 2 * sx + lean[:, 1] * up
                    V[:, vi, 2] = z[sl] + h[sl] * up
                    UV[:, vi, 0] = u0 + (0.0 if sx < 0 else du)
                    UV[:, vi, 1] = 1.0 - up
            F = []
            for q in range(ncards):
                b = q * 4
                F += [(b, b + 1, b + 2), (b, b + 2, b + 3), (b, b + 2, b + 1), (b, b + 3, b + 2)]
            F = np.asarray(F)
            faces = (F[None] + (np.arange(m) * per)[:, None, None]).reshape(-1, 3)
            p = MeshPart(f"{prefix}_{name}_{len([q for q in parts if name in q.name]):03d}", mat,
                         V.reshape(-1, 3), faces, UV.reshape(-1, 2))
            nrm = np.zeros((m * per, 3))
            nrm[:, 2] = 1.0
            p.normals = nrm
            p.lod_out = LOD_OUT[kind]
            parts.append(p)
    return parts


def atlas(size=256) -> Image.Image:
    """RGBA grass atlas, 4 columns of size x size: short green | short dry | tall green | tall dry."""
    S = 3
    W = size * S
    rng = np.random.default_rng(31)
    img = Image.new("RGBA", (W * GRASS_COLS, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    palettes = [
        [(70, 110, 40), (92, 132, 52), (58, 92, 34), (104, 140, 60)],          # lawn green
        [(126, 128, 66), (148, 140, 82), (104, 108, 56), (160, 150, 92)],      # dry lawn
        [(76, 112, 44), (96, 128, 54), (62, 96, 38), (118, 138, 66)],          # tall green
        [(160, 146, 92), (182, 166, 108), (138, 124, 76), (120, 116, 70)],     # straw / dry
    ]
    for col in range(GRASS_COLS):
        ox = col * W
        tall = col >= 2
        nblades = 150 if tall else 220
        for _ in range(nblades):
            bx = ox + W * float(np.clip(0.5 + rng.normal(0, 0.16), 0.18, 0.82))
            top = W * (rng.uniform(0.02, 0.35) if tall else rng.uniform(0.15, 0.55))
            bend = float(np.clip(rng.normal(0, W * (0.12 if tall else 0.08)), ox + W * 0.04 - bx, ox + W * 0.96 - bx))
            c = palettes[col][rng.integers(0, 4)]
            c = tuple(int(np.clip(v + rng.integers(-12, 12), 0, 255)) for v in c) + (255,)
            pts = [(bx + bend * t * t, W - (W - top) * t) for t in np.linspace(0, 1, 6)]
            wid = max(2, int(W * (0.012 if tall else 0.016)))
            d.line(pts, fill=c, width=wid)
            if tall and rng.random() < 0.18:                                  # seed heads
                hx, hy = pts[-1]
                d.ellipse([hx - W * 0.012, hy - W * 0.03, hx + W * 0.012, hy + W * 0.01],
                          fill=(170, 150, 100, 255))
    img = img.resize((size * GRASS_COLS, size), Image.LANCZOS)
    arr = np.asarray(img).copy()
    a = arr[..., 3]
    arr[..., 3] = np.where(a > 100, 255, 0)
    from .landcover import clear_margins

    clear_margins(arr, size, size)
    arr[a <= 100, :3] = [96, 112, 60]
    return Image.fromarray(arr, "RGBA")

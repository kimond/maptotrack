"""Land cover map (Overture/ESA WorldCover + OpenStreetMap), tree scattering and tree meshes."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from .meshes import MAX_VERTS, MeshPart

GRASS, FOREST, SHRUB, CROP, URBAN, WATER, BARREN, WETLAND, SNOW = range(9)
CLASS_NAMES = ["grass", "forest", "shrub", "crop", "urban", "water", "barren", "wetland", "snow"]

OVERTURE_CLASS = {"forest": FOREST, "mangrove": FOREST, "shrub": SHRUB, "grass": GRASS, "crop": CROP,
                  "urban": URBAN, "barren": BARREN, "wetland": WETLAND, "moss": WETLAND, "snow": SNOW}
OSM_CLASS = {"wood": FOREST, "forest": FOREST, "scrub": SHRUB, "grass": GRASS, "meadow": GRASS,
             "park": GRASS, "water": WATER}
# paint order: later wins
OVERTURE_ORDER = [GRASS, CROP, URBAN, BARREN, SNOW, WETLAND, SHRUB, FOREST]


class LandCover:
    """Class raster in the local track frame. Row 0 = north edge."""

    def __init__(self, bounds, res=2.0):
        x0, y0, x1, y1 = bounds
        self.x0, self.y1, self.res = x0, y1, res
        self.w = int(np.ceil((x1 - x0) / res)) + 1
        self.h = int(np.ceil((y1 - y0) / res)) + 1
        self.img = Image.new("L", (self.w, self.h), GRASS)
        self.block = Image.new("L", (self.w, self.h), 0)   # 1 = no trees (buildings)
        self.sources = []

    def _px(self, xy):
        xy = np.asarray(xy, float)
        return [((x - self.x0) / self.res, (self.y1 - y) / self.res) for x, y in xy]

    def paint(self, outer_xy, cls, holes_xy=()):
        d = ImageDraw.Draw(self.img)
        if len(outer_xy) >= 3:
            d.polygon(self._px(outer_xy), fill=int(cls))
            for h in holes_xy:
                if len(h) >= 3:
                    d.polygon(self._px(h), fill=GRASS)

    def block_polygon(self, xy):
        if len(xy) >= 3:
            ImageDraw.Draw(self.block).polygon(self._px(xy), fill=1)

    def finish(self):
        # buildings: grow the blocked area by ~2 m so trees don't stick into walls
        self.block = self.block.filter(ImageFilter.MaxFilter(3))
        self.cls = np.asarray(self.img, dtype=np.uint8)
        self.blocked = np.asarray(self.block, dtype=bool)

    def _idx(self, x, y):
        i = np.clip(((np.asarray(x) - self.x0) / self.res).astype(int), 0, self.w - 1)
        j = np.clip(((self.y1 - np.asarray(y)) / self.res).astype(int), 0, self.h - 1)
        return i, j

    def sample(self, x, y):
        i, j = self._idx(x, y)
        return self.cls[j, i]

    def is_blocked(self, x, y):
        i, j = self._idx(x, y)
        return self.blocked[j, i]


def build_landcover(bounds, proj, osm_data, overture_lc, res=2.0):
    lc = LandCover(bounds, res)
    if overture_lc:
        lc.sources.append("ESA WorldCover via Overture Maps")
        by_cls = {}
        for f in overture_lc:
            c = OVERTURE_CLASS.get(f["subtype"])
            if c is not None:
                by_cls.setdefault(c, []).append(f)
        for c in OVERTURE_ORDER:
            for f in by_cls.get(c, []):
                for outer, inner in f["rings"]:
                    lc.paint(proj.latlngs_to_xy(outer), c, [proj.latlngs_to_xy(h) for h in inner])
    areas = osm_data.get("areas", [])
    if areas:
        lc.sources.append("OpenStreetMap")
    order = {GRASS: 0, SHRUB: 1, FOREST: 2, WATER: 3}
    for a in sorted(areas, key=lambda a: order.get(OSM_CLASS.get(a["kind"], -1), -1)):
        c = OSM_CLASS.get(a["kind"])
        if c is None:
            continue
        for ring in a["outer"]:
            lc.paint(proj.latlngs_to_xy(ring), c)
    for b in osm_data.get("buildings", []):
        for ring in b["outer"]:
            lc.block_polygon(proj.latlngs_to_xy(ring))
    lc.finish()
    return lc


# --------------------------------------------------------------------------- trees
# probability that a jittered-grid candidate becomes a tree, per class
TREE_PROB = {FOREST: 0.95, SHRUB: 0.35, WETLAND: 0.12, GRASS: 0.012, URBAN: 0.015}


def _value_noise(x, y, scale, seed):
    """Smooth, continuous noise in [0,1] on world coordinates (metres)."""
    gx, gy = np.asarray(x) / scale, np.asarray(y) / scale
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = gx - ix, gy - iy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)

    def h(a, b):
        v = (a * 374761393 + b * 668265263 + seed * 2147483647) & 0xFFFFFFFF
        v = (v ^ (v >> 13)) * 1274126177 & 0xFFFFFFFF
        return ((v ^ (v >> 16)) & 0xFFFF) / 65535.0

    a, b, c, d = h(ix, iy), h(ix + 1, iy), h(ix, iy + 1), h(ix + 1, iy + 1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def tree_ok(x, y, lc, tm, clear_extra, streets):
    """Mask of tree positions that keep clear of the track, streets, buildings and water."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0:
        return np.zeros(0, bool)
    d, _, hw = tm.road_info3(x, y)
    ok = d > hw + clear_extra
    if lc is not None:
        ok &= ~lc.is_blocked(x, y)
        ok &= lc.sample(x, y) != WATER
    if streets is not None:
        pts, half = streets
        if len(pts):
            ds, idx = cKDTree(pts).query(np.column_stack([x, y]))
            ok &= ds > half[idx] + 1.5
    return ok


def scatter_trees(lc: LandCover, tm, cl, bounds, clear_extra, streets, osm_trees, proj,
                  density=1.0, conifer_ratio=0.6, seed=1, covered=None):
    """Random trees from land-cover classes. `covered(x, y)` marks areas handled by photo detection."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = bounds
    sp = 5.5 / max(density, 0.05) ** 0.5
    gx, gy = np.meshgrid(np.arange(x0, x1, sp), np.arange(y0, y1, sp))
    x = gx.ravel() + rng.uniform(-0.45, 0.45, gx.size) * sp
    y = gy.ravel() + rng.uniform(-0.45, 0.45, gx.size) * sp
    cls = lc.sample(x, y)
    p = np.zeros(len(x))
    for c, pr in TREE_PROB.items():
        p[cls == c] = pr
    # natural clumping
    p *= np.clip(0.35 + 1.1 * _value_noise(x, y, 40.0, seed), 0, 1.3)
    keep = rng.random(len(x)) < p
    x, y, cls = x[keep], y[keep], cls[keep]

    # exact OSM trees (street trees etc.)
    n_osm = 0
    if osm_trees:
        ox, oy = proj.to_xy([t["lat"] for t in osm_trees], [t["lon"] for t in osm_trees])
        ins = (ox > x0) & (ox < x1) & (oy > y0) & (oy < y1)
        x = np.concatenate([x, ox[ins]])
        y = np.concatenate([y, oy[ins]])
        cls = np.concatenate([cls, np.full(ins.sum(), URBAN, np.uint8)])
        n_osm = int(ins.sum())

    ok = tree_ok(x, y, lc, tm, clear_extra, streets)
    if covered is not None and len(x):
        ok &= ~covered(x, y)
    x, y, cls = x[ok], y[ok], cls[ok]

    n = len(x)
    kind = np.where(rng.random(n) < conifer_ratio, 0, 1)
    kind[cls == URBAN] = 1
    bush = (cls == SHRUB) & (rng.random(n) < 0.6)
    kind[bush] = 2
    height = np.where(kind == 0, rng.uniform(9, 20, n), rng.uniform(7, 15, n))
    height[kind == 2] = rng.uniform(1.5, 3.5, bush.sum())
    width = np.where(kind == 0, height * rng.uniform(0.38, 0.5, n), height * rng.uniform(0.7, 0.95, n))
    width[kind == 2] = height[kind == 2] * rng.uniform(1.2, 1.8, bush.sum())
    return {"x": x, "y": y, "h": height, "w": width, "kind": kind, "n_osm": n_osm}


def merge_trees(sets, tm, max_trees, seed=2):
    """Concatenate tree dicts and cap the count, keeping trees near the track first."""
    keys = ("x", "y", "h", "w", "kind")
    sets = [t for t in sets if t is not None and len(t["x"])]
    if not sets:
        return {k: np.zeros(0) for k in keys}
    out = {k: np.concatenate([np.asarray(t[k]) for t in sets]) for k in keys}
    out["kind"] = out["kind"].astype(int)
    if len(out["x"]) > max_trees:
        rng = np.random.default_rng(seed)
        d, _ = tm.road_info(out["x"], out["y"])
        sel = np.argsort(d + rng.uniform(0, 60, len(d)))[:max_trees]
        out = {k: v[sel] for k, v in out.items()}
    return out


def spatial_order(x, y, cell=48.0):
    """Order points cell by cell so that each mesh chunk covers one small area (better culling)."""
    cx = np.floor(np.asarray(x) / cell).astype(np.int64)
    cy = np.floor(np.asarray(y) / cell).astype(np.int64)
    return np.lexsort((np.asarray(x), cx, cy))


ATLAS_COLS = 4          # tree atlas: conifer side | broadleaf side | bush side | top views


def tree_meshes(trees, tm, prefix="TREES"):
    """Multi-plane trees: three vertical leaf cards at 60 degrees, plus a horizontal crown card for
    broadleaf trees and bushes (seen from above / under the canopy). Cards are double-sided by sharing
    vertices between a front and a back triangle pair; normals point mostly up so foliage is evenly lit."""
    n = len(trees["x"])
    if n == 0:
        return []
    order = spatial_order(trees["x"], trees["y"])
    trees = {k: np.asarray(v)[order] for k, v in trees.items()}
    rng = np.random.default_rng(3)
    z = tm.surface_height(trees["x"], trees["y"]) - 0.25
    ang = rng.uniform(0, np.pi, n)
    kind = trees["kind"].astype(int)
    per_tree = 16                                   # 3 vertical cards + 1 horizontal, 4 vertices each
    max_per_part = MAX_VERTS // per_tree
    col = np.where(kind == 0, 0, np.where(kind == 2, 2, 1))
    parts = []
    for start in range(0, n, max_per_part):
        sl = slice(start, min(start + max_per_part, n))
        m = sl.stop - sl.start
        V = np.zeros((m, per_tree, 3))
        N = np.zeros((m, per_tree, 3))
        UV = np.zeros((m, per_tree, 2))
        cx, cy, zz, hh, ww = trees["x"][sl], trees["y"][sl], z[sl], trees["h"][sl], trees["w"][sl]
        u0 = col[sl] / ATLAS_COLS
        du = 1.0 / ATLAS_COLS
        for k in range(3):
            a = ang[sl] + k * np.pi / 3
            dx, dy = np.cos(a), np.sin(a)
            for c, (sx, up) in enumerate(((-1, 0), (1, 0), (1, 1), (-1, 1))):
                vi = k * 4 + c
                V[:, vi, 0] = cx + dx * ww / 2 * sx
                V[:, vi, 1] = cy + dy * ww / 2 * sx
                V[:, vi, 2] = zz + hh * up
                UV[:, vi, 0] = u0 + (0.0 if sx < 0 else du)
                UV[:, vi, 1] = 1.0 - up
        # horizontal crown card (top view texture: broadleaf in the upper half, conifer in the lower half)
        conif = kind[sl] == 0
        zc = zz + hh * np.where(conif, 0.45, np.where(kind[sl] == 2, 0.7, 0.58))
        sc = ww * np.where(conif, 0.75, 0.92) / 2
        a = ang[sl]
        ca, sa = np.cos(a), np.sin(a)
        for c, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
            vi = 12 + c
            V[:, vi, 0] = cx + (ca * sx - sa * sy) * sc
            V[:, vi, 1] = cy + (sa * sx + ca * sy) * sc
            V[:, vi, 2] = zc
            UV[:, vi, 0] = 3 * du + (0.0 if sx < 0 else du)
            UV[:, vi, 1] = np.where(conif, 0.5, 0.0) + (0.0 if sy > 0 else 0.5)
        N[..., 2] = 1.0
        N[:, :12, 0] = rng.uniform(-0.15, 0.15, (m, 1))
        N[:, :12, 1] = rng.uniform(-0.15, 0.15, (m, 1))
        N /= np.linalg.norm(N, axis=2, keepdims=True)
        F = []
        for q in range(4):
            b = q * 4
            F += [(b, b + 1, b + 2), (b, b + 2, b + 3), (b, b + 2, b + 1), (b, b + 3, b + 2)]
        F = np.asarray(F)
        faces = (F[None, :, :] + (np.arange(m) * per_tree)[:, None, None]).reshape(-1, 3)
        part = MeshPart(f"{prefix}_{len(parts):03d}", "trees", V.reshape(-1, 3), faces, UV.reshape(-1, 2))
        part.normals = N.reshape(-1, 3)
        parts.append(part)
    return parts


def street_points(roads, proj, bounds):
    """Dense points along OSM streets with their half widths (for keeping trees off streets)."""
    from .meshes import SIDE_ROAD_SKIP
    from .track import resample

    x0, y0, x1, y1 = bounds
    pts, half = [], []
    for r in roads:
        if r["highway"] in SIDE_ROAD_SKIP and r["highway"] not in ("footway", "path", "cycleway"):
            continue
        xy = proj.latlngs_to_xy(r["coords"])
        if xy[:, 0].max() < x0 or xy[:, 0].min() > x1 or xy[:, 1].max() < y0 or xy[:, 1].min() > y1:
            continue
        try:
            p = resample(xy, 2.0, False)
        except ValueError:
            p = xy
        pts.append(p)
        half.append(np.full(len(p), r["width"] / 2))
    if not pts:
        return np.zeros((0, 2)), np.zeros(0)
    return np.vstack(pts), np.concatenate(half)


# --------------------------------------------------------------------------- textures
def _foliage_blobs(d, rng, cx, cy, rx, ry, n, base, shade_top=True, H=1.0, rmin=0.04, rmax=0.09, W=1.0):
    for _ in range(n):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, 1) ** 0.6
        px = cx + np.cos(ang) * rad * rx
        py = cy + np.sin(ang) * rad * ry
        r = W * rng.uniform(rmin, rmax)
        shade = 1.0 - 0.35 * (py - (cy - ry)) / (2 * ry) if shade_top else rng.uniform(0.75, 1.05)
        col = tuple(int(np.clip(c * shade + rng.integers(-14, 14), 0, 255)) for c in base) + (255,)
        d.ellipse([px - r, py - r, px + r, py + r], fill=col)
    # leaf-sized detail on top: small light and dark speckles inside the crown
    for _ in range(n * 4):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, 1) ** 0.5 * 0.95
        px = cx + np.cos(ang) * rad * rx
        py = cy + np.sin(ang) * rad * ry
        r = W * rng.uniform(0.008, 0.02)
        k = rng.uniform(0.6, 1.35) * (1.0 - 0.25 * (py - (cy - ry)) / (2 * ry) if shade_top else 1.0)
        col = tuple(int(np.clip(c * k, 0, 255)) for c in base) + (255,)
        d.ellipse([px - r, py - r, px + r, py + r], fill=col)


def tree_atlas(size=512) -> Image.Image:
    """RGBA atlas, 4 columns of size x 2*size: conifer side | broadleaf side | bush side |
    top views (broadleaf crown in the upper half, conifer in the lower half)."""
    S = 2  # supersample
    W, H = size * S, size * 2 * S
    rng = np.random.default_rng(11)
    img = Image.new("RGBA", (W * ATLAS_COLS, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # conifer (col 0)
    cx = W / 2
    d.rectangle([cx - W * 0.03, H * 0.80, cx + W * 0.03, H], fill=(82, 60, 40, 255))
    tiers = 10
    for t in range(tiers):
        top = H * (0.02 + 0.08 * t)
        bot = H * (0.18 + 0.075 * t)
        half = W * (0.08 + 0.042 * t)
        g = int(70 + rng.integers(-12, 12))
        d.polygon([(cx, top), (cx - half, bot), (cx + half, bot)], fill=(28, g, 40, 255))
        for _ in range(45):  # ragged needles
            px = cx + rng.uniform(-half, half)
            py = rng.uniform(top + (bot - top) * 0.4, bot)
            r = W * 0.025
            d.polygon([(px, py - r * 2), (px - r, py + r), (px + r, py + r)],
                      fill=(30 + rng.integers(0, 20), 72 + rng.integers(0, 35), 42, 255))

    # broadleaf (col 1): trunk + branches + round crown
    cx = W * 1.5
    d.polygon([(cx - W * 0.05, H), (cx + W * 0.05, H), (cx + W * 0.025, H * 0.55), (cx - W * 0.025, H * 0.55)],
              fill=(88, 68, 50, 255))
    for k in range(5):
        a = rng.uniform(-0.9, 0.9)
        y0 = H * rng.uniform(0.5, 0.7)
        d.line([(cx, y0), (cx + np.sin(a) * W * 0.25, y0 - np.cos(a) * H * 0.18)], fill=(80, 62, 46, 255), width=int(W * 0.015))
    _foliage_blobs(d, rng, cx, H * 0.41, W * 0.42, H * 0.36, 380, (64, 110, 46), W=W)

    # bush (col 2): dense, rounded, down to the ground
    cx = W * 2.5
    _foliage_blobs(d, rng, cx, H * 0.56, W * 0.40, H * 0.40, 420, (66, 104, 48), W=W, rmin=0.05, rmax=0.1)

    # top views (col 3)
    cx = W * 3.5
    _foliage_blobs(d, rng, cx, H * 0.25, W * 0.42, H * 0.21, 260, (66, 112, 48), shade_top=False, W=W)
    for k in range(9):           # conifer seen from above: star of branches
        a = k * 2 * np.pi / 9 + rng.uniform(-0.2, 0.2)
        L = W * rng.uniform(0.3, 0.42)
        p0 = (cx, H * 0.75)
        p1 = (cx + np.cos(a) * L, H * 0.75 + np.sin(a) * L)
        d.line([p0, p1], fill=(30, 74, 42, 255), width=int(W * 0.09))
    _foliage_blobs(d, rng, cx, H * 0.75, W * 0.18, W * 0.18, 60, (32, 80, 44), shade_top=False, W=W, rmin=0.03, rmax=0.06)

    img = img.resize((size * ATLAS_COLS, size * 2), Image.LANCZOS)
    arr = np.asarray(img).copy()
    a = arr[..., 3]
    arr[..., 3] = np.where(a > 110, 255, 0)
    clear_margins(arr, size, size * 2, extra_rows=[(3, size)])
    # bleed colour into transparent pixels to avoid dark fringes with alpha testing
    arr[a <= 110, :3] = [45, 85, 40]
    return Image.fromarray(arr, "RGBA")


def clear_margins(arr, cell_w, cell_h, extra_rows=()):
    """Transparent border round every atlas image (in place). Without it, far away (small mip levels) a card
    picks up its neighbour's edge, or through the texture repeat its own opposite edge: thin floating lines.
    extra_rows: (column, y) splits inside a column (the two top views)."""
    pad = max(2, cell_w // 64)
    H, W = arr.shape[:2]
    for x in range(0, W + 1, cell_w):
        arr[:, max(0, x - pad):min(W, x + pad), 3] = 0
    for y in range(0, H + 1, cell_h):
        arr[max(0, y - pad):min(H, y + pad), :, 3] = 0
    for col, y in extra_rows:
        arr[max(0, y - pad):min(H, y + pad), col * cell_w:(col + 1) * cell_w, 3] = 0
    return arr


CLASS_COLOR = {
    GRASS: (92, 120, 62), FOREST: (58, 74, 44), SHRUB: (88, 102, 58), CROP: (148, 140, 92),
    URBAN: (118, 120, 108), WATER: (52, 78, 92), BARREN: (150, 138, 112), WETLAND: (84, 96, 64), SNOW: (230, 232, 236),
}


def synth_ground(lc: LandCover, xs, ys, seed=5) -> np.ndarray:
    """Colour image (H,W,3) uint8 for a grid of world coordinates, from land-cover classes + noise."""
    X, Y = np.meshgrid(xs, ys)
    cls = lc.sample(X, Y)
    pal = np.array([CLASS_COLOR[i] for i in range(9)], float)
    col = pal[cls]
    n = (0.5 * _value_noise(X, Y, 18.0, seed) + 0.3 * _value_noise(X, Y, 4.0, seed + 1)
         + 0.2 * _value_noise(X, Y, 1.2, seed + 2))
    col *= (0.78 + 0.44 * n)[..., None]
    img = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(1.2))
    return np.asarray(img)


def photo_tinted_ground(pm, lc, xs, ys, seed=5):
    """'Land cover colours' when a photo exists: each surface type (trees, lawn, bare/sand, paved)
    gets the *average* colour the photo shows for it nearby (~6 m), plus procedural grain - so the
    palette matches the place but there are no cars, shadows or photo detail on the ground."""
    from . import photo as photo_mod

    f = max(1, int(round(1.0 / abs(xs[1] - xs[0])))) if len(xs) > 1 else 1
    xs1, ys1 = xs[::f], ys[::f]
    rgb, ok = pm.grid(xs1, ys1)
    F = photo_mod.features(rgb)
    v, s, exg = F["v"], F["s"], F["exg"]
    veg = exg > 0.045
    vref = np.percentile(v[veg & ok], 50) if (veg & ok).sum() > 50 else 0.4
    tree = veg & (v < vref * 0.82)
    lawn = veg & ~tree
    lab = None
    from .roadwidth import rgb_to_lab
    lab = rgb_to_lab(rgb)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    bare = ~veg & (chroma > 9) & (lab[..., 2] > 4)
    paved = ~veg & ~bare
    classes = [tree, lawn, bare, paved]
    sig = 6.0
    soft, cols = [], []
    for m in classes:
        m = m & ok
        w = ndi_gauss(m.astype(np.float32), sig)
        acc = np.stack([ndi_gauss(rgb[..., c] * m, sig) for c in range(3)], -1)
        glob = rgb[m].mean(0) if m.sum() > 20 else np.array([110, 120, 90], np.float32)
        col = np.where(w[..., None] > 1e-3, acc / np.maximum(w[..., None], 1e-3), glob)
        cols.append(col)
        soft.append(ndi_gauss(m.astype(np.float32), 1.0))
    soft = np.stack(soft, 0)
    soft /= np.maximum(soft.sum(0, keepdims=True), 1e-6)
    base = (soft[..., None] * np.stack(cols, 0)).sum(0)
    # areas without photo: plain land-cover colours
    lc_col = synth_ground(lc, xs1, ys1).astype(np.float32)
    base = np.where(ok[..., None], base, lc_col)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((len(xs), len(ys)), Image.BICUBIC)
    out = np.asarray(img, np.float32)
    X, Y = np.meshgrid(xs, ys)
    n = (0.45 * _value_noise(X, Y, 5.0, seed) + 0.35 * _value_noise(X, Y, 1.3, seed + 1)
         + 0.20 * _value_noise(X, Y, 0.45, seed + 2))
    out *= (0.82 + 0.36 * n)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def ndi_gauss(a, sigma):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(a, sigma)


def paint_paved(textures, tiles, polys, pm, seed=11):
    """Paint driveways / parking lots crisply into painted (non-photo) ground textures.

    Each area gets the median colour the photo shows inside it (so a grey concrete pad stays grey and
    fresh black asphalt stays black), fine asphalt grain, a slightly darker rim and an anti-aliased edge - instead of the blurred 'paved' class of the tinted ground."""
    from PIL import ImageDraw

    if not polys or not tiles:
        return 0
    cols = []
    for p in polys:
        x0, y0, x1, y1 = p.bounds
        xs = np.arange(x0, x1, 0.5)
        ys = np.arange(y1, y0, -0.5)
        col = np.array([118, 118, 116], np.float32)
        if pm is not None and len(xs) and len(ys):
            from shapely import contains_xy

            ph, ok = pm.grid(xs, ys)
            X, Y = np.meshgrid(xs, ys)
            m = ok & contains_xy(p, X, Y)
            if m.sum() > 8:
                px = ph[m]
                lum = px.mean(1)
                lo, hi = np.percentile(lum, [20, 80])     # drop cars, paint and shadows
                sel = px[(lum >= lo) & (lum <= hi)]
                if len(sel):
                    col = np.median(sel, 0)
        # clearly readable against grass: dark asphalt, or light concrete when the photo shows concrete
        g = float(col.mean())
        g = 175.0 if g > 150 else float(np.clip(g * 0.72, 62, 105))
        cols.append(np.array([g, g, g * 1.03]))
    from .parallel import pmap

    def paint(t):
        img = textures.get(t["material"])
        if img is None:
            return None
        W, H = img.size
        sx = W / (t["x1"] - t["x0"])
        sy = H / (t["y1"] - t["y0"])
        ss = 2                                            # supersample for clean edges
        mask = Image.new("L", (W * ss, H * ss), 0)
        colimg = Image.new("RGB", (W * ss, H * ss), (0, 0, 0))
        dm, dc = ImageDraw.Draw(mask), ImageDraw.Draw(colimg)
        hit = False
        for p, c in zip(polys, cols):
            bx0, by0, bx1, by1 = p.bounds
            if bx1 < t["x0"] or bx0 > t["x1"] or by1 < t["y0"] or by0 > t["y1"]:
                continue
            hit = True
            ring = np.asarray(p.exterior.coords)
            pts = [((x - t["x0"]) * sx * ss, (t["y1"] - y) * sy * ss) for x, y in ring]
            dm.polygon(pts, fill=255)
            dc.polygon(pts, fill=tuple(int(v) for v in c))
            for hole in p.interiors:
                hp = [((x - t["x0"]) * sx * ss, (t["y1"] - y) * sy * ss) for x, y in np.asarray(hole.coords)]
                dm.polygon(hp, fill=0)
        if not hit:
            return None
        a = np.asarray(mask.resize((W, H), Image.BOX), np.float32) / 255.0
        c = np.asarray(colimg.resize((W, H), Image.BOX), np.float32)
        # colour was box-filtered together with black outside: un-premultiply
        c = c / np.maximum(a[..., None], 1e-3)
        xs = np.linspace(t["x0"], t["x1"], W)
        ys = np.linspace(t["y1"], t["y0"], H)
        X, Y = np.meshgrid(xs, ys)
        grain = (0.55 * _value_noise(X, Y, 0.12, seed) + 0.30 * _value_noise(X, Y, 0.6, seed + 1)
                 + 0.15 * _value_noise(X, Y, 3.0, seed + 2))
        c = c * (0.86 + 0.28 * grain)[..., None]
        # darker rim where the pad meets grass (dirt / curb shadow), ~0.3 m
        rim = np.clip(1.0 - ndi_gauss(a, 0.25 * sx), 0, 1)
        c = c * (1.0 - 0.25 * rim * a)[..., None]
        base = np.asarray(img, np.float32)
        out = base * (1 - a[..., None]) + c * a[..., None]
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")

    n = 0
    for t, img in zip(tiles, pmap(paint, tiles)):
        if img is not None:
            textures[t["material"]] = img
            n += 1
    return n


# --------------------------------------------------------------------------- stylised ground
# 'Land cover colours' with a photo: every surface type read on the photo gets a clean, saturated colour
# (like a well-kept game track) instead of the photo's washed-out average. The photo only decides WHERE
# each surface is, plus a faint trace of its light/dark detail.
GROUND_STYLE = {
    #  name        base RGB          alt RGB (mixed in by noise)   brightness var  grain
    "lawn":     ((84, 132, 46),   (102, 146, 54),   0.10, 0.10),
    "wild":     ((104, 124, 54),  (128, 132, 66),   0.12, 0.16),
    "dry":      ((150, 144, 86),  (128, 128, 70),   0.10, 0.16),
    "forest":   ((50, 72, 36),    (62, 70, 40),     0.12, 0.20),
    "sand":     ((184, 168, 130), (168, 150, 112),  0.06, 0.10),
    "soil":     ((124, 102, 78),  (108, 92, 70),    0.08, 0.14),
    "gravel":   ((150, 142, 122), (134, 128, 112),  0.06, 0.18),
    "paved":    ((86, 87, 90),    (100, 100, 102),  0.05, 0.12),
}
_STYLE_ORDER = list(GROUND_STYLE)


def ground_classes(an):
    """Hard class index map (analysis grid) from the photo analysis; -1 = no photo here."""
    from . import photo as photo_mod

    L, a, b = an.L, an.a, an.b
    exg = photo_mod.features(an.rgb)["exg"]
    veg = ((exg > 0.02) | (a < -2.5))
    cls = np.full(L.shape, _STYLE_ORDER.index("gravel"), np.int8)
    sand = ~veg & (L > 58) & (b > 7)
    soil = ~veg & ~sand & (a > 2.5) & (b > 8)
    cls[soil] = _STYLE_ORDER.index("soil")
    cls[sand] = _STYLE_ORDER.index("sand")
    cls[an.dry] = _STYLE_ORDER.index("dry")
    cls[veg] = _STYLE_ORDER.index("lawn")
    cls[an.wild] = _STYLE_ORDER.index("wild")
    cls[an.lawn] = _STYLE_ORDER.index("lawn")
    cls[getattr(an, "pool", np.zeros_like(veg))] = _STYLE_ORDER.index("lawn")   # no pools on the track
    cls[an.canopy] = _STYLE_ORDER.index("forest")
    cls[an.paved | getattr(an, "street", np.zeros_like(veg))] = _STYLE_ORDER.index("paved")
    cls = tidy_classes(cls, an.res, getattr(an, "block", None))
    cls[~an.valid] = -1
    return cls


BARE = ("gravel", "sand", "soil", "dry")
TIDY_MODE_M = 1.25        # m: radius of the majority vote that straightens the borders between surfaces
TIDY_BARE_WIDTH = 4.0     # m: bare strips narrower than this (fringes round driveways, specks) become grass
TIDY_HOUSE_M = 3.0        # m: bare ground this close to a house (eave shadows, mixed roof pixels) is lawn
TIDY_MIN_AREA = {"gravel": 60.0, "sand": 60.0, "soil": 60.0, "dry": 80.0, "forest": 6.0, "lawn": 20.0,
                 "wild": 20.0}


def tidy_classes(cls, res, houses=None):
    """Fewer, larger, simpler surfaces: what a groundskeeper would leave, not every colour change of the photo.
      1. majority vote within TIDY_MODE_M: smooth, logical borders instead of ragged pixel edges;
      2. sand / gravel / soil / dry grass narrower than TIDY_BARE_WIDTH (the light fringe round driveways and
         along kerbs, worn spots in a lawn) is grass; bare ground stays only where there is a real area of it;
      3. patches under TIDY_MIN_AREA take the surface around them.
    Paved areas and streets are kept exactly as found (they are drawn from their outlines)."""
    from scipy.ndimage import binary_opening, distance_transform_edt, label, uniform_filter

    paved_k = _STYLE_ORDER.index("paved")
    paved = cls == paved_k
    nk = len(_STYLE_ORDER)
    r = max(1, int(round(TIDY_MODE_M / res)))
    votes = np.stack([uniform_filter((cls == k).astype(np.float32), 2 * r + 1) if k != paved_k
                      else np.full(cls.shape, -1.0, np.float32) for k in range(nk)], 0)
    cls = np.where(paved, paved_k, np.argmax(votes, 0)).astype(np.int8)
    del votes

    def refill(drop):
        nonlocal cls
        drop &= ~paved
        if drop.any() and (~drop).any():
            _, (ri, ci) = distance_transform_edt(drop, return_indices=True)
            cls = cls[ri, ci]

    # thin bare strips -> the vegetation next to them (never into paved)
    bare = np.isin(cls, [_STYLE_ORDER.index(n) for n in BARE])
    rb = max(1, int(round(TIDY_BARE_WIDTH / 2 / res)))
    yy, xx = np.mgrid[-rb:rb + 1, -rb:rb + 1]
    disk = xx * xx + yy * yy <= rb * rb
    thin = bare & ~binary_opening(bare, structure=disk)
    if houses is not None and houses.any():
        thin |= bare & (distance_transform_edt(~houses) * res < TIDY_HOUSE_M)
    if thin.any():
        veg = ~bare & ~paved & (cls >= 0)
        if veg.any():
            _, (ri, ci) = distance_transform_edt(~veg, return_indices=True)
            cls = np.where(thin, cls[ri, ci], cls)
    # small patches take the surface around them
    small = np.zeros(cls.shape, bool)
    px = res ** 2
    for k, name in enumerate(_STYLE_ORDER):
        if name == "paved":
            continue
        lab, n = label(cls == k)
        if n == 0:
            continue
        area = np.bincount(lab.ravel())[1:] * px
        small |= np.isin(lab, np.nonzero(area < TIDY_MIN_AREA.get(name, 8.0))[0] + 1)
    refill(small)
    return cls


def tidy_masks(an):
    """Tidy ground classes on the analysis (an.cls) and make the grass masks follow them: short tufts only on
    lawn, tall clumps only on wild / dry grass, nothing on sand, gravel, soil or paved ground."""
    cls = ground_classes(an)
    an.cls = cls
    an.lawn = cls == _STYLE_ORDER.index("lawn")
    an.wild = np.isin(cls, [_STYLE_ORDER.index("wild"), _STYLE_ORDER.index("dry")])
    an.dry = cls == _STYLE_ORDER.index("dry")
    return cls


def stylised_ground(an, lc, tiles, textures, res, seed=5, log=print):
    """Paint every ground tile from the photo classes with the GROUND_STYLE palette."""
    from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter

    cls = an.cls if getattr(an, "cls", None) is not None else ground_classes(an)
    soft = max(0.6, 0.35 / an.res)                    # ~0.35 m soft edge between surfaces
    weights = [gaussian_filter((cls == k).astype(np.float32), soft) for k in range(len(_STYLE_ORDER))]
    nophoto = gaussian_filter((cls < 0).astype(np.float32), soft)
    # faint trace of the photo's own light/dark detail inside each surface
    L = an.L.astype(np.float32)
    mean = uniform_filter(L, 9)
    sd = np.sqrt(np.clip(uniform_filter(L * L, 9) - mean * mean, 1, None))
    detail = np.clip((L - mean) / sd, -2.5, 2.5)
    counts = {k: float((cls == i).mean()) for i, k in enumerate(_STYLE_ORDER)}
    log("  ground: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in counts.items() if v > 0.005))
    from .parallel import pmap

    def paint(t):
        wpx = int(min(np.ceil((t["x1"] - t["x0"]) / res), 4096))
        hpx = int(min(np.ceil((t["y1"] - t["y0"]) / res), 4096))
        xs = np.linspace(t["x0"], t["x1"], wpx)
        ys = np.linspace(t["y1"], t["y0"], hpx)
        X, Y = np.meshgrid(xs, ys)
        rr = (an.y1 - Y) / an.res - 0.5
        cc = (X - an.x0) / an.res - 0.5
        coords = [rr.ravel(), cc.ravel()]

        def samp(arr, cval=0.0):
            return map_coordinates(arr, coords, order=1, mode="constant", cval=cval).reshape(X.shape)

        big = _value_noise(X, Y, 23.0, seed)
        # two rotated noises averaged: no visible square grid
        c30, s30 = 0.866, 0.5
        mid = 0.5 * _value_noise(X, Y, 5.0, seed + 1) + 0.5 * _value_noise(c30 * X - s30 * Y, s30 * X + c30 * Y, 3.1, seed + 7)
        fine = 0.6 * _value_noise(X, Y, 0.35, seed + 2) + 0.4 * _value_noise(X, Y, 0.9, seed + 3)
        det = samp(detail)
        out = np.zeros(X.shape + (3,), np.float32)
        wsum = np.zeros(X.shape, np.float32)
        for k, name in enumerate(_STYLE_ORDER):
            w = samp(weights[k])
            if w.max() < 1e-3:
                continue
            base, alt, var, grain = GROUND_STYLE[name]
            mix = np.clip((mid - 0.5) * 1.6 + 0.5, 0, 1)[..., None]
            col = np.array(base, np.float32) * (1 - mix) + np.array(alt, np.float32) * mix
            shade = (1 + var * (big - 0.5) * 2) * (1 + grain * (fine - 0.5) * 2) * (1 + 0.06 * det)
            out += w[..., None] * col * shade[..., None]
            wsum += w
        # outside the photo: plain land-cover palette
        npw = samp(nophoto, cval=1.0)
        f = max(1, int(round(1.0 / res)))
        if npw.max() > 1e-3:
            small = synth_ground(lc, xs[::f], ys[::f])
            lcimg = np.asarray(Image.fromarray(small).resize((wpx, hpx), Image.BICUBIC), np.float32)
            out += npw[..., None] * lcimg
            wsum += npw
        rgb = out / np.maximum(wsum, 1e-4)[..., None]
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

    for t, img in zip(tiles, pmap(paint, tiles)):         # one tile per core
        textures[t["material"]] = img

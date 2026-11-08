"""Ground materials from ambientCG photos (asphalt, grass, gravel, dirt / sand), in two ways:

1. Baked: each ground tile (already painted: land-cover colours or aerial photo, asphalt, driveways) gets the
   photos' structure at its own resolution. At 25 cm per pixel this is colour variation (patches, worn spots)
   more than detail: grass blades and gravel stones are smaller than a pixel.
2. Close-up detail for Assetto Corsa (ksMultilayer, the shader Kunos uses for terrain and roads): per tile a
   low-resolution mask says how much of each surface is where (R asphalt, G grass, B gravel, A dirt / sand) and
   four small photo textures, repeated every few metres, add their grain near the car. Far away the tile
   texture alone is seen.

The surface of each pixel is read from the painted tile itself (so it works the same for land-cover colours
and aerial photos): green -> grass, dark grey -> asphalt, warm and saturated -> dirt / sand, the rest -> gravel.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import materials

# surface: (photo key, mask channel, bake: metres per repeat and strength, close-up: metres per repeat, contrast)
SURFACES = {
    "asphalt": ("asphalt", 0, 8.0, 0.07, 3.0, 0.7),
    "grass": ("grass", 1, 10.0, 0.12, 2.0, 1.0),
    "gravel": ("gravel", 2, 6.0, 0.10, 2.0, 0.9),
    "dirt": ("dirt", 3, 10.0, 0.10, 3.0, 0.8),
}
MASK_RES = 0.5          # m per mask pixel
DETAIL_PX = 1024


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def surface_weights(rgb):
    """(H, W, 3) colours -> (H, W, 4) weights asphalt, grass, gravel, dirt (sum 1)."""
    from scipy.ndimage import gaussian_filter

    c = np.asarray(rgb, np.float32)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = c.max(-1), c.min(-1)
    V = mx / 255.0
    sat = (mx - mn) / np.maximum(mx, 1.0)
    gr = (g - r) / (r + g + b + 1.0)              # green over red: grass and forest, not sand / dry soil
    grass = _sig((gr - 0.03) / 0.012) * _sig((g - b - 8.0) / 4.0)
    asphalt = (1 - grass) * _sig((0.13 - sat) / 0.035) * _sig((0.43 - V) / 0.04)
    dirt = (1 - grass) * (1 - asphalt) * _sig((sat - 0.24) / 0.03) * _sig((r - b - 18.0) / 6.0)
    gravel = np.clip(1.0 - grass - asphalt - dirt, 0, 1)
    w = np.stack([asphalt, grass, gravel, dirt], -1)
    w = np.stack([gaussian_filter(w[..., k], 0.7) for k in range(4)], -1)
    return w / np.maximum(w.sum(-1, keepdims=True), 1e-4)


def _tiled(d, X, Y, repeat_m):
    """Sample a seamless detail array (zero mean) at world coordinates, one repeat every repeat_m metres."""
    h, w = d.shape
    ix = (np.floor((X / repeat_m % 1.0) * w).astype(np.int64)) % w
    iy = (np.floor((-Y / repeat_m % 1.0) * h).astype(np.int64)) % h
    return d[iy, ix]


def finish_ground(textures, tiles, res, photos, bake=True, close_up=True, log=print):
    """Bake the photos into the ground tiles and/or build the ksMultilayer masks and details.
    -> {tile material: {"textures": {...}, "props": {...}}} for the KN5 (empty when close_up is off)."""
    from .parallel import pmap

    have = {k for k, (key, *_r) in SURFACES.items() if key in photos}
    if bake and have:
        log(f"  baking photo materials into the ground ({', '.join(sorted(have))})")
    bake_d = {}
    for name, (key, ch, rep_b, strength, _rc, _c) in SURFACES.items():
        if bake and key in photos:
            px = max(8, int(round(rep_b / res)))
            bake_d[name] = (materials.detail(photos[key], px, px, highpass=0.12), rep_b, strength)

    def work(t):
        img = textures.get(t["material"])
        if img is None:
            return None, None
        W, H = img.size
        mw = max(4, int(round((t["x1"] - t["x0"]) / MASK_RES)))
        mh = max(4, int(round((t["y1"] - t["y0"]) / MASK_RES)))
        small = np.asarray(img.resize((mw, mh), Image.BOX), np.float32)
        wts = surface_weights(small)
        out_img = None
        if bake_d:
            full = np.asarray(img, np.float32)
            wb = np.stack([np.asarray(Image.fromarray(wts[..., k]).resize((W, H), Image.BILINEAR), np.float32)
                           for k in range(4)], -1)
            X, Y = np.meshgrid(np.linspace(t["x0"], t["x1"], W, dtype=np.float32),
                               np.linspace(t["y1"], t["y0"], H, dtype=np.float32))
            mod = np.ones((H, W), np.float32)
            for name, (d, rep_b, strength) in bake_d.items():
                ch = SURFACES[name][1]
                mod += wb[..., ch] * strength * _tiled(d, X, Y, rep_b)
            out_img = Image.fromarray(np.clip(full * mod[..., None], 0, 255).astype(np.uint8), "RGB")
        mask = Image.fromarray(np.clip(wts * 255 + 0.5, 0, 255).astype(np.uint8), "RGBA") if close_up else None
        return out_img, mask

    results = pmap(work, tiles)
    multilayer = {}
    if close_up:
        names = ["asphalt", "grass", "gravel", "dirt"]
        for i, name in enumerate(names):
            key = SURFACES[name][0]
            textures[f"gdetail_{name}"] = materials.close_up_detail(photos.get(key), DETAIL_PX, SURFACES[name][5],
                                                                    seed=40 + i)
    for t, (img, mask) in zip(tiles, results):
        if img is not None:
            textures[t["material"]] = img
        if mask is not None:
            mname = "gmask_" + t["material"][len("ground_"):]
            textures[mname] = mask
            size = t["x1"] - t["x0"]
            multilayer[t["material"]] = {
                "textures": {"txMask": mname, "txDetailR": "gdetail_asphalt", "txDetailG": "gdetail_grass",
                             "txDetailB": "gdetail_gravel", "txDetailA": "gdetail_dirt"},
                "props": {"magicMult": 2.0,
                          "multR": round(size / SURFACES["asphalt"][4], 3),
                          "multG": round(size / SURFACES["grass"][4], 3),
                          "multB": round(size / SURFACES["gravel"][4], 3),
                          "multA": round(size / SURFACES["dirt"][4], 3)},
            }
    return multilayer


# ---------------------------------------------------------------- sharp ground next to the track
# A painted ground texture can't show grass blades or soil from the driver's seat (even at 7 cm per pixel), so
# the ground within near_ground m of the track is split by surface and each surface gets a real photo repeated
# every few metres (shared textures: costs vertices, not texture memory). The surface of each triangle is the
# land-cover class whose colour is nearest to the ground texture there; each material is tinted to the median
# colour of its triangles, so it matches the ground further away.
NEAR_CLASSES = {   # class: (photo, metres per repeat, contrast)
    "lawn": ("grass", 2.0, 1.0), "wild": ("grass", 2.0, 1.1), "dry": ("grass", 2.0, 1.0),
    "forest": ("dirt", 3.0, 1.0), "sand": ("dirt", 3.0, 0.6), "soil": ("dirt", 3.0, 0.9),
    "gravel": ("gravel", 2.0, 0.9), "paved": ("asphalt", 2.5, 1.0),
}


def sample_ground(textures, tiles, x, y):
    """Colours of the ground textures at world points (bilinear)."""
    from scipy.ndimage import map_coordinates

    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.zeros((len(x), 3), np.float32)
    for t in tiles:
        img = textures.get(t["material"])
        if img is None:
            continue
        sel = (x >= t["x0"]) & (x < t["x1"]) & (y >= t["y0"]) & (y < t["y1"])
        if not sel.any():
            continue
        arr = np.asarray(img, np.float32)
        H, W = arr.shape[:2]
        cc = (x[sel] - t["x0"]) / (t["x1"] - t["x0"]) * W - 0.5
        rr = (t["y1"] - y[sel]) / (t["y1"] - t["y0"]) * H - 0.5
        out[sel] = np.stack([map_coordinates(arr[..., c], [rr, cc], order=1, mode="nearest") for c in range(3)], -1)
    return out


def near_regions(textures, tiles, ngeom, paved=(), exclude=None, res=0.5, log=print):
    """The surfaces of the ground near the track, as smooth regions:
    colours sampled every res m (only inside ngeom), averaged over ~1.5 m, nearest land-cover palette colour,
    driveways / lots (paved polygons) on top, specks under 3 m2 merged into their neighbours, then each class
    blurred and the strongest kept (round shapes). Its outlines, simplified to remove the pixel steps, become
    vertices of the ground so the materials meet on smooth lines.
    -> classify(x, y) -> class name, median colour per class, outline points (n, 2)."""
    from rasterio import features
    from rasterio.transform import from_origin
    from scipy.ndimage import gaussian_filter
    from shapely.geometry import shape

    from .landcover import GROUND_STYLE

    names = list(NEAR_CLASSES)
    x0, y0, x1, y1 = ngeom.bounds
    x0, y0, x1, y1 = x0 - 2, y0 - 2, x1 + 2, y1 + 2
    W, H = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    T = from_origin(x0, y1, res, res)
    inside = features.rasterize([(ngeom.buffer(1.0), 1)], out_shape=(H, W), transform=T, fill=0,
                                dtype="uint8").astype(bool)
    rr, cc = np.nonzero(inside)
    xs = x0 + (cc + 0.5) * res
    ys = y1 - (rr + 0.5) * res
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[rr, cc] = sample_ground(textures, tiles, xs, ys)
    paved = [p for p in paved if p is not None and not p.is_empty]
    pm = (features.rasterize([(p, 1) for p in paved], out_shape=(H, W), transform=T, fill=0, dtype="uint8") > 0) \
        if paved else np.zeros((H, W), bool)
    # colours to read the surface from: not the asphalt painted under streets / track / driveways (it would
    # spill a dark fringe around them), filled in from the ground around (normalised blur)
    wgt = inside & ~pm
    if exclude is not None and not exclude.is_empty:
        wgt &= ~(features.rasterize([(exclude.buffer(1.2), 1)], out_shape=(H, W), transform=T, fill=0,
                                    dtype="uint8") > 0)
    wf = wgt.astype(np.float32)
    m = gaussian_filter(wf, 3.0)
    blur = np.stack([gaussian_filter(rgb[..., c] * wf, 3.0) for c in range(3)], -1)
    for _ in range(3):                                   # grow into wide masked areas
        hole = m < 0.05
        if not hole.any():
            break
        m2 = gaussian_filter(m, 6.0)
        blur = np.where(hole[..., None], np.stack([gaussian_filter(blur[..., c], 6.0) for c in range(3)], -1), blur)
        m = np.where(hole, m2, m)
    blur = blur / np.maximum(m, 1e-4)[..., None]
    pal = np.array([(np.asarray(GROUND_STYLE[n][0], float) + GROUND_STYLE[n][1]) / 2 for n in names])
    pch = pal / pal.sum(1, keepdims=True)
    pbr = np.log(pal.sum(1))
    v = blur[rr, cc]
    ch = v / np.maximum(v.sum(1, keepdims=True), 1.0)
    br = np.log(np.maximum(v.sum(1), 1.0))
    dist = 120.0 * ((ch[:, None, :] - pch[None]) ** 2).sum(-1) + 0.08 * (br[:, None] - pbr[None]) ** 2
    ip = names.index("paved")
    dist[:, ip] = np.inf                                 # 'paved' only where a driveway / lot was found
    cls = np.full((H, W), 255, np.uint8)
    cls[rr, cc] = dist.argmin(1)
    cls = features.sieve(cls, size=int(3.0 / res / res), mask=inside)
    # round shapes: blur each class, keep the strongest
    score = np.stack([gaussian_filter((cls == i).astype(np.float32), 1.6) for i in range(len(names))], -1)
    score[..., ip] = -1.0
    cls = np.where(inside, score.argmax(-1), 255).astype(np.uint8)
    base = cls.copy()                                    # the surface under / around the driveways
    cls[pm & inside] = ip
    pu = None
    if paved:
        from shapely.ops import unary_union

        pu = unary_union(paved).buffer(0)
    colours = {}
    for i, n in enumerate(names):
        sel = cls[rr, cc] == i
        if sel.sum() > 20:
            colours[n] = np.median(rgb[rr[sel], cc[sel]], axis=0)
    # outlines -> ground vertices
    pts = []
    for geom, val in features.shapes(cls, mask=inside & (cls != 255), transform=T):
        poly = shape(geom).simplify(0.35)
        if poly.is_empty or poly.area < 2.0:
            continue
        for ring in [poly.exterior] + list(poly.interiors):
            L = ring.length
            if L < 1.0:
                continue
            for d in np.arange(0, L, 0.8):
                q = ring.interpolate(d)
                pts.append((q.x, q.y))
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts):
        from shapely import contains_xy
        from scipy.spatial import cKDTree

        ok = contains_xy(ngeom.buffer(-0.3), pts[:, 0], pts[:, 1])
        if pu is not None:
            # driveway edges come from their own outlines (straight lines), not from the 0.5 m raster
            from shapely import distance, points as _points

            ok &= distance(pu.boundary, _points(pts)) > 0.5
            ring_pts = []
            for g in getattr(pu, "geoms", [pu]):
                for ring in [g.exterior] + list(g.interiors):
                    for d in np.arange(0, ring.length, 0.5):
                        q = ring.interpolate(d)
                        ring_pts.append((q.x, q.y))
                    ring_pts += list(ring.coords)[:-1]       # the corners themselves
            rp = np.asarray(ring_pts, float).reshape(-1, 2)
            pts = np.vstack([pts, rp])
            ok = np.concatenate([ok, contains_xy(ngeom.buffer(-0.3), rp[:, 0], rp[:, 1])])
        if exclude is not None:
            ok &= ~contains_xy(exclude.buffer(0.5), pts[:, 0], pts[:, 1])
        pts = pts[ok]
        # neighbouring regions share their outline: keep one point per 0.35 m
        if len(pts):
            keep = np.ones(len(pts), bool)
            tree = cKDTree(pts)
            for i, j in sorted(tree.query_pairs(0.35)):
                if keep[i] and keep[j]:
                    keep[j] = False
            pts = pts[keep]

    def classify(x, y):
        c = np.clip(((np.asarray(x) - x0) / res).astype(int), 0, W - 1)
        r = np.clip(((y1 - np.asarray(y)) / res).astype(int), 0, H - 1)
        k = base[r, c].copy()
        if pu is not None:
            from shapely import contains_xy

            on = contains_xy(pu, np.asarray(x, float), np.asarray(y, float))
            k = np.where(on, ip, np.where(k == ip, 255, k))
        k = np.where(k == 255, names.index("lawn"), k)
        return np.array(names)[k]

    log(f"  ground beside the track: {len(colours)} surfaces, {len(pts):,} outline points")
    return classify, colours, pts


def near_textures(colours, photos, size=1024):
    """Tiled photo texture per near class (tinted to its median colour)."""
    from . import ground as ground_mod
    from .landcover import GROUND_STYLE

    out = {}
    for i, (n, (key, _rep, contrast)) in enumerate(NEAR_CLASSES.items()):
        c = colours.get(n)
        if c is None:
            c = (np.asarray(GROUND_STYLE[n][0], float) + GROUND_STYLE[n][1]) / 2
        out[f"near_{n}"] = ground_mod.tiled_texture(photos.get(key), tuple(float(v) for v in c), size,
                                                    contrast=contrast, seed=60 + i)
    return out

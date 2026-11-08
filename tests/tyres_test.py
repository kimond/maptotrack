"""Tyre walls read from a (synthetic) aerial photo - no network.

  python tests/tyres_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from app import elevation, tyres
from app.photo import PhotoMap
from app.track import build_centerline
from markings_test import Proj


def kart_photo(centre, wall_off, wall_s, res=0.2, half=4.0, ss=4, seed=0):
    """Light asphalt on grass; a beaded dark tyre wall (0.6 m tyres) centred wall_off m left of the centre line,
    from wall_s[0] to wall_s[1] m along the road; JPEG-like blur."""
    rng = np.random.default_rng(seed)
    x0, y0 = centre.min(0) - 15
    x1, y1 = centre.max(0) + 15
    W, H = int((x1 - x0) / res) * ss, int((y1 - y0) / res) * ss
    xs = x0 + (np.arange(W) + 0.5) * res / ss
    ys = y1 - (np.arange(H) + 0.5) * res / ss
    X, Y = np.meshgrid(xs, ys)
    t = np.gradient(centre, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))])
    P = np.column_stack([X.ravel(), Y.ravel()])
    d, i = cKDTree(centre).query(P)
    off = ((P - centre[i]) * n[i]).sum(1).reshape(H, W)
    along = s[i].reshape(H, W)
    del P, d, i
    img = np.empty((H, W, 3), np.float32)
    img[:] = (92, 120, 66)                                        # grass
    img[np.abs(off) < half] = (160, 158, 152)                     # asphalt
    # tyres: rings of dark rubber, a lighter centre (the hole shows the ground / the tyre below)
    du = (along % 0.6) - 0.3
    dv = off - wall_off
    r = np.hypot(du, dv)
    on = (along > wall_s[0]) & (along < wall_s[1])
    img[on & (r < 0.3)] = (38, 38, 40)
    img[on & (r < 0.12)] = (70, 70, 66)
    img = img.reshape(H // ss, ss, W // ss, ss, 3).mean((1, 3))
    img += rng.normal(0, 2.5, img.shape)
    img = ndi.gaussian_filter(img, (0.9, 0.9, 0))
    lum = img.mean(-1, keepdims=True)
    img = lum + ndi.gaussian_filter(img - lum, (1.2, 1.2, 0))      # JPEG: colour at half resolution
    return PhotoMap(np.clip(img, 0, 255).astype(np.uint8), np.ones((H // ss, W // ss), bool), x0, y1, res)


def main():
    proj = Proj()
    xs = np.linspace(0, 200, 201)
    centre = np.column_stack([xs, 5 * np.sin(xs / 30.0)])
    ll = [(y / 1e5, x / 1e5) for x, y in centre[::5]]
    cl = build_centerline(ll, proj, elevation.FlatDEM(), False, corner_smoothing=0)
    dense = np.column_stack([np.interp(np.linspace(0, cl.length, 3000), cl.s, cl.p[:, 0]),
                             np.interp(np.linspace(0, cl.length, 3000), cl.s, cl.p[:, 1])])
    hl = np.full(len(cl.p), 4.0)
    hr = hl.copy()
    # a tyre wall 1.5 m outside the left edge (centre at 5.5 m), from 40 to 160 m
    pm = kart_photo(dense, 5.5, (40.0, 160.0))
    walls, info = tyres.detect(pm, cl, hl, hr)
    print(info, [(w["side"], round(w["s0"]), round(w["s1"]), round(float(np.median(w["rel"])), 2)) for w in walls])
    assert walls and all(w["side"] == "left" for w in walls), "wall on the wrong side / none"
    w = max(walls, key=lambda w: w["s1"] - w["s0"])
    assert w["s0"] < 50 and w["s1"] > 150, (w["s0"], w["s1"])
    assert abs(float(np.median(w["rel"])) - 1.5) < 0.3, np.median(w["rel"])
    assert sum(x["s1"] - x["s0"] for x in walls) < 135, "wall longer than painted"

    # a plain road on grass: no wall
    pm2 = kart_photo(dense, 5.5, (0.0, 0.0), seed=1)
    walls2, _ = tyres.detect(pm2, cl, hl, hr)
    assert not walls2, [(w["s0"], w["s1"]) for w in walls2]

    # meshes: one stack every ~0.6 m, standing on the ground, never on the asphalt
    class TM:
        def surface_height(self, x, y):
            return np.zeros(np.shape(x))

    parts = tyres.meshes(walls, TM())
    st = tyres.stacks(walls)
    print(f"{len(st)} stacks, {sum(len(p.vertices) for p in parts)} vertices")
    vis = [p for p in parts if p.name.startswith("TYRES_")]
    col = [p for p in parts if p.name.startswith("1WALL_TYRES")]
    assert vis and col and all(p.material == "tyres" for p in parts)
    assert abs(len(st) - 120 / 0.58) < 25
    d, _ = cKDTree(cl.p).query(st)
    assert d.min() > 4.0 + 0.3 - 0.05, d.min()
    v = np.vstack([p.vertices for p in vis])
    assert 0.55 < v[:, 2].max() < 0.9, v[:, 2].max()            # 3-4 tyres
    assert all(getattr(p, "lod_out", 0) > 0 for p in vis)        # detailed tyres only up close
    # the collision stack stays inside the tyres (never seen around them)
    c = np.vstack([p.vertices for p in col])
    dc, _ = cKDTree(st).query(c[:, :2])
    assert dc.max() < 0.28
    # a tyre: rounded, with its hole (no vertex between r 0.0 and 0.18 m of its axis)
    V, F, UV, N = tyres._tyre(0, 0, 0, 0, 1, 1, (0, 0))
    r = np.hypot(V[:, 0], V[:, 1])
    assert r.min() > 0.18 and abs(r.max() - 0.30) < 0.005
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert ((fn * N[F[:, 0]]).sum(1) > 0).all(), "faces not facing out"

    # texture: same layout with or without the rubber photo (only its grain is used)
    from PIL import Image

    from app import materials

    rng = np.random.default_rng(3)
    fake = Image.fromarray(rng.integers(20, 90, (512, 512, 3), dtype=np.uint8))
    t0, t1 = np.asarray(tyres.texture(None), float), np.asarray(tyres.texture(fake), float)
    assert t0.shape == t1.shape == (256, 1024, 3)
    assert abs(t0.mean() - t1.mean()) < 4 and t1.std() > t0.std(), (t0.mean(), t1.mean(), t0.std(), t1.std())
    assert t1[:, 780:].mean() < 20, "the inside of the hole is dark"
    m = materials.EXTRA["rubber"]["match"]
    assert materials._colour_member(["x/plastic_0022_normal_1k.png", "x/plastic_0022_basecolor_1k.jpg",
                                     "x/plastic_0022_ao_1k.jpg"], m) == "x/plastic_0022_basecolor_1k.jpg"
    assert materials._colour_member(["a_roughness.jpg", "a_color.jpg"], m) == "a_color.jpg"

    # auto mode: on when OSM maps a raceway along the track
    road = [{"highway": "raceway", "coords": ll}]
    assert tyres.is_race_track(road, proj, cl)
    assert not tyres.is_race_track([{"highway": "residential", "coords": ll}], proj, cl)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

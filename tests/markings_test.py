"""Road markings read from a (synthetic) aerial photo - no network.

  python tests/markings_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import ndimage as ndi

from app import elevation, markings
from app.photo import PhotoMap
from app.track import build_centerline


class Proj:
    """Local metres <-> fake lat/lon (1 m = 1e-5 deg), enough for build_centerline."""

    def latlngs_to_xy(self, ll):
        a = np.asarray(ll, float)
        return np.column_stack([a[:, 1] * 1e5, a[:, 0] * 1e5])

    def to_latlon(self, x, y):
        return np.asarray(y) / 1e5, np.asarray(x) / 1e5


def road_photo(centre, painted, res=0.2, half=3.6, shadow=None, seed=0, ss=4):
    """A light-grey road on grass, painted lines [(offset, colour, dash, period)] at 12 cm, blurred to the
    photo's resolution. centre: dense (x, y) of the road centre."""
    rng = np.random.default_rng(seed)
    x0, y0 = centre.min(0) - 25
    x1, y1 = centre.max(0) + 25
    W, H = int((x1 - x0) / res) * ss, int((y1 - y0) / res) * ss      # drawn 4x finer, then averaged
    xs = x0 + (np.arange(W) + 0.5) * res / ss
    ys = y1 - (np.arange(H) + 0.5) * res / ss
    X, Y = np.meshgrid(xs, ys)
    from scipy.spatial import cKDTree

    t = np.gradient(centre, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))])
    d, i = cKDTree(centre).query(np.column_stack([X.ravel(), Y.ravel()]))
    off = ((np.column_stack([X.ravel(), Y.ravel()]) - centre[i]) * n[i]).sum(1).reshape(H, W)
    along = s[i].reshape(H, W)
    img = np.empty((H, W, 3), np.float32)
    img[:] = (96, 128, 64)                                       # grass
    road = np.abs(off) < half
    img[road] = (168, 166, 160)                                  # light asphalt (as in Saguenay)
    for o, colour, dash, period in painted:
        on = np.abs(off - o) < 0.06
        if dash:
            on &= (along % period) < dash
        img[on & road] = (235, 235, 230) if colour == "white" else (240, 216, 120)
    if shadow is not None:
        sx, sy, r = shadow
        img[np.hypot(X - sx, Y - sy) < r] *= 0.45
    del X, Y, off, along, road
    img = img.reshape(H // ss, ss, W // ss, ss, 3).mean((1, 3))
    H, W = H // ss, W // ss
    img += rng.normal(0, 2.5, img.shape)
    img = ndi.gaussian_filter(img, (0.9, 0.9, 0))               # 20 cm photo, a little soft
    # JPEG keeps colour at a lower resolution than lightness
    lum = img.mean(-1, keepdims=True)
    chroma = ndi.gaussian_filter(img - lum, (2.5, 2.5, 0))
    img = np.clip(lum + chroma, 0, 255).astype(np.uint8)
    return PhotoMap(img, np.ones((H, W), bool), x0, y1, res)


def main():
    proj = Proj()
    # a gently winding 400 m road
    xs = np.linspace(0, 400, 401)
    centre = np.column_stack([xs, 6 * np.sin(xs / 45.0)])
    ll = [(y / 1e5, x / 1e5) for x, y in centre[::10]]
    cl = build_centerline(ll, proj, elevation.FlatDEM(), False, corner_smoothing=0)
    dense = np.column_stack([np.interp(np.linspace(0, cl.length, 4000), cl.s, cl.p[:, 0]),
                             np.interp(np.linspace(0, cl.length, 4000), cl.s, cl.p[:, 1])])
    hl = np.full(len(cl.p), 3.6)
    hr = hl.copy()

    # 1. white edge lines 25 cm inside, yellow dashed centre line (3 m every 12 m), a tree shadow on it
    pm = road_photo(dense, [(3.35, "white", 0, 0), (-3.35, "white", 0, 0), (0.0, "yellow", 3.0, 12.0)],
                    shadow=(200, float(np.interp(200, cl.p[:, 0], cl.p[:, 1])), 3.0))
    mk = markings.detect(pm, cl, hl, hr)
    print(mk.info)
    by = {ln.side: ln for ln in mk.lines}
    assert set(by) == {"left", "right", "centre"}, list(by)
    for side in ("left", "right"):
        ln = by[side]
        assert ln.colour == "white" and ln.style == "solid", (side, ln.colour, ln.style)
        painted = sum(b - a for a, b in ln.pieces)
        assert painted > 0.9 * cl.length, (side, painted)
        inset = np.median(3.6 - np.abs(ln.offset))
        assert abs(inset - 0.25) < 0.12, (side, inset)
    c = by["centre"]
    assert c.colour == "yellow" and c.style == "dashed", (c.colour, c.style)
    lens = np.array([b - a for a, b in c.pieces])
    starts = np.array([a for a, _ in c.pieces])
    print(f"centre dashes: {len(lens)}, length {np.median(lens):.1f} m, period {np.median(np.diff(starts)):.1f} m")
    assert abs(np.median(lens) - 3.0) < 0.8 and abs(np.median(np.diff(starts)) - 12.0) < 0.8
    # the dashes are where they are painted (phase): dash starts at multiples of 12 m along the road
    ph = (starts % 12.0)
    ph = np.minimum(ph, 12.0 - ph)
    assert np.median(ph) < 1.0, ph
    # under the shadow (s ~ 200 m) the dashes carry on
    assert any(190 < a < 215 for a in starts), "dashes stop under the tree shadow"

    # 2. a road with no paint -> no lines (as in real life), but the photo was readable
    pm2 = road_photo(dense, [], seed=1)
    mk2 = markings.detect(pm2, cl, hl, hr)
    print(mk2.info)
    assert mk2.info["usable"] and mk2.lines == [], mk2.lines

    # 3. meshes: one white and one yellow mesh, strips just above the road
    parts = markings.meshes(mk, cl, hl, hr, camber=0.05)
    names = {p.material for p in parts}
    assert names == {"road_paint", "road_paint_yellow"}, names
    assert all(p.vertices[:, 2].max() < 0.1 for p in parts)
    for p in parts:                                   # faces up (seen from above, not culled)
        v = p.vertices[p.faces]
        nz = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])[:, 2]
        assert (nz > 0).all(), "markings face down"
    std = markings.standard(cl, hl, hr)
    assert len(std.lines) == 3 and len(markings.meshes(std, cl, hl, hr, 0.05)) == 1
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

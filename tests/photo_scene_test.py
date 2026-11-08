"""Offline end-to-end test of photo reading with a synthetic aerial photo (no network).

A fake WMS paints a known scene: the track at 7 m wide on one half and 11 m on the other,
tree crowns, a forest patch, driveways and roofs. The test checks the generator finds them.   python tests/photo_scene_test.py
"""
import io
import os
import sys

os.environ["MAPTOTRACK_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from app import imagery, osm, pipeline, routing
from app.geo import LocalProjection
from app.track import resample

WPS = [[48.49847, -71.25462], [48.49794, -71.25224], [48.49647, -71.25231], [48.49690, -71.25560]]
r = routing.route(WPS, ["free"] * 4, loop=True)
coords = np.array(r["coords"])
proj = LocalProjection((coords[:, 0].min() + coords[:, 0].max()) / 2, (coords[:, 1].min() + coords[:, 1].max()) / 2)
route_xy = resample(proj.latlngs_to_xy(coords), 0.5, True)
s_route = np.arange(len(route_xy)) * 0.5
total = s_route[-1]
tree_route = cKDTree(route_xy)
rng = np.random.default_rng(0)

# scene objects in local metres (centre of the loop is around 0,0)
c = route_xy.mean(0)
crowns = [(c[0] + rng.uniform(-60, 60), c[1] + rng.uniform(-40, 40), rng.uniform(2.0, 3.5)) for _ in range(25)]
forest = (c[0] - 260, c[1] - 120, 120, 90)                 # x, y, w, h
pool_round = (c[0] + 30, c[1] + 55, 2.6)
pool_rect = (c[0] - 40, c[1] - 55, 8.0, 4.0)
houses = [(c[0] + dx, c[1] + dy, 12, 9) for dx, dy in [(-20, 20), (25, 15), (-30, -25), (15, -30)]]
drives = [(hx + 12, hy - 1, 12, 5) for hx, hy, _, _ in houses]   # 12 m x 5 m next to each house


def paint(X, Y):
    img = np.zeros(X.shape + (3,), np.float32)
    img[:] = (100, 140, 62)                                      # lawn
    img += rng.normal(0, 6, img.shape)
    d, idx = tree_route.query(np.column_stack([X.ravel(), Y.ravel()]))
    d = d.reshape(X.shape)
    half = np.where(s_route[idx].reshape(X.shape) < total / 2, 3.5, 5.5)
    img[d < half] = (86, 86, 90)
    img[(d > half - 0.35) & (d < half - 0.15)] = (230, 230, 225)
    fx, fy, fw, fh = forest
    img[(X > fx) & (X < fx + fw) & (Y > fy) & (Y < fy + fh)] = (38, 66, 34)
    for tx, ty, tr in crowns:
        m = (X - tx) ** 2 + (Y - ty) ** 2 < tr * tr
        img[m] = (40, 72, 36)
    for hx, hy, w, h in houses:
        img[(np.abs(X - hx) < w / 2) & (np.abs(Y - hy) < h / 2)] = (120, 70, 60)
    for x, y, w, h in drives:
        img[(np.abs(X - x) < w / 2) & (np.abs(Y - y) < h / 2)] = (125, 125, 128)
    px, py, pr = pool_round
    img[(X - px) ** 2 + (Y - py) ** 2 < pr * pr] = (70, 175, 215)
    x, y, w, h = pool_rect
    img[(np.abs(X - x) < w / 2) & (np.abs(Y - y) < h / 2)] = (60, 165, 210)
    return np.clip(img, 0, 255).astype(np.uint8)


def fake_get(url, params=None):
    W, H = int(params["WIDTH"]), int(params["HEIGHT"])
    mx0, my0, mx1, my1 = [float(v) for v in params["BBOX"].split(",")]
    mx = mx0 + (np.arange(W) + 0.5) / W * (mx1 - mx0)
    my = my1 - (np.arange(H) + 0.5) / H * (my1 - my0)
    MX, MY = np.meshgrid(mx, my)
    lon = np.degrees(MX / imagery.R_EARTH)
    lat = np.degrees(2 * np.arctan(np.exp(MY / imagery.R_EARTH)) - np.pi / 2)
    X, Y = proj.to_xy(lat, lon)
    b = io.BytesIO()
    Image.fromarray(paint(X, Y)).save(b, "PNG")
    return b.getvalue()


def fake_osm(bbox, log=print):
    blds = []
    for i, (hx, hy, w, h) in enumerate(houses):
        ring = [(hx - w / 2, hy - h / 2), (hx + w / 2, hy - h / 2), (hx + w / 2, hy + h / 2), (hx - w / 2, hy + h / 2),
                (hx - w / 2, hy - h / 2)]
        la, lo = proj.to_latlon([p[0] for p in ring], [p[1] for p in ring])
        blds.append({"id": f"w{i}", "outer": [list(zip(la.tolist(), lo.tolist()))], "inner": [],
                     "tags": {"building": "house"}, "height": 7.0})
    return {"buildings": blds, "roads": [], "trees": [], "areas": [], "barriers": []}


imagery._get = fake_get
osm.fetch = fake_osm

rep = pipeline.generate({"name": "Photo scene test", "coords": r["coords"], "loop": True, "margin": 320,
                         "width": 9.0, "corner_smoothing": 6, "walls": "none", "photo_paved": True}, log=print)
prof = rep["profile"]
w = np.array(prof["w"])
s = np.array(prof["s"])
first = w[(s > total * 0.1) & (s < total * 0.4)]
second = w[(s > total * 0.6) & (s < total * 0.9)]
print("\n=== RESULTS ===")
print("width info:", rep["width"])
print(f"width first half median {np.median(first):.1f} m (truth 7), second half {np.median(second):.1f} m (truth 11)")
print("photo:", rep["photo"])
print("trees:", rep["trees"], rep["tree_sources"])
print("warnings:", rep["warnings"])
print("meshes:", sorted({m['name'].rsplit('_', 1)[0] for m in rep['meshes']}))
assert abs(np.median(first) - 7) < 0.5 and abs(np.median(second) - 11) < 0.5, "road width"
assert rep["photo"]["paved_areas"] >= 3, "driveways"
assert rep["tree_sources"].get("photo", 0) >= 20, "photo trees"
# road markings: the scene paints white edge lines 15-35 cm inside both edges, nothing in the middle
mk = rep["markings"]["info"]
print("markings:", [(l["side"], l["colour"], l["style"], l["painted_m"]) for l in mk["lines"]])
assert mk["source"] == "photo" and all(l["colour"] == "white" and l["side"] in ("left", "right") for l in mk["lines"])
for side in ("left", "right"):
    painted = sum(l["painted_m"] for l in mk["lines"] if l["side"] == side)
    assert painted > 0.8 * total, (side, painted, total)
# manual width override: 9.0 m around the first waypoint
rep2 = pipeline.generate({"name": "Photo scene test", "coords": r["coords"], "loop": True, "margin": 320,
                          "width": 9.0, "corner_smoothing": 6, "walls": "none", "trees": False, "photo_paved": False,
                          "width_overrides": [{"lat": WPS[0][0], "lon": WPS[0][1], "width": 9.0, "span": 30}]},
                         log=lambda m: None)
w0 = rep2["profile"]["w"][0]
print(f"override at start: {w0:.2f} m (asked 9.0)")
assert abs(w0 - 9.0) < 0.05, "override"
print("ALL CHECKS PASSED")

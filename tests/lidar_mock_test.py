"""Offline test of the LiDAR path with synthetic GeoTIFFs (Quebec Lambert, EPSG:32198).

MNT = a 3% slope rising to the east; MHC = 12 cone-shaped trees (15 m) and 6 bushes (1.5 m).
Run:  python tests/lidar_mock_test.py
"""
import os
import sys
import tempfile

os.environ["MAPTOTRACK_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from app import lidar, osm, pipeline, routing

WPS = [[48.49847, -71.25462], [48.49794, -71.25224], [48.49647, -71.25231], [48.49690, -71.25560]]
r = routing.route(WPS, ["free"] * 4, loop=True)
lat0, lon0 = 48.4975, -71.2535
tr = Transformer.from_crs("EPSG:4326", "EPSG:32198", always_xy=True)
cx, cy = tr.transform(lon0, lat0)
size, res = 1400, 1.0
west, north = cx - size / 2, cy + size / 2
X, Y = np.meshgrid(west + (np.arange(size) + 0.5) * res, north - (np.arange(size) + 0.5) * res)
mnt = (150 + 0.03 * (X - cx)).astype(np.float32)
mhc = np.zeros_like(mnt)
rng = np.random.default_rng(1)
truth = []
for i in range(18):
    ang = rng.uniform(0, 2 * np.pi)
    rad = rng.uniform(220, 380)       # outside the loop, away from the road
    tx, ty = cx + np.cos(ang) * rad, cy + np.sin(ang) * rad
    h = 15.0 if i < 12 else 1.5
    rr = 3.0 if i < 12 else 1.0
    d = np.hypot(X - tx, Y - ty)
    mhc = np.maximum(mhc, np.where(d < rr, h * (1 - d / rr * 0.6), 0))
    truth.append((tx, ty, h))
tmp = tempfile.mkdtemp()
paths = {}
for name, arr in (("MNT", mnt), ("MHC", mhc)):
    p = os.path.join(tmp, f"{name}.tif")
    with rasterio.open(p, "w", driver="GTiff", width=size, height=size, count=1, dtype="float32",
                       crs="EPSG:32198", transform=from_origin(west, north, res, res), nodata=-9999) as ds:
        ds.write(arr, 1)
    paths[name] = p

lidar.sheets_for = lambda bbox, log=print: [{"Feuillet20K": "TEST", "MNT": "x", "MHC": "y"}]
lidar.product_files = lambda sheet, product, log=print: [paths[product]]
osm.fetch = lambda bbox, log=print: {"buildings": [], "roads": [], "trees": [], "areas": [], "barriers": []}

rep = pipeline.generate({"name": "Lidar test", "coords": r["coords"], "loop": True, "margin": 350, "lidar": True,
                         "ground": "landcover", "width_mode": "fixed", "elev_smoothing": 25}, log=print)
print("\n=== RESULTS ===")
print("lidar:", rep["lidar"], "trees:", rep["trees"], rep["tree_sources"], "warnings:", rep["warnings"])
print("elevation:", rep["elevation"])
# expected road heights: plane 150 + 0.03 * dx
lons = np.array(r["coords"])[:, 1]
lats = np.array(r["coords"])[:, 0]
ex, ey = tr.transform(lons, lats)
exp = 150 + 0.03 * (np.asarray(ex) - cx)
print(f"expected road height {exp.min():.1f}-{exp.max():.1f} m")
assert rep["lidar"], "lidar used"
assert abs(rep["elevation"]["min"] - exp.min()) < 1.5 and abs(rep["elevation"]["max"] - exp.max()) < 1.5, "heights"
assert 15 <= rep["tree_sources"].get("lidar", 0) <= 24, "tree tops"
print("ALL CHECKS PASSED")

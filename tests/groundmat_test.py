"""Ground photo materials: surface classes, baking, ksMultilayer materials in the KN5, square tiles."""
import os
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image

os.environ.setdefault("MAPTOTRACK_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import config, ground, groundmat, kn5, materials, pipeline, routing  # noqa: E402

# fake ambientCG downloads in a private cache (never the app's own cache)
config.CACHE_DIR = pathlib.Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
for key, name in materials.SOURCES.items():
    p = materials._path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(60, 200, (256, 256, 3), dtype=np.uint8)).save(p)
materials._LOADED.clear()

# surface classes from colours
cols = np.array([[[74, 74, 77], [84, 132, 46], [138, 130, 112], [184, 168, 130], [124, 102, 78], [50, 72, 36]]], np.float32)
w = groundmat.surface_weights(np.repeat(np.repeat(cols, 5, 0), 5, 1))[2, 2::5]
names = ["asphalt", "grass", "gravel", "dirt"]
got = [names[i] for i in w.argmax(-1)]
print("classes:", got)
assert got == ["asphalt", "grass", "gravel", "dirt", "dirt", "grass"], got
assert np.allclose(w.sum(-1), 1, atol=1e-4)

# square tiles
for b in [(0, 0, 700, 520), (-300, -300, 1500, 400), (0, 0, 100, 100)]:
    ts = ground.ground_tiles(b)
    assert all(abs((t["x1"] - t["x0"]) - (t["y1"] - t["y0"])) < 1e-6 for t in ts)
    assert max(t["x1"] for t in ts) >= b[2] - 1e-6 and max(t["y1"] for t in ts) >= b[3] - 1e-6
    assert all(t["size"] <= 512 + 1e-6 for t in ts)

# whole generation (land-cover ground): baked tiles + ksMultilayer
wps = [[45.5017, -73.5673], [45.5040, -73.5630], [45.5075, -73.5655], [45.5068, -73.5712]]
route = routing.route(wps, ["free"] * 4, loop=True)
rep = pipeline.generate({"name": "groundmat test", "coords": route["coords"], "loop": True, "ml_ground": True},
                        log=lambda *a: None)
assert "Textures: ambientCG (CC0)" in rep["credits"], rep["credits"]
r = kn5.read(rep["files"]["kn5"])
tex = {t[0] for t in r["textures"]}
ml = [m for m in r["materials"] if m["name"].startswith("ground_")]
assert ml and all(m["shader"] == "ksMultilayer" for m in ml), [m["shader"] for m in ml]
for m in ml:
    assert set(m["textures"]) == {"txDiffuse", "txMask", "txDetailR", "txDetailG", "txDetailB", "txDetailA"}
    assert all(t in tex for t in m["textures"].values())
    assert m["textures"]["txDetailG"] == "gdetail_grass.dds" and m["textures"]["txMask"].endswith(".png")
    pv = {k: (v[0] if isinstance(v, tuple) else v) for k, v in m["props"].items()}
    assert 100 < pv["multG"] < 300 and pv["magicMult"] == 2.0, pv
print("ground materials:", len(ml), "ksMultilayer;", pv)
mats = {m["name"]: m for m in r["materials"]}
assert "asphalt_tiled" in mats and "gravel_tiled" in mats, "tiled photo asphalt / gravel"
names = [n["name"] for n in r["nodes"] if n["class"] == 2]
assert any(n.startswith("1ROAD_ASPH") for n in names) and any(n.startswith("1GRASS_SHLD") for n in names)
assert any(n.startswith("1GRASS_NEAR") for n in names), "sharp ground beside the track"
# switched off -> plain ksPerPixel
rep2 = pipeline.generate({"name": "groundmat test2", "coords": route["coords"], "loop": True, "ml_ground": False},
                         log=lambda *a: None)
r2 = kn5.read(rep2["files"]["kn5"])
assert all(m["shader"] == "ksPerPixel" for m in r2["materials"] if m["name"].startswith("ground_"))
assert not any(t[0].startswith(("gmask_", "gdetail_")) for t in r2["textures"])
# every ground tile mesh stays on its own tile (texture coordinates 0..1: nothing beyond the map area)
import trimesh  # noqa: E402
sc = trimesh.load(rep["files"]["folder"] + "/blender/track.glb")
for n, g in sc.geometry.items():
    if n.startswith(("1GRASS_GND", "1ROAD_GND")):
        uv = g.visual.uv
        assert uv.min() > -0.02 and uv.max() < 1.02, (n, uv.min(0), uv.max(0))
print("ALL CHECKS PASSED")

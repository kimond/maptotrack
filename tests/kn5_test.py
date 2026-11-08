"""KN5 writer: byte layout round trip + the generated (offline demo) track's .kn5 is complete."""
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("MAPTOTRACK_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import kn5, pipeline, routing  # noqa: E402

# ---- 1. tiny scene round trip
kb = kn5.Kn5Builder()
kb.add_texture("t.png", b"\x89PNG fake")
kb.material("asphalt", texture="t.png")
kb.material("trees", texture="t.png")
V = np.array([[0, 0, 0], [10, 0, 0], [10, 0, -10], [0, 0, -10]], float)     # AC: y up, z = -north
F = np.array([[0, 1, 2], [0, 2, 3]])
UV = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)
kb.add_mesh("1ROAD_TEST", "asphalt", V, F, uv=UV)
kb.add_dummy("AC_START_0", kn5.dummy_matrix([1, 2, 3], [0, 0, -1]))
tmp = os.path.join(tempfile.mkdtemp(), "t.kn5")
kb.write(tmp)
r = kn5.read(tmp, with_data=True)
assert r["version"] == 5 and r["textures"][0][0] == "t.png"
mats = {m["name"]: m for m in r["materials"]}
assert mats["asphalt"]["shader"] == "ksPerPixel" and mats["asphalt"]["textures"]["txDiffuse"] == "t.png"
assert mats["trees"]["shader"] == "ksPerPixelAT_NS" and mats["trees"]["alpha_tested"]
mesh = [n for n in r["nodes"] if n["class"] == 2][0]
assert np.allclose(mesh["verts"]["pos"], V) and np.allclose(mesh["verts"]["uv"], UV)
assert np.allclose(mesh["verts"]["nrm"], [0, 1, 0]), "CCW from above -> normal +Y"
assert list(mesh["idx"]) == [0, 1, 2, 0, 2, 3]
d = [n for n in r["nodes"] if n["name"] == "AC_START_0"][0]["matrix"]
assert np.allclose(d[3, :3], [1, 2, 3]) and np.allclose(d[1, :3], [0, 1, 0]) and np.allclose(d[2, :3], [0, 0, -1])
assert np.allclose(np.cross(d[1, :3], d[2, :3]), d[0, :3]), "right-handed axes: X = Y x Z"

# ---- 2. splitting above 65535 vertices
n = 70000
P = np.random.rand(n, 3)
Fb = np.arange(n - n % 3).reshape(-1, 3)
kb2 = kn5.Kn5Builder()
kb2.add_mesh("1GRASS_BIG", "ground", P, Fb)
assert len(kb2.meshes) == 2 and all(len(m["pos"]) <= kn5.MAX_VERTS for m in kb2.meshes)
assert all(m["name"].startswith("1GRASS") for m in kb2.meshes)

# ---- 3. the generated track
wps = [[45.5017, -73.5673], [45.5040, -73.5630], [45.5075, -73.5655], [45.5068, -73.5712]]
route = routing.route(wps, ["free"] * 4, loop=True)
rep = pipeline.generate({"name": "kn5 test", "coords": route["coords"], "loop": True}, log=lambda *a: None)
assert rep["kn5"], rep["warnings"]
path = rep["files"]["kn5"]
r = kn5.read(path)
names = [nd["name"] for nd in r["nodes"]]
assert any(x.startswith("1ROAD") for x in names) and any(x.startswith("1GRASS") for x in names)
assert "AC_START_0" in names and "AC_PIT_0" in names and "AC_HOTLAP_START_0" in names
tex = {t[0] for t in r["textures"]}
for m in r["materials"]:
    for t in m["textures"].values():
        assert t in tex, f"{m['name']} uses missing texture {t}"
assert all(nd["nverts"] <= kn5.MAX_VERTS for nd in r["nodes"] if nd["class"] == 2)
surf = os.path.join(os.path.dirname(path), "data", "surfaces.ini")
assert "KEY=WALL" not in open(surf).read(), "WALL must not be a surface (walls would become drivable)"
assert any(t[2] == b"DDS " for t in r["textures"]), "opaque textures should be DDS"
from app.ac_export import dds_bc1  # noqa: E402
from PIL import Image  # noqa: E402
import struct  # noqa: E402
d = dds_bc1(Image.new("RGB", (1000, 500), (60, 120, 40)))
_, _, _, Hd, Wd, _, _, mips = struct.unpack_from("<4sIIIIIII", d, 0)
assert (Wd, Hd) == (1024, 512) and mips == 8, (Wd, Hd, mips)
assert os.path.exists(surf) and "KEY=ROAD" in open(surf).read()
print(f"kn5: {len(names)} nodes, {len(r['materials'])} materials, {len(tex)} textures, "
      f"{os.path.getsize(path) / 1e6:.1f} MB")
print("ALL CHECKS PASSED")

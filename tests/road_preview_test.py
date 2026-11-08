"""Road preview: the road alone, fast, and the same road as a full generation."""
import os
import sys
import time

import numpy as np

os.environ.setdefault("MAPTOTRACK_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import pipeline, routing  # noqa: E402

wps = [[45.5017, -73.5673], [45.5040, -73.5630], [45.5075, -73.5655], [45.5068, -73.5712]]
route = routing.route(wps, ["free"] * 4, loop=True)
params = {"name": "preview test", "coords": route["coords"], "loop": True, "width_mode": "fixed", "width": 9.0,
          "shoulder": 1.5, "walls": "concrete",
          "width_overrides": [{"lat": route["coords"][len(route["coords"]) // 2][0], "lon": route["coords"][len(route["coords"]) // 2][1], "width": 12.0, "span": 40}]}
t = time.time()
pv = pipeline.road_preview(params, log=lambda *a: None)
dt = time.time() - t
rep = pipeline.generate(params, log=lambda *a: None)
print(f"preview {dt:.2f} s vs generation {rep['elapsed_s']} s")
assert dt < 5.0
for k in ("built_line", "built_s", "built_w", "edges", "length_m", "min_radius_m"):
    assert np.allclose(np.asarray(pv[k], float), np.asarray(rep[k], float)), k
assert [o["name"] for o in pv["objects"]] == [o["name"] for o in rep["objects"]]
assert np.allclose([o["pos"] for o in pv["objects"]], [o["pos"] for o in rep["objects"]])
assert max(pv["built_w"]) > 11.5 and min(pv["built_w"]) < 9.5, "manual width applied"
assert len(pv["shoulder_lines"]) == 2 and len(pv["walls_lines"]) == 2
assert pv["preview"] and pv["profile"]["s"]
print("ALL CHECKS PASSED")

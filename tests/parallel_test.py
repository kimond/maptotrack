"""Parallel generation: downloads overlap, and the result is identical with 1 or several workers."""
import os
import sys
import time

os.environ["MAPTOTRACK_OFFLINE"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import elevation, kn5, parallel, pipeline, routing  # noqa: E402

wps = [[45.5017, -73.5673], [45.5040, -73.5630], [45.5075, -73.5655], [45.5068, -73.5712]]
route = routing.route(wps, ["free"] * 4, loop=True)
params = {"name": "parallel test", "coords": route["coords"], "loop": True, "trees": True, "grass": True}

# 1. slow "downloads" must overlap: elevation and OSM each take 2 s
real_dem, real_scenery = elevation.make_dem, pipeline.load_scenery


def slow_dem(*a, **k):
    time.sleep(2.0)
    return real_dem(*a, **k)


def slow_scenery(*a, **k):
    time.sleep(2.0)
    return real_scenery(*a, **k)


elevation.make_dem, pipeline.load_scenery = slow_dem, slow_scenery
os.environ["MAPTOTRACK_WORKERS"] = "1"
t = time.time()
pipeline.generate(dict(params), log=lambda *a: None)
t_seq = time.time() - t
os.environ["MAPTOTRACK_WORKERS"] = "4"
t = time.time()
rep = pipeline.generate(dict(params), log=lambda *a: None)
t_par = time.time() - t
elevation.make_dem, pipeline.load_scenery = real_dem, real_scenery
print(f"with 2 s + 2 s of downloads: sequential {t_seq:.1f} s, parallel {t_par:.1f} s")
assert t_seq - t_par > 1.5, "downloads did not overlap"

# 2. same track with 1 and 4 workers
def summary(rep):
    r = kn5.read(rep["files"]["kn5"], with_data=True)
    return {n["name"]: (n["verts"].tobytes(), n["idx"].tobytes()) for n in r["nodes"] if n["class"] == 2}


os.environ["MAPTOTRACK_WORKERS"] = "1"
a = summary(pipeline.generate(dict(params), log=lambda *a: None))
os.environ["MAPTOTRACK_WORKERS"] = "4"
b = summary(pipeline.generate(dict(params), log=lambda *a: None))
assert a.keys() == b.keys() and all(a[k] == b[k] for k in a), "parallel result differs"
print(f"identical geometry with 1 and 4 workers ({len(a)} meshes); this machine has {os.cpu_count()} cores")
print("ALL CHECKS PASSED")

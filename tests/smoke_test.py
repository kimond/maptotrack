import os, sys, json
os.environ["MAPTOTRACK_OFFLINE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import routing, pipeline
wps = [[45.5017,-73.5673],[45.5040,-73.5630],[45.5075,-73.5655],[45.5068,-73.5712],[45.5035,-73.5725]]
loop = len(sys.argv) < 2 or sys.argv[1] != "open"
r = routing.route(wps, ["free"]*5, loop=loop)
print("route pts", len(r["coords"]), round(r["length_m"]))
rep = pipeline.generate({"name": "Smoke " + ("loop" if loop else "open"), "coords": r["coords"], "loop": loop}, log=print)
print(json.dumps({k: rep[k] for k in ("length_m","min_radius_m","elevation","buildings","streets","total_verts","warnings")}, indent=1))
print(max(m["verts"] for m in rep["meshes"]), len(rep["meshes"]))

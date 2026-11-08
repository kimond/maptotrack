"""Street furniture: junction rules, facing, clearances, and the offline demo track gets props in its kn5."""
import os
import sys

import numpy as np

os.environ.setdefault("MAPTOTRACK_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import props, kn5, pipeline, routing  # noqa: E402
from app.geo import LocalProjection  # noqa: E402

proj = LocalProjection(48.4, -71.1)


def ll(pts):
    lat, lon = proj.to_latlon(np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float))
    return list(zip(lat.tolist(), lon.tolist()))


def road(i, pts, hw, width, speed=None, oneway=0):
    return {"id": f"w{i}", "coords": ll(pts), "highway": hw, "width": width, "maxspeed": speed, "oneway": oneway,
            "lit": None, "roundabout": False, "bridge": False, "tunnel": False}


# a secondary road west-east through (0,0) and (200,0); a residential street crossing it at (0,0) (4-way),
# another one ending on it at (200,0) (T); the residential limit 30 vs 50 on the main road;
# a signal mapped near a third junction at (400,0) between two secondaries.
R = [road(1, [(-300, 0), (0, 0)], "secondary", 10, 50), road(2, [(0, 0), (200, 0)], "secondary", 10, 50),
     road(3, [(0, -200), (0, 0)], "residential", 7, 30), road(4, [(0, 0), (0, 200)], "residential", 7, 30),
     road(5, [(200, 0), (200, 150)], "residential", 7, 30), road(6, [(200, 0), (400, 0)], "secondary", 10, 50),
     road(7, [(400, 0), (700, 0)], "secondary", 10, 50), road(8, [(400, -200), (400, 0)], "secondary", 10, 50),
     road(9, [(400, 0), (400, 200)], "tertiary", 8, 50)]
sig_ll = ll([(408, 6)])
furn = {"signals": [{"lat": sig_ll[0][0], "lon": sig_ll[0][1]}], "stops": [], "giveways": [], "lamps": [],
        "towers": [], "lines": [{"id": "pl", "kind": "line", "voltage": 315000.0, "cables": 6,
                                 "coords": ll([(-280, 300), (-280, -10), (-280, -300)])}]}


class TM:            # a terrain model stand-in: flat ground, the 'track' far away at y = 900
    def __init__(self):
        from scipy.spatial import cKDTree
        self.tree = cKDTree(np.column_stack([np.linspace(-500, 900, 300), np.full(300, 900.0)]))
        self.hwc = np.full(300, 5.0)

    def road_info3(self, x, y):
        d, i = self.tree.query(np.column_stack([np.ravel(x), np.ravel(y)]))
        return d, np.zeros_like(d), self.hwc[i]

    def surface_height(self, x, y):
        return np.zeros(np.shape(x))


from shapely.geometry import LineString  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
streets = unary_union([LineString(proj.latlngs_to_xy(r["coords"])).buffer(r["width"] / 2, cap_style=2) for r in R])
sc = props.Scene(TM(), 1.5, streets, None)
parts, tex, st = props.build(R, furn, proj, sc, (-400, -400, 900, 400),
                             {"power_lines": True, "road_signs": True, "street_lights": True}, "qc", log=print)
items = sc.items
print(st)
stops = [i for i in items if i[0] == "stop"]
sigs = [i for i in items if i[0] == "signal"]
speeds = [i for i in items if i[0].startswith("speed")]
# 4-way at (0,0): the two residential arms get a stop, the secondary ones none; T at (200,0): its stem
assert len([s for s in stops if abs(s[1]) < 20 and abs(s[2]) < 20]) == 2, stops
assert len([s for s in stops if abs(s[1] - 200) < 20]) == 1, stops
# every stop sign faces away from its junction (towards the drivers coming in) and stands beside the road
for k, x, y, fx, fy in stops:
    c = np.array([0, 0]) if abs(x) < 50 else np.array([200, 0])
    assert np.dot([fx, fy], np.array([x, y]) - c) > 0, (x, y, fx, fy)
    assert not streets.contains(__import__("shapely.geometry", fromlist=["Point"]).Point(x, y))
# drivers' right: an approach coming north (from y<0) has its sign east of the street (x > 0)
s_south = [s for s in stops if abs(s[1]) < 20 and s[2] < 0][0]
assert s_south[1] > 0, s_south
# the signalled junction at (400,0): one head per arm (4)
assert len([s for s in sigs if abs(s[1] - 400) < 25]) == 4, sigs
# speed signs: entering the 30 km/h streets from the 50 road, and the 50 road from the 30 streets
assert any(s[0] == "speed30" for s in speeds) and any(s[0] == "speed50" for s in speeds), speeds
assert st["pylons"] == 3 and st["utility_poles"] > 5 and st["street_lights"] > 2, st
names = {p.name.rsplit("_", 1)[0] for p in parts}
assert {"1WALL_PROPS", "PROPS", "PYLONS", "1WALL_PYLON", "WIRES_HV", "WIRES"} <= names, names
assert tex["props"].mode == "RGBA" and tex["props"].size == (props.COLS * props.CELL, props.ROWS * props.CELL)

# Québec detection
assert props.sign_style(48.42, -71.07) == "qc" and props.sign_style(45.42, -75.69) == "ca"
assert props.sign_style(45.50, -73.57) == "qc" and props.sign_style(40.7, -74.0) == "us"
assert props.sign_style(48.85, 2.35) == "eu" and props.sign_style(43.65, -79.38) == "ca"

# ---- crossings, guard rails, substations
cw_ll = ll([(100, 3), (102, -3), (0, 120)])          # two nodes of one zebra on the secondary, lines on a street
furn2 = {"crossings": [{"lat": cw_ll[0][0], "lon": cw_ll[0][1], "style": "zebra"},
                       {"lat": cw_ll[1][0], "lon": cw_ll[1][1], "style": "zebra"},
                       {"lat": cw_ll[2][0], "lon": cw_ll[2][1], "style": "signals"}],
         "guardrails": [{"id": "g", "coords": ll([(500, 8), (560, 8), (620, 9)])},
                        {"id": "g2", "coords": ll([(0, 150), (0, 170)])}],          # on a street: left out
         "substations": [{"id": "s", "voltage": 120000.0, "kind": "transmission",
                          "coords": ll([(-200, 60), (-140, 60), (-140, 110), (-200, 110), (-200, 60)])}],
         "transformers": [{"lat": ll([(-170, 85)])[0][0], "lon": ll([(-170, 85)])[0][1]}]}
sc2 = props.Scene(TM(), 1.5, streets, None)
parts2, tex2, st2 = props.build(R, furn2, proj, sc2, (-400, -400, 900, 400),
                                {"power_lines": False, "road_signs": False, "street_lights": False,
                                 "crosswalks": True, "guardrails": True, "substations": True}, "qc", log=print)
assert st2["crosswalks"] == 2, st2
assert st2["substations"] == 1 and st2["transformers"] == 1, st2
assert 110 < st2["guard_rails_m"] < 130, st2
paint = [p for p in parts2 if p.material == "road_paint"]
assert paint and paint[0].name.startswith("ROAD_CROSSWALKS"), [p.name for p in parts2]
V = np.vstack([p.vertices for p in paint])
zebra = V[np.abs(V[:, 0] - 100) < 5]
assert np.allclose(zebra[:, 2], props.PAINT_LIFT) and np.ptp(zebra[:, 1]) > 8.5, np.ptp(zebra[:, 1])  # 10 m road
assert np.ptp(zebra[:, 0]) < props.CROSS_LEN + 0.1
lines = V[np.abs(V[:, 1] - 120) < 5]                 # 'signals' in Québec: two lines across the 7 m street
assert len(lines) and np.ptp(lines[:, 0]) <= 7.01 and np.ptp(lines[:, 1]) > 2.9
solid2 = np.vstack([p.vertices for p in parts2 if p.name.startswith("1WALL")])
gr = solid2[(solid2[:, 0] > 490) & (solid2[:, 0] < 630)]
assert gr[:, 2].max() < 0.9 and gr[:, 1].min() > 5.5, (gr[:, 2].max(), gr[:, 1].min())     # beside the road
fence = solid2[(solid2[:, 0] < -130) & (solid2[:, 0] > -210)]
assert 2.3 < fence[:, 2].max() < 12 and len(fence) > 100
# nothing else stands inside the substation now
assert not sc2.ok(-170, 85, spacing=0.0)
atlas_img = np.asarray(tex2["props"])
ch = atlas_img[(props.CHAIN // props.COLS) * props.CELL:(props.CHAIN // props.COLS + 1) * props.CELL,
               (props.CHAIN % props.COLS) * props.CELL:(props.CHAIN % props.COLS + 1) * props.CELL, 3]
assert 0.15 < (ch > 0).mean() < 0.5, (ch > 0).mean()   # a see-through mesh
print("crossings / guard rails / substations OK:", st2)

# ---- the offline demo track: props in the kn5, nothing solid on the track
wps = [[45.5017, -73.5673], [45.5040, -73.5630], [45.5075, -73.5655], [45.5068, -73.5712]]
route = routing.route(wps, ["free"] * 4, loop=True)
rep = pipeline.generate({"name": "props test", "coords": route["coords"], "loop": True}, log=lambda *a: None)
print("report props:", rep["props"], rep["warnings"])
assert rep["props"].get("pylons", 0) > 0 and rep["props"].get("utility_poles", 0) > 0, rep["props"]
r = kn5.read(rep["files"]["kn5"], with_data=True)
mats = {m["name"]: m for m in r["materials"]}
assert mats["props"]["shader"] == "ksPerPixelAT"
solid = np.vstack([n["verts"]["pos"] for n in r["nodes"] if n["class"] == 2 and n["name"].startswith("1WALL_P")])
from scipy.spatial import cKDTree  # noqa: E402
from app.geo import LocalProjection as LP  # noqa: E402
pr = LP(rep["origin"]["lat"], rep["origin"]["lon"])
line = pr.latlngs_to_xy(rep["built_line"])
half = np.interp(np.linspace(0, 1, len(line)), np.linspace(0, 1, len(rep["built_w"])), np.asarray(rep["built_w"])) / 2
d, i = cKDTree(line).query(np.column_stack([solid[:, 0], -solid[:, 2]]))      # AC (x, y up, -north)
gap = (d - half[i]).min()
print(f"closest solid prop to the track edge: {gap:.2f} m")
assert gap > 0.5
print("ALL CHECKS PASSED")

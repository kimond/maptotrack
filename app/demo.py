"""Synthetic data for offline/demo mode (MAPTOTRACK_OFFLINE=1) and tests."""
from __future__ import annotations

import random

import numpy as np

from .geo import LocalProjection
from .osm import building_height


class SyntheticDEM:
    def __init__(self, bbox):
        s, w, n, e = bbox
        self.proj = LocalProjection((s + n) / 2, (w + e) / 2)

    def sample(self, lat, lon):
        x, y = self.proj.to_xy(lat, lon)
        return 120 + 18 * np.sin(x / 260.0) * np.cos(y / 310.0) + 6 * np.sin((x + y) / 90.0) + 0.6 * np.sin(x / 7.0)


def synthetic_osm(bbox):
    s, w, n, e = bbox
    proj = LocalProjection((s + n) / 2, (w + e) / 2)
    x0, y0 = proj.to_xy(s, w)
    x1, y1 = proj.to_xy(n, e)
    rnd = random.Random(42)
    buildings, roads, trees = [], [], []
    bid = 0
    step = 45.0
    for gx in np.arange(x0 + 20, x1 - 20, step):
        for gy in np.arange(y0 + 20, y1 - 20, step):
            if rnd.random() < 0.45:
                continue
            wdt, dep = rnd.uniform(10, 26), rnd.uniform(10, 22)
            cx, cy = gx + rnd.uniform(-6, 6), gy + rnd.uniform(-6, 6)
            ring = [(cx - wdt / 2, cy - dep / 2), (cx + wdt / 2, cy - dep / 2), (cx + wdt / 2, cy + dep / 2),
                    (cx - wdt / 2, cy + dep / 2), (cx - wdt / 2, cy - dep / 2)]
            if rnd.random() < 0.2:  # L-shaped
                ring = [(cx - wdt / 2, cy - dep / 2), (cx + wdt / 2, cy - dep / 2), (cx + wdt / 2, cy),
                        (cx, cy), (cx, cy + dep / 2), (cx - wdt / 2, cy + dep / 2), (cx - wdt / 2, cy - dep / 2)]
            lat, lon = proj.to_latlon([p[0] for p in ring], [p[1] for p in ring])
            bid += 1
            tags = {"building": rnd.choice(["house", "apartments", "commercial", "yes"])}
            if rnd.random() < 0.3:
                tags["building:levels"] = str(rnd.randint(2, 8))
            buildings.append({"id": f"w{bid}", "outer": [list(zip(lat.tolist(), lon.tolist()))], "inner": [],
                              "tags": tags, "height": building_height(tags, bid)})
    for gx in np.arange(x0 + 20 + step / 2, x1, step * 3):
        lat, lon = proj.to_latlon([gx, gx], [y0, y1])
        roads.append({"id": f"w9{len(roads)}", "coords": list(zip(lat.tolist(), lon.tolist())),
                      "highway": "residential", "width": 6.5, "layer": 0, "bridge": False, "tunnel": False})
    for _ in range(60):
        lat, lon = proj.to_latlon(rnd.uniform(x0, x1), rnd.uniform(y0, y1))
        trees.append({"id": f"n{len(trees)}", "lat": float(lat), "lon": float(lon)})
    areas = []
    for k in range(10):
        cx, cy = rnd.uniform(x0, x1), rnd.uniform(y0, y1)
        r = rnd.uniform(60, 220)
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        rad = r * (0.75 + 0.5 * np.array([rnd.random() for _ in ang]))
        lat, lon = proj.to_latlon(cx + rad * np.cos(ang), cy + rad * np.sin(ang))
        ring = list(zip(lat.tolist(), lon.tolist()))
        ring.append(ring[0])
        kind = ["wood", "wood", "wood", "scrub", "water", "meadow"][k % 6]
        areas.append({"id": f"a{k}", "kind": kind, "outer": [ring]})
    return {"buildings": buildings, "roads": roads, "trees": trees, "areas": areas, "barriers": []}


def synthetic_furniture(bbox):
    """Offline street furniture: a transmission line across the area, a low-voltage line, a few signals,
    stop signs and lamps on the synthetic streets."""
    s, w, n, e = bbox
    proj = LocalProjection((s + n) / 2, (w + e) / 2)
    x0, y0 = proj.to_xy(s, w)
    x1, y1 = proj.to_xy(n, e)

    def ll(x, y):
        lat, lon = proj.to_latlon(x, y)
        return float(lat), float(lon)

    k = np.linspace(0, 1, 9)
    hv = [ll(x0 + (x1 - x0) * t, y0 + (y1 - y0) * (0.15 + 0.6 * t)) for t in k]
    lv = [ll(x0 + (x1 - x0) * t, y0 + (y1 - y0) * 0.85) for t in np.linspace(0.1, 0.5, 12)]
    gx = x0 + 20 + 45.0 / 2                       # the first synthetic street (synthetic_osm)
    sig = [{"lat": ll(gx, y0 + (y1 - y0) * 0.4)[0], "lon": ll(gx, y0 + (y1 - y0) * 0.4)[1]}]
    lamps = [dict(zip(("lat", "lon"), ll(gx + 5.5, y0 + (y1 - y0) * t))) for t in (0.55, 0.6, 0.65)]
    cw = [{"lat": ll(gx, y0 + (y1 - y0) * 0.3)[0], "lon": ll(gx, y0 + (y1 - y0) * 0.3)[1], "style": "zebra"}]
    gr = [ll(gx - 9.0, y0 + (y1 - y0) * t) for t in (0.45, 0.5, 0.55)]
    sub = [ll(x1 - 60, y0 + 40), ll(x1 - 20, y0 + 40), ll(x1 - 20, y0 + 70), ll(x1 - 60, y0 + 70), ll(x1 - 60, y0 + 40)]
    return {"crossings": cw, "guardrails": [{"id": "w_gr", "coords": gr}],
            "substations": [{"id": "w_sub", "coords": sub, "voltage": 120000.0, "kind": "distribution"}],
            "transformers": [],
            "signals": sig, "stops": [], "giveways": [], "lamps": lamps,
            "towers": [{"lat": p[0], "lon": p[1], "kind": "tower"} for p in hv],
            "lines": [{"id": "w_hv", "coords": hv, "kind": "line", "voltage": 230000.0, "cables": 6},
                      {"id": "w_lv", "coords": lv, "kind": "minor_line", "voltage": 0.0, "cables": 3}]}

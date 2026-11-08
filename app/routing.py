"""Turn user waypoints into a continuous route.

Each segment between two consecutive waypoints is either
  * "road": snapped to real streets with OSRM, or
  * "free": a straight line (smoothed later by the corner-smoothing step).
"""
from __future__ import annotations

import math
from functools import lru_cache

import requests

from . import config


class RoutingError(Exception):
    pass


def _haversine(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


@lru_cache(maxsize=2048)
def _osrm(a_lat, a_lng, b_lat, b_lng, ignore_oneway: bool):
    base = config.OSRM_NO_ONEWAY_URL if ignore_oneway else config.OSRM_CAR_URL
    url = f"{base}/{a_lng:.7f},{a_lat:.7f};{b_lng:.7f},{b_lat:.7f}"
    params = {"overview": "full", "geometries": "geojson", "continue_straight": "true"}
    try:
        r = requests.get(url, params=params, headers={"User-Agent": config.USER_AGENT}, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # network, HTTP, JSON
        raise RoutingError(f"Routing service unavailable ({exc})") from exc
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RoutingError(f"No road route found between waypoints ({data.get('code')})")
    coords = data["routes"][0]["geometry"]["coordinates"]
    return tuple((c[1], c[0]) for c in coords)


def route(waypoints, modes, loop: bool, ignore_oneway: bool = False):
    """waypoints: [[lat,lng],...]; modes: per-segment 'road'|'free' (len = n-1, or n if loop).

    Returns dict with 'segments' (list of lists of [lat,lng]) and 'coords' (merged polyline).
    """
    pts = [tuple(map(float, p)) for p in waypoints]
    if len(pts) < 2:
        return {"segments": [], "coords": [list(p) for p in pts], "warnings": []}
    pairs = list(zip(pts[:-1], pts[1:]))
    if loop and len(pts) >= 3:
        pairs.append((pts[-1], pts[0]))
    segments, warnings = [], []
    for i, (a, b) in enumerate(pairs):
        mode = modes[i] if i < len(modes) else "road"
        if mode == "road" and not config.OFFLINE:
            try:
                seg = list(_osrm(round(a[0], 7), round(a[1], 7), round(b[0], 7), round(b[1], 7), bool(ignore_oneway)))
                # OSRM snaps to the nearest road; connect the clicked points to the snapped ends.
                if _haversine(a, seg[0]) > 1.0:
                    seg.insert(0, a)
                if _haversine(b, seg[-1]) > 1.0:
                    seg.append(b)
            except RoutingError as exc:
                warnings.append(f"Segment {i + 1}: {exc} - used a straight line instead.")
                seg = [a, b]
        else:
            seg = [a, b]
        segments.append([list(p) for p in seg])
    merged = []
    for seg in segments:
        for p in seg:
            if not merged or _haversine(merged[-1], p) > 0.3:
                merged.append(p)
    length = sum(_haversine(merged[i], merged[i + 1]) for i in range(len(merged) - 1))
    return {"segments": segments, "coords": merged, "length_m": length, "warnings": warnings}

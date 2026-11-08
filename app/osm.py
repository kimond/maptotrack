"""Fetch scenery (buildings, roads, trees, barriers, land areas) from OpenStreetMap via Overpass."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time

import requests

from . import config


class OSMError(Exception):
    pass


QUERY = """
[out:json][timeout:120];
(
  way["building"]({b});
  relation["building"]["type"="multipolygon"]({b});
  way["highway"]({b});
  node["natural"="tree"]({b});
  way["natural"~"^(wood|water|scrub)$"]({b});
  way["landuse"~"^(forest|grass|meadow|park)$"]({b});
  way["leisure"="park"]({b});
  way["barrier"~"^(wall|fence|guard_rail|retaining_wall)$"]({b});
);
out geom;
"""

DEFAULT_BUILDING_HEIGHT = {
    "house": 7, "detached": 7, "semidetached_house": 7, "terrace": 8, "bungalow": 4,
    "residential": 12, "apartments": 15, "commercial": 12, "retail": 6, "office": 18,
    "industrial": 9, "warehouse": 9, "garage": 3, "garages": 3, "shed": 3, "roof": 4,
    "hut": 3, "cabin": 4, "church": 16, "cathedral": 30, "school": 10, "hospital": 18,
    "hotel": 20, "farm_auxiliary": 6, "barn": 7, "kiosk": 3, "service": 4,
}

ROAD_WIDTH = {
    "motorway": 14, "trunk": 12, "primary": 10, "secondary": 9, "tertiary": 8,
    "unclassified": 6.5, "residential": 6.5, "living_street": 5, "service": 4.5,
    "motorway_link": 6, "trunk_link": 6, "primary_link": 6, "secondary_link": 6,
    "tertiary_link": 6, "track": 3.5, "pedestrian": 5, "footway": 2, "cycleway": 2,
    "path": 1.5, "steps": 2,
}
SKIP_HIGHWAYS = {"proposed", "construction", "abandoned", "platform", "elevator", "corridor", "bus_stop"}


def _num(text):
    if text is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", str(text).replace(",", "."))
    if not m:
        return None
    val = float(m.group())
    if "ft" in str(text) or "'" in str(text):
        val *= 0.3048
    return val


def building_height(tags: dict, osm_id) -> float:
    h = _num(tags.get("height")) or _num(tags.get("building:height"))
    if h and h > 1:
        return min(h, 400.0)
    lv = _num(tags.get("building:levels"))
    if lv and lv > 0:
        return lv * 3.1 + (1.5 if tags.get("roof:levels") else 0.5)
    base = DEFAULT_BUILDING_HEIGHT.get(tags.get("building", ""), 8)
    rnd = random.Random(str(osm_id))
    return base * rnd.uniform(0.85, 1.25)


def _merge_rings(ways):
    """Join way geometries (lists of (lat,lon)) into closed rings."""
    pending = [list(w) for w in ways if len(w) >= 2]
    rings = []
    while pending:
        cur = pending.pop(0)
        changed = True
        while cur[0] != cur[-1] and changed:
            changed = False
            for i, w in enumerate(pending):
                if w[0] == cur[-1]:
                    cur += w[1:]
                elif w[-1] == cur[-1]:
                    cur += w[::-1][1:]
                elif w[-1] == cur[0]:
                    cur = w[:-1] + cur
                elif w[0] == cur[0]:
                    cur = w[::-1][:-1] + cur
                else:
                    continue
                pending.pop(i)
                changed = True
                break
        if len(cur) >= 4 and cur[0] == cur[-1]:
            rings.append(cur)
    return rings


def _geom(el):
    return [(p["lat"], p["lon"]) for p in el.get("geometry", []) if p]


def parse(data: dict) -> dict:
    out = {"buildings": [], "roads": [], "trees": [], "areas": [], "barriers": []}
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        t = el.get("type")
        if t == "node":
            if tags.get("natural") == "tree":
                out["trees"].append({"id": f"n{el['id']}", "lat": el["lat"], "lon": el["lon"]})
            continue
        if t == "way":
            g = _geom(el)
            if len(g) < 2:
                continue
            wid = f"w{el['id']}"
            if "building" in tags and len(g) >= 4 and g[0] == g[-1]:
                out["buildings"].append({
                    "id": wid, "outer": [g], "inner": [], "tags": tags,
                    "height": building_height(tags, wid),
                })
            elif "highway" in tags:
                hw = tags["highway"]
                if hw in SKIP_HIGHWAYS or tags.get("area") == "yes":
                    continue
                width = _num(tags.get("width"))
                if not width:
                    lanes = _num(tags.get("lanes"))
                    width = lanes * 3.3 if lanes and hw not in ("footway", "path", "cycleway") else ROAD_WIDTH.get(hw, 5)
                out["roads"].append({
                    "id": wid, "coords": g, "highway": hw, "width": float(width),
                    "layer": int(_num(tags.get("layer")) or 0),
                    "bridge": tags.get("bridge") not in (None, "no"),
                    "tunnel": tags.get("tunnel") not in (None, "no"),
                    "maxspeed": maxspeed(tags.get("maxspeed")),
                    "lit": tags.get("lit"),
                    "oneway": {"yes": 1, "true": 1, "1": 1, "-1": -1}.get(tags.get("oneway", ""),
                                                                         1 if tags.get("junction") == "roundabout" else 0),
                    "roundabout": tags.get("junction") in ("roundabout", "circular"),
                    "service": tags.get("service"),
                })
            elif "barrier" in tags:
                out["barriers"].append({"id": wid, "coords": g, "kind": tags["barrier"]})
            elif g[0] == g[-1]:
                kind = tags.get("natural") or tags.get("landuse") or tags.get("leisure")
                out["areas"].append({"id": wid, "kind": kind, "outer": [g]})
        elif t == "relation" and "building" in tags:
            outer = _merge_rings([_geom(m) for m in el.get("members", []) if m.get("role") == "outer"])
            inner = _merge_rings([_geom(m) for m in el.get("members", []) if m.get("role") == "inner"])
            if outer:
                rid = f"r{el['id']}"
                out["buildings"].append({
                    "id": rid, "outer": outer, "inner": inner, "tags": tags,
                    "height": building_height(tags, rid),
                })
    return out


def maxspeed(text):
    """OSM maxspeed -> km/h (None when absent or not a number, e.g. 'signals', 'none')."""
    if not text:
        return None
    m = re.search(r"\d+(\.\d+)?", str(text))
    if not m:
        return None
    v = float(m.group())
    return round(v * 1.609344) if "mph" in str(text) else round(v)


# Street furniture and power lines: a separate download (own cache files f2_*.json, fu_*.json before crossings,
# guard rails and substations were added), so the scenery cache made before this existed stays valid.
FURNITURE_QUERY = """
[out:json][timeout:120];
(
  node["highway"~"^(traffic_signals|stop|give_way|street_lamp|turning_circle|turning_loop)$"]({b});
  node["power"~"^(tower|pole|transformer)$"]({b});
  node["highway"="crossing"]({b});
  way["power"~"^(line|minor_line|substation)$"]({b});
  way["barrier"="guard_rail"]({b});
);
out geom;
"""


def parse_furniture(data: dict) -> dict:
    out = {"signals": [], "stops": [], "giveways": [], "lamps": [], "towers": [], "lines": [], "turning": [],
           "crossings": [], "guardrails": [], "substations": [], "transformers": []}
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el.get("type") == "node":
            hw, pw = tags.get("highway"), tags.get("power")
            item = {"lat": el["lat"], "lon": el["lon"]}
            if hw == "traffic_signals":
                out["signals"].append(item)
            elif hw in ("stop", "give_way"):
                item["direction"] = tags.get("direction")
                item["all_way"] = tags.get("stop") == "all"
                out["stops" if hw == "stop" else "giveways"].append(item)
            elif hw == "street_lamp":
                out["lamps"].append(item)
            elif hw in ("turning_circle", "turning_loop"):
                item["diameter"] = _num(tags.get("diameter"))
                item["loop"] = hw == "turning_loop"
                out["turning"].append(item)
            elif hw == "crossing":
                style = crossing_style(tags)
                if style:
                    item["style"] = style
                    out["crossings"].append(item)
            elif pw in ("tower", "pole"):
                item["kind"] = pw
                out["towers"].append(item)
            elif pw == "transformer":
                out["transformers"].append(item)
        elif el.get("type") == "way" and tags.get("barrier") == "guard_rail":
            g = _geom(el)
            if len(g) >= 2 and not tags.get("layer", "0").startswith("-"):
                out["guardrails"].append({"id": f"w{el['id']}", "coords": g})
        elif el.get("type") == "way" and tags.get("power") == "substation":
            g = _geom(el)
            if len(g) >= 4 and g[0] == g[-1] and tags.get("location") not in ("underground", "indoor", "roof") \
                    and tags.get("substation") != "minor_distribution":
                out["substations"].append({"id": f"w{el['id']}", "coords": g,
                                           "voltage": _num(str(tags.get("voltage", "")).split(";")[0]) or 0.0,
                                           "kind": tags.get("substation", "")})
        elif el.get("type") == "way" and tags.get("power") in ("line", "minor_line"):
            g = _geom(el)
            if len(g) < 2 or tags.get("location") in ("underground", "underwater") or tags.get("layer", "0").startswith("-"):
                continue
            out["lines"].append({"id": f"w{el['id']}", "coords": g, "kind": tags["power"],
                                 "voltage": _num(str(tags.get("voltage", "")).split(";")[0]) or 0.0,
                                 "cables": int(_num(tags.get("cables")) or 0)})
    return out


def crossing_style(tags: dict) -> str | None:
    """How a mapped pedestrian crossing is painted: 'zebra' (stripes along the road), 'lines' (two lines across
    it), 'signals' (a signalled crossing: lines in North America, zebra elsewhere) or None (not painted)."""
    m = (tags.get("crossing:markings") or "").lower()
    c = (tags.get("crossing") or "").lower()
    ref = (tags.get("crossing_ref") or "").lower()
    if m == "no" or c in ("unmarked", "no", "informal"):
        return None
    if m in ("lines", "dashes", "dots", "surface"):
        return "lines"
    if m.startswith(("zebra", "ladder")) or c == "zebra" or ref in ("zebra", "pelican", "toucan", "puffin", "pegasus"):
        return "zebra"
    if c == "traffic_signals":
        return "signals"
    if c in ("marked", "uncontrolled", "yes") or m == "yes":       # a bare highway=crossing: unknown, not painted
        return "zebra"
    return None


CACHE_GRID = 0.004       # degrees (~450 m north-south): small route edits reuse the same download


def _snap(bbox):
    s, w, n, e = bbox
    g = CACHE_GRID
    return (math.floor(s / g) * g, math.floor(w / g) * g, math.ceil(n / g) * g, math.ceil(e / g) * g)


def _cached(bbox, prefix="bb_"):
    """A cached download that covers bbox (exact old-style key, or any larger new-style area)."""
    d = config.CACHE_DIR / "osm"
    b = ",".join(f"{v:.6f}" for v in bbox)
    old = d / f"{hashlib.sha1(b.encode()).hexdigest()[:16]}.json"
    if prefix == "bb_" and old.exists():
        return old
    s, w, n, e = bbox
    best = None
    for f in d.glob(f"{prefix}*.json") if d.exists() else []:
        try:
            cs, cw, cn, ce = (float(v) for v in f.stem[len(prefix):].split("_"))
        except ValueError:
            continue
        if cs <= s and cw <= w and cn >= n and ce >= e:
            area = (cn - cs) * (ce - cw)
            if best is None or area < best[0]:
                best = (area, f)
    return best[1] if best else None


def fetch(bbox, log=print) -> dict:
    """bbox = (south, west, north, east). Cached on disk; any earlier download covering the area is reused.

    The public Overpass servers are free and often overloaded (HTTP 502/503/504 or 429, or very slow), so
    several servers are tried, then the whole list once more after a pause."""
    if config.OFFLINE:
        from .demo import synthetic_osm

        return synthetic_osm(bbox)
    return parse(_download(bbox, QUERY, "bb_", "OpenStreetMap data", log))


def fetch_furniture(bbox, log=print) -> dict:
    """Traffic lights, stop / give-way signs, street lamps, power towers and lines (OSM), cached like fetch()."""
    if config.OFFLINE:
        from .demo import synthetic_furniture

        return synthetic_furniture(bbox)
    return parse_furniture(_download(bbox, FURNITURE_QUERY, "f2_", "OpenStreetMap street furniture", log))


def _download(bbox, query, prefix, label, log):
    hit = _cached(bbox, prefix)
    if hit is not None:
        log(f"{label}: using the saved copy")
        return json.loads(hit.read_text(encoding="utf-8"))
    sb = _snap(bbox)
    b = ",".join(f"{v:.6f}" for v in sb)
    cache = config.CACHE_DIR / "osm" / (prefix + "_".join(f"{v:.3f}" for v in sb) + ".json")
    q = query.format(b=b)
    last = None
    for attempt in range(2):
        if attempt:
            log("  all OpenStreetMap servers are busy - trying again in 20 s ...")
            time.sleep(20)
        for url in config.OVERPASS_URLS:
            host = url.split("/")[2]
            t0 = time.time()
            try:
                log(f"Downloading {label} from {host} ...")
                r = requests.post(url, data={"data": q}, headers={"User-Agent": config.USER_AGENT},
                                  timeout=(10, 120))
                if r.status_code in (429, 502, 503, 504):
                    raise OSMError(f"server busy (HTTP {r.status_code})")
                r.raise_for_status()
                data = r.json()
                if "remark" in data and not data.get("elements"):
                    raise OSMError(f"server error: {data['remark'][:120]}")
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data), encoding="utf-8")
                log(f"  done in {time.time() - t0:.0f} s")
                return data
            except requests.exceptions.Timeout:
                last = f"{host} did not answer in time"
                log(f"  {host}: no answer after {time.time() - t0:.0f} s - trying the next server")
            except Exception as exc:
                last = exc
                log(f"  {host}: {exc} - trying the next server")
    raise OSMError("Could not download OpenStreetMap data - the public servers are busy. Try again in a few "
                   f"minutes. (last error: {last})")

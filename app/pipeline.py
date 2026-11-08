"""End-to-end generation: route -> centerline -> meshes -> files."""
from __future__ import annotations

import io
import json
import time

import numpy as np
from PIL import Image

from . import csp, driveways, parallel
from . import network as network_mod
from . import markings as markings_mod
from . import tyres as tyres_mod
from . import ai_line, hrdem
from . import ac_export, config, elevation, imagery, landcover, lidar, meshes, osm, overture, photo, roadwidth
from . import buildings as buildings_mod
from . import ground as ground_mod
from . import grass as grass_mod
from . import props as props_mod
from . import materials
from . import groundmat
from .textures import house_textures
from .geo import LocalProjection, bbox_of, slugify
from .textures import make_textures
from .track import build_centerline, profile_roughness

DEFAULTS = {
    "name": "My Track",
    "loop": True,
    "reverse": False,
    "width": 10.0,
    "camber": 0.06,
    "shoulder": 1.5,
    "corner_smoothing": 8.0,
    "elev_smoothing": 25.0,
    "width_smoothing": 30.0,      # m: photo-measured width averaged over this length (0 = raw measurement)
    "start_offset": 0.0,
    "walls": "concrete",          # none | concrete
    "wall_offset": 3.0,
    "wall_height": 1.0,
    "margin": 250.0,
    "terrain": True,
    "terrain_spacing": 8.0,
    "dem_zoom": 14,
    "buildings": True,
    "buildings_collide": True,
    "side_roads": True,
    "overture_buildings": True,
    "ground": "auto",             # auto | <imagery source id> | landcover | plain
    "ground_res": 0.25,           # metres per texture pixel
    "near_ground": 30.0,          # m beside the track with tiled photo ground (grass, soil, gravel...); 0 = off
    "ml_ground": False,       # AC, experimental: ksMultilayer close-up detail on the ground (greyed the ground
                                  # in game, 2026-11-02); the track / streets / shoulders use tiled photo materials
    "trees": True,
    "tree_density": 1.0,
    "max_trees": 20000,
    "grass": True,
    "wild_grass": True,
    "grass_density": 1.0,
    "grass_fx": True,
    "csp_trees": True,                # CSP 3D trees from csp_trees/*.bin at the real tree positions (if any)
    "tyres": "auto",                  # tyre walls from the photo: "auto" (when OSM maps a race track), "on", "off"
    "markings": "photo",              # road lines: "photo" (as painted), "standard" (edges + dashed centre), "none"                 # CSP Grass FX config (extension/ext_config.ini)
    "max_grass": 30000,
    "max_tall_grass": 25000,
    "conifer_ratio": 0.6,
    "overture_landcover": True,
    "width_mode": "photo",        # photo | fixed
    "photo_paved": True,
    "width_overrides": [],         # [{lat, lon, width, span}] set on the map
    "photo_trees": True,
    "photo_buildings": True,      # find unmapped buildings, snap mapped outlines, LiDAR heights
    "lidar": False,
    "hrdem": True,                # NRCan HRDEM 1 m / 2 m LiDAR terrain (Canada; only the track's window is read)
    "power_lines": True,          # OSM power lines: pylons / poles and their wires
    "road_signs": True,           # traffic lights, stop / give-way and speed-limit signs
    "street_lights": True,        # street lamps (+ wooden utility poles along streets in the Americas)
    "crosswalks": True,           # OSM pedestrian crossings painted on the road (zebra / two lines)
    "guardrails": True,           # OSM barrier=guard_rail: W-beam rails on posts (collide)
    "csp_lights": True,           # CSP: the street lamps light the road at night (ext_config.ini)
    "csp_rain": True,             # CSP RainFX: puddles, wet surfaces, slippery paint (ext_config.ini)
    "sign_style": "auto",         # auto | qc | ca | us | eu
    "kn5": True,
    "kn5_dds": True,
    "grid": 10,
    "pits": 10,
    "mode": "route",                  # route (a track along waypoints) | area (every street in a rectangle)
    "area": None,                     # area mode: [south, west, north, east] (degrees)
    "spawn": None,                    # area mode: [lat, lon] of the start spot (None = the main street near the centre)
    "force_include": [],
    "force_exclude": [],
    "author": "",
    "city": "",
    "country": "",
}


class ShiftedDEM:
    def __init__(self, dem, dz):
        self.dem, self.dz = dem, dz

    def sample(self, lat, lon):
        return np.asarray(self.dem.sample(lat, lon), dtype=float) - self.dz


def _params(p: dict) -> dict:
    out = dict(DEFAULTS)
    out.update({k: v for k, v in p.items() if v is not None})
    for k in ("width", "camber", "shoulder", "corner_smoothing", "elev_smoothing", "width_smoothing", "near_ground",
              "start_offset",
              "wall_offset", "wall_height", "margin", "terrain_spacing", "ground_res", "tree_density",
              "conifer_ratio"):
        out[k] = float(out[k])
    for k in ("grid", "pits", "dem_zoom", "max_trees", "max_grass", "max_tall_grass"):
        out[k] = int(out[k])
    out["terrain_spacing"] = float(np.clip(out["terrain_spacing"], 2.0, 50.0))
    if out.get("mode") == "area":
        # free roam on city streets: no gravel shoulders, barriers or tyre walls; the area is the rectangle
        out["coords"] = area_corners(out)
        out.update({"shoulder": 0.0, "walls": "none", "tyres": "off", "loop": False})
    out["width"] = float(np.clip(out["width"], 3.0, 40.0))
    return out


def area_corners(P):
    """Area mode: the area's outline [(lat, lon)]: the polygon drawn on the map (`area_poly`), else the
    rectangle `area` [s, w, n, e]. Checked: at most MAX_AREA_KM2, bounding box at most MAX_SIDE_M a side."""
    from shapely.geometry import Polygon

    poly = P.get("area_poly")
    if poly and len(poly) >= 3:
        ring = [[float(p[0]), float(p[1])] for p in poly]
        if ring[0] == ring[-1]:
            ring = ring[:-1]
    else:
        a = P.get("area")
        if not a or len(a) != 4:
            raise ValueError("Draw the area on the map first.")
        s_, w_, n_, e_ = (float(v) for v in a)
        s_, n_ = min(s_, n_), max(s_, n_)
        w_, e_ = min(w_, e_), max(w_, e_)
        ring = [[s_, w_], [s_, e_], [n_, e_], [n_, w_]]
    if len(ring) < 3:
        raise ValueError("The area needs at least 3 corners.")
    lat = np.array([p[0] for p in ring])
    lon = np.array([p[1] for p in ring])
    lat0 = float(lat.mean())
    xy = np.column_stack([(lon - lon.mean()) * 111320.0 * np.cos(np.radians(lat0)), (lat - lat0) * 111320.0])
    g = Polygon(xy)
    if not g.is_valid:
        raise ValueError("The area's outline crosses itself - redraw it.")
    w, h = float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))
    lim = network_mod.MAX_SIDE_M * 1.01
    if h > lim or w > lim:
        raise ValueError(f"The area spans {w / 1000:.2f} x {h / 1000:.2f} km - at most "
                         f"{network_mod.MAX_SIDE_M / 1000:.0f} x {network_mod.MAX_SIDE_M / 1000:.0f} km.")
    if g.area > network_mod.MAX_AREA_KM2 * 1e6 * 1.01:
        raise ValueError(f"The area is {g.area / 1e6:.1f} km² - at most {network_mod.MAX_AREA_KM2:.0f} km².")
    if h < 50 or w < 50 or g.area < 2500:
        raise ValueError("The area is too small.")
    return ring


def area_scale(bounds):
    """Big areas: (photo / ground texture resolution, photo analysis resolution, factor). The pixel counts stay
    those of a 2 x 2 km area at full detail (memory, generation time, .kn5 size)."""
    x0, y0, x1, y1 = bounds
    km2 = (x1 - x0) * (y1 - y0) / 1e6
    f = float(np.sqrt(km2 / network_mod.BIG_AREA_KM2)) if km2 > network_mod.BIG_AREA_KM2 else 1.0
    ground = next((r for r in (0.25, 0.5, 1.0) if r >= 0.25 * f - 1e-6), 1.0)
    return ground, max(0.5, 0.5 * f), f


def _area_bounds(proj, coords):
    xy = proj.latlngs_to_xy(coords)
    return (float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max()))


def _area_samples(coords, n=9):
    a = np.asarray(coords, float)
    la = np.linspace(a[:, 0].min(), a[:, 0].max(), n)
    lo = np.linspace(a[:, 1].min(), a[:, 1].max(), n)
    LA, LO = np.meshgrid(la, lo)
    return np.column_stack([LA.ravel(), LO.ravel()])


def _projection(coords):
    arr = np.asarray(coords, float)
    return LocalProjection((arr[:, 0].min() + arr[:, 0].max()) / 2, (arr[:, 1].min() + arr[:, 1].max()) / 2)


def load_scenery(bbox, P, log=print, warnings=None):
    """OSM data, optionally topped up with Overture building footprints where OSM has none."""
    data = osm.fetch(bbox, log=log)
    n_osm = len(data["buildings"])
    data["n_osm_buildings"] = n_osm
    data["n_extra_buildings"] = 0
    if P.get("overture_buildings") and not config.OFFLINE:
        try:
            data["n_extra_buildings"] = overture.merge_into(data, bbox, log=log)
            log(f"Buildings: {n_osm} from OpenStreetMap + {data['n_extra_buildings']} from Overture Maps")
        except Exception as exc:
            msg = f"Extra buildings (Overture Maps) failed: {exc}"
            log("WARNING: " + msg)
            if warnings is not None:
                warnings.append(msg)
    return data


def clearance(P):
    return P["width"] / 2 + P["shoulder"] + 1.0


def road_preview(params: dict, log=print) -> dict:
    """Just the road, for the map, in seconds: centreline, width (measured on the photo along the track only,
    smoothed, manual settings), elevation profile, start / pit spots and the warnings. The same functions as a
    full generation, so what the map shows is what will be built (LiDAR heights excepted)."""
    t0 = time.time()
    P = _params(params)
    if P.get("mode") == "area":
        return area_preview(P, log, t0)
    coords = P["coords"]
    if len(coords) < 2:
        raise ValueError("Add at least two waypoints")
    proj = _projection(coords)
    warnings = []
    bbox = bbox_of(coords, 60.0)
    log("Elevation ...")
    try:
        dem_raw = elevation.make_dem(bbox, P["dem_zoom"], log=log)
    except Exception as exc:
        warnings.append(f"Elevation unavailable ({exc}) - flat profile.")
        dem_raw = elevation.FlatDEM()
    arr = np.asarray(coords, float)
    z = dem_raw.sample(arr[:, 0], arr[:, 1])
    z0 = float(np.round(np.nanmedian(z))) if np.isfinite(z).any() else 0.0
    cl = build_centerline(coords, proj, ShiftedDEM(dem_raw, z0), P["loop"], spacing=2.0,
                          corner_smoothing=P["corner_smoothing"], elev_smoothing=P["elev_smoothing"],
                          start_offset=P["start_offset"], reverse=P["reverse"])
    warnings += profile_warnings(cl)
    pm = None
    if P["width_mode"] == "photo" or P["markings"] == "photo" or P["tyres"] != "off":
        src = imagery.pick_source(P["ground"] if P["ground"] not in ("landcover", "plain") else "auto", bbox)
        if src:
            try:
                pm = photo.corridor_photomap(src, proj, cl, res=max(P["ground_res"], 0.1), log=log)
                if pm.valid.sum() < 2000:
                    warnings.append("The aerial photo source returned no imagery along the track.")
                    pm = None
            except Exception as exc:
                warnings.append(f"Aerial imagery failed ({exc}).")
                pm = None
    hl, hr, conf, width_info = road_widths(P, cl, proj, pm, log, warnings)
    warnings += corner_warnings(cl, hl, hr)
    wmean = float(np.mean(hl + hr))
    objs = ac_export.ac_objects(cl, wmean, P["grid"], P["pits"], widths=(hl, hr), camber=P["camber"])
    out = road_overlay(cl, hl, hr, conf, proj, z0, objs, width_info)
    mk = road_markings(P, cl, hl, hr, pm, log, warnings)
    out["markings"] = {"info": mk.info, "lines": markings_mod.overlay(mk, cl, proj)}
    roads = None
    if P["tyres"] == "auto" and pm is not None:
        try:
            roads = osm.fetch(bbox_of(coords, P["margin"]), log=log)["roads"]
        except Exception:
            roads = None
    walls, tinfo = tyre_walls(P, cl, hl, hr, pm, proj, roads, log, warnings)
    out["tyres"] = {"info": tinfo, "lines": tyres_mod.overlay(walls, proj)}
    # outer lines: shoulder and barriers
    k = max(len(cl.p) // 1500, 1)
    extra = {}
    if P["shoulder"] > 0:
        extra["shoulder"] = P["shoulder"]
    if P["walls"] == "concrete":
        extra["walls"] = P["wall_offset"]
    for name, off in extra.items():
        lines = []
        for sgn, h in ((1, hl), (-1, hr)):
            q = cl.p + cl.n * (sgn * (h + off))[:, None]
            la, lo = proj.to_latlon(q[::k, 0], q[::k, 1])
            lines.append(np.column_stack([la, lo]).round(7).tolist())
        out[f"{name}_lines"] = lines
    out.update({"preview": True, "warnings": warnings, "elapsed_s": round(time.time() - t0, 1),
                "origin": {"lat": proj.lat0, "lon": proj.lon0, "z0": z0}})
    log(f"Road preview ready in {out['elapsed_s']} s.")
    return out


def area_preview(P, log=print, t0=None):
    """Area mode 'Preview road on map': every street of the area with its measured width and markings, and
    the start spots - the same functions as a full generation (LiDAR heights excepted)."""
    t0 = time.time() if t0 is None else t0
    coords = P["coords"]
    proj = _projection(coords)
    bounds = _area_bounds(proj, coords)
    warnings = []
    log("Elevation, streets" + (" and aerial photo" if P["width_mode"] == "photo" or P["markings"] == "photo" else "")
        + " ...")
    sbox = bbox_of(coords, 25.0)
    f_dem = parallel.prefetch(elevation.make_dem, bbox_of(coords, 60.0), P["dem_zoom"], log=log)
    f_osm = parallel.prefetch(osm.fetch, sbox, log=log)
    f_furn = parallel.prefetch(osm.fetch_furniture, sbox, log=log)
    pm = None
    if P["width_mode"] == "photo" or P["markings"] == "photo":
        src = imagery.pick_source(P["ground"] if P["ground"] not in ("landcover", "plain") else "auto", sbox)
        if src:
            # only the photo along the streets is read, piece by piece (cached pieces, any area size)
            pm = photo.TiledPhoto(src, proj, res=0.25)
    try:
        dem_raw = f_dem.result()
    except Exception as exc:
        warnings.append(f"Elevation unavailable ({exc}) - flat.")
        dem_raw = elevation.FlatDEM()
    arr = _area_samples(coords)
    z = dem_raw.sample(arr[:, 0], arr[:, 1])
    z0 = float(np.round(np.nanmedian(z))) if np.isfinite(z).any() else 0.0
    try:
        roads = f_osm.result()["roads"]
    except osm.OSMError as exc:
        raise ValueError(f"No street data for this area: {exc}")
    try:
        turning = f_furn.result().get("turning", [])
    except Exception:
        turning = []
    log("Streets of the area ...")
    from shapely.geometry import Polygon as _Poly

    net = network_mod.build(roads, proj, bounds, ShiftedDEM(dem_raw, z0), P, pm, log=log, warnings=warnings,
                            turning=turning, clip=_Poly(proj.latlngs_to_xy(coords)).buffer(0))
    network_mod.markings(net, P, pm, log=log, warnings=warnings)
    scl, shl, shr, s0, _ = network_mod.spawn_street(net, proj, P, warnings)
    objs, n_spots = ac_export.spawn_objects(scl, shl, shr, s0, P["grid"], P["pits"], camber=P["camber"])
    if n_spots < max(P["grid"], P["pits"]):
        warnings.append(f"The start street is short: {n_spots} start / pit spots fit on it.")
    nm = sum(c.measured for c in net.chains)
    ws = [float(np.median(c.hl + c.hr)) for c in net.chains]
    width_info = {"used": nm > 0, "streets": len(net.chains), "measured": nm, "median": round(float(np.median(ws)), 1),
                  "min": round(float(min(ws)), 1), "max": round(float(max(ws)), 1)}
    out = area_overlay(net, proj, z0, objs, width_info)
    out["markings"] = area_markings(net, proj)
    out.update({"preview": True, "warnings": warnings, "elapsed_s": round(time.time() - t0, 1),
                "origin": {"lat": proj.lat0, "lon": proj.lon0, "z0": z0}, "tyres": {"info": {"used": False}, "lines": []}})
    log(f"Area preview ready in {out['elapsed_s']} s.")
    return out


def scenery_preview(params: dict, log=print) -> dict:
    """Fetch OSM for the route area and flag buildings that sit on the track (area mode: on the streets)."""
    P = _params(params)
    coords = P["coords"]
    proj = _projection(coords)
    area = P.get("mode") == "area"
    bbox = bbox_of(coords, 25.0 if area else P["margin"])
    warn = []
    data = load_scenery(bbox, P, log=log, warnings=warn)
    if area:
        from shapely.geometry import LineString
        from shapely.ops import unary_union

        from shapely.geometry import Point, Polygon as _Poly

        clip = _Poly(proj.latlngs_to_xy(coords)).buffer(0)
        ways = network_mod.ways_in(data["roads"], proj, _area_bounds(proj, coords), clip=clip)
        geom = unary_union([LineString(xy).buffer(r["width"] / 2 + 1.0) for xy, r in ways]) if ways else None
        auto = set(overlapping_buildings(data["buildings"], proj, geom)) if geom is not None else set()
        x0, y0, x1, y1 = _area_bounds(proj, coords)
        inside = []
        for b in data["buildings"]:
            c = proj.latlngs_to_xy(b["outer"][0]).mean(0)
            if x0 < c[0] < x1 and y0 < c[1] < y1 and clip.contains(Point(c)):
                inside.append(b)
        data["buildings"] = inside
    else:
        cl = build_centerline(coords, proj, elevation.FlatDEM(), P["loop"], corner_smoothing=P["corner_smoothing"],
                              reverse=P["reverse"])
        auto = set(meshes.classify_buildings(data["buildings"], proj, cl, clearance(P)))
    feats = []
    for b in data["buildings"]:
        feats.append({
            "type": "Feature",
            "properties": {"id": b["id"], "height": round(b["height"], 1), "auto_excluded": b["id"] in auto,
                           "kind": b["tags"].get("building", "yes"), "source": "overture" if b["id"].startswith("o") else "osm", "name": b["tags"].get("name", "")},
            "geometry": {"type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in ring] for ring in b["outer"][:1]]},
        })
    return {
        "buildings": {"type": "FeatureCollection", "features": feats},
        "warnings": warn,
        "counts": {"buildings": len(data["buildings"]), "osm_buildings": data["n_osm_buildings"],
                   "extra_buildings": data["n_extra_buildings"],
                   "roads": len(data["roads"]), "trees": len(data["trees"]),
                   "auto_excluded": len(auto)},
        "bbox": bbox,
    }


def _ll_polys(polys, proj, limit=3000, tol=0.4):
    out = []
    for p in polys[:limit]:
        q = p.simplify(tol)
        xy = np.asarray(q.exterior.coords)
        la, lo = proj.to_latlon(xy[:, 0], xy[:, 1])
        out.append(np.column_stack([la, lo]).round(7).tolist())
    return out


def _roof_colours(blds, skip, proj, pm):
    """Typical roof colour of every building from the photo (picks the matching shingle colour)."""
    from shapely import contains_xy

    for i, b in enumerate(blds):
        if b["id"] in skip or "roof_rgb" in b:
            continue
        try:
            polys = meshes._polygons_for_building(b, proj)
        except Exception:
            continue
        if not polys:
            continue
        p = polys[0].buffer(-0.6)
        if p.is_empty or p.area < 6:
            continue
        x0, y0, x1, y1 = p.bounds
        xs, ys = np.arange(x0, x1, 0.5), np.arange(y1, y0, -0.5)
        if not len(xs) or not len(ys):
            continue
        ph, ok = pm.grid(xs, ys)
        X, Y = np.meshgrid(xs, ys)
        m = ok & contains_xy(p, X, Y)
        if m.sum() < 10:
            continue
        px = ph[m]
        lum = px.mean(1)
        lo, hi = np.percentile(lum, [25, 75])    # skip the shadowed slope / specular / vents
        sel = px[(lum >= lo) & (lum <= hi)]
        if len(sel):
            blds[i] = dict(b, roof_rgb=[float(v) for v in np.median(sel, 0)])
    # Aerial haze lifts dark shingles to mid-grey, so pick shingles by rank within this photo
    # (darkest roofs -> black, typical -> charcoal, lightest -> grey) and by hue (brown / red).
    from .roadwidth import rgb_to_lab

    idx = [i for i, b in enumerate(blds) if b.get("roof_rgb") and "roof_mat" not in b]
    if len(idx) >= 5:
        lab = rgb_to_lab(np.array([blds[i]["roof_rgb"] for i in idx], np.float32)[None])[0]
        rank = lab[:, 0].argsort().argsort() / max(len(idx) - 1, 1)
        for j, i in enumerate(idx):
            L, a, bb = lab[j]
            if a > 5.0 and bb > 3.0:
                mat = "shingle_red"
            elif bb > 3.5 and a > 1.5:
                mat = "shingle_brown"
            elif rank[j] < 0.2:
                mat = "shingle_black"
            elif rank[j] < 0.7:
                mat = "shingle_charcoal"
            else:
                mat = "shingle_grey"
            blds[i] = dict(blds[i], roof_mat=mat)


def road_widths(P, cl, proj, pm, log=print, warnings=None):
    """Half widths left / right along the centreline: measured on the photo (+ width smoothing) or fixed,
    then the manual width settings. -> hl, hr, edge confidence, width info."""
    warnings = [] if warnings is None else warnings
    width_info = {"used": False}
    hl = np.full(len(cl.p), P["width"] / 2)
    hr = hl.copy()
    if P["width_mode"] == "photo":
        if pm is not None:
            log("Measuring road width on the photo ...")
            hl, hr, width_info = roadwidth.measure_road_edges(pm, cl, P["width"])
            if width_info.get("used"):
                log(f"  width {width_info['min']}-{width_info['max']} m (median {width_info['median']} m), "
                    f"clear edges on {width_info['confident_fraction'] * 100:.0f}% of the track")
                if P["width_smoothing"] > 0:
                    hl, hr = roadwidth.smooth_widths(cl, hl, hr, width_info.get("confidence"), P["width_smoothing"])
                    wd = hl + hr
                    width_info.update({"min": round(float(wd.min()), 1), "max": round(float(wd.max()), 1),
                                       "median": round(float(np.median(wd)), 1),
                                       "smoothing_m": P["width_smoothing"]})
                    log(f"  smoothed over {P['width_smoothing']:.0f} m: {width_info['min']}-{width_info['max']} m")
            else:
                warnings.append(f"Could not read the road width on the photo ({width_info.get('reason')}) "
                                f"- using {P['width']} m.")
        else:
            log("No aerial photo here - using the fixed road width.")
    conf = np.asarray(width_info.get("confidence", np.zeros(len(cl.p))), float)
    if P["width_overrides"]:
        ovs = []
        for ov in P["width_overrides"]:
            try:
                ox, oy = proj.to_xy(float(ov["lat"]), float(ov["lon"]))
                i = int(np.argmin(np.hypot(cl.p[:, 0] - ox, cl.p[:, 1] - oy)))
                ovs.append({"s": float(cl.s[i]), "width": float(ov["width"]), "span": float(ov.get("span", 40))})
            except (KeyError, TypeError, ValueError):
                continue
        hl, hr = roadwidth.apply_overrides(cl, hl, hr, ovs)
        log(f"  {len(ovs)} manual width setting(s) applied")
    width_info = {k: v for k, v in width_info.items() if k != "confidence"}
    return hl, hr, conf, width_info


def road_markings(P, cl, hl, hr, pm, log=print, warnings=None):
    """Which lines to paint on the track: read on the photo (as painted in real life), the standard layout,
    or none. -> markings.Markings (info['source'] says which)."""
    warnings = [] if warnings is None else warnings
    mode = P.get("markings", "photo")
    if mode == "none":
        return markings_mod.Markings([], {"source": "none"})
    if mode == "photo":
        if pm is None:
            log("No aerial photo here - standard road markings.")
        else:
            log("Reading the road markings on the photo ...")
            try:
                mk = markings_mod.detect(pm, cl, hl, hr, log=log)
            except Exception as exc:                     # never lose the track over its paint
                mk = None
                warnings.append(f"Reading the road markings failed ({exc}) - standard markings.")
            if mk is not None and mk.info.get("usable"):
                mk.info["source"] = "photo"
                if mk.lines:
                    desc = ", ".join(f"{ln.colour} {ln.style} {ln.side} line "
                                     f"({sum(b - a for a, b in ln.pieces):.0f} m)" for ln in mk.lines)
                    log(f"  {desc}")
                else:
                    log("  no painted lines on this road (as on the photo)")
                return mk
            if mk is not None:
                warnings.append(f"Road markings: {mk.info.get('reason', 'the photo is unclear')} - standard markings.")
    mk = markings_mod.standard(cl, hl, hr)
    return mk


def tyre_walls(P, cl, hl, hr, pm, proj, roads=None, log=print, warnings=None):
    """Tyre walls read on the photo, when asked for (or 'auto' and OSM maps this as a race / kart track).
    -> (walls, info)."""
    mode = P.get("tyres", "auto")
    if mode == "off" or pm is None:
        return [], {"used": False}
    if mode == "auto":
        if roads is None or not tyres_mod.is_race_track(roads, proj, cl):
            return [], {"used": False, "reason": "not mapped as a race track"}
    log("Finding tyre walls on the photo ...")
    try:
        walls, info = tyres_mod.detect(pm, cl, hl, hr, log=log)
    except Exception as exc:
        if warnings is not None:
            warnings.append(f"Finding tyre walls failed ({exc}).")
        return [], {"used": False}
    info["used"] = True
    log(f"  {info['walls']} tyre walls, {info['length_m']} m")
    return walls, info


def profile_warnings(cl):
    out = []
    grade = np.abs(np.diff(cl.z)) / cl.spacing
    if grade.max() > 0.25:
        out.append(f"Max gradient {grade.max() * 100:.0f}% - consider more elevation smoothing.")
    bump, dgrade = profile_roughness(cl.s, cl.z, cl.loop)
    if bump > 0.03 or dgrade > 3.0:
        out.append(f"The road height has bumps (up to {bump * 100:.0f} cm, slope changing by {dgrade:.0f}% "
                   f"within 2 m) - raise Elevation smoothing (10-15 m for a kart track, 25 m for roads).")
    return out


def corner_warnings(cl, hl, hr):
    inner = np.where(cl.radius > 0, hl, hr)
    if ((np.abs(cl.radius) - inner) < 1.0).any():
        return [f"Tightest corner radius is {float(np.min(np.abs(cl.radius))):.1f} m, close to the road half-width - "
                f"increase corner smoothing if the inside edge looks folded."]
    return []


def road_overlay(cl, hl, hr, conf, proj, z0, objs, width_info):
    """What the map shows of the road (shared by the preview and the full generation)."""
    lat, lon = proj.to_latlon(cl.p[:, 0], cl.p[:, 1])
    k = max(len(cl.p) // 1500, 1)
    le = cl.p + cl.n * hl[:, None]
    re_ = cl.p - cl.n * hr[:, None]
    la1, lo1 = proj.to_latlon(le[::k, 0], le[::k, 1])
    la2, lo2 = proj.to_latlon(re_[::k, 0], re_[::k, 1])
    obj_ll = []
    for o in objs:
        la, lo = proj.to_latlon(o["pos"][0], o["pos"][1])
        obj_ll.append({"name": o["name"], "lat": float(la), "lon": float(lo), "pos": o["pos"], "forward": o["forward"]})
    grade = np.abs(np.diff(cl.z)) / cl.spacing
    step = max(len(cl.s) // 400, 1)
    return {
        "length_m": round(cl.length, 1),
        "loop": cl.loop,
        "min_radius_m": round(float(np.min(np.abs(cl.radius))), 1),
        "elevation": {"min": round(float(cl.z.min() + z0), 1), "max": round(float(cl.z.max() + z0), 1),
                      "gain": round(float(np.sum(np.clip(np.diff(cl.z), 0, None))), 1),
                      "max_grade_pct": round(float(grade.max() * 100), 1)},
        "profile": {"s": cl.s[::step].round(1).tolist(), "z": (cl.z[::step] + z0).round(2).tolist(),
                    "w": (hl + hr)[::step].round(2).tolist()},
        "width": width_info,
        "objects": obj_ll,
        "built_line": np.column_stack([lat, lon])[::k].round(7).tolist(),
        "built_s": cl.s[::k].round(1).tolist(),
        "built_w": (hl + hr)[::k].round(2).tolist(),
        "edge_conf": conf[::k].round(2).tolist(),
        "edges": [np.column_stack([la1, lo1]).round(7).tolist(), np.column_stack([la2, lo2]).round(7).tolist()],
    }


def overlapping_buildings(buildings, proj, geom):
    """Ids of buildings that overlap the streets (area mode: removed like those on a track)."""
    from shapely.prepared import prep

    g = prep(geom)
    hits = []
    for b in buildings:
        for poly in meshes._polygons_for_building(b, proj):
            if g.intersects(poly):
                hits.append(b["id"])
                break
    return hits


def area_overlay(net, proj, z0, objs, width_info):
    """What the map shows of an area: every street (edges, width), the start spots, the totals."""
    ov = net.overlay(proj, objs)
    ov["elevation"] = {"min": round(ov["elevation"]["min"] + z0, 1), "max": round(ov["elevation"]["max"] + z0, 1)}
    ov.update({"area": True, "loop": False, "width": width_info})
    return ov


def area_markings(net, proj):
    lines, src = [], {}
    for c in net.chains:
        if c.mk is None:
            continue
        s_ = c.mk.info.get("source", "none")
        src[s_] = src.get(s_, 0) + 1
        if c.mk.lines:
            lines += markings_mod.overlay(c.mk, c.cl, proj)
    return {"info": {"source": "area", "streets": src}, "lines": lines}


def generate(params: dict, log=print) -> dict:
    t0 = time.time()
    P = _params(params)
    coords = P["coords"]
    if len(coords) < 2:
        raise ValueError("Add at least two waypoints.")
    area = P.get("mode") == "area"
    slug = slugify(P["name"])
    out = config.OUTPUT_DIR / slug
    bl = out / "blender"
    acf = out / "ac_track" / slug
    bl.mkdir(parents=True, exist_ok=True)
    (out / "project.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    proj = _projection(coords)
    bbox = bbox_of(coords, 60.0 if area else P["margin"] + 50)
    sbox = bbox_of(coords, 25.0 if area else P["margin"])
    warnings, credits = [], []

    # ------------------------------------------------------------ start every download at once
    # The track's ground plan doesn't depend on heights, so the area (and the photo) is known now.
    clip = None                      # area mode: the area's outline (local metres)
    an_res = 0.5                     # photo analysis grid (m)
    if area:
        from shapely.geometry import Polygon as _Poly

        bounds = _area_bounds(proj, coords)
        clip = _Poly(proj.latlngs_to_xy(coords)).buffer(0)
        g_res, an_res, f_big = area_scale(bounds)
        if f_big > 1.0:
            # a big area: the photo / ground textures and the photo analysis get coarser so memory, time and the
            # .kn5 stay as for a 2 x 2 km area (street widths and markings still read at full resolution)
            if P["ground_res"] < g_res:
                P["ground_res"] = g_res
            P["near_ground"] = min(P["near_ground"], 15.0)
            log(f"Big area ({clip.area / 1e6:.1f} km², box {(bounds[2] - bounds[0]) / 1000:.1f} x "
                f"{(bounds[3] - bounds[1]) / 1000:.1f} km): ground detail {P['ground_res']} m/px, photo analysis "
                f"{an_res:.2f} m, sharp ground {P['near_ground']:.0f} m beside the streets")
    else:
        plan = build_centerline(coords, proj, elevation.FlatDEM(), P["loop"], spacing=2.0,
                                corner_smoothing=P["corner_smoothing"], elev_smoothing=0.0,
                                start_offset=P["start_offset"], reverse=P["reverse"])
        m = P["margin"]
        bounds = (plan.p[:, 0].min() - m, plan.p[:, 1].min() - m, plan.p[:, 0].max() + m, plan.p[:, 1].max() + m)
    use_ground = P["terrain"] and P["ground"] != "plain"
    photo_ground = False
    want_photo = (use_ground or P["width_mode"] == "photo" or P["markings"] == "photo" or P["tyres"] != "off"
                  or P["photo_trees"] or P["photo_paved"]
                  or P["photo_buildings"])
    src = imagery.pick_source(P["ground"] if P["ground"] not in ("landcover", "plain") else "auto", sbox) \
        if want_photo else None
    want_props = P["power_lines"] or P["road_signs"] or P["street_lights"] or P["crosswalks"] or P["guardrails"]
    want_scenery = (P["buildings"] or P["side_roads"] or P["trees"] or use_ground or src is not None or want_props
                    or area)
    want_lc = (P["trees"] or use_ground) and P["overture_landcover"] and not config.OFFLINE
    log("Downloading elevation" + (", LiDAR" if P["lidar"] else "") + (", aerial photo" if src else "")
        + (", OpenStreetMap" if want_scenery else "") + (", land cover" if want_lc else "")
        + f" in parallel ({parallel.workers()} cores) ...")
    f_dem = parallel.prefetch(elevation.make_dem, bbox, P["dem_zoom"], log=log)
    f_lidar = parallel.prefetch(lidar.open_products, bbox, log=log) if P["lidar"] else None
    f_hrdem = (parallel.prefetch(hrdem.open_dem, bbox, log=log)
               if P["hrdem"] and hrdem.in_canada(bbox) and not config.OFFLINE else None)
    f_photo = (parallel.prefetch(photo.build_photomap, src, proj, bounds, res=max(P["ground_res"], 0.1), log=log)
               if src else None)
    f_scenery = parallel.prefetch(load_scenery, sbox, P, log=log, warnings=warnings) if want_scenery else None
    f_lc = parallel.prefetch(overture.fetch_land_cover, sbox, log=log) if want_lc else None
    f_furn = parallel.prefetch(osm.fetch_furniture, sbox, log=log) if (want_props or area) else None
    f_mat = parallel.prefetch(materials.fetch, log=log)      # ambientCG photos (cached after the first time)
    photos_cache = []

    def photo_materials():
        if not photos_cache:
            try:
                photos_cache.append(f_mat.result())
            except Exception as exc:
                log(f"  photo materials unavailable: {exc} - procedural grain")
                photos_cache.append({})
            if photos_cache[0]:
                credits.append(materials.CREDIT)
        return photos_cache[0]

    # ------------------------------------------------------------ elevation (+ LiDAR)
    dem_raw = f_dem.result()
    arr = np.asarray(_area_samples(coords) if area else coords, float)
    chm = None
    elev_smooth = P["elev_smoothing"]
    if f_hrdem is not None:                  # Canada-wide LiDAR terrain under the Quebec sheets (if any)
        try:
            comp = lidar.CompositeDEM(f_hrdem.result(), dem_raw, arr[:, 0], arr[:, 1])
            if comp.coverage > 0.2:
                dem_raw = comp
                credits.append(hrdem.ATTRIBUTION)
                log(f"  HRDEM LiDAR terrain covers {comp.coverage * 100:.0f}% of the route")
            else:
                log("  HRDEM: no LiDAR terrain along this route - using the standard elevation")
        except Exception as exc:
            log(f"  HRDEM unavailable ({exc}) - using the standard elevation")
    if P["lidar"]:
        try:
            prods = f_lidar.result()
            if "MNT" in prods:
                comp = lidar.CompositeDEM(prods["MNT"], dem_raw, arr[:, 0], arr[:, 1])
                if comp.coverage > 0.2:
                    dem_raw = comp
                    log(f"  LiDAR terrain covers {comp.coverage * 100:.0f}% of the route "
                        f"(elevation smoothing {elev_smooth:.0f} m)")
            chm = prods.get("MHC")
            if prods:
                credits.append(lidar.ATTRIBUTION)
            else:
                warnings.append("LiDAR: no data could be loaded for this area.")
        except Exception as exc:
            warnings.append(f"LiDAR unavailable: {exc}")
    z0 = float(np.round(np.nanmedian(dem_raw.sample(arr[:, 0], arr[:, 1]))))
    dem = ShiftedDEM(dem_raw, z0)

    cl = net = None
    if not area:
        log("Building centerline ...")
        cl = build_centerline(coords, proj, dem, P["loop"], spacing=2.0, corner_smoothing=P["corner_smoothing"],
                              elev_smoothing=elev_smooth, start_offset=P["start_offset"], reverse=P["reverse"])
        warnings += profile_warnings(cl)
        m = P["margin"]
        bounds = (cl.p[:, 0].min() - m, cl.p[:, 1].min() - m, cl.p[:, 0].max() + m, cl.p[:, 1].max() + m)
    x0, y0, x1, y1 = bounds

    # ------------------------------------------------------------ aerial photo
    if P["ground"] not in ("auto", "landcover", "plain") and src is None:
        warnings.append(f"Unknown imagery source '{P['ground']}'.")
    pm = None
    if src:
        log(f"Aerial photo: {src['name']}")
        try:
            pm = f_photo.result()
            if pm.valid.mean() < 0.05:
                warnings.append("The aerial photo source returned no imagery for this area.")
                pm = None
            else:
                credits.append(src["attribution"])
        except Exception as exc:
            warnings.append(f"Aerial imagery failed ({exc}) - using land cover colours.")
            pm = None

    parts = []
    wall_extent = 0.0
    sp = P["terrain_spacing"]
    if area:
        # ------------------------------------------------------------ every street of the area
        log("Streets of the area ...")
        try:
            data0 = f_scenery.result()
        except osm.OSMError as exc:
            raise ValueError(f"No street data for this area: {exc}")
        try:
            turning = f_furn.result().get("turning", [])
        except Exception:
            turning = []
        # street widths / markings: the photo at full resolution (a big area's photo map is coarser)
        pm_st = pm
        if pm is not None and pm.res > 0.3 and src is not None:
            pm_st = photo.TiledPhoto(src, proj, res=0.25)
        net = network_mod.build(data0["roads"], proj, bounds, dem, P, pm_st, log=log, warnings=warnings,
                                turning=turning, clip=clip)
        network_mod.markings(net, P, pm_st, log=log, warnings=warnings)
        pm_st = None
        hl = hr = conf = None
        mk = None
        nm = sum(c.measured for c in net.chains)
        width_info = {"used": nm > 0, "streets": len(net.chains), "measured": nm,
                      "median": round(float(np.median([np.median(c.hl + c.hr) for c in net.chains])), 1)}
        wmean = width_info["median"]
        tm = network_mod.NetTerrain(proj, dem, net, extra=0.5 + 1.5 * sp, blend=max(25.0, 2 * sp))
    else:
        # ------------------------------------------------------------ road width
        hl, hr, conf, width_info = road_widths(P, cl, proj, pm, log, warnings)
        widths = (hl, hr)
        wmean = float(np.mean(hl + hr))
        warnings += corner_warnings(cl, hl, hr)

        # the road is part of the ground (one continuous surface, built below); only the markings are a mesh
        log("Road surface ...")
        mk = road_markings(P, cl, hl, hr, pm, log, warnings)
        parts += markings_mod.meshes(mk, cl, hl, hr, P["camber"])
        if P["walls"] == "concrete":
            log("Walls ...")
            parts += meshes.build_walls(cl, widths, P["wall_offset"], P["wall_height"])
            wall_extent = P["wall_offset"] + 0.6

        extra = max(P["shoulder"], wall_extent + 0.5) + 1.5 * sp
        tm = meshes.TerrainModel(proj, dem, cl, extra=extra, half_width=np.maximum(hl, hr), blend=max(25.0, 2 * sp))

    # ------------------------------------------------------------ OSM / Overture scenery + land cover
    scenery_xy = {"roads": [], "buildings": []}
    n_bld = n_side = 0
    auto = []
    data = {"buildings": [], "roads": [], "trees": [], "areas": []}
    if f_scenery is not None:
        try:
            data = f_scenery.result()
        except osm.OSMError as exc:
            warnings.append(str(exc))
    walls, tyre_info = tyre_walls(P, cl, hl, hr, pm, proj, data["roads"], log, warnings)
    lc = None
    if P["trees"] or use_ground:
        lc_feats = []
        if f_lc is not None:
            try:
                lc_feats = f_lc.result()
            except Exception as exc:
                warnings.append(f"Land cover (Overture Maps) failed: {exc} - using OpenStreetMap areas only.")
        lc = landcover.build_landcover(bounds, proj, data, lc_feats)
        credits += lc.sources

    # ------------------------------------------------------------ terrain + ground textures
    textures = make_textures()
    painted_ground = False
    deferred_ground = False          # 'land cover colours' with a photo: painted after the photo analysis
    street_lines = []
    if P["side_roads"] and not area:
        for r in data["roads"]:
            if r["highway"] in meshes.SIDE_ROAD_SKIP or r.get("tunnel"):
                continue
            xy = proj.latlngs_to_xy(r["coords"])
            if len(xy) >= 2:
                for piece in ground_mod.off_track_pieces(xy, r["width"] / 2, cl, tm):
                    street_lines.append((piece, r["width"] / 2))
    tiles = ground_mod.ground_tiles(bounds)
    ground_keep = None               # None = the whole area is ground
    if area:
        from shapely.geometry import box

        log("Terrain ... (one surface with the streets)")
        if P["terrain"]:
            tm.tin = ground_mod.build_tin_net(tm, net, bounds, sp, near=P["near_ground"], log=log, keep=clip)
            ground_keep = clip
        else:
            strip = 3.0
            tm.tin = ground_mod.build_tin_net(tm, net, bounds, sp, band=strip + ground_mod.FINE, far_points=False,
                                              log=log, keep=clip)
            ground_keep = net.corridor(strip + 2.0).intersection(clip)
        tiles = [t for t in tiles if ground_keep.intersects(box(t["x0"], t["y0"], t["x1"], t["y1"]))]
    elif P["terrain"]:
        log("Terrain ... (one surface with the road)")
        tm.tin = ground_mod.build_tin(tm, cl, hl, hr, P["camber"], bounds, sp, street_lines,
                                      shoulder=P["shoulder"], near=P["near_ground"], log=log)
    else:
        # terrain off: only a strip of ground along the road (shoulder / barriers + 3 m) and the streets
        from shapely.geometry import LineString
        from shapely.ops import unary_union

        log("Ground strip along the road ... (terrain off)")
        strip = max(P["shoulder"], wall_extent) + 3.0
        tm.tin = ground_mod.build_tin(tm, cl, hl, hr, P["camber"], bounds, sp, street_lines,
                                      band=strip + ground_mod.FINE, far_points=False, shoulder=P["shoulder"],
                                      log=log)
        ground_keep = unary_union([meshes.corridor_polygon(cl, hl + strip, hr + strip)]
                                  + [LineString(xy).buffer(h + 2.0, cap_style=2) for xy, h in street_lines]).buffer(0)
        from shapely.geometry import box

        tiles = [t for t in tiles if ground_keep.intersects(box(t["x0"], t["y0"], t["x1"], t["y1"]))]
    res = max(P["ground_res"], 0.1)
    if use_ground:
        photo_ground = pm is not None and P["ground"] not in ("landcover", "plain")
        painted_ground = not photo_ground
        deferred_ground = painted_ground and pm is not None
        if not deferred_ground:
            log("Ground textures ... (" + ("aerial photo" if photo_ground else "land cover colours") + ")")
        def ground_tile(t, photo_ground=photo_ground, res=res):
            wpx = int(min(np.ceil((t["x1"] - t["x0"]) / res), 4096))
            hpx = int(min(np.ceil((t["y1"] - t["y0"]) / res), 4096))
            xs = np.linspace(t["x0"], t["x1"], wpx)
            ys = np.linspace(t["y1"], t["y0"], hpx)
            if pm is not None and not photo_ground:
                rgb = landcover.photo_tinted_ground(pm, lc, xs, ys).astype(np.float32)
            else:
                f = max(1, int(round(1.0 / res)))
                small = landcover.synth_ground(lc, xs[::f], ys[::f])
                rgb = np.asarray(Image.fromarray(small).resize((wpx, hpx), Image.BICUBIC), dtype=np.float32)
                if photo_ground:
                    ph, ok = pm.grid(xs, ys)
                    rgb = np.where(ok[..., None], ph, rgb)
            return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

        todo = [] if deferred_ground else tiles
        for t, img in zip(todo, parallel.pmap(ground_tile, todo)):      # one tile per core
            textures[t["material"]] = img
    else:
        log("Ground textures ... (plain grass)")
        for t, img in zip(tiles, parallel.pmap(lambda t: ground_mod.plain_tile(t, res), tiles)):
            textures[t["material"]] = img
    ground_tiles = tiles

    # ------------------------------------------------------------ streets & buildings
    street_geom = None
    if P["side_roads"]:             # streets are painted into the ground and their triangles are ROAD
        from shapely.geometry import LineString
        from shapely.ops import unary_union

        geoms = [LineString(xy).buffer(h, cap_style=2, join_style=1) for xy, h in street_lines]
        street_geom = unary_union(geoms) if geoms else None
        n_side = len(geoms)
    for r in data["roads"]:
        scenery_xy["roads"].append({"xy": proj.latlngs_to_xy(r["coords"]), "width": r["width"]})
    from shapely.geometry import Point

    bld_polys = []
    in_box = []
    for b in data["buildings"]:
        c = proj.latlngs_to_xy(b["outer"][0]).mean(0)
        if x0 < c[0] < x1 and y0 < c[1] < y1 and (clip is None or clip.contains(Point(c))):
            in_box.append(b)
    bstats = {}
    det_buildings = []
    if P["buildings"] and P["photo_buildings"] and (pm is not None or chm is not None):
        log("Finding buildings on the photo" + (" and LiDAR" if chm is not None else "") + " ...")
        from shapely import affinity
        from shapely.geometry import LineString

        try:
            known, owner = [], []
            for bi, b in enumerate(in_box):
                for poly in meshes._polygons_for_building(b, proj):
                    known.append(poly)
                    owner.append(bi)
            corridor_b = net.corridor(0.5) if area else \
                meshes.corridor_polygon(cl, hl + P["shoulder"] + 0.5, hr + P["shoulder"] + 0.5)
            street_b = []
            for r in data["roads"]:
                if r["highway"] in meshes.SIDE_ROAD_SKIP:
                    continue
                xy = proj.latlngs_to_xy(r["coords"])
                if len(xy) >= 2:
                    street_b.append(LineString(xy).buffer(r["width"] / 2 + 0.8))
            hint = None
            if len(known) < 5 and chm is None and pm is not None and not config.OFFLINE:
                # nothing mapped to learn roof colours from: borrow Overture footprints (training only)
                try:
                    tmp = {"buildings": []}
                    overture.merge_into(tmp, sbox, log=log)
                    hint = [q for b in tmp["buildings"] for q in meshes._polygons_for_building(b, proj)]
                except Exception as exc:
                    log(f"  (Overture footprints unavailable for training: {exc})")
            found = buildings_mod.detect(pm, chm, proj, bounds, known, corridor_b, street_b, log=log,
                                         train_hint=hint, res=an_res)
            if clip is not None:
                found["new"] = [f for f in found["new"] if clip.contains(f[0].representative_point())]
            # snap mapped outlines + LiDAR heights back onto the building records
            moved, lh = set(), 0
            for kp, new_p, bi, h in zip(known, found["known"], owner, found["known_h"]):
                b = in_box[bi]
                if new_p is not kp and bi not in moved:
                    dx = new_p.centroid.x - kp.centroid.x
                    dy = new_p.centroid.y - kp.centroid.y

                    def _mv(ring, dx=dx, dy=dy):
                        xy = proj.latlngs_to_xy(ring)
                        la, lo = proj.to_latlon(xy[:, 0] + dx, xy[:, 1] + dy)
                        return list(zip(la.tolist(), lo.tolist()))
                    b = dict(b, outer=[_mv(r) for r in b["outer"]], inner=[_mv(r) for r in b.get("inner", [])])
                    in_box[bi] = b
                    moved.add(bi)
                if h is not None:
                    in_box[bi] = dict(in_box[bi], eave=h[0], height=h[1])
                    lh += 1
            for k, (poly, h, src) in enumerate(found["new"]):
                xy = np.asarray(poly.exterior.coords)
                la, lo = proj.to_latlon(xy[:, 0], xy[:, 1])
                fp_m2 = poly.area                       # (not "area": that is the area-mode flag)
                nb = {"id": f"d{k}", "outer": [list(zip(la.tolist(), lo.tolist()))], "inner": [],
                      "tags": {"building": "house" if fp_m2 < 250 else "yes", "source": src},
                      "height": h[1] if h else buildings_mod.default_height(fp_m2)}
                if h:
                    nb["eave"] = h[0]
                in_box.append(nb)
                det_buildings.append(poly)
            hs = [f[1] for f in found["new"] if f[1]]
            if hs:
                log(f"  LiDAR heights: eave median {np.median([h[0] for h in hs]):.1f} m, "
                    f"ridge median {np.median([h[1] for h in hs]):.1f} m")
            bstats = {"found_photo": sum(1 for f in found["new"] if f[2] == "photo"),
                      "found_lidar": sum(1 for f in found["new"] if f[2] == "lidar"),
                      "snapped": len(moved), "lidar_heights": lh}
            log(f"  {len(found['new'])} new buildings ({bstats['found_lidar']} LiDAR, {bstats['found_photo']} photo), "
                f"{len(moved)} mapped outlines snapped to the photo, {lh} real heights from LiDAR")
        except Exception as exc:
            warnings.append(f"Building detection failed: {exc}")
    if P["buildings"]:
        log("Buildings ...")
        if area:
            auto = overlapping_buildings(in_box, proj, net.corridor(1.0))
        else:
            auto = meshes.classify_buildings(in_box, proj, cl, hl + P["shoulder"] + 1.0, hr + P["shoulder"] + 1.0)
        skip = (set(auto) | set(P["force_exclude"])) - set(P["force_include"])
        if pm is not None:
            _roof_colours(in_box, skip, proj, pm)
        # footprints made buildable now (rectangles / squared outlines): the photo analysis, driveways and
        # trees use them; the houses themselves are built after the driveways are known (garage doors)
        bplans = meshes.plan_buildings(in_box, proj, skip)
        bld_polys += meshes.footprints(bplans)
        for b in in_box:
            if b["id"] not in skip:
                scenery_xy["buildings"].append(proj.latlngs_to_xy(b["outer"][0]))

    # ------------------------------------------------------------ photo analysis
    detect = {"paved": []}
    stats_photo = {}
    an = None
    streets = landcover.street_points(data["roads"], proj, bounds)
    if pm is not None and (P["photo_trees"] or P["photo_paved"] or P["grass"] or deferred_ground):
        log("Reading the photo (vegetation, paved areas) ...")
        from shapely.geometry import LineString

        an0 = photo.Grid(pm, an_res)
        block = an0.mask_from_polys(bld_polys, buffer=1.0)
        corridor = net.corridor(0.5) if area else \
            meshes.corridor_polygon(cl, hl + P["shoulder"] + 0.5, hr + P["shoulder"] + 0.5)
        street_geoms = [corridor]
        for r in data["roads"]:
            if r["highway"] in meshes.SIDE_ROAD_SKIP:
                continue
            xy = proj.latlngs_to_xy(r["coords"])
            if len(xy) >= 2:
                street_geoms.append(LineString(xy).buffer(r["width"] / 2 + 0.8))
        street_mask = an0.mask_from_polys(street_geoms)
        chm_grid = None
        if chm is not None:
            rr, cc = np.mgrid[0:an0.H, 0:an0.W]
            gx, gy = an0.xy_of(rr.ravel(), cc.ravel())
            la_, lo_ = proj.to_latlon(gx, gy)
            chm_grid = chm.sample(la_, lo_).reshape(an0.H, an0.W)
        an = photo.Analysis(pm, an_res, block_mask=block, street_mask=street_mask, chm_grid=chm_grid)
        if P["photo_paved"] and ground_tiles:
            from shapely.ops import unary_union

            raw = an.polygons(an.paved, 12.0, 6000.0, simplify=0.4)
            # clean shapes joined to their street; no bush, car or shadow left in the middle
            # streets to join them to (not the parts the track runs along: the measured track is the edge there)
            if area:
                axes = driveways.Axes([(net.P, np.column_stack([net.N[:, 1], -net.N[:, 0]]), net.HL, net.HR)])
            else:
                lines = [(piece, r["width"] / 2) for r in data["roads"]
                         if r["highway"] not in meshes.SIDE_ROAD_SKIP and not r.get("tunnel") and len(r["coords"]) >= 2
                         for piece in ground_mod.off_track_pieces(proj.latlngs_to_xy(r["coords"]), r["width"] / 2,
                                                                  cl, tm)]
                axes = driveways.Axes.build(cl, hl, hr, lines)
            paved = driveways.regularize(raw, axes, unary_union(bld_polys) if bld_polys else None, log=log)
            if clip is not None:                       # nothing beyond the area's outline
                cut = [p.intersection(clip) for p in paved]
                paved = [q for p in cut for q in getattr(p, "geoms", [p]) if q.geom_type == "Polygon" and q.area > 4]
            pmask = an.mask_from_polys(paved) & an.valid
            an.paved = pmask
            an.canopy = an.canopy & ~pmask
            an.tree = an.canopy
            detect["paved"] = paved
            stats_photo["paved_areas"] = len(paved)
            stats_photo["paved_m2"] = round(float(sum(p.area for p in paved)))
            log(f"  paved areas (driveways, parking): {len(paved)}")
        # fewer, larger, simpler surfaces; grass tufts follow them (none on sand / gravel / driveways)
        landcover.tidy_masks(an)
        stats_photo["canopy_m2"] = round(float(an.canopy.sum() * an.res ** 2))
        stats_photo["wild_m2"] = round(float(an.wild.sum() * an.res ** 2))
        if deferred_ground:
            log("Ground textures ... (land cover colours, surfaces read on the photo)")
            landcover.stylised_ground(an, lc, tiles, textures, max(P["ground_res"], 0.1), log=log)
            deferred_ground = False
        if painted_ground and detect["paved"]:
            landcover.paint_paved(textures, ground_tiles, detect["paved"], pm)
        stats_photo["lawn_m2"] = round(float(an.lawn.sum() * an.res ** 2))

    if P["buildings"]:
        roads_xy = np.vstack([net.P if area else cl.p] + ([streets[0]] if len(streets[0]) else []))
        bparts, n_bld = meshes.build_buildings(in_box, proj, tm, skip, collide=P["buildings_collide"], plans=bplans,
                                               roads_xy=roads_xy, driveways=detect.get("paved"))
        parts += bparts
        textures.update(house_textures(photo_materials()))
        st_ = getattr(meshes.build_buildings, "last_stats", {})
        kinds = ", ".join(f"{v} {k}" for k, v in sorted(st_.get("kinds", {}).items(), key=lambda kv: -kv[1]))
        log(f"  {n_bld} buildings ({kinds}): {st_.get('doors', 0)} doors, {st_.get('garages', 0)} garage doors, "
            f"{st_.get('porches', 0)} porches, {st_.get('canopies', 0)} door canopies")

    if deferred_ground:                                   # photo analysis unavailable: older tinted look
        res = max(P["ground_res"], 0.1)

        def tinted(t):
            wpx = int(min(np.ceil((t["x1"] - t["x0"]) / res), 4096))
            hpx = int(min(np.ceil((t["y1"] - t["y0"]) / res), 4096))
            return Image.fromarray(landcover.photo_tinted_ground(pm, lc, np.linspace(t["x0"], t["x1"], wpx),
                                                                 np.linspace(t["y1"], t["y0"], hpx)), "RGB")

        for t, img in zip(tiles, parallel.pmap(tinted, tiles)):
            textures[t["material"]] = img

    # ------------------------------------------------------------ asphalt painted into the ground
    track_geom = net.asphalt if area else meshes.corridor_polygon(cl, hl, hr)
    if ground_tiles:
        layers = []
        if street_geom is not None:
            layers.append((street_geom, ground_mod.STREET_ASPHALT))
        if P["shoulder"] > 0:
            sh = meshes.corridor_polygon(cl, hl + P["shoulder"], hr + P["shoulder"]).difference(track_geom)
            layers.append((sh, ground_mod.SHOULDER_GRAVEL))
        layers.append((track_geom, ground_mod.TRACK_ASPHALT))
        ground_mod.paint_roads(textures, ground_tiles, layers)
        textures["road_paint"] = ground_mod.paint_texture()
        textures["road_paint_yellow"] = markings_mod.paint_texture("yellow")

    # ------------------------------------------------------------ ground photo materials (bake + close-up detail)
    multilayer = {}
    if ground_tiles:
        photos = photo_materials()
        need = ("asphalt", "grass", "gravel", "dirt")
        close = P["ml_ground"] and P["kn5"] and all(k in photos for k in need)
        if P["ml_ground"] and P["kn5"] and not close:
            log("  close-up ground detail needs the ambientCG photos - skipped this time")
        bake = bool(photos) and not photo_ground          # an aerial photo already has its own texture
        if bake or close:
            log("Ground materials ..." + (" (close-up detail for AC)" if close else ""))
            multilayer = groundmat.finish_ground(textures, ground_tiles, res, photos, bake=bake, close_up=close,
                                                 log=log)

    # ------------------------------------------------------------ street furniture and power lines
    n_props = {}
    lamp_lights, lamp_dist = [], None
    if want_props:
        log("Street furniture and power lines ...")
        furn = {}
        try:
            furn = f_furn.result()
        except Exception as exc:
            warnings.append(f"Street furniture download failed: {exc} - using rules only (no power lines, "
                            "no mapped traffic lights).")
        try:
            from shapely.geometry import LineString
            from shapely.ops import unary_union

            pstreets = street_geom
            if pstreets is None:
                g = [LineString(proj.latlngs_to_xy(r["coords"])).buffer(r["width"] / 2, cap_style=2)
                     for r in data["roads"] if r["highway"] not in meshes.SIDE_ROAD_SKIP and len(r["coords"]) >= 2]
                pstreets = unary_union(g) if g else None
            # buildings and driveways: no pole or sign stands on them
            keep_off = list(bld_polys) + list(detect.get("paved") or [])
            bunion = unary_union(keep_off).buffer(0.5) if keep_off else None
            style = props_mod.sign_style(proj.lat0, proj.lon0, P["sign_style"])
            scene = props_mod.Scene(tm, max(1.2, wall_extent + 0.6), pstreets, bunion)
            pparts, ptex, n_props = props_mod.build(
                data["roads"], furn, proj, scene, bounds,
                {"power_lines": P["power_lines"], "road_signs": P["road_signs"], "street_lights": P["street_lights"],
                 "substations": P["power_lines"], "crosswalks": P["crosswalks"], "guardrails": P["guardrails"]},
                style, log=log, keep=clip)
            parts += pparts
            textures.update(ptex)
            if any(pt.material == "road_paint" for pt in pparts) and "road_paint" not in textures:
                textures["road_paint"] = ground_mod.paint_texture()
            n_props["sign_style"] = style
            lamp_lights = scene.lights
            if lamp_lights:
                L = np.asarray(lamp_lights)
                d_, _z, hw_ = tm.road_info3(L[:, 0], L[:, 1])
                lamp_dist = d_ - hw_
        except Exception as exc:
            warnings.append(f"Street furniture failed: {exc}")

    # ------------------------------------------------------------ tyre walls (after the ground: they stand on it)
    if walls:
        parts += tyres_mod.meshes(walls, tm)
        rubber = photo_materials().get("rubber")
        textures["tyres"] = tyres_mod.texture(rubber)
        if rubber is not None:
            credits.append(materials.EXTRA_CREDIT["rubber"])

    # ------------------------------------------------------------ trees
    n_trees = 0
    csp_trees = None
    tree_src = {}
    if P["trees"] and lc is not None:
        log("Trees ...")
        clear = max(P["shoulder"], wall_extent) + 2.5
        sets = []
        covered = None
        if an is not None and (P["photo_trees"] or chm is not None):
            # canopy read from the photo (+ LiDAR heights): every tree, bush and hedge where it really is
            tops = lidar.chm_treetops(chm, proj, bounds) if chm is not None else None
            t_, tinfo = photo.trees_from_canopy(an, tops)
            ok = landcover.tree_ok(t_["x"], t_["y"], lc, tm, clear, streets)
            t_ = {k: v[ok] for k, v in t_.items()}
            sets.append(t_)
            n_hedge = int(ok[len(ok) - tinfo["hedge_bushes"]:].sum()) if tinfo["hedge_bushes"] else 0
            tree_src["lidar" if chm is not None else "photo"] = int(ok.sum()) - n_hedge
            if n_hedge:
                tree_src["hedges"] = n_hedge
            covered = lambda x, y: an.valid[an.rc_of(x, y)]
        elif chm is not None:
            tx, ty, th = lidar.chm_treetops(chm, proj, bounds)
            if len(tx):
                conifer = np.random.default_rng(8).random(len(tx)) < P["conifer_ratio"]
                t_ = photo.trees_from_treetops(tx, ty, th, conifer)
                ok = landcover.tree_ok(t_["x"], t_["y"], lc, tm, clear, streets)
                t_ = {k: v[ok] for k, v in t_.items()}
                sets.append(t_)
                tree_src["lidar"] = int(ok.sum())
                covered = lambda x, y: np.isfinite(chm.sample(*proj.to_latlon(x, y)))
        rest = landcover.scatter_trees(lc, tm, cl, bounds, clear, streets,
                                       [] if covered is not None else data.get("trees", []), proj,
                                       density=P["tree_density"], conifer_ratio=P["conifer_ratio"], covered=covered)
        tree_src["land_cover"] = int(len(rest["x"]))
        sets.append(rest)
        trees = landcover.merge_trees(sets, tm, P["max_trees"])
        if clip is not None and len(trees["x"]):         # no tree where no ground is built
            from shapely import contains_xy

            inside = contains_xy(clip.buffer(-0.5), trees["x"], trees["y"])
            trees = {k: (v[inside] if hasattr(v, "__len__") and len(v) == len(inside) else v) for k, v in trees.items()}
        if walls and len(trees["x"]):
            # no tree in a tyre wall
            from scipy.spatial import cKDTree as _KD

            dt, _ = _KD(tyres_mod.stacks(walls)).query(np.column_stack([trees["x"], trees["y"]]))
            keep_t = dt > np.maximum(0.6, 0.5 * trees["w"])
            trees = {k: (v[keep_t] if hasattr(v, "__len__") and len(v) == len(keep_t) else v) for k, v in trees.items()}
        if detect["paved"] and len(trees["x"]):
            # nothing grows on a driveway (land-cover trees and LiDAR tops included)
            from shapely import contains_xy
            from shapely.ops import unary_union

            onp = contains_xy(unary_union(detect["paved"]).buffer(0.5), trees["x"], trees["y"])
            if onp.any():
                trees = {k: (v[~onp] if hasattr(v, "__len__") and len(v) == len(onp) else v) for k, v in trees.items()}
        n_trees = len(trees["x"])
        parts += landcover.tree_meshes(trees, tm)
        if P["csp_trees"] and n_trees:
            models = csp.find_tree_models(log=log)
            if models:
                csp_trees = (trees, tm.surface_height(trees["x"], trees["y"]), models)
            else:
                log("  CSP trees: no .bin tree models in csp_trees/ - skipped (see csp_trees/README.txt)")
        textures["trees"] = landcover.tree_atlas()
        scenery_xy["trees"] = np.column_stack([trees["x"], trees["y"], trees["w"]]) if n_trees else []
        log(f"  {n_trees:,} trees ({', '.join(f'{v:,} from {k}' for k, v in tree_src.items())})")

    # ------------------------------------------------------------ grass
    n_grass = {}
    if P["grass"] and (an is not None or lc is not None):
        log("Grass ...")
        if an is not None:
            g = grass_mod.scatter(an, tm, P["shoulder"], streets, P["grass_density"], P["max_grass"],
                                  P["max_tall_grass"], wild=P["wild_grass"])
        else:
            g = grass_mod.scatter_landcover(lc, tm, bounds, P["shoulder"], streets, P["grass_density"],
                                            P["max_grass"], P["max_tall_grass"], wild=P["wild_grass"])
        if clip is not None and len(g["x"]):
            from shapely import contains_xy

            inside = contains_xy(clip.buffer(-0.3), g["x"], g["y"])
            g = {k: (v[inside] if hasattr(v, "__len__") and len(v) == len(inside) else v) for k, v in g.items()}
        if len(g["x"]):
            parts += grass_mod.meshes(g, tm)
            textures["veg_grass"] = grass_mod.atlas()
            textures["veg_grass_short"] = textures["veg_grass"]
        n_grass = {"lawn": int((g["kind"] == grass_mod.LAWN).sum()), "tall": int((g["kind"] == grass_mod.WILD).sum())}
        log(f"  {n_grass['lawn']:,} grass tufts, {n_grass['tall']:,} tall grass clumps")

    # ------------------------------------------------------------ write
    from shapely.ops import unary_union

    an = None                                             # the photo analysis is done: free its grids
    import gc

    gc.collect()

    drivable = [track_geom] + ([street_geom] if street_geom is not None else []) + list(detect.get("paved") or [])
    asphalt_geom = unary_union([track_geom] + ([street_geom] if street_geom is not None else [])).buffer(0)
    shoulder_geom = None
    if P["shoulder"] > 0:
        shoulder_geom = meshes.corridor_polygon(cl, hl + P["shoulder"], hr + P["shoulder"]).difference(track_geom)
    # close to the car: photo asphalt / gravel repeated every few metres (sharp), not the 25 cm ground texture;
    # the ground within near_ground m of the track: high-resolution tiles with the photo grain
    photos = photo_materials()
    near = None
    if P["near_ground"] > 0 and ground_tiles:
        from shapely.geometry import box as _box

        ngeom = (net.corridor(P["near_ground"]) if area else
                 meshes.corridor_polygon(cl, hl + P["near_ground"], hr + P["near_ground"])).intersection(_box(*bounds))
        if ground_keep is not None:
            ngeom = ngeom.intersection(ground_keep)
        classify, near_colours, outline = groundmat.near_regions(
            textures, ground_tiles, ngeom, paved=list(detect.get("paved") or []),
            exclude=unary_union([asphalt_geom] + ([shoulder_geom] if shoulder_geom is not None else [])), log=log,
            res=an_res)
        if len(outline):
            tm.tin = tm.tin.with_points(outline, tm.height(outline[:, 0], outline[:, 1]))
        near = (ngeom, classify)
    if area:                                              # lines laid on the finished surface
        parts += network_mod.marking_meshes(net, tm)
    parts += ground_mod.ground_meshes(tm.tin, tiles, unary_union(drivable).buffer(0), keep=ground_keep,
                                      asphalt_geom=asphalt_geom, shoulder_geom=shoulder_geom, near=near)
    textures["asphalt_tiled"] = ground_mod.tiled_texture(photos.get("asphalt"), ground_mod.TRACK_ASPHALT, 1024,
                                                         contrast=1.0, seed=3)
    textures["gravel_tiled"] = ground_mod.tiled_texture(photos.get("gravel"), ground_mod.SHOULDER_GRAVEL, 1024,
                                                        contrast=0.9, seed=4)
    if near is not None:
        textures.update(groundmat.near_textures(near_colours, photos))
        n_near = sum(1 for p in parts if p.name.startswith(("1GRASS_NEAR", "1ROAD_NEAR")))
        log(f"  ground beside the track: tiled photo grass / soil / gravel ({n_near} meshes, "
            f"{', '.join(sorted(near_colours))})")

    log("Writing 3D model ...")
    tex_dir = bl / "texture"
    tex_dir.mkdir(exist_ok=True)
    used = {p.material for p in parts}

    def encode(k):
        """PNG once (texture folder + KN5), JPEG for big ground tiles inside the preview .glb."""
        img = textures[k]
        buf = io.BytesIO()
        img.save(buf, "PNG", compress_level=6)
        png = buf.getvalue()
        (tex_dir / f"{k}.png").write_bytes(png)
        glb_img = img
        if k.startswith("ground_"):
            jb = io.BytesIO()
            img.save(jb, format="JPEG", quality=88)
            jb.seek(0)
            glb_img = Image.open(jb)
            glb_img.load()
        # in the .kn5: opaque textures as DDS/BC1 (8x less video memory than PNG); alpha cut-outs stay PNG
        dds = ac_export.dds_bc1(img) if (P["kn5"] and P["kn5_dds"] and img.mode == "RGB") else None
        return k, png, glb_img, dds

    extra = {k for m in multilayer.values() for k in m["textures"].values()}
    names = [k for k in textures if k in used or k in extra]
    encoded = parallel.pmap(encode, names)                # Pillow encodes on all cores
    png_bytes = {x[0]: x[1] for x in encoded}
    glb_tex = {x[0]: x[2] for x in encoded if x[0] in used}
    dds_bytes = {x[0]: x[3] for x in encoded if x[3] is not None}
    ac_export.write_glb(parts, glb_tex, bl / "track.glb", origin_shift=np.zeros(3))

    if area:
        scl, shl, shr, s0, _ = network_mod.spawn_street(net, proj, P, warnings)
        objs, n_spots = ac_export.spawn_objects(scl, shl, shr, s0, P["grid"], P["pits"])
        if n_spots < max(P["grid"], P["pits"]):
            warnings.append(f"The start street is short: {n_spots} start / pit spots fit on it.")
        for o in objs:                                    # exactly on the built surface
            o["pos"][2] = round(float(tm.surface_height(o["pos"][0], o["pos"][1])) + 0.3, 3)
        npits = min(max(P["pits"], 1), n_spots)
    else:
        objs = ac_export.ac_objects(cl, wmean, P["grid"], P["pits"], widths=(hl, hr), camber=P["camber"])
        npits = max(P["pits"], 1)
    (bl / "ac_objects.json").write_text(json.dumps(objs, indent=1), encoding="utf-8")
    # the driving line for other sims (rFactor 2: AI waypoints, sectors, cameras): the track, or in area mode the
    # street the cars start on
    _l_cl, _l_hl, _l_hr = (scl, shl, shr) if area else (cl, hl, hr)
    (bl / "track_line.json").write_text(json.dumps({
        "mode": "area" if area else "route", "loop": bool(_l_cl.loop), "length": float(_l_cl.length),
        "p": np.round(_l_cl.p, 3).tolist(), "z": np.round(_l_cl.z, 3).tolist(), "s": np.round(_l_cl.s, 3).tolist(),
        "hl": np.round(np.asarray(_l_hl, float), 3).tolist(), "hr": np.round(np.asarray(_l_hr, float), 3).tolist(),
        "shoulder": float(P["shoulder"]), "walls": P["walls"], "wall_offset": float(P["wall_offset"]),
        "lat": proj.lat0, "lon": proj.lon0, "z0": z0, "name": P["name"], "city": P["city"], "country": P["country"],
        "author": P["author"], "total_length": float(net.length) if area else float(cl.length),
        # every street of an area (menu icons)
        "map": [np.round(xy[::max(1, len(xy) // 200)], 1).tolist() for xy, _w in net.lines()] if area else [],
    }), encoding="utf-8")

    log("Writing Assetto Corsa files ...")
    if area:
        prev = ac_export.write_preview(scl, wmean, scenery_xy, out / "preview.png", lines=net.lines(), bounds=bounds,
                                       start_s=s0)
    else:
        prev = ac_export.write_preview(cl, wmean, scenery_xy, out / "preview.png")
    info = {"name": P["name"], "lat": proj.lat0, "lon": proj.lon0, "author": P["author"], "city": P["city"],
            "country": P["country"], "clockwise": False if area else ac_export.clockwise(cl), "credits": credits}
    if area:
        info["length"] = net.length
    ac_export.write_ac_folder(acf, slug, info, scl if area else cl, wmean, prev, npits,
                              lines=net.lines() if area else None)
    if not area:                                          # AI line + pit lane (free-roam areas have no race line)
        try:
            ai_line.write_ai_lines(acf / "ai", {"p": cl.p, "z": cl.z, "hl": hl, "hr": hr, "loop": cl.loop}, objs,
                                   log=log)
        except Exception as exc:
            warnings.append(f"AI line not written: {exc}")
    if csp.write_ext_config(acf, {pt.material for pt in parts}, grass_fx=P["grass_fx"] and P["grass"],
                            density=P["grass_density"], trees=csp_trees, log=log,
                            lights=csp.pick_lights(lamp_lights, lamp_dist) if P["csp_lights"] else None,
                            rain=P["csp_rain"]) and P["grass_fx"] and P["grass"]:
        log("  CSP Grass FX: 3D grass on lawns and wild grass (extension/ext_config.ini)")
    ac_export.write_blender_kit(bl, slug, acf)
    kn5_info = None
    if P["kn5"]:
        log(f"Writing {slug}.kn5 ...")
        try:
            kn5_info = ac_export.write_kn5(parts, textures, objs, acf / f"{slug}.kn5", log=log, png_bytes=png_bytes,
                                           dds_bytes=dds_bytes, multilayer=multilayer)
        except Exception as exc:
            warnings.append(f"KN5 export failed: {exc}")

    report = {
        "slug": slug,
        "name": P["name"],
        **(area_overlay(net, proj, z0, objs, width_info) if area else
           road_overlay(cl, hl, hr, conf, proj, z0, objs, width_info)),
        "buildings": n_bld,
        "trees": n_trees,
        "grass": n_grass,
        "tree_sources": tree_src,
        "photo": stats_photo,
        "lidar": chm is not None or isinstance(dem_raw, lidar.CompositeDEM),
        "hrdem": any(c == hrdem.ATTRIBUTION for c in credits),
        "credits": credits,
        "buildings_auto_excluded": auto,
        **({} if area else {"streets": n_side}),        # area mode: "streets" is the list of streets (overlay)
        "props": n_props,
        "meshes": [{"name": p.name, "material": p.material, "verts": p.nverts, "tris": len(p.faces)} for p in parts],
        "total_verts": int(sum(p.nverts for p in parts)),
        "total_tris": int(sum(len(p.faces) for p in parts)),
        "detections": {"paved": _ll_polys(detect["paved"], proj), "buildings": _ll_polys(det_buildings, proj)},
        "markings": area_markings(net, proj) if area else {"info": mk.info, "lines": markings_mod.overlay(mk, cl, proj)},
        "tyres": {"info": tyre_info, "lines": tyres_mod.overlay(walls, proj)},
        "building_stats": bstats,
        "kn5": kn5_info,
        "warnings": warnings,
        "elapsed_s": round(time.time() - t0, 1),
        "files": {
            "glb": f"/output/{slug}/blender/track.glb",
            "preview": f"/output/{slug}/preview.png",
            "folder": str(out),
            "kn5": str(acf / f"{slug}.kn5") if kn5_info else None,
        },
        "origin": {"lat": proj.lat0, "lon": proj.lon0, "z0": z0},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(f"Done in {report['elapsed_s']} s - {report['total_verts']:,} vertices in {len(parts)} meshes.")
    return report

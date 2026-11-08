"""Extra building footprints from Overture Maps (OSM + Microsoft/Google/Esri AI-detected footprints).

Used to fill areas where OpenStreetMap has no buildings mapped. Data is ODbL, read anonymously
from Overture's public S3 bucket with the `overturemaps` package.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import config
from .osm import building_height


class OvertureError(Exception):
    pass


def _rings(geom):
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom] if geom.geom_type == "Polygon" else []
    out = []
    for p in polys:
        outer = [(round(y, 7), round(x, 7)) for x, y in p.exterior.coords]
        inner = [[(round(y, 7), round(x, 7)) for x, y in r.coords] for r in p.interiors]
        out.append((outer, inner))
    return out


STAC_CATALOG_URL = "https://stac.overturemaps.org/catalog.json"


def _use_certifi():
    """overturemaps uses urllib, which on Windows trusts the system certificate store. Outdated or
    expired roots there cause 'CERTIFICATE_VERIFY_FAILED: certificate has expired'. Use the same
    up-to-date certifi bundle that requests uses for OSM/OSRM instead."""
    import ssl

    try:
        import certifi
    except ImportError:
        return
    cafile = certifi.where()
    ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(cafile=cafile)


def _latest_release() -> str:
    import requests

    try:
        r = requests.get(STAC_CATALOG_URL, headers={"User-Agent": config.USER_AGENT}, timeout=20)
        r.raise_for_status()
        latest = r.json().get("latest")
        if latest:
            return latest
    except Exception as exc:
        raise OvertureError(f"Could not reach Overture Maps catalog: {exc}") from exc
    raise OvertureError("Overture Maps catalog did not list a release.")


def _batches(overture_type, bbox, log):
    """Yield pyarrow record batches of an Overture type inside bbox=(s,w,n,e)."""
    s, w, n, e = bbox
    try:
        from overturemaps import core
    except ImportError as exc:
        raise OvertureError("Package 'overturemaps' is not installed (run start.bat again).") from exc
    _use_certifi()
    release = _latest_release()
    log(f"  Overture release {release}, layer '{overture_type}'")
    reader = None
    for stac in (True, False):
        try:
            reader = core.record_batch_reader(overture_type, bbox=(w, s, e, n), release=release, stac=stac,
                                              connect_timeout=20, request_timeout=60)
            break
        except Exception as exc:  # STAC index unavailable -> plain scan
            last = exc
            reader = None
    else:
        raise OvertureError(f"Overture Maps unavailable: {last}")
    if reader is None:
        return
    yield from reader


def _col(batch, name):
    return batch.column(name).to_pylist() if name in batch.schema.names else [None] * batch.num_rows


def _download(bbox, log):
    import shapely

    log("Downloading extra building footprints from Overture Maps (can take up to a minute) ...")
    feats = []
    for batch in _batches("building", bbox, log):
        geoms = shapely.from_wkb(batch.column("geometry").to_pylist())
        for g, i, hh, ff, uu, ss in zip(geoms, _col(batch, "id"), _col(batch, "height"), _col(batch, "num_floors"),
                                        _col(batch, "is_underground"), _col(batch, "sources")):
            if g is None or uu:
                continue
            dataset = ""
            if ss:
                dataset = ",".join(sorted({str(x.get("dataset")) for x in ss if x and x.get("dataset")}))
            feats.append({"id": str(i), "rings": _rings(g), "height": hh, "floors": ff, "dataset": dataset})
    return feats


def fetch_land_cover(bbox, log=print):
    """Overture land_cover polygons (vectorised ESA WorldCover 10 m): [{'subtype', 'rings'}]. Cached."""
    import shapely
    from shapely.geometry import box

    key = hashlib.sha1(("lc" + ",".join(f"{v:.6f}" for v in bbox)).encode()).hexdigest()[:16]
    cache = config.CACHE_DIR / "overture" / f"landcover_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    log("Downloading land cover (Overture Maps / ESA WorldCover) ...")
    s, w, n, e = bbox
    clip = box(w, s, e, n)
    feats = []
    for batch in _batches("land_cover", bbox, log):
        geoms = shapely.from_wkb(batch.column("geometry").to_pylist())
        for g, st, mz in zip(geoms, _col(batch, "subtype"), _col(batch, "min_zoom")):
            if g is None or not st:
                continue
            g = g.intersection(clip)
            if g.is_empty:
                continue
            g = g.simplify(0.00002)
            feats.append({"subtype": st, "min_zoom": mz, "rings": _rings(g)})
    # Overture ships several generalisations (by min_zoom); keep the most detailed one.
    zooms = [f["min_zoom"] for f in feats if f["min_zoom"] is not None]
    if zooms:
        top = max(zooms)
        feats = [f for f in feats if f["min_zoom"] is None or f["min_zoom"] == top]
    if feats:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(feats), encoding="utf-8")
    return feats


def fetch(bbox, log=print):
    """Return Overture building features for bbox=(s,w,n,e), cached on disk."""
    key = hashlib.sha1(",".join(f"{v:.6f}" for v in bbox).encode()).hexdigest()[:16]
    cache = config.CACHE_DIR / "overture" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    feats = _download(bbox, log)
    if feats:  # don't cache empty results - they may come from a failed download
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(feats), encoding="utf-8")
    return feats


def merge_into(data: dict, bbox, log=print) -> int:
    """Add Overture buildings that don't overlap an existing OSM building. Returns number added."""
    from shapely.geometry import Point, Polygon
    from shapely.strtree import STRtree

    feats = fetch(bbox, log)
    osm_polys = []
    for b in data["buildings"]:
        ring = b["outer"][0]
        if len(ring) >= 4:
            osm_polys.append(Polygon([(lon, lat) for lat, lon in ring]).buffer(0))
    tree = STRtree(osm_polys) if osm_polys else None
    added = 0
    for f in feats:
        for k, (outer, inner) in enumerate(f["rings"]):
            if len(outer) < 4:
                continue
            arr = np.asarray(outer)
            c = Point(arr[:, 1].mean(), arr[:, 0].mean())
            if tree is not None and len(tree.query(c, predicate="within")):
                continue  # already in OSM
            # footprint area (m^2) -> small ones are houses/sheds, which are lower than the default 8 m
            lat0 = np.radians(arr[:, 0].mean())
            px, py = arr[:, 1] * 111320 * np.cos(lat0), arr[:, 0] * 110540
            area = 0.5 * abs(np.dot(px, np.roll(py, 1)) - np.dot(py, np.roll(px, 1)))
            kind = "shed" if area < 40 else "house" if area < 250 else "yes"
            tags = {"building": kind, "source": f.get("dataset") or "overture"}
            if f.get("floors"):
                tags["building:levels"] = str(f["floors"])
            if f.get("height"):
                tags["height"] = str(f["height"])
            bid = f"o{f['id']}" + (f"_{k}" if k else "")
            data["buildings"].append({"id": bid, "outer": [outer], "inner": inner, "tags": tags,
                                      "height": building_height(tags, bid)})
            added += 1
    return added

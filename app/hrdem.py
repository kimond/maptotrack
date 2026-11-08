"""NRCan HRDEM: Canada's High Resolution Digital Elevation Model (LiDAR terrain, 1 m / 2 m), all provinces.

The national mosaics are Cloud Optimized GeoTIFFs (EPSG:3979, CGVD2013 heights) found through NRCan's STAC
catalogue (collections hrdem-mosaic-1m / hrdem-mosaic-2m, asset 'dtm'). Only the window around the track is
read over HTTP (GDAL range requests) and saved in cache/hrdem/ as a small GeoTIFF, so a second run needs no
network. Where the mosaic has no LiDAR (gaps, outside Canada) the samples are NaN and the coarse DEM fills in
(lidar.CompositeDEM). https://nrcan.github.io/CanElevation/
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import requests

from . import config
from .lidar import LidarError, LidarRaster

STAC_SEARCH = "https://datacube.services.geo.ca/stac/api/search"
S3_BASE = "https://canelevation-dem.s3.ca-central-1.amazonaws.com"
COLLECTIONS = {1: "hrdem-mosaic-1m", 2: "hrdem-mosaic-2m"}
ATTRIBUTION = ("HRDEM elevation (c) Natural Resources Canada, Open Government Licence - Canada")
HRDEM_DIR = config.CACHE_DIR / "hrdem"
MAX_PIXELS = 40e6                 # per read; bigger areas are read at 2 m, then from the COG's overviews
PAD_M = 150.0
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
            "GDAL_HTTP_MULTIRANGE": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES", "VSI_CACHE": "TRUE",
            "GDAL_HTTP_TIMEOUT": "60", "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "2"}


def in_canada(bbox) -> bool:
    """Roughly: the HRDEM mosaics' extent (Canada + a margin)."""
    s, w, n, e = bbox
    return n > 41.5 and s < 83.5 and e > -141.5 and w < -52.0


def _href(item, asset):
    a = item.get("assets", {}).get(asset)
    if not a:
        return None
    h = a.get("href", "")
    if h.startswith("http"):
        return h
    return f"{S3_BASE}/{item.get('collection', '')}/{h.lstrip('./')}"


def find_cogs(bbox, res=1, log=print):
    """bbox (s, w, n, e) -> [COG url] of the mosaic tiles covering it."""
    s, w, n, e = bbox
    try:
        r = requests.get(STAC_SEARCH, params={"collections": COLLECTIONS[res], "bbox": f"{w},{s},{e},{n}",
                                              "limit": 20},
                         headers={"User-Agent": config.USER_AGENT}, timeout=60)
        r.raise_for_status()
        feats = r.json().get("features", [])
    except Exception as exc:
        raise LidarError(f"HRDEM catalogue unavailable: {exc}") from exc
    return [h for h in (_href(f, "dtm") for f in feats) if h]


def _choose_res(bbox):
    s, w, n, e = bbox
    h = (n - s) * 111_320 + 2 * PAD_M
    wd = (e - w) * 111_320 * math.cos(math.radians((s + n) / 2)) + 2 * PAD_M
    return (1 if h * wd <= MAX_PIXELS else 2), h * wd


def _cache_path(url, bbox, res):
    key = hashlib.sha1(f"{url}|{res}|" .encode() + ",".join(f"{v:.4f}" for v in bbox).encode()).hexdigest()[:16]
    return HRDEM_DIR / f"{url.rsplit('/', 1)[-1].split('.')[0]}_{key}.tif"


def _covering(bbox, res):
    """A cached window that already covers bbox (same resolution), or None."""
    if not HRDEM_DIR.exists():
        return []
    import rasterio
    from pyproj import Transformer

    s, w, n, e = bbox
    hits = []
    for p in HRDEM_DIR.glob(f"*-{res}m-*.tif"):
        try:
            with rasterio.open(p) as ds:
                tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
                xs, ys = tr.transform([w, e, w, e], [s, s, n, n])
                b = ds.bounds
                if b.left <= min(xs) and b.right >= max(xs) and b.bottom <= min(ys) and b.top >= max(ys):
                    hits.append(p)
        except Exception:
            continue
    return hits[:1]


def fetch_window(url, bbox, res, log=print) -> Path | None:
    """Read the bbox window (+PAD_M) of one COG and save it as a local GeoTIFF. None if it has no data there."""
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    dest = _cache_path(url, bbox, res)
    if dest.exists():
        return dest
    s, w, n, e = bbox
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(url) as ds:
            tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            xs, ys = tr.transform([w, e, w, e], [s, s, n, n])
            l, b = max(min(xs) - PAD_M, ds.bounds.left), max(min(ys) - PAD_M, ds.bounds.bottom)
            r, t = min(max(xs) + PAD_M, ds.bounds.right), min(max(ys) + PAD_M, ds.bounds.top)
            if l >= r or b >= t:
                return None
            win = from_bounds(l, b, r, t, ds.transform).round_offsets().round_lengths()
            npx = win.width * win.height
            scale = max(1.0, math.sqrt(npx / MAX_PIXELS))     # very large areas: from the overviews
            shape = (max(1, int(win.height / scale)), max(1, int(win.width / scale)))
            log(f"  HRDEM: reading {(r - l) / 1000:.1f} x {(t - b) / 1000:.1f} km at "
                f"{res * scale:.0f} m from {url.rsplit('/', 1)[-1]} ...")
            arr = ds.read(1, window=win, out_shape=shape, masked=True)
            data = arr.astype(np.float32).filled(np.nan)
            data[(data < -500) | (data > 9000)] = np.nan
            if not np.isfinite(data).any():
                return None
            wt = ds.window_transform(win) * rasterio.Affine.scale(win.width / shape[1], win.height / shape[0])
            prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "width": shape[1], "height": shape[0],
                    "crs": ds.crs, "transform": wt, "nodata": np.nan, "compress": "deflate", "tiled": True,
                    "predictor": 3}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **prof) as out:
        out.write(data, 1)
    tmp.replace(dest)
    return dest


def open_dem(bbox, log=print) -> LidarRaster:
    """HRDEM terrain around bbox (s, w, n, e) as a sampler (NaN where the mosaic has no LiDAR)."""
    if config.OFFLINE:
        raise LidarError("offline")
    if not in_canada(bbox):
        raise LidarError("HRDEM covers Canada only.")
    res, npx = _choose_res(bbox)
    paths = _covering(bbox, res)
    if paths:
        log(f"HRDEM ({res} m): using the saved copy")
    else:
        urls = find_cogs(bbox, res, log)
        if not urls and res == 1:                      # no 1 m mosaic here: the 2 m one
            res = 2
            urls = find_cogs(bbox, res, log)
        if not urls:
            raise LidarError("No HRDEM coverage for this area.")
        log(f"HRDEM ({res} m): {len(urls)} mosaic tile(s)")
        for u in urls:
            try:
                p = fetch_window(u, bbox, res, log)
            except Exception as exc:
                log(f"  HRDEM tile {u.rsplit('/', 1)[-1]} failed: {exc}")
                continue
            if p is not None:
                paths.append(p)
        if not paths:
            raise LidarError("HRDEM has no LiDAR data for this area.")
    return LidarRaster(paths, bbox)

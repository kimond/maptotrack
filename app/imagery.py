"""Aerial / satellite imagery for ground textures.

Sources are fetched as Web-Mercator rasters (WMS GetMap or XYZ tiles), cached on disk and
resampled onto the local track frame, one texture per terrain tile.
Add your own sources to IMAGERY_SOURCES (any WMS that supports EPSG:3857, or an XYZ tile URL).
"""
from __future__ import annotations

import hashlib
import io
import math

import numpy as np
import requests
from PIL import Image
from scipy.ndimage import map_coordinates

from . import config

R_EARTH = 6378137.0

IMAGERY_SOURCES = [
    {
        "id": "qc_saglac_2025",
        "name": "Québec – Saguenay–Lac-Saint-Jean 2025 aerial photos (20 cm)",
        "type": "wms",
        "url": "https://servicesmatriciels.mern.gouv.qc.ca/erdas-iws/ogc/wms/Imagerie",
        "layer": "Orthomosaique_2025_2025_Partenariat_SaguenayLacSaintJean_20cm_RVB",
        "coverage": (47.9779, -72.9746, 49.2407, -69.8576),  # s, w, n, e
        "max_px": 4000,
        "best_res": 0.2,
        "attribution": "Orthophotos 2025 (c) Gouvernement du Quebec / MRC du Fjord-du-Saguenay, CC BY 4.0",
        "auto": True,
    },
    {
        "id": "esri",
        "name": "Esri World Imagery (personal use only - check Esri terms before publishing)",
        "type": "xyz",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "max_zoom": 19,
        "best_res": 0.3,
        "attribution": "Imagery (c) Esri, Maxar, Earthstar Geographics",
        "auto": False,
    },
]


class ImageryError(Exception):
    pass


def list_sources():
    return [{"id": s["id"], "name": s["name"], "type": s["type"], "coverage": s.get("coverage"),
             "attribution": s.get("attribution", ""), "max_zoom": s.get("max_zoom", 21),
             "url": s["url"] if s["type"] == "xyz" else None} for s in IMAGERY_SOURCES]


def tile(src_id: str, z: int, x: int, y: int, size: int = 256) -> bytes:
    """One Web-Mercator map tile (JPEG) of a WMS source, for the map view. Cached on disk."""
    src = next((s for s in IMAGERY_SOURCES if s["id"] == src_id), None)
    if src is None:
        raise ImageryError(f"unknown imagery source {src_id}")
    if src["type"] != "wms":
        return _get(src["url"].format(z=z, x=x, y=y))
    world = 2 * math.pi * R_EARTH
    span = world / (2 ** z)
    x0 = -world / 2 + x * span
    y1 = world / 2 - y * span
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": src["layer"], "STYLES": "",
        "CRS": "EPSG:3857", "BBOX": f"{x0:.2f},{y1 - span:.2f},{x0 + span:.2f},{y1:.2f}",
        "WIDTH": str(size), "HEIGHT": str(size), "FORMAT": "image/jpeg",
    }
    return _get(src["url"], params)


def _covers(src, bbox):
    cov = src.get("coverage")
    if not cov:
        return True
    s, w, n, e = bbox
    return s >= cov[0] and w >= cov[1] and n <= cov[2] and e <= cov[3]


def pick_source(choice: str, bbox):
    """choice: 'auto' | source id | 'landcover' | 'plain'. Returns source dict or None."""
    if choice in ("landcover", "plain", "none", None):
        return None
    if choice == "auto":
        for s in IMAGERY_SOURCES:
            if s.get("auto") and _covers(s, bbox):
                return s
        return None
    for s in IMAGERY_SOURCES:
        if s["id"] == choice:
            return s
    return None


def _merc(lat, lon):
    lat = np.clip(np.asarray(lat, float), -85.0511, 85.0511)
    x = R_EARTH * np.radians(lon)
    y = R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def _get(url, params=None):
    key = hashlib.sha1((url + repr(sorted((params or {}).items()))).encode()).hexdigest()
    path = config.CACHE_DIR / "imagery" / key[:2] / f"{key}.img"
    if path.exists():
        return path.read_bytes()
    r = requests.get(url, params=params, headers={"User-Agent": config.USER_AGENT}, timeout=90)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "xml" in ctype or "text" in ctype:
        raise ImageryError(f"Imagery server returned an error: {r.text[:300]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return r.content


class _Raster:
    """An RGB image covering a Web-Mercator rectangle."""

    def __init__(self, img: np.ndarray, mx0, my1, mres_x, mres_y):
        self.img, self.mx0, self.my1, self.rx, self.ry = img, mx0, my1, mres_x, mres_y

    def sample(self, lat, lon):
        mx, my = _merc(lat, lon)
        col = (mx - self.mx0) / self.rx - 0.5
        row = (self.my1 - my) / self.ry - 0.5
        out = np.stack([map_coordinates(self.img[..., c].astype(np.float32), [row, col], order=1, mode="nearest")
                        for c in range(3)], axis=-1)
        return out


def _fetch_wms(src, lat_s, lon_w, lat_n, lon_e, res_m):
    mx0, my0 = _merc(lat_s, lon_w)
    mx1, my1 = _merc(lat_n, lon_e)
    scale = 1.0 / math.cos(math.radians((lat_s + lat_n) / 2))   # mercator metres per ground metre
    mres = max(res_m, src.get("best_res", 0.1)) * scale
    W = int(math.ceil((mx1 - mx0) / mres))
    H = int(math.ceil((my1 - my0) / mres))
    mp = src.get("max_px", 4000)
    if W > mp or H > mp:
        f = max(W, H) / mp
        W, H = int(W / f), int(H / f)
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": src["layer"], "STYLES": "",
        "CRS": "EPSG:3857", "BBOX": f"{mx0:.2f},{my0:.2f},{mx1:.2f},{my1:.2f}",
        "WIDTH": str(W), "HEIGHT": str(H), "FORMAT": "image/jpeg",
    }
    data = _get(src["url"], params)
    img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    return _Raster(img, float(mx0), float(my1), (mx1 - mx0) / img.shape[1], (my1 - my0) / img.shape[0])


def _fetch_xyz(src, lat_s, lon_w, lat_n, lon_e, res_m):
    lat_c = (lat_s + lat_n) / 2
    z = math.ceil(math.log2(156543.03392 * math.cos(math.radians(lat_c)) / max(res_m, src.get("best_res", 0.1))))
    z = int(min(max(z, 1), src.get("max_zoom", 19)))
    n = 2 ** z

    def tile_xy(lat, lon):
        x = (lon + 180) / 360 * n
        y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
        return x, y

    tx0, ty0 = tile_xy(lat_n, lon_w)
    tx1, ty1 = tile_xy(lat_s, lon_e)
    tx0, ty0, tx1, ty1 = int(tx0), int(ty0), int(tx1), int(ty1)
    if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 400:
        raise ImageryError("Imagery area too large")
    from .parallel import io_map

    keys = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

    def one(k):
        data = _get(src["url"].format(z=z, x=k[0], y=k[1]))
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB").resize((256, 256)))

    tiles = dict(zip(keys, io_map(one, keys, threads=6)))
    img = np.vstack([np.hstack([tiles[(tx, ty)] for tx in range(tx0, tx1 + 1)]) for ty in range(ty0, ty1 + 1)])
    world = 2 * math.pi * R_EARTH
    tile_m = world / n
    mx0 = -world / 2 + tx0 * tile_m
    my1 = world / 2 - ty0 * tile_m
    return _Raster(img, mx0, my1, tile_m / 256, tile_m / 256)


def fetch_raster(src, lat_s, lon_w, lat_n, lon_e, res_m):
    if src["type"] == "wms":
        return _fetch_wms(src, lat_s, lon_w, lat_n, lon_e, res_m)
    return _fetch_xyz(src, lat_s, lon_w, lat_n, lon_e, res_m)


def ground_image(src, proj, xs, ys):
    """RGB (H,W,3) float + valid mask for the local grid xs (west->east), ys (north->south)."""
    X, Y = np.meshgrid(xs, ys)
    lat, lon = proj.to_latlon(X, Y)
    pad = 0.0003
    res = abs(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    r = fetch_raster(src, float(lat.min()) - pad, float(lon.min()) - pad, float(lat.max()) + pad,
                     float(lon.max()) + pad, res)
    rgb = r.sample(lat.ravel(), lon.ravel()).reshape(X.shape + (3,))
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    valid = ~((mn > 248) | (mx < 4))   # pure white / black = outside coverage
    return rgb, valid

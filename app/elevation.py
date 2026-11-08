"""Terrain elevation from free Terrarium-encoded DEM tiles (AWS Terrain Tiles / Mapzen).

height = R*256 + G + B/256 - 32768  (metres)
Sources: SRTM, NED, Copernicus/GMTED etc. ~10-30 m resolution depending on region.
"""
from __future__ import annotations

import io
import math

import numpy as np
import requests
from PIL import Image

from . import config


class ElevationError(Exception):
    pass


def _lonlat_to_global_px(lon, lat, z):
    n = 256 * (2 ** z)
    lat = np.clip(lat, -85.0511, 85.0511)
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    phi = np.radians(lat)
    y = (1.0 - np.log(np.tan(phi) + 1.0 / np.cos(phi)) / math.pi) / 2.0 * n
    return x, y


def _fetch_tile(z, x, y) -> np.ndarray:
    path = config.CACHE_DIR / "terrarium" / str(z) / str(x) / f"{y}.png"
    if path.exists():
        data = path.read_bytes()
    else:
        url = config.TERRARIUM_URL.format(z=z, x=x, y=y)
        try:
            r = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=30)
            r.raise_for_status()
        except Exception as exc:
            raise ElevationError(f"Could not download elevation tile {z}/{x}/{y}: {exc}") from exc
        data = r.content
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float64)
    return img[:, :, 0] * 256.0 + img[:, :, 1] + img[:, :, 2] / 256.0 - 32768.0


class TerrariumDEM:
    def __init__(self, bbox, zoom: int = 14):
        s, w, n, e = bbox
        self.z = zoom
        x0, y0 = _lonlat_to_global_px(w, n, zoom)
        x1, y1 = _lonlat_to_global_px(e, s, zoom)
        tx0, ty0 = int(x0 // 256), int(y0 // 256)
        tx1, ty1 = int(x1 // 256), int(y1 // 256)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 400:
            raise ElevationError("Area too large for elevation download - reduce the margin or zoom.")
        from .parallel import io_map

        keys = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
        tiles = dict(zip(keys, io_map(lambda k: _fetch_tile(zoom, k[0], k[1]), keys, threads=8)))
        self.grid = np.vstack([np.hstack([tiles[(tx, ty)] for tx in range(tx0, tx1 + 1)])
                               for ty in range(ty0, ty1 + 1)])
        self.ox, self.oy = tx0 * 256, ty0 * 256

    def sample(self, lat, lon) -> np.ndarray:
        gx, gy = _lonlat_to_global_px(lon, lat, self.z)
        fx = np.asarray(gx) - self.ox - 0.5
        fy = np.asarray(gy) - self.oy - 0.5
        h, w = self.grid.shape
        fx = np.clip(fx, 0, w - 1.001)
        fy = np.clip(fy, 0, h - 1.001)
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - ix, fy - iy
        g = self.grid
        return (
            g[iy, ix] * (1 - tx) * (1 - ty)
            + g[iy, ix + 1] * tx * (1 - ty)
            + g[iy + 1, ix] * (1 - tx) * ty
            + g[iy + 1, ix + 1] * tx * ty
        )


class FlatDEM:
    def __init__(self, height: float = 0.0):
        self.h = height

    def sample(self, lat, lon):
        return np.full(np.shape(lat), self.h, dtype=float)


def make_dem(bbox, zoom: int = 14, log=print):
    if config.OFFLINE:
        from .demo import SyntheticDEM

        return SyntheticDEM(bbox)
    try:
        return TerrariumDEM(bbox, zoom)
    except ElevationError as exc:
        log(f"WARNING: {exc} - using flat terrain.")
        return FlatDEM()

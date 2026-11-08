"""Geographic helpers: local metric projection around a track origin.

Local frame used everywhere in the generator ("track frame"):
    x = east (m), y = north (m), z = up (m)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LocalProjection:
    lat0: float
    lon0: float

    def __post_init__(self):
        phi = math.radians(self.lat0)
        # Metres per degree (WGS84 series expansion) - accurate to cm over a few km.
        self.m_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
        self.m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)

    def to_xy(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        return (lon - self.lon0) * self.m_lon, (lat - self.lat0) * self.m_lat

    def to_latlon(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return self.lat0 + y / self.m_lat, self.lon0 + x / self.m_lon

    def latlngs_to_xy(self, latlngs) -> np.ndarray:
        arr = np.asarray(latlngs, dtype=float).reshape(-1, 2)
        x, y = self.to_xy(arr[:, 0], arr[:, 1])
        return np.column_stack([x, y])


def bbox_of(latlngs, margin_m: float = 0.0):
    """Return (south, west, north, east) of a list of [lat, lng], grown by margin metres."""
    arr = np.asarray(latlngs, dtype=float).reshape(-1, 2)
    s, n = arr[:, 0].min(), arr[:, 0].max()
    w, e = arr[:, 1].min(), arr[:, 1].max()
    proj = LocalProjection((s + n) / 2, (w + e) / 2)
    dlat = margin_m / proj.m_lat
    dlon = margin_m / proj.m_lon
    return (s - dlat, w - dlon, n + dlat, e + dlon)


def slugify(name: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "_" for c in name.strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "track"

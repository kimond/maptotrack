"""Centerline processing: resampling, corner smoothing, elevation profile, frames."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def resample(xy: np.ndarray, step: float, loop: bool) -> np.ndarray:
    pts = np.vstack([xy, xy[:1]]) if loop else xy
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    if total < step * 2:
        raise ValueError("Track is too short.")
    if loop:
        n = max(int(round(total / step)), 8)
        si = np.arange(n) * (total / n)
    else:
        n = max(int(round(total / step)), 2)
        si = np.linspace(0, total, n + 1)
    return np.column_stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])])


def _dedupe(xy: np.ndarray, tol=0.5) -> np.ndarray:
    keep = [0]
    for i in range(1, len(xy)):
        if np.linalg.norm(xy[i] - xy[keep[-1]]) > tol:
            keep.append(i)
    return xy[keep]


@dataclass
class Centerline:
    p: np.ndarray        # (N,2) x,y metres
    z: np.ndarray        # (N,)  road height
    t: np.ndarray        # (N,2) unit tangent (driving direction)
    n: np.ndarray        # (N,2) unit left normal
    s: np.ndarray        # (N,)  distance from start line
    loop: bool
    length: float
    spacing: float
    radius: np.ndarray   # (N,) signed turn radius (inf on straights)

    def xyz(self, i):
        return np.array([self.p[i, 0], self.p[i, 1], self.z[i]])

    def frame_at(self, dist: float):
        """Interpolated point/tangent/normal/z at a distance along the track (wraps on loops)."""
        if self.loop:
            dist = dist % self.length
        else:
            dist = float(np.clip(dist, 0, self.length))
        f = dist / self.spacing
        i = int(np.floor(f))
        a = f - i
        n = len(self.p)
        j = (i + 1) % n if self.loop else min(i + 1, n - 1)
        i = i % n if self.loop else min(i, n - 1)
        p = self.p[i] * (1 - a) + self.p[j] * a
        z = self.z[i] * (1 - a) + self.z[j] * a
        t = self.t[i] * (1 - a) + self.t[j] * a
        t = t / (np.linalg.norm(t) + 1e-9)
        nrm = np.array([-t[1], t[0]])
        return p, z, t, nrm

    def kdtree(self, step=1.0):
        """Dense (x,y) samples + their heights for nearest-road queries."""
        dense = resample(self.p, step, self.loop)
        tree = cKDTree(dense)
        _, idx = cKDTree(self.p).query(dense)
        return tree, self.z[idx], dense


MIN_ELEV_SMOOTHING = 4.0     # metres (gaussian sigma) - floor for the road height smoothing


def profile_roughness(s, z, loop):
    """Bumps of the road profile: (largest deviation from a 10 m average in m, largest grade change per
    2 m in %). A kart or car feels > 3 cm / > 3 % as a hit."""
    from scipy.ndimage import gaussian_filter1d as g1

    if len(z) < 5:
        return 0.0, 0.0
    ds = float(np.median(np.diff(s))) or 2.0
    bump = z - g1(z, 5.0 / ds, mode="wrap" if loop else "nearest")
    grade = np.diff(z) / ds * 100
    k = max(1, int(round(2.0 / ds)))
    dg = np.abs(grade[k:] - grade[:-k]) if len(grade) > k else np.zeros(1)
    return float(np.abs(bump).max()), float(dg.max())


def build_centerline(latlngs, proj, dem, loop: bool, spacing=2.0, corner_smoothing=8.0,
                     elev_smoothing=25.0, start_offset=0.0, reverse=False, elev_offset=0.0) -> Centerline:
    xy = proj.latlngs_to_xy(latlngs)
    if reverse:
        xy = xy[::-1]
    xy = _dedupe(xy)
    if loop and len(xy) > 2 and np.linalg.norm(xy[0] - xy[-1]) < 2.0:
        xy = xy[:-1]
    if len(xy) < 2:
        raise ValueError("Need at least two distinct points.")
    mode = "wrap" if loop else "nearest"

    dense = resample(xy, 1.0, loop)
    if corner_smoothing > 0:
        sm = gaussian_filter1d(dense, sigma=corner_smoothing, axis=0, mode=mode)
        if not loop:  # keep the exact start/end positions
            k = min(len(sm) // 4, int(corner_smoothing * 3))
            if k > 0:
                w = np.linspace(0, 1, k)[:, None]
                sm[:k] = dense[:k] * (1 - w) + sm[:k] * w
                sm[-k:] = sm[-k:] * (1 - w[::-1]) + dense[-k:] * w[::-1]
        dense = sm
    pts = resample(dense, spacing, loop)
    if loop and start_offset:
        pts = np.roll(pts, -int(round(start_offset / spacing)), axis=0)

    if loop:
        d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    else:
        d = np.gradient(pts, axis=0)
    t = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    n = np.column_stack([-t[:, 1], t[:, 0]])

    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]) if loop else pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    length = float(s[-1])
    s = s[: len(pts)]

    # Road height: median of 5 samples across the road (-2..+2 m) instead of one point, so LiDAR noise,
    # tyre walls, parked karts/cars or a kerb under the centre line don't become bumps.
    zs = []
    for off in (-2.0, -1.0, 0.0, 1.0, 2.0):
        q = pts + n * off
        lat, lon = proj.to_latlon(q[:, 0], q[:, 1])
        zs.append(np.asarray(dem.sample(lat, lon), dtype=float))
    zs = np.vstack(zs)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        z = np.where(np.isfinite(zs).any(0), np.nanmedian(zs, axis=0), zs[2])
    # Always remove bumps shorter than ~10 m (sensor noise; no real road has them), whatever the setting.
    sigma_m = max(float(elev_smoothing), MIN_ELEV_SMOOTHING)
    z = gaussian_filter1d(z, sigma=max(sigma_m / spacing, 0.5), mode=mode)
    z = z + elev_offset

    heading = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    dh = np.gradient(heading)
    if loop:
        dh = (np.roll(heading, -1) - np.roll(heading, 1))
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        dh /= 2
    with np.errstate(divide="ignore"):
        radius = np.where(np.abs(dh) > 1e-6, spacing / dh, np.inf)

    return Centerline(p=pts, z=z, t=t, n=n, s=s, loop=loop, length=length,
                      spacing=length / len(pts) if loop else length / (len(pts) - 1),
                      radius=radius)

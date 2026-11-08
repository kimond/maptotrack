"""Assetto Corsa AI lines: ai/fast_lane.ai (the racing line the AI drive) and ai/pit_lane.ai, written from the
track's centre line instead of recording a lap in game.

File layout (version 7, as Content Manager's AcTools/AiFile reads it; all little-endian):
    int32 version (7), int32 point count, int32 lap time (0), int32 sample count
    point x count:  vec3 position, float32 length along the line, int32 id                   (20 bytes)
    int32 extra count, extra x count: float32 speed, gas, brake, obsolete lat G, radius, side left, side right,
        camber, direction, vec3 normal, float32 length, vec3 forward, float32 tag, grade       (72 bytes)
    int32 has grid (0 here: AC builds it when it loads the line)
Positions are in AC's frame: (x east, y up, z south) = (local x, local z, -local y).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

STEP = 1.0                  # m between AI points (a recorded line has about one per metre)
LAT_G = 9.0                 # m/s2 lateral grip for the target speed in corners
TOP_SPEED = 70.0            # m/s
BRAKE = 7.0                 # m/s2 when slowing down for a corner
ACCEL = 3.0                 # m/s2 when speeding up


def _ac(p):
    p = np.asarray(p, float)
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


def resample(p, z, hl, hr, loop, step=STEP):
    """Points every `step` m along the line -> (xyz local, half widths left/right, cumulative length)."""
    p = np.asarray(p, float)
    z = np.asarray(z, float)
    if loop:
        p, z = np.vstack([p, p[:1]]), np.concatenate([z, z[:1]])
        hl, hr = np.concatenate([hl, hl[:1]]), np.concatenate([hr, hr[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    L = float(s[-1])
    n = max(int(L / step), 8)
    si = np.linspace(0, L, n, endpoint=not loop)
    xyz = np.column_stack([np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1]), np.interp(si, s, z)])
    return xyz, np.interp(si, s, hl), np.interp(si, s, hr), si, L


def speeds(xyz, loop, lat_g=LAT_G, top=TOP_SPEED, brake=BRAKE, accel=ACCEL):
    """Target speed (m/s) and corner radius (m) at every point: v = sqrt(a * R), then limited so the car can
    brake into each corner and accelerate out of it."""
    n = len(xyz)
    xy = xyz[:, :2]
    nxt = np.roll(xy, -1, 0) if loop else np.vstack([xy[1:], xy[-1:]])
    k = 6                                               # curvature over +-6 m (smooth)
    a = np.roll(xy, k, 0) if loop else np.vstack([np.repeat(xy[:1], k, 0), xy[:-k]])
    c = np.roll(xy, -k, 0) if loop else np.vstack([xy[k:], np.repeat(xy[-1:], k, 0)])
    ab, bc, ca = np.linalg.norm(xy - a, axis=1), np.linalg.norm(c - xy, axis=1), np.linalg.norm(a - c, axis=1)
    cross = np.abs((xy[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (xy[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    radius = np.where(cross > 1e-6, ab * bc * ca / np.maximum(2 * cross, 1e-9), 1e4)
    radius = np.clip(radius, 3.0, 1e4)
    v = np.minimum(np.sqrt(lat_g * radius), top)
    ds = np.linalg.norm(nxt - xy, axis=1)
    for _ in range(3 if loop else 1):                   # braking (backwards) and acceleration (forwards) limits
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            if not loop and i == n - 1:
                continue
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * brake * ds[i]))
        for i in range(n):
            j = (i - 1) % n
            if not loop and i == 0:
                continue
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * accel * ds[j]))
    return v, radius


def ai_bytes(xyz, hl, hr, s, loop, speed=None, radius=None) -> bytes:
    n = len(xyz)
    if speed is None:
        speed, radius = speeds(xyz, loop)
    P = _ac(xyz)
    fwd = (np.roll(P, -1, 0) - np.roll(P, 1, 0)) if loop else np.gradient(P, axis=0)
    fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    seg = np.linalg.norm((np.roll(P, -1, 0) if loop else np.vstack([P[1:], P[-1:]])) - P, axis=1)
    grade = np.gradient(xyz[:, 2]) / np.maximum(np.gradient(s), 1e-6)
    dv = (np.roll(speed, -1) if loop else np.r_[speed[1:], speed[-1]]) - speed
    gas = np.where(dv >= -0.05, 1.0, 0.0)
    brk = np.where(dv < -0.05, np.clip(-dv * 4, 0.2, 1.0), 0.0)
    heading = np.arctan2(fwd[:, 0], fwd[:, 2])
    out = [struct.pack("<4i", 7, n, 0, n)]
    for i in range(n):
        out.append(struct.pack("<4fi", P[i, 0], P[i, 1], P[i, 2], float(s[i]), i))
    out.append(struct.pack("<i", n))
    for i in range(n):
        out.append(struct.pack("<18f", float(speed[i]), float(gas[i]), float(brk[i]), 0.0, float(radius[i]),
                               float(hl[i]), float(hr[i]), 0.0, float(heading[i]), 0.0, 1.0, 0.0,
                               float(seg[i]), float(fwd[i, 0]), float(fwd[i, 1]), float(fwd[i, 2]), 0.0,
                               float(grade[i])))
    out.append(struct.pack("<i", 0))
    return b"".join(out)


def read_ai(data: bytes) -> dict:
    """Parse a version 7 .ai file (tests, checks)."""
    ver, n, lap, samples = struct.unpack_from("<4i", data, 0)
    o = 16
    pts = np.frombuffer(data, dtype=np.dtype([("p", "<f4", 3), ("len", "<f4"), ("id", "<i4")]), count=n, offset=o)
    o += 20 * n
    ne = struct.unpack_from("<i", data, o)[0]
    o += 4
    ext = np.frombuffer(data, dtype="<f4", count=18 * ne, offset=o).reshape(ne, 18)
    o += 72 * ne
    grid = struct.unpack_from("<i", data, o)[0]
    return {"version": ver, "n": n, "points": pts, "extra": ext, "has_grid": grid, "size": o + 4}


def write_ai_lines(ai_dir: Path, line: dict, objs: list, log=print) -> dict:
    """fast_lane.ai along the centre line; pit_lane.ai along the road past the pit spots (our tracks have no
    separate pit lane): from 40 m before the first pit spot to 40 m after the last one."""
    ai_dir.mkdir(parents=True, exist_ok=True)
    loop = bool(line.get("loop"))
    n0 = len(line["p"])
    xyz, hl, hr, s, L = resample(line["p"], line["z"], np.broadcast_to(np.asarray(line["hl"], float), (n0,)),
                                 np.broadcast_to(np.asarray(line["hr"], float), (n0,)), loop)
    v, r = speeds(xyz, loop)
    (ai_dir / "fast_lane.ai").write_bytes(ai_bytes(xyz, hl, hr, s, loop, v, r))
    info = {"points": len(xyz), "length": L, "lap_s": float(np.sum(np.diff(np.r_[s, L if loop else s[-1]]) /
                                                                    np.maximum(v, 1.0)))}
    pits = [o for o in objs if o["name"].startswith("AC_PIT_")]
    if pits:
        idx = [int(np.argmin(np.hypot(xyz[:, 0] - o["pos"][0], xyz[:, 1] - o["pos"][1]))) for o in pits]
        n = len(xyz)
        m = int(40.0 / STEP)
        if loop:
            order = sorted(idx)
            gaps = [((order[(i + 1) % len(order)] - order[i]) % n or n, i) for i in range(len(order))]
            _g, gi = max(gaps)
            a, b = order[(gi + 1) % len(order)], order[gi]
            span = min(((b - a) % n) + 1 + 2 * m, n)
            ids = [(a - m + j) % n for j in range(span)]
        else:
            ids = list(range(max(0, min(idx) - m), min(n, max(idx) + m + 1)))
        px = xyz[ids]
        ps = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(px[:, :2], axis=0), axis=1))])
        pv = np.minimum(v[ids], 60 / 3.6)                     # pit speed
        (ai_dir / "pit_lane.ai").write_bytes(ai_bytes(px, hl[ids], hr[ids], ps, False, pv, r[ids]))
        info["pit_points"] = len(ids)
    log(f"  AI line: {info['points']} points, {L:.0f} m, estimated lap {info['lap_s']:.1f} s"
        + (f"; pit lane {info.get('pit_points', 0)} points" if pits else ""))
    return info

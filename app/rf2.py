"""rFactor 2 export: a generated track as an rF2 Location for ModDev (Mod Mode).

    <rF2>\\ModDev\\Locations\\<Loc>\\
        <Loc>.tdf                    surfaces (grip, sounds, effects) by material name
        <Loc>\\<Loc>.scn             scene: one Instance per mesh (collision / HAT flags), lights, sky, timing gates
        <Loc>\\<Loc>.gdb             track data (names, length, place, time zone, pit speed...)
        <Loc>\\<Loc>.aiw             AI waypoints, grid, garages / pit spots, sectors
        <Loc>\\<Loc>.cam             TV cameras
        Assets\\GMT\\*.gmt           3D meshes - written by Studio 397's Blender exporter (io_rfactor2_gmt, for
                                     Blender 5.0+), driven by blender/export_rf2_gmt.py with rf2_meshes.json
        Assets\\Maps\\*.dds          textures

The text files follow the official samples shipped with rF2 (ModDev\\Locations\\Joesville, LochDrummond).
Packaging into a .mas / .rfcmp for normal rF2 is a manual step (rF2's MAS2 tool), see README.

Coordinates: maptotrack's local frame is x east, y north, z up (metres); rF2 is y up. RF2_AXES maps one to the
other and must match what the GMT exporter does with Blender's axes (Blender = the local frame).
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image

# local (x east, y north, z up) -> rF2 (X, Y up, Z). gMotor exporters map Max/Blender (x, y, z-up) to (x, z, y).
RF2_AXES = ("x", "z", "y")
SECTOR_FRACTIONS = (1 / 3, 2 / 3)

# surfaces: our material codes in the .tdf (rF2 matches material names that contain them)
SURFACES = {
    "mtroad": ("Asphalt", "Wear=1.00 Dry=1.00 Wet=0.84 Roughness=(0.5,0.25) Resistance=0 BumpAmp=RoadBumpAmp "
               "BumpWavelen=RoadBumpLen Legal=true Spring=0 Damper=0 CollFrict=0.4 Sparks=1 Scraping=1 Sound=dry",
               "road"),
    "mtgrav": ("Gravel shoulder", "Wear=0.7 Dry=0.80 Wet=0.55 Roughness=(0.45,0.10) Resistance=500 BumpAmp=0.02 "
               "BumpWavelen=1.6 Legal=false Spring=200 Damper=200 CollFrict=0.8 Sparks=1 Scraping=1 Sink=0.010 "
               "Sound=gravel", "dirt"),
    "mtgras": ("Grass (lawn)", "Wear=0.6 Dry=0.6 Wet=0.35 Roughness=(0.4,0.04) Resistance=2000 BumpAmp=0.03 "
               "BumpWavelen=2.0 Legal=false Spring=300 Damper=300 CollFrict=1.5 Sparks=0 Scraping=0 Sink=0.025 "
               "Sound=grass", "grass"),
    "mtrugh": ("Tall / wild grass", "Wear=0.5 Dry=0.45 Wet=0.25 Roughness=(0.4,0.06) Resistance=3000 BumpAmp=0.04 "
               "BumpWavelen=2.2 Legal=false Spring=400 Damper=400 CollFrict=2 Sparks=0 Scraping=0 Sink=0.03 "
               "Sound=grass", "grass"),
    "mtdirt": ("Dirt, forest floor", "Wear=0.4 Dry=0.48 Wet=0.22 Roughness=(0.4,0.25) Resistance=1000 BumpAmp=0.044 "
               "BumpWavelen=1.2 Legal=false Spring=200 Damper=200 CollFrict=1.9 Sparks=0 Scraping=0 Sink=0.042 "
               "Sound=gravel", "dirt"),
    "mtsand": ("Sand", "Wear=0.75 Dry=0.6 Wet=0.5 Roughness=(0.4,0.25) Resistance=20000 BumpAmp=0.035 "
               "BumpWavelen=1.2 Legal=false Spring=400 Damper=200 CollFrict=2.5 Sparks=0 Scraping=0 Sink=0.04 "
               "Sound=gravel", "dirt"),
    "mtwall": ("Walls, buildings, poles", "Wear=0 Dry=1.0 Wet=1.0 Resistance=0 BumpAmp=0 BumpWavelen=1 Legal=false "
               "Spring=0 Damper=0 CollFrict=0.5 Sparks=1 Scraping=1 Sound=solidwall", None),
    "mttyre": ("Tyre walls", "Wear=0 Dry=1.0 Wet=1.0 Resistance=0 BumpAmp=0 BumpWavelen=1 Legal=false "
               "Spring=60000 Damper=6000 CollFrict=0.9 Sparks=0 Scraping=0 Sound=guardrail", None),
}

_EFFECTS = {
    "road": [
        "Reaction=tiresmoke Tex=SMOKETire.tga Max=1024 Scale=(0.5,0.5,0.5) Growth=(2.5,2.0,2.0) "
        "ASDEnvelope=(0.1,0.7,3.8) Suspension=0.98 DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
        "Reaction=skid Tex=skidhard.tga Max=2500 Pixel=NoReduceDetail Particle=Plane+Deformable+SingleSided "
        "DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
        "Reaction=wetskid Tex=skidwet.tga Max=1024 Duration=0.40 Pixel=NoReduceDetail "
        "Particle=Plane+Deformable+SingleSided DestBlend=One SrcBlend=SrcAlpha",
        "Reaction=spray Tex=rainspray.tga Max=1024 Scale=(1.3,0.02,0.02) Growth=(2.0,0.19,0.19) "
        "GrowthVel=(0.17,0.13,0.13) Power=0.41 RampSpeed=90 OffsetVel=0.10 ASDEnvelope=(0.01,3.0,2.0) "
        "DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
    ],
    "grass": [
        "Reaction=softskid Tex=skidgreen.tga Max=1024 Pixel=NoReduceDetail Particle=Plane+Deformable+SingleSided "
        "DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
        "Reaction=dust Tex=DIRT_Cloud.tga Max=128 TopSpeed=105 Scale=(1.4,1.0,1.0) Growth=(3.0,2.0,2.0) "
        "Suspension=0.847 ASDEnvelope=(1.0,1.2,3.5) DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
        "Reaction=dirt Tex=GrassDirtKick.tga Max=64 Scale=(0.7,0.7,0.7) Growth=(0.85,0.85,0.85) Suspension=0.693 "
        "ASDEnvelope=(0.2,0.8,0.2) DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
    ],
    "dirt": [
        "Reaction=dust Tex=DIRT_Cloud.tga Max=512 TopSpeed=110 Scale=(1.0,1.0,1.0) Growth=(3.0,2.0,2.0) "
        "Suspension=0.99 ASDEnvelope=(0.1,2.0,3.5) DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
        "Reaction=dirt Tex=DirtKick.tga Max=64 Scale=(0.8,0.8,0.8) Growth=(1.2,0.9,0.9) Suspension=0.65 "
        "ASDEnvelope=(0.01,3.5,2.1) DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
        "Reaction=softskid Tex=skidbrown.tga Max=1024 Pixel=NoReduceDetail Particle=Plane+Deformable+SingleSided "
        "DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
    ],
}

NEAR_SURFACE = {"lawn": "mtgras", "wild": "mtrugh", "dry": "mtrugh", "forest": "mtdirt", "sand": "mtsand",
                "gravel": "mtgrav", "soil": "mtdirt", "dirt": "mtdirt", "paved": "mtroad"}

# meshes that are not worth a collision in rF2 / are drawn only
VIEW_DISTANCE = [("VEG_GRASS", 90.0), ("VEG_TALLGRASS", 200.0), ("TREES", 1200.0), ("BLD_", 2000.0),
                 ("1WALL_BLD", 2000.0), ("PROPS", 900.0), ("1WALL_PROPS", 900.0), ("WIRES_HV", 2500.0),
                 ("WIRES", 500.0), ("PYLONS", 4000.0), ("1WALL_PYLON", 4000.0), ("TYRES", 120.0)]


def loc_name(slug: str) -> str:
    """rF2 folder / file name: letters, digits, underscores."""
    n = re.sub(r"[^A-Za-z0-9_]+", "_", slug).strip("_") or "maptotrack"
    return n[:40]


def to_rf2(p):
    """local (x, y, z) -> rF2 (X, Y, Z) per RF2_AXES."""
    p = np.asarray(p, float)
    idx = {"x": 0, "y": 1, "z": 2}
    cols = []
    for a in RF2_AXES:
        sign = -1.0 if a.startswith("-") else 1.0
        cols.append(sign * p[..., idx[a.lstrip("-")]])
    return np.stack(cols, -1)


def yaw_of(forward_xy):
    """rF2 Ori yaw for a forward direction in the local frame: a vehicle's forward is -Z in its own frame, so
    after the yaw rotation forward = (-sin yaw, -cos yaw) in (X, Z)."""
    f = to_rf2(np.array([forward_xy[0], forward_xy[1], 0.0]))
    return math.atan2(-f[0], -f[2])


def _v(a, fmt="{:.3f}"):
    return "(" + ",".join(fmt.format(float(x)) for x in a) + ")"


# ---------------------------------------------------------------------------------------------- surfaces
def surface_of(mesh_name: str, material: str) -> str | None:
    """rF2 surface code for a mesh (None = drawn only, no collision)."""
    n = mesh_name.upper()
    if n.startswith("1ROAD"):
        return "mtroad"
    if n.startswith("1GRASS_SHLD"):
        return "mtgrav"
    if n.startswith("1GRASS_NEAR") or n.startswith("1ROAD_NEAR"):
        m = re.match(r"1(?:GRASS|ROAD)_NEAR_([A-Z]+)", n)
        return NEAR_SURFACE.get(m.group(1).lower(), "mtgras") if m else "mtgras"
    if n.startswith("1GRASS"):
        return "mtgras"
    if n.startswith("1WALL_TYRES"):
        return "mttyre"
    if n.startswith("1WALL"):
        return "mtwall"
    return None


def view_distance(name: str) -> float:
    return next((d for pre, d in VIEW_DISTANCE if name.startswith(pre)), 0.0)


def tdf_text(loc: str) -> str:
    out = [f"// {loc}: surfaces written by maptotrack. A material belongs to a surface when its name contains the",
           "// code on the Materials= line (e.g. mtroad_asphalt_tiled).",
           "[TRACKVARS]", "RoadDryGrip=1.00", "RoadWetGrip=0.85", "RoadBumpAmp=0.0002", "RoadBumpLen=0.5",
           "HATFilterMaxOffset=0.002", ""]
    for code, (title, line, fx) in SURFACES.items():
        out += [f"// {title}", "[FEEDBACK]", line] + _EFFECTS.get(fx, []) + [f"Materials={code}", ""]
    out += ["// effects not tied to a surface (as in every stock track): rF2 looks them up when the session starts",
            "// and crashes on the first frame when they are missing", "[LOOSE]"] + LOOSE_REACTIONS
    return "\r\n".join(out) + "\r\n"


LOOSE_REACTIONS = [
    "Reaction=spark Tex=spark.tga Max=256 Duration=0.6 Scale=(.08,.05,.05) Particle=Plane+Deformable "
    "DestBlend=InvSrcAlpha SrcBlend=SrcAlpha Shader=AlphaChannel+NoShade",
    "Reaction=whitesmoke Tex=smokewhite.tga Max=256 Scale=(1.0,1.0,1.0) Growth=(2.5,2.0,2.0) "
    "ASDEnvelope=(1.0,1.2,3.3) DriftMult=0.15 Suspension=0.999 DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
    "Reaction=blacksmoke Tex=smokeblack.tga Max=1024 Scale=(1.5,1.5,1.5) Growth=(2.0,2.0,2.0) "
    "ASDEnvelope=(0.5,3.8,1.2) DriftMult=0.3 Suspension=0.99 DestBlend=Zero SrcBlend=SrcAlpha",
    "Reaction=flames Tex=fire.tga Max=256 Scale=(0.8,0.8,0.8) Growth=(6.3,1.75,1.0) ASDEnvelope=(0.08,0.17,0.05) "
    "DriftMult=0.75 Suspension=0.8 DestBlend=One SrcBlend=SrcAlpha Shader=AlphaChannel+NoShade",
    "Reaction=wallskid Tex=skidwall.tga Max=128 Scale=(3.25,1.10,1.0) Pixel=NoReduceDetail "
    "Particle=Plane+Deformable+SingleSided DestBlend=InvSrcAlpha SrcBlend=SrcAlpha",
]


# ---------------------------------------------------------------------------------------------- gdb
def gdb_text(loc: str, line: dict, objs_count: int) -> str:
    name = line.get("name") or loc
    km = line.get("total_length", line.get("length", 0.0)) / 1000.0
    area = line.get("mode") == "area"
    lon = float(line.get("lon", 0.0))
    tz = round(lon / 15.0)
    place = ", ".join(x for x in (line.get("city"), line.get("country")) if x) or "maptotrack"
    return "\r\n".join([
        f"{loc}",
        "{",
        "  Filter Properties = *",
        f"  TrackName = {name}",
        f"  EventName = {name}",
        f"  VenueName = {name}",
        f"  Location = {place}",
        f"  Length = {km:.3f} KM / {km * 0.621371:.2f} Miles",
        f"  TrackType = {'Free Roam' if area else ('Road Course' if line.get('loop') else 'Point to Point')}",
        f"  TerrainDataFile=..\\{loc}.tdf",
        "  HeadlightsRequired = false",
        f"  Max Vehicles = {max(1, objs_count)}",
        "  FormationAndStart=0",
        "  PitlaneBoundary = 0     // no pit lane: request a pit stop to be serviced",
        "  GarageDepth = 2.0",
        "  RacePitKPH = 60.0",
        "  NormalPitKPH = 60.0",
        "  FormationSpeedKPH = 80.0",
        "  TestDayStart = 12:00",
        "  RaceLaps = 10",
        "  RaceTime = 60",
        "  NumStartingLights = 5",
        f"  Latitude = {float(line.get('lat', 0.0)):.5f}",
        f"  Longitude = {lon:.5f}",
        f"  Altitude = {float(line.get('z0', 0.0)):.1f}",
        f"  TimezoneRelativeGMT = {tz:.1f}",
        "  CutLeavingThresh = 2",
        "  CutJoiningThresh = 0.95",
        f"  SettingsFolder = {loc}",
        "}",
    ]) + "\r\n"


# ---------------------------------------------------------------------------------------------- aiw
def _resample(p, z, hl, hr, s, step, loop):
    L = float(s[-1] + (np.linalg.norm(p[0] - p[-1]) if loop else 0.0))
    n = max(int(round(L / step)), 4)
    si = np.arange(n) * (L / n) if loop else np.linspace(0.0, s[-1], n + 1)
    S = np.concatenate([s, [L]]) if loop else s
    ext = (lambda a: np.concatenate([a, a[:1]], 0)) if loop else (lambda a: a)
    P, Z, HL, HR = ext(p), ext(z), ext(hl), ext(hr)
    q = np.column_stack([np.interp(si, S, P[:, 0]), np.interp(si, S, P[:, 1])])
    return q, np.interp(si, S, Z), np.interp(si, S, HL), np.interp(si, S, HR), si, L


BRANCH_LEAD = 3.6             # a grid / pit box spot is reached by a 2-waypoint branch starting this far behind it
PIT_LANE_MARGIN = 30.0        # the pit lane starts / ends this far before / after the pit boxes (and after the grid)
PIT_BOX_GAP = 8.0             # pit boxes in single file
# the 3 driving lines of every waypoint (offset to the right of the waypoint, m), like the stock AIWs:
# path / pit lane / branch waypoints
PATH_OFFSETS = {0: (0.0, -1.5, 1.5), 1: (0.0, -1.5, 1.5), 2: (0.0, 0.0, 0.0)}


def aiw_text(line: dict, objs: list, step=4.0) -> tuple[str, dict]:
    """The .aiw: AI waypoints along the driving line (one closed path), a pit lane and the grid / pit box branches
    the way rF2's AIW editor writes them (as in the stock Joesville / Loch Drummond files), the grid / pit / garage
    spots from the AC spawn spots, sector distances. -> (text, info).

    rF2 needs the pit lane and pit boxes even when PitlaneBoundary is 0: without them the first physics step after
    loading crashes (cars "Exiting Pits Under Red", then the AI looking for their pit box). Our tracks have no separate
    pit lane, so it runs on the road beside the racing line past rF2's own pit boxes (see below).
    Waypoint links WP_PTRS=(previous, next, branch, own branch id): branch 0 = the racing path, 1 = the pit lane,
    2.. = one short non-path branch (bitfield 1) per grid spot (from the path) and per pit box (from the pit lane)."""
    p = np.asarray(line["p"], float)
    z = np.asarray(line["z"], float)
    s = np.asarray(line["s"], float)
    hl = np.asarray(line["hl"], float)
    hr = np.asarray(line["hr"], float)
    loop = bool(line.get("loop"))
    q, zq, hlq, hrq, si, L = _resample(p, z, hl, hr, s, step, loop)
    n = len(q)
    t = np.gradient(q, axis=0)
    if loop:
        t = np.roll(q, -1, 0) - np.roll(q, 1, 0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    right = np.column_stack([t[:, 1], -t[:, 0]])                        # local frame, to the driver's right
    grade = np.gradient(zq) / max(L / n, 1e-6)
    pitch = np.arctan(grade)
    s1, s2 = L * SECTOR_FRACTIONS[0], L * SECTOR_FRACTIONS[1]
    extra = float(line.get("shoulder", 0.0)) + 2.0

    def sector(d):
        return 0 if d < s1 else (1 if d < s2 else 2)

    def near(xy):
        return int(np.argmin(np.hypot(q[:, 0] - xy[0], q[:, 1] - xy[1])))

    grid = sorted([o for o in objs if o["name"].startswith("AC_START_")], key=lambda o: int(o["name"].rsplit("_", 1)[1]))
    n_pits = max(len([o for o in objs if o["name"].startswith("AC_PIT_")]), len(grid), 1)

    W = []                                         # waypoints: dict(xy, z, right, pitch, yaw, w, d, sc, br, bf, ptr)

    def add(**kw):
        W.append(kw)
        return len(W) - 1

    for k in range(n):                             # 0. the path
        add(xy=q[k], z=zq[k], right=right[k], pitch=pitch[k], yaw=yaw_of(t[k]),
            w=(hlq[k], hrq[k], hlq[k] + extra, hrq[k] + extra), sc=si[k], br=0, bf=0,
            ptr=[(k - 1) % n, (k + 1) % n, -1, 0])     # always a closed chain (rF2 laps)

    # 1. pit boxes and the pit lane. AC's pit spots are the grid spots; rF2 needs its own: rF2 finds a team's pit box
    # waypoint by its position, and one sharing a grid spot's position was not taken for a pit box ("No pit box
    # waypoints found", then a crash when the AI leave). So: boxes in single file PIT_BOX_GAP apart from
    # PIT_LANE_MARGIN after the grid (after the start line on a loop), at the right of the road; the pit lane runs
    # beside the racing line (not on it) from PIT_LANE_MARGIN before the first box to PIT_LANE_MARGIN after the last.
    Lw = L if loop else float(si[-1])
    g_s = [si[near(o["pos"][:2])] for o in grid] or [0.0]
    if loop:
        g_s = [((v + L / 2) % L) - L / 2 for v in g_s]            # behind the start line -> negative
    gap = PIT_BOX_GAP if (Lw > 4 * PIT_LANE_MARGIN + PIT_BOX_GAP * n_pits) else max(4.0, (Lw / 2) / n_pits)
    first = max(g_s) + PIT_LANE_MARGIN
    if not loop and first + gap * n_pits + PIT_LANE_MARGIN > Lw:           # no room after the grid: before it
        first = max(min(g_s) - PIT_LANE_MARGIN - gap * (n_pits - 1), PIT_LANE_MARGIN)
    box_s = [first + gap * i for i in range(n_pits)]

    def at(d):                                     # point / tangent / height / half widths at distance d
        d = d % L if loop else float(np.clip(d, 0.0, Lw))
        S = np.concatenate([si, [L]]) if loop else si
        ext = (lambda v: np.concatenate([v, v[:1]], 0)) if loop else (lambda v: v)
        xy = np.array([np.interp(d, S, ext(q)[:, 0]), np.interp(d, S, ext(q)[:, 1])])
        k = near(xy)
        return xy, t[k], right[k], float(np.interp(d, S, ext(zq))), float(min(hlq[k], hrq[k])), k, d

    half = min(at(d)[4] for d in np.linspace(box_s[0] - PIT_LANE_MARGIN, box_s[-1] + PIT_LANE_MARGIN, 12))
    lane_off = float(np.clip(0.4 * half, min(0.8, half), max(half - 1.0, min(0.8, half))))
    box_off = float(min(lane_off + 1.2, max(half - 0.5, lane_off)))
    pits = []
    for d in box_s:
        xy, tt, rr, zz, _h, _k, _d = at(d)
        p3 = xy + rr * box_off
        pits.append({"name": "rF2_PIT", "pos": [float(p3[0]), float(p3[1]), zz + 0.3], "forward": [tt[0], tt[1], 0.0]})
    k0 = near(at(box_s[0] - PIT_LANE_MARGIN)[0])
    k1 = near(at(box_s[-1] + PIT_LANE_MARGIN)[0])
    span = ((k1 - k0) % n if loop else k1 - k0) + 1
    span = int(np.clip(span, 3, n - 6))
    lane_k = [(k0 + j) % n for j in range(span)] if loop else list(range(max(k0, 1), min(k0 + span, n - 1)))
    lane = []
    for j, k in enumerate(lane_k):
        lane.append(add(xy=q[k] + right[k] * lane_off, z=zq[k], right=right[k], pitch=pitch[k], yaw=yaw_of(t[k]),
                        w=(1.5, 1.5, 1.5, 1.5), sc=si[k], br=1, bf=0, ptr=[-1, -1, k, 1]))
    for j, i in enumerate(lane):
        W[i]["ptr"][0] = lane[j - 1] if j else (lane_k[0] - 1) % n          # leaves the path before its first wp
        W[i]["ptr"][1] = lane[j + 1] if j < len(lane) - 1 else (lane_k[-1] + 1) % n   # and joins it after the last
    W[(lane_k[0] - 1) % n]["ptr"][2] = lane[0]

    def link(i, target):                           # a waypoint's branch link (one per waypoint)
        for c in (i, W[i]["ptr"][0]):
            if W[c]["ptr"][2] in (-1,) or (W[c]["br"] == 1 and W[W[c]["ptr"][2]]["br"] == 0):
                W[c]["ptr"][2] = target
                return c
        return i

    def branch(o, from_ids, bid, exit_link):
        """2 waypoints into a spot: from the waypoint before it on `from_ids` (path or pit lane)."""
        f = np.asarray(o["forward"][:2], float)
        f /= max(np.linalg.norm(f), 1e-9)
        end_xy = np.asarray(o["pos"][:2], float)
        ent_xy = end_xy - f * BRANCH_LEAD
        dd = [np.hypot(*(W[i]["xy"] - ent_xy)) for i in from_ids]
        src = from_ids[max(int(np.argmin(dd)) - 1, 0)]
        r = np.array([f[1], -f[0]])
        k = near(end_xy)
        common = dict(z=float(o["pos"][2]), right=r, pitch=0.0, yaw=yaw_of(f), w=(2.0, 2.0, 2.0, 2.0), sc=si[k],
                      br=bid, bf=1)
        end = add(xy=end_xy, ptr=[-1, -1, -1, bid], **common)          # the spot first, as the AIW editor
        ent = add(xy=ent_xy, ptr=[-1, -1, -1, bid], **common)          # writes it, then the lead-in
        W[end]["ptr"] = [ent, end, -1, bid]       # a branch ends on itself
        W[ent]["ptr"] = [link(src, ent), end, -1, bid]
        if exit_link:                             # pit boxes lead back to the pit lane a little further on
            j = from_ids.index(src)
            W[ent]["ptr"][2] = from_ids[min(j + 2, len(from_ids) - 1)]
            W[end]["ptr"][2] = from_ids[min(j + 3, len(from_ids) - 1)]
        return ent

    bid = 2
    path_ids = list(range(n))
    for o in grid:
        branch(o, path_ids, bid, False)
        bid += 1
    for o in pits:
        branch(o, lane, bid, True)
        bid += 1

    lines = []
    for wp in W:
        pos = to_rf2([wp["xy"][0], wp["xy"][1], wp["z"]])
        perp = to_rf2([wp["right"][0], wp["right"][1], 0.0])
        w = wp["w"]
        lines += [
            f"wp_pos={_v(pos, '{:.4f}')}",
            f"wp_perp={_v(perp, '{:.4f}')}",
            f"wp_normal={_v(to_rf2([0.0, 0.0, 1.0]), '{:.4f}')}",
        ]
        for pi, po in enumerate(PATH_OFFSETS[wp["br"] if wp["br"] < 2 else 2]):   # FASTEST, LEFT, RIGHT
            lines += [f"wp_pathinfo2=({pi},{po:.4f},-1.0000,-1.0000)",
                      f"wp_oriantation=({pi},{wp['pitch']:.4f},{wp['yaw']:.4f},0.0000)",
                      f"wp_pathflags=({pi},0)"]
        lines += [
            f"wp_width=({w[0]:.3f},{w[1]:.3f},{w[2]:.3f},{w[3]:.3f}) // road left, road right, far left, far right",
            f"wp_dwidth=({w[2]:.3f},{w[3]:.3f},0.000,0.000) // collision left, collision right, cut left, cut right",
            "wp_lockedAlpha=(0)",
            "wp_test_speed=(0.000000)",
            "wp_reverb=(0)",
            f"wp_score=({sector(wp['sc'])},{wp['sc']:.3f})",
            "wp_wpse=(0,0)",
            f"wp_branchID=({wp['br']})",
            f"wp_bitfields=({wp['bf']})",
            "wp_pitlane=(1)",
            "WP_PTRS=({},{},{},{})".format(*wp["ptr"]),
        ]

    def spot(o):
        pos = to_rf2(o["pos"])
        return f"Pos={_v(pos)}", f"Ori=(0.000,{yaw_of(o['forward'][:2]):.3f},0.000)"

    out = ["//[[gMa1.002f (c)2015    ]] [[            ]]", "[Features]", "pitlanes=1",
           f"startinggrid={len(grid)}", f"pitspots={len(pits)}", f"garagespots={1 if pits else 0}", "auxspots=0",
           "acceptabledriverlinenoise=1.000000", "StartingStretch=20.000000", "definepath=FASTEST",
           f"pathtime={L / 30.0:.4f}", "definepath=LEFT", f"pathtime={L / 25.0:.4f}", "definepath=RIGHT",
           f"pathtime={L / 25.0:.4f}", ""]
    for name, rows in (("GRID", grid), ("ALTGRID", grid), ("TELEPORT", grid)):
        out.append(f"[{name}]")
        for i, o in enumerate(rows):
            a, b = spot(o)
            out += [f"GridIndex={i}", a, b]
        out.append("")
    out.append("[PITS]")
    for i, o in enumerate(pits):
        pos = to_rf2(o["pos"])
        yaw = yaw_of(o["forward"][:2])
        out += [f"TeamIndex={i}", f"PitPos={_v(pos)}", f"PitOri=(0.000,{yaw:.3f},0.000)",
                f"GarPos=(0,{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})", f"GarOri=(0,0.000,{yaw:.3f},0.000)"]
    tele = near(grid[0]["pos"][:2]) if grid else 0
    out += ["", "[AUX]", "", "[Waypoint]", "trackstate=0", "drivinglines=1", f"teleportwp=({tele})",
            "pitlanepaths=(2,3)", f"number_waypoints={len(W)}", f"lap_length={L:.6f}",
            f"sector_1_length={s1:.6f}", f"sector_2_length={s2:.6f}", "LeftHandedPits=0", "FuelUse=9869.000000",
            "AIBrakingStiffness=(1.0000,1.0000,0.9000)", "slowwhenpushed=1.00", "DelayPitCrewLoad=0",
            "LaneSpacing=4.50", "ultracarefulyellow=0", "SafetyCarReleaseDist=(0.8300, 0.8800)",
            "WorstAdjust=(0.8000)", "MidAdjust=(1.0000)", "BestAdjust=(1.2000)", "AIRange=(0.1000)",
            "AISpec=(0.0000,0.0000,1.0000,0.0000)", "AIDraftStickiness=(4.0000)"]
    out += lines
    info = {"waypoints": n, "lap_length": L, "sectors": (s1, s2), "grid": len(grid), "pits": len(pits),
            "pit_lane": len(lane), "branches": bid - 2, "pit_boxes": [p["pos"] for p in pits],
            "lane_offset": lane_off}
    return "\r\n".join(out) + "\r\n", info


def gates(line: dict) -> dict:
    """Timing gates (invisible collision quads across the road, 6 m high): finish line at the start of the
    line, sectors at 1/3 and 2/3. -> {name: 4 corners (local x, y, z)}."""
    p = np.asarray(line["p"], float)
    z = np.asarray(line["z"], float)
    s = np.asarray(line["s"], float)
    hl = np.asarray(line["hl"], float)
    hr = np.asarray(line["hr"], float)
    L = float(line["length"])
    loop = bool(line.get("loop"))
    out = {}
    at = {"xfinish": 0.0 if loop else min(30.0, 0.1 * L), "xsector1": L * SECTOR_FRACTIONS[0],
          "xsector2": L * SECTOR_FRACTIONS[1]}
    for name, d in at.items():
        i = int(np.clip(np.searchsorted(s, d), 1, len(s) - 2))
        t = p[i + 1] - p[i - 1]
        t /= max(np.linalg.norm(t), 1e-9)
        nl = np.array([-t[1], t[0]])
        a = p[i] + nl * (hl[i] + 6.0)
        b = p[i] - nl * (hr[i] + 6.0)
        z0, z1 = z[i] - 2.0, z[i] + 6.0
        out[name] = [[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]]
    return out


# ---------------------------------------------------------------------------------------------- cam
_CAM = """TrackingCam={name}
{{
  Fov=(38.000000, 30.000000)
  Clear=FALSE
  Color=(0, 0, 0)
  ClipPlanes=(1.050000, 2500.000000)
  LODMultiplier=(6.000000)
  Size=(1.000000, 1.000000)
  Center=(0.500000, 0.500000)
  MipmapLODBias=(0.000000)
  Flags1=(1)
  Flags2=(0)
  SoundName=""
  SoundParams=(1.000,1,15)
  MinShadowRange=(0.100)
  MaxShadowRange=(450.000)
  ShadowSplitRatio=(0.920)
  Position={pos}
  Orientation=(0.000000, {yaw:.6f}, 0.000000)
  ActivationLocation={act}
  ActivationRadius=({radius:.6f})
  ListenerPos=(-3.402823e+038, -3.402823e+038, -3.402823e+038)
  ListenerVol=(1.200000)
  RainVol=(1.000000)
  Groups=1
  TrackingRate=(0.000000)
  PositionOffset=(0.000000, 1.000000, 0.000000)
  MovementRate=(0.000000)
}}
"""


def cam_text(line: dict, spacing=250.0) -> str:
    """TV cameras every `spacing` m beside the road, 5 m up, each active while the car is within reach."""
    p = np.asarray(line["p"], float)
    z = np.asarray(line["z"], float)
    s = np.asarray(line["s"], float)
    hl = np.asarray(line["hl"], float)
    L = float(line["length"])
    n = max(1, int(L // spacing))
    out = ["//[[gMa1.002f (c)2015    ]] [[            ]]"]
    for k in range(n):
        d = (k + 0.5) * L / n
        i = int(np.clip(np.searchsorted(s, d), 1, len(s) - 2))
        t = p[i + 1] - p[i - 1]
        t /= max(np.linalg.norm(t), 1e-9)
        nl = np.array([-t[1], t[0]])
        c = p[i] + nl * (hl[i] + 8.0)
        pos = to_rf2([c[0], c[1], z[i] + 5.0])
        act = to_rf2([p[i, 0], p[i, 1], z[i]])
        out.append(_CAM.format(name=f"TRACKING{k + 1:03d}", pos=_v(pos, "{:.6f}"), act=_v(act, "{:.6f}"),
                               yaw=yaw_of(-nl), radius=min(spacing * 0.75, 200.0)))
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- scn
def scn_text(loc: str, instances: list, sky: bool, centre=(0.0, 0.0, 0.0)) -> str:
    out = ["CUBEASF", "", f"SearchPath=.", f"SearchPath={loc}", f"SearchPath={loc}\\{loc}",
           f"SearchPath={loc}\\Assets\\GMT", f"SearchPath={loc}\\Assets\\Maps",
           f"SearchPath={loc}\\Assets\\Materials"]
    out += ["", "MASFile=COMMONMAPS.MAS", "",
            "View=mainview", "{", "  Clear=False", "  Color=(0, 0, 0)", "  Size=(1.00, 1.00) Center=(0.5, 0.5)",
            "  FOV=(77.75, 31.25)", "  ClipPlanes=(0.50, 6000.00)", "  View=rearview", "  {", "    Clear=False",
            "    Color=(0, 0, 0)", "    Size=(0.200, 0.100) Center=(0.50, 0.01)", "    FOV=(62.5, 62.5)",
            "    ClipPlanes=(0.50, 300.00)", "  }", "}", "",
            "GroupMethod=Dynamic", "MaxShadowRange=(450.00)", "AmbientColor=(126, 126, 126)",
            "FogMode=LINEAR FogIn=(50.00) FogOut=(5000.00) FogDensity=(0.00015) FogColor=(205, 215, 235)", "",
            "Light=Direct00", "{", " Type=Directional Color=(220, 220, 220) Dir=(0.5, -0.9, 0.5)", "}", ""]
    for ins in instances:
        flags = [f"MeshFile={ins['gmt']}.gmt", f"CollTarget={'True' if ins['coll'] else 'False'}",
                 f"HATTarget={'True' if ins['hat'] else 'False'}"]
        if ins.get("shadow"):
            flags.append("ShadowCaster=(Static, Solid) ShadowGroups=(12)")
        head = []
        if ins.get("render") is False:
            head.append("Render=False")
        if ins.get("lod_out"):
            head.append(f"LODOut=({ins['lod_out']:.0f})")
        if ins.get("deformable"):
            flags.insert(1, "Deformable=True")
        out += [f"Instance={ins['gmt']}", "{"]
        if head:
            out.append("  " + " ".join(head))
        out.append("  " + " ".join(flags))
        if ins.get("response"):
            out.append(f"  Response={ins['response']}")
        out += ["}", ""]
    if sky:                                   # as in Joesville (the dome is drawn by rF2's sky shader)
        out += ["Instance=SkyBoxi", "{", "  Planes=(4) ReflectPlane=(0.000, 1.000, 0.000, 0.550)",
                "  MeshFile=SkyBoxi.gmt CollTarget=False HATTarget=False Reflect=True", "}", ""]
    # reflection maps: the renderer needs STATIC01 and REFMAP0 ("Error loading background mappers for SSR. Please
    # check Static01 and Refmap0 at least, are defined"), without them the first frame on track crashes. As in the
    # stock Loch Drummond: a planar map of the walls / buildings, a static cube map at the middle of the track, a cube
    # map that follows the car, a global irradiance probe.
    planar = [i["gmt"] for i in instances if i.get("render") is not False and i["gmt"].startswith(("CWALL", "BLD_"))]
    cx, cy, cz = (float(v) for v in centre)
    out += ["ReflectionMapper=REFLECTEDENV", "{", "  Type=Planar", "  TextureSize=(1024)", "  UpdateRate=(100.000)",
            "  StaticSwitch=(500.000)", "  TrackingIns=NULL"] + [f"  IncludeIns={g}" for g in planar[:16]] + ["}", "",
            "ReflectionMapper=STATIC01", "{", "  Type=Cubic", "  TextureSize=(512)", "  UpdateRate=(0.100)",
            "  StaticSwitch=(300.000)", f"  Pos=({cx:.2f}, {cy + 5.0:.2f}, {cz:.2f})", "}", "",
            "ReflectionMapper=REFMAP0", "{", "  Type=Cubic", "  TextureSize=(1024)", "  UpdateRate=(100.000)",
            "  StaticSwitch=(300.000)", "  TrackingIns=True", "}", "",
            "IRRADIANCE_PROBE=IRRAD_GLOBAL", "{", "  INFLUENCE_RANGE = (0.0)",
            f"  PROBE_POSITION = ({cx:.2f},{cy + 5.0:.2f},{cz:.2f})", "}", ""]
    return "\r\n".join(out) + "\r\n"


# ---------------------------------------------------------------------------------------------- textures
def dds_rgba(img):
    """Texture with alpha -> uncompressed 32-bit BGRA DDS with mips (no encoder needed; cut-outs stay sharp:
    alpha coverage is kept per mip by scaling)."""
    im = img.convert("RGBA")
    w, h = im.size
    w2 = int(min(2048, max(4, 2 ** round(np.log2(w)))))
    h2 = int(min(2048, max(4, 2 ** round(np.log2(h)))))
    if (w2, h2) != (w, h):
        im = im.resize((w2, h2), Image.LANCZOS)
    a0 = np.asarray(im, np.float32)[..., 3]
    cov0 = float((a0 > 127).mean())
    levels, cw, ch = [], w2, h2
    cur = im
    while True:
        arr = np.asarray(cur, np.uint8).copy()
        if levels and cov0 > 0:                       # keep the alpha-test coverage of the full-size image
            a = arr[..., 3].astype(np.float32)
            lo, hi = 0.25, 4.0
            for _ in range(12):
                m = (lo + hi) / 2
                if float((np.clip(a * m, 0, 255) > 127).mean()) < cov0:
                    lo = m
                else:
                    hi = m
            arr[..., 3] = np.clip(a * hi, 0, 255).astype(np.uint8)
        levels.append(arr[..., [2, 1, 0, 3]].tobytes())
        if cw <= 1 and ch <= 1:
            break
        cw, ch = max(1, cw // 2), max(1, ch // 2)
        cur = im.resize((cw, ch), Image.BOX)
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x8          # caps, height, width, pixelformat, mipcount, pitch
    header = struct.pack("<4sIIIIIII44x", b"DDS ", 124, flags, h2, w2, w2 * 4, 0, len(levels))
    header += struct.pack("<II4sIIIII", 32, 0x41, b"\0\0\0\0", 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    header += struct.pack("<IIIII", 0x1000 | 0x8 | 0x400000, 0, 0, 0, 0)
    return header + b"".join(levels)


def write_textures(tex_dir: Path, names, maps_dir: Path, log=print):
    """PNG textures of a generation -> DDS in Assets\\Maps (BC1 opaque, 32-bit with alpha). -> {name: dds}."""
    from .ac_export import dds_bc1

    maps_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in sorted(set(names)):
        src = tex_dir / f"{name}.png"
        if not src.exists():
            continue
        img = Image.open(src)
        img.load()
        alpha = img.mode in ("RGBA", "LA") and np.asarray(img.convert("RGBA"))[..., 3].min() < 250
        data = dds_rgba(img) if alpha else dds_bc1(img)
        fn = f"{name}.dds"
        (maps_dir / fn).write_bytes(data)
        out[name] = {"dds": fn, "alpha": bool(alpha)}
    log(f"  {len(out)} textures as DDS")
    return out


# ---------------------------------------------------------------------------------------------- menu images
RF2_FORMAT = 8                # bump when the location files change (a ModDev copy of an older format is rebuilt)
MARKER = "maptotrack_rf2.json"


def _map_polylines(line: dict) -> list:
    """The track (or every street of an area) as polylines in local x/y."""
    if line.get("map"):
        return [np.asarray(m, float) for m in line["map"] if len(m) > 1]
    p = np.asarray(line["p"], float)
    return [np.vstack([p, p[:1]]) if line.get("loop") else p]


def _draw_map(size, polys, color, width, pad, bg=None, start=None):
    """polylines fitted into an RGBA image of `size` (north up)."""
    from PIL import ImageDraw

    w, h = size
    s = 4                                      # drawn 4x larger, then reduced (smooth lines)
    im = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if bg:
        d.rounded_rectangle([0, 0, w * s - 1, h * s - 1], radius=10 * s, fill=bg)
    allp = np.vstack(polys)
    lo, hi = allp.min(0), allp.max(0)
    span = np.maximum(hi - lo, 1.0)
    k = min((w - 2 * pad) / span[0], (h - 2 * pad) / span[1]) * s
    off = np.array([w * s, h * s]) / 2 - (lo + hi) / 2 * np.array([k, -k])

    def px(q):
        q = np.asarray(q, float)
        return [tuple(v) for v in np.column_stack([q[:, 0] * k + off[0], -q[:, 1] * k + off[1]])]

    r = width * s / 2
    for q in polys:
        pts = px(q)
        d.line(pts, fill=color, width=int(width * s))
        for x, y in pts:                       # round joints (PIL's own leave notches on thick lines)
            d.ellipse([x - r, y - r, x + r, y + r], fill=color)
    if start is not None:                      # the start: a small white dot with a dark ring
        x, y = px([start[0]])[0]
        rr = 0.8 * width * s
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(255, 255, 255, 255), outline=(20, 20, 20, 255), width=s * 2)
    return im.resize((w, h), Image.LANCZOS)


def write_menu_images(lay: Path, loc: str, line: dict, preview: Path | None = None, log=print):
    """The track's pictures in rF2's menus, named like the stock tracks: <layout>icon.tga (102 x 87, the track on a
    red tile), <layout>Thmb.tga (190 x 128, the track in yellow) and <layout>_loading.jpg (1920 x 1080, the
    generated preview)."""
    polys = _map_polylines(line)
    p = np.asarray(line["p"], float)
    start = (p[0], p[min(5, len(p) - 1)])
    _draw_map((102, 87), polys, (255, 255, 255, 255), 4.5 if not line.get("map") else 2.0, 14,
              bg=(170, 12, 12, 255)).save(lay / f"{loc}icon.tga")
    _draw_map((190, 128), polys, (255, 220, 0, 255), 9 if not line.get("map") else 3.5, 14,
              start=start).save(lay / f"{loc}Thmb.tga")
    if preview and Path(preview).exists():
        im = Image.open(preview).convert("RGB")
        r = 16 / 9
        w, h = im.size
        if w / h > r:
            cw = int(h * r)
            im = im.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h))
        else:
            ch = int(w / r)
            im = im.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch))
        im.resize((1920, 1080), Image.LANCZOS).save(lay / f"{loc}_loading.jpg", quality=90)
    else:
        big = _draw_map((1920, 1080), polys, (255, 220, 0, 255), 14, 120, bg=(25, 28, 32, 255))
        big.convert("RGB").save(lay / f"{loc}_loading.jpg", quality=90)


# ---------------------------------------------------------------------------------------------- the location
def gmt_name(mesh: str, used: set) -> str:
    """GMT / instance name from a mesh name: letters, digits, underscores, at most 31 characters, unique."""
    n = re.sub(r"[^A-Za-z0-9_]+", "_", mesh)
    if n[:1].isdigit():
        n = "C" + n[1:]                                    # 1ROAD_GND_00_01 -> CROAD_GND_00_01 (collision)
    n = n[:31]
    base, k = n, 1
    while n.upper() in used:
        k += 1
        n = f"{base[:27]}_{k}"
    used.add(n.upper())
    return n


GMT_MAX_TRIS = 21000          # the exporter writes 3 vertices per triangle; a GMT holds at most 65536 (uint16)
ALPHA_MATERIALS = {"trees", "veg_grass", "veg_grass_short", "props", "tyres"}
# how shiny each kind of material is: (specular colour, specular power, fresnel max, double sided). The exporter's
# default (0.75 / 5 / fresnel 4) makes grass and ground glare lime-white at a low sun and at grazing angles.
SHADING = {"asphalt": (0.22, 10, 0.5, False), "paved": (0.22, 10, 0.5, False), "concrete": (0.15, 8, 0.4, False),
           "ground": (0.03, 2, 0.1, False), "near": (0.03, 2, 0.1, False), "gravel": (0.04, 2, 0.12, False),
           "trees": (0.02, 2, 0.08, True), "veg": (0.02, 2, 0.08, True), "siding": (0.10, 6, 0.3, False),
           "shingle": (0.08, 4, 0.25, False), "props": (0.15, 8, 0.3, False), "tyres": (0.12, 6, 0.25, False)}


# PBR (rF2's IBL standard shader, as in the stock tracks): (roughness, specular) per kind of material
PBR = {"asphalt": (0.72, 0.5), "paved": (0.75, 0.5), "concrete": (0.8, 0.5), "ground": (0.95, 0.25),
       "near": (0.95, 0.25), "gravel": (0.95, 0.3), "trees": (0.85, 0.3), "veg": (0.85, 0.3), "siding": (0.8, 0.5),
       "shingle": (0.85, 0.4), "props": (0.6, 0.5), "tyres": (0.9, 0.4)}
USE_PBR = True


def shading_of(material: str) -> dict:
    m = material.lower()
    for key in ("paved", "asphalt", "concrete", "gravel", "ground", "near", "trees", "veg", "siding", "shingle",
                "props", "tyres"):
        if key in m:
            sp, pw, fr, ds = SHADING[key]
            rough, spec = PBR[key]
            break
    else:
        sp, pw, fr, ds = 0.08, 4, 0.25, False
        rough, spec = 0.8, 0.5
    return {"spec": sp, "power": pw, "fresnel": fr, "double": ds, "pbr": USE_PBR, "roughness": rough,
            "specular": spec}


def _tex_slot(name, texture="", stage="CUBETST_BUMPMAP", channel=0, chroma=False):
    active = bool(texture)
    flags = ["CUBETF_AUTOMIPMAP", "CUBETF_TRILINEAR"] + (["CUBETF_CHROMAKEY"] if chroma else []) + \
        ["CUBETF_OWNEDBYMATERIAL", "CUBETF_NOZBUFFERREQUIRED"]
    return {"AnisoLevel": 0, "ChromaColor": 0, "Flags": flags if active else None, "FrameCount": 1, "MipBias": 0,
            "MipLevels": -1 if active else 1, "Name": name, "ShaderPass": 0, "ShadowMipBias": 0, "StageType": stage,
            "TexChannel": channel, "Texture": texture, "hasDefault": active, "inactive": not active, "runtime": False}


def material_json(name: str, texture: str, shading: dict, cutout: bool) -> dict:
    """rF2's material file (Assets/Materials/<NAME>.json, what the in-game material editor writes) for an IBL
    standard material: our colour map, no other maps, roughness / specular as constants. Same layout as the stock
    IBL_STANDARD.json of Loch Drummond."""
    one = [1, 1, 1, 1]
    flags = {k: "@Undef" for k in ("AOSPECALTMAP", "AOSPECMAP", "BILLBOARD", "DETAILAO", "DETAILMAP", "DETAILNORMAL",
                                   "DETAILSPEC", "EMISSIVEMAP", "EMISSIVEMODE", "METALLICROUGHNESSMAP", "NORMALMAP",
                                   "RETROREFLECTOR", "TRANSLUCENCY", "TRANSLUCENCYALBEDO", "VERTEXALPHA",
                                   "VERTEXCOLOR")}
    flags.update({"ALBEDOMAP": "@Null", "ALBEDOUV": "UV_A", "AOSPECALTUV": "UV_A", "AOSPECUV": "UV_C",
                  "DETAILALBEDO": "@Null", "DETAILMETALLIC": "@Null", "DETAILROUGHNESS": "@Null", "DETAILUV": "UV_B",
                  "EMISSIVEUV": "UV_C", "METALROUGHUV": "UV_A", "NORMALUV": "UV_A"})
    return {
        "dstBlend": "CUBEAB_ZERO",
        "flags": ["CUBEMAT_SPECULAR", "CUBEMAT_RELIGHT", "CUBEMAT_AMBIENTMAT", "CUBEMAT_DIFFUSEMAT",
                  "CUBEMAT_SPECLARDIR", "CUBEMAT_EMISSIVMAT", "CUBEMAT_EMISSIVDIR"]
                 + (["CUBEMAT_TWOSIDED"] if shading.get("double") else []),
        "name": name,
        "parameters": {
            "StandardInfo": {"ambientColour": one, "blendPercent": 0, "bumpScale": 1,
                             "cubeMapFresnel": {"Exp": 4, "Max": 1, "Min": 0}, "diffuseColour": one,
                             "emissiveColour": [0, 0, 0, 1], "refEnhance": 0,
                             "specFresnel": {"Exp": 4, "Max": 1, "Min": 0}, "specPower": 1, "specularColour": one},
            "cbParameters": {
                "Base_Material": {"AO": 1, "Metallic": 0, "Roughness": shading["roughness"],
                                  "Specular": shading["specular"]},
                "Material_Adjust": {"AlbedoTint": one, "Metallic": 0.5, "NormalIntensity": 1,
                                    "NormalShadowInfluence": 0.5, "Roughness": 0.5, "Specular": 1},
                "UV_Scale": {"Albedo": [1, 1]}}},
        "preprocessorFlags": dict(sorted(flags.items())),
        "refName": "STATIC01",
        "shader": "L2IBLSTANDARD",
        "sortOffset": 0,
        "srcBlend": "CUBEAB_ONE",
        "textures": [_tex_slot("albedoMap", texture, "CUBETST_DIFFUSEMAP", 0, cutout)]
                    + [_tex_slot(f":{i}") for i in range(2, 7)],
    }


def load_meshes(glb_path):
    """The generated model (track.glb) -> {mesh name: (vertices local x/y/z, faces, uv (OpenGL v), material)}."""
    import trimesh

    sc = trimesh.load(str(glb_path), force="scene", process=False)
    out = {}
    for name, g in sc.geometry.items():
        v = np.asarray(g.vertices, float)
        loc = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])          # glTF (x, up, south) -> local (x, north, up)
        uv = getattr(g.visual, "uv", None)
        uv = np.zeros((len(v), 2)) if uv is None else np.asarray(uv, float)
        mat = getattr(getattr(g.visual, "material", None), "name", None) or name
        out[name] = (loc, np.asarray(g.faces, np.int64), uv, mat)
    return out


def chunks(faces, max_tris=GMT_MAX_TRIS):
    """Face index ranges of at most max_tris triangles."""
    n = len(faces)
    k = max(1, int(math.ceil(n / max_tris)))
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def plan(report: dict, line: dict, meshes: dict) -> dict:
    """What the GMT exporter builds and the .scn lists: one GMT per mesh of the generated model (big meshes cut
    into GMTs of at most GMT_MAX_TRIS triangles)."""
    used = set()
    gmts, instances = [], []
    for m in report.get("meshes", []):
        name = m["name"]
        if name not in meshes:
            continue
        v, f, uv, mat = meshes[name]
        mat = m.get("material") or mat
        code = surface_of(name, mat)
        rf2_mat = (f"{code}_{mat}" if code else mat)[:60].upper()   # upper case, like the stock materials
        lod = view_distance(name)
        base = gmt_name(name, used)
        parts = chunks(f)
        for k, (a, b) in enumerate(parts):
            g = base if k == 0 else gmt_name(f"{base}_{k}", used)
            gmts.append({"mesh": name, "gmt": g, "faces": (a, b), "material": mat, "rf2_material": rf2_mat,
                         "texture": mat, "lod_out": lod, "shading": shading_of(mat)})
            instances.append({"gmt": g, "coll": code is not None,
                              "hat": code is not None and code not in ("mtwall", "mttyre"),
                              "deformable": code == "mtroad",
                              "shadow": name.startswith(("BLD_", "1WALL_BLD", "TREES", "PROPS", "PYLONS")),
                              "lod_out": lod})
    for gname in gates(line):
        instances.append({"gmt": gname, "coll": True, "hat": False, "render": False,
                          "response": "VEHICLE,TIMING"})
    return {"meshes": gmts, "instances": instances}


SKY_GMT = "skyboxi.gmt"
SKY_MAPS = ("BKA.DDS",)       # the sky dome's texture (material BKA)


def _find_ci(folder: Path, name: str):
    if not folder.is_dir():
        return None
    return next((f for f in folder.iterdir() if f.name.lower() == name.lower()), None)


def copy_sky(root: Path, rf2_root, log=print) -> bool:
    """Copy the stock sky dome of the Joesville sample (ModDev) and its texture into the location, so the track
    doesn't need Joesville once packaged. Also repairs a location copied without the texture. -> sky available."""
    jv = Path(rf2_root) / "ModDev" / "Locations" / "Joesville" / "Assets"
    gmt = _find_ci(jv / "GMT", SKY_GMT)
    maps = [_find_ci(jv / "Maps", n) for n in SKY_MAPS]
    if gmt is None or any(m is None for m in maps):
        return False
    (root / "Assets" / "GMT").mkdir(parents=True, exist_ok=True)
    (root / "Assets" / "Maps").mkdir(parents=True, exist_ok=True)
    if _find_ci(root / "Assets" / "GMT", SKY_GMT) is None:
        shutil.copyfile(gmt, root / "Assets" / "GMT" / SKY_GMT)
    for m in maps:
        if _find_ci(root / "Assets" / "Maps", m.name) is None:
            shutil.copyfile(m, root / "Assets" / "Maps" / m.name)
            log(f"  sky texture {m.name} added")
    return True


def write_location(out_dir: Path, slug: str, report: dict, objs: list, line: dict, tex_dir: Path,
                   rf2_root: Path | None = None, glb_path: Path | None = None, meshes: dict | None = None,
                   preview: Path | None = None, log=print) -> dict:
    """Everything of the rF2 location except the .gmt files, plus what the GMT exporter reads: rf2_src/*.npz
    (vertices in the local frame, triangles, UVs) and the manifest."""
    loc = loc_name(slug)
    root = out_dir / loc
    lay = root / loc
    gmt_dir = root / "Assets" / "GMT"
    maps = root / "Assets" / "Maps"
    src_dir = out_dir / "rf2_src"
    for d in (lay, gmt_dir, maps, src_dir):
        d.mkdir(parents=True, exist_ok=True)
    if meshes is None:
        meshes = load_meshes(glb_path)
    pl = plan(report, line, meshes)
    sky = copy_sky(root, rf2_root, log=log) if rf2_root else False
    tex = write_textures(tex_dir, [m["texture"] for m in pl["meshes"]], maps, log=log)
    for m in pl["meshes"]:
        t = tex.get(m["texture"])
        m["dds"] = t["dds"] if t else "white.dds"
        m["alpha"] = bool((t and t["alpha"]) or m["material"] in ALPHA_MATERIALS)
        v, f, uv, _ = meshes[m["mesh"]]
        a, b = m["faces"]
        ff = f[a:b]
        used, inv = np.unique(ff, return_inverse=True)
        npz = src_dir / f"{m['gmt']}.npz"
        np.savez_compressed(npz, v=v[used].astype(np.float32), f=inv.reshape(-1, 3).astype(np.int32),
                            uv=uv[used].astype(np.float32))
        m["npz"] = str(npz)
        m.pop("faces")
    mats_dir = root / "Assets" / "Materials"
    done_mats = set()
    for m in pl["meshes"]:                         # PBR material files, one per material
        if m["shading"].get("pbr") and m["rf2_material"] not in done_mats:
            done_mats.add(m["rf2_material"])
            mats_dir.mkdir(parents=True, exist_ok=True)
            js = material_json(m["rf2_material"], Path(m["dds"]).stem, m["shading"], m["alpha"])
            (mats_dir / f"{m['rf2_material']}.json").write_text(json.dumps(js, indent=2), encoding="utf-8")
    if not (maps / "white.dds").exists():
        from .ac_export import dds_bc1

        (maps / "white.dds").write_bytes(dds_bc1(Image.new("RGB", (8, 8), (255, 255, 255))))
    (root / f"{loc}.tdf").write_text(tdf_text(loc), encoding="utf-8", newline="")
    aiw, info = aiw_text(line, objs)
    (lay / f"{loc}.aiw").write_text(aiw, encoding="utf-8", newline="")
    (lay / f"{loc}.gdb").write_text(gdb_text(loc, line, info["grid"]), encoding="utf-8", newline="")
    (lay / f"{loc}.cam").write_text(cam_text(line), encoding="utf-8")
    _p = np.asarray(line["p"], float)
    centre = to_rf2([float(_p[:, 0].mean()), float(_p[:, 1].mean()), float(np.mean(line["z"]))])
    (lay / f"{loc}.scn").write_text(scn_text(loc, pl["instances"], sky, centre), encoding="utf-8", newline="")
    write_menu_images(lay, loc, line, preview, log=log)
    (root / MARKER).write_text(json.dumps({"format": RF2_FORMAT, "slug": slug}), encoding="utf-8")
    manifest = {"loc": loc, "gmt_dir": str(gmt_dir), "axes": list(RF2_AXES), "meshes": pl["meshes"],
                "gates": gates(line)}
    log(f"  rF2 location {loc}: {len(pl['meshes'])} GMTs from {len(report.get('meshes', []))} meshes, "
        f"{info['waypoints']} AI waypoints, {info['grid']} grid / {info['pits']} pit spots"
        f"{'' if sky else ' (no sky: Joesville not found)'}")
    return {"loc": loc, "root": root, "manifest": manifest, "info": info}


def install(root: Path, rf2_root: Path, log=print) -> Path:
    """Copy the location into <rF2>\\ModDev\\Locations (replacing an older copy of the same track)."""
    dest = Path(rf2_root) / "ModDev" / "Locations" / root.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(root, dest)
    log(f"  installed to {dest}")
    return dest


# ---------------------------------------------------------------------------------------------- .gmt files
GMT_CACHE_VERSION = 3         # bump when blender/export_rf2_gmt.py writes different .gmt files
GMT_WORKER_TRIS = 60000       # at least this much work per extra Blender (each start costs a few seconds)


def addon_dir(zip_path: str) -> Path:
    """Studio 397's exporter zip unpacked once into cache/rf2_addon/ (loaded from there by each Blender, nothing is
    installed into the user's Blender)."""
    import zipfile

    from . import config

    st = os.stat(zip_path)
    d = config.CACHE_DIR / "rf2_addon" / f"{Path(zip_path).stem}_{st.st_size}_{int(st.st_mtime)}"
    if not (d / ".ok").exists():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(d)
        (d / ".ok").write_text("ok")
    return d


def _gmt_key(item: dict, addon: str) -> str:
    """What a .gmt depends on: its geometry, name, material and texture, the exporter and this app's script."""
    import hashlib

    h = hashlib.sha1()
    h.update(json.dumps([GMT_CACHE_VERSION, Path(addon).name, os.path.getsize(addon), item.get("gmt"),
                         item.get("rf2_material"), item.get("dds"), bool(item.get("alpha")), item.get("lod_out"),
                         item.get("corners"), item.get("shading")], sort_keys=True).encode())
    if item.get("npz"):
        h.update(Path(item["npz"]).read_bytes())
    return h.hexdigest()


def _split(items: list, k: int) -> list:
    """Items into k lists of about equal triangle count (largest first, each to the lightest list)."""
    bins = [[0, []] for _ in range(k)]
    for it in sorted(items, key=lambda i: -i["tris"]):
        b = min(bins, key=lambda b: b[0])
        b[0] += it["tris"]
        b[1].append(it)
    return [b[1] for b in bins if b[1]]


def write_gmts(manifest: dict, stage: Path, cache: Path, blender: str, addon: str, log=print) -> int:
    """All .gmt files of the manifest: unchanged ones from the cache (a re-export after changing only the track's
    text files needs no Blender), the others written by several Blenders at once (the exporter is single-threaded
    Python, ~20 000 triangles a second)."""
    import subprocess
    import time

    from . import config

    gmt_dir = Path(manifest["gmt_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    items = []
    for m in manifest["meshes"]:
        items.append(dict(m, kind="mesh", tris=int(np.load(m["npz"])["f"].shape[0]) if m.get("npz") else 0))
    for g, corners in manifest.get("gates", {}).items():
        items.append({"kind": "gate", "gmt": g, "corners": corners, "tris": 4})
    todo, hit = [], 0
    for it in items:
        it["key"] = _gmt_key(it, addon)
        c = cache / f"{it['key']}.gmt"
        if c.exists():
            shutil.copyfile(c, gmt_dir / f"{it['gmt']}.gmt")
            hit += 1
        else:
            todo.append(it)
    if hit:
        log(f"  {hit} .gmt files unchanged (from the cache)")
    if todo:
        total = sum(i["tris"] for i in todo)
        k = max(1, min((os.cpu_count() or 2) - 1, 8, int(math.ceil(total / GMT_WORKER_TRIS)), len(todo)))
        adir = addon_dir(addon)
        script = config.BLENDER_TEMPLATES / "export_rf2_gmt.py"
        log(f"Writing {len(todo)} .gmt files ({total / 1000:.0f}k triangles) with {k} x {Path(blender).name} ...")
        t0 = time.time()
        procs = []
        for w, part in enumerate(_split(todo, k), 1):
            man = {"loc": manifest["loc"], "gmt_dir": str(gmt_dir), "axes": manifest.get("axes"),
                   "meshes": [{kk: v for kk, v in i.items() if kk not in ("kind", "tris", "key")}
                              for i in part if i["kind"] == "mesh"],
                   "gates": {i["gmt"]: i["corners"] for i in part if i["kind"] == "gate"}}
            mp = stage / f"rf2_meshes_{w}.json"
            mp.write_text(json.dumps(man), encoding="utf-8")
            lp = open(stage / f"rf2_blender_{w}.log", "w", encoding="utf-8", errors="replace")
            procs.append((subprocess.Popen([blender, "--background", "--factory-startup", "--python", str(script),
                                            "--", "--manifest", str(mp), "--addon-dir", str(adir),
                                            "--worker", str(w)],
                                           cwd=str(stage), stdout=lp, stderr=subprocess.STDOUT), lp, w))
        last = 0
        while any(p.poll() is None for p, _l, _w in procs):
            time.sleep(1.0)
            done = sum(1 for i in todo if (gmt_dir / f"{i['gmt']}.gmt").exists())
            if done - last >= max(5, len(todo) // 10):
                log(f"  {done}/{len(todo)} .gmt files, {time.time() - t0:.0f} s")
                last = done
            if time.time() - t0 > 3600:
                for p, _l, _w in procs:
                    p.kill()
                raise RuntimeError("Blender took more than an hour writing the .gmt files.")
        failed = False
        for p, lp, w in procs:
            lp.close()
            txt = (stage / f"rf2_blender_{w}.log").read_text(encoding="utf-8", errors="replace")
            for ln in txt.splitlines():
                if "Warning" in ln or "maptotrack-progress" in ln:
                    continue
                if ln.startswith("maptotrack:") or "Error" in ln or "Traceback" in ln:
                    log(ln)
            failed |= p.returncode != 0
        missing = [i["gmt"] for i in todo if not (gmt_dir / f"{i['gmt']}.gmt").exists()]
        if failed or missing:
            raise RuntimeError(f"Blender failed writing {len(missing) or 'some'} .gmt files - see the log "
                               f"(rf2_blender_*.log in {stage}).")
        for i in todo:
            shutil.copyfile(gmt_dir / f"{i['gmt']}.gmt", cache / f"{i['key']}.gmt")
        log(f"  {len(todo)} .gmt files written in {time.time() - t0:.0f} s")
    keep = {f"{i['key']}.gmt" for i in items}            # the cache holds this track's current files only
    for f in cache.glob("*.gmt"):
        if f.name not in keep:
            f.unlink()
    return len(list(gmt_dir.glob("*.gmt")))


# ---------------------------------------------------------------------------------------------- the whole export
def export(slug: str, settings: dict, install_to_rf2=True, log=print) -> dict:
    """A generated project -> rF2 location in output/<slug>/rf2/ (+ ModDev install). The .gmt files are written by
    Studio 397's exporter in Blender 5; without it everything else is still written and the result says so."""
    import os
    import subprocess

    from . import config

    out = config.OUTPUT_DIR / slug
    bl = out / "blender"
    rep_p, obj_p, line_p = out / "report.json", bl / "ac_objects.json", bl / "track_line.json"
    for p in (rep_p, obj_p, line_p, bl / "track.glb"):
        if not p.exists():
            raise RuntimeError(f"Generate the track again first ({p.name} is missing - made by an older version).")
    report = json.loads(rep_p.read_text(encoding="utf-8"))
    objs = json.loads(obj_p.read_text(encoding="utf-8"))
    line = json.loads(line_p.read_text(encoding="utf-8"))
    rf2_root = Path(settings.get("rf2_path") or "")
    rf2_ok = bool(settings.get("rf2_path")) and (rf2_root / "ModDev").is_dir()
    stage = out / "rf2"
    if stage.exists():
        shutil.rmtree(stage)
    log("rFactor 2: track files ...")
    res = write_location(stage, slug, report, objs, line, bl / "texture", rf2_root if rf2_ok else None,
                         glb_path=bl / "track.glb", preview=out / "preview.png", log=log)
    man_p = stage / "rf2_meshes.json"
    man_p.write_text(json.dumps(res["manifest"], indent=1), encoding="utf-8")

    missing = []
    addon = config.find_rf2_addon()
    blender = config.rf2_blender(settings, addon) if addon else ""
    if not addon:
        missing.append("Studio 397's exporter zip (io_rfactor2_gmt*.zip) in the maptotrack folder")
    elif not blender:
        tried = []
        for p in config._blender_candidates(settings):
            v, py = config.blender_version(p), config.blender_python(p)
            if v:
                tried.append(f"{Path(p).parent.name or p}: Blender {v[0]}.{v[1]}"
                             + (f", Python {py[0]}.{py[1]}" if py else ""))
        missing.append(f"{config.rf2_blender_needed(addon)} for the .gmt exporter - set it in Settings, or unzip the "
                       "portable blender-5.0.x-windows-x64.zip into the maptotrack folder"
                       + (f" (found: {'; '.join(tried)})" if tried else ""))
    n_gmt = 0
    if not missing:
        n_gmt = write_gmts(res["manifest"], stage, out / "rf2_gmt_cache", blender, addon, log=log)
    else:
        log("  .gmt files NOT written - missing: " + "; ".join(missing))
    dest = None
    if install_to_rf2 and rf2_ok and n_gmt:
        dest = install(res["root"], rf2_root, log=log)
    elif install_to_rf2 and not rf2_ok:
        log("  not installed: set the rFactor 2 folder in Settings")
    return {"loc": res["loc"], "folder": str(res["root"]), "installed": str(dest) if dest else None,
            "gmt": n_gmt, "missing": missing, **{k: v for k, v in res["info"].items() if k != "sectors"}}


# ---------------------------------------------------------------------------------------------- packaging
# Outside Mod Mode rF2 only loads installed components: .mas archives + a signed manifest in
# <rF2>\Installed\Locations\<name>\<version>, installed from an .rfcmp package. The .mas files are made with ModMgr's
# command line (-m). The .rfcmp can only be made in Studio 397's MAS2 tool (its header is not documented), so the app
# writes the component (name, version, .mas list, output file) into MAS2's own list (%APPDATA%\cmpinfo.dat, plain
# text), opens MAS2 with it selected, waits for the .rfcmp and installs it with `ModMgr -i`.
LOCATION_TYPE = 1             # cmpinfo.dat / .mft Type of a location component
PACKAGE_WAIT_S = 20 * 60


def _exe(args: list) -> list:
    """Windows programs; under MAPTOTRACK_WINE (tests on Linux) through wine."""
    import os

    w = os.environ.get("MAPTOTRACK_WINE")
    return ([w] if w else []) + [str(a) for a in args]


def _win(p) -> str:
    """A path as the Windows programs see it (tests: a wine C: drive given by MAPTOTRACK_WINE_C)."""
    import os

    p = str(p)
    c = os.environ.get("MAPTOTRACK_WINE_C")
    if c and p.startswith(c):
        return "C:" + p[len(c):].replace("/", "\\")
    return p


def _run(args, timeout=1800, **kw):
    import subprocess
    import sys

    flags = 0x08000000 if sys.platform == "win32" else 0          # CREATE_NO_WINDOW
    return subprocess.run(_exe(args), capture_output=True, text=True, timeout=timeout, creationflags=flags, **kw)


def mas_groups(loc_dir: Path, loc: str) -> dict:
    """Which files go into which .mas (Studio 397's layout for a track component: overall files, GMTs, maps, and one
    mas per layout with its .gdb/.scn/.aiw/.cam, menu pictures and the collision cache rF2 builds in Mod Mode)."""
    lay = loc_dir / loc
    return {f"{loc}.mas": [loc_dir / f"{loc}.tdf"],
            f"{loc}_gmts.mas": sorted((loc_dir / "Assets" / "GMT").glob("*.gmt")),
            f"{loc}_maps.mas": sorted(f for f in (loc_dir / "Assets" / "Maps").iterdir() if f.is_file()),
            f"{loc}_materials.mas": sorted((loc_dir / "Assets" / "Materials").glob("*.json"))
            if (loc_dir / "Assets" / "Materials").is_dir() else [],
            f"{loc}_layout.mas": sorted(f for f in lay.iterdir() if f.is_file())}


def make_mas(modmgr: Path, name: str, files: list, out_dir: Path, work: Path, log=print) -> Path:
    """`ModMgr -m<name> *.*` over a folder holding just these files -> out_dir/<name>."""
    import os

    src = work / name.replace(".", "_")
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    total = 0
    for f in files:
        try:
            os.link(f, src / f.name)               # same drive: no copy
        except OSError:
            shutil.copyfile(f, src / f.name)
        total += f.stat().st_size
    for old in out_dir.glob("*"):
        if old.name.lower() == name.lower():
            old.unlink()
    p = _run([modmgr, f"-c{_win(src)}", f"-o{_win(out_dir)}", f"-m{name}", "*.*"], cwd=str(modmgr.parent.parent))
    shutil.rmtree(src, ignore_errors=True)
    out = next((f for f in out_dir.iterdir() if f.name.lower() == name.lower()), None)
    if out is None or out.stat().st_size < max(64, total // 100):
        raise RuntimeError(f"ModMgr did not write {name} ({(p.stdout + p.stderr).strip()[-300:]})")
    if out.name != name:                           # ModMgr writes the name in lower case
        out = out.rename(out_dir / name)
    log(f"  {name}: {len(files)} files, {out.stat().st_size / 1e6:.1f} MB")
    return out


def installed_versions(rf2_root: Path, name: str) -> list:
    d = Path(rf2_root) / "Installed" / "Locations" / name
    out = []
    for v in (d.iterdir() if d.is_dir() else []):
        try:
            out.append((float(v.name), v))
        except ValueError:
            pass
    return sorted(out)


def next_version(rf2_root: Path, name: str, last: str | None) -> str:
    vs = [v for v, _ in installed_versions(rf2_root, name)] + ([float(last)] if last else [])
    return f"{(max(vs) + 0.01) if vs else 1.0:.2f}"


def cmpinfo_text(old: str, comp: dict) -> str:
    """MAS2's component list with `comp` added (or replacing the one of the same name) and selected."""
    blocks, cur = [], None
    for ln in old.replace("\r\n", "\n").split("\n"):
        if ln.strip() == "[Component]":
            cur = [ln]
            blocks.append(cur)
        elif ln.startswith("CurComponent="):
            cur = None
        elif cur is not None and ln.strip():
            cur.append(ln)
    blocks = [b for b in blocks if f"Name={comp['name']}" not in b]
    blocks.append(["[Component]", f"Name={comp['name']}", f"Type={LOCATION_TYPE}", f"Version={comp['version']}",
                   "BaseVersion=", "MinVersion=", f"Author={comp.get('author', '')}", "Date=0", "ID=", "URL=",
                   f"Desc={comp.get('desc', '')}", "Category=0", "Origin=0", "Flags=0", "CurLocation=0",
                   "NumLocations=1", f"Location={comp['rfcmp']}", f"NumMASFiles={len(comp['mas'])}"]
                  + [f"MASFile={m}" for m in comp["mas"]] + ["rFmFile=", "IconFile=", "SmallIconFile="])
    return "".join("\r\n".join(b) + "\r\n\r\n" for b in blocks) + f"CurComponent={len(blocks) - 1}\r\n"


def _mas2_running() -> bool:
    import sys

    if sys.platform != "win32":
        return False
    try:
        out = _run(["tasklist", "/FI", "IMAGENAME eq MAS2.exe", "/NH"], timeout=30).stdout
        return "mas2.exe" in out.lower()
    except Exception:
        return False


def package(slug: str, settings: dict, log=print, wait_s: float = PACKAGE_WAIT_S) -> dict:
    """The track as an installed rF2 component (drivable outside Mod Mode) and a .rfcmp to share."""
    import os
    import subprocess
    import time

    from . import config

    rf2_root = Path(settings.get("rf2_path") or "")
    modmgr, mas2 = rf2_root / "Bin64" / "ModMgr.exe", rf2_root / "Support" / "Tools" / "MAS2.exe"
    if not (settings.get("rf2_path") and modmgr.exists() and mas2.exists()):
        raise RuntimeError("Set the rFactor 2 folder in Settings (it needs Bin64\\ModMgr.exe and "
                           "Support\\Tools\\MAS2.exe).")
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is not set (MAS2 keeps its package list there).")
    if _mas2_running():
        raise RuntimeError("Close MAS2 first, then click Package again (MAS2 rewrites its package list on exit).")
    out = config.OUTPUT_DIR / slug
    loc = loc_name(slug)
    dev = rf2_root / "ModDev" / "Locations" / loc
    marker = dev / MARKER
    rep = out / "report.json"
    fresh = False
    if marker.exists() and rep.exists():
        try:
            fresh = (json.loads(marker.read_text(encoding="utf-8")).get("format", 0) >= RF2_FORMAT
                     and marker.stat().st_mtime >= rep.stat().st_mtime)
        except Exception:
            fresh = False
    if not fresh:
        log("rFactor 2: exporting the location first ...")
        r = export(slug, settings, install_to_rf2=True, log=log)
        if not r.get("installed"):
            raise RuntimeError("The export did not finish (" + "; ".join(r.get("missing") or ["see the log"]) + ").")
    else:
        log(f"rFactor 2: packaging ModDev\\Locations\\{loc}"
            + (" (with the collision cache from your Mod Mode session)" if any(dev.glob(f"{loc}/*.cbash")) else ""))

    if _find_ci(dev / "Assets" / "GMT", SKY_GMT) is not None:      # a copy made before the sky texture was added
        copy_sky(dev, rf2_root, log=log)
    pk = out / "rf2_package"
    pk.mkdir(exist_ok=True)
    state_p = pk / "state.json"
    state = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {}
    ver = next_version(rf2_root, loc, state.get("version"))
    vdir = pk / ver
    if vdir.exists():
        shutil.rmtree(vdir)
    vdir.mkdir()
    log(f"Packing .mas files (version {ver}) ...")
    mas = [make_mas(modmgr, n, fs, vdir, pk / "_work", log=log) for n, fs in mas_groups(dev, loc).items() if fs]
    shutil.rmtree(pk / "_work", ignore_errors=True)

    rfcmp = vdir / f"{loc}_v{ver.replace('.', '')}.rfcmp"
    cmpinfo = Path(appdata) / "cmpinfo.dat"
    old = cmpinfo.read_text(encoding="latin-1") if cmpinfo.exists() else ""
    if old and not (Path(appdata) / "cmpinfo_before_maptotrack.dat").exists():
        (Path(appdata) / "cmpinfo_before_maptotrack.dat").write_text(old, encoding="latin-1", newline="")
    cmpinfo.write_text(cmpinfo_text(old, {"name": loc, "version": ver, "rfcmp": _win(rfcmp),
                                          "mas": [_win(m) for m in mas], "author": settings.get("author", ""),
                                          "desc": "Made with maptotrack"}), encoding="latin-1", newline="")
    state.update(version=ver)
    state_p.write_text(json.dumps(state), encoding="utf-8")

    log(f"Opening MAS2 with \"{loc}\" {ver} ready to package. In MAS2: click the yellow box button (Mod Packager) "
        "-> Create Single Cmp Package -> Package, then close MAS2. The app installs it when the package appears.")
    t0 = time.time()
    proc = subprocess.Popen(_exe([mas2]), cwd=str(rf2_root))
    last = -1
    while True:
        if rfcmp.exists():
            sz = rfcmp.stat().st_size
            if sz == last and sz > sum(m.stat().st_size for m in mas):
                break
            last = sz
        elif proc.poll() is not None:
            raise RuntimeError(f"MAS2 was closed before the package was made. Click Package again (version {ver}).")
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"No package after {wait_s / 60:.0f} minutes. In MAS2: Mod Packager -> Create Single "
                               f"Cmp Package -> Package (it should write {rfcmp.name}).")
        time.sleep(2)
    log(f"  {rfcmp.name}: {rfcmp.stat().st_size / 1e6:.1f} MB")

    dest = rf2_root / "Installed" / "Locations" / loc / ver
    for v, d in installed_versions(rf2_root, loc):
        if d != dest:                              # one version in the track list (ModMgr can't uninstall these)
            shutil.rmtree(d, ignore_errors=True)
            log(f"  removed the older installed version {d.name}")
    if not any(dest.glob("*.mft")):               # not already installed with MAS2's own Install button
        p = _run([modmgr, "-q", f"-i{rfcmp.name}", f"-p{_win(rfcmp.parent)}", f"-d{_win(rf2_root)}"], timeout=1800,
                 cwd=str(rf2_root))                 # ModMgr only installs when started in the rF2 folder
        if not any(dest.glob("*.mft")):
            raise RuntimeError(f"ModMgr did not install {rfcmp.name} ({(p.stdout + p.stderr).strip()[-300:]})")
    log(f"Installed: {dest} - start rFactor 2 normally and pick \"{loc}\".")
    return {"loc": loc, "version": ver, "rfcmp": str(rfcmp), "installed": str(dest), "mas": [m.name for m in mas]}

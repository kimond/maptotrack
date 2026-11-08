"""Paths, endpoints and persisted user settings."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"
BLENDER_TEMPLATES = ROOT / "blender"
SETTINGS_FILE = ROOT / "settings.json"

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

USER_AGENT = "maptotrack/0.1 (Assetto Corsa track generator; personal use)"

# Data providers. All can be overridden with environment variables, e.g. to
# point at a self-hosted OSRM or Overpass instance.
OSRM_CAR_URL = os.environ.get("MAPTOTRACK_OSRM_URL", "https://router.project-osrm.org/route/v1/driving")
# FOSSGIS foot profile ignores one-way restrictions - useful for closed street circuits.
OSRM_NO_ONEWAY_URL = os.environ.get(
    "MAPTOTRACK_OSRM_NOONEWAY_URL", "https://routing.openstreetmap.de/routed-foot/route/v1/driving"
)
OVERPASS_URLS = [
    u
    for u in os.environ.get(
        "MAPTOTRACK_OVERPASS_URLS",
        "https://overpass-api.de/api/interpreter,https://overpass.private.coffee/api/interpreter,"
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ).split(",")
    if u
]
TERRARIUM_URL = os.environ.get(
    "MAPTOTRACK_TERRARIUM_URL", "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)
NOMINATIM_URL = os.environ.get("MAPTOTRACK_NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")

# Offline/demo mode: synthetic routing, buildings and elevation (no network).
OFFLINE = os.environ.get("MAPTOTRACK_OFFLINE", "0") == "1"

PORT = int(os.environ.get("MAPTOTRACK_PORT", "8765"))


def _guess_blender() -> str:
    patterns = [
        r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
        r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe",
        "/usr/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[-1]
    return ""


def _guess_ac_tracks() -> str:
    for p in [
        r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\content\tracks",
        r"D:\SteamLibrary\steamapps\common\assettocorsa\content\tracks",
    ]:
        if os.path.isdir(p):
            return p
    return r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\content\tracks"


def _guess_rf2() -> str:
    """rFactor 2's folder in a Steam library (the default library, or another drive's SteamLibrary)."""
    cands = [r"C:\Program Files (x86)\Steam\steamapps\common\rFactor 2"]
    cands += [rf"{d}:\SteamLibrary\steamapps\common\rFactor 2" for d in "CDEFGH"]
    cands += [rf"{d}:\Steam\steamapps\common\rFactor 2" for d in "DEFGH"]
    for p in cands:
        if os.path.isdir(p):
            return p
    return ""


RF2_BLENDER_MIN = (5, 0)      # Studio 397's GMT exporter R25.11 is built for Blender 5.0 (Python 3.11)
_BV_CACHE = {}
_PY_CACHE = {}


def blender_version(path: str):
    """(major, minor) of a Blender executable (from `--version`, cached), None if it can't be run."""
    import re
    import subprocess

    if not path or not os.path.exists(path):
        return None
    if path in _BV_CACHE:
        return _BV_CACHE[path]
    ver = None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout
        mm = re.search(r"Blender (\d+)\.(\d+)", out)
        if mm:
            ver = (int(mm.group(1)), int(mm.group(2)))
    except Exception:
        m = re.search(r"Blender[ _-]?(\d+)\.(\d+)", path, re.I)
        ver = (int(m.group(1)), int(m.group(2))) if m else None
    _BV_CACHE[path] = ver
    return ver


def blender_python(path: str):
    """(major, minor) of the Python inside a Blender (Blender 4.x / 5.0: 3.11, 5.1 and later: 3.13), None if
    unknown. Asks Blender itself; falls back on the known versions."""
    import re
    import subprocess

    if not path or not os.path.exists(path):
        return None
    if path in _PY_CACHE:
        return _PY_CACHE[path]
    py = None
    try:
        out = subprocess.run([path, "--background", "--factory-startup", "--python-expr",
                              "import sys; print('MAPTOTRACK_PY', sys.version_info[0], sys.version_info[1])"],
                             capture_output=True, text=True, timeout=180).stdout
        m = re.search(r"MAPTOTRACK_PY (\d+) (\d+)", out)
        py = (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:
        py = None
    if py is None:
        v = blender_version(path)
        if v:
            py = (3, 10) if v < (4, 0) else (3, 11) if v < (5, 1) else (3, 13)
    _PY_CACHE[path] = py
    return py


def addon_python(zip_path: str):
    """The Python version a compiled add-on zip was built for (from the magic number of its .pyc files), None if it
    has no .pyc (plain .py add-ons run in any Blender)."""
    import zipfile

    try:
        with zipfile.ZipFile(zip_path) as z:
            pyc = next((n for n in z.namelist() if n.endswith(".pyc")), None)
            if not pyc:
                return None
            magic = int.from_bytes(z.read(pyc)[:2], "little")
    except Exception:
        return None
    # CPython magic numbers: 3.10 34xx (3430-3439), 3.11 3450-3499, 3.12 3500-3549, 3.13 3550-3599, 3.14 3600-3649
    for lo, hi, ver in ((3430, 3449, (3, 10)), (3450, 3499, (3, 11)), (3500, 3549, (3, 12)), (3550, 3599, (3, 13)),
                        (3600, 3649, (3, 14))):
        if lo <= magic <= hi:
            return ver
    return None


def _blender_candidates(settings: dict) -> list:
    """Blenders to try for the rF2 exporter: the one set for it, the main one, then installed or unzipped Blender 5.0
    copies (Program Files, or a portable blender-5.0.x-windows-x64 folder inside the maptotrack folder)."""
    c = [settings.get("rf2_blender_path") or "", settings.get("blender_path") or ""]
    for pat in (str(ROOT / "blender-*" / "blender.exe"), str(ROOT / "blender-*" / "blender"),
                str(ROOT / "*" / "blender-*" / "blender.exe"),
                r"C:\Program Files\Blender Foundation\Blender*\blender.exe"):
        c += sorted(glob.glob(pat), reverse=True)
    seen, out = set(), []
    for p in c:
        if p and p not in seen and os.path.exists(p):
            seen.add(p)
            out.append(p)
    return out


def rf2_blender(settings: dict, addon: str | None = None) -> str:
    """The Blender that runs the rF2 GMT exporter: Blender 5.0+ whose Python matches the add-on's compiled files
    (the R25.11 zip only loads in Blender 5.0.x: Python 3.11)."""
    addon = find_rf2_addon() if addon is None else addon
    need = addon_python(addon) if addon else None
    for p in _blender_candidates(settings):
        v = blender_version(p)
        if v is None or v < RF2_BLENDER_MIN:
            continue
        if need is None or blender_python(p) == need:
            return p
    return ""


def rf2_blender_needed(addon: str) -> str:
    """Plain words for the Blender the add-on needs."""
    need = addon_python(addon) if addon else None
    if need == (3, 11):
        return "Blender 5.0.x (the exporter is built for Python 3.11; Blender 5.1 and later use Python 3.13)"
    if need:
        return f"Blender 5 with Python {need[0]}.{need[1]} (the exporter's compiled files need it)"
    return "Blender 5.0 or newer"


def find_rf2_addon() -> str:
    """Studio 397's Blender GMT exporter zip (io_rfactor2_gmt*.zip / io_rf2_gmt_release*.zip) in the maptotrack
    folder (the newest)."""
    hits = []
    for pat in ("io_rfactor2_gmt*.zip", "io_rf2_gmt_release*.zip"):
        hits += glob.glob(str(ROOT / pat)) + glob.glob(str(ROOT / "*" / pat))
    hits.sort(key=lambda p: os.path.getmtime(p))
    return hits[-1] if hits else ""


def load_settings() -> dict:
    data = {}
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault("blender_path", _guess_blender())
    data.setdefault("ac_tracks_path", _guess_ac_tracks())
    data.setdefault("author", "")
    data.setdefault("rf2_path", _guess_rf2())
    data.setdefault("rf2_blender_path", "")
    return data


def save_settings(data: dict) -> dict:
    cur = load_settings()
    cur.update({k: v for k, v in data.items() if k in ("blender_path", "ac_tracks_path", "author", "rf2_path",
                                                             "rf2_blender_path")})
    SETTINGS_FILE.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    return cur

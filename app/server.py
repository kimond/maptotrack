"""maptotrack local web server.  Run:  python -m app.server"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import traceback
import uuid
import webbrowser
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, imagery, pipeline, routing
from .geo import slugify

app = FastAPI(title="maptotrack")
JOBS: dict[str, dict] = {}


# ------------------------------------------------------------------ jobs
def run_job(fn, *args) -> str:
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "status": "running", "log": [], "result": None, "error": None}
    JOBS[jid] = job

    def log(msg):
        job["log"].append(str(msg))
        print(msg, flush=True)

    def target():
        try:
            job["result"] = fn(*args, log=log)
            job["status"] = "done"
        except Exception as exc:
            traceback.print_exc()
            job["error"] = str(exc)
            job["status"] = "error"
            log(f"ERROR: {exc}")

    threading.Thread(target=target, daemon=True).start()
    return jid


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[jid]


# ------------------------------------------------------------------ config
@app.get("/api/config")
def get_config():
    import os

    addon = config.find_rf2_addon()
    return {"offline": config.OFFLINE, "settings": config.load_settings(), "defaults": pipeline.DEFAULTS,
            "imagery": imagery.list_sources(), "rf2_addon": os.path.basename(addon) if addon else "",
            "rf2_need": config.rf2_blender_needed(addon) if addon else ""}


@app.post("/api/settings")
def set_settings(body: dict):
    return config.save_settings(body)


# ------------------------------------------------------------------ routing & scenery
class RouteReq(BaseModel):
    waypoints: list[list[float]]
    modes: list[str] = []
    loop: bool = True
    ignore_oneway: bool = False


@app.post("/api/route")
def api_route(req: RouteReq):
    return routing.route(req.waypoints, req.modes, req.loop, req.ignore_oneway)


@app.post("/api/scenery")
def api_scenery(body: dict):
    if body.get("mode") == "area":
        if not body.get("area"):
            raise HTTPException(400, "Draw the area on the map first.")
    elif len(body.get("coords", [])) < 2:
        raise HTTPException(400, "Build a route first.")
    return {"job": run_job(pipeline.scenery_preview, body)}


@app.post("/api/road_preview")
def api_road_preview(body: dict):
    return {"job": run_job(pipeline.road_preview, body)}


@app.post("/api/generate")
def api_generate(body: dict):
    if body.get("mode") == "area":
        if not body.get("area"):
            raise HTTPException(400, "Draw the area on the map first.")
    elif len(body.get("coords", [])) < 2:
        raise HTTPException(400, "Build a route first.")
    return {"job": run_job(pipeline.generate, body)}


@app.get("/api/tiles/{src_id}/{z}/{x}/{y}.jpg")
def map_tile(src_id: str, z: int, x: int, y: int, size: int = 256):
    """Aerial photo tiles for the map view (the WMS is fetched once, then served from the cache)."""
    from fastapi import Response

    if config.OFFLINE or not (0 <= z <= 22) or size not in (256, 512):
        raise HTTPException(404, "no tile")
    try:
        data = imagery.tile(src_id, z, x, y, size)
    except Exception as exc:
        raise HTTPException(404, f"tile unavailable: {exc}")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800"})


@app.get("/api/geocode")
def geocode(q: str):
    if config.OFFLINE:
        return []
    r = requests.get(config.NOMINATIM_URL, params={"q": q, "format": "json", "limit": 6},
                     headers={"User-Agent": config.USER_AGENT}, timeout=15)
    return [{"name": x["display_name"], "lat": float(x["lat"]), "lon": float(x["lon"])} for x in r.json()]


# ------------------------------------------------------------------ projects
@app.get("/api/projects")
def list_projects():
    out = []
    for d in sorted(config.OUTPUT_DIR.iterdir(), key=lambda p: -p.stat().st_mtime):
        if (d / "project.json").exists():
            out.append({"slug": d.name, "has_report": (d / "report.json").exists(),
                        "has_kn5": (d / "ac_track" / d.name / f"{d.name}.kn5").exists(),
                        "has_fbx": (d / "blender" / "track.fbx").exists()})
    return out


@app.get("/api/projects/{slug}")
def get_project(slug: str):
    d = config.OUTPUT_DIR / slugify(slug)
    if not (d / "project.json").exists():
        raise HTTPException(404, "not found")
    rep = d / "report.json"
    return {"project": json.loads((d / "project.json").read_text(encoding="utf-8")),
            "report": json.loads(rep.read_text(encoding="utf-8")) if rep.exists() else None}


@app.post("/api/projects/{slug}/save")
def save_project(slug: str, body: dict):
    d = config.OUTPUT_DIR / slugify(slug)
    d.mkdir(parents=True, exist_ok=True)
    (d / "project.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    return {"ok": True, "slug": d.name}


def _blender_job(slug, blender, log=print):
    bl = config.OUTPUT_DIR / slug / "blender"
    if not blender or not os.path.exists(blender):
        raise RuntimeError("Blender not found - set the path to blender.exe in Settings.")
    log(f"Running Blender: {blender}")
    proc = subprocess.run([blender, "--background", "--factory-startup", "--python", str(bl / "build_scene.py")],
                          cwd=str(bl), capture_output=True, text=True, timeout=1800)
    for line in (proc.stdout + proc.stderr).splitlines():
        if "maptotrack" in line or "Error" in line or "WARNING" in line:
            log(line)
    if proc.returncode != 0 or not (bl / "track.fbx").exists():
        raise RuntimeError("Blender failed - see log.")
    return {"blend": str(bl / "track.blend"), "fbx": str(bl / "track.fbx")}


@app.post("/api/projects/{slug}/blender")
def build_blender(slug: str, body: dict | None = None):
    s = config.load_settings()
    blender = (body or {}).get("blender_path") or s.get("blender_path")
    return {"job": run_job(_blender_job, slugify(slug), blender)}


def _kn5_job(slug, blender, source, path, log=print):
    """FBX / Blender scene -> <slug>.kn5 by running blender/export_kn5.py in background Blender."""
    bl = config.OUTPUT_DIR / slug / "blender"
    out = config.OUTPUT_DIR / slug / "ac_track" / slug / f"{slug}.kn5"
    if not blender or not os.path.exists(blender):
        raise RuntimeError("Blender not found - set the path to blender.exe in Settings.")
    if source == "blend":
        src = bl / "track.blend"
        if not src.exists():
            raise RuntimeError("No track.blend yet - click 'Build Blender scene' first (and save your edits in Blender).")
    elif source == "fbx":
        src = Path(path) if path else bl / "track.fbx"
        if not src.exists():
            raise RuntimeError(f"FBX not found: {src}")
    else:
        raise RuntimeError(f"unknown source {source}")
    bl.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    # always use the current exporter (older projects may have older copies)
    shutil.copy(config.BLENDER_TEMPLATES / "export_kn5.py", bl / "export_kn5.py")
    shutil.copy(config.ROOT / "app" / "kn5.py", bl / "kn5.py")
    cmd = [blender, "--background", "--factory-startup", "--python", str(bl / "export_kn5.py"), "--",
           "--" + source, str(src), "--out", str(out)]
    ov = config.ROOT / "kn5_materials.json"
    if ov.exists():
        cmd += ["--overrides", str(ov)]
    log(f"Converting {src.name} -> {out.name} with Blender ...")
    tmp = out.with_suffix(".kn5.tmp")
    cmd[cmd.index(str(out))] = str(tmp)
    proc = subprocess.run(cmd, cwd=str(bl), capture_output=True, text=True, timeout=1800)
    for line in (proc.stdout + proc.stderr).splitlines():
        if "maptotrack" in line or "Error" in line or "Traceback" in line:
            log(line)
    if proc.returncode != 0 or not tmp.exists():
        raise RuntimeError("Blender failed - see log.")
    from . import kn5 as kn5_mod

    try:
        info = kn5_mod.read(str(tmp))      # refuse to install a broken file
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"The exported KN5 is not valid ({exc}) - the previous one was kept.")
    os.replace(tmp, out)
    n_mesh = sum(1 for n in info["nodes"] if n["class"] == 2)
    starts = [n["name"] for n in info["nodes"] if n["name"].startswith("AC_START_")]
    log(f"{out.name}: {n_mesh} meshes, {len(info['materials'])} materials, {len(info['textures'])} textures, "
        f"{len(starts)} start positions")
    return {"kn5": str(out), "meshes": n_mesh, "starts": len(starts)}


@app.post("/api/projects/{slug}/kn5")
def build_kn5(slug: str, body: dict | None = None):
    body = body or {}
    s = config.load_settings()
    blender = body.get("blender_path") or s.get("blender_path")
    return {"job": run_job(_kn5_job, slugify(slug), blender, body.get("source", "blend"), body.get("path") or None)}


def _rf2_job(slug, log=print):
    from . import rf2

    return rf2.export(slug, config.load_settings(), log=log)


@app.post("/api/projects/{slug}/rf2")
def export_rf2(slug: str, body: dict | None = None):
    """The generated track as an rFactor 2 location, installed into rF2's ModDev (Mod Mode)."""
    return {"job": run_job(_rf2_job, slugify(slug))}


def _rf2_package_job(slug, log=print):
    from . import rf2

    return rf2.package(slug, config.load_settings(), log=log)


@app.post("/api/projects/{slug}/rf2_package")
def package_rf2(slug: str, body: dict | None = None):
    """The rF2 location packed (.mas), packaged in MAS2 (.rfcmp) and installed: drivable outside Mod Mode."""
    return {"job": run_job(_rf2_package_job, slugify(slug))}


@app.post("/api/projects/{slug}/install")
def install(slug: str, body: dict | None = None):
    slug = slugify(slug)
    src = config.OUTPUT_DIR / slug / "ac_track" / slug
    dest_root = (body or {}).get("ac_tracks_path") or config.load_settings()["ac_tracks_path"]
    if not src.exists():
        raise HTTPException(404, "Generate the track first.")
    if not os.path.isdir(dest_root):
        raise HTTPException(400, f"Assetto Corsa tracks folder not found: {dest_root}")
    dest = os.path.join(dest_root, slug)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    has_kn5 = (src / f"{slug}.kn5").exists()
    return {"dest": dest, "has_kn5": has_kn5,
            "note": None if has_kn5 else f"No {slug}.kn5 yet - generate the track again or use 'Build KN5'."}


@app.post("/api/projects/{slug}/open")
def open_folder(slug: str, body: dict | None = None):
    d = config.OUTPUT_DIR / slugify(slug) / ((body or {}).get("sub") or "")
    if not d.exists():
        raise HTTPException(404, "not found")
    if sys.platform.startswith("win"):
        os.startfile(str(d))  # noqa
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(d)])
    else:
        subprocess.Popen(["xdg-open", str(d)])
    return {"path": str(d)}


# ------------------------------------------------------------------ static
app.mount("/output", StaticFiles(directory=str(config.OUTPUT_DIR)), name="output")
app.mount("/web", StaticFiles(directory=str(config.WEB_DIR)), name="web")


@app.get("/")
def index():
    return FileResponse(config.WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


def main():
    url = f"http://127.0.0.1:{config.PORT}"
    print(f"maptotrack running at {url}  (Ctrl+C to stop)")
    if os.environ.get("MAPTOTRACK_NO_BROWSER") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()

"""Write the generated track: glTF for Blender/preview, AC spawn/timing objects, AC track folder files."""
from __future__ import annotations

import json
import shutil

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from . import config
from .meshes import MeshPart
from .track import Centerline

GRID_SPACING = 7.0


# ---------------------------------------------------------------- glTF
def to_gltf_coords(v):
    """track frame (x east, y north, z up) -> glTF (x, y up, z south)."""
    return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])


MASK_MATERIALS = {"trees", "veg_grass", "veg_grass_short", "props"}


def write_glb(parts: list[MeshPart], textures: dict, path, origin_shift):
    scene = trimesh.Scene()
    mats = {}
    for name, img in textures.items():
        kw = {}
        if name in MASK_MATERIALS:
            kw = {"alphaMode": "MASK", "alphaCutoff": 0.5}
        mats[name] = trimesh.visual.material.PBRMaterial(
            name=name, baseColorTexture=img, metallicFactor=0.0, roughnessFactor=0.95, **kw)
    for p in parts:
        v = to_gltf_coords(p.vertices - origin_shift)
        # Our UVs use the glTF/image convention (v = 0 at the top row of the image). trimesh stores
        # OpenGL-style UVs (v = 0 at the bottom) and flips v when writing glTF, so pre-flip here.
        uv = np.column_stack([p.uv[:, 0], 1.0 - p.uv[:, 1]])
        mesh = trimesh.Trimesh(vertices=v, faces=p.faces, process=False,
                               visual=trimesh.visual.TextureVisuals(uv=uv, material=mats[p.material]))
        nrm = getattr(p, "normals", None)
        if nrm is not None:
            mesh.vertex_normals = to_gltf_coords(nrm)
        else:
            _ = mesh.vertex_normals  # cache -> exported
        scene.add_geometry(mesh, node_name=p.name, geom_name=p.name)
    scene.export(str(path), file_type="glb")


# ---------------------------------------------------------------- KN5 (Assetto Corsa model)
def _png_bytes(img):
    import io

    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def dds_bc1(img):
    """Opaque texture -> DDS bytes, BC1 (DXT1) compressed with its mip chain.

    AC unpacks PNG/JPG to raw RGBA in video memory (a 2048 px ground tile = 16 MB, 22 MB with mipmaps);
    BC1 keeps it compressed: 2.7 MB with mipmaps, and the .kn5 is ~3x smaller. Sides are rounded to a
    power of two (BC1 works on 4x4 blocks, mipmaps halve; UVs are 0-1 over the image, so it doesn't show)."""
    import io
    import struct

    im = img.convert("RGB")
    w, h = im.size
    # power-of-two sides: BC1 needs multiples of 4, and the full mip chain (down to 4 px) needs halvable sizes
    w4 = int(min(4096, max(4, 2 ** round(np.log2(w)))))
    h4 = int(min(4096, max(4, 2 ** round(np.log2(h)))))
    if (w4, h4) != (w, h):
        im = im.resize((w4, h4), Image.LANCZOS)
        w, h = w4, h4
    levels = []
    while True:
        buf = io.BytesIO()
        im.save(buf, "DDS", pixel_format="DXT1")
        levels.append(buf.getvalue()[128:])                 # Pillow writes a 128-byte DDS header first
        if w <= 4 or h <= 4:
            break
        w, h = w // 2, h // 2
        im = im.resize((w, h), Image.BOX)
    W, H = w4, h4
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000   # caps, height, width, pixelformat, mipcount, linearsize
    header = struct.pack("<4sIIIIIII44x", b"DDS ", 124, flags, H, W, len(levels[0]), 0, len(levels))
    header += struct.pack("<II4sIIIII", 32, 0x4, b"DXT1", 0, 0, 0, 0, 0)          # pixel format: FourCC DXT1
    header += struct.pack("<IIIII", 0x1000 | 0x8 | 0x400000, 0, 0, 0, 0)         # texture, complex, mipmap
    return header + b"".join(levels)


def write_kn5(parts: list[MeshPart], textures: dict, objects: list, path, origin_shift=np.zeros(3), log=print,
              png_bytes=None, dds_bytes=None, multilayer=None):
    """The generated track straight to <slug>.kn5 (no Blender / ksEditor needed).
    multilayer: {material: {"textures": {slot: texture key}, "props": {...}}} -> ksMultilayer materials."""
    from . import kn5

    kb = kn5.Kn5Builder(overrides=load_kn5_overrides())
    used = sorted({p.material for p in parts})          # sorted: identical files run after run
    added = {}

    def tex_of(key):
        if key in added:
            return added[key]
        img = textures.get(key)
        if img is None:
            return None
        dds = (dds_bytes or {}).get(key)
        if dds is not None:
            added[key] = kb.add_texture(f"{key}.dds", dds)
        else:
            data = (png_bytes or {}).get(key) or _png_bytes(img)
            added[key] = kb.add_texture(f"{key}.png", data)
        return added[key]

    for name in used:
        tex = tex_of(name)
        ml = (multilayer or {}).get(name)
        if ml and tex is not None and all(tex_of(k) for k in ml["textures"].values()):
            kb.material(name, texture=tex, shader="ksMultilayer",
                        extra_textures={slot: tex_of(k) for slot, k in ml["textures"].items()},
                        extra_props=ml["props"])
        else:
            kb.material(name, texture=tex)
    for p in parts:
        v = to_gltf_coords(p.vertices - origin_shift)
        nrm = getattr(p, "normals", None)
        if nrm is not None:
            nrm = to_gltf_coords(nrm)
        kb.add_mesh(p.name, p.material, v, p.faces, uv=p.uv, normals=nrm, lod_out=getattr(p, "lod_out", None))
    for o in objects:
        pos = to_gltf_coords(np.asarray([o["pos"]], float))[0]
        fwd = to_gltf_coords(np.asarray([o["forward"]], float))[0]
        kb.add_dummy(o["name"], kn5.dummy_matrix(pos, fwd))
    kb.write(path)
    return {"meshes": len(kb.meshes), "materials": len(kb.materials), "textures": len(kb.textures)}


def load_kn5_overrides():
    """Optional user shader settings: <app>/kn5_materials.json  {"regex": {"shader": ..., "props": {...}}}."""
    f = config.ROOT / "kn5_materials.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# No WALL surface on purpose: AC treats 1WALL... meshes as walls only while WALL is NOT a surface key;
# defined as a surface, every wall becomes drivable ground and cars pass through barriers and buildings
# (Kunos' tracks don't define it either; found by Map2Lap).
SURFACES_INI = """[SURFACE_0]
KEY=ROAD
FRICTION=0.99
DAMPING=0
WAV=
WAV_PITCH=0
FF_EFFECT=NULL
DIRT_ADDITIVE=0
IS_VALID_TRACK=1
BLACK_FLAG_TIME=0
SIN_HEIGHT=0
SIN_LENGTH=0
IS_PITLANE=0
VIBRATION_GAIN=0
VIBRATION_LENGTH=0

[SURFACE_1]
KEY=GRASS
FRICTION=0.60
DAMPING=0.1
WAV=grass.wav
WAV_PITCH=1.3
FF_EFFECT=NULL
DIRT_ADDITIVE=1
IS_VALID_TRACK=0
BLACK_FLAG_TIME=0
SIN_HEIGHT=0.03
SIN_LENGTH=0.5
IS_PITLANE=0
VIBRATION_GAIN=0.2
VIBRATION_LENGTH=0.6
"""
_OLD_SURFACES_WITH_WALL = """[SURFACE_0]
KEY=ROAD
FRICTION=0.99
DAMPING=0
WAV=
WAV_PITCH=0
FF_EFFECT=NULL
DIRT_ADDITIVE=0
IS_VALID_TRACK=1
BLACK_FLAG_TIME=0
SIN_HEIGHT=0
SIN_LENGTH=0
IS_PITLANE=0
VIBRATION_GAIN=0
VIBRATION_LENGTH=0

[SURFACE_1]
KEY=GRASS
FRICTION=0.60
DAMPING=0.1
WAV=grass.wav
WAV_PITCH=1.3
FF_EFFECT=NULL
DIRT_ADDITIVE=1
IS_VALID_TRACK=0
BLACK_FLAG_TIME=0
SIN_HEIGHT=0.03
SIN_LENGTH=0.5
IS_PITLANE=0
VIBRATION_GAIN=0.2
VIBRATION_LENGTH=0.6

[SURFACE_2]
KEY=WALL
FRICTION=0.3
DAMPING=0
WAV=
WAV_PITCH=0
FF_EFFECT=NULL
DIRT_ADDITIVE=0
IS_VALID_TRACK=0
BLACK_FLAG_TIME=0
SIN_HEIGHT=0
SIN_LENGTH=0
IS_PITLANE=0
VIBRATION_GAIN=0
VIBRATION_LENGTH=0
"""


# ---------------------------------------------------------------- AC objects
def _obj(name, p, z, t, lateral, n, dz):
    pos = np.array([p[0] + n[0] * lateral, p[1] + n[1] * lateral, z + dz])
    return {"name": name, "pos": pos.round(3).tolist(), "forward": [round(float(t[0]), 5), round(float(t[1]), 5), 0.0]}


def ac_objects(cl: Centerline, width: float, grid: int, pits: int, widths=None, camber: float = 0.0):
    """Spawn and timing points ON the built road.

    widths = (hl, hr) per centreline point: the asphalt really spans [-hr, +hl] around the centreline
    (the photo width can be off-centre), so cars are placed in the middle of the real asphalt, at its
    surface height (crown included), and timing gates just outside its edges."""
    objs = []
    if widths is None:
        hl = hr = np.full(len(cl.p), width / 2)
    else:
        hl, hr = (np.asarray(w, float) for w in widths)
    spots = max(grid, pits)

    def road_at(dist):
        p, z, t, n = cl.frame_at(dist)
        i = int(round((dist % cl.length if cl.loop else np.clip(dist, 0, cl.length)) / cl.spacing)) % len(cl.p)
        return p, z, t, n, float(hl[i]), float(hr[i])

    def surface(z, lateral, l, r):
        tc = (l - lateral) / max(l + r, 1e-6)          # 0 = left edge, 1 = right edge (as the road cross-section)
        return z + camber * (1 - (2 * tc - 1) ** 2)

    def place(name, dist, side, dz):
        """side: -1 right lane, 0 middle, +1 left lane; dz: height above the asphalt."""
        p, z, t, n, l, r = road_at(dist)
        mid = (l - r) / 2
        lane = min((l + r) / 4, max((l + r) / 2 - 1.0, 0.0))
        lat = mid + side * lane
        return _obj(name, p, surface(z, lat, l, r), t, lat, n, dz)

    def gate(name, dist, left):
        p, z, t, n, l, r = road_at(dist)
        lat = (l + 1.0) if left else -(r + 1.0)
        return _obj(name, p, z, t, lat, n, 0.5)

    if cl.loop:
        start = 0.0
        objs += [gate("AC_TIME_0_L", start, True), gate("AC_TIME_0_R", start, False)]
        for k, frac in ((1, 1 / 3), (2, 2 / 3)):
            d = cl.length * frac
            objs += [gate(f"AC_TIME_{k}_L", d, True), gate(f"AC_TIME_{k}_R", d, False)]
        hot = -min(150.0, cl.length * 0.1)
    else:
        start = min(12.0 + GRID_SPACING * spots, cl.length * 0.3)
        objs += [gate("AC_AB_START_L", start, True), gate("AC_AB_START_R", start, False)]
        objs += [gate("AC_AB_FINISH_L", cl.length - 15.0, True), gate("AC_AB_FINISH_R", cl.length - 15.0, False)]
        mid_d = (start + cl.length) / 2
        objs += [gate("AC_TIME_0_L", mid_d, True), gate("AC_TIME_0_R", mid_d, False)]
        hot = -6.0
    objs.append(place("AC_HOTLAP_START_0", start + hot, 0, 0.3))
    for i in range(spots):
        d = start - 6.0 - i * GRID_SPACING
        side = 1 if i % 2 == 0 else -1
        if i < grid:
            objs.append(place(f"AC_START_{i}", d, side, 0.3))
        if i < pits:
            objs.append(place(f"AC_PIT_{i}", d, side, 0.3))
    return objs


def spawn_objects(cl: Centerline, hl, hr, s0: float, grid: int, pits: int, camber: float = 0.0):
    """Area mode (free roam): the cars start on one street, no timing gates. The pit and grid spots line up
    behind s0 (the hot-lap start), two by two on a street wide enough, else in single file.
    -> (objects, number of spots that fit)."""
    hl, hr = np.asarray(hl, float), np.asarray(hr, float)
    spots = max(grid, pits, 1)
    wide = float(np.median(hl + hr)) >= 5.5
    gap = GRID_SPACING if wide else GRID_SPACING + 1.0
    span = 6.0 + (spots - 1) * gap
    if cl.length < span + 10.0:
        spots = max(1, int((cl.length - 16.0) / gap) + 1)
        span = 6.0 + (spots - 1) * gap
    s0 = float(np.clip(s0, min(span + 4.0, cl.length - 4.0), cl.length - 4.0))

    def place(name, dist, side, dz):
        p, z, t, n = cl.frame_at(dist)
        i = int(np.clip(round(dist / cl.spacing), 0, len(cl.p) - 1))
        l, r = float(hl[i]), float(hr[i])
        mid = (l - r) / 2
        lane = min((l + r) / 4, max((l + r) / 2 - 1.0, 0.0)) if wide else 0.0
        lat = mid + side * lane
        tc = (l - lat) / max(l + r, 1e-6)
        return _obj(name, p, z + camber * (1 - (2 * tc - 1) ** 2), t, lat, n, dz)

    objs = [place("AC_HOTLAP_START_0", s0, 0, 0.3)]
    for i in range(spots):
        d = s0 - 6.0 - i * gap
        side = (1 if i % 2 == 0 else -1) if wide else 0
        if i < max(grid, 1):
            objs.append(place(f"AC_START_{i}", d, side, 0.3))
        if i < max(pits, 1):
            objs.append(place(f"AC_PIT_{i}", d, side, 0.3))
    return objs, spots


# ---------------------------------------------------------------- 2D images
def _draw_path(draw, pts, width, fill, loop):
    pts = [tuple(map(float, p)) for p in pts]
    if loop:
        pts = pts + [pts[0]]
    draw.line(pts, fill=fill, width=max(int(width), 1), joint="curve")


def write_map(cl: Centerline, width, folder, lines=None):
    """map.png + data/map.ini. lines: [(xy, width)] of every street (area mode) instead of the track."""
    margin = 20
    if lines:
        allp = np.vstack([xy for xy, _ in lines])
        ax, az = allp[:, 0], -allp[:, 1]
        width = max(w for _, w in lines)
    else:
        ax, az = cl.p[:, 0], -cl.p[:, 1]           # AC world X / Z (Blender FBX export -Z forward, Y up)
    minx, maxx, minz, maxz = ax.min() - width, ax.max() + width, az.min() - width, az.max() + width
    W = int(np.ceil(maxx - minx)) + 2 * margin
    H = int(np.ceil(maxz - minz)) + 2 * margin
    xo, zo = -minx + margin, -minz + margin
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if lines:
        for xy, w in lines:
            _draw_path(d, np.column_stack([xy[:, 0] + xo, -xy[:, 1] + zo]), w + 4, (40, 40, 40, 200), False)
        for xy, w in lines:
            _draw_path(d, np.column_stack([xy[:, 0] + xo, -xy[:, 1] + zo]), w, (255, 255, 255, 230), False)
    else:
        pts = np.column_stack([ax + xo, az + zo])
        _draw_path(d, pts, width + 4, (40, 40, 40, 200), cl.loop)
        _draw_path(d, pts, width, (255, 255, 255, 230), cl.loop)
    img.save(folder / "map.png")
    (folder / "data").mkdir(exist_ok=True)
    (folder / "data" / "map.ini").write_text(
        "[PARAMETERS]\n"
        f"WIDTH={W}\nHEIGHT={H}\nMARGIN={margin}\nSCALE_FACTOR=1\n"
        f"X_OFFSET={xo:.3f}\nZ_OFFSET={zo:.3f}\nDRAWING_SIZE=10\n", encoding="utf-8")


def write_outline(cl: Centerline, path, size=512, lines=None):
    pad = 30
    paths = [(xy.copy(), False) for xy, _ in lines] if lines else [(cl.p.copy(), cl.loop)]
    for p, _ in paths:
        p[:, 1] *= -1
    allp = np.vstack([p for p, _ in paths])
    mn, mx = allp.min(0), allp.max(0)
    sc = (size - 2 * pad) / max((mx - mn).max(), 1)
    off = (size - (mx - mn) * sc) / 2
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for p, loop in paths:
        _draw_path(d, (p - mn) * sc + off, 9 if not lines else 4, (255, 255, 255, 255), loop)
    img.save(path)


def write_preview(cl: Centerline, width, scenery_xy, path, size=(1280, 720), lines=None, bounds=None, start_s=None):
    """Top-down schematic: terrain, streets, buildings, track, start line. Area mode: lines = every street
    (drawn as the track), cl = the street the cars start on."""
    Wd, Hd = size
    pts = cl.p
    if bounds is not None:
        mn, mx = np.array(bounds[:2], float), np.array(bounds[2:], float)
    else:
        mn, mx = pts.min(0) - 60, pts.max(0) + 60
    sc = min(Wd / (mx[0] - mn[0]), Hd / (mx[1] - mn[1]))
    cx, cy = (mn + mx) / 2

    def tr(xy):
        xy = np.asarray(xy)
        return np.column_stack([(xy[:, 0] - cx) * sc + Wd / 2, Hd / 2 - (xy[:, 1] - cy) * sc])

    img = Image.new("RGB", size, (74, 98, 58))
    d = ImageDraw.Draw(img)
    for road in scenery_xy.get("roads", []):
        if len(road["xy"]) > 1:
            d.line([tuple(p) for p in tr(road["xy"])], fill=(92, 92, 94), width=max(int(road["width"] * sc), 1))
    for poly in scenery_xy.get("buildings", []):
        if len(poly) > 2:
            d.polygon([tuple(p) for p in tr(poly)], fill=(190, 180, 165), outline=(120, 110, 100))
    trees = scenery_xy.get("trees")
    if trees is not None and len(trees):
        tq = tr(np.asarray(trees)[:, :2])
        rr = np.maximum(np.asarray(trees)[:, 2] * sc / 2, 1.0)
        for (px, py), r in zip(tq, rr):
            d.ellipse([px - r, py - r, px + r, py + r], fill=(44, 72, 36))
    if lines:
        for xy, w in lines:
            _draw_path(d, tr(xy), (w + 2) * sc, (30, 30, 32), False)
        for xy, w in lines:
            _draw_path(d, tr(xy), w * sc, (70, 70, 74), False)
        p, _, t, n = cl.frame_at(cl.length / 2 if start_s is None else start_s)
    else:
        q = tr(pts)
        _draw_path(d, q, (width + 2) * sc, (30, 30, 32), cl.loop)
        _draw_path(d, q, width * sc, (70, 70, 74), cl.loop)
        _draw_path(d, q, max(sc * 0.3, 1), (230, 230, 230), cl.loop)
        p, _, t, n = cl.frame_at(0.0 if cl.loop else min(12 + GRID_SPACING * 2, cl.length * 0.3))
    a = tr([p + n * width / 2, p - n * width / 2])
    d.line([tuple(a[0]), tuple(a[1])], fill=(220, 30, 30), width=max(int(2 * sc), 3))
    img.save(path)
    return img


# ---------------------------------------------------------------- AC folder + Blender kit
def write_ac_folder(folder, slug, info: dict, cl: Centerline, width, preview_img, pitboxes, lines=None):
    """lines: area mode, every street [(xy, width)] (map, outline, 'free roam' tags)."""
    folder.mkdir(parents=True, exist_ok=True)
    ui = folder / "ui"
    ui.mkdir(exist_ok=True)
    ui_json = {
        "name": info.get("name", slug),
        "description": info.get("description", "Generated with maptotrack. Map data (c) OpenStreetMap contributors, "
                                "Overture Maps Foundation (ODbL)."
                                + "".join(f" {c}." for c in info.get("credits", []) if "OpenStreetMap" not in c)),
        "tags": ["street", "maptotrack", "free roam" if lines else ("circuit" if cl.loop else "point to point")],
        "geotags": [f"{info['lat']:.5f}", f"{info['lon']:.5f}"],
        "country": info.get("country", ""),
        "city": info.get("city", ""),
        "length": f"{info.get('length', cl.length):.0f}",
        "width": f"{width:.0f}",
        "pitboxes": str(pitboxes),
        "run": "clockwise" if info.get("clockwise") else "counter-clockwise",
        "author": info.get("author", ""),
        "version": "0.1",
        "url": "",
        "year": info.get("year", ""),
    }
    (ui / "ui_track.json").write_text(json.dumps(ui_json, indent=2, ensure_ascii=False), encoding="utf-8")
    write_outline(cl, ui / "outline.png", lines=lines)
    preview_img.resize((1024, 576)).save(ui / "preview.png")
    write_map(cl, width, folder, lines=lines)
    data = folder / "data"
    data.mkdir(exist_ok=True)
    surf = data / "surfaces.ini"
    # keep a file the user tuned, but replace our own older one that wrongly defined WALL as a surface
    if not surf.exists() or surf.read_text(encoding="utf-8") == _OLD_SURFACES_WITH_WALL:
        surf.write_text(SURFACES_INI, encoding="utf-8")
    (folder / "README_KN5.txt").write_text(
        f"{slug}.kn5 is the 3D model. maptotrack writes it when generating, and again with 'Build KN5'\n"
        "(from your Blender scene or an FBX file).\n"
        "Then copy this whole folder to  assettocorsa/content/tracks/  (or use 'Install' in maptotrack).\n"
        "AI line: ai/fast_lane.ai and ai/pit_lane.ai are generated along the road centre. For a better racing line,\n"
        "drive a lap with the AI line recording app (Content Manager) and replace ai/fast_lane.ai.\n",
        encoding="utf-8")


def write_blender_kit(folder, slug, ac_folder):
    folder.mkdir(parents=True, exist_ok=True)
    for f in ("build_scene.py", "export_fbx.py", "export_kn5.py"):
        shutil.copy(config.BLENDER_TEMPLATES / f, folder / f)
    shutil.copy(config.ROOT / "app" / "kn5.py", folder / "kn5.py")   # used by export_kn5.py inside Blender
    (folder / "track_meta.json").write_text(json.dumps({
        "slug": slug, "glb": "track.glb", "objects": "ac_objects.json",
        "fbx": "track.fbx", "blend": "track.blend", "ac_folder": str(ac_folder),
    }, indent=2), encoding="utf-8")


def clockwise(cl: Centerline) -> bool:
    x, y = cl.p[:, 0], cl.p[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return area < 0

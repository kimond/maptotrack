"""maptotrack -> rFactor 2 .gmt files, with Studio 397's Blender exporter (io_rfactor2_gmt, R25.11, Blender 5.0+).

    blender --background --factory-startup --python export_rf2_gmt.py -- --manifest rf2_meshes.json --addon io_rfactor2_gmt_50.zip

The manifest (written by app/rf2.py) lists one GMT per entry: an .npz with the vertices (maptotrack's local frame:
x east, y north, z up = Blender's axes; the exporter turns Blender (x, y, z) into rF2 (x, z, y)), the triangles and
the UVs, the rF2 material name (its surface code), the .dds texture and whether it is an alpha cut-out. Timing gates
are quads given by their corners.

How the exporter is driven (worked out from its code, it has no headless API): a GMT is an EMPTY whose child MESH
objects are written with mesh.data.materials[0]; rF2 properties live directly on Object / Mesh / Material and on the
Image Texture node named 'sTex0' (L2DIFFUSET0 shader, 'sTex0|DIFFUSE-MODULATE|True'). The texture name written is
image.name, so the image is named after the .dds. The mesh needs a 'UV_A' layer and a 'Diffuse Color' corner colour
layer. rF2_GMT_Export.save_gmt(directory, gmt, False) writes <directory>/<gmt name>.gmt.
"""
import importlib
import importlib.util
import json
import os
import sys
import time
import zipfile

import bpy
import numpy as np


def log(*a):
    print("maptotrack:", *a, flush=True)


argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
args = {argv[i].lstrip("-"): argv[i + 1] for i in range(0, len(argv) - 1, 2)}
MAN = json.load(open(args["manifest"], encoding="utf-8"))
ZIP = args.get("addon")
ADDON_DIR = args.get("addon-dir")             # the zip already unpacked by the app: loaded from there, nothing is
OUT = MAN["gmt_dir"]                           # installed into Blender (several exporters can run at once)
WORKER = args.get("worker", "")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- the add-on
if ADDON_DIR:
    MOD = next(d for d in os.listdir(ADDON_DIR) if os.path.isfile(os.path.join(ADDON_DIR, d, "__init__.py")))
    pyc = next((os.path.join(r, f) for r, _d, fs in os.walk(os.path.join(ADDON_DIR, MOD)) for f in fs
                if f.endswith(".pyc")), None)
    magic = open(pyc, "rb").read(4) if pyc else None
    SRC = MOD
else:
    with zipfile.ZipFile(ZIP) as z:
        MOD = next(n.split("/")[0] for n in z.namelist() if n.count("/") == 1 and n.endswith("/__init__.py"))
        pyc = next((n for n in z.namelist() if n.endswith(".pyc")), None)
        magic = z.read(pyc)[:4] if pyc else None
    SRC = os.path.basename(ZIP)
if magic and magic != importlib.util.MAGIC_NUMBER:     # compiled for another Python: it can't load here
    log(f"ERROR: {SRC} is compiled for another Python than this Blender's "
        f"({bpy.app.version_string}, Python {sys.version_info[0]}.{sys.version_info[1]}). "
        "Studio 397's R25.11 exporter needs Blender 5.0.x (Python 3.11).")
    sys.exit(2)
if ADDON_DIR:
    sys.path.insert(0, ADDON_DIR)
    importlib.import_module(MOD).register()
else:
    bpy.ops.preferences.addon_install(filepath=ZIP, overwrite=True)
    bpy.ops.preferences.addon_enable(module=MOD)
EXP = importlib.import_module(MOD + ".rF2_GMT_Export")
UTL = importlib.import_module(MOD + ".rF2_Utilities")
STF = importlib.import_module(MOD + ".rF2_GMT_Stuff")
UTL.MessageBox = lambda messages=(), title="Message Box", icon="INFO": print("[rF2]", title, *messages, sep="\n   ")
if not WORKER or WORKER == "1":
    log(f"rF2 exporter {MOD} enabled (Blender {bpy.app.version_string})")

# an empty scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
scene = bpy.context.scene
_images = {}


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:                     # e.g. an attribute renamed in this Blender
        print("[rF2] ignored:", type(e).__name__, e)


def image(dds):
    """A placeholder image named exactly like the texture file (the exporter writes image.name)."""
    im = _images.get(dds)
    if im is None:
        im = bpy.data.images.new(dds, 4, 4)
        if im.name != dds:                     # a clash would give 'name.dds.001'
            im.name = dds
        _images[dds] = im
    return im


def corner_colours(me):
    n = len(me.loops)
    try:
        vc = me.vertex_colors.get("Diffuse Color") or me.vertex_colors.new(name="Diffuse Color")
        vc.data.foreach_set("color", [1.0] * (4 * n))
    except AttributeError:                     # no legacy API: the same layer as a colour attribute
        ca = me.color_attributes.new("Diffuse Color", "BYTE_COLOR", "CORNER")
        ca.data.foreach_set("color", [1.0] * (4 * n))


def build_mesh(name, v, f, uv):
    me = bpy.data.meshes.new(name + "_mesh")
    me.from_pydata(v.tolist(), [], f.tolist())
    me.update()
    uvl = me.uv_layers.new(name="UV_A")
    loop_v = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", loop_v)
    uvl.data.foreach_set("uv", uv[loop_v].astype(np.float32).ravel())
    try:
        me.shade_smooth()
    except AttributeError:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    corner_colours(me)
    return me


IBL_SLOTS = [("albedoMap", None), ("detailMap", "_DEFTEX_DCA_BC7.DDS"), ("aoSpecularMap", "_DEFTEX_AOS_BC5.DDS"),
             ("metalRoughMap", "_DEFTEX_MR_BC5.DDS"), ("normalMap", "_DEFTEX_NM_BC5.DDS"),
             ("emissiveMap", "_DEFTEX_AOS_BC5.DDS"), ("altAoSpecularMap", "_DEFTEX_AOS_BC5.DDS")]
IBL_TEXTURES = ("albedoMap/detailMap/aoSpecularMap/metalRoughMap/normalMap/emissiveMap/altAoSpecularMap|"
                "DIFFUSE-NULL/BUMP-NULL/BUMP-NULL/BUMP-NULL/BUMP-NULL/BUMP-NULL/BUMP-NULL|"
                "False/False/False/False/False/False/False")


def build_material(name, dds, cutout, shading=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    if "Principled BSDF" not in nt.nodes:
        p = nt.nodes.new("ShaderNodeBsdfPrincipled")
        p.name = "Principled BSDF"
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.name = uvn.label = "UV_A"
    uvn.uv_map = "UV_A"
    pbr = bool(shading and shading.get("pbr"))
    # PBR (rF2's IBL standard shader, as the stock tracks): our colour map + rF2's neutral default maps; the
    # roughness etc. come from the material's .json (written by the app). Otherwise the old one-texture shader.
    slots = IBL_SLOTS if pbr else [("sTex0", None)]
    nodes = []
    for slot, default in slots:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.name = tex.label = slot
        tex.image = image(default or dds)
        nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
        nodes.append(tex)
    tex = nodes[0]
    nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    if pbr:
        mat.ShaderName = "L2IBLSTANDARD"
        mat.CommonName = "IBL Standard Shader"
        mat.ShaderTextures = IBL_TEXTURES
    else:
        mat.ShaderName = "L2DIFFUSET0"
        mat.CommonName = "Diffuse T1"
        mat.ShaderTextures = "sTex0|DIFFUSE-MODULATE|True"
    for n in nodes:
        _safe(UTL.texture_set_defaults, n)
    _safe(UTL.material_set_defaults, mat, mat.ShaderName)
    try:
        mat.Blend_List = "SIMPLE" if cutout else "REPLACE"     # SIMPLE: colour key + alpha reference (cut-out)
        if cutout:
            tex.Alpha_Reference = 128
    except Exception as e:
        print("[rF2] blend:", e)
    if shading:                                # how shiny (the exporter's default 0.75 / 5 glares on grass)
        sp = float(shading.get("spec", 0.1))
        _safe(setattr, mat, "Specular_Color", (sp, sp, sp, 1.0))
        _safe(setattr, mat, "Specular_Power", int(shading.get("power", 4)))
        _safe(setattr, mat, "FresSpecMin", 0.0)
        _safe(setattr, mat, "FresSpecMax", float(shading.get("fresnel", 0.25)))
        if shading.get("double"):                 # foliage cards seen from both sides
            _safe(setattr, mat, "DoubleSided", True)
    # never Zero/Zero (the property default): One / Zero = opaque
    if getattr(mat, "Blend_Source_List", "2") == "1" and getattr(mat, "Blend_Destination_List", "1") == "1":
        _safe(setattr, mat, "Blend_Source_List", "2")
    return mat


def export_one(gname, me, mat, lod_out):
    me.materials.clear()
    me.materials.append(mat)
    gmt = bpy.data.objects.new(gname, None)
    child = bpy.data.objects.new(gname + "_m", me)
    scene.collection.objects.link(gmt)
    scene.collection.objects.link(child)
    child.parent = gmt
    UTL.set_gmt_defaults(gmt)
    gmt.LOD_In = 0.0
    gmt.LOD_Out = float(lod_out) if lod_out else 20000.0
    vl = bpy.context.view_layer
    for o in vl.objects:
        o.select_set(False)
    gmt.select_set(True)
    vl.objects.active = gmt
    EXP.save_gmt(OUT, gmt, False)
    path = os.path.join(OUT, STF.GetName(gmt.name, 64) + ".gmt")
    ok = os.path.isfile(path)
    bpy.data.objects.remove(child)
    bpy.data.objects.remove(gmt)
    bpy.data.meshes.remove(me)
    bpy.data.materials.remove(mat)
    return ok


t0 = time.time()
done = failed = 0
for m in MAN["meshes"]:
    d = np.load(m["npz"])
    v, f, uv = d["v"].astype(np.float64), d["f"].astype(np.int64), d["uv"].astype(np.float64)
    if not len(f):
        continue
    me = build_mesh(m["gmt"], v, f, uv)
    mat = build_material(m["rf2_material"], m["dds"], bool(m.get("alpha")), m.get("shading"))
    if export_one(m["gmt"], me, mat, m.get("lod_out")):
        done += 1
        print(f"maptotrack-progress {m['gmt']}", flush=True)
    else:
        failed += 1
        log(f"ERROR: {m['gmt']} was not written")
for gname, corners in MAN.get("gates", {}).items():
    v = np.asarray(corners, float)
    v = np.vstack([v, v])                      # both sides, each with its own vertices
    me = build_mesh(gname, v, np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]), np.zeros((8, 2)))
    mat = build_material(gname + "_mat", "white.dds", False)
    if export_one(gname, me, mat, 0):
        done += 1
    else:
        failed += 1
        log(f"ERROR: timing gate {gname} was not written")
log(f"{'worker ' + WORKER + ': ' if WORKER else ''}{done} .gmt files written, {failed} failed, {time.time() - t0:.0f} s")
sys.exit(1 if failed or not done else 0)

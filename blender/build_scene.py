"""maptotrack -> Blender scene for Assetto Corsa.

Run either:
  * from the maptotrack app ("Build Blender scene" button), or
  * blender --background --python build_scene.py
  * or open Blender > Scripting tab > Open this file > Run Script

It will:
  1. import track.glb (road, terrain, walls, buildings, streets)
  2. point every material at the PNGs in ./texture (so ksEditor finds them)
  3. create the AC_ spawn / timing empties from ac_objects.json
  4. save track.blend and export track.fbx ready for ksEditor
Tested with Blender 4.x.
"""
import json
import os
import sys

import bpy
from mathutils import Matrix, Vector

# Set to True if cars spawn facing backwards in game.
FLIP_SPAWN_DIRECTION = False


def _here():
    for cand in (globals().get("__file__"), bpy.data.filepath):
        if cand and os.path.exists(cand):
            return os.path.dirname(os.path.abspath(cand))
    return os.getcwd()


HERE = _here()
META = json.load(open(os.path.join(HERE, "track_meta.json"), encoding="utf-8"))

GROUPS = [  # (collection, name prefix)
    ("AC_OBJECTS", "AC_"),
    ("PROPS", "1WALL_PROPS"),
    ("PROPS", "PROPS"),
    ("PROPS", "PYLONS"),
    ("PROPS", "1WALL_PYLON"),
    ("PROPS", "WIRES"),
    ("ROAD", "1ROAD_GND"),
    ("ROAD", "1ROAD_ASPH"),
    ("ROAD", "1GRASS_SHLD"),
    ("ROAD", "ROAD_LINES"),
    ("TERRAIN", "1GRASS_GND"),
    ("TERRAIN", "1GRASS_NEAR"),
    ("ROAD", "1ROAD"),
    ("BUILDINGS", "1WALL_BLD"),
    ("BUILDINGS", "BLD_"),
    ("WALLS", "1WALL"),
    ("WALLS", "TYRES"),
    ("VEGETATION", "TREES"),
    ("VEGETATION", "VEG_"),
]


def collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def move_to(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


def fix_materials():
    tex_dir = os.path.join(HERE, "texture")
    for mat in list(bpy.data.materials):
        base = mat.name.split(".")[0]
        png = os.path.join(tex_dir, base + ".png")
        if not mat.use_nodes or not os.path.exists(png):
            continue
        img = bpy.data.images.load(png, check_existing=True)
        img.name = base + ".png"
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE":
                node.image = img
        mat.name = base
        if base in ("trees", "veg_grass", "veg_grass_short"):  # alpha-tested foliage (ksPerPixelAT)
            try:
                mat.blend_method = "CLIP"
            except Exception:
                pass
    # merge duplicated materials (asphalt.001 -> asphalt)
    for obj in bpy.data.objects:
        for slot in obj.material_slots:
            if slot.material and "." in slot.material.name:
                base = bpy.data.materials.get(slot.material.name.split(".")[0])
                if base:
                    slot.material = base
    for img in list(bpy.data.images):
        if img.users == 0:
            bpy.data.images.remove(img)


def add_ac_objects():
    col = collection("AC_OBJECTS")
    objs = json.load(open(os.path.join(HERE, META["objects"]), encoding="utf-8"))
    for o in objs:
        f = Vector(o["forward"]).normalized()
        if FLIP_SPAWN_DIRECTION:
            f = -f
        up = Vector((0.0, 0.0, 1.0))
        x = up.cross(f).normalized()
        # Assetto Corsa dummies: local Y = up, local Z = driving direction
        m = Matrix((
            (x.x, up.x, f.x, o["pos"][0]),
            (x.y, up.y, f.y, o["pos"][1]),
            (x.z, up.z, f.z, o["pos"][2]),
            (0, 0, 0, 1),
        ))
        e = bpy.data.objects.get(o["name"]) or bpy.data.objects.new(o["name"], None)
        e.empty_display_type = "ARROWS" if not o["name"].startswith("AC_TIME") and "_AB_" not in o["name"] else "CUBE"
        e.empty_display_size = 1.5 if e.empty_display_type == "ARROWS" else 0.5
        e.matrix_world = m
        if e.name not in col.objects:
            for c in list(e.users_collection):
                c.objects.unlink(e)
            col.objects.link(e)


def organise():
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for col_name, prefix in GROUPS:
            if obj.name.startswith(prefix):
                move_to(obj, collection(col_name))
                break


def main():
    # start from an empty scene (without resetting the UI / this script)
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(block):
            if item.users == 0:
                block.remove(item)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0
    bpy.ops.import_scene.gltf(filepath=os.path.join(HERE, META["glb"]))
    # glTF importer may add a root "world" parent on some versions - clear parents, keep transforms
    for obj in bpy.data.objects:
        if obj.parent:
            mw = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = mw
    fix_materials()
    organise()
    add_ac_objects()
    blend = os.path.join(HERE, META["blend"])
    bpy.ops.wm.save_as_mainfile(filepath=blend, relative_remap=True)
    sys.path.insert(0, HERE)
    import export_fbx  # noqa: E402
    export_fbx.export(os.path.join(HERE, META["fbx"]))
    print("maptotrack: saved", blend)


main()

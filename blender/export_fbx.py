"""Export the current Blender scene to track.fbx with settings that work with ksEditor.

After fine-tuning the track in Blender, run this from the Scripting tab
(or: blender track.blend --background --python export_fbx.py).
"""
import os

import bpy

# If the track shows up 100x too big or too small in ksEditor, try "FBX_SCALE_UNITS" or "FBX_SCALE_NONE".
APPLY_SCALE = "FBX_SCALE_ALL"
MAX_VERTS = 65000


def check():
    problems = []
    for obj in bpy.data.objects:
        if obj.type == "MESH" and len(obj.data.vertices) > MAX_VERTS:
            problems.append(f"{obj.name}: {len(obj.data.vertices)} vertices (KN5 limit ~65k, split it)")
    for name in ("AC_START_0", "AC_PIT_0", "AC_HOTLAP_START_0"):
        if name not in bpy.data.objects:
            problems.append(f"missing {name}")
    return problems


def export(path=None):
    if path is None:
        base = os.path.dirname(bpy.data.filepath) or os.getcwd()
        path = os.path.join(base, "track.fbx")
    for p in check():
        print("maptotrack WARNING:", p)
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=False,
        object_types={"MESH", "EMPTY"},
        apply_unit_scale=True,
        apply_scale_options=APPLY_SCALE,
        global_scale=1.0,
        axis_forward="-Z",
        axis_up="Y",
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        use_tspace=False,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode="RELATIVE",
        embed_textures=False,
    )
    print("maptotrack: exported", path)
    return path


if __name__ == "__main__":
    export()

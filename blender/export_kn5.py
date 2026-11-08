"""Export a Blender scene (or an FBX file) straight to an Assetto Corsa .kn5 - no ksEditor needed.

Used by the maptotrack app ("Build KN5"), but also works on its own:
  * Blender > Scripting tab > open this file > Run Script   (exports the open scene)
  * blender track.blend --background --python export_kn5.py
  * blender --background --factory-startup --python export_kn5.py -- --fbx some.fbx --out some.kn5

What gets exported:
  * every visible-in-render mesh object (modifiers applied), one kn5 mesh per material.
    Object names matter: 1ROAD..., 1GRASS..., 1WALL... are the physical surfaces (data/surfaces.ini).
  * empties named AC_... (AC_START_0, AC_PIT_0, AC_TIME_0_L, AC_HOTLAP_START_0 ...): local Y up, Z forward.
  * the first Image Texture of each material as txDiffuse (PNG/DDS/JPG embedded as is).
  * shaders: ksPerPixel, ksPerPixelAT (alpha tested) for materials named trees*; edit MATERIAL_RULES in
    kn5.py or add kn5_materials.json next to the app to change them.
Objects or materials whose name starts with "__" are skipped.
"""
import json
import os
import sys
import tempfile

import bpy
import numpy as np


def _here():
    for cand in (globals().get("__file__"), bpy.data.filepath):
        if cand and os.path.exists(cand):
            return os.path.dirname(os.path.abspath(cand))
    return os.getcwd()


HERE = _here()
sys.path.insert(0, HERE)
import kn5  # noqa: E402  (copied next to this script by maptotrack)


def log(*a):
    print("maptotrack:", *a, flush=True)


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {}
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            out[argv[i][2:]] = argv[i + 1]
            i += 2
        else:
            i += 1
    return out


def to_ac(v):
    """Blender (x, y, z up) -> Assetto Corsa (x, y up, z = -y)."""
    v = np.asarray(v, np.float64)
    return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])


def image_bytes(img):
    """Bytes of an image for the kn5 (packed data, the file on disk, or a PNG re-save)."""
    if img.packed_file is not None:
        return img.packed_file.data
    path = bpy.path.abspath(img.filepath) if img.filepath else ""
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            data = fh.read()
        if data[:4] in (b"\x89PNG", b"DDS ") or data[:2] == b"\xff\xd8":
            return data
    tmp = os.path.join(tempfile.gettempdir(), "maptotrack_kn5_tex.png")
    old = (img.filepath_raw, img.file_format)
    try:
        img.filepath_raw, img.file_format = tmp, "PNG"
        img.save()
    finally:
        img.filepath_raw, img.file_format = old
    with open(tmp, "rb") as fh:
        return fh.read()


def material_texture(mat):
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    nodes = [n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and n.image is not None]
    # prefer the one wired into Base Color
    for n in nodes:
        for link in n.outputs["Color"].links:
            if link.to_socket.name in ("Base Color", "Color"):
                return n.image
    return nodes[0].image if nodes else None


def tex_name(img):
    name = img.name
    if not os.path.splitext(name)[1]:
        name += ".png"
    return name


def mesh_arrays(obj, dg):
    """Triangulated corner data of an evaluated mesh object in world space."""
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        if nt == 0:
            return None
        tri_loops = np.empty(nt * 3, np.int64)
        me.loop_triangles.foreach_get("loops", tri_loops)
        tri_mat = np.empty(nt, np.int64)
        me.loop_triangles.foreach_get("material_index", tri_mat)
        nl = len(me.loops)
        loop_vert = np.empty(nl, np.int64)
        me.loops.foreach_get("vertex_index", loop_vert)
        co = np.empty(len(me.vertices) * 3, np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        # corner normals (Blender 4.1+: corner_normals; older: calc_normals_split + loop.normal)
        nrm = np.empty(nl * 3, np.float64)
        if hasattr(me, "corner_normals"):
            me.corner_normals.foreach_get("vector", nrm)
        else:
            me.calc_normals_split()
            me.loops.foreach_get("normal", nrm)
        nrm = nrm.reshape(-1, 3)
        uv = np.zeros((nl, 2), np.float64)
        if me.uv_layers.active is not None:
            buf = np.empty(nl * 2, np.float64)
            me.uv_layers.active.data.foreach_get("uv", buf)
            uv = buf.reshape(-1, 2)
        mats = [s.material for s in obj.material_slots] or [None]
    finally:
        ev.to_mesh_clear()
    M = np.array(obj.matrix_world, np.float64)
    pos = co @ M[:3, :3].T + M[:3, 3]
    Nm = np.linalg.inv(M[:3, :3]).T
    nrm = nrm @ Nm.T
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    flip = np.linalg.det(M[:3, :3]) < 0      # mirrored objects: reverse the winding
    return {"pos": pos, "loop_vert": loop_vert, "nrm": nrm, "uv": uv, "tri_loops": tri_loops.reshape(-1, 3),
            "tri_mat": tri_mat, "mats": mats, "flip": flip}


def export(out_path, overrides=None):
    dg = bpy.context.evaluated_depsgraph_get()
    kb = kn5.Kn5Builder(overrides=overrides)
    n_obj = n_dummy = 0
    problems = []
    for obj in bpy.context.scene.objects:
        if obj.name.startswith("__") or obj.hide_render:
            continue
        if obj.type == "EMPTY":
            if obj.name.startswith("AC_"):
                M = np.array(obj.matrix_world, np.float64)
                R = M[:3, :3] / np.maximum(np.linalg.norm(M[:3, :3], axis=0, keepdims=True), 1e-12)
                axes = to_ac(R.T)                    # rows: local X, Y, Z axes in AC coords
                m = np.eye(4)
                m[:3, :3] = axes
                m[3, :3] = to_ac(M[:3, 3][None])[0]
                kb.add_dummy(obj.name, m)
                n_dummy += 1
            continue
        if obj.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT"):
            continue
        d = mesh_arrays(obj, dg)
        if d is None:
            continue
        for mi in np.unique(d["tri_mat"]):
            mat = d["mats"][min(mi, len(d["mats"]) - 1)]
            if mat is not None and mat.name.startswith("__"):
                continue
            mname = mat.name if mat is not None else "default"
            img = material_texture(mat)
            tname = None
            if img is not None:
                tname = tex_name(img)
                if tname not in kb.textures:
                    try:
                        kb.add_texture(tname, image_bytes(img))
                    except Exception as exc:
                        problems.append(f"texture {img.name}: {exc}")
                        tname = None
            if mname not in kb._mat_index:
                kb.material(mname, texture=tname)
            loops = d["tri_loops"][d["tri_mat"] == mi]
            if d["flip"]:
                loops = loops[:, ::-1]
            corners = loops.ravel()
            P = to_ac(d["pos"][d["loop_vert"][corners]])
            N = to_ac(d["nrm"][corners])
            U = d["uv"][corners].copy()
            U[:, 1] = 1.0 - U[:, 1]                  # Blender v-up -> image convention
            # merge identical corners into shared vertices
            key = np.column_stack([np.round(P, 4), np.round(N, 3), np.round(U, 5)])
            uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
            faces = inv.reshape(-1, 3)
            kb.add_mesh(obj.name, mname, P[first], faces, uv=U[first], normals=N[first])   # draw distance by name
        n_obj += 1
    if not any(n.startswith("AC_START_") for n, _ in kb.dummies):
        problems.append("no AC_START_0 empty - the game needs at least one start position")
    kb.write(out_path)
    log(f"wrote {out_path}: {n_obj} objects -> {len(kb.meshes)} meshes, {len(kb.materials)} materials, "
        f"{len(kb.textures)} textures, {n_dummy} AC_ empties")
    for p in problems:
        log("WARNING", p)
    return out_path


def main():
    a = args()
    if "fbx" in a:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=a["fbx"], axis_forward="-Z", axis_up="Y")
        log("imported", a["fbx"])
    elif "blend" in a:
        bpy.ops.wm.open_mainfile(filepath=a["blend"])
    out = a.get("out")
    if not out:
        meta = os.path.join(HERE, "track_meta.json")
        if os.path.exists(meta):
            m = json.load(open(meta, encoding="utf-8"))
            out = os.path.join(m["ac_folder"], m["slug"] + ".kn5")
        else:
            out = os.path.join(HERE, "track.kn5")
    overrides = None
    if a.get("overrides") and os.path.exists(a["overrides"]):
        overrides = json.load(open(a["overrides"], encoding="utf-8"))
    export(out, overrides)


if __name__ == "__main__":
    main()

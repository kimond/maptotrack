"""Assetto Corsa .kn5 writer (and a small reader to check files).

Pure Python + numpy, no Blender or PIL needed, so the same module runs inside the app and inside
Blender (blender/export_kn5.py imports a copy of it).

KN5 v5 layout (little endian; string = uint32 length + utf-8 bytes):
    "sc6969", uint32 version (5)
    textures:  int32 n, then per texture: int32 active(1), string name, uint32 size + image bytes (PNG/DDS)
    materials: int32 n, then per material: string name, string shader, uint8 blend mode, bool alpha tested,
               int32 depth mode, uint32 n props {string name, f32 A, f32x2 B, f32x3 C, f32x4 D},
               uint32 n textures {string slot name, uint32 slot, string texture name}
    nodes (depth first, starting with one root):
        dummy (class 1): uint32 class, string name, uint32 children, bool active, f32x16 matrix
        mesh  (class 2): uint32 class, string name, uint32 children(0), bool active, bool cast shadows,
                         bool visible, bool transparent, uint32 n verts,
                         n x (pos f32x3, normal f32x3, uv f32x2, tangent f32x3), uint32 n idx, uint16 idx[],
                         uint32 material, uint32 layer, f32 lod in, f32 lod out, f32x3 sphere centre,
                         f32 sphere radius, bool renderable
Coordinates are Assetto Corsa's: x, y up, z = -north (same as glTF). Matrices are row-vector style
(rows = local X, Y, Z axes, then translation). UVs use the image convention (v = 0 at the top).
"""
from __future__ import annotations

import re
import struct

import numpy as np

MAGIC = b"sc6969"
MAX_VERTS = 65535

_VERT = np.dtype([("pos", "<f4", 3), ("nrm", "<f4", 3), ("uv", "<f4", 2), ("tan", "<f4", 3)])

# ------------------------------------------------------------------ default shaders per material
# (shader, alpha tested, properties). Property values: float or tuple (stored in the matching slot).
_BASE = {"ksAmbient": 0.45, "ksDiffuse": 0.45, "ksSpecular": 0.0, "ksSpecularEXP": 1.0,
         "ksEmissive": (0.0, 0.0, 0.0), "ksAlphaRef": 0.0}
MATERIAL_RULES = [
    # regex on the material name, settings
    # leaf / grass cards: ksPerPixelAT_NS - cards with up-pointing normals shade themselves in stripes with
    # ksPerPixelAT (Map2Lap's finding, checked in game there)
    (r"^trees", {"shader": "ksPerPixelAT_NS", "alpha_tested": True, "shadows": True,
                 "props": {"ksAmbient": 0.5, "ksDiffuse": 0.4}}),
    (r"^veg_", {"shader": "ksPerPixelAT_NS", "alpha_tested": True, "shadows": False,
                "props": {"ksAmbient": 0.55, "ksDiffuse": 0.35}}),
    (r"^props", {"shader": "ksPerPixelAT", "alpha_tested": True, "shadows": True,
                 "props": {"ksAmbient": 0.45, "ksDiffuse": 0.5, "ksSpecular": 0.1, "ksSpecularEXP": 20.0}}),
    (r"^tyres", {"props": {"ksAmbient": 0.45, "ksDiffuse": 0.45, "ksSpecular": 0.08, "ksSpecularEXP": 12.0},
                 "shadows": True}),
    (r"^(asphalt|road|paved)", {"props": {"ksAmbient": 0.4, "ksDiffuse": 0.4, "ksSpecular": 0.12,
                                           "ksSpecularEXP": 30.0}, "shadows": False}),
    (r"^(ground|grass|shoulder|terrain|near_|gravel)", {"shadows": False}),
    (r"^(siding|facade|wall|concrete|shingle|roof)", {"props": {"ksSpecular": 0.05, "ksSpecularEXP": 10.0},
                                                      "shadows": True}),
]


# Draw distance (kn5 "LOD out", metres) by mesh name prefix; 0 = always drawn. AC loads the whole track and
# draws everything in view: small things far away cost frame time for nothing. First match wins.
VIEW_DISTANCE = [
    ("VEG_GRASS", 90.0), ("VEG_TALLGRASS", 200.0),
    ("TREES", 1200.0),
    ("1WALL_BLD", 2000.0), ("BLD_", 2000.0),
    ("1WALL_PROPS", 900.0), ("PROPS", 900.0), ("WIRES_HV", 2500.0), ("WIRES", 500.0), ("PYLONS", 4000.0), ("1WALL_PYLON", 4000.0),
    ("1WALL_", 1500.0),
]


def view_distance(name: str) -> float:
    return next((d for pre, d in VIEW_DISTANCE if name.startswith(pre)), 0.0)


def material_settings(name: str, overrides: dict | None = None) -> dict:
    """Shader settings for a material name; `overrides` maps regex -> partial settings (user file)."""
    s = {"shader": "ksPerPixel", "alpha_tested": False, "blend": 0, "depth": 0, "shadows": True,
         "props": dict(_BASE)}
    rules = list(MATERIAL_RULES) + list((overrides or {}).items())
    for pat, r in rules:
        if re.match(pat, name, re.IGNORECASE):
            s.update({k: v for k, v in r.items() if k != "props"})
            s["props"] = dict(s["props"], **r.get("props", {}))
    return s


# ------------------------------------------------------------------ geometry helpers
def vertex_normals(pos, faces):
    """Area-weighted vertex normals (shared vertices are smoothed, split vertices stay flat)."""
    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(pos, dtype=np.float64)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), np.array([0.0, 1.0, 0.0]))
    return n


def tangents(pos, nrm, uv, faces):
    """Per-vertex tangents from UV gradients (needed by normal-mapped shaders; harmless otherwise)."""
    p0, p1, p2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    t0, t1, t2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) > 1e-12, 1.0 / np.where(np.abs(r) > 1e-12, r, 1.0), 0.0)
    ft = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    t = np.zeros_like(pos, dtype=np.float64)
    for k in range(3):
        np.add.at(t, faces[:, k], ft)
    t -= nrm * np.sum(t * nrm, axis=1, keepdims=True)          # Gram-Schmidt against the normal
    ln = np.linalg.norm(t, axis=1, keepdims=True)
    fallback = np.cross(nrm, np.array([0.0, 0.0, 1.0]))
    fallback /= np.maximum(np.linalg.norm(fallback, axis=1, keepdims=True), 1e-9)
    return np.where(ln > 1e-9, t / np.maximum(ln, 1e-9), fallback)


def split_mesh(pos, nrm, uv, faces, limit=MAX_VERTS):
    """Split into pieces with < limit vertices (uint16 indices)."""
    if len(pos) <= limit:
        yield pos, nrm, uv, faces
        return
    start = 0
    while start < len(faces):
        # grow a face range until it references too many distinct vertices
        lo, hi = start + 1, len(faces)
        best = start + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if len(np.unique(faces[start:mid])) < limit:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        sub = faces[start:best]
        used, inv = np.unique(sub, return_inverse=True)
        yield pos[used], nrm[used], uv[used], inv.reshape(-1, 3)
        start = best


def identity():
    return np.eye(4)


def dummy_matrix(pos, forward, up=(0.0, 1.0, 0.0)):
    """AC dummy (spawn / timing point) in AC coords: local Y = up, local Z = forward."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    u = np.asarray(up, float)
    x = np.cross(u, f)
    x /= np.linalg.norm(x)
    u = np.cross(f, x)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = x, u, f, pos
    return m


# ------------------------------------------------------------------ writer
class _W:
    def __init__(self, f):
        self.f = f

    def raw(self, b):
        self.f.write(b)

    def u32(self, v):
        self.f.write(struct.pack("<I", int(v)))

    def i32(self, v):
        self.f.write(struct.pack("<i", int(v)))

    def u8(self, v):
        self.f.write(struct.pack("<B", int(v)))

    def bool(self, v):
        self.f.write(struct.pack("<?", bool(v)))

    def f32(self, *v):
        self.f.write(struct.pack(f"<{len(v)}f", *[float(x) for x in v]))

    def str(self, s):
        b = s.encode("utf-8")
        self.u32(len(b))
        self.f.write(b)

    def blob(self, b):
        self.u32(len(b))
        self.f.write(b)


class Kn5Builder:
    """Collects textures, materials, meshes and dummies, then writes one .kn5."""

    def __init__(self, overrides: dict | None = None):
        self.textures = {}            # name -> bytes
        self.materials = []           # dicts
        self._mat_index = {}
        self.meshes = []              # dicts
        self.dummies = []             # (name, matrix)
        self.overrides = overrides or {}

    def add_texture(self, name, data: bytes):
        self.textures[name] = data
        return name

    def material(self, name, texture=None, extra_textures=None, extra_props=None, **kw):
        """Index of material `name` (created with default shader settings on first use).
        extra_textures: more slots after txDiffuse ({slot name: texture name}); extra_props: more shader
        properties (e.g. ksMultilayer's multR...). User overrides (kn5_materials.json) still win."""
        if name in self._mat_index:
            return self._mat_index[name]
        s = material_settings(name, None)
        s.update(kw)
        if extra_props:
            s["props"] = dict(s["props"], **extra_props)
        if self.overrides:
            o = material_settings(name, self.overrides)
            base = material_settings(name, None)
            for k, v in o.items():
                if k == "props":
                    s["props"].update({pk: pv for pk, pv in v.items() if base["props"].get(pk) != pv})
                elif base.get(k) != v:
                    s[k] = v
        s["name"] = name
        s["textures"] = {"txDiffuse": texture} if texture else {}
        if extra_textures:
            s["textures"].update(extra_textures)
        self._mat_index[name] = len(self.materials)
        self.materials.append(s)
        return self._mat_index[name]

    def add_mesh(self, name, material, pos, faces, uv=None, normals=None, shadows=None, visible=True, lod_out=None):
        if lod_out is None:
            lod_out = view_distance(name)
        pos = np.asarray(pos, np.float64)
        faces = np.asarray(faces, np.int64).reshape(-1, 3)
        if len(faces) == 0:
            return
        uv = np.zeros((len(pos), 2)) if uv is None else np.asarray(uv, np.float64)
        nrm = vertex_normals(pos, faces) if normals is None else np.asarray(normals, np.float64)
        tan = tangents(pos, nrm, uv, faces)
        mi = material if isinstance(material, int) else self.material(material)
        if shadows is None:
            shadows = self.materials[mi]["shadows"]
        pieces = list(split_mesh(np.hstack([pos, tan]), nrm, uv, faces))
        for k, (pt, n, u, f) in enumerate(pieces):
            self.meshes.append({"name": name if len(pieces) == 1 else f"{name}_{k}", "mat": mi,
                                "pos": pt[:, :3], "tan": pt[:, 3:], "nrm": n, "uv": u, "faces": f,
                                "shadows": shadows, "visible": visible, "lod_out": float(lod_out or 0.0)})

    def add_dummy(self, name, matrix):
        self.dummies.append((name, np.asarray(matrix, float)))

    # -------------------------------------------------------------- output
    def write(self, path, root_name="maptotrack"):
        used_tex = {t for m in self.materials for t in m["textures"].values()}
        missing = used_tex - set(self.textures)
        if missing:
            raise ValueError(f"textures referenced but not added: {sorted(missing)}")
        tex_names = [n for n in self.textures if n in used_tex]
        with open(path, "wb") as fh:
            w = _W(fh)
            w.raw(MAGIC)
            w.u32(5)
            w.i32(len(tex_names))
            for n in tex_names:
                w.i32(1)
                w.str(n)
                w.blob(self.textures[n])
            w.i32(len(self.materials))
            for m in self.materials:
                w.str(m["name"])
                w.str(m["shader"])
                w.u8(m.get("blend", 0))
                w.bool(m.get("alpha_tested", False))
                w.i32(m.get("depth", 0))
                props = m["props"]
                w.u32(len(props))
                for pn, pv in props.items():
                    w.str(pn)
                    vals = list(pv) if isinstance(pv, (tuple, list)) else [pv]
                    a = vals[0] if len(vals) == 1 else 0.0
                    b = vals if len(vals) == 2 else [0.0, 0.0]
                    c = vals if len(vals) == 3 else [0.0, 0.0, 0.0]
                    d = vals if len(vals) == 4 else [0.0, 0.0, 0.0, 0.0]
                    w.f32(a)
                    w.f32(*b)
                    w.f32(*c)
                    w.f32(*d)
                w.u32(len(m["textures"]))
                for slot, (sname, tname) in enumerate(m["textures"].items()):
                    w.str(sname)
                    w.u32(slot)
                    w.str(tname)
            # root dummy with every mesh and dummy as direct children
            w.u32(1)
            w.str(root_name)
            w.u32(len(self.meshes) + len(self.dummies))
            w.bool(True)
            w.f32(*identity().ravel())
            for name, mtx in self.dummies:
                w.u32(1)
                w.str(name)
                w.u32(0)
                w.bool(True)
                w.f32(*mtx.ravel())
            for m in self.meshes:
                self._write_mesh(w, m)

    @staticmethod
    def _write_mesh(w, m):
        w.u32(2)
        w.str(m["name"])
        w.u32(0)
        w.bool(True)                 # active
        w.bool(m["shadows"])         # cast shadows
        w.bool(m["visible"])         # visible
        w.bool(False)                # transparent
        n = len(m["pos"])
        if n > MAX_VERTS:
            raise ValueError(f"{m['name']}: {n} vertices (max {MAX_VERTS})")
        w.u32(n)
        v = np.empty(n, _VERT)
        v["pos"], v["nrm"], v["uv"], v["tan"] = m["pos"], m["nrm"], m["uv"], m["tan"]
        w.raw(v.tobytes())
        idx = np.asarray(m["faces"], np.uint16).ravel()
        w.u32(len(idx))
        w.raw(idx.astype("<u2").tobytes())
        w.u32(m["mat"])
        w.u32(0)                     # layer
        w.f32(0.0, m.get("lod_out", 0.0))   # lod in / out in metres (0 = always drawn)
        lo, hi = m["pos"].min(0), m["pos"].max(0)
        c = (lo + hi) / 2
        w.f32(*c)
        w.f32(float(np.linalg.norm(hi - lo) / 2))
        w.bool(True)                 # renderable


# ------------------------------------------------------------------ reader (checks / tests)
def read(path, with_data=False):
    """Parse a .kn5 -> {'version', 'textures': [(name, size, magic)], 'materials': [...], 'nodes': [...]}."""
    data = open(path, "rb").read()
    o = 0

    def take(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, data, o)
        o += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]

    def string():
        nonlocal o
        n = take("I")
        s = data[o:o + n].decode("utf-8")
        o += n
        return s

    assert data[:6] == MAGIC, "not a kn5 file"
    o = 6
    ver = take("I")
    if ver > 5:
        take("i")
    out = {"version": ver, "textures": [], "materials": [], "nodes": []}
    for _ in range(take("i")):
        take("i")
        name = string()
        n = take("I")
        out["textures"].append((name, n, data[o:o + 4]))
        o += n
    for _ in range(take("i")):
        m = {"name": string(), "shader": string(), "blend": take("B"), "alpha_tested": take("?"),
             "depth": take("i"), "props": {}, "textures": {}}
        for _ in range(take("I")):
            pn = string()
            vals = take("10f")
            m["props"][pn] = vals
        for _ in range(take("I")):
            sn = string()
            take("I")
            m["textures"][sn] = string()
        out["materials"].append(m)

    def node(depth):
        nonlocal o
        cls = take("I")
        name = string()
        nch = take("I")
        take("?")
        nd = {"class": cls, "name": name, "depth": depth}
        if cls == 1:
            nd["matrix"] = np.array(take("16f")).reshape(4, 4)
        elif cls == 2:
            nd["shadows"], nd["visible"], nd["transparent"] = take("?"), take("?"), take("?")
            nv = take("I")
            verts = np.frombuffer(data, _VERT, nv, o)
            o += nv * _VERT.itemsize
            ni = take("I")
            idx = np.frombuffer(data, "<u2", ni, o)
            o += ni * 2
            nd["material"] = take("I")
            take("I")
            take("2f")
            nd["sphere"] = take("4f")
            take("?")
            nd["nverts"], nd["nidx"] = nv, ni
            if with_data:
                nd["verts"], nd["idx"] = verts.copy(), idx.copy()
        else:
            raise ValueError(f"unsupported node class {cls} ({name})")
        out["nodes"].append(nd)
        for _ in range(nch):
            node(depth + 1)

    node(0)
    assert o == len(data), f"trailing bytes: {len(data) - o}"
    return out

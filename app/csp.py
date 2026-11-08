"""Custom Shaders Patch (CSP) extras written into the track folder: extension/ext_config.ini.

Grass FX: CSP grows real 3D grass (thousands of blades, swaying in the wind, shadowed) on the ground materials
listed in [GRASS_FX] - https://cup.acstuff.club/docs/csp/tracks/grass-fx . Without CSP the file is ignored and
the track keeps the app's own grass cards.

Which grass where (one surface class per near-track material, see groundmat.NEAR_CLASSES):
  * lawn (near_lawn): short, tidy, mown grass                       -> configuration A
  * wild / dry grass (near_wild, near_dry): tall, untidy, uncut      -> configuration B
  * the ground further out (ground_jj_ii tiles: every surface painted into one texture): Grass FX's colour mask
    decides - green parts grow grass, sand / gravel / asphalt colours don't (base [GRASS_FX] settings)
  * nothing on asphalt, road paint, gravel shoulder, driveways, sand, gravel or soil; grass fades out softly
    at sand / gravel / soil edges (OCCLUDING_MATERIALS_ALPHA)
Our own short lawn tufts (veg_grass_short) are hidden when Grass FX runs (ORIGINAL_GRASS_MATERIALS); the tall
grass clumps (veg_grass) stay - they are taller than Grass FX blades.
"""
from __future__ import annotations

import numpy as np

MARKER = "; Written by maptotrack"

LAWN = ("near_lawn",)
WILD = ("near_wild", "near_dry")
NO_GRASS = ("asphalt_tiled", "road_paint", "gravel_tiled", "near_paved")
SOFT_EDGE = ("near_gravel", "near_sand", "near_soil")
OWN_TUFTS = ("veg_grass_short",)

# spawn on every pixel of the material (the material itself says "grass here")
_EVERYWHERE = """MASK_MAIN_THRESHOLD = -1
MASK_RED_THRESHOLD = 0
MASK_MIN_LUMINANCE = -1
MASK_MAX_LUMINANCE = 1"""


def _list(names):
    return ", ".join(sorted(names))


def grass_fx_ini(materials, density=1.0) -> str:
    """The [GRASS_FX] sections for a track using these material names (only the ones present are listed).
    density: the app's grass density (0.5 / 1 / 1.8) - Grass FX's SHAPE_SIZE works the other way round
    (bigger = fewer, larger blades), so it only nudges the size a little."""
    mats = set(materials)
    lawn = [m for m in LAWN if m in mats]
    wild = [m for m in WILD if m in mats]
    ground = sorted(m for m in mats if m.startswith("ground_"))
    grass = lawn + wild + ground
    if not grass:
        return ""
    occl = [m for m in NO_GRASS if m in mats]
    soft = [m for m in SOFT_EDGE if m in mats]
    tufts = [m for m in OWN_TUFTS if m in mats]
    k = 1.0 / max(0.6, min(density, 1.6)) ** 0.25
    lines = [
        "[GRASS_FX]",
        f"GRASS_MATERIALS = {_list(grass)}",
        f"OCCLUDING_MATERIALS = {_list(occl)}",
        f"OCCLUDING_MATERIALS_ALPHA = {_list(soft)}",
        f"ORIGINAL_GRASS_MATERIALS = {_list(tufts)}",
        "; ground tiles: only where the texture is green (lawns, fields; not sand, gravel or driveways)",
        "MASK_MAIN_THRESHOLD = 0.35",
        "MASK_RED_THRESHOLD = 0.05",
        "MASK_MIN_LUMINANCE = 0.02",
        "MASK_MAX_LUMINANCE = 0.6",
        "MASK_BLUR = 1",
        f"SHAPE_SIZE = {1.3 * k:.2f}",
        "SHAPE_TIDY = 0.3",
        "SHAPE_CUT = 0.2",
        "SHAPE_WIDTH = 1.0",
        "",
    ]
    if lawn:
        lines += [
            "; A: mown lawn - short, even, cut every week",
            "[GRASS_FX_CONFIGURATION_A]",
            _EVERYWHERE,
            "TRIM_PERIOD = 7",
            f"SHAPE_SIZE = {0.8 * k:.2f}, {1.1 * k:.2f}",
            "SHAPE_TIDY = 1, 0.6",
            "SHAPE_CUT = 1, 0.5",
            "SHAPE_WIDTH = 1.0",
            "",
            "[GRASS_FX_ADJUSTMENT_0]",
            f"MATERIALS = {_list(lawn)}",
            "MAP = A, 1",
            "",
        ]
    if wild:
        lines += [
            "; B: wild and dry grass - tall, untidy, never cut",
            "[GRASS_FX_CONFIGURATION_B]",
            _EVERYWHERE,
            "MASK_BLUR = 2",
            f"SHAPE_SIZE = {1.8 * k:.2f}",
            "SHAPE_TIDY = 0",
            "SHAPE_CUT = 0",
            "SHAPE_WIDTH = 0.9",
            "",
            "[GRASS_FX_ADJUSTMENT_1]",
            f"MATERIALS = {_list(wild)}",
            "MAP = B, 1",
            "",
        ]
    return "\n".join(lines)


# ------------------------------------------------------------------ CSP trees
# CSP draws real 3D trees (LODs, billboards far away, seasons, baked shadows) from a list of positions and
# tree models in its own .bin format (made with Content Manager's tree converter, Content > Tools > Creator
# tools). CSP ships no trees: the user puts .bin trees in maptotrack/csp_trees/ and the app places them where
# the real trees are. https://cup.acstuff.club/docs/csp/tracks/trees
TREE_KINDS = {0: "conifer", 1: "broadleaf", 2: "bush"}
_KIND_WORDS = {
    0: ("pine", "spruce", "fir", "conifer", "cedar", "larch", "sapin", "epinette", "pin_", "thuja", "cypress"),
    2: ("bush", "shrub", "hedge", "buisson", "arbuste", "haie"),
    1: ("birch", "maple", "oak", "poplar", "aspen", "willow", "elm", "ash", "beech", "linden", "broadleaf",
        "deciduous", "bouleau", "erable", "chene", "peuplier", "tremble", "tree"),
}
DEFAULT_HEIGHT = {0: 15.0, 1: 12.0, 2: 2.5}       # m: assumed height of a model when trees.json doesn't say
README_TREES = """Put CSP tree models (.bin) here - maptotrack places them where the real trees are.

* Make them with Content Manager: Content > Tools > Creator tools > tree converter (from an FBX with up to
  three LODs), or use trees from a track / tree pack whose licence lets you reuse them.
* The kind comes from the file name: pine / spruce / fir ... = conifer; birch / maple / oak ... = broadleaf;
  bush / shrub / hedge = bush. Several models of a kind are mixed.
* Optional trees.json gives the kind and the model's own height (m), so the trees get their real size:
    {"my_pine.bin": {"kind": "conifer", "height": 16}, "birch_a.bin": {"kind": "broadleaf", "height": 11}}
* Without CSP the track keeps maptotrack's own trees.
"""


def tree_dir():
    from . import config

    return config.ROOT / "csp_trees"


def find_tree_models(folder=None, log=print):
    """{kind: [(file name, model height m)]} of the .bin trees in csp_trees/ (created with a README if missing).
    A kind without a model borrows one of another kind (bushes: a small broadleaf)."""
    import json

    folder = folder or tree_dir()
    if not folder.exists():
        try:
            folder.mkdir(parents=True)
            (folder / "README.txt").write_text(README_TREES, encoding="utf-8")
        except OSError:
            pass
        return {}
    meta = {}
    if (folder / "trees.json").exists():
        try:
            meta = json.loads((folder / "trees.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log(f"  csp_trees/trees.json unreadable ({exc}) - using file names")
    names = {v: k for k, v in TREE_KINDS.items()}
    out = {0: [], 1: [], 2: []}
    for f in sorted(folder.glob("*.bin")):
        m = meta.get(f.name, {})
        kind = names.get(str(m.get("kind", "")).lower())
        if kind is None:
            low = f.name.lower()
            kind = next((k for k in (0, 2, 1) if any(w in low for w in _KIND_WORDS[k])), 1)
        out[kind].append((f.name, float(m.get("height", DEFAULT_HEIGHT[kind]))))
    if not any(out.values()):
        return {}
    for k, alt in ((0, 1), (1, 0), (2, 1), (2, 0)):
        if not out[k] and out[alt]:
            out[k] = list(out[alt])
    return out


def tree_list(trees, ground_z, models, seed=6):
    """trees.txt for CSP: one line per tree, the model of its kind (picked at random), turned at random, sized
    to its measured height (and crown width relative to height). Positions in AC coordinates (x, up, -y)."""
    rng = np.random.default_rng(seed)
    lines = ["# maptotrack: trees where the aerial photo / LiDAR / land cover put them",
             "configure: color variance = 0.06",
             "configure: brightness variance = 0.08"]
    kind = np.asarray(trees["kind"], int)
    for x, y, z, h, w, k in zip(trees["x"], trees["y"], ground_z, trees["h"], trees["w"], kind):
        opts = models.get(int(k)) or models.get(1) or models.get(0)
        if not opts:
            continue
        name, mh = opts[rng.integers(len(opts))]
        size = float(np.clip(h / max(mh, 0.5), 0.2, 4.0))
        ratio = {0: 0.45, 1: 0.8, 2: 1.4}.get(int(k), 0.8)
        width = float(np.clip((w / max(h, 0.1)) / ratio, 0.6, 1.6))
        lines.append(f"tree: {name}; pos={x:.2f}, {z:.2f}, {-y:.2f}; angle = {rng.uniform(0, 360):.0f}; "
                     f"size = {size:.2f}; width = {width:.2f}")
    return "\n".join(lines) + "\n"


def trees_ini(materials):
    """[TREES] plus hiding maptotrack's own card trees while CSP draws its trees."""
    surf = sorted(m for m in materials if m.startswith(("ground_", "near_", "gravel_tiled")))
    out = ["; CSP trees: real 3D trees from csp_trees/*.bin at the trees' real positions", "[TREES]",
           "LIST_0 = trees/trees.txt"]
    if surf:
        out.append(f"SURFACE_MATERIALS = {_list(surf)}")
    out += ["", "; maptotrack's own trees (flat cards) are hidden when CSP draws the 3D ones",
            "[SHADER_REPLACEMENT_0_HIDE_CARD_TREES]", "MATERIALS = trees", "LOD_IN = 100000", ""]
    return "\n".join(out)


def write_trees(folder, trees, ground_z, models, log=print):
    """extension/trees/: the .bin models used and trees.txt. -> number of trees listed."""
    import shutil

    tdir = folder / "extension" / "trees"
    tdir.mkdir(parents=True, exist_ok=True)
    used = {n for v in models.values() for n, _ in v}
    for n in used:
        shutil.copy(tree_dir() / n, tdir / n)
    text = tree_list(trees, ground_z, models)
    (tdir / "trees.txt").write_text(text, encoding="utf-8")
    return text.count("\ntree: ") + text.startswith("tree: ")


# ------------------------------------------------------------------ night lights and rain
# Street lamps light the road at night: one CSP light under every lamp head (a [LIGHT_SERIES_] of positions),
# the lamp lenses (material props_lit) glow - both switched by CSP's built-in NIGHT_SMOOTH condition (sun below
# 88 degrees). https://cup.acstuff.club/docs/csp/tracks/lights
MAX_LIGHTS = 1500
LAMP_COLOR = "1, 0.84, 0.66, 4"         # warm white (sodium / 3000 K LED), intensity
LAMP_RANGE = 26.0


def lights_ini(lights) -> str:
    """lights: [(x, y, z)] in maptotrack's frame (x east, y north, z up) -> AC's (x, y up, z south)."""
    if not len(lights):
        return ""
    out = ["; street lamps: a light under every lamp head, on at night",
           "[LIGHT_SERIES_0]",
           "DESCRIPTION = street lamps (maptotrack)",
           "SPOT = 130",
           "SPOT_SHARPNESS = 0.35",
           f"RANGE = {LAMP_RANGE:.0f}",
           "RANGE_GRADIENT_OFFSET = 0.3",
           "DIRECTION = 0, -1, 0",
           f"COLOR = {LAMP_COLOR}",
           "FADE_AT = 450",
           "FADE_SMOOTH = 60",
           "DIFFUSE_CONCENTRATION = 0.8",
           "CONDITION = NIGHT_SMOOTH"]
    for i, (x, y, z) in enumerate(lights):
        out.append(f"POSITION_{i} = {x:.2f}, {z - 0.12:.2f}, {-y:.2f}")
    out += ["",
            "[MATERIAL_ADJUSTMENT_0]",
            "DESCRIPTION = lamp lenses glow at night",
            "MATERIALS = props_lit",
            "KEY_0 = ksEmissive",
            "VALUE_0 = 255, 214, 170, 0.04",
            "VALUE_0_OFF = 0",
            "CONDITION = NIGHT_SMOOTH",
            ""]
    return "\n".join(out)


def pick_lights(lights, dist=None, n=MAX_LIGHTS):
    """At most n lamps, the ones nearest the track first (dist: their distance to it)."""
    lights = list(lights)
    if len(lights) <= n or dist is None:
        return lights[:n]
    order = np.argsort(np.asarray(dist, float))[:n]
    return [lights[i] for i in sorted(order)]


# RainFX (CSP's rain): which materials get puddles, soak (darker), stay rough (grass), get drops (metal), and the
# painted lines (slippery when wet). https://cup.acstuff.club/docs/csp/tracks/rainfx
RAIN_GROUPS = (
    ("PUDDLES_MATERIALS", ("asphalt?", "road_asph?", "street_asph?")),
    ("SOAKING_MATERIALS", ("asphalt?", "gravel?", "near_gravel", "near_soil", "near_sand", "near_dry", "near_paved",
                           "concrete", "tyres")),
    ("SMOOTH_MATERIALS", ("props", "props_lit")),
    ("ROUGH_MATERIALS", ("ground_?", "near_lawn", "near_wild", "near_forest", "veg_?", "trees", "shingle_?",
                         "roof_?")),
    ("LINES_MATERIALS", ("road_paint", "road_paint_yellow")),
)


def _matches(pattern, materials):
    if pattern.endswith("?"):
        return any(m.startswith(pattern[:-1]) for m in materials)
    return pattern in materials


def rain_ini(materials) -> str:
    mats = set(materials)
    out = ["; rain (CSP RainFX): puddles on the asphalt, wet dark gravel and soil, rough grass, slippery paint",
           "[RAIN_FX]"]
    n = 0
    for key, pats in RAIN_GROUPS:
        use = [p for p in pats if _matches(p, mats)]
        if use:
            out.append(f"{key} = {', '.join(use)}")
            n += 1
    return "\n".join(out) + "\n" if n else ""


def write_ext_config(folder, materials, grass_fx=True, density=1.0, trees=None, log=print, lights=None,
                     rain=False):
    """folder/extension/ext_config.ini. A file the user wrote themselves (no maptotrack marker) is kept.
    trees: (trees dict, ground heights, models) for CSP trees, or None. lights: street lamp positions (night
    lights), rain: RainFX material groups."""
    path = folder / "extension" / "ext_config.ini"
    if path.exists() and not path.read_text(encoding="utf-8", errors="replace").startswith(MARKER):
        log("  extension/ext_config.ini was edited by hand - kept as it is")
        return None
    body = grass_fx_ini(materials, density) if grass_fx else ""
    if trees is not None:
        n = write_trees(folder, *trees, log=log)
        if n:
            body += ("\n" if body else "") + trees_ini(materials)
            log(f"  CSP trees: {n:,} trees from csp_trees/ ("
                + ", ".join(f"{TREE_KINDS[k]}: {', '.join(sorted({a for a, _ in v}))}" for k, v in trees[2].items() if v)
                + ")")
    if lights:
        body += ("\n" if body else "") + lights_ini(lights)
        log(f"  CSP night lights: {len(lights):,} street lamps light the road at night")
    if rain:
        r = rain_ini(materials)
        if r:
            body += ("\n" if body else "") + r
            log("  CSP RainFX: puddles, wet surfaces and slippery paint set up")
    if not body:
        if path.exists():
            path.unlink()
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (f"{MARKER}. Custom Shaders Patch settings for this track (ignored without CSP).\n"
            "; Delete this first line to keep your own edits when the track is generated again.\n\n" + body)
    path.write_text(text, encoding="utf-8")
    return path

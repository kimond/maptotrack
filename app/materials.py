"""Photo materials: CC0 colour maps from ambientCG (https://ambientcg.com), the same ones Map2Lap uses, and a
CC0 tyre rubber from TextureCan (https://www.texturecan.com, Plastic 0022) for the tyre walls.
Downloaded once into cache/textures (a few MB each). Only their structure is used (plaster, concrete pores,
asphalt aggregate, grass blades, gravel stones, soil), laid over our own layouts and colours:
  * house walls and roofs (textures.py),
  * the ground: baked into the ground tiles, and as close-up detail layers for AC's ksMultilayer (groundmat.py).
Without them (offline, download blocked) a procedural grain is used instead.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
from PIL import Image

from . import config

SOURCES = {"plaster": "Plaster001", "concrete": "Concrete034", "asphalt": "Asphalt031",
           "grass": "Grass005", "gravel": "Gravel043", "dirt": "Ground037"}
URL = "https://ambientcg.com/get?file={}_1K-JPG.zip"
CREDIT = "Textures: ambientCG (CC0)"
# other CC0 sources: key -> zip URL, the colour map's file name in the cache, words that pick it in the zip.
# A photo dropped into cache/textures under that file name is used as it is (no download).
EXTRA = {
    "rubber": {"url": "https://www.texturecan.com/downloads/plastic_0022/plastic_0022_1k_mfWrU6.zip",
               "file": "TextureCan_Plastic0022_1K_Color.jpg", "match": ("basecolor", "base_color", "color",
                                                                        "albedo", "diffuse", "col")},
}
EXTRA_CREDIT = {"rubber": "Tyre rubber: TextureCan (CC0)"}


def _path(name):
    return config.CACHE_DIR / "textures" / f"{name}_1K_Color.jpg"


def _colour_member(names, match):
    """The colour map in a texture zip (not the normal / roughness / height ... maps)."""
    skip = ("normal", "rough", "height", "disp", "ao", "ambient", "metal", "spec", "opacity", "bump", "preview")
    imgs = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    for word in match:
        for n in imgs:
            base = n.lower().rsplit("/", 1)[-1]
            if word in base and not any(k in base for k in skip):
                return n
    return None


_LOADED = {}


def fetch(log=print) -> dict:
    """{key: RGB image} of the photo materials available (downloads the missing ones)."""
    if _LOADED and len(_LOADED) == len(SOURCES) + len(EXTRA):
        return dict(_LOADED)
    out = {}
    missing = []
    for key, name in SOURCES.items():
        p = _path(name)
        if not p.exists() and not config.OFFLINE:
            try:
                import requests

                if not missing:
                    log("Downloading building materials (ambientCG, once) ...")
                missing.append(name)
                r = requests.get(URL.format(name), headers={"User-Agent": config.USER_AGENT}, timeout=(10, 120))
                r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    member = next(n for n in z.namelist() if n.endswith("_Color.jpg"))
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(z.read(member))
            except Exception as exc:
                log(f"  {name}: {exc} - using a procedural grain")
                continue
        if p.exists():
            try:
                out[key] = Image.open(p).convert("RGB")
            except OSError:
                continue
    for key, src in EXTRA.items():
        p = config.CACHE_DIR / "textures" / src["file"]
        if not p.exists() and not config.OFFLINE:
            try:
                import requests

                log(f"Downloading the {key} texture (once) ...")
                r = requests.get(src["url"], headers={"User-Agent": config.USER_AGENT}, timeout=(10, 120))
                r.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    member = _colour_member(z.namelist(), src["match"])
                    if member is None:
                        raise ValueError(f"no colour map in {z.namelist()[:6]}")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    img = Image.open(io.BytesIO(z.read(member))).convert("RGB")
                    img.save(p, quality=92)
            except Exception as exc:
                log(f"  {key} texture: {exc} - using a procedural grain")
                continue
        if p.exists():
            try:
                out[key] = Image.open(p).convert("RGB")
            except OSError:
                continue
    _LOADED.update(out)
    return out


def detail(img, w, h, fallback_seed=0, highpass=None):
    """Fine structure of a material, zero mean and unit spread, resized to w x h pixels (one seamless tile).
    img: a photo material or None for a procedural grain. Use it as colour * (1 + strength * detail).
    highpass: remove stains larger than this fraction of the tile (photos often have big blotches that look
    like camouflage on a wall and repeat visibly when tiled)."""
    if img is not None:
        g = np.asarray(img.convert("L").resize((w, h), Image.LANCZOS), np.float32)
    else:
        rng = np.random.default_rng(fallback_seed)
        g = np.zeros((h, w), np.float32)
        for cells, amp in ((8, 0.12), (32, 0.2), (128, 0.35), (w // 2, 0.4)):
            small = rng.random((max(2, cells * h // w), cells)).astype(np.float32)
            big = np.tile(small, (3, 3))
            up = np.asarray(Image.fromarray((big * 255).astype(np.uint8)).resize((w * 3, h * 3), Image.BICUBIC),
                            np.float32)[h:2 * h, w:2 * w] / 255.0
            g += up * amp
    if highpass:
        from scipy.ndimage import gaussian_filter

        g = g - gaussian_filter(g, (highpass * h, highpass * w), mode="wrap")
    g = g - float(g.mean())
    return g / max(float(g.std()), 1e-3)


def close_up_detail(img, size=1024, contrast=0.8, seed=0):
    """A detail texture for ksMultilayer: the material's light / dark structure around mid grey (the shader
    multiplies it with the ground colour, x2), so the ground keeps its colour and gains the photo's grain."""
    from PIL import Image as _I

    d = detail(img, size, size, fallback_seed=seed, highpass=0.04)
    v = np.clip(0.5 * (1.0 + contrast * 0.22 * d), 0.08, 0.95)
    g = (v * 255).astype(np.uint8)
    return _I.fromarray(np.stack([g, g, g], -1), "RGB")

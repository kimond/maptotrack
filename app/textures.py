"""Procedural textures: road/ground placeholders and the house facades and roofs (house_textures)."""
from __future__ import annotations

import numpy as np
from PIL import Image


def _noise(size, cells, rng):
    """Roughly tileable value noise in [0,1]."""
    small = rng.random((cells, cells))
    big = np.tile(small, (3, 3))
    img = Image.fromarray((big * 255).astype(np.uint8)).resize((size * 3, size * 3), Image.BICUBIC)
    a = np.asarray(img, dtype=float)[size:2 * size, size:2 * size] / 255.0
    return a


def _fbm(size, rng, octaves=((4, 0.45), (16, 0.3), (64, 0.15), (256, 0.1))):
    return sum(_noise(size, c, rng) * w for c, w in octaves)


def _rgb(base, n, amount):
    base = np.asarray(base, float)
    img = base[None, None, :] * (1 - amount / 2 + amount * n[..., None])
    return np.clip(img, 0, 255)


def make_textures(size=512) -> dict:
    rng = np.random.default_rng(7)
    tex = {}

    n = _fbm(size, rng)
    asphalt = _rgb([68, 68, 70], n, 0.35)
    plain = asphalt.copy()
    H, W = size, size
    u = np.arange(W) / W
    v = np.arange(H) / H
    white = np.array([225, 225, 220])
    for lo, hi in ((0.015, 0.035), (0.965, 0.985)):
        cols = (u >= lo) & (u <= hi)
        asphalt[:, cols] = white * (0.9 + 0.1 * n[:, cols, None])
    dash = (v < 0.5)[:, None] & ((u >= 0.494) & (u <= 0.506))[None, :]
    asphalt[dash] = white * 0.9
    tex["asphalt"] = asphalt
    tex["asphalt_plain"] = plain

    n = _fbm(size, rng)
    tex["grass"] = _rgb([78, 112, 52], n, 0.5)
    n = _fbm(size, rng)
    tex["shoulder"] = _rgb([112, 106, 84], n, 0.6)
    n = _fbm(size, rng)
    conc = _rgb([182, 180, 172], n, 0.2)
    conc[int(size * 0.85):] *= 0.75
    tex["concrete"] = conc
    n = _fbm(size, rng)
    tex["roof"] = _rgb([88, 84, 82], n, 0.3)

    # facade: one tile = 4 m wide x 3 m tall, with one window
    n = _fbm(size, rng, ((8, 0.6), (64, 0.4)))
    fac = _rgb([196, 184, 162], n, 0.25)
    x0, x1 = int(size * 0.22), int(size * 0.78)
    y0, y1 = int(size * 0.22), int(size * 0.72)
    fac[y0 - 6:y1 + 6, x0 - 6:x1 + 6] = [120, 112, 100]
    glass = np.linspace(70, 110, y1 - y0)[:, None, None] * np.array([0.75, 0.85, 1.0])[None, None, :]
    fac[y0:y1, x0:x1] = glass
    tex["facade"] = fac

    return {k: Image.fromarray(np.clip(v, 0, 255).astype(np.uint8), "RGB") for k, v in tex.items()}


# ---------------------------------------------------------------- houses
# Facade texture layout (FAC_W x FAC_H px, u wraps horizontally, one bay = BAY_M of wall across):
#   rows A and B: one storey (STOREY_M) with a window centred in the bay; B has shutters
#   row P: plain wall, PLAIN_M tall (gables, walls too short for a window)
#   row F: foundation concrete
#   row D: the ground-floor bay with the front door (door at the bottom of the storey; steps are geometry)
#   row G: the ground-floor bay with a garage door, GARAGE_M tall (it reaches down to the ground)
# Patterns are periodic over the bay (bricks: 15 per bay) and over a storey (16 lap boards, 36 brick courses), so
# bays and storeys join without seams.
BAY_M, STOREY_M, PLAIN_M, FOUND_M = 3.4, 2.8, 5.6, 1.6
GARAGE_M = 3.1                 # storey + the foundation band: a garage floor is at the ground
FAC_W, FAC_H = 512, 4096
ROW_A, ROW_B, ROW_P, ROW_F = (0, 448), (448, 896), (896, 1792), (1792, 2048)
ROW_D, ROW_G = (2048, 2496), (2496, 2992)
# commercial / apartment blocks and house trim:
#   row S: a shop front (glass from 0.5 m to 3.0 m, mullions, sign band), SHOP_M tall
#   row C: a storey of an office / shop block (wide ribbon window), BLOCK_M tall
#   row K: cornice / parapet band, CORNICE_M tall (mouldings, a cap)
#   row T: plain trim (posts, fascias, porch decks), TRIM_M tall
SHOP_M, BLOCK_M, CORNICE_M, TRIM_M = 4.0, 3.2, 0.9, 0.4
ROW_S, ROW_C, ROW_K, ROW_T = (2992, 3440), (3440, 3888), (3888, 4032), (4032, 4096)

FACADES = {   # name: (kind, wall colour, shutter colour, trim colour)
    "siding_white": ("boards", (232, 232, 226), (36, 38, 40), (246, 246, 242)),
    "siding_cream": ("boards", (226, 214, 184), (52, 78, 56), (246, 244, 236)),
    "siding_beige": ("boards", (196, 180, 150), (92, 64, 46), (244, 240, 230)),
    "siding_grey": ("boards", (176, 180, 182), (40, 42, 46), (244, 244, 242)),
    "siding_blue": ("boards", (132, 152, 170), (240, 240, 236), (244, 244, 242)),
    "siding_sage": ("boards", (150, 164, 136), (244, 242, 236), (244, 244, 240)),
    "siding_yellow": ("boards", (226, 204, 138), (54, 74, 58), (246, 244, 236)),
    "siding_charcoal": ("boards", (78, 82, 86), (232, 230, 224), (236, 236, 232)),
    "siding_brick": ("brick", (148, 70, 52), (238, 236, 230), (238, 236, 230)),
    "siding_brick_brown": ("brick", (112, 74, 58), (236, 232, 222), (232, 228, 218)),
    "siding_brick_buff": ("brick", (188, 160, 118), (70, 54, 42), (238, 234, 224)),
    "siding_stucco_white": ("stucco", (228, 224, 214), (82, 58, 42), (246, 244, 238)),
    "siding_stucco_sand": ("stucco", (206, 186, 150), (78, 56, 40), (242, 238, 228)),
}
SHINGLES = {
    "shingle_charcoal": (62, 64, 68), "shingle_brown": (96, 74, 58), "shingle_grey": (122, 122, 120),
    "shingle_red": (122, 62, 48), "shingle_black": (44, 47, 54),
}


def row_v(row, frac0=0.0, frac1=1.0, guard=2):
    """v coordinates (0 = top of the image) of a fraction of a texture row, with a guard band."""
    a, b = row
    a, b = a + guard, b - guard
    return (a + (b - a) * frac0) / FAC_H, (a + (b - a) * frac1) / FAC_H


def _wall_base(kind, rgb, h_px, h_m, grain, seed):
    """Wall material for a block FAC_W x h_px covering BAY_M x h_m metres (float RGB)."""
    W = FAC_W
    rng = np.random.default_rng(seed)
    y = (np.arange(h_px)[:, None] + 0.5) / h_px * h_m          # metres from the top
    x = (np.arange(W)[None, :] + 0.5) / W * BAY_M
    col = np.broadcast_to(np.asarray(rgb, np.float32), (h_px, W, 3)).copy()
    g = np.tile(grain, (int(np.ceil(h_px / grain.shape[0])), int(np.ceil(W / grain.shape[1]))))[:h_px, :W]
    if kind == "boards":                       # vinyl / wood lap siding, 16 boards per storey
        bh = STOREY_M / 16
        f = (y / bh) % 1.0                     # 0 at the top of a board, 1 at its lower lip
        shade = 0.9 + 0.13 * f - 0.28 * np.clip((f - 0.94) / 0.06, 0, 1)
        board = np.floor(y / bh).astype(int)
        tone = 1.0 + 0.025 * (((board * 7919) % 13) / 6.0 - 1.0)
        col *= (shade * tone * (1 + 0.022 * g))[..., None]
    elif kind == "brick":                      # 36 courses per storey, 15 bricks per bay, half-bond
        ch = STOREY_M / 36
        bl = BAY_M / 15
        row = np.floor(y / ch).astype(int)
        fy = (y / ch) % 1.0
        xs = x / bl + 0.5 * (row % 2)
        bid = np.floor(xs).astype(int) % 15
        fx = xs % 1.0
        per = rng.uniform(0.82, 1.14, (64, 16)).astype(np.float32)
        tint = per[row % 64, bid % 16]
        warm = rng.uniform(-0.05, 0.05, (64, 16, 3)).astype(np.float32)[row % 64, bid % 16]
        mortar = (fy > 0.86) | (fx < 0.045)
        brick = col * (tint * (1 + 0.10 * g))[..., None] * (1 + warm)
        mcol = np.array([186, 182, 172], np.float32) * (1 + 0.06 * g)[..., None]
        col = np.where(mortar[..., None], mcol, brick)
        col *= (1 - 0.18 * np.clip((fy - 0.70) / 0.16, 0, 1) * ~mortar)[..., None]   # soft shade under each brick
    else:                                      # stucco
        col *= (1 + 0.035 * g)[..., None]
    return col


def _window_layer(img, top_m, rows_px, trim, shutters, shutter_rgb, S):
    """Draw the window (supersampled S times) into img (PIL RGB, FAC_W*S wide, one storey tall)."""
    from PIL import ImageDraw, ImageFilter

    d = ImageDraw.Draw(img)
    pxm = rows_px / STOREY_M * S                # px per metre (vertical; horizontal is FAC_W / BAY_M)
    pxx = FAC_W / BAY_M * S
    ww, wh, head = 1.06, 1.36, 0.52            # opening, top of the opening below the storey top
    cx = FAC_W * S / 2
    x0, x1 = cx - ww / 2 * pxx, cx + ww / 2 * pxx
    y0, y1 = head * pxm, (head + wh) * pxm
    t = 0.09 * pxx
    # soft shadow around the frame
    sh = Image.new("L", img.size, 0)
    ds = ImageDraw.Draw(sh)
    ds.rectangle([x0 - t * 1.2, y0 - t * 0.6, x1 + t * 1.6, y1 + t * 2.4], fill=90)
    sh = sh.filter(ImageFilter.GaussianBlur(t * 0.8))
    dark = Image.new("RGB", img.size, (0, 0, 0))
    img.paste(Image.composite(dark, img, sh), (0, 0))
    d = ImageDraw.Draw(img)
    if shutters:
        sw = 0.44 * pxx
        for sx0 in (x0 - t - 0.05 * pxx - sw, x1 + t + 0.05 * pxx):
            d.rectangle([sx0, y0 - t * 0.3, sx0 + sw, y1 + t * 0.3], fill=shutter_rgb)
            dk = tuple(int(c * 0.72) for c in shutter_rgb)
            lt = tuple(min(255, int(c * 1.12 + 10)) for c in shutter_rgb)
            yy = y0 + 0.12 * pxm
            while yy < y1 - 0.08 * pxm:
                d.line([(sx0 + 0.06 * pxx, yy), (sx0 + sw - 0.06 * pxx, yy)], fill=dk, width=max(1, int(0.012 * pxm)))
                d.line([(sx0 + 0.06 * pxx, yy + 0.014 * pxm), (sx0 + sw - 0.06 * pxx, yy + 0.014 * pxm)], fill=lt,
                       width=max(1, int(0.006 * pxm)))
                yy += 0.05 * pxm
            d.rectangle([sx0, y0 - t * 0.3, sx0 + sw, y1 + t * 0.3], outline=dk, width=max(1, int(0.03 * pxx)))
    # trim (casing) with a bevel
    d.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], fill=trim)
    d.line([(x0 - t, y1 + t), (x1 + t, y1 + t), (x1 + t, y0 - t)], fill=tuple(int(c * 0.8) for c in trim),
           width=max(1, int(t * 0.18)))
    # glass: sky reflection fading to a dark interior, a diagonal glint
    gh = int(y1 - y0)
    for i in range(gh):
        k = i / max(gh - 1, 1)
        c = (int(150 - 112 * k), int(168 - 116 * k), int(186 - 116 * k))
        d.line([(x0, y0 + i), (x1, y0 + i)], fill=c)
    glint = Image.new("L", img.size, 0)
    dg = ImageDraw.Draw(glint)
    gw = (x1 - x0) * 0.22
    dg.polygon([(x0 + (x1 - x0) * 0.15, y0), (x0 + (x1 - x0) * 0.15 + gw, y0), (x0 + gw * 0.3, y1), (x0, y1),
                (x0, y0 + (y1 - y0) * 0.75)], fill=60)
    glint = glint.filter(ImageFilter.GaussianBlur(t * 0.3))
    img.paste(Image.composite(Image.new("RGB", img.size, (235, 240, 245)), img, glint), (0, 0))
    d = ImageDraw.Draw(img)
    # sash frame, centre mullion, one rail (four panes)
    st = 0.045 * pxx
    d.rectangle([x0, y0, x1, y1], outline=trim, width=max(1, int(st)))
    d.line([(cx, y0), (cx, y1)], fill=trim, width=max(1, int(st)))
    ry = y0 + (y1 - y0) * 0.42
    d.line([(x0, ry), (x1, ry)], fill=trim, width=max(1, int(st)))
    # sill with its shadow
    sx0, sx1 = x0 - t * 1.6, x1 + t * 1.6
    sy0 = y1 + t
    sh2 = Image.new("L", img.size, 0)
    ImageDraw.Draw(sh2).rectangle([sx0, sy0, sx1, sy0 + 0.1 * pxm], fill=120)
    sh2 = sh2.filter(ImageFilter.GaussianBlur(0.03 * pxm))
    img.paste(Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, sh2), (0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([sx0, sy0 - 0.01 * pxm, sx1, sy0 + 0.055 * pxm], fill=tuple(int(c * 0.93) for c in trim))
    d.line([(sx0, sy0 + 0.055 * pxm), (sx1, sy0 + 0.055 * pxm)], fill=tuple(int(c * 0.7) for c in trim),
           width=max(1, int(0.01 * pxm)))
    return img


def _door_layer(img, rows_px, trim, door_rgb, S, height_m=STOREY_M):
    """A front door (0.95 x 2.05 m, panelled, three small lights, handle) standing on the bottom of the bay,
    centred, with its casing (img: PIL RGB, FAC_W*S wide, one row tall = height_m)."""
    from PIL import ImageDraw, ImageFilter

    pxm = rows_px / height_m * S
    pxx = FAC_W / BAY_M * S
    H = rows_px * S
    cx = FAC_W * S / 2
    dw, dh = 0.95, 2.05
    x0, x1 = cx - dw / 2 * pxx, cx + dw / 2 * pxx
    y1 = H - 1
    y0 = H - dh * pxm
    t = 0.10 * pxx
    sh = Image.new("L", img.size, 0)
    ImageDraw.Draw(sh).rectangle([x0 - t * 1.3, y0 - t * 1.0, x1 + t * 1.8, y1], fill=80)
    sh = sh.filter(ImageFilter.GaussianBlur(t * 0.7))
    img.paste(Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, sh), (0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([x0 - t, y0 - t, x1 + t, y1], fill=trim)                          # casing
    d.rectangle([x0 - t, y0 - t * 1.25, x1 + t, y0 - t * 0.7], fill=tuple(int(c * 0.85) for c in trim))   # head
    d.rectangle([x0, y0, x1, y1], fill=door_rgb)
    dk = tuple(int(c * 0.72) for c in door_rgb)
    lt = tuple(min(255, int(c * 1.18 + 12)) for c in door_rgb)
    pw = (x1 - x0)
    # two raised panels below, three small lights above
    for (a, b) in ((0.55, 0.95),):
        for px0, px1 in ((0.12, 0.46), (0.54, 0.88)):
            r = [x0 + pw * px0, y0 + (y1 - y0) * a, x0 + pw * px1, y0 + (y1 - y0) * b]
            d.rectangle(r, outline=dk, width=max(1, int(0.02 * pxx)))
            d.line([(r[0] + 2, r[1] + 2), (r[2] - 2, r[1] + 2)], fill=lt, width=max(1, int(0.01 * pxx)))
    for k in range(3):
        gx0 = x0 + pw * (0.14 + k * 0.25)
        r = [gx0, y0 + (y1 - y0) * 0.12, gx0 + pw * 0.2, y0 + (y1 - y0) * 0.36]
        d.rectangle(r, fill=(96, 112, 128), outline=dk, width=max(1, int(0.015 * pxx)))
    hx, hy = x0 + pw * 0.86, y0 + (y1 - y0) * 0.52
    d.ellipse([hx - 0.025 * pxx, hy - 0.025 * pxm, hx + 0.025 * pxx, hy + 0.025 * pxm], fill=(196, 180, 120))
    return img


def _garage_layer(img, rows_px, trim, S, height_m=GARAGE_M, seed=0):
    """A sectional garage door (2.6 x 2.15 m, four panel rows, white or almond) at the bottom of the bay."""
    from PIL import ImageDraw, ImageFilter

    pxm = rows_px / height_m * S
    pxx = FAC_W / BAY_M * S
    H = rows_px * S
    cx = FAC_W * S / 2
    gw, gh = 2.6, 2.15
    x0, x1 = cx - gw / 2 * pxx, cx + gw / 2 * pxx
    y1 = H - 1
    y0 = H - gh * pxm
    t = 0.09 * pxx
    sh = Image.new("L", img.size, 0)
    ImageDraw.Draw(sh).rectangle([x0 - t, y0 - t, x1 + t * 1.5, y1], fill=90)
    sh = sh.filter(ImageFilter.GaussianBlur(t * 0.6))
    img.paste(Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, sh), (0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([x0 - t, y0 - t, x1 + t, y1], fill=trim)
    door = (236, 234, 228) if seed % 3 else (214, 202, 178)
    d.rectangle([x0, y0, x1, y1], fill=door)
    dk = tuple(int(c * 0.78) for c in door)
    lt = tuple(min(255, int(c * 1.06 + 6)) for c in door)
    for k in range(1, 4):
        yy = y0 + (y1 - y0) * k / 4
        d.line([(x0, yy), (x1, yy)], fill=dk, width=max(1, int(0.02 * pxm)))
        d.line([(x0, yy + 0.015 * pxm), (x1, yy + 0.015 * pxm)], fill=lt, width=max(1, int(0.01 * pxm)))
    for k in range(4):                                      # raised panels: 8 per row
        ya, yb = y0 + (y1 - y0) * (k + 0.18) / 4, y0 + (y1 - y0) * (k + 0.82) / 4
        for j in range(8):
            xa, xb = x0 + (x1 - x0) * (j + 0.12) / 8, x0 + (x1 - x0) * (j + 0.88) / 8
            d.rectangle([xa, ya, xb, yb], outline=dk, width=max(1, int(0.012 * pxx)))
    return img


def _glass(d, x0, y0, x1, y1, dark=1.0):
    gh = int(y1 - y0)
    for i in range(max(gh, 1)):
        k = i / max(gh - 1, 1)
        c = (int((132 - 96 * k) * dark), int((150 - 102 * k) * dark), int((168 - 104 * k) * dark))
        d.line([(x0, y0 + i), (x1, y0 + i)], fill=c)


def _storefront_layer(img, rows_px, trim, S, height_m=SHOP_M):
    """A shop window bay: kick plate up to 0.5 m, glass to 3.0 m in three panes with dark metal mullions, a
    sign band above (wall colour with a fascia line)."""
    from PIL import ImageDraw, ImageFilter

    pxm = rows_px / height_m * S
    pxx = FAC_W / BAY_M * S
    H = rows_px * S
    x0, x1 = 0.08 * pxx, FAC_W * S - 0.08 * pxx
    y1 = H - 0.5 * pxm
    y0 = H - 3.0 * pxm
    sh = Image.new("L", img.size, 0)
    ImageDraw.Draw(sh).rectangle([x0, y0 - 0.12 * pxm, x1, y1 + 0.05 * pxm], fill=70)
    sh = sh.filter(ImageFilter.GaussianBlur(0.05 * pxm))
    img.paste(Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, sh), (0, 0))
    d = ImageDraw.Draw(img)
    metal = (58, 60, 64)
    _glass(d, x0, y0, x1, y1, 0.92)
    mw = max(1, int(0.06 * pxx))
    d.rectangle([x0, y0, x1, y1], outline=metal, width=mw)
    for j in (1, 2):
        xx = x0 + (x1 - x0) * j / 3
        d.line([(xx, y0), (xx, y1)], fill=metal, width=mw)
    d.line([(x0, y0 + 0.55 * pxm), (x1, y0 + 0.55 * pxm)], fill=metal, width=max(1, mw // 2))   # transom
    d.rectangle([x0, y1, x1, H], fill=tuple(int(c * 0.55) for c in trim))                       # kick plate
    fy = H - 3.25 * pxm                                                                       # sign band fascia
    d.rectangle([0, fy - 0.06 * pxm, FAC_W * S, fy], fill=tuple(int(c * 0.8) for c in trim))
    return img


def _ribbon_layer(img, rows_px, trim, S, height_m=BLOCK_M):
    """An office / shop storey: one wide window per bay (0.9 m to 2.5 m above the floor), dark frames."""
    from PIL import ImageDraw

    pxm = rows_px / height_m * S
    pxx = FAC_W / BAY_M * S
    H = rows_px * S
    x0, x1 = 0.22 * pxx, FAC_W * S - 0.22 * pxx
    y1, y0 = H - 0.9 * pxm, H - 2.5 * pxm
    d = ImageDraw.Draw(img)
    metal = (62, 64, 68)
    d.rectangle([x0 - 0.05 * pxx, y0 - 0.05 * pxm, x1 + 0.05 * pxx, y1 + 0.08 * pxm],
                fill=tuple(int(c * 0.6) for c in trim))
    _glass(d, x0, y0, x1, y1)
    mw = max(1, int(0.04 * pxx))
    d.rectangle([x0, y0, x1, y1], outline=metal, width=mw)
    d.line([((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)], fill=metal, width=mw)
    d.rectangle([x0 - 0.08 * pxx, y1, x1 + 0.08 * pxx, y1 + 0.06 * pxm], fill=tuple(int(c * 0.9) for c in trim))
    return img


def _cornice(trim, h_px, grain):
    """Cornice band: a cap, a frieze, a row of dentils and mouldings with their shadows (float RGB)."""
    W = FAC_W
    y = (np.arange(h_px)[:, None] + 0.5) / h_px * CORNICE_M
    x = (np.arange(W)[None, :] + 0.5) / W * BAY_M
    col = np.broadcast_to(np.asarray(trim, np.float32) * 0.92, (h_px, W, 3)).copy()
    g = np.tile(grain, (int(np.ceil(h_px / grain.shape[0])), int(np.ceil(W / grain.shape[1]))))[:h_px, :W]
    shade = np.ones((h_px, W), np.float32)
    shade[(y < 0.08).ravel(), :] = 1.05                                    # cap
    shade[((y >= 0.08) & (y < 0.12)).ravel(), :] = 0.62                     # shadow under the cap
    dent = (y >= 0.2) & (y < 0.32) & (((x / 0.17) % 1.0) < 0.55)            # dentils
    gapd = (y >= 0.2) & (y < 0.32) & ~dent
    shade = np.where(dent, 1.0, shade)
    shade = np.where(gapd, 0.7, shade)
    shade[((y >= 0.32) & (y < 0.36)).ravel(), :] = 0.75
    shade[((y >= 0.8) & (y < 0.84)).ravel(), :] = 0.8                       # bed moulding
    shade[(y >= 0.86).ravel(), :] = 0.7                                     # shadow on the wall below
    return col * (shade * (1 + 0.03 * g))[..., None]


def facade_texture(name, photos=None, seed=0):
    from . import materials

    kind, rgb, shutter, trim = FACADES[name]
    photos = photos or {}
    src = photos.get("concrete")          # fine concrete grain (the plaster photo repeats visibly as a pattern)
    # stucco: a smaller, finer tile (rough surface); siding / brick: 0.85 m tile
    gw, gh = (64, 56) if kind == "stucco" else (128, 112)
    grain = materials.detail(src, gw, gh, fallback_seed=seed + 1, highpass=0.15 if kind == "stucco" else 0.08)
    out = np.zeros((FAC_H, FAC_W, 3), np.float32)
    S = 2
    for row, shut in ((ROW_A, False), (ROW_B, True)):
        h = row[1] - row[0]
        base = _wall_base(kind, rgb, h, STOREY_M, grain, seed)
        big = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((FAC_W * S, h * S), Image.BICUBIC)
        big = _window_layer(big, 0.0, h, trim, shut, shutter, S)
        out[row[0]:row[1]] = np.asarray(big.resize((FAC_W, h), Image.LANCZOS), np.float32)
    door_rgb = shutter if kind != "brick" else (66, 46, 34)
    for row, hm, layer in ((ROW_D, STOREY_M, "door"), (ROW_G, GARAGE_M, "garage")):
        h = row[1] - row[0]
        base = _wall_base(kind, rgb, h, hm, grain, seed)
        big = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((FAC_W * S, h * S), Image.BICUBIC)
        big = (_door_layer(big, h, trim, door_rgb, S, hm) if layer == "door"
               else _garage_layer(big, h, trim, S, hm, seed))
        out[row[0]:row[1]] = np.asarray(big.resize((FAC_W, h), Image.LANCZOS), np.float32)
    for row, hm, layer in ((ROW_S, SHOP_M, _storefront_layer), (ROW_C, BLOCK_M, _ribbon_layer)):
        h = row[1] - row[0]
        base = _wall_base(kind, rgb, h, hm, grain, seed)
        big = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).resize((FAC_W * S, h * S), Image.BICUBIC)
        big = layer(big, h, trim, S, hm)
        out[row[0]:row[1]] = np.asarray(big.resize((FAC_W, h), Image.LANCZOS), np.float32)
    out[ROW_K[0]:ROW_K[1]] = _cornice(trim, ROW_K[1] - ROW_K[0], grain)
    hT = ROW_T[1] - ROW_T[0]
    out[ROW_T[0]:ROW_T[1]] = np.asarray(trim, np.float32) * (0.97 + 0.03 * np.tile(
        grain, (int(np.ceil(hT / grain.shape[0])), int(np.ceil(FAC_W / grain.shape[1]))))[:hT, :FAC_W])[..., None]
    out[ROW_P[0]:ROW_P[1]] = _wall_base(kind, rgb, ROW_P[1] - ROW_P[0], PLAIN_M, grain, seed)
    cg = materials.detail(photos.get("concrete"), 256, 128, fallback_seed=seed + 2, highpass=0.08)
    hF = ROW_F[1] - ROW_F[0]
    conc = np.broadcast_to(np.array([148, 146, 140], np.float32), (hF, FAC_W, 3)).copy()
    cgt = np.tile(cg, (int(np.ceil(hF / cg.shape[0])), int(np.ceil(FAC_W / cg.shape[1]))))[:hF, :FAC_W]
    conc *= (1 + 0.12 * cgt)[..., None]
    conc[:6] *= 0.8                                              # the sill plate's shadow line
    out[ROW_F[0]:ROW_F[1]] = conc
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def shingle_texture(base, photos=None, seed=0, size=512):
    """Three-tab asphalt shingles, one tile = 3 x 3 m, granules from the asphalt photo."""
    from . import materials

    rng = np.random.default_rng(seed)
    yy = (np.arange(size)[:, None] + 0.5) / size * 3.0
    xx = (np.arange(size)[None, :] + 0.5) / size * 3.0
    g = materials.detail((photos or {}).get("asphalt"), 128, 128, fallback_seed=seed + 3, highpass=0.1)
    g = np.tile(g, (size // 128, size // 128))
    exp = 3.0 / 21                                             # ~14 cm exposure, 21 courses per tile
    row = np.floor(yy / exp).astype(int)
    fy = (yy / exp) % 1
    tabw = 3.0 / 9
    xs = xx / tabw + (row % 3) / 3.0
    tab = np.floor(xs).astype(int)
    fx = xs % 1
    tone = rng.uniform(0.86, 1.12, (32, 16))[row % 32, tab % 16]
    img = np.asarray(base, np.float32)[None, None, :] * (tone * (1 + 0.16 * g))[..., None]
    gap = fx < 0.025
    lip = fy > 0.9
    img *= np.where(gap | lip, 0.55, 1.0)[..., None]
    img *= (0.94 + 0.08 * fy)[..., None]                       # slightly lighter towards the lower edge
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def house_textures(photos=None) -> dict:
    """All facade and roof textures (photos: materials.fetch(), or None for procedural grain)."""
    from . import materials
    from .parallel import pmap

    jobs = [(name, None, 11 + i) for i, name in enumerate(FACADES)] + \
           [(name, base, 31 + i) for i, (name, base) in enumerate(SHINGLES.items())]
    imgs = pmap(lambda j: facade_texture(j[0], photos, seed=j[2]) if j[1] is None
                else shingle_texture(j[1], photos, seed=j[2]), jobs)
    out = {j[0]: im for j, im in zip(jobs, imgs)}

    g = materials.detail((photos or {}).get("concrete"), 256, 256, fallback_seed=5)
    flat = np.array([112, 110, 106], np.float32)[None, None, :] * (1 + 0.1 * np.tile(g, (2, 2)))[..., None]
    out["roof_flat"] = Image.fromarray(np.clip(flat, 0, 255).astype(np.uint8), "RGB")
    return out

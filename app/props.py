"""Street furniture and power lines (the way Map2Lap / BeamNG.Earth does it, from OpenStreetMap):

* power lines: OSM power=line / minor_line ways (underground ones left out). A tower at each mapped tower
  node (or at each vertex of the line), steel lattice pylons for transmission lines, wooden poles for
  low-voltage lines, conductors hung between them with a sag.
* utility poles (the Americas, where most low-voltage lines are overhead): wooden poles along the town streets
  every POLE_SPACING m on one side, with their wires, a cobra-head street light on every other one. Skipped
  where OSM maps the low-voltage lines itself.
* street lights: OSM highway=street_lamp nodes; in Europe also posts along lit / built-up streets.
* traffic lights at junctions with a mapped highway=traffic_signals node (or mid-block signals); one per way in,
  at the kerb on the drivers' side, facing the traffic coming in (North America: on a mast arm over the lane too).
* stop / give-way signs: mapped highway=stop / give_way nodes, otherwise the usual rules: stop on the minor
  roads meeting a major road, on the stem of a minor T, all-way stops on half of the minor 4-way junctions
  (always the same ones), give way when entering a roundabout.
* speed-limit signs where the OSM maxspeed changes, facing the traffic entering the new limit.

Posts and poles collide (1WALL_PROPS); signs, heads, arms, wires are visual only. Everything keeps clear of the
track (and its barriers), the streets' carriageways and the buildings. One RGBA atlas ('props', alpha-tested).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .meshes import MeshBuilder

MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary",
         "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}
MINOR = {"residential", "unclassified", "living_street"}
CTRL = MAJOR | MINOR
POLE_CLASSES = {"secondary", "tertiary", "residential", "unclassified"}
LAMP_CLASSES = {"primary", "secondary", "tertiary", "residential", "unclassified", "living_street"}
POLE_SPACING = 40.0          # m, utility poles along a street (North American practice: 35-50 m)
EU_LAMP_SPACING = 30.0       # m, lamp posts along a European street (Map2Lap's measurement: 25-30 m)
SIGNAL_REACH = 15.0          # m beyond the widest road's edge a mapped signal still belongs to a junction
STOP_REACH = 30.0
KERB = 1.5                    # m from the carriageway edge to a post

# ---------------------------------------------------------------- atlas
COLS, ROWS, CELL = 8, 4, 128
(METAL, GALV, WOOD, DARK, SIG_BODY, SIG_RED, SIG_GREEN, LAMP_LIT,
 STOP, STOP_BACK, YIELD, YIELD_BACK, RECT_BACK, CIRCLE_BACK, INSULATOR, CONCRETE) = range(16)
SPEED0 = 16                  # 12 speed-limit cells
N_SPEED = 12
CHAIN, TRANSFORMER, RAIL_W, BUSHING = 28, 29, 30, 31   # chain-link fence (cut-out), transformer tank, W-beam, porcelain


def sign_style(lat, lon, choice="auto"):
    """'qc' (Québec), 'ca' (rest of Canada), 'us', 'eu'. Auto: rough borders, good enough for sign faces."""
    if choice and choice != "auto":
        return choice
    if lon > -30:
        return "eu"
    if _in_quebec(lat, lon):
        return "qc"
    if lat >= 49.0 or (lon > -95 and lat >= 46.0) or (-83.5 < lon < -74.3 and lat >= 43.6 and not (
            lon > -79.0 and lat < 44.95)) or (lon > -67.8 and lat >= 44.5):
        return "ca"
    return "us"


_QC = [(-79.52, 62.6), (-64.5, 60.3), (-63.5, 52.0), (-57.1, 51.4), (-64.2, 48.5), (-66.4, 48.0), (-67.8, 47.9),
       (-68.3, 47.35), (-69.2, 47.45), (-70.0, 46.7), (-71.5, 45.0), (-74.5, 45.0), (-74.4, 45.57), (-75.0, 45.6),
       (-75.7, 45.47), (-76.7, 45.6), (-77.3, 45.9), (-78.7, 46.33), (-79.45, 46.6)]


def _in_quebec(lat, lon):
    inside = False
    n = len(_QC)
    for i in range(n):
        x1, y1 = _QC[i]
        x2, y2 = _QC[(i + 1) % n]
        if (y1 > lat) != (y2 > lat) and lon < x1 + (lat - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _font(size):
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fit_text(d, xy, text, max_w, size, fill):
    f = _font(size)
    while size > 8 and d.textlength(text, font=f) > max_w:
        size -= 2
        f = _font(size)
    d.text(xy, text, fill=fill, font=f, anchor="mm")


def atlas(style="us", speeds=()) -> Image.Image:
    """RGBA atlas, COLS x ROWS cells. Sign cells are cut out by alpha (octagon, triangle, circle)."""
    S = 2
    C = CELL * S
    img = Image.new("RGBA", (COLS * C, ROWS * C), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def box(k):
        return (k % COLS) * C, (k // COLS) * C

    def flat(k, rgb, grain=0):
        x, y = box(k)
        d.rectangle([x, y, x + C - 1, y + C - 1], fill=rgb + (255,))
        if grain:
            rng = np.random.default_rng(k)
            for _ in range(40):
                yy = y + int(rng.integers(0, C))
                c = tuple(int(np.clip(v + rng.integers(-grain, grain), 0, 255)) for v in rgb) + (255,)
                d.line([(x, yy), (x + C - 1, yy + int(rng.integers(-3, 3)))], fill=c, width=S * 2)

    flat(METAL, (120, 124, 128), 10)
    flat(GALV, (168, 172, 174), 8)
    flat(WOOD, (98, 76, 56), 16)
    flat(DARK, (34, 34, 36))
    flat(SIG_BODY, (214, 176, 38) if style in ("us", "ca", "qc") else (38, 40, 42))
    flat(INSULATOR, (120, 98, 80), 6)
    flat(CONCRETE, (150, 148, 142), 8)
    flat(LAMP_LIT, (255, 236, 190))
    for k, lit in ((SIG_RED, 0), (SIG_GREEN, 2)):            # signal faces: three lenses, one lit
        x, y = box(k)
        body = (214, 176, 38) if style in ("us", "ca", "qc") else (38, 40, 42)
        d.rectangle([x, y, x + C - 1, y + C - 1], fill=body + (255,))
        d.rectangle([x + C * 0.12, y, x + C * 0.88, y + C - 1], fill=(26, 26, 28, 255))
        for j, (on, off) in enumerate((((255, 40, 30), (70, 18, 16)), ((255, 170, 30), (70, 50, 14)),
                                       ((60, 255, 120), (14, 60, 30)))):
            cy = y + C * (0.18 + 0.32 * j)
            rx, ry = C * 0.3, C * 0.1           # the face is ~3x taller than wide: round lenses once mapped
            d.ellipse([x + C / 2 - rx, cy - ry, x + C / 2 + rx, cy + ry], fill=(on if j == lit else off) + (255,))
    flat(RECT_BACK, (150, 154, 158))
    # stop sign: red octagon, white border, ARRÊT (Québec) / STOP
    for k, back in ((STOP, False), (STOP_BACK, True)):
        x, y = box(k)
        cx, cy = x + C / 2, y + C / 2
        octo = [(cx + C * 0.49 * math.cos(math.pi / 8 + i * math.pi / 4),
                 cy + C * 0.49 * math.sin(math.pi / 8 + i * math.pi / 4)) for i in range(8)]
        d.polygon(octo, fill=(150, 154, 158, 255) if back else (245, 245, 240, 255))
        if not back:
            d.polygon([(cx + 0.9 * (px - cx), cy + 0.9 * (py - cy)) for px, py in octo], fill=(196, 28, 28, 255))
            _fit_text(d, (cx, cy), "ARRÊT" if style == "qc" else "STOP", C * 0.78, int(C * 0.27), (250, 250, 250))
    # give way / yield: inverted triangle, red border, white inside
    for k, back in ((YIELD, False), (YIELD_BACK, True)):
        x, y = box(k)
        tri = [(x + C * 0.02, y + C * 0.08), (x + C * 0.98, y + C * 0.08), (x + C / 2, y + C * 0.92)]
        d.polygon(tri, fill=(150, 154, 158, 255) if back else (196, 28, 28, 255))
        if not back:
            cx, cy = x + C / 2, y + C * 0.36
            d.polygon([(cx + 0.66 * (px - cx), cy + 0.66 * (py - cy)) for px, py in tri], fill=(245, 245, 240, 255))
            if style == "us":
                _fit_text(d, (x + C / 2, y + C * 0.3), "YIELD", C * 0.4, int(C * 0.12), (196, 28, 28))
    x, y = box(CIRCLE_BACK)
    d.ellipse([x + C * 0.02, y + C * 0.02, x + C * 0.98, y + C * 0.98], fill=(150, 154, 158, 255))
    # speed limits
    for i, v in enumerate(list(speeds)[:N_SPEED]):
        x, y = box(SPEED0 + i)
        cx = x + C / 2
        if style == "eu":
            d.ellipse([x + C * 0.02, y + C * 0.02, x + C * 0.98, y + C * 0.98], fill=(200, 30, 30, 255))
            d.ellipse([x + C * 0.13, y + C * 0.13, x + C * 0.87, y + C * 0.87], fill=(245, 245, 240, 255))
            _fit_text(d, (cx, y + C / 2), str(v), C * 0.62, int(C * 0.42), (20, 20, 20))
            continue
        # rectangular signs (the face is 0.8 wide x 1.0 high: drawn in the middle 80% of the cell)
        x0, x1 = x + C * 0.1, x + C * 0.9
        d.rectangle([x0, y, x1, y + C - 1], fill=(20, 20, 20, 255))
        d.rectangle([x0 + C * 0.03, y + C * 0.03, x1 - C * 0.03, y + C * 0.97], fill=(245, 245, 240, 255))
        if style == "qc":                     # number in a red ring, km/h below
            r = C * 0.3
            cy = y + C * 0.42
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 30, 30, 255))
            d.ellipse([cx - r * 0.8, cy - r * 0.8, cx + r * 0.8, cy + r * 0.8], fill=(245, 245, 240, 255))
            _fit_text(d, (cx, cy), str(v), r * 1.3, int(C * 0.24), (20, 20, 20))
            _fit_text(d, (cx, y + C * 0.83), "km/h", C * 0.5, int(C * 0.12), (20, 20, 20))
        elif style == "ca":
            _fit_text(d, (cx, y + C * 0.18), "MAXIMUM", C * 0.68, int(C * 0.12), (20, 20, 20))
            _fit_text(d, (cx, y + C * 0.6), str(v), C * 0.66, int(C * 0.42), (20, 20, 20))
        else:
            _fit_text(d, (cx, y + C * 0.15), "SPEED", C * 0.66, int(C * 0.13), (20, 20, 20))
            _fit_text(d, (cx, y + C * 0.3), "LIMIT", C * 0.66, int(C * 0.13), (20, 20, 20))
            _fit_text(d, (cx, y + C * 0.66), str(v), C * 0.66, int(C * 0.4), (20, 20, 20))
    flat(TRANSFORMER, (104, 114, 106), 6)
    flat(BUSHING, (132, 92, 66), 4)
    x, y = box(RAIL_W)                         # W-beam: galvanised, two ridges catching the light
    for j in range(C):
        t = j / C
        g = 150 + 50 * math.sin(t * 4 * math.pi) ** 2 - 25 * (abs(t - 0.5) < 0.05)
        d.line([(x, y + j), (x + C - 1, y + j)], fill=(int(g), int(g) + 3, int(g) + 6, 255))
    img = img.resize((COLS * CELL, ROWS * CELL), Image.LANCZOS)
    arr = np.asarray(img).copy()
    a = arr[..., 3]
    arr[..., 3] = np.where(a > 110, 255, 0)
    arr[a <= 110, :3] = [150, 154, 158]
    # chain-link fence: a diamond mesh of galvanised wire (drawn at the final size so the wires survive),
    # a top rail along the cell's upper edge; the holes are cut out
    cx, cy = (CHAIN % COLS) * CELL, (CHAIN // COLS) * CELL
    jj, ii = np.mgrid[0:CELL, 0:CELL]
    pitch = 16
    wire_ = ((ii + jj) % pitch < 2) | ((ii - jj) % pitch < 2) | (jj < 5) | (jj >= CELL - 3)
    cell = arr[cy:cy + CELL, cx:cx + CELL]
    cell[...] = [150, 154, 158, 0]
    cell[wire_] = [170, 174, 176, 255]
    return Image.fromarray(arr, "RGBA")


def cell_uv(k, inset=0.0):
    u0, v0 = (k % COLS) / COLS, (k // COLS) / ROWS
    du, dv = 1.0 / COLS, 1.0 / ROWS
    return u0 + du * inset, v0 + dv * inset, u0 + du * (1 - inset), v0 + dv * (1 - inset)


# ---------------------------------------------------------------- geometry
class Geo:
    """Quads with their own vertices (flat shading) in world coordinates, atlas UVs."""

    def __init__(self):
        self.v, self.f, self.uv = [], [], []
        self.n = 0
        self.lens = None            # lamp lenses go to their own mesh (material props_lit: lit at night by CSP)
        self.lights = []            # (x, y, z) under every lamp lens

    def quad(self, p0, p1, p2, p3, cell, out=None, uv=None, double=False):
        P = np.array([p0, p1, p2, p3], float)
        if uv is None:
            u0, v0, u1, v1 = cell_uv(cell, 0.25)
            uv = [(u0, v1), (u1, v1), (u1, v0), (u0, v0)]
        UV = np.array(uv, float)
        if out is not None:
            nrm = np.cross(P[1] - P[0], P[3] - P[0])
            if np.dot(nrm, out) < 0:
                P, UV = P[[0, 3, 2, 1]], UV[[0, 3, 2, 1]]
        self._add(P, UV)
        if double:
            self._add(P[[0, 3, 2, 1]], UV[[0, 3, 2, 1]])

    def _add(self, P, UV):
        self.v.append(P)
        self.uv.append(UV)
        self.f.append(np.array([[0, 1, 2], [0, 2, 3]]) + self.n)
        self.n += 4

    def beam(self, a, b, r, cell, sides=4):
        """Prism of radius r between points a and b."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-6:
            return
        d /= L
        ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(d, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        ang = np.arange(sides) * 2 * np.pi / sides + np.pi / sides
        ring = [np.cos(t) * e1 + np.sin(t) * e2 for t in ang]
        u0, v0, u1, v1 = cell_uv(cell, 0.25)
        for i in range(sides):
            o1, o2 = ring[i] * r, ring[(i + 1) % sides] * r
            out = ring[i] + ring[(i + 1) % sides]
            self.quad(a + o1, a + o2, b + o2, b + o1, cell, out=out,
                      uv=[(u0, v1), (u1, v1), (u1, v0), (u0, v0)])

    def box(self, c, along, half_a, half_b, z0, z1, cell, front_cell=None, front=None):
        """Box around centre c (x, y), axes along (unit 2D) and its normal, from z0 to z1. front_cell textures the
        face whose outward normal is `front` (2D)."""
        a = np.array([along[0], along[1], 0.0])
        b = np.array([-along[1], along[0], 0.0])
        c = np.array([c[0], c[1], 0.0])
        z = np.array([0, 0, 1.0])
        for nrm, ha, hb in ((a, half_a, half_b), (-a, half_a, half_b), (b, half_b, half_a), (-b, half_b, half_a)):
            t = np.cross(z, nrm)
            base = c + nrm * ha
            p = [base - t * hb + z * z0, base + t * hb + z * z0, base + t * hb + z * z1, base - t * hb + z * z1]
            k = cell
            if front_cell is not None and front is not None and np.dot(nrm[:2], front) > 0.9:
                k = front_cell
                u0, v0, u1, v1 = cell_uv(k)
                self.quad(*p, k, out=nrm, uv=[(u0, v1), (u1, v1), (u1, v0), (u0, v0)])
            else:
                self.quad(*p, k, out=nrm)
        p = [c + a * half_a + b * half_b, c - a * half_a + b * half_b, c - a * half_a - b * half_b,
             c + a * half_a - b * half_b]
        self.quad(*[q + z * z1 for q in p], cell, out=z)

    def face(self, c, z_mid, facing, w, h, cell, back_cell, uv_box=None):
        """A sign panel at c facing `facing` (2D unit), width w, height h, with its back."""
        f = np.array([facing[0], facing[1], 0.0])
        r = np.array([-facing[1], facing[0], 0.0])        # the viewer's right
        c = np.array([c[0], c[1], z_mid]) + f * 0.03
        z = np.array([0, 0, 1.0])
        p = [c - r * w / 2 - z * h / 2, c + r * w / 2 - z * h / 2, c + r * w / 2 + z * h / 2, c - r * w / 2 + z * h / 2]
        u0, v0, u1, v1 = uv_box if uv_box is not None else cell_uv(cell)
        self.quad(*p, cell, out=f, uv=[(u0, v1), (u1, v1), (u1, v0), (u0, v0)])
        b0, c0, b1, c1 = cell_uv(back_cell)
        q = [x - f * 0.02 for x in p]
        self.quad(q[1], q[0], q[3], q[2], back_cell, out=-f, uv=[(b0, c1), (b1, c1), (b1, c0), (b0, c0)])

    def mesh(self):
        if not self.n:
            return None
        return np.vstack(self.v), np.vstack(self.f), np.vstack(self.uv)


# ---------------------------------------------------------------- models
def _yaw_vec(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def post(G, x, y, z, h, r=0.045, cell=GALV):
    G.beam((x, y, z - 0.3), (x, y, z + h), r, cell)


def stop_sign(col, vis, x, y, z, facing, kind):
    post(col, x, y, z, 2.6)
    if kind == "stop":
        vis.face((x, y), z + 2.15, facing, 0.75, 0.75, STOP, STOP_BACK)
    else:
        vis.face((x, y), z + 2.15, facing, 0.9, 0.9, YIELD, YIELD_BACK)


def speed_sign(col, vis, x, y, z, facing, cell, style):
    post(col, x, y, z, 2.6)
    if style == "eu":
        vis.face((x, y), z + 2.1, facing, 0.6, 0.6, cell, CIRCLE_BACK)
    else:
        u0, v0, u1, v1 = cell_uv(cell)
        du = (u1 - u0) * 0.1
        vis.face((x, y), z + 2.0, facing, 0.6, 0.75, cell, RECT_BACK, uv_box=(u0 + du, v0, u1 - du, v1))


def signal_head(vis, cx, cy, zb, facing, lit):
    f = _yaw_vec(facing)
    vis.box((cx, cy), f, 0.13, 0.16, zb, zb + 0.95, SIG_BODY, front_cell=lit, front=f)
    # visor plate behind (the black backboard)
    r = np.array([-f[1], f[0]])
    b = np.array([cx, cy]) - f * 0.135
    vis.quad(np.r_[b - r * 0.3, zb - 0.1], np.r_[b + r * 0.3, zb - 0.1], np.r_[b + r * 0.3, zb + 1.05],
             np.r_[b - r * 0.3, zb + 1.05], DARK, out=np.r_[f, 0], double=True)


def traffic_light(col, vis, x, y, z, facing, lane_dx, style, lit):
    """Post at the kerb; NA: mast arm reaching over the lane (lane_dx: 2D vector from the post to the lane)."""
    na = style in ("us", "ca", "qc")
    h = 6.6 if na else 3.6
    post(col, x, y, z, h, r=0.1, cell=GALV if na else DARK)
    signal_head(vis, x + facing[0] * 0.25, y + facing[1] * 0.25, z + 2.4, facing, lit)
    if na and np.linalg.norm(lane_dx) > 1.0:
        e = np.array([x, y]) + lane_dx
        vis.beam((x, y, z + 6.3), (e[0], e[1], z + 6.3), 0.08, GALV)
        vis.beam((x, y, z + 5.2), (x + lane_dx[0] * 0.35, y + lane_dx[1] * 0.35, z + 6.25), 0.04, GALV)
        signal_head(vis, e[0], e[1], z + 5.25, facing, lit)


def lamp_post(col, vis, x, y, z, toward):
    t = _yaw_vec(toward)
    post(col, x, y, z, 8.0, r=0.09, cell=METAL)
    e = np.array([x, y]) + t * 1.6
    vis.beam((x, y, z + 7.8), (e[0], e[1], z + 8.1), 0.05, METAL)
    vis.box(e + t * 0.25, t, 0.35, 0.15, z + 7.95, z + 8.15, METAL)
    _lamp_lens(vis, e + t * 0.25, t, z + 7.94)


def _lamp_lens(vis, c, t, z):
    r = np.array([-t[1], t[0]])
    pts = [c + t * 0.3 + r * 0.12, c - t * 0.3 + r * 0.12, c - t * 0.3 - r * 0.12, c + t * 0.3 - r * 0.12]
    (vis.lens or vis).quad(*[np.r_[p, z] for p in pts], LAMP_LIT, out=np.array([0, 0, -1.0]))
    vis.lights.append((float(c[0]), float(c[1]), float(z)))


UTILITY_WIRES = [(-1.05, 10.45), (0.0, 11.05), (1.05, 10.45), (0.25, 7.2)]     # (lateral, height): 3 phases + telecom


def utility_pole(col, vis, x, y, z, axis, lamp_toward=None):
    """Wooden pole (11 m), cross arm along `axis` (2D unit, across the line), insulators."""
    a = _yaw_vec(axis)
    col.beam((x, y, z - 0.5), (x, y, z + 11.0), 0.14, WOOD, sides=6)
    p = np.array([x, y])
    vis.box(p, a, 1.25, 0.05, z + 10.2, z + 10.32, WOOD)
    for lat in (-1.05, 1.05):
        q = p + a * lat
        vis.beam((q[0], q[1], z + 10.32), (q[0], q[1], z + 10.5), 0.05, INSULATOR, sides=4)
    vis.beam((x, y, z + 11.0), (x, y, z + 11.1), 0.05, INSULATOR, sides=4)
    vis.box(p + a * 0.25, a, 0.12, 0.08, z + 7.0, z + 7.3, DARK)        # telecom clamp
    if lamp_toward is not None:
        t = _yaw_vec(lamp_toward)
        e = p + t * 2.2
        vis.beam((x, y, z + 8.6), (e[0], e[1], z + 9.0), 0.04, GALV)
        vis.box(e + t * 0.2, t, 0.32, 0.14, z + 8.9, z + 9.08, GALV)
        _lamp_lens(vis, e + t * 0.2, t, z + 8.89)


def wooden_hv_pole(col, vis, x, y, z, axis):
    a = _yaw_vec(axis)
    col.beam((x, y, z - 0.5), (x, y, z + 15.0), 0.18, WOOD, sides=6)
    vis.box(np.array([x, y]), a, 1.9, 0.07, z + 13.6, z + 13.8, WOOD)
    for lat in (-1.7, 0.0, 1.7):
        q = np.array([x, y]) + a * lat
        vis.beam((q[0], q[1], z + 13.8), (q[0], q[1], z + 14.3), 0.07, INSULATOR, sides=4)


HV_WOOD_WIRES = [(-1.7, 14.3), (0.0, 14.3), (1.7, 14.3)]
PYLON_ARMS = [(20.0, 5.0), (24.5, 6.6), (29.0, 5.0)]       # (height, half span) of the three cross arms
PYLON_WIRES = [(s * hs, h - 2.2) for h, hs in PYLON_ARMS for s in (-1, 1)]


def pylon(col, vis, x, y, z, axis):
    """Steel lattice tower (~33 m), double circuit: three cross arms along `axis`."""
    a = np.r_[_yaw_vec(axis), 0.0]
    b = np.array([-a[1], a[0], 0.0])
    o = np.array([x, y, z])
    zt = np.array([0, 0, 1.0])
    body = 28.0

    def half(h):
        return 3.2 + (0.9 - 3.2) * min(max(h, 0.0), body) / body

    def corner(h, sa, sb):
        return o + half(h) * (sa * a + sb * b) + zt * h
    corners = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    levels = [0.0, 5.0, 10.0, 14.5, 18.5, 22.0, 26.0, body]
    for sa, sb in corners:
        col.beam(corner(-1.0, sa, sb), corner(5.0, sa, sb), 0.16, METAL)
        vis.beam(corner(5.0, sa, sb), corner(body, sa, sb), 0.12, METAL)
        vis.beam(corner(body, sa, sb), o + zt * 33.0, 0.07, METAL)
    for i, h in enumerate(levels[1:], 1):
        for k in range(4):
            c1, c2 = corners[k], corners[(k + 1) % 4]
            vis.beam(corner(h, *c1), corner(h, *c2), 0.05, METAL)
            h0 = levels[i - 1]
            vis.beam(corner(h0, *c1), corner(h, *c2), 0.035, METAL)          # X bracing
            vis.beam(corner(h0, *c2), corner(h, *c1), 0.035, METAL)
    for h, hs in PYLON_ARMS:
        for s in (-1, 1):
            root = half(h)
            vis.beam(o + zt * h + s * a * root + b * 0.5, o + zt * h + s * a * hs, 0.06, METAL)
            vis.beam(o + zt * h + s * a * root - b * 0.5, o + zt * h + s * a * hs, 0.06, METAL)
            vis.beam(o + zt * (h + 2.0) + s * a * root, o + zt * h + s * a * hs, 0.05, METAL)
            tip = o + zt * h + s * a * hs
            vis.beam(tip, tip - zt * 2.2, 0.09, INSULATOR, sides=6)


def wire(G, p, q, sag, r=0.022):
    """A conductor from p to q (3D) hanging `sag` m at mid-span: two crossed ribbons, both sides."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    L = np.linalg.norm(q[:2] - p[:2])
    n = int(np.clip(L / 8.0, 4, 40))
    t = np.linspace(0, 1, n + 1)
    pts = p[None] + (q - p)[None] * t[:, None]
    pts[:, 2] -= 4 * sag * t * (1 - t)
    d = _yaw_vec((q - p)[:2])
    side = np.array([-d[1], d[0], 0.0]) * r
    up = np.array([0, 0, r])
    for i in range(n):
        a, b = pts[i], pts[i + 1]
        G.quad(a - side, b - side, b + side, a + side, DARK, double=True)
        G.quad(a - up, b - up, b + up, a + up, DARK, double=True)


# ---------------------------------------------------------------- placement helpers
def _h01(*vals):
    h = hashlib.md5("|".join(map(str, vals)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _arclen(p):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def _at(p, s, cum=None):
    """Point and unit tangent at arc length s along polyline p."""
    cum = _arclen(p) if cum is None else cum
    s = float(np.clip(s, 0, cum[-1]))
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(p) - 2))
    seg = p[i + 1] - p[i]
    L = max(np.linalg.norm(seg), 1e-9)
    return p[i] + seg * (s - cum[i]) / L, seg / L


# ---------------------------------------------------------------- crossings, guard rails, substations
ZEBRA_W, ZEBRA_GAP, CROSS_LEN = 0.5, 0.5, 3.0     # m: stripe width / gap across the road, crossing width along it
CROSS_LINE_W = 0.3
PAINT_LIFT = 0.025                                 # above the ground surface (the centre lines are at 0.015)
GR_POST_SPACING = 2.0                              # m between guard rail posts
GR_RAIL = (0.5, 0.82)                              # m, bottom / top of the W-beam above the ground
GR_CLEAR = 0.8                                     # m, a guard rail stays this far off the track's asphalt


def _paint_strip(sc, a, b, w, step=1.0):
    """A painted strip of width w from a to b (2D), draped on the built ground surface -> (V, F)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 0.05:
        return None
    t = (b - a) / L
    n = np.array([-t[1], t[0]]) * w / 2
    k = max(1, int(math.ceil(L / step)))
    c = a + np.outer(np.linspace(0, 1, k + 1), b - a)
    xy = np.vstack([c + n, c - n])
    z = sc.tm.surface_height(xy[:, 0], xy[:, 1]) + PAINT_LIFT
    V = np.column_stack([xy, z])
    m = k + 1
    i = np.arange(k)
    # left row (c + n) first: counter-clockwise seen from above
    F = np.vstack([np.column_stack([i, m + i + 1, i + 1]), np.column_stack([i, m + i, m + i + 1])])
    return V, F


def crossing_frame(sc, p, roads, rtree, seg_ri):
    """Where a mapped crossing node is: (centre, unit direction of the road, half width) on the track or on the
    nearest street, or None."""
    d, idx = sc.tm.tree.query(p)
    hw = float(sc.tm.hwc[idx])
    if d < hw + 1.0:                                       # on the track
        pts = sc.tm.tree.data
        i0, i1 = max(idx - 2, 0), min(idx + 2, len(pts) - 1)
        tan = _yaw_vec(pts[i1] - pts[i0])
        c = pts[idx]
        # the node may sit off the centre line (a one-side sidewalk node): use the track's centre there
        return np.asarray(c, float), tan, hw
    if rtree is None:
        return None
    d, i = rtree.query(p)
    r = roads[seg_ri[i]]
    if d > r["half"] + 3.0 or r["bridge"]:
        return None
    cum = _arclen(r["xy"])
    c, tan = _at(r["xy"], _project(r["xy"], p, cum), cum)
    return c, tan, float(r["half"])


def _project(xy, p, cum=None):
    """Arc length of the point of polyline xy nearest to p."""
    cum = _arclen(xy) if cum is None else cum
    a, b = xy[:-1], xy[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
    d = np.linalg.norm(a + ab * t[:, None] - p, axis=1)
    i = int(np.argmin(d))
    return float(cum[i] + t[i] * np.linalg.norm(ab[i]))


def crosswalk(sc, mb, c, tan, half, style):
    """Zebra (stripes along the road, across its whole width) or two lines across the road."""
    nrm = np.array([-tan[1], tan[0]])
    pieces = []
    if style == "zebra":
        n = max(2, int((2 * half - ZEBRA_GAP) // (ZEBRA_W + ZEBRA_GAP)) + 1)
        span = n * ZEBRA_W + (n - 1) * ZEBRA_GAP
        for k in range(n):
            o = -span / 2 + ZEBRA_W / 2 + k * (ZEBRA_W + ZEBRA_GAP)
            q = c + nrm * o
            pieces.append(_paint_strip(sc, q - tan * CROSS_LEN / 2, q + tan * CROSS_LEN / 2, ZEBRA_W))
    else:
        for sgn in (-1, 1):
            q = c + tan * sgn * (CROSS_LEN / 2)
            pieces.append(_paint_strip(sc, q - nrm * half, q + nrm * half, CROSS_LINE_W))
    n_ok = 0
    for pc in pieces:
        if pc is not None:
            mb.add(pc[0], pc[1], np.zeros((len(pc[0]), 2)))
            n_ok += 1
    return n_ok


def guard_rail(sc, col, xy, road_pts):
    """A W-beam guard rail on posts along the polyline xy. Pieces on the track's asphalt are pushed to its edge,
    pieces on a street or in a building are left out. Returns the length built (m)."""
    xy = resample_line(xy, GR_POST_SPACING)
    if len(xy) < 2:
        return 0.0
    pts = []
    for p in xy:
        d, idx = sc.tm.tree.query(p)
        hw = float(sc.tm.hwc[idx])
        if d - hw < GR_CLEAR:                       # OSM's line drifts onto the track: move it to the verge
            c = sc.tm.tree.data[idx]
            v = _yaw_vec(p - c) if d > 0.05 else np.array([1.0, 0.0])
            p = c + v * (hw + GR_CLEAR)
        from shapely.geometry import Point

        pt = Point(*p)
        bad = (sc.streets is not None and sc.streets.contains(pt)) or (sc.blds is not None and sc.blds.contains(pt))
        pts.append(None if bad else np.asarray(p, float))
    built = 0.0
    run = []
    for p in pts + [None]:
        if p is not None:
            run.append(p)
            continue
        if len(run) >= 2:
            built += _rail_run(sc, col, np.array(run), road_pts)
        run = []
    return built


def _rail_run(sc, col, P, road_pts):
    z = sc.tm.surface_height(P[:, 0], P[:, 1])
    # the road side: towards the nearest road point (the beam hangs on that side of the posts)
    mid = P[len(P) // 2]
    t = _yaw_vec(P[-1] - P[0])
    nrm = np.array([-t[1], t[0]])
    if road_pts is not None:
        _d, i = road_pts.query(mid)
        if np.dot(road_pts.data[i] - mid, nrm) < 0:
            nrm = -nrm
    L = 0.0
    for i, p in enumerate(P):
        col.box(p, t, 0.05, 0.08, z[i] - 0.3, z[i] + GR_RAIL[1] - 0.02, GALV)
        if i:
            a, b = P[i - 1], p
            seg = b - a
            L += float(np.linalg.norm(seg))
            sn = np.array([-seg[1], seg[0]]) / max(np.linalg.norm(seg), 1e-9)
            if np.dot(sn, nrm) < 0:
                sn = -sn
            off = sn * 0.14
            za, zb = z[i - 1], z[i]
            u0, v0, u1, v1 = cell_uv(RAIL_W)
            q = [np.r_[a + off, za + GR_RAIL[0]], np.r_[b + off, zb + GR_RAIL[0]],
                 np.r_[b + off, zb + GR_RAIL[1]], np.r_[a + off, za + GR_RAIL[1]]]
            col.quad(*q, RAIL_W, out=np.r_[sn, 0], uv=[(u0, v1), (u1, v1), (u1, v0), (u0, v0)])
            back = [x - np.r_[sn * 0.04, 0] for x in q]
            col.quad(*back, RAIL_W, out=np.r_[-sn, 0], uv=[(u0, v1), (u1, v1), (u1, v0), (u0, v0)])
            col.quad(q[3], q[2], back[2], back[3], GALV, out=np.array([0, 0, 1.0]))
    # the ends turned down into the ground
    for k, j in ((0, 1), (len(P) - 1, len(P) - 2)):
        a = P[k]
        e = a + _yaw_vec(a - P[j]) * 2.0
        ze = float(sc.tm.surface_height(np.array([e[0]]), np.array([e[1]]))[0])
        off = nrm * 0.14
        col.quad(np.r_[a + off, z[k] + GR_RAIL[0]], np.r_[e + off, ze - 0.05], np.r_[e + off, ze + 0.1],
                 np.r_[a + off, z[k] + GR_RAIL[1]], RAIL_W, double=True)
    return L


SUB_FENCE_H = 2.4
SUB_POST = 3.0


def substation(sc, col, vis, poly_xy, voltage, tr_xy=None):
    """Fence along the outline (chain-link, collides), transformers, and steel gantries with insulators for the
    high-voltage yards. Returns the number of transformers built, or None when the substation is in the way."""
    from shapely.geometry import Point, Polygon

    poly = Polygon(poly_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 30:
        return None
    c = poly.representative_point()
    if not sc.ok(c.x, c.y, need=sc.track_clear, check_streets=False, spacing=0.0):
        return None
    ring = np.asarray(poly.exterior.coords if poly.geom_type == "Polygon" else
                      max(poly.geoms, key=lambda g: g.area).exterior.coords)
    # fence
    for a, b in zip(ring[:-1], ring[1:]):
        L = float(np.linalg.norm(b - a))
        k = max(1, int(math.ceil(L / SUB_POST)))
        for j in range(k):
            p, q = a + (b - a) * j / k, a + (b - a) * (j + 1) / k
            m = (p + q) / 2
            if not sc.ok(m[0], m[1], need=1.0, check_streets=True, spacing=0.0):
                continue
            zp, zq = (float(v) for v in sc.tm.surface_height(np.array([p[0], q[0]]), np.array([p[1], q[1]])))
            u0, v0, u1, v1 = cell_uv(CHAIN)
            col.quad(np.r_[p, zp], np.r_[q, zq], np.r_[q, zq + SUB_FENCE_H], np.r_[p, zp + SUB_FENCE_H], CHAIN,
                     uv=[(u0, v1), (u1, v1), (u1, v0), (u0, v0)], double=True)
            col.beam((p[0], p[1], zp - 0.3), (p[0], p[1], zp + SUB_FENCE_H + 0.1), 0.04, GALV)
    # equipment on a grid inside the yard, along its long axis
    inner = poly.buffer(-4.0)
    if inner.is_empty:
        return 0
    mrr = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4]
    e1, e2 = mrr[1] - mrr[0], mrr[2] - mrr[1]
    ax = _yaw_vec(e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2)
    side = np.array([-ax[1], ax[0]])
    hv = voltage >= 69000 or (voltage == 0 and poly.area > 2500)
    n_tr = 0
    spots = []
    if tr_xy is not None and len(tr_xy):
        spots = [p for p in tr_xy if inner.contains(Point(*p))]
    if not spots:
        want = int(np.clip(poly.area / 700.0, 1, 6))
        cx, cy = inner.centroid.x, inner.centroid.y
        cands = [np.array([cx, cy]) + ax * (i - (want - 1) / 2) * 9.0 for i in range(want)]
        spots = [p for p in cands if inner.contains(Point(*p))] or [np.array([cx, cy])]
    for p in spots:
        z = sc.z(p[0], p[1])
        col.box(p, ax, 1.8, 1.3, z - 0.2, z + 2.8, TRANSFORMER)          # tank
        vis.box(p + side * 1.6, ax, 1.4, 0.3, z + 0.3, z + 2.5, TRANSFORMER)  # radiators
        vis.box(p - ax * 1.2, ax, 0.5, 0.5, z + 2.8, z + 3.6, TRANSFORMER)   # conservator
        for k in (-1, 0, 1):                                                 # bushings
            b = p + ax * 0.6 + side * 0.7 * k
            vis.beam((b[0], b[1], z + 2.8), (b[0], b[1], z + 4.2), 0.09, BUSHING, sides=6)
        n_tr += 1
    if hv:
        # gantries across the yard every 14 m: two lattice columns, a beam, insulator strings
        L = float(np.ptp(inner.exterior.coords, axis=0).max()) if inner.geom_type == "Polygon" else 0.0
        cx, cy = inner.centroid.x, inner.centroid.y
        n_g = int(np.clip(L // 14, 1, 6))
        width = 9.0
        for i in range(n_g):
            m = np.array([cx, cy]) + ax * (i - (n_g - 1) / 2) * 14.0 + side * 6.0
            a, b = m - side * width / 2, m + side * width / 2
            if not (inner.contains(Point(*a)) and inner.contains(Point(*b))):
                continue
            za, zb = sc.z(*a), sc.z(*b)
            zt = max(za, zb) + 10.0
            col.beam((a[0], a[1], za - 0.3), (a[0], a[1], zt), 0.18, GALV)
            col.beam((b[0], b[1], zb - 0.3), (b[0], b[1], zt), 0.18, GALV)
            vis.beam((a[0], a[1], zt - 0.2), (b[0], b[1], zt - 0.2), 0.16, GALV)
            for k in (-1, 0, 1):
                q = m + side * k * 2.6
                vis.beam((q[0], q[1], zt - 0.3), (q[0], q[1], zt - 2.1), 0.07, BUSHING, sides=6)
                ze = sc.z(*q)
                vis.beam((q[0], q[1], ze), (q[0], q[1], ze + 3.2), 0.1, GALV)       # insulator post / breaker
                vis.beam((q[0], q[1], ze + 3.2), (q[0], q[1], ze + 4.4), 0.08, BUSHING, sides=6)
    return n_tr


class Scene:
    """Everything needed to keep props in sensible places."""

    def __init__(self, tm, track_clear, streets_geom=None, buildings=None):
        from shapely import prepared

        self.tm = tm
        self.track_clear = track_clear
        self.streets = prepared.prep(streets_geom.buffer(0.3)) if streets_geom is not None else None
        self.blds = prepared.prep(buildings) if buildings is not None else None
        self.keep_out = []              # areas taken (substations): no pole or sign inside
        self.lights = []                # street lamps built: (x, y, z) under each lens (CSP night lights)
        self.placed = []                # (x, y) of the posts placed so far
        self.items = []                 # (kind, x, y, facing x, facing y): for tests / debugging

    def ok(self, x, y, need=None, check_streets=True, spacing=0.8):
        from shapely.geometry import Point

        d, _, hw = self.tm.road_info3(np.array([x]), np.array([y]))
        if d[0] - hw[0] < (self.track_clear if need is None else need):
            return False
        pt = Point(x, y)
        if check_streets and self.streets is not None and self.streets.contains(pt):
            return False
        if self.blds is not None and self.blds.contains(pt):
            return False
        if any(k.contains(pt) for k in self.keep_out):
            return False
        for px, py in self.placed[-400:]:
            if (px - x) ** 2 + (py - y) ** 2 < spacing ** 2:
                return False
        return True

    def add_keep_out(self, geom):
        from shapely import prepared

        self.keep_out.append(prepared.prep(geom))

    def push_off_track(self, x, y):
        """A point too close to the track moved straight out to the clearance line (signs at junctions on the
        track: the measured asphalt is often wider than OSM's)."""
        d, idx = self.tm.tree.query([x, y])
        hw = self.tm.hwc[idx]
        if d - hw >= self.track_clear:
            return x, y
        c = self.tm.tree.data[idx]
        v = np.array([x, y]) - c
        v = v / max(np.linalg.norm(v), 1e-6)
        q = c + v * (hw + self.track_clear + 0.1)
        return float(q[0]), float(q[1])

    def z(self, x, y):
        return float(self.tm.surface_height(np.array([x]), np.array([y]))[0])


def _roads_xy(roads, proj):
    out = []
    for r in roads:
        if r["highway"] not in CTRL and r["highway"] != "service":
            continue
        if r.get("tunnel"):
            continue
        xy = proj.latlngs_to_xy(r["coords"])
        if len(xy) < 2:
            continue
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.05]
        xy = xy[keep]
        if len(xy) < 2:
            continue
        out.append({"id": r["id"], "xy": xy, "cls": r["highway"], "half": r["width"] / 2,
                    "maxspeed": r.get("maxspeed"), "oneway": r.get("oneway", 0) or 0,
                    "ring": bool(r.get("roundabout")), "lit": r.get("lit"), "bridge": r.get("bridge")})
    return out


def _key(p):
    return (int(round(p[0] * 20)), int(round(p[1] * 20)))


def junctions(roads):
    """{key: {"c": xy, "arms": [arm]}} where 3+ road arms of CTRL roads meet. arm: road index, vertex index, u (unit
    vector away from the junction), inbound (traffic can drive in on it), far (key of the arm's other end)."""
    vmap = {}
    for ri, r in enumerate(roads):
        if r["cls"] not in CTRL:
            continue
        for vi, p in enumerate(r["xy"]):
            vmap.setdefault(_key(p), []).append((ri, vi))
    jkeys = set()
    for k, lst in vmap.items():
        n = sum((vi > 0) + (vi < len(roads[ri]["xy"]) - 1) for ri, vi in lst)
        if n >= 3:
            jkeys.add(k)
    out = {}
    for k in jkeys:
        arms = []
        c = None
        for ri, vi in vmap[k]:
            r = roads[ri]
            p = r["xy"]
            c = p[vi]
            for step in (-1, 1):
                j = vi + step
                if j < 0 or j >= len(p):
                    continue
                # direction 6 m along the arm, and the arm's far end (next junction or road end)
                seg = p[vi::step] if step > 0 else p[vi::-1]
                cum = _arclen(seg)
                q, _ = _at(seg, min(6.0, cum[-1]), cum)
                u = _yaw_vec(q - c)
                far = None
                for m in range(1, len(seg)):
                    if _key(seg[m]) in jkeys:
                        far = (_key(seg[m]), cum[m])
                        break
                # traffic drives in along -u: along the way (+index) when the arm is behind it (step -1)
                inbound = r["oneway"] == 0 or (r["oneway"] == 1 and step == -1) or (r["oneway"] == -1 and step == 1)
                arms.append({"ri": ri, "vi": vi, "u": u, "inbound": inbound, "far": far, "step": step})
        out[k] = {"c": c, "arms": arms}
    return out


def _controls(j, roads, signalled, k):
    """[(arm, 'signal' | 'stop' | 'giveway')] by the usual rules (Map2Lap's junction_controls)."""
    arms = j["arms"]
    cls = [roads[a["ri"]]["cls"] for a in arms]
    if sum(roads[a["ri"]]["ring"] for a in arms) >= 2:
        return [(a, "giveway") for a in arms if not roads[a["ri"]]["ring"] and a["inbound"]]
    if signalled:
        return [(a, "signal") for a in arms if a["inbound"]]
    major = [a for a, c in zip(arms, cls) if c in MAJOR]
    minor = [a for a, c in zip(arms, cls) if c in MINOR]
    if major:
        return [(a, "stop") for a in minor if a["inbound"]]
    if len(arms) == 3:
        def straightest(a):
            return max(-float(a["u"] @ b["u"]) for b in arms if b is not a)
        a = min(arms, key=straightest)
        return [(a, "stop")] if a["inbound"] else []
    key = tuple(sorted(roads[a["ri"]]["id"] for a in arms))
    return [(a, "stop") for a in minor if a["inbound"]] if _h01("allway", key) < 0.5 else []


def _approach_spot(j, arm, roads, back=1.2):
    """Kerb spot on the drivers' side of an approach, just before the junction."""
    r = roads[arm["ri"]]
    wide = max(roads[a["ri"]]["half"] for a in j["arms"] if a is not arm)
    u = arm["u"]
    inc = -u
    right = np.array([inc[1], -inc[0]])
    q = j["c"] + u * (wide + back + 0.5) + right * (r["half"] + KERB)
    return q, right


# ---------------------------------------------------------------- build
def build(roads_raw, furniture, proj, sc: Scene, bounds, opts, style, log=print, keep=None):
    """-> (parts, textures dict, stats). opts: power_lines, road_signs, street_lights (bools)."""
    from scipy.spatial import cKDTree

    roads = _roads_xy(roads_raw, proj)
    col, vis, colp, pyl, whv, wlv = Geo(), Geo(), Geo(), Geo(), Geo(), Geo()
    lens = vis.lens = Geo()
    stats = {"traffic_lights": 0, "stop_signs": 0, "give_way_signs": 0, "speed_signs": 0, "street_lights": 0,
             "utility_poles": 0, "pylons": 0, "crosswalks": 0, "guard_rails_m": 0, "substations": 0,
             "transformers": 0}
    paint = MeshBuilder("ROAD_CROSSWALKS", "road_paint")
    x0, y0, x1, y1 = bounds
    inb = lambda x, y, m=0.0: x0 - m <= x <= x1 + m and y0 - m <= y <= y1 + m      # noqa: E731
    if keep is not None:                    # an area polygon: nothing stands where no ground is built
        from shapely import contains_xy

        _inb = inb
        inb = lambda x, y, m=0.0: _inb(x, y, m) and bool(contains_xy(keep, x, y))      # noqa: E731
    fx = lambda items: (proj.latlngs_to_xy([(i["lat"], i["lon"]) for i in items])  # noqa: E731
                        if items else np.zeros((0, 2)))
    speeds_used = []

    def place(x, y, need=None, check_streets=True, spacing=0.8):
        if not inb(x, y) or not sc.ok(x, y, need, check_streets, spacing):
            return None
        sc.placed.append((x, y))
        return sc.z(x, y)

    # ------------------------------------------------ junction controls
    if opts.get("road_signs"):
        J = junctions(roads)
        sig = fx(furniture.get("signals", []))
        stops = furniture.get("stops", []) + [dict(g, kind="giveway") for g in furniture.get("giveways", [])]
        stop_xy = fx(stops)
        jk = list(J)
        jc = np.array([J[k]["c"] for k in jk]) if jk else np.zeros((0, 2))
        reach = {k: max(roads[a["ri"]]["half"] for a in J[k]["arms"]) for k in jk}
        signalled = set()
        used_sig = np.zeros(len(sig), bool)
        if len(sig) and len(jc):
            t = cKDTree(sig)
            for k, c in zip(jk, jc):
                idx = t.query_ball_point(c, reach[k] + SIGNAL_REACH)
                if idx:
                    signalled.add(k)
                    used_sig[idx] = True
        # mapped stop / give-way nodes -> the approach they are on
        mapped = {}
        if len(stop_xy) and len(jc):
            tj = cKDTree(jc)
            for s, p in zip(stops, stop_xy):
                d, i = tj.query(p)
                if d > STOP_REACH:
                    continue
                k = jk[i]
                kind = "giveway" if s.get("kind") == "giveway" else "stop"
                J_ = J[k]
                if d < 3.0 or s.get("all_way"):                     # on the junction node itself: use the rules
                    mapped.setdefault(k, {})["rules"] = kind
                    if s.get("all_way"):
                        mapped[k]["all"] = True
                    continue
                arm = max(J_["arms"], key=lambda a: float(a["u"] @ _yaw_vec(p - J_["c"])))
                mapped.setdefault(k, {}).setdefault("arms", []).append((arm, kind))
        for k in jk:
            j = J[k]
            if k in signalled:
                put = _controls(j, roads, True, k)
                # dual carriageways: no head on a short link to the next signalled junction
                put = [(a, w) for a, w in put if not (a["far"] and a["far"][1] < 30.0 and a["far"][0] in signalled)]
            elif k in mapped and "arms" in mapped[k]:
                put = [(a, w) for a, w in mapped[k]["arms"] if a["inbound"]]
            elif k in mapped and mapped[k].get("all"):
                put = [(a, "stop") for a in j["arms"] if a["inbound"]]
            else:
                put = _controls(j, roads, False, k)
            major_arm = {id(a): roads[a["ri"]]["cls"] in MAJOR for a in j["arms"]}
            for arm, what in put:
                if roads[arm["ri"]]["cls"] not in CTRL:
                    continue
                q, right = _approach_spot(j, arm, roads)
                x, y = sc.push_off_track(*q)
                z = place(x, y)
                if z is None:
                    continue
                facing = arm["u"]
                if what == "signal":
                    lane = -right * (KERB + roads[arm["ri"]]["half"] * (0.5 if roads[arm["ri"]]["oneway"] == 0 else 1.0))
                    lit = SIG_GREEN if major_arm[id(arm)] else SIG_RED
                    traffic_light(col, vis, x, y, z, facing, lane, style, lit)
                    stats["traffic_lights"] += 1
                    sc.items.append(("signal", x, y, *facing))
                else:
                    stop_sign(col, vis, x, y, z, facing, what)
                    sc.items.append((what, x, y, *facing))
                    stats["stop_signs" if what == "stop" else "give_way_signs"] += 1
        # mid-block signals (pedestrian crossings): both directions where the road allows
        if len(sig):
            seg_pts, seg_ri = [], []
            for ri, r in enumerate(roads):
                if r["cls"] in CTRL:
                    seg_pts.append(r["xy"])
                    seg_ri += [ri] * len(r["xy"])
            if seg_pts:
                allp = np.vstack(seg_pts)
                tr = cKDTree(allp)
                done = []
                for p in sig[~used_sig]:
                    if any(np.hypot(*(p - q)) < 25 for q in done):
                        continue
                    d, i = tr.query(p)
                    r = roads[seg_ri[i]]
                    if d > r["half"] + 4:
                        continue
                    done.append(p)
                    cum = _arclen(r["xy"])
                    sp = cum[np.argmin(np.linalg.norm(r["xy"] - p, axis=1))]
                    pt, tan = _at(r["xy"], sp, cum)
                    for sgn in (1, -1):
                        if r["oneway"] and sgn != r["oneway"]:
                            continue
                        travel = tan * sgn
                        right = np.array([travel[1], -travel[0]])
                        q = pt - travel * 1.5 + right * (r["half"] + KERB)
                        x, y = sc.push_off_track(*q)
                        z = place(x, y)
                        if z is None:
                            continue
                        lane = -right * (KERB + r["half"] * (0.5 if not r["oneway"] else 1.0))
                        traffic_light(col, vis, x, y, z, -travel, lane, style, SIG_RED)
                        stats["traffic_lights"] += 1
                        sc.items.append(("signal", x, y, *(-travel)))
        # speed limits where the mapped limit changes
        vmap = {}
        for ri, r in enumerate(roads):
            if r["cls"] in CTRL:
                vmap.setdefault(_key(r["xy"][0]), []).append((ri, 0))
                vmap.setdefault(_key(r["xy"][-1]), []).append((ri, len(r["xy"]) - 1))
        for ri, r in enumerate(roads):
            v = r["maxspeed"]
            if not v or r["cls"] not in CTRL:
                continue
            cum = _arclen(r["xy"])
            if cum[-1] < 50:
                continue
            for end, sgn in ((0, 1), (len(r["xy"]) - 1, -1)):
                if r["oneway"] and sgn != r["oneway"]:
                    continue
                others = [(oi, ov) for oi, ov in vmap.get(_key(r["xy"][end]), []) if oi != ri]
                if not others:
                    continue                       # the road comes in from outside the area
                # a sign when traffic can come in from a road with another (known) limit, turning or straight on
                feeders = [roads[oi]["maxspeed"] for oi, ov in others
                           if not roads[oi]["oneway"] or (roads[oi]["oneway"] == 1) == (ov == len(roads[oi]["xy"]) - 1)]
                if not any(f and f != v for f in feeders):
                    continue
                s = 25.0 if sgn == 1 else cum[-1] - 25.0
                pt, tan = _at(r["xy"], s, cum)
                travel = tan * sgn
                right = np.array([travel[1], -travel[0]])
                x, y = sc.push_off_track(*(pt + right * (r["half"] + KERB)))
                z = place(x, y)
                if z is None:
                    continue
                shown = v if style != "us" else int(round(v / 1.609344 / 5.0) * 5)
                if shown not in speeds_used:
                    if len(speeds_used) >= N_SPEED:
                        shown = min(speeds_used, key=lambda w: abs(w - shown))
                    else:
                        speeds_used.append(shown)
                speed_sign(col, vis, x, y, z, -travel, SPEED0 + speeds_used.index(shown), style)
                stats["speed_signs"] += 1
                sc.items.append((f"speed{shown}", x, y, *(-travel)))

    # ------------------------------------------------ power lines
    lines = furniture.get("lines", []) if opts.get("power_lines") else []
    minor_lines = []
    if lines:
        towers = fx(furniture.get("towers", []))
        tt = cKDTree(towers) if len(towers) else None
        for ln in lines:
            xy = proj.latlngs_to_xy(ln["coords"])
            hv = ln["kind"] == "line" and (ln["voltage"] == 0 or ln["voltage"] >= 69000)
            if ln["kind"] == "minor_line":
                minor_lines.append(xy)
            # supports: the mapped tower / pole nodes on the line, else its vertices
            sup = [p for p in xy if tt is None or tt.query(p)[0] < 3.0] if tt is not None else list(xy)
            if len(sup) < 2:
                sup = list(xy)
            sup = np.array(sup)
            keep = [inb(p[0], p[1]) for p in sup]
            prev = None
            for i, p in enumerate(sup):
                if not keep[i]:
                    prev = None
                    continue
                d_prev = sup[i] - sup[i - 1] if i > 0 else sup[1] - sup[0]
                d_next = sup[i + 1] - sup[i] if i + 1 < len(sup) else d_prev
                along = _yaw_vec(_yaw_vec(d_prev) + _yaw_vec(d_next))
                axis = np.array([-along[1], along[0]])
                need = 8.0 if hv else 2.5
                if not sc.ok(p[0], p[1], need=need, check_streets=True, spacing=3.0):
                    continue                       # no tower on the track / a street / in a house: span over it
                z = sc.z(p[0], p[1])
                sc.placed.append((p[0], p[1]))
                if hv:
                    pylon(colp, pyl, p[0], p[1], z, axis)
                elif ln["kind"] == "line":
                    wooden_hv_pole(colp, pyl, p[0], p[1], z, axis)
                else:
                    utility_pole(col, vis, p[0], p[1], z, axis)
                stats["pylons" if ln["kind"] == "line" else "utility_poles"] += 1
                arms = PYLON_WIRES if hv else (HV_WOOD_WIRES if ln["kind"] == "line" else UTILITY_WIRES[:3])
                here = [np.r_[p + axis * lat, z + h] for lat, h in arms]
                if prev is not None:
                    span = float(np.linalg.norm(p - prev[0]))
                    if 10.0 < span < 1500.0:
                        for a, b in zip(prev[1], here):
                            wire(whv if ln["kind"] == "line" else wlv, a, b, (0.025 if hv else 0.02) * span,
                                 r=0.025 if hv else 0.015)
                prev = (p, here)

    # ------------------------------------------------ substations (before the poles: nothing else inside them)
    if opts.get("substations", opts.get("power_lines")):
        tr_xy = fx(furniture.get("transformers", []))
        for sub in furniture.get("substations", []):
            xy = proj.latlngs_to_xy(sub["coords"])
            if not all(inb(p[0], p[1], 50.0) for p in xy) or not any(inb(p[0], p[1]) for p in xy):
                continue
            n_tr = substation(sc, colp, pyl, xy, sub.get("voltage", 0.0), tr_xy)
            if n_tr is None:
                continue
            stats["substations"] += 1
            stats["transformers"] += n_tr
            from shapely.geometry import Polygon

            sc.add_keep_out(Polygon(xy).buffer(1.0))

    # ------------------------------------------------ pedestrian crossings and guard rails
    seg_pts, seg_ri = [], []
    for ri, r in enumerate(roads):
        seg_pts.append(resample_line(r["xy"], 2.0))
        seg_ri += [ri] * len(seg_pts[-1])
    rtree = cKDTree(np.vstack(seg_pts)) if seg_pts else None
    if opts.get("crosswalks", opts.get("road_signs")):
        na = style in ("us", "ca", "qc")
        done = []
        cw = furniture.get("crossings", [])
        for item, p in zip(cw, fx(cw)):
            if not inb(p[0], p[1]):
                continue
            fr = crossing_frame(sc, p, roads, rtree, seg_ri)
            if fr is None:
                continue
            c, tan, half = fr
            if any(np.hypot(*(c - q)) < 6.0 for q in done):      # both sidewalk nodes of one crossing
                continue
            st = item.get("style", "zebra")
            st = ("lines" if na else "zebra") if st == "signals" else st
            if crosswalk(sc, paint, c, tan, half, st):
                done.append(c)
                stats["crosswalks"] += 1
    if opts.get("guardrails", opts.get("road_signs")):
        for gr in furniture.get("guardrails", []):
            xy = proj.latlngs_to_xy(gr["coords"])
            keep_pts = np.array([inb(p[0], p[1]) for p in xy])
            if keep_pts.sum() < 2:
                continue
            stats["guard_rails_m"] += int(round(guard_rail(sc, col, xy[keep_pts], rtree)))

    # ------------------------------------------------ utility poles and street lights along the streets
    lamps = fx(furniture.get("lamps", [])) if opts.get("street_lights") else np.zeros((0, 2))
    if opts.get("street_lights"):
        for p in lamps:
            ok_r = [r for r in roads if r["cls"] in CTRL]
            if not ok_r:
                break
            best = min(ok_r, key=lambda r: np.min(np.linalg.norm(r["xy"] - p, axis=1)))
            i = int(np.argmin(np.linalg.norm(best["xy"] - p, axis=1)))
            toward = best["xy"][i] - p
            x, y = sc.push_off_track(*p)
            z = place(x, y)
            if z is None:
                continue
            lamp_post(col, vis, x, y, z, toward)
            stats["street_lights"] += 1
        lamp_tree = cKDTree(lamps) if len(lamps) else None
        ml_tree = cKDTree(np.vstack([resample_line(m, 5.0) for m in minor_lines])) if minor_lines else None
        na = style in ("us", "ca", "qc")
        for r in roads:
            if r["ring"] or r["bridge"]:
                continue
            if na:
                if r["cls"] not in POLE_CLASSES:
                    continue
                spacing = POLE_SPACING
            else:
                if r["cls"] not in LAMP_CLASSES or r["lit"] == "no":
                    continue
                if r["lit"] != "yes" and r["cls"] not in ("residential", "living_street", "tertiary"):
                    continue
                spacing = EU_LAMP_SPACING
            cum = _arclen(r["xy"])
            if cum[-1] < 20:
                continue
            side = 1 if _h01("side", r["id"]) < 0.5 else -1
            phase = _h01("phase", r["id"]) * spacing
            prev, n_on = None, 0
            for s in np.arange(phase, cum[-1], spacing):
                pt, tan = _at(r["xy"], s, cum)
                right = np.array([tan[1], -tan[0]]) * side
                q = pt + right * (r["half"] + 1.6)
                if ml_tree is not None and ml_tree.query(q)[0] < 15.0:
                    prev = None
                    continue
                x, y = sc.push_off_track(*q)
                z = place(x, y, spacing=4.0)
                if z is None:
                    prev = None
                    continue
                if na:
                    near_lamp = lamp_tree is not None and lamp_tree.query([x, y])[0] < 12.0
                    lamp = (n_on % 2 == 0) and not near_lamp and r["lit"] != "no"
                    axis = right
                    utility_pole(col, vis, x, y, z, axis, lamp_toward=-right if lamp else None)
                    stats["utility_poles"] += 1
                    stats["street_lights"] += int(lamp)
                    here = [np.r_[np.array([x, y]) + axis * lat, z + h] for lat, h in UTILITY_WIRES]
                    if prev is not None and np.linalg.norm(np.array([x, y]) - prev[0]) < 70.0:
                        span = float(np.linalg.norm(np.array([x, y]) - prev[0]))
                        for a, b in zip(prev[1], here):
                            wire(wlv, a, b, 0.02 * span, r=0.014)
                    prev = (np.array([x, y]), here)
                else:
                    if lamp_tree is not None and lamp_tree.query([x, y])[0] < 12.0:
                        continue
                    lamp_post(col, vis, x, y, z, -right)
                    stats["street_lights"] += 1
                n_on += 1

    parts = []
    for G, prefix in ((col, "1WALL_PROPS"), (vis, "PROPS"), (colp, "1WALL_PYLON"), (pyl, "PYLONS"),
                      (whv, "WIRES_HV"), (wlv, "WIRES"), (lens, "PROPS_LENS")):
        m = G.mesh()
        if m is None:
            continue
        mb = MeshBuilder(prefix, "props_lit" if G is lens else "props")
        V, F, UV = m
        per = 30000                               # quads have their own vertices: split on quad boundaries
        nq = len(V) // 4
        for a in range(0, nq, per // 4):
            b = min(nq, a + per // 4)
            mb.add(V[a * 4:b * 4], F[a * 2:b * 2] - a * 4, UV[a * 4:b * 4])
        parts += mb.done()
    tex = {"props": atlas(style, speeds_used)} if parts else {}
    if lens.n and tex:
        tex["props_lit"] = tex["props"]
    sc.lights = list(vis.lights)
    parts += paint.done()
    log("  " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in stats.items() if v) if any(stats.values())
        else "  none in this area")
    return parts, tex, stats


def resample_line(xy, step):
    xy = np.asarray(xy, float)
    cum = _arclen(xy)
    if cum[-1] < step:
        return xy
    s = np.arange(0, cum[-1], step)
    return np.column_stack([np.interp(s, cum, xy[:, 0]), np.interp(s, cum, xy[:, 1])])

"""Road markings read from the aerial photo: which lines are really painted on the track, where, solid or
dashed, white or yellow.

Method
  1. Rectify the photo along the track: one row every 0.25 m of track, one column every 5 cm across it.
  2. Paint response: lightness minus its surroundings across the road (difference of Gaussians, 0.1 m vs
     0.6 m). A painted line (10-15 cm, blurred to ~0.3 m at 20 cm per pixel) is a narrow bright ridge with
     darker asphalt on BOTH sides - the asphalt / gravel edge (a step) does not count.
  3. Three search bands per row: the left edge line, the right edge line (within 0.9 m of each asphalt edge)
     and the centre line (within 0.9 m of the middle). In each band the strongest ridge above the noise of
     the asphalt nearby is a line candidate. Rows hidden by trees, shadows or cars are 'not seen'.
  4. Along the track, in 24 m windows: no line / solid / dashed (short runs repeating every 6-15 m), from
     the share of the seen rows that have paint and the run lengths. Windows not seen take the style of
     their neighbours. Pieces under 15 m are merged into the neighbours.
  5. Dashed lines get the measured dash length, period and phase (the dashes are where they are in real
     life); solid lines are painted from start to end of their piece. Colour: yellow when the paint is
     clearly yellower than the asphalt, else white. The line's lateral position follows the photo
     (smoothed), so an edge line 40 cm inside the asphalt stays 40 cm inside.

-> Markings (list of lines: side, colour, style, offset per centreline station, painted intervals along s)
or None when the photo cannot tell (too little of the road seen).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

DS = 0.25            # m along the track per row
DL = 0.05            # m across per column
BAND = 0.9           # m: search band width at each edge / around the middle
WIN = 24.0           # m: style window
MIN_PIECE = 15.0     # m: shortest piece of one style
LINE_W = {"edge": 0.12, "centre": 0.12}   # m: painted width
DOUBLE_GAP = 0.12    # m between the two lines of a double line


@dataclass
class Line:
    side: str                    # "left" | "right" | "centre"
    colour: str                  # "white" | "yellow"
    style: str                   # "solid" | "dashed" | "double"
    pieces: list                 # [(s0, s1)] painted along the track (m)
    offset: np.ndarray           # lateral offset (m, left +) per centreline station
    width: float = 0.12


@dataclass
class Markings:
    lines: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


# ------------------------------------------------------------------ geometry along the track
def _stations(cl, ds=DS):
    """s values every ds m and the interpolated point / normal / station index (fraction)."""
    L = cl.length
    s = np.arange(0.0, L if cl.loop else L + 1e-6, ds)
    S = np.concatenate([cl.s, [cl.length]]) if cl.loop else cl.s
    P = np.vstack([cl.p, cl.p[:1]]) if cl.loop else cl.p
    N = np.vstack([cl.n, cl.n[:1]]) if cl.loop else cl.n
    px = np.column_stack([np.interp(s, S, P[:, 0]), np.interp(s, S, P[:, 1])])
    nn = np.column_stack([np.interp(s, S, N[:, 0]), np.interp(s, S, N[:, 1])])
    nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
    return s, px, nn, S


def _at(s, S, v, loop):
    vv = np.concatenate([v, v[:1]]) if loop else v
    return np.interp(s, S, vv)


# ------------------------------------------------------------------ detection
def detect(pm, cl, hl, hr, log=print) -> Markings | None:
    from .roadwidth import rgb_to_lab

    s, px, nn, S = _stations(cl)
    hls, hrs = _at(s, S, hl, cl.loop), _at(s, S, hr, cl.loop)
    lo, hi = -float(hrs.max()) - 0.7, float(hls.max()) + 0.7
    offs = np.arange(lo, hi + DL / 2, DL)
    X = px[:, None, 0] + nn[:, None, 0] * offs[None]
    Y = px[:, None, 1] + nn[:, None, 1] * offs[None]
    rgb, ok = pm.sample(X.ravel(), Y.ravel(), order=1)
    rgb = rgb.reshape(len(s), len(offs), 3)
    ok = ok.reshape(len(s), len(offs))
    lab = rgb_to_lab(rgb)
    Lc, bc = lab[..., 0], lab[..., 2]
    a = lab[..., 1]
    asphalt = (offs[None] < hls[:, None] - 0.02) & (offs[None] > -hrs[:, None] + 0.02) & ok
    # paint is looked for a little beyond the measured edges too (an edge measured 30-50 cm too narrow)
    span = (offs[None] >= -hrs[:, None] - 0.6) & (offs[None] <= hls[:, None] + 0.6) & ok
    # asphalt reference per 20 m (median of the road surface): shadows / vegetation / cars are 'not seen'
    k = max(1, int(20.0 / DS))
    Lm = np.where(asphalt, Lc, np.nan)
    with np.errstate(all="ignore"):
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            rowL = np.nanmedian(Lm, axis=1)
    rowL = np.where(np.isfinite(rowL), rowL, np.nanmedian(rowL) if np.isfinite(rowL).any() else 50.0)
    refL = ndi.median_filter(rowL, size=k, mode="wrap" if cl.loop else "nearest")
    exg_like = (a < -11) & (bc > 8)                      # green: grass / leaves over the road
    dark = Lc < refL[:, None] - 14                       # tree shadow, dark car
    hidden = ~ok | exg_like | dark
    # ridge response (difference of Gaussians across, light smoothing along)
    g1 = ndi.gaussian_filter(Lc, (1.0, 0.10 / DL))
    g2 = ndi.gaussian_filter(Lc, (1.0, 0.60 / DL))
    R = g1 - g2
    side = int(round(0.4 / DL))
    left_ = np.roll(g1, side, axis=1)
    right_ = np.roll(g1, -side, axis=1)
    sym = np.minimum(g1 - left_, g1 - right_)            # darker on both sides
    peak = (R >= ndi.maximum_filter1d(R, int(round(0.3 / DL)) | 1, axis=1)) & span & ~hidden
    noise = 1.4826 * float(np.median(np.abs(R[asphalt & ~hidden] - np.median(R[asphalt & ~hidden])))) \
        if (asphalt & ~hidden).sum() > 100 else 1.0
    thr = max(2.5, 3.0 * noise)
    cand = peak & (R > thr) & (sym > thr * 0.5)
    # yellow paint on light asphalt is hardly lighter (half the white lines' contrast) but yellower. The
    # photo's colour is blurred (JPEG keeps colour at a lower resolution than lightness), so the position
    # comes from lightness and the colour from a broader yellowness bump around it
    gb1 = ndi.gaussian_filter(bc, (1.0, 0.20 / DL))
    gb2 = ndi.gaussian_filter(bc, (1.0, 1.20 / DL))
    Rb = gb1 - gb2
    cand |= peak & (R > max(1.6, 0.55 * thr)) & (sym > 0.25 * thr) & (Rb > 0.4)
    mid = (hls - hrs) / 2
    # how much of each row's road can be seen
    seen_row = ((asphalt & ~hidden).sum(1) / np.maximum(asphalt.sum(1), 1)) > 0.6
    info = {"seen": round(float(seen_row.mean()), 2), "threshold": round(thr, 1), "lines": []}
    if seen_row.mean() < 0.4:
        info["reason"] = f"only {seen_row.mean() * 100:.0f}% of the road visible on the photo"
        return Markings([], dict(info, usable=False))
    chains = _join_guided(_chain_strokes(cand, offs, cl.loop), cand, offs)
    lines = []
    rows = np.arange(len(s))
    for cov, tr in chains:
        jt = np.clip(np.searchsorted(offs, np.nan_to_num(tr, nan=offs[0])), 0, len(offs) - 1)
        seen = seen_row & np.isfinite(tr) & ~hidden[rows, jt] & span[rows, jt]
        has = cov & seen | cov                      # paint seen (a stroke counts even through a shadow edge)
        # hysteresis along the line: the paint response at the line's position, half as strong as on the
        # detected rows, still counts when it touches them (full dash length, not just its brightest core)
        jj = np.clip(jt[:, None] + np.arange(-2, 3)[None], 0, len(offs) - 1)
        prof = np.where(np.isfinite(tr), R[rows[:, None], jj].max(1), -np.inf)
        if has.any():
            low = prof > 0.5 * float(np.median(prof[has]))
            lab_, _ = ndi.label(low)
            keep = np.unique(lab_[has & low])
            has = has | (np.isin(lab_, keep[keep > 0]) & ~hidden[rows, jt])
        pos = np.where(cov, tr, np.nan)
        yel = np.where(cov, Rb[rows, jt], np.nan)
        m = float(np.nanmedian(tr - mid))
        hw = float(np.median((hls + hrs) / 2))
        name = "left" if np.nanmedian(hls - tr) < 1.0 else ("right" if np.nanmedian(tr + hrs) < 1.0 else
                                                          ("centre" if abs(m) < 0.25 * hw + 0.5 else "lane"))
        line = _classify(name, s, has, seen | has, pos, yel, cl, S, hls, hrs, mid, track=tr)
        if line is None:
            continue
        lines.append(line)
        info["lines"].append({"side": name, "style": line.style, "colour": line.colour,
                              "painted_m": round(sum(b - a_ for a_, b in line.pieces)),
                              "paint_share": round(float(has.sum() / max((seen | has).sum(), 1)), 2),
                              "position_m": round(float(np.nanmedian(tr)), 2)})
    info["usable"] = True
    return Markings(lines, info)


def _chain_strokes(cand, offs, loop, R=None, max_gap_m=16.0):
    """Follow each painted line row by row (every 0.25 m), like tracking objects in a video: the paint
    peaks of a row go to the nearest line being followed (within 0.2 m of where it was heading, more after a
    gap - a dashed line's gaps, a car, a shadow); peaks nobody claims start a new line; a line not seen for
    max_gap_m ends. Lines that end and start again in line with each other are joined.
    -> [(rows with paint (bool per row), lateral position per row (nan outside))], lines at least 15 m long
    with at least 5 m of paint."""
    n, nb = cand.shape
    merge = max(1, int(round(0.15 / DL)))
    gapr = int(max_gap_m / DS)
    active = []          # dict: rows, pos, last, slope
    done = []
    for r in range(n):
        cols = np.nonzero(cand[r])[0]
        peaks = []
        if len(cols):
            grp = np.split(cols, np.nonzero(np.diff(cols) > merge)[0] + 1)
            peaks = [float(offs[g].mean()) for g in grp]
        # predictions
        claims = []
        for li, L in enumerate(active):
            dr = r - L["last"]
            pred = L["pos"][-1] + L["slope"] * dr
            tol = min(0.2 + 0.03 * dr * DS, 0.6)
            for pi, p in enumerate(peaks):
                d = abs(p - pred)
                if d <= tol:
                    claims.append((d, li, pi))
        claims.sort()
        used_l, used_p = set(), set()
        for d, li, pi in claims:
            if li in used_l or pi in used_p:
                continue
            L = active[li]
            L["rows"].append(r)
            L["pos"].append(peaks[pi])
            L["last"] = r
            k = min(len(L["rows"]), int(3.0 / DS))
            if k >= 4:
                rr = np.asarray(L["rows"][-k:], float)
                pp = np.asarray(L["pos"][-k:], float)
                L["slope"] = float(np.clip(np.polyfit(rr, pp, 1)[0], -0.05, 0.05))
            used_l.add(li)
            used_p.add(pi)
        for pi, p in enumerate(peaks):
            if pi not in used_p:
                active.append({"rows": [r], "pos": [p], "last": r, "slope": 0.0})
        still = []
        for L in active:
            (done if r - L["last"] > gapr else still).append(L)
        active = still
    done += active
    done = [L for L in done if len(L["rows"]) >= 3]
    # join lines that continue each other
    done.sort(key=lambda L: L["rows"][0])
    joined = []
    for L in done:
        best = None
        for J in joined:
            if J["rows"][-1] < L["rows"][0] and (L["rows"][0] - J["rows"][-1]) * DS <= 2 * max_gap_m:
                if abs(J["pos"][-1] - L["pos"][0]) <= 0.5 and (best is None or J["rows"][-1] > best["rows"][-1]):
                    best = J
        if best is not None:
            best["rows"] += L["rows"]
            best["pos"] += L["pos"]
        else:
            joined.append(L)
    out = []
    for L in joined:
        rows = np.asarray(L["rows"])
        pos = np.asarray(L["pos"])
        if (rows[-1] - rows[0]) * DS < 15.0:
            continue
        cov = np.zeros(n, bool)
        cov[rows] = True
        cov = ndi.binary_closing(cov, structure=np.ones(3, bool)) | cov       # 0.5 m holes
        if cov.sum() * DS < 5.0:
            continue
        pr = np.full(n, np.nan)
        rr = np.arange(rows[0], rows[-1] + 1)
        pr[rr] = np.interp(rr, rows, pos)
        out.append((cov, pr))
    return out


def _join_guided(chains, cand, offs, max_gap_m=120.0, tol=0.5):
    """Worn stretches split a line into pieces. Lines run parallel, so a long line next to it (a guide)
    shows where the missing stretch runs: two pieces are joined when, kept at the same distance from the
    guide, they meet within `tol` m; paint seen along that path in the gap counts, the rest of the gap is
    painted too when the pieces line up (a line doesn't stop for 50 m and start again in the same place)."""
    if len(chains) < 2:
        return chains
    ext = lambda pr: np.nonzero(np.isfinite(pr))[0]
    changed = True
    while changed:
        changed = False
        chains.sort(key=lambda c: len(ext(c[1])), reverse=True)
        for ia, (ca, pa) in enumerate(chains):
            ea = ext(pa)
            for ib, (cb, pb) in enumerate(chains):
                if ib == ia:
                    continue
                eb = ext(pb)
                if not len(ea) or not len(eb) or eb[0] <= ea[-1]:
                    continue
                g0, g1 = ea[-1], eb[0]
                if (g1 - g0) * DS > max_gap_m:
                    continue
                guide = None
                for ig, (cg, pg) in enumerate(chains):
                    if ig in (ia, ib):
                        continue
                    if np.isfinite(pg[g0:g1 + 1]).all():
                        guide = pg
                        break
                if guide is None:
                    if (g1 - g0) * DS > 20.0:
                        continue
                    rel0, rel1 = pa[g0], pb[g1]
                    path = np.interp(np.arange(g0, g1 + 1), [g0, g1], [rel0, rel1])
                else:
                    d0, d1 = pa[g0] - guide[g0], pb[g1] - guide[g1]
                    if abs(d1 - d0) > tol:
                        continue
                    path = guide[g0:g1 + 1] + np.interp(np.arange(g0, g1 + 1), [g0, g1], [d0, d1])
                if guide is None and abs(pb[g1] - pa[g0]) > tol:
                    continue
                cov = ca | cb
                pr = np.where(np.isfinite(pa), pa, pb)
                pr[g0:g1 + 1] = path
                jj = np.clip(np.searchsorted(offs, path), 0, len(offs) - 1)
                for k, r in enumerate(range(g0, g1 + 1)):
                    lo_, hi_ = max(jj[k] - 5, 0), min(jj[k] + 6, len(offs))
                    if cand[r, lo_:hi_].any():
                        cov[r] = True
                w20 = int(20.0 / DS)
                solid = ca[max(0, g0 - w20):g0 + 1].mean() > 0.5 and cb[g1:g1 + w20].mean() > 0.5
                if solid:
                    cov[g0:g1 + 1] = True              # a solid line worn away here: still a line
                chains[ia] = (cov, pr)
                del chains[ib]
                changed = True
                break
            if changed:
                break
    return chains


def _runs(mask):
    """[(i0, i1)] index runs (inclusive start, exclusive end) where mask is True."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _periodic(runs, lo=5.0, hi=16.0, tol=0.8):
    """Are these paint runs [(start, end) in m] dashes: about the same length, repeating at one period
    (missing dashes allowed)? Worn stretches of a solid line break up at random instead."""
    if len(runs) < 3:
        return False
    starts = np.sort(np.array([a for a, _ in runs]))
    lens = np.array([b - a for a, b in runs])
    if np.std(lens) > 0.5 * np.mean(lens):
        return False
    d = np.diff(starts)
    d1 = d[(d >= lo) & (d <= hi)]
    if len(d1) < 2:
        return False
    P = float(np.median(d1))
    k = np.maximum(np.round(d / P), 1)
    return float((np.abs(d - k * P) <= tol).mean()) >= 0.8


def _classify(name, s, has, seen, pos, yel, cl, S, hls, hrs, mid, track=None):
    """Style per window -> pieces of one style -> a Line (or None when nothing is painted in this band)."""
    n = len(s)
    if has.sum() * DS < 10.0:
        return None
    # close tiny gaps inside a painted stretch (blur, small cracks): 0.5 m
    painted = ndi.binary_closing(has, structure=np.ones(3, bool)) | has
    w = max(4, int(WIN / DS))
    # a dashed line as a whole: its dashes repeat at one period (some missed on the photo)
    # (only the short runs: a centre line can be solid in no-passing zones and dashed elsewhere)
    short = [(a * DS, b * DS) for a, b in _runs(painted) if 0.5 <= (b - a) * DS < 6.0]
    dashed_line = len(short) >= 6 and _periodic(short) and \
        sum(b - a for a, b in short) >= 0.15 * painted.sum() * DS
    styles = []
    for i0 in range(0, n, w):
        sl = slice(i0, min(n, i0 + w))
        sn = seen[sl].sum()
        if track is not None and np.isfinite(track[sl]).mean() < 0.5:
            styles.append("none")                       # outside this line's stretch
            continue
        if sn < 0.3 * (sl.stop - sl.start):
            styles.append("?")
            continue
        share = painted[sl][seen[sl]].mean()
        # dashes: look at 48 m around the window (a 24 m window holds only two)
        wide = slice(max(0, i0 - w // 2), min(n, i0 + w + w // 2))
        runs = [(a, b) for a, b in _runs(painted[wide]) if (b - a) * DS >= 0.5]
        lens = np.array([(b - a) * DS for a, b in runs]) if runs else np.zeros(0)
        if share > 0.7:
            styles.append("solid")
        elif len(lens) and np.median(lens) < 6.0 and (
                (dashed_line and len(runs) >= 1) or
                _periodic([(a * DS, b * DS) for a, b in runs if (b - a) * DS < 6.0])):
            styles.append("dashed")                     # regular dashes (some may be missed on the photo)
        elif share > 0.45:
            styles.append("solid")                      # a worn solid line
        else:
            styles.append("none")
    # unseen windows take the neighbours' style (previous, else next)
    for i, st in enumerate(styles):
        if st == "?":
            prev = next((styles[j] for j in range(i - 1, -1, -1) if styles[j] != "?"), None)
            nxt = next((styles[j] for j in range(i + 1, len(styles)) if styles[j] != "?"), None)
            styles[i] = prev or nxt or "none"
    if all(st == "none" for st in styles):
        return None
    # pieces of one style, short ones merged into the longer neighbour
    win_s = np.minimum(np.arange(len(styles)) * w * DS, s[-1])
    pieces = []
    for st, i in zip(styles, range(len(styles))):
        a, b = win_s[i], min(s[-1] + DS, win_s[i] + w * DS)
        if pieces and pieces[-1][0] == st:
            pieces[-1][2] = b
        else:
            pieces.append([st, a, b])
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for i, (st, a, b) in enumerate(pieces):
            if b - a < MIN_PIECE:
                j = i - 1 if i > 0 and (i == len(pieces) - 1 or pieces[i - 1][2] - pieces[i - 1][1] >=
                                        pieces[i + 1][2] - pieces[i + 1][1]) else i + 1
                lo_, hi_ = min(i, j), max(i, j)
                keep = pieces[j][0]
                pieces[lo_:hi_ + 1] = [[keep, pieces[lo_][1], pieces[hi_][2]]]
                changed = True
                break
    # a solid piece starts / ends where its paint does (not at a window boundary)
    for k, pc in enumerate(pieces):
        st, a, b = pc
        if st != "solid":
            continue
        rows_p = np.nonzero(painted[int(a / DS):int(min(n, b / DS))])[0]
        if not len(rows_p):
            continue
        if k == 0 or pieces[k - 1][0] == "none":
            pc[1] = a + rows_p[0] * DS
        if k == len(pieces) - 1 or pieces[k + 1][0] == "none":
            pc[2] = a + (rows_p[-1] + 1) * DS
    # colour: yellow when the paint is clearly yellower than the asphalt
    dy = float(np.nanmedian(yel)) if np.isfinite(yel).any() else 0.0
    colour = "yellow" if dy > 0.35 else "white"
    # lateral position: measured, smoothed, filled from the track; per centreline station
    good = np.isfinite(pos)
    base = track if track is not None else np.where(good, pos, np.nanmedian(pos))
    base = np.where(np.isfinite(base), base, np.nanmedian(base))
    o = np.where(good, pos, base)
    o = ndi.median_filter(o, size=int(8.0 / DS) | 1, mode="wrap" if cl.loop else "nearest")
    o = ndi.gaussian_filter1d(o, 4.0 / DS, mode="wrap" if cl.loop else "nearest")
    # paint stays on the asphalt (a line seen past the measured edge means the edge is a bit off)
    o = np.clip(o, -hrs + 0.1, hls - 0.1)
    off_st = np.interp(cl.s, s, o)
    # painted intervals
    out = []
    style_main = max(("solid", "dashed"), key=lambda k: sum(b - a for st, a, b in pieces if st == k))
    for st, a, b in pieces:
        if st == "solid":
            out.append((a, b))
        elif st == "dashed":
            out += _dash_train(s, painted, a, b)
    out = [(max(0.0, a), min(cl.length, b)) for a, b in out if b - a > 0.3]
    if not out:
        return None
    return Line(name, colour, style_main, out, off_st, LINE_W["centre" if name == "centre" else "edge"])


def _dash_train(s, painted, a, b):
    """Dashes from a to b with the measured length, period and phase (regular train fitted to the dashes
    seen, so they also continue under trees and cars)."""
    i0, i1 = int(a / DS), int(min(len(s), b / DS))
    runs = [(x + i0, y + i0) for x, y in _runs(painted[i0:i1]) if (y - x) * DS >= 0.5]
    if len(runs) < 2:
        return [(a, b)] if not runs else [(s[runs[0][0]], s[runs[0][1] - 1] + DS)]
    starts = np.array([s[x] for x, _ in runs])
    lens = np.array([(y - x) * DS for x, y in runs])
    gaps = np.diff(starts)
    period = float(np.median(gaps[(gaps > 4.0) & (gaps < 17.0)])) if ((gaps > 4.0) & (gaps < 17.0)).any() else 12.0
    dash = float(np.clip(np.median(lens), 1.0, period * 0.7))
    ph = np.angle(np.mean(np.exp(2j * np.pi * starts / period)))
    phase = (ph / (2 * np.pi)) * period % period
    first = phase + np.floor((a - phase) / period) * period
    out = []
    x = first
    while x < b:
        d0, d1 = max(x, a), min(x + dash, b)
        if d1 - d0 > 0.3:
            out.append((d0, d1))
        x += period
    return out


# ------------------------------------------------------------------ the old fixed layout
def standard(cl, hl, hr) -> Markings:
    """Edge lines 12-27 cm inside the asphalt and a dashed centre line (the layout used before, when the
    photo can't tell): white."""
    wmean = float(np.mean(hl + hr))
    period = max(wmean, 6.0)
    mid = (hl - hr) / 2
    full = [(0.0, cl.length)]
    dashes = [(x, min(x + period / 2, cl.length)) for x in np.arange(0.0, cl.length, period)]
    return Markings([
        Line("left", "white", "solid", full, hl - 0.195, 0.15),
        Line("right", "white", "solid", full, -hr + 0.195, 0.15),
        Line("centre", "white", "dashed", dashes, mid, 0.12),
    ], {"source": "standard"})


# ------------------------------------------------------------------ mesh
def meshes(mk: Markings, cl, hl, hr, camber, lift=0.015, step=0.5):
    """Thin strips just above the road surface (visual only): materials road_paint / road_paint_yellow."""
    from .meshes import MeshBuilder

    parts = []
    S = np.concatenate([cl.s, [cl.length]]) if cl.loop else cl.s
    builders = {}
    for ln in mk.lines:
        mat = "road_paint_yellow" if ln.colour == "yellow" else "road_paint"
        mb = builders.setdefault(mat, MeshBuilder("ROAD_LINES" if mat == "road_paint" else "ROAD_LINES_Y", mat))
        offs = [0.0] if ln.style != "double" else [-(DOUBLE_GAP + ln.width) / 2, (DOUBLE_GAP + ln.width) / 2]
        for a, b in ln.pieces:
            ss = np.unique(np.concatenate([np.arange(a, b, step), [b]]))
            if len(ss) < 2:
                continue
            for d in offs:
                V, F = _ribbon(cl, S, hl, hr, camber, ss, ln.offset, d, ln.width, lift)
                mb.add(V, F, np.zeros((len(V), 2)))
    for mb in builders.values():
        parts += mb.done()
    return parts


def _ribbon(cl, S, hl, hr, camber, ss, offset, d, width, lift):
    loop = cl.loop
    sq = ss % cl.length if loop else ss
    P = np.vstack([cl.p, cl.p[:1]]) if loop else cl.p
    N = np.vstack([cl.n, cl.n[:1]]) if loop else cl.n
    p = np.column_stack([np.interp(sq, S, P[:, 0]), np.interp(sq, S, P[:, 1])])
    n = np.column_stack([np.interp(sq, S, N[:, 0]), np.interp(sq, S, N[:, 1])])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    z = _at(sq, S, cl.z, loop)
    hlq, hrq = _at(sq, S, hl, loop), _at(sq, S, hr, loop)
    o = _at(sq, S, offset, loop) + d
    rows = []
    for e in (o + width / 2, o - width / 2):
        tc = (hlq - e) / np.maximum(hlq + hrq, 1e-6)
        zz = z + camber * (1 - (2 * tc - 1) ** 2) + lift
        xy = p + n * e[:, None]
        rows.append(np.column_stack([xy, zz]))
    m = len(ss)
    V = np.vstack([rows[0], rows[1]])
    i = np.arange(m - 1)
    # counter-clockwise seen from above (normal up): left edge row first, then the right edge row
    F = np.vstack([np.column_stack([i, m + i + 1, i + 1]), np.column_stack([i, m + i, m + i + 1])])
    return V, F


def paint_texture(colour="white", size=16):
    rng = np.random.default_rng(3 if colour == "white" else 4)
    from PIL import Image

    a = np.clip(rng.normal(0, 6, (size, size)), -20, 20)
    base = (228, 228, 221) if colour == "white" else (214, 170, 52)
    img = np.stack([np.clip(c + a, 0, 255) for c in base], -1).astype(np.uint8)
    return Image.fromarray(img, "RGB")


def overlay(mk: Markings, cl, proj, step=1.0):
    """For the map: [{colour, style, side, segments: [[[lat, lon], ...], ...]}]."""
    out = []
    S = np.concatenate([cl.s, [cl.length]]) if cl.loop else cl.s
    P = np.vstack([cl.p, cl.p[:1]]) if cl.loop else cl.p
    N = np.vstack([cl.n, cl.n[:1]]) if cl.loop else cl.n
    for ln in mk.lines:
        segs = []
        for a, b in ln.pieces:
            ss = np.unique(np.concatenate([np.arange(a, b, step), [b]]))
            sq = ss % cl.length if cl.loop else ss
            p = np.column_stack([np.interp(sq, S, P[:, 0]), np.interp(sq, S, P[:, 1])])
            n = np.column_stack([np.interp(sq, S, N[:, 0]), np.interp(sq, S, N[:, 1])])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            q = p + n * _at(sq, S, ln.offset, cl.loop)[:, None]
            la, lo = proj.to_latlon(q[:, 0], q[:, 1])
            segs.append(np.column_stack([la, lo]).round(7).tolist())
        out.append({"side": ln.side, "colour": ln.colour, "style": ln.style, "segments": segs})
    return out

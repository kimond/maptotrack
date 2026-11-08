"""Exact road edges from the aerial photo.

Method
  1. Rectify: sample the photo on a strip that follows the centreline
     (one row per 2 m station, one column per 0.1 m of lateral offset).
  2. Asphalt model: CIE-Lab colour + fine texture of the road centre, learned locally
     (sliding 80 m window) so worn/patched/new asphalt all work. Each strip pixel gets a
     "road-likeness" from its Mahalanobis distance to that model. Thin bright paint lines
     count as road; sand, dry grass, gravel and concrete differ in colour and/or texture.
  3. Edge tracing: for each side, dynamic programming (Viterbi) finds the edge offset
     per station that maximises the road->non-road contrast, prefers the innermost strong
     edge, and moves smoothly from station to station. Occlusions (cars, tree shadows,
     side streets, driveways) are bridged by the smoothness term instead of creating spikes.
  4. A second pass re-learns the asphalt model from the whole detected road surface.

Returns half widths left/right of the centreline (so an off-centre OSM line is corrected)
and a per-station confidence.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

STEP = 0.1          # lateral sampling (m)
REACH = 15.0        # search up to 15 m each side
MIN_HALF = 1.2
LW = 0.9           # weight of lightness vs colour (asphalt is mostly told apart by brightness)


def rgb_to_lab(rgb):
    c = np.asarray(rgb, np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], -1)


def _rectify(pm, cl):
    offs = np.arange(-REACH, REACH + STEP / 2, STEP)
    N = len(cl.p)
    X = cl.p[:, None, 0] + cl.n[:, None, 0] * offs[None, :]
    Y = cl.p[:, None, 1] + cl.n[:, None, 1] * offs[None, :]
    rgb, ok = pm.sample(X.ravel(), Y.ravel(), order=1)
    return offs, rgb.reshape(N, len(offs), 3), ok.reshape(N, len(offs))


def _features(lab):
    L = lab[..., 0]
    # fine texture across the strip (~0.7 m window): gravel / grass are rougher than asphalt
    m1 = ndi.uniform_filter1d(L, 7, axis=1)
    m2 = ndi.uniform_filter1d(L * L, 7, axis=1)
    tex = np.sqrt(np.clip(m2 - m1 * m1, 0, None))
    # L is down-weighted so tree shadows on the road are still "road"
    return np.stack([L * LW, lab[..., 1], lab[..., 2], tex * 0.8], -1)


def _window_stats(F, w, N, win, loop):
    """Per-station mean/covariance of features F (N,M,4) weighted by w (N,M), over +-win stations."""
    D = F.shape[-1]
    s0 = (w).sum(1)
    s1 = (w[..., None] * F).sum(1)
    s2 = (w[..., None, None] * F[..., :, None] * F[..., None, :]).sum(1)
    mode = "wrap" if loop else "nearest"
    k = 2 * win + 1
    S0 = ndi.uniform_filter1d(s0, k, mode=mode) * k
    S1 = ndi.uniform_filter1d(s1, k, axis=0, mode=mode) * k
    S2 = ndi.uniform_filter1d(s2.reshape(N, -1), k, axis=0, mode=mode).reshape(N, D, D) * k
    S0 = np.maximum(S0, 1e-6)
    mean = S1 / S0[:, None]
    cov = S2 / S0[:, None, None] - mean[:, :, None] * mean[:, None, :]
    cov += np.eye(D) * np.array([1.0, 0.6, 0.6, 0.5])  # floor: avoid over-confident models
    return mean, cov, S0


def _roadness(F, mean, cov):
    inv = np.linalg.inv(cov)
    d = F - mean[:, None, :]
    m2 = np.einsum("nmi,nij,nmj->nm", d, inv, d)
    dist = np.sqrt(np.maximum(m2, 0))
    return np.clip(1.0 - (dist - 2.5) / 3.0, 0.0, 1.0)


def _paint(lab, road_L, road_chroma):
    """Thin bright stripes no more colourful than the asphalt (lane / edge markings, often blurred
    to a faint light line at 20 cm/pixel) - count as road."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    bright = (L > road_L[:, None] + 5) & (np.hypot(a, b) < road_chroma[:, None] + 3)
    wide = ndi.binary_opening(bright, structure=np.ones((1, 7), bool))   # >= 0.7 m wide = not paint
    return bright & ~wide


def _viterbi(cost, max_jump=2, lam=0.12, loop=False):
    """cost (N, M) -> state per row minimising sum(cost) + lam*|jump|, jumps <= max_jump."""
    N, M = cost.shape
    if loop:  # unroll 3 laps and keep the middle one
        res = _viterbi(np.vstack([cost, cost, cost]), max_jump, lam, False)
        return res[N:2 * N]
    acc = cost[0].copy()
    back = np.zeros((N, M), np.int32)
    idx = np.arange(M)
    for i in range(1, N):
        best = np.full(M, np.inf)
        arg = np.zeros(M, np.int32)
        for dj in range(-max_jump, max_jump + 1):
            src = idx - dj
            valid = (src >= 0) & (src < M)
            cand = np.full(M, np.inf)
            cand[valid] = acc[src[valid]] + lam * abs(dj)
            better = cand < best
            best[better] = cand[better]
            arg[better] = src[better]
        acc = best + cost[i]
        back[i] = arg
    path = np.zeros(N, np.int32)
    path[-1] = int(np.argmin(acc))
    for i in range(N - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def _trace_side(R, offs, sign, valid, prior_half):
    """Edge offsets (m, positive) for one side. R: roadness (N, M)."""
    mid = len(offs) // 2
    if sign > 0:
        r = R[:, mid:]
        v = valid[:, mid:]
    else:
        r = R[:, mid::-1]
        v = valid[:, mid::-1]
    N, M = r.shape
    o = np.arange(M) * STEP
    w = 5  # 0.5 m windows either side of the candidate edge
    c = np.concatenate([np.zeros((N, 1)), np.cumsum(r, 1)], 1)
    inside = np.full((N, M), np.nan)
    outside = np.full((N, M), np.nan)
    j = np.arange(w, M - w)
    inside[:, j] = (c[:, j] - c[:, j - w]) / w
    outside[:, j] = (c[:, j + w] - c[:, j]) / w
    contrast = np.nan_to_num(inside - outside, nan=-1.0)
    # non-road fraction between 1 m from centre and the candidate: prefer the innermost real edge
    nr = np.concatenate([np.zeros((N, 1)), np.cumsum(1 - r, 1)], 1)
    j0 = int(0.8 / STEP)
    gapfrac = np.zeros((N, M))
    jj = np.arange(M)
    span = np.maximum(jj - j0, 1)
    gapfrac[:, :] = np.clip((nr[:, np.maximum(jj, j0)] - nr[:, j0][:, None]) / span, 0, 1)
    cost = -contrast + 1.2 * gapfrac + 0.15 * np.abs(o - prior_half)[None, :] / 10.0
    cost[:, : int(MIN_HALF / STEP)] = 5.0
    cost[~v] = 0.5
    path = _viterbi(cost, loop=False)
    edge = o[path]
    conf = np.clip(contrast[np.arange(N), path], 0, 1)
    return edge, conf, cost


class _Line:
    def __init__(self, p, n):
        self.p, self.n = p, n


def _seed_centre(R, offs, ok, loop, spacing):
    """Per station, the middle of the asphalt run nearest the OSM line (corrects off-centre OSM data)."""
    N, M = R.shape
    mid = M // 2
    road = (R > 0.5) & ok
    c = np.full(N, np.nan)
    lim = int(6.0 / STEP)
    for i in range(N):
        row = road[i]
        lab_, n = ndi.label(row)
        if n == 0:
            continue
        best, best_score = None, -1e9
        for k in range(1, n + 1):
            idx = np.nonzero(lab_ == k)[0]
            if len(idx) < int(2.0 / STEP):
                continue
            dist = 0 if idx[0] <= mid <= idx[-1] else min(abs(idx[0] - mid), abs(idx[-1] - mid))
            if dist > lim:
                continue
            score = len(idx) * STEP - 2.0 * dist * STEP
            if score > best_score:
                best, best_score = idx, score
        if best is not None:
            c[i] = offs[(best[0] + best[-1]) // 2]
    good = np.isfinite(c)
    if good.mean() < 0.3:
        return np.zeros(N)
    idx = np.arange(N)
    if loop:
        g = np.nonzero(good)[0]
        c[~good] = np.interp(idx[~good], np.concatenate([g - N, g, g + N]), np.tile(c[g], 3))
    else:
        c[~good] = np.interp(idx[~good], idx[good], c[good])
    mode = "wrap" if loop else "nearest"
    c = ndi.median_filter(c, size=max(3, int(20 / spacing) | 1), mode=mode)
    return ndi.gaussian_filter1d(c, max(1.0, 8.0 / spacing), mode=mode)


def _model(F, w, N, win, loop, lab):
    mean, cov, _ = _window_stats(F, w, N, win, loop)
    R = _roadness(F, mean, cov)
    return np.maximum(R, _paint(lab, mean[:, 0] / LW, np.hypot(mean[:, 1], mean[:, 2])))


def _refine(F, offs, edge, sign, ok, contrast_out=None):
    """Sub-pixel edge: the 50 % point of the colour transition between the asphalt just inside
    (0.4-1.4 m) and the ground just outside (0.4-1.4 m), searched within +-0.6 m of the DP edge.
    Unbiased for blurred edges, whatever the contrast."""
    N, M, D = F.shape
    mid = M // 2
    out = edge.copy()
    k = np.arange(-6, 7)
    for i in range(N):
        j = mid + sign * int(round(edge[i] / STEP))
        ji = j - sign * np.arange(4, 15)
        jo = j + sign * np.arange(4, 15)
        if ji.min() < 0 or jo.min() < 0 or ji.max() >= M or jo.max() >= M:
            continue
        if not (ok[i, ji].all() and ok[i, jo].all()):
            continue
        fin, fout = F[i, ji].mean(0), F[i, jo].mean(0)
        d = fin - fout
        dd = float(d @ d)
        if contrast_out is not None:
            contrast_out[i] = np.sqrt(dd)
        if dd < 4.0:          # no real contrast here - keep the DP edge
            continue
        js = j + sign * k
        t = (F[i, js] - fout) @ d / dd        # 1 = asphalt, 0 = outside
        below = np.nonzero(t < 0.5)[0]
        if len(below) == 0 or below[0] == 0:
            continue
        b = below[0]
        t0, t1 = t[b - 1], t[b]
        frac = (t0 - 0.5) / max(t0 - t1, 1e-6)
        pos = (k[b - 1] + frac) * STEP
        out[i] = edge[i] + pos
    return out


def measure_road_edges(pm, cl, default_w: float, min_w=3.0, max_w=24.0):
    """Returns (half_left, half_right, info). info['confidence'] per station in [0,1]."""
    N = len(cl.p)
    win = max(5, int(40 / cl.spacing))
    offs, rgb, ok = _rectify(pm, cl)
    lab = rgb_to_lab(rgb)
    F = _features(lab)
    core = np.abs(offs) <= 1.2
    centre_valid = ok[:, core].all(1)
    if centre_valid.mean() < 0.3:
        return _fallback(N, default_w, "no photo along the track")

    # pass 0: global asphalt model from the OSM centre line (robust to cars / off-centre spots)
    Fc = F[:, core][centre_valid].reshape(-1, F.shape[-1])
    med = np.median(Fc, axis=0)
    mad = np.median(np.abs(Fc - med), axis=0) * 1.4826 + 0.5
    good = (np.abs(F[:, core] - med) / mad < 3.0).all(-1) & ok[:, core]
    w = np.zeros(ok.shape, np.float32)
    w[:, core] = good
    R0 = _model(F, w, N, win, cl.loop, lab)
    centre = _seed_centre(R0, offs, ok, cl.loop, cl.spacing)

    # re-sample along the corrected centre line
    line = _Line(cl.p + cl.n * centre[:, None], cl.n)
    offs, rgb, ok = _rectify(pm, line)
    lab = rgb_to_lab(rgb)
    F = _features(lab)
    core = np.abs(offs) <= 1.0
    w = np.zeros(ok.shape, np.float32)
    w[:, core] = ok[:, core] & (np.abs(F[:, core] - med) / mad < 3.5).all(-1)
    R = _model(F, w, N, win, cl.loop, lab)
    R[~ok] = 0.5
    el, cl_conf, _ = _trace_side(R, offs, +1, ok, default_w / 2)
    er, cr_conf, _ = _trace_side(R, offs, -1, ok, default_w / 2)

    # pass 2: re-learn from the whole detected road surface, trace again
    inside = (offs[None, :] < el[:, None] - 0.4) & (offs[None, :] > -er[:, None] + 0.4) & ok
    R = _model(F, (inside & (R > 0.3)).astype(np.float32), N, win, cl.loop, lab)
    R[~ok] = 0.5
    el, cl_conf, _ = _trace_side(R, offs, +1, ok, float(np.median(el)))
    er, cr_conf, _ = _trace_side(R, offs, -1, ok, float(np.median(er)))

    kl, kr = np.zeros(N), np.zeros(N)
    el = _refine(F, offs, el, +1, ok, kl)
    er = _refine(F, offs, er, -1, ok, kr)
    # confidence: how visible the asphalt/ground boundary is on the photo (colour step across the edge)
    conf = np.clip((np.minimum(kl, kr) - 3.0) / 6.0, 0, 1)
    conf = ndi.minimum_filter1d(ndi.maximum_filter1d(conf, 3), 3)
    mode = "wrap" if cl.loop else "nearest"
    el = ndi.gaussian_filter1d(el, 1.5, mode=mode)
    er = ndi.gaussian_filter1d(er, 1.5, mode=mode)
    hl, hr = centre + el, er - centre          # back to the original centre line
    tot = hl + hr
    scale = np.clip(tot, min_w, max_w) / np.maximum(tot, 1e-6)
    hl, hr = hl * scale, hr * scale
    hl, hr = _limit(cl, hl, hr, min_w)
    wdt = hl + hr
    good_frac = float((conf > 0.3).mean())
    info = {
        "used": True, "method": "photo-edges",
        "min": round(float(wdt.min()), 1), "median": round(float(np.median(wdt)), 1),
        "max": round(float(wdt.max()), 1),
        "confident_fraction": round(good_frac, 2),
        "centre_shift_median": round(float(np.median(np.abs(centre))), 2),
        "confidence": conf.round(2),
    }
    if good_frac < 0.2:
        out = _fallback(N, default_w, f"edges unclear on the photo ({good_frac * 100:.0f}% confident)")
        out[2]["confident_fraction"] = round(good_frac, 2)
        return out
    return hl, hr, info


def _limit(cl, hl, hr, min_w=3.0):
    """Keep the inside of tight corners from folding, and the road at least min_w wide."""
    Rr = np.abs(cl.radius)
    left_inside = cl.radius > 0
    lim = np.maximum(Rr - 1.0, MIN_HALF)
    hl = np.where(left_inside, np.minimum(hl, lim), hl)
    hr = np.where(~left_inside, np.minimum(hr, lim), hr)
    return np.maximum(hl, -hr + min_w), np.maximum(hr, -hl + min_w)


def smooth_widths(cl, hl, hr, conf, length_m, min_w=3.0):
    """Width smoothing along the track: the width and the road's centre offset are averaged over about
    `length_m` metres, so the road edges run evenly instead of following every bump of the measurement.

    1. a running median over length_m removes short spikes (a driveway or side street read as road, a car
       or a tree shadow read as verge) without rounding off real changes of width;
    2. a Gaussian (sigma = length_m / 2.5) weighted by the edge confidence: stretches where the photo edges are
       clear count most, unclear ones take their width from their neighbours.
    Manual width settings are applied after this, so they stay exact."""
    if length_m <= 0 or len(hl) < 5:
        return hl, hr
    sp = float(np.median(np.diff(cl.s))) if len(cl.s) > 1 else 2.0
    mode = "wrap" if cl.loop else "nearest"
    half = (hl + hr) / 2
    centre = (hl - hr) / 2
    k = max(3, int(round(length_m / sp)) | 1)
    k = min(k, len(hl))
    half = ndi.median_filter(half, size=k, mode=mode)
    centre = ndi.median_filter(centre, size=k, mode=mode)
    sig = max(0.5, length_m / 2.5 / sp)
    w = 0.15 + np.clip(np.asarray(conf, float), 0, 1) if conf is not None and len(conf) == len(hl) else np.ones(len(hl))
    den = ndi.gaussian_filter1d(w, sig, mode=mode)
    half = ndi.gaussian_filter1d(half * w, sig, mode=mode) / den
    centre = ndi.gaussian_filter1d(centre * w, sig, mode=mode) / den
    return _limit(cl, centre + half, half - centre, min_w)


def _fallback(N, default_w, why):
    return (np.full(N, default_w / 2), np.full(N, default_w / 2),
            {"used": False, "reason": why, "confidence": np.zeros(N)})


def apply_overrides(cl, hl, hr, overrides):
    """overrides: [{'s': metres along track, 'width': m, 'span': m}] -> blended into hl/hr
    keeping the measured centre (midpoint of the edges)."""
    if not overrides:
        return hl, hr
    hl, hr = hl.copy(), hr.copy()
    for ov in overrides:
        s0, wd = float(ov["s"]), float(ov["width"])
        span = max(float(ov.get("span", 40.0)), 4.0)
        ds = np.abs(cl.s - s0)
        if cl.loop:
            ds = np.minimum(ds, cl.length - ds)
        wgt = np.clip(1.0 - (ds - span / 2) / 10.0, 0, 1)   # flat over span, 10 m blend
        centre = (hl - hr) / 2
        half = (hl + hr) / 2 * (1 - wgt) + wd / 2 * wgt
        hl, hr = centre + half, half - centre
    return np.maximum(hl, 0.5), np.maximum(hr, 0.5)

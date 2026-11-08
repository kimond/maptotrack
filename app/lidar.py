"""Quebec LiDAR derived products (1 m): terrain (MNT) and canopy height (MHC).

Source: "Lidar - Modèles numériques (terrain, canopée, pente, courbe de niveau)", Gouvernement du
Québec (MRNF), CC BY 4.0, via Données Québec. Data is distributed per 1:20 000 map sheet; each
product is downloaded once into cache/lidar/ (several hundred MB per sheet).
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
from scipy.ndimage import map_coordinates

from . import config

INDEX_URL = ("https://diffusion.mffp.gouv.qc.ca/Diffusion/DonneeGratuite/Foret/IMAGERIE/"
             "Produits_derives_LiDAR/Produit_derive_lidar/03-Telechargement/URL_Lidar.geojson")
ATTRIBUTION = "LiDAR terrain/canopy models (c) Gouvernement du Quebec (MRNF), CC BY 4.0"
LIDAR_DIR = config.CACHE_DIR / "lidar"


class LidarError(Exception):
    pass


def _session():
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    return s


def load_index(log=print):
    path = LIDAR_DIR / "URL_Lidar.geojson"
    if not path.exists() or time.time() - path.stat().st_mtime > 30 * 86400:
        log("Downloading Quebec LiDAR sheet index ...")
        try:
            r = _session().get(INDEX_URL, timeout=120)
            r.raise_for_status()
        except Exception as exc:
            if path.exists():
                log(f"  index refresh failed ({exc}) - using cached copy")
                return json.loads(path.read_text(encoding="utf-8"))
            raise LidarError(f"Could not download the LiDAR index: {exc}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return json.loads(path.read_text(encoding="utf-8"))


def sheets_for(bbox, log=print):
    """bbox = (s, w, n, e) -> list of sheet property dicts intersecting it."""
    from shapely.geometry import box, shape

    idx = load_index(log)
    q = box(bbox[1], bbox[0], bbox[3], bbox[2])
    out = []
    for f in idx.get("features", []):
        try:
            g = shape(f["geometry"])
        except Exception:
            continue
        if g.intersects(q):
            out.append(f.get("properties", {}))
    return out


def _download(url, dest: Path, log):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _session().get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        log(f"  downloading {dest.name}" + (f" ({total / 1e6:.0f} MB, one time only)" if total else "") + " ...")
        done, next_report = 0, 0.1
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done / total >= next_report:
                    log(f"    {done / total * 100:.0f}%")
                    next_report += 0.1
    tmp.replace(dest)
    return dest


def product_files(sheet: dict, product: str, log=print):
    """Download one product ('MNT' or 'MHC') for a sheet; return the GeoTIFF paths."""
    url = (sheet.get(product) or "").strip()
    if not url:
        return []
    folder = LIDAR_DIR / str(sheet.get("Feuillet20K", "sheet")) / product
    name = url.rstrip("/").split("/")[-1].split("?")[0] or f"{product}.tif"
    path = _download(url, folder / name, log)
    if path.suffix.lower() == ".zip":
        out = folder / "unzipped"
        if not out.exists():
            log(f"  extracting {path.name} ...")
            with zipfile.ZipFile(path) as z:
                z.extractall(out)
        return sorted([p for p in out.rglob("*") if p.suffix.lower() in (".tif", ".tiff")])
    return [path]


class LidarRaster:
    """Windowed read of one or more GeoTIFFs around a lat/lon bbox; bilinear sampling, NaN outside."""

    def __init__(self, paths, bbox, pad_deg=0.002):
        import rasterio
        from pyproj import Transformer
        from rasterio.windows import from_bounds

        s, w, n, e = bbox
        self.parts = []
        for p in paths:
            ds = rasterio.open(p)
            tr = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
            xs, ys = tr.transform([w - pad_deg, e + pad_deg, w - pad_deg, e + pad_deg],
                                  [s - pad_deg, s - pad_deg, n + pad_deg, n + pad_deg])
            l, b, r, t = max(min(xs), ds.bounds.left), max(min(ys), ds.bounds.bottom), \
                min(max(xs), ds.bounds.right), min(max(ys), ds.bounds.top)
            if l >= r or b >= t:
                ds.close()
                continue
            win = from_bounds(l, b, r, t, ds.transform).round_offsets().round_lengths()
            arr = ds.read(1, window=win, masked=True).astype(np.float32).filled(np.nan)
            arr[(arr < -500) | (arr > 9000)] = np.nan
            self.parts.append((arr, ~ds.window_transform(win), tr))
            ds.close()
        if not self.parts:
            raise LidarError("LiDAR files don't cover the track area.")

    def sample(self, lat, lon):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        out = np.full(lat.shape, np.nan)
        for arr, inv, tr in self.parts:
            todo = np.isnan(out)
            if not todo.any():
                break
            X, Y = tr.transform(lon[todo], lat[todo])
            col, row = inv * (np.asarray(X), np.asarray(Y))
            col, row = np.asarray(col) - 0.5, np.asarray(row) - 0.5
            inside = (col >= 0) & (row >= 0) & (col <= arr.shape[1] - 1) & (row <= arr.shape[0] - 1)
            nanmask = np.isnan(arr)
            val = map_coordinates(np.nan_to_num(arr), [row, col], order=1, mode="nearest")
            bad = map_coordinates(nanmask.astype(np.float32), [row, col], order=1, mode="nearest") > 0.01
            val = np.where(inside & ~bad, val, np.nan)
            sub = out[todo]
            sub[np.isfinite(val)] = val[np.isfinite(val)]
            out[todo] = sub
        return out


class CompositeDEM:
    """LiDAR terrain where available, the coarse DEM elsewhere (shifted to match the LiDAR datum)."""

    def __init__(self, lidar: LidarRaster, fallback, probe_lat, probe_lon):
        self.lidar, self.fallback = lidar, fallback
        a = lidar.sample(probe_lat, probe_lon)
        b = np.asarray(fallback.sample(probe_lat, probe_lon), float)
        ok = np.isfinite(a)
        self.offset = float(np.median(a[ok] - b[ok])) if ok.sum() > 10 else 0.0
        self.coverage = float(ok.mean()) if len(ok) else 0.0

    def sample(self, lat, lon):
        a = self.lidar.sample(lat, lon)
        bad = ~np.isfinite(a)
        if bad.any():
            b = np.asarray(self.fallback.sample(np.asarray(lat)[bad], np.asarray(lon)[bad]), float)
            a[bad] = b + self.offset
        return a


def open_products(bbox, products=("MNT", "MHC"), log=print):
    """Return {product: LidarRaster} for the sheets covering bbox (downloads on first use)."""
    sheets = sheets_for(bbox, log)
    if not sheets:
        raise LidarError("No Quebec LiDAR coverage for this area.")
    log(f"LiDAR: {len(sheets)} map sheet(s): " + ", ".join(str(s.get("Feuillet20K")) for s in sheets))
    from .parallel import io_map

    # all (product, sheet) downloads at the same time (each is a single large file)
    jobs = [(prod, sh) for prod in products for sh in sheets]

    def fetch(job):
        prod, sh = job
        try:
            return prod, product_files(sh, prod, log)
        except Exception as exc:
            log(f"  {prod} for sheet {sh.get('Feuillet20K')} failed: {exc}")
            return prod, []

    files = {prod: [] for prod in products}
    for prod, fs in io_map(fetch, jobs, threads=4):
        files[prod] += fs

    def open_one(prod):
        if not files[prod]:
            return prod, None
        try:
            return prod, LidarRaster(files[prod], bbox)
        except Exception as exc:
            log(f"  {prod}: {exc}")
            return prod, None

    return {prod: r for prod, r in io_map(open_one, products) if r is not None}


def chm_sampler(chm: LidarRaster, proj):
    """f(x, y, radius) -> max canopy height within ~radius of local points (NaN if no data)."""
    def f(x, y, r):
        x, y, r = np.asarray(x), np.asarray(y), np.asarray(r)
        best = np.full(x.shape, np.nan)
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7)]:
            lat, lon = proj.to_latlon(x + dx * r * 0.6, y + dy * r * 0.6)
            v = chm.sample(lat, lon)
            best = np.fmax(best, v)
        return best
    return f


def chm_treetops(chm: LidarRaster, proj, bounds, res=1.0, min_h=0.8):
    """Local maxima of the canopy height model -> (x, y, h)."""
    from scipy import ndimage as ndi

    x0, y0, x1, y1 = bounds
    xs = np.arange(x0 + res / 2, x1, res)
    ys = np.arange(y1 - res / 2, y0, -res)
    X, Y = np.meshgrid(xs, ys)
    lat, lon = proj.to_latlon(X, Y)
    h = chm.sample(lat.ravel(), lon.ravel()).reshape(X.shape)
    h = np.nan_to_num(h, nan=0.0)
    hs = ndi.gaussian_filter(h, 0.6)
    hmax = ndi.maximum_filter(h, size=3)        # height of the crown around the peak (not blurred)
    # window grows with height: small for bushes, ~5 m for big trees
    peaks = np.zeros_like(hs, bool)
    for lo, hi, win in ((min_h, 4, 3), (4, 12, 4), (12, 99, 6)):
        band = (hmax >= lo) & (hmax < hi)
        peaks |= band & (hs == ndi.maximum_filter(hs, size=win)) & (hs > 0.3)
    r, c = np.nonzero(peaks)
    return X[r, c], Y[r, c], hmax[r, c]

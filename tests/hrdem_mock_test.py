"""Offline test of the NRCan HRDEM path with a synthetic mosaic tile (EPSG:3979, like the real COGs).

  python tests/hrdem_mock_test.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from app import config, hrdem, lidar


class Flat:
    def sample(self, lat, lon):
        return np.full(np.shape(lat), 40.0)


def main():
    lat0, lon0 = 45.50, -73.57                     # Montreal
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3979", always_xy=True)
    cx, cy = tr.transform(lon0, lat0)
    size = 3000
    west, north = round(cx) - size / 2, round(cy) + size / 2
    X, Y = np.meshgrid(west + np.arange(size) + 0.5, north - np.arange(size) - 0.5)
    dtm = (60 + 0.02 * (X - cx) + 0.01 * (Y - cy)).astype(np.float32)
    dtm[:, :400] = -32767.0                        # a gap in the mosaic (west edge): nodata
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tile = d / "9_2-mosaic-1m-dtm.tif"
        with rasterio.open(tile, "w", driver="GTiff", width=size, height=size, count=1, dtype="float32",
                           crs="EPSG:3979", transform=from_origin(west, north, 1.0, 1.0), nodata=-32767.0,
                           tiled=True, blockxsize=256, blockysize=256, compress="deflate") as ds:
            ds.write(dtm, 1)
        hrdem.HRDEM_DIR = d / "cache"
        config.OFFLINE = False
        calls = []
        hrdem.find_cogs = lambda bbox, res=1, log=print: calls.append(res) or [str(tile)]
        bbox = (lat0 - 0.006, lon0 - 0.008, lat0 + 0.006, lon0 + 0.008)   # ~1.3 x 1.2 km
        dem = hrdem.open_dem(bbox, log=print)
        assert calls == [1]
        cached = list(hrdem.HRDEM_DIR.glob("*.tif"))
        assert len(cached) == 1 and cached[0].stat().st_size < tile.stat().st_size
        lats = lat0 + np.array([0.0, 0.003, -0.004])
        lons = lon0 + np.array([0.0, 0.005, -0.002])
        xs, ys = tr.transform(lons, lats)
        want = 60 + 0.02 * (np.asarray(xs) - cx) + 0.01 * (np.asarray(ys) - cy)
        got = dem.sample(lats, lons)
        assert np.allclose(got, want, atol=0.05), (got, want)
        # a gap in the mosaic: NaN from HRDEM, filled by the coarse DEM (+ its offset) in the composite
        glon, glat = Transformer.from_crs("EPSG:3979", "EPSG:4326", always_xy=True).transform(west + 200.0, cy)
        wide = hrdem.open_dem((glat - 0.002, glon - 0.002, lat0 + 0.002, lon0 + 0.002), log=print)
        assert not np.isfinite(wide.sample(np.array([glat]), np.array([glon])))[0]
        cw = lidar.CompositeDEM(wide, Flat(), lats, lons)
        assert abs(cw.sample(np.array([glat]), np.array([glon]))[0] - (40.0 + cw.offset)) < 1e-6
        calls.clear()
        comp = lidar.CompositeDEM(dem, Flat(), lats, lons)
        assert comp.coverage == 1.0
        # second run: the saved window, no catalogue search
        dem2 = hrdem.open_dem((lat0 - 0.004, lon0 - 0.005, lat0 + 0.004, lon0 + 0.005), log=print)
        assert calls == [], calls
        assert np.allclose(dem2.sample(lats[:1], lons[:1]), want[:1], atol=0.05)
        # an area too big for 1 m is read at 2 m
        res, npx = hrdem._choose_res((45.4, -73.7, 45.6, -73.4))
        assert res == 2, npx
        assert hrdem.in_canada(bbox) and not hrdem.in_canada((48.8, 2.3, 48.9, 2.4))
        # STAC asset hrefs, full or relative
        assert hrdem._href({"collection": "hrdem-mosaic-1m", "assets": {"dtm": {"href": "9_3-mosaic-1m-dtm.tif"}}},
                           "dtm") == f"{hrdem.S3_BASE}/hrdem-mosaic-1m/9_3-mosaic-1m-dtm.tif"
        assert hrdem._href({"assets": {"dtm": {"href": "https://x/y.tif"}}}, "dtm") == "https://x/y.tif"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

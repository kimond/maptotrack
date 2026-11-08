"""Area mode (every street in a rectangle, free roam) - no network.

  MAPTOTRACK_OFFLINE=1 python tests/area_test.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MAPTOTRACK_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from shapely.geometry import Point

from app import ac_export, network
from app.geo import LocalProjection

PROJ = LocalProjection(48.5, -71.25)


class SlopeDEM:
    """A hillside (5 % east, 2 % north) with 30 cm of noise: the streets must come out smooth and meet level."""

    def sample(self, lat, lon):
        x, y = PROJ.to_xy(np.asarray(lat, float), np.asarray(lon, float))
        noise = 0.3 * np.sin(x * 1.7) * np.cos(y * 2.3)
        return 100.0 + 0.05 * x + 0.02 * y + noise


def way(wid, hw, width, pts, **kw):
    xy = np.asarray(pts, float)
    la, lo = PROJ.to_latlon(xy[:, 0], xy[:, 1])
    r = {"id": wid, "highway": hw, "width": width, "coords": list(zip(la.tolist(), lo.tolist())), "oneway": 0}
    r.update(kw)
    return r


def roads():
    ring = [(-120 + 30 * np.sin(a), 80 - 30 * np.cos(a)) for a in np.linspace(0, 2 * np.pi, 41)]
    return [
        # main street, split in two OSM ways at the cross (as mapped); its vertices are the junction nodes
        way("w1", "tertiary", 9.0, [(-200, 0), (-120, 0), (-60, 0), (0, 0)]),
        way("w2", "tertiary", 9.0, [(0, 0), (50, 0), (100, 0), (200, 0)]),
        way("w3", "residential", 6.5, [(0, -150), (0, -60), (0, 0), (0, 60), (0, 150)]),     # cross street
        way("w4", "residential", 6.5, [(100, 0), (100, 60), (100, 120)]),                # T, cul-de-sac
        way("w5", "residential", 6.5, [(-120, 0), (-120, 25), (-120, 50)]),              # T, to the ring
        way("w6", "residential", 6.5, ring),                                            # a crescent loop
        way("w7", "footway", 2.0, [(-50, -50), (50, -50)]),                              # not a street
        way("w8", "service", 4.5, [(150, 0), (150, -30)], service="driveway"),          # a driveway: the photo's job
    ]


def main():
    P = {"elev_smoothing": 25.0, "width_mode": "photo", "width_smoothing": 30.0, "camber": 0.06, "markings": "standard"}
    bounds = (-250.0, -200.0, 250.0, 200.0)
    turning = [{"lat": float(PROJ.to_latlon(100.0, 120.0)[0]), "lon": float(PROJ.to_latlon(100.0, 120.0)[1]),
                "diameter": None}]
    net = network.build(roads(), PROJ, bounds, SlopeDEM(), P, pm=None, log=print, turning=turning)

    # 1. the graph: footway and driveway left out; 4 junctions (a cross, two T's, the ring's T)
    kinds = sorted(c.highway for c in net.chains)
    print("chains:", kinds, "junctions:", {k: net.degree[k] for k in net.junctions})
    assert "footway" not in kinds and "service" not in kinds
    assert len(net.junctions) == 4, net.junctions
    assert sorted(net.degree[k] for k in net.junctions) == [3, 3, 3, 4]
    # the two OSM ways of the main street between junctions are one street each
    assert sum(1 for c in net.chains if c.highway == "tertiary") == 4

    # 2. streets that meet share the junction's height exactly, and the profile is smooth (no DEM noise)
    for k, (xy, R) in net.junctions.items():
        zs = [c.cl.z[0] if c.ends[0] == k else c.cl.z[-1] for c in net.chains if c.ends and k in c.ends]
        assert np.ptp(zs) < 1e-9, zs
    for c in net.chains:
        if c.cl.length > 60:
            d2 = np.abs(np.diff(c.cl.z, 2)) / c.cl.spacing ** 2
            assert d2.max() < 0.004, (c.highway, d2.max())             # < 0.4 % grade change per metre
    main = [c for c in net.chains if c.highway == "tertiary" and c.cl.p[:, 0].min() < -150][0]
    zx = np.interp(-160.0, main.cl.p[:, 0] if main.cl.p[0, 0] < main.cl.p[-1, 0] else main.cl.p[::-1, 0],
                   main.cl.z if main.cl.p[0, 0] < main.cl.p[-1, 0] else main.cl.z[::-1])
    assert abs(zx - (100.0 - 0.05 * 160)) < 0.35, zx                   # follows the hill

    # 3. asphalt: junctions, rounded kerbs (7 m), the cul-de-sac's turning circle, the loop's hole
    A = net.asphalt
    assert A.contains(Point(0, 0)) and A.contains(Point(100, 0)) and A.contains(Point(-120, 50))
    assert A.contains(Point(3.25 + 1.0, 4.5 + 1.0)), "kerb corner not rounded"
    assert not A.contains(Point(3.25 + 6.0, 4.5 + 6.0))
    assert A.contains(Point(108.0, 120.0)) and A.contains(Point(100.0, 129.0)), "no turning circle"
    assert not A.contains(Point(-120, 80)), "the loop's middle must stay ground"

    # 4. one continuous surface: no step anywhere on the asphalt, crown on the street, flat at junctions
    xs, ys = np.meshgrid(np.arange(-25, 25, 0.25), np.arange(-25, 25, 0.25))
    z, e = net.query(xs, ys)
    on = e <= 0
    gx = np.abs(np.diff(z, axis=1))[on[:, 1:] & on[:, :-1]]
    gy = np.abs(np.diff(z, axis=0))[on[1:, :] & on[:-1, :]]
    print("max step on the cross junction: %.3f m per 25 cm" % max(gx.max(), gy.max()))
    assert max(gx.max(), gy.max()) < 0.03
    zc, _ = net.query(np.array([-60.0, -60.0]), np.array([0.0, 4.4]))
    assert 0.04 < zc[0] - zc[1] < 0.08, zc                             # crown mid-block
    zj, _ = net.query(np.array([0.0, 0.0]), np.array([0.0, 4.0]))
    assert abs(zj[0] - zj[1] - (0.0 * 4.0)) < 0.025 + 0.02 * 4.0 / 1.0  # ~flat (just the hill)

    # 5. terrain over the network: road height near streets, back to the DEM far away
    tm = network.NetTerrain(PROJ, SlopeDEM(), net, extra=12.5, blend=25.0)
    hz = tm.height(np.array([-60.0, -60.0]), np.array([4.0, 190.0]))
    assert abs(hz[0] - net.query(np.array([-60.0]), np.array([4.0]))[0][0]) < 1e-6
    assert abs(hz[1] - SlopeDEM().sample(*PROJ.to_latlon(np.array([-60.0]), np.array([190.0])))[0]) < 1e-6

    # 6. markings: the main street's centre line stops in the cross, runs through the T's; edge lines stop
    # across the side streets' mouths; residential streets have none
    network.markings(net, P, None, log=print)
    for c in net.chains:
        if c.highway == "residential":
            assert not c.mk.lines
    east = [c for c in net.chains if c.highway == "tertiary" and c.cl.p[:, 0].min() >= -1 and c.cl.p[:, 0].max() < 110][0]

    def painted(c, side, x, tol=0.0):
        """Paint within tol m of the station at x (a dashed line has gaps)."""
        ln = [l for l in c.mk.lines if l.side == side][0]
        s = float(c.cl.s[int(np.argmin(np.abs(c.cl.p[:, 0] - x)))])
        return any(a <= s + tol and b >= s - tol for a, b in ln.pieces)

    assert not painted(east, "centre", 2.0, 1.0), "centre line through the cross"
    assert painted(east, "centre", 50.0, 5.0)
    assert painted(east, "centre", 97.0, 3.0), "centre line must run through the T"
    left_up = "left" if east.cl.n[len(east.cl.p) // 2, 1] > 0 else "right"      # edge line on the north side
    assert not painted(east, left_up, 100.0), "edge line across the side street"
    assert painted(east, left_up, 60.0)

    # 7. start spot: on the street nearest to the click, on the asphalt, facing along it
    sp = PROJ.to_latlon(52.0, 8.0)
    cl, hl, hr, s0, ci = network.spawn_street(net, PROJ, {"spawn": [float(sp[0]), float(sp[1])]})
    assert net.chains[ci].highway == "tertiary"
    objs, n = ac_export.spawn_objects(cl, hl, hr, s0, 6, 6)
    assert n == 6 and [o["name"] for o in objs].count("AC_PIT_5") == 1
    assert not any(o["name"].startswith("AC_TIME") for o in objs), "free roam: no timing gates"
    for o in objs:
        assert A.contains(Point(o["pos"][0], o["pos"][1])), o
        assert abs(o["forward"][1]) < 0.05                               # along the east-west street
    # a short dead end: fewer spots, never off the street
    sp2 = PROJ.to_latlon(100.0, 110.0)
    cl2, hl2, hr2, s02, _ = network.spawn_street(net, PROJ, {"spawn": [float(sp2[0]), float(sp2[1])]})
    objs2, n2 = ac_export.spawn_objects(cl2, hl2, hr2, s02, 20, 20)
    assert n2 < 20 and all(A.buffer(0.2).contains(Point(o["pos"][0], o["pos"][1])) for o in objs2)

    # 8. the ground surface: street outline and cross-sections fixed, nothing loose on the asphalt
    from app import ground

    tm.tin = ground.build_tin_net(tm, net, bounds, 8.0, near=15.0, log=print)
    pts = tm.tin.xy[~tm.tin.fixed]
    assert not A.buffer(-0.1).contains(Point(0, 0)) or not any(A.buffer(-0.1).contains(Point(p)) for p in pts[::50])
    hz = tm.surface_height(np.array([-60.0, 0.0]), np.array([0.0, 0.0]))
    hq, _ = net.query(np.array([-60.0, 0.0]), np.array([0.0, 0.0]))
    assert np.all(np.abs(hz - hq) < 0.02), (hz, hq)

    # 9. the whole pipeline in area mode (offline demo data)
    from app import config, pipeline

    with tempfile.TemporaryDirectory() as d:
        config.OUTPUT_DIR = Path(d)
        la, lo = PROJ.to_latlon(np.array([-300.0, 300.0]), np.array([-300.0, 300.0]))
        rep = pipeline.generate({"mode": "area", "area": [float(la[0]), float(lo[0]), float(la[1]), float(lo[1])],
                                 "name": "Area smoke", "trees": False, "grass": False, "terrain_spacing": 16,
                                 "kn5_dds": False}, log=lambda *a: None)
        assert rep["area"] and rep["streets"] and rep["n_streets"] >= 1, rep.get("n_streets")
        assert rep["kn5"] and (Path(d) / "area_smoke" / "ac_track" / "area_smoke" / "area_smoke.kn5").exists()
        names = [o["name"] for o in rep["objects"]]
        assert "AC_START_0" in names and "AC_PIT_0" in names and not any(n.startswith("AC_TIME") for n in names)
        ui = (Path(d) / "area_smoke" / "ac_track" / "area_smoke" / "ui" / "ui_track.json").read_text()
        assert "free roam" in ui
        poly = [[float(a), float(b)] for a, b in zip(*PROJ.to_latlon(np.array([-280.0, 280.0, 0.0]),
                                                                       np.array([-250.0, -250.0, 280.0])))]
        rep2 = pipeline.generate({"mode": "area", "area": [float(la[0]), float(lo[0]), float(la[1]), float(lo[1])],
                                  "area_poly": poly, "name": "Area tri", "trees": False, "grass": False,
                                  "terrain_spacing": 16, "kn5_dds": False}, log=lambda *a: None)
        assert rep2["area"] and rep2["kn5"]
        from shapely.geometry import Polygon as _P2

        tri_xy = _P2(PROJ.latlngs_to_xy(poly)).buffer(1.0)
        objs = [o for o in rep2["objects"]]
        assert objs and all(tri_xy.contains(Point(PROJ.to_xy(o["lat"], o["lon"]))) for o in objs)
        # too big an area is refused
        try:
            pipeline._params({"mode": "area", "area": [48.0, -71.0, 48.05, -70.95]})
            raise AssertionError("a 5 km area should be refused")
        except ValueError as exc:
            print("refused:", exc)
    # 10. polygon areas: streets, asphalt and ground only inside the outline; size limits; big-area scaling
    from shapely.geometry import Polygon as _Poly

    tri = _Poly([(-240, -190), (240, -190), (240, 60), (-240, 60)])       # cuts off the north (cul-de-sac, ring)
    net2 = network.build(roads(), PROJ, bounds, SlopeDEM(), P, pm=None, log=lambda *a: None, clip=tri)
    assert all(c.cl.p[:, 1].max() <= 60.5 for c in net2.chains), "a street runs out of the polygon"
    assert net2.asphalt.difference(tri.buffer(0.05)).area < 1.0
    tm2 = network.NetTerrain(PROJ, SlopeDEM(), net2, extra=12.5, blend=25.0)
    tin2 = ground.build_tin_net(tm2, net2, bounds, 8.0, near=15.0, log=lambda *a: None, keep=tri)
    assert tin2.xy[:, 1].max() <= 60.05 and tin2.xy[:, 1].min() >= -190.05

    from app import pipeline as pl

    def ll(xy):
        la, lo = PROJ.to_latlon(np.array([q[0] for q in xy], float), np.array([q[1] for q in xy], float))
        return [[float(a), float(b)] for a, b in zip(la, lo)]

    ok = pl.area_corners({"area_poly": ll([(0, 0), (4000, 0), (4000, 2000), (0, 2000)])})     # 8 km²
    assert len(ok) == 4
    for bad, why in ((ll([(0, 0), (4000, 0), (4000, 3000), (0, 3000)]), "12 km²"),
                     (ll([(0, 0), (6500, 0), (6500, 300), (0, 300)]), "6.5 km across"),
                     (ll([(0, 0), (100, 100), (100, 0), (0, 100)]), "self-crossing")):
        try:
            pl.area_corners({"area_poly": bad})
            raise AssertionError(f"{why} should be refused")
        except ValueError as exc:
            print("refused:", exc)
    assert pl.area_scale((0, 0, 2000, 2000))[2] == 1.0
    g_res, a_res, f = pl.area_scale((0, 0, 4000, 2500))
    assert g_res == 0.5 and abs(a_res - 0.5 * f) < 1e-9 and f > 1.5

    # the street photo read piece by piece gives the same pixels as a whole-area photo
    from app import imagery, photo

    rng = np.random.default_rng(0)
    big = (rng.random((900, 900, 3)) * 255).astype(np.uint8)

    def fake_raster(src, s_, w_, n_, e_, res):
        mx0, my0 = imagery._merc(s_ - 0.01, w_ - 0.01)
        mx1, my1 = imagery._merc(n_ + 0.01, e_ + 0.01)
        return imagery._Raster(big, mx0, my1, (mx1 - mx0) / 900, (my1 - my0) / 900)

    real = imagery.fetch_raster
    imagery.fetch_raster = fake_raster
    try:
        tp = photo.TiledPhoto({"type": "wms"}, PROJ, res=0.25, max_cells=2)
        xs, ys = rng.uniform(-300, 300, 500), rng.uniform(-300, 300, 500)
        v, okv = tp.sample(xs, ys)
        assert okv.mean() > 0.95 and v.shape == (500, 3) and tp.loaded >= 1
        la, lo = PROJ.to_latlon(xs, ys)
        ci = np.floor(la / photo.CELL_DEG[0]).astype(int)
        cj = np.floor(lo / photo.CELL_DEG[1]).astype(int)
        k0 = (ci == ci[0]) & (cj == cj[0])
        r = fake_raster(None, ci[0] * photo.CELL_DEG[0] - 0.0003, cj[0] * photo.CELL_DEG[1] - 0.0003,
                        (ci[0] + 1) * photo.CELL_DEG[0] + 0.0003, (cj[0] + 1) * photo.CELL_DEG[1] + 0.0003, 0.25)
        assert np.allclose(v[k0], r.sample(la[k0], lo[k0]), atol=0.51)
    finally:
        imagery.fetch_raster = real

    # 11. the area-mode flag in generate() is never overwritten by a local of the same name (a footprint's
    # "area" once turned route generations into area mode when buildings were found on the photo)
    import ast

    src = (Path(__file__).resolve().parents[1] / "app" / "pipeline.py").read_text(encoding="utf-8")
    for fn in ast.parse(src).body:
        if isinstance(fn, ast.FunctionDef) and fn.name == "generate":
            stores = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "area"
                      and isinstance(n.ctx, ast.Store)]
            assert len(stores) == 1, f"'area' assigned more than once in generate(): lines {stores}"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

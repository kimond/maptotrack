"""rFactor 2 export: the text files of a location (scn, gdb, tdf, aiw, cam), surfaces, textures - no network.

  MAPTOTRACK_OFFLINE=1 python tests/rf2_test.py
"""
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MAPTOTRACK_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from app import rf2


def circle_line(R=80.0, n=200):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.column_stack([R * np.cos(a), R * np.sin(a)])                 # counter-clockwise loop
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return {"mode": "route", "loop": True, "length": float(s[-1] + np.linalg.norm(p[0] - p[-1])),
            "p": p.tolist(), "z": (2.0 * np.sin(a)).tolist(), "s": s.tolist(), "hl": [4.0] * n, "hr": [4.5] * n,
            "shoulder": 1.5, "lat": 48.5, "lon": -71.25, "z0": 120.0, "name": "Test ring", "city": "Saguenay",
            "country": "Canada", "author": "", "total_length": 500.0}


def main():
    line = circle_line()
    # cars at the start, facing along the loop (counter-clockwise: at angle 0 the road heads north, +y)
    objs = [{"name": f"AC_START_{i}", "pos": [80.0, -7.0 * i, 0.3], "forward": [0.0, 1.0, 0.0]} for i in range(4)]
    objs += [{"name": f"AC_PIT_{i}", "pos": [80.0, -7.0 * i, 0.3], "forward": [0.0, 1.0, 0.0]} for i in range(4)]

    # 1. axes and headings: local north -> rF2 +Z, up -> +Y; a car facing north has forward (-sin, -cos) = (0, +1)
    assert np.allclose(rf2.to_rf2([1.0, 2.0, 3.0]), [1.0, 3.0, 2.0])
    yaw = rf2.yaw_of([0.0, 1.0])
    f = np.array([-math.sin(yaw), -math.cos(yaw)])                      # rF2 (X, Z)
    assert np.allclose(f, [0.0, 1.0], atol=1e-9), (yaw, f)
    assert np.allclose([-math.sin(rf2.yaw_of([1.0, 0.0])), -math.cos(rf2.yaw_of([1.0, 0.0]))], [1.0, 0.0], atol=1e-9)

    # 2. surfaces: every collision mesh class gets a code that the .tdf defines, drawn-only meshes none
    cases = {("1ROAD_ASPH_00_00", "asphalt_tiled"): "mtroad", ("1ROAD_GND_00_01", "ground_00_01"): "mtroad",
             ("1GRASS_GND_00_01", "ground_00_01"): "mtgras", ("1GRASS_SHLD_00", "gravel_tiled"): "mtgrav",
             ("1GRASS_NEAR_WILD_501_499_000", "near_wild"): "mtrugh", ("1GRASS_NEAR_SAND_1_2_000", "near_sand"): "mtsand",
             ("1ROAD_NEAR_PAVED_1_2_000", "near_paved"): "mtroad", ("1WALL_000", "concrete"): "mtwall",
             ("1WALL_TYRES", "tyres"): "mttyre", ("TREES_000", "trees"): None, ("ROAD_LINES_000", "road_paint"): None}
    for (mesh, mat), want in cases.items():
        assert rf2.surface_of(mesh, mat) == want, (mesh, rf2.surface_of(mesh, mat), want)
    tdf = rf2.tdf_text("ring")
    codes = re.findall(r"^Materials=(\w+)", tdf, re.M)
    assert set(codes) == set(rf2.SURFACES), codes
    assert all(not a in b for a in codes for b in codes if a != b), "a code must not be part of another"
    assert tdf.count("[FEEDBACK]") == len(rf2.SURFACES)
    loose = tdf[tdf.index("[LOOSE]"):]                                          # global effects, as in stock tracks
    for r in ("spark", "whitesmoke", "blacksmoke", "flames", "wallskid"):
        assert f"Reaction={r} " in loose, r
    stock_sounds = {"dry", "grass", "gravel", "guardrail", "rumble", "solidwall"}
    assert set(re.findall(r"Sound=(\w+)", tdf)) <= stock_sounds, set(re.findall(r"Sound=(\w+)", tdf))

    # 3. AI waypoints: a closed chain, sectors in order, widths and positions in rF2 axes
    aiw, info = rf2.aiw_text(line, objs)
    n = info["waypoints"]
    assert abs(info["lap_length"] - line["length"]) < 1.0 and 100 < n < 140
    wps = re.split(r"\r\nwp_pos=", aiw.split("[Waypoint]")[1])[1:]
    W = []
    for blk in wps:
        W.append({"pos": [float(v) for v in blk.split(")")[0].strip("(").split(",")],
                  "br": int(re.search(r"wp_branchID=\((\d+)\)", blk).group(1)),
                  "bf": int(re.search(r"wp_bitfields=\((\d+)\)", blk).group(1)),
                  "sec": int(re.search(r"wp_score=\((\d),", blk).group(1)),
                  "ptr": [int(v) for v in re.search(r"WP_PTRS=\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)", blk).groups()]})
    N = len(W)
    assert f"number_waypoints={N}" in aiw and "pitlanes=1" in aiw and aiw.count("definepath=") == 3
    assert all(len(re.findall(r"wp_pathinfo2=", blk)) == 3 for blk in wps)
    assert all(-1 <= v < N for w in W for v in w["ptr"][:3]) and all(w["ptr"][3] == w["br"] for w in W)
    main = [i for i, w in enumerate(W) if w["br"] == 0]
    assert main == list(range(n))                                                # the path first, a closed chain
    assert all(W[i]["ptr"][:2] == [(i - 1) % n, (i + 1) % n] for i in main)
    secs = [W[i]["sec"] for i in main]
    assert secs == sorted(secs) and set(secs) == {0, 1, 2}
    pos = np.array([W[i]["pos"] for i in main])
    assert np.allclose(np.hypot(pos[:, 0], pos[:, 2]), 80.0, atol=0.2)          # on the ring (X, Z)
    assert abs(pos[:, 1].max() - 2.0) < 0.1                                     # heights in Y
    # the pit lane: one chain leaving the path and joining it again, through the pit spots
    lane = [i for i, w in enumerate(W) if w["br"] == 1]
    assert len(lane) == info["pit_lane"] > 10 and all(W[i]["bf"] == 0 for i in lane)
    assert W[lane[0]]["ptr"][0] in main and W[lane[-1]]["ptr"][1] in main
    assert W[W[lane[0]]["ptr"][0]]["ptr"][2] == lane[0]                         # the path links to its start
    assert all(W[a]["ptr"][1] == b and W[b]["ptr"][0] == a for a, b in zip(lane, lane[1:]))
    # one 2-waypoint branch per grid spot (from the path) and per pit box (from the pit lane), ending on the spot
    br = {}
    for i, w in enumerate(W):
        if w["br"] >= 2:
            br.setdefault(w["br"], []).append(i)
    assert len(br) == 8 == info["branches"] and all(len(v) == 2 for v in br.values())
    gspots = [rf2.to_rf2(o["pos"]) for o in objs if o["name"].startswith("AC_START_")]
    boxes = [rf2.to_rf2(b) for b in info["pit_boxes"]]
    assert len(boxes) == 4 and info["lane_offset"] > 0.5                        # the lane is beside the racing line
    assert min(np.linalg.norm(np.subtract(b, g)) for b in boxes for g in gspots) > 5.0   # boxes apart from the grid
    pitpos = [[float(v) for v in m] for m in re.findall(r"PitPos=\(([-\d.]+),([-\d.]+),([-\d.]+)\)", aiw)]
    assert np.allclose(pitpos, boxes, atol=2e-3)
    for b, ids in sorted(br.items()):
        end, ent = ids                                                          # the spot first, then the lead-in
        assert W[ent]["bf"] == W[end]["bf"] == 1
        assert W[ent]["ptr"][1] == end and W[end]["ptr"][:2] == [ent, end]        # a branch ends on itself
        src = W[ent]["ptr"][0]
        assert W[src]["ptr"][2] == ent                                          # its source waypoint links to it
        assert W[src]["br"] == (0 if b < 6 else 1)                              # grid from the path, pits from the lane
        spots = gspots if b < 6 else boxes
        assert min(np.linalg.norm(np.subtract(W[end]["pos"], sp)) for sp in spots) < 2e-3
        if b >= 6:                                                              # nothing else on a pit box
            others = [w["pos"] for j, w in enumerate(W) if j != end]
            assert min(np.linalg.norm(np.subtract(W[end]["pos"], o)) for o in others) > 1.0
        assert abs(np.linalg.norm(np.subtract(W[ent]["pos"], W[end]["pos"])) - rf2.BRANCH_LEAD) < 0.05
    assert "startinggrid=4" in aiw and aiw.count("TeamIndex=") == 4 and "[GRID]" in aiw
    gpos = re.search(r"\[GRID\]\r\nGridIndex=0\r\nPos=\(([-\d.]+),([-\d.]+),([-\d.]+)\)", aiw)
    assert gpos and abs(float(gpos.group(1)) - 80.0) < 1e-6 and abs(float(gpos.group(2)) - 0.3) < 1e-6

    # 4. timing gates across the road, finish at the start
    g = rf2.gates(line)
    assert set(g) == {"xfinish", "xsector1", "xsector2"}
    fin = np.array(g["xfinish"])
    assert np.ptp(fin[:, 2]) > 7 and np.ptp(fin[:, 0]) > 15                    # tall, across the road (x)

    # 5. a whole location from a (fake) generation: files, names, scene flags
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        tex = d / "texture"
        tex.mkdir()
        Image.new("RGB", (300, 200), (90, 90, 90)).save(tex / "asphalt_tiled.png")
        leaf = Image.new("RGBA", (64, 64), (40, 120, 40, 0))
        leaf.paste((40, 120, 40, 255), (16, 16, 48, 48))
        leaf.save(tex / "trees.png")
        report = {"meshes": [{"name": "1ROAD_ASPH_00_00", "material": "asphalt_tiled"},
                             {"name": "1ROAD_ASPH_00_01", "material": "asphalt_tiled"},
                             {"name": "TREES_000", "material": "trees"},
                             {"name": "1WALL_000", "material": "concrete"},
                             {"name": "NOT_IN_MODEL", "material": "concrete"}]}

        def grid(nx, ny, x0=0.0):                   # a flat nx * ny quad grid -> 2 * nx * ny triangles
            xs, ys = np.meshgrid(np.arange(nx + 1, dtype=float), np.arange(ny + 1, dtype=float))
            v = np.column_stack([xs.ravel() + x0, ys.ravel(), np.zeros(xs.size)])
            i = (np.arange(ny)[:, None] * (nx + 1) + np.arange(nx)[None, :]).ravel()
            f = np.vstack([np.column_stack([i, i + 1, i + nx + 2]), np.column_stack([i, i + nx + 2, i + nx + 1])])
            return v, f, v[:, :2] / max(nx, ny), None

        meshes = {"1ROAD_ASPH_00_00": grid(150, 150), "1ROAD_ASPH_00_01": grid(4, 4, 200.0),
                  "TREES_000": grid(2, 2), "1WALL_000": grid(3, 1)}
        res = rf2.write_location(d / "out", "My ring (test)", report, objs, line, tex, None, meshes=meshes,
                                 log=print)
        root = res["root"]
        loc = res["loc"]
        assert loc == "My_ring_test"
        for f in (root / f"{loc}.tdf", root / loc / f"{loc}.scn", root / loc / f"{loc}.gdb", root / loc / f"{loc}.aiw",
                  root / loc / f"{loc}.cam", root / "Assets" / "Maps" / "asphalt_tiled.dds",
                  root / "Assets" / "Maps" / "trees.dds"):
            assert f.exists(), f
        scn = (root / loc / f"{loc}.scn").read_text()
        for mapper in ("ReflectionMapper=STATIC01", "ReflectionMapper=REFMAP0", "IRRADIANCE_PROBE=IRRAD_GLOBAL"):
            assert mapper in scn, mapper
        assert "IncludeIns=CWALL_000" in scn
        assert "Instance=CROAD_ASPH_00_00" in scn and "Instance=xfinish" in scn and "Response=VEHICLE,TIMING" in scn
        block = scn[scn.index("Instance=CROAD_ASPH_00_00"):].split("}")[0]
        assert "CollTarget=True" in block and "HATTarget=True" in block and "Deformable=True" in block
        block = scn[scn.index("Instance=TREES_000"):].split("}")[0]
        assert "CollTarget=False" in block and "LODOut=(1200)" in block
        block = scn[scn.index("Instance=CWALL_000"):].split("}")[0]
        assert "CollTarget=True" in block and "HATTarget=False" in block
        gdb = (root / loc / f"{loc}.gdb").read_text()
        assert f"TerrainDataFile=..\\{loc}.tdf" in gdb and "Latitude = 48.50000" in gdb and "TimezoneRelativeGMT = -5.0" in gdb
        man = res["manifest"]
        # 45000 triangles -> 3 GMTs of at most GMT_MAX_TRIS (the exporter's 65536 vertex limit), all faces kept
        road = [m for m in man["meshes"] if m["mesh"] == "1ROAD_ASPH_00_00"]
        assert [m["gmt"] for m in road] == ["CROAD_ASPH_00_00", "CROAD_ASPH_00_00_1", "CROAD_ASPH_00_00_2"], road
        nf = 0
        for m in road:
            z = np.load(m["npz"])
            assert len(z["f"]) <= rf2.GMT_MAX_TRIS and z["f"].max() < len(z["v"]) == len(z["uv"])
            nf += len(z["f"])
        assert nf == 2 * 150 * 150
        for g in ("CROAD_ASPH_00_00_1", "CROAD_ASPH_00_00_2"):
            assert f"Instance={g}" in scn
        assert "NOT_IN_MODEL" not in scn
        mats = {m["mesh"]: m for m in man["meshes"]}
        assert mats["1ROAD_ASPH_00_00"]["rf2_material"] == "MTROAD_ASPHALT_TILED"
        assert mats["TREES_000"]["alpha"] and not mats["1ROAD_ASPH_00_00"]["alpha"]
        assert len({m["gmt"].upper() for m in man["meshes"]}) == len(man["meshes"])
        # DDS headers: BC1 for opaque, 32-bit BGRA with mips for alpha
        a = (root / "Assets" / "Maps" / "asphalt_tiled.dds").read_bytes()
        t = (root / "Assets" / "Maps" / "trees.dds").read_bytes()
        assert a[:4] == b"DDS " and a[84:88] == b"DXT1"
        assert t[:4] == b"DDS " and int.from_bytes(t[88:92], "little") == 32 and int.from_bytes(t[28:32], "little") == 7
        im = Image.open(root / "Assets" / "Maps" / "trees.dds")
        im.load()
        assert im.size == (64, 64) and im.mode == "RGBA" and np.asarray(im)[32, 32, 3] == 255
        cam = (root / loc / f"{loc}.cam").read_text()
        assert cam.count("TrackingCam=") == 2
        json.dumps(man)

        # menu pictures (stock names and sizes) and the format marker
        lay = root / loc
        from PIL import Image as _I
        assert _I.open(lay / f"{loc}icon.tga").size == (102, 87)
        assert _I.open(lay / f"{loc}Thmb.tga").size == (190, 128)
        assert _I.open(lay / f"{loc}_loading.jpg").size == (1920, 1080)
        assert json.loads((root / rf2.MARKER).read_text())["format"] == rf2.RF2_FORMAT
        assert "Joesville" not in scn                                      # self-contained (packaged tracks)

        # packaging: which file goes in which .mas
        groups = rf2.mas_groups(root, loc)
        assert set(groups) == {f"{loc}.mas", f"{loc}_gmts.mas", f"{loc}_maps.mas", f"{loc}_materials.mas",
                               f"{loc}_layout.mas"}
        mjs = {f.stem: json.loads(f.read_text()) for f in groups[f"{loc}_materials.mas"]}
        assert mjs["TREES"]["shader"] == "L2IBLSTANDARD" and "CUBETF_CHROMAKEY" in mjs["TREES"]["textures"][0]["Flags"]
        assert mjs["TREES"]["textures"][0]["Texture"] == "trees" and "CUBEMAT_TWOSIDED" in mjs["TREES"]["flags"]
        assert mjs["MTROAD_ASPHALT_TILED"]["parameters"]["cbParameters"]["Base_Material"]["Roughness"] < 0.8
        assert "CUBETF_CHROMAKEY" not in mjs["MTROAD_ASPHALT_TILED"]["textures"][0]["Flags"]
        assert [f.name for f in groups[f"{loc}.mas"]] == [f"{loc}.tdf"]
        lay_names = {f.name for f in groups[f"{loc}_layout.mas"]}
        assert {f"{loc}.gdb", f"{loc}.scn", f"{loc}.aiw", f"{loc}.cam", f"{loc}icon.tga"} <= lay_names
        assert all(f.suffix == ".dds" for f in groups[f"{loc}_maps.mas"])

        # 6. optional: the real .gmt export (Blender 5 + Studio 397's add-on), e.g.
        #    MAPTOTRACK_TEST_BLENDER=C:\\...\\blender.exe MAPTOTRACK_TEST_RF2_ADDON=io_rfactor2_gmt_50.zip
        blender, addon = os.environ.get("MAPTOTRACK_TEST_BLENDER"), os.environ.get("MAPTOTRACK_TEST_RF2_ADDON")
        if blender and addon:
            import subprocess

            man_p = d / "out" / "rf2_meshes.json"
            man_p.write_text(json.dumps(man), encoding="utf-8")
            script = Path(__file__).resolve().parents[1] / "blender" / "export_rf2_gmt.py"
            p = subprocess.run([blender, "--background", "--factory-startup", "--python", str(script), "--",
                                "--manifest", str(man_p), "--addon", addon], capture_output=True, text=True)
            assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
            names = sorted(f.stem for f in (root / "Assets" / "GMT").glob("*.gmt"))
            want = sorted([m["gmt"] for m in man["meshes"]] + list(man["gates"]))
            assert names == want, (names, want)
            g = (root / "Assets" / "GMT" / "xfinish.gmt").read_bytes()
            assert b"white.dds" in g and b"L2DIFFUSET0" in g
            # vertices in rF2 axes: local (x, y, z) is written as (x, z, y)
            c = np.array(man["gates"]["xfinish"][0], np.float32)[[0, 2, 1]]
            assert c.tobytes() in g, "xfinish corner not found in rF2 axes"
            print("  .gmt export checked:", len(names), "files")
        else:
            print("  (.gmt export not checked: set MAPTOTRACK_TEST_BLENDER and MAPTOTRACK_TEST_RF2_ADDON)")
    # 6b. the sky dome and its texture come with the location (a packaged track can't reach Joesville)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        jv = d / "rf2" / "ModDev" / "Locations" / "Joesville" / "Assets"
        (jv / "GMT").mkdir(parents=True)
        (jv / "Maps").mkdir(parents=True)
        (jv / "GMT" / "skyboxi.gmt").write_bytes(b"sky")
        (jv / "Maps" / "BKA.DDS").write_bytes(b"DDS sky")
        tex = d / "texture"
        tex.mkdir()
        res = rf2.write_location(d / "out", "Sky test", {"meshes": []}, objs, line, tex, d / "rf2", meshes={},
                                 log=lambda *a: None)
        r = res["root"]
        assert (r / "Assets" / "GMT" / "skyboxi.gmt").exists() and (r / "Assets" / "Maps" / "BKA.DDS").exists()
        assert "Instance=SkyBoxi" in (r / res["loc"] / f"{res['loc']}.scn").read_text()
        (r / "Assets" / "Maps" / "BKA.DDS").unlink()                    # an older copy without the texture: repaired
        assert rf2.copy_sky(r, d / "rf2") and (r / "Assets" / "Maps" / "BKA.DDS").exists()

    # 6b2. shininess per material: grass / ground nearly matt, asphalt a little, foliage double-sided
    assert rf2.shading_of("ground_00_01")["spec"] < 0.05 and rf2.shading_of("near_lawn")["spec"] < 0.05
    assert rf2.shading_of("asphalt_tiled")["spec"] > rf2.shading_of("near_lawn")["spec"]
    assert rf2.shading_of("trees")["double"] and rf2.shading_of("veg_grass_short")["double"]
    assert not rf2.shading_of("siding_blue")["double"]

    # 6c. .gmt jobs: split evenly over the Blenders, cache keys follow the geometry
    parts = rf2._split([{"tris": t} for t in (20000, 20000, 20000, 15000, 5000, 4000, 1000)], 3)
    loads = sorted(sum(i["tris"] for i in p) for p in parts)
    assert len(parts) == 3 and sum(loads) == 85000 and loads[-1] <= 35000, loads
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        z = d / "addon.zip"
        z.write_bytes(b"zip")
        np.savez(d / "a.npz", v=np.zeros((3, 3)), f=np.array([[0, 1, 2]]), uv=np.zeros((3, 2)))
        it = {"gmt": "A", "rf2_material": "m", "dds": "a.dds", "npz": str(d / "a.npz")}
        k1 = rf2._gmt_key(it, str(z))
        assert k1 == rf2._gmt_key(dict(it), str(z)) and k1 != rf2._gmt_key(dict(it, dds="b.dds"), str(z))
        np.savez(d / "a.npz", v=np.ones((3, 3)), f=np.array([[0, 1, 2]]), uv=np.zeros((3, 2)))
        assert rf2._gmt_key(it, str(z)) != k1

    # 7. MAS2's component list: ours added and selected, others kept, same name replaced
    other = ("[Component]\r\nName=Other\r\nType=2\r\nVersion=3.0\r\nNumMASFiles=1\r\nMASFile=C:\\a.mas\r\n"
             "\r\nCurComponent=0\r\n")
    comp = {"name": "My_ring_test", "version": "1.00", "rfcmp": r"C:\o\My_ring_test_v100.rfcmp",
            "mas": [r"C:\o\My_ring_test.mas", r"C:\o\My_ring_test_layout.mas"]}
    t1 = rf2.cmpinfo_text(other, comp)
    assert t1.count("[Component]") == 2 and "Name=Other" in t1 and t1.rstrip().endswith("CurComponent=1")
    assert "NumMASFiles=2" in t1 and r"Location=C:\o\My_ring_test_v100.rfcmp" in t1 and "Type=1" in t1
    t2 = rf2.cmpinfo_text(t1, dict(comp, version="1.01"))
    assert t2.count("[Component]") == 2 and "Version=1.01" in t2 and "Version=1.00" not in t2
    assert rf2.cmpinfo_text("", comp).rstrip().endswith("CurComponent=0")
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "Installed" / "Locations" / "My_ring_test"
        assert rf2.next_version(Path(d), "My_ring_test", None) == "1.00"
        (base / "1.00").mkdir(parents=True)
        (base / "1.04").mkdir()
        assert rf2.next_version(Path(d), "My_ring_test", "1.02") == "1.05"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

"""CSP Grass FX config (no network).

  python tests/csp_test.py
"""
import configparser
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import csp


def parse(text):
    cp = configparser.ConfigParser(strict=False, inline_comment_prefixes=(";",), comment_prefixes=(";",))
    cp.optionxform = str
    cp.read_string(text)
    return cp


def items(v):
    return {x.strip() for x in v.split(",") if x.strip()}


def main():
    mats = {"ground_00_00", "ground_00_01", "near_lawn", "near_wild", "near_dry", "near_sand", "near_paved",
            "near_forest", "asphalt_tiled", "gravel_tiled", "road_paint", "veg_grass", "veg_grass_short", "trees"}
    cp = parse(csp.grass_fx_ini(mats))
    g = cp["GRASS_FX"]
    grass = items(g["GRASS_MATERIALS"])
    assert grass == {"ground_00_00", "ground_00_01", "near_lawn", "near_wild", "near_dry"}, grass
    assert items(g["OCCLUDING_MATERIALS"]) == {"asphalt_tiled", "gravel_tiled", "road_paint", "near_paved"}
    assert items(g["OCCLUDING_MATERIALS_ALPHA"]) == {"near_sand"}
    assert items(g["ORIGINAL_GRASS_MATERIALS"]) == {"veg_grass_short"}, "tall grass must stay"
    assert cp["GRASS_FX_ADJUSTMENT_0"]["MATERIALS"].strip() == "near_lawn"
    assert cp["GRASS_FX_ADJUSTMENT_0"]["MAP"].replace(" ", "") == "A,1"
    assert items(cp["GRASS_FX_ADJUSTMENT_1"]["MATERIALS"]) == {"near_wild", "near_dry"}
    assert float(cp["GRASS_FX_CONFIGURATION_B"]["SHAPE_SIZE"]) > float(cp["GRASS_FX_CONFIGURATION_A"]["SHAPE_SIZE"].split(",")[0])
    # never grass on materials that are not ground
    every = set().union(*(items(cp[s][k]) for s in cp.sections() for k in cp[s] if k.endswith("MATERIALS")))
    assert every <= mats, every - mats

    # no grass materials -> no file; a hand-edited file is kept
    assert csp.grass_fx_ini({"asphalt_tiled"}) == ""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        p = csp.write_ext_config(d, mats, log=print)
        assert p is not None and p.read_text().startswith(csp.MARKER)
        p.write_text("[GRASS_FX]\nGRASS_MATERIALS = mine\n")
        assert csp.write_ext_config(d, mats, log=print) is None
        assert "mine" in p.read_text()
        p.write_text(csp.MARKER + "\n")
        csp.write_ext_config(d, mats, grass_fx=False, log=print)
        assert not p.exists(), "our own file should go when Grass FX is switched off"
    print(csp.grass_fx_ini(mats))

    # CSP trees: .bin models from csp_trees/, kinds from names / trees.json, a line per tree
    import json

    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "csp_trees"
        assert csp.find_tree_models(src) == {} and (src / "README.txt").exists(), "README not created"
        (src / "red_pine.bin").write_bytes(b"x")
        (src / "birch_a.bin").write_bytes(b"y")
        (src / "odd_name.bin").write_bytes(b"z")
        (src / "trees.json").write_text(json.dumps({"odd_name.bin": {"kind": "broadleaf", "height": 20}}))
        models = csp.find_tree_models(src)
        assert [n for n, _ in models[0]] == ["red_pine.bin"]
        assert sorted(n for n, _ in models[1]) == ["birch_a.bin", "odd_name.bin"]
        assert models[2] == models[1], "bushes borrow broadleaf models"
        assert dict(models[1])["odd_name.bin"] == 20.0 and models[0][0][1] == csp.DEFAULT_HEIGHT[0]
        csp.tree_dir = lambda: src
        trees = {"x": np.array([10.0, 20.0, 30.0]), "y": np.array([5.0, -7.0, 0.0]), "h": np.array([15.0, 12.0, 2.5]),
                 "w": np.array([6.75, 9.6, 3.5]), "kind": np.array([0, 1, 2])}
        z = np.array([100.0, 101.0, 102.0])
        out = d / "track"
        p = csp.write_ext_config(out, mats, grass_fx=True, trees=(trees, z, models), log=print)
        cp = parse(p.read_text())
        assert cp["TREES"]["LIST_0"].strip() == "trees/trees.txt"
        assert "ground_00_00" in items(cp["TREES"]["SURFACE_MATERIALS"])
        hide = [s for s in cp.sections() if s.startswith("SHADER_REPLACEMENT_")]
        assert hide and cp[hide[0]]["MATERIALS"].strip() == "trees" and float(cp[hide[0]]["LOD_IN"]) > 1000
        lines = [ln for ln in (out / "extension" / "trees" / "trees.txt").read_text().splitlines() if ln.startswith("tree: ")]
        print("\n".join(lines))
        assert len(lines) == 3
        assert lines[0].startswith("tree: red_pine.bin; pos=10.00, 100.00, -5.00;") and "size = 1.00" in lines[0]
        assert "pos=20.00, 101.00, 7.00" in lines[1]
        assert (out / "extension" / "trees" / "red_pine.bin").exists()
        # no models: no [TREES]
        p2 = csp.write_ext_config(d / "t2", mats, grass_fx=True, trees=None, log=print)
        assert "[TREES]" not in p2.read_text()
        # night lights and rain
        lamps = [(10.0, 20.0, 108.0), (40.0, -5.0, 107.5)]
        p3 = csp.write_ext_config(d / "t3", mats | {"props", "props_lit", "road_paint_yellow"}, grass_fx=False,
                                  log=print, lights=lamps, rain=True)
        cp = parse(p3.read_text())
        ls = cp["LIGHT_SERIES_0"]
        assert ls["CONDITION"].strip() == "NIGHT_SMOOTH" and ls["DIRECTION"].strip() == "0, -1, 0"
        assert ls["POSITION_0"].strip() == "10.00, 107.88, -20.00" and ls["POSITION_1"].strip() == "40.00, 107.38, 5.00"
        ma = cp["MATERIAL_ADJUSTMENT_0"]
        assert ma["MATERIALS"].strip() == "props_lit" and ma["KEY_0"].strip() == "ksEmissive"
        rf = cp["RAIN_FX"]
        assert items(rf["PUDDLES_MATERIALS"]) == {"asphalt?"}
        assert {"asphalt?", "gravel?"} <= items(rf["SOAKING_MATERIALS"])
        assert items(rf["LINES_MATERIALS"]) == {"road_paint", "road_paint_yellow"}
        assert "ground_?" in items(rf["ROUGH_MATERIALS"]) and "near_lawn" in items(rf["ROUGH_MATERIALS"])
        assert "GRASS_FX" not in cp
        # the lamps nearest the track are kept
        assert csp.pick_lights(list(range(5)), [5, 1, 4, 2, 3], n=2) == [1, 3]
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

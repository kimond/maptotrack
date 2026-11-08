"""AC AI line files (no network).

  python tests/ai_line_test.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import ai_line


def oval():
    """A 400 m stadium: two 100 m straights and two half circles of radius ~31.8 m, counter-clockwise."""
    r = 100 / np.pi
    pts = []
    for x in np.arange(0, 100, 2.0):
        pts.append((x, -r))
    for a in np.arange(-np.pi / 2, np.pi / 2, 2.0 / r):
        pts.append((100 + r * np.cos(a), r * np.sin(a)))
    for x in np.arange(100, 0, -2.0):
        pts.append((x, r))
    for a in np.arange(np.pi / 2, 3 * np.pi / 2, 2.0 / r):
        pts.append((r * np.cos(a), r * np.sin(a)))
    p = np.array(pts)
    return p, 10 + 0.02 * p[:, 0], r


def main():
    p, z, r = oval()
    n = len(p)
    objs = [{"name": f"AC_PIT_{i}", "pos": [20 + 8 * i, -r + 3, 10.5]} for i in range(4)]
    with tempfile.TemporaryDirectory() as d:
        info = ai_line.write_ai_lines(Path(d) / "ai", {"p": p, "z": z, "hl": np.full(n, 5.0), "hr": 4.0,
                                                        "loop": True}, objs, log=print)
        a = ai_line.read_ai((Path(d) / "ai" / "fast_lane.ai").read_bytes())
        assert a["version"] == 7 and a["has_grid"] == 0
        assert a["size"] == (Path(d) / "ai" / "fast_lane.ai").stat().st_size
        assert a["n"] == info["points"] and abs(info["length"] - 400) < 10, info
        P = a["points"]["p"]
        # AC frame: y up = local z, z = -local y
        assert np.allclose(P[0], [p[0, 0], z[0], -p[0, 1]], atol=0.01), P[0]
        assert np.all(np.diff(a["points"]["len"]) > 0) and list(a["points"]["id"][:3]) == [0, 1, 2]
        ex = a["extra"]
        speed, radius, sl, sr, fwd = ex[:, 0], ex[:, 4], ex[:, 5], ex[:, 6], ex[:, 13:16]
        assert np.allclose(sl, 5) and np.allclose(sr, 4)
        assert np.allclose(np.linalg.norm(fwd, axis=1), 1, atol=1e-3)
        assert abs(fwd[0, 0] - 1) < 0.05                    # along +x on the first straight
        mid_bend = np.argmin(np.hypot(P[:, 0] - (100 + r), P[:, 2]))
        mid_straight = np.argmin(np.hypot(P[:, 0] - 50, P[:, 2] - r))
        assert 20 < radius[mid_bend] < 45, radius[mid_bend]
        assert speed[mid_bend] < speed[mid_straight] - 3, (speed[mid_bend], speed[mid_straight])
        assert np.all(speed > 3) and np.all(speed <= ai_line.TOP_SPEED + 1e-3)
        assert 15 < info["lap_s"] < 60, info
        pit = ai_line.read_ai((Path(d) / "ai" / "pit_lane.ai").read_bytes())
        assert 90 < pit["n"] < 150, pit["n"]                 # 24 m of pit spots + 40 m each side
        assert np.all(pit["extra"][:, 0] <= 60 / 3.6 + 1e-3)
        assert pit["points"]["p"][0][0] < 20 and pit["points"]["p"][-1][0] > 44

        # point to point: no wrap
        q = np.column_stack([np.arange(0, 300, 2.0), np.zeros(150)])
        info2 = ai_line.write_ai_lines(Path(d) / "ai2", {"p": q, "z": np.zeros(150), "hl": 3, "hr": 3,
                                                          "loop": False}, [], log=print)
        b = ai_line.read_ai((Path(d) / "ai2" / "fast_lane.ai").read_bytes())
        assert b["n"] == info2["points"] and not (Path(d) / "ai2" / "pit_lane.ai").exists()
        assert abs(b["points"]["len"][-1] - 298) < 1.5
    print("ai_line_test OK")


if __name__ == "__main__":
    main()

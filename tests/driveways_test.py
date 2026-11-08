"""Driveway clean-up and tidy ground classes (no network).

  python tests/driveways_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box

from app import driveways, landcover


def ragged(poly, seed, amp=0.6):
    """A photo-like blob: frayed edge, a bush-sized hole."""
    rng = np.random.default_rng(seed)
    xy = np.asarray(poly.exterior.coords)[:-1]
    pts = []
    for a, b in zip(xy, np.roll(xy, -1, 0)):
        for f in np.linspace(0, 1, 12, endpoint=False):
            p = a + (b - a) * f
            pts.append(p + rng.uniform(-amp, amp, 2))
    return Polygon(pts).buffer(0)


def main():
    # a 7 m street along x (centre y = 0)
    street = np.column_stack([np.linspace(-100, 100, 201), np.zeros(201)])
    axes = driveways.Axes.build(lines=[(street, 3.5)])
    street_poly = LineString(street).buffer(3.5)
    house = box(-4, 26, 8, 36)

    # 1. straight driveway, 4 m wide, starting 3 m from the street edge, a bush (hole) in the middle
    ideal = box(-2, 6.5, 2, 25)
    d1 = ragged(ideal, 1).difference(Point(0, 15).buffer(1.0))
    # 2. diagonal driveway (45 deg) stopping 2.5 m short of the street
    diag = LineString([(30, 6.0), (42, 18)]).buffer(1.8, cap_style=2)
    d2 = ragged(diag, 2, 0.4)
    # 3. parking lot along the street, 20 x 12 m
    d3 = ragged(box(-60, 8, -40, 20), 3, 0.8)
    # 4. behind a house, far from the street
    d4 = ragged(box(60, 40, 66, 50), 4)
    out = driveways.regularize([d1, d2, d3, d4], axes, house, log=print)
    assert len(out) == 4, len(out)
    by = lambda p: min(out, key=lambda o: o.centroid.distance(p.centroid))
    o1, o2, o3, o4 = by(d1), by(d2), by(d3), by(d4)
    for name, o in (("straight", o1), ("diagonal", o2), ("lot", o3)):
        assert o.intersection(street_poly).area > 1.0, f"{name} driveway does not reach the street"
        assert len(o.interiors) == 0, f"{name} driveway has a hole"
    iou = o1.intersection(box(-2, 2.5, 2, 25)).area / o1.union(box(-2, 2.5, 2, 25)).area
    print(f"straight driveway: IoU {iou:.2f} with the ideal rectangle, width {o1.bounds[2] - o1.bounds[0]:.1f} m")
    assert iou > 0.8, iou
    assert o1.contains(Point(0, 15)), "the bush hole was not filled"
    # diagonal: still runs at 45 deg (its long side)
    mr = np.asarray(o2.minimum_rotated_rectangle.exterior.coords)
    e = max((mr[1] - mr[0], mr[2] - mr[1]), key=np.linalg.norm)
    ang = np.degrees(np.arctan2(abs(e[1]), abs(e[0])))
    print(f"diagonal driveway at {ang:.0f} deg")
    assert 35 < ang < 55, ang
    assert o4.distance(street_poly) > 20, "driveway behind a house was pulled to the street"
    assert all(o.intersection(house).area < 0.5 for o in out), "driveway under a house"

    # the measured track (half width 4.5 m) under a narrower mapped street (3.5 m): the driveway must reach
    # the track's edge, not stop at the street's
    class CL:
        p = street
        t = np.tile([1.0, 0.0], (len(street), 1))
    axes2 = driveways.Axes.build(CL, 4.5, 4.5, [(street, 3.5)])
    o = driveways.regularize([ragged(box(-2, 8, 2, 25), 5)], axes2, None, log=print)[0]
    print(f"driveway on a wide track starts at y = {o.bounds[1]:.1f} m (track edge 4.5 m)")
    assert o.bounds[1] < 4.0, o.bounds

    # tidy classes: a thin sand fringe round a driveway and a gravel speck in a lawn become lawn; a big sand
    # area stays; the paved area is untouched
    K = {k: i for i, k in enumerate(landcover._STYLE_ORDER)}
    res = 0.5
    cls = np.full((200, 200), K["lawn"], np.int8)
    cls[60:140, 90:100] = K["paved"]                     # 5 m driveway
    cls[60:140, 88:90] = K["sand"]                       # 1 m fringe
    cls[60:140, 100:102] = K["sand"]
    cls[20:24, 20:24] = K["gravel"]                       # 4 m2 speck
    cls[150:190, 150:190] = K["sand"]                     # 400 m2 sand pit
    houses = np.zeros_like(cls, bool)
    houses[10:30, 150:170] = True
    cls[30:34, 150:170] = K["gravel"]                     # eave shadow strip
    t = landcover.tidy_classes(cls.copy(), res, houses)
    assert (t[60:140, 90:100] == K["paved"]).all(), "paved changed"
    assert (t[60:140, 88:90] == K["lawn"]).all() and (t[60:140, 100:102] == K["lawn"]).all(), "fringe kept"
    assert (t[20:24, 20:24] == K["lawn"]).all(), "speck kept"
    assert (t[155:185, 155:185] == K["sand"]).all(), "sand pit lost"
    assert (t[30:34, 150:170] == K["lawn"]).all(), "strip by the house kept"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

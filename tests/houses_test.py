"""House footprints made buildable, doors and garage doors - no network.

  python tests/houses_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import LineString, Polygon, box

from app import houses, meshes


def main():
    # 1. a traced footprint with cut corners and a notch (as the photo gives) -> one clean rectangle, same area
    p = affinity.rotate(Polygon([(0, 0), (14, 0), (15, 1), (15, 9), (9, 9), (9, 10), (6, 10), (6, 9), (1, 9), (0, 8)]),
                        30, origin=(0, 0))
    kind, geo = houses.shape(p)
    assert kind == "rects" and len(geo) == 1, (kind, len(geo))
    r = Polygon(geo[0])
    assert abs(r.area - p.area) / p.area < 0.05, (r.area, p.area)
    e = np.diff(np.asarray(r.exterior.coords)[:3], axis=0)
    ang = np.degrees(np.arctan2(e[:, 1], e[:, 0])) % 90
    assert all(min(abs(a - 30), abs(a - 30 + 90), abs(a - 30 - 90)) < 2 for a in ang), ang   # house's own direction
    # 2. L and T houses -> a main block and wings (all right angles)
    L = Polygon([(0, 0), (12, 0), (12, 6), (6, 6), (6, 14), (0, 14)])
    kind, geo = houses.shape(affinity.rotate(L, -17, origin=(0, 0)))
    assert kind == "rects" and len(geo) == 2 and abs(sum(Polygon(g).area for g in geo) - L.area) < 6
    T = Polygon([(0, 0), (16, 0), (16, 7), (11, 7), (11, 13), (5, 13), (5, 7), (0, 7)])
    kind, geo = houses.shape(T)
    assert kind == "rects" and len(geo) == 2
    # 3. an odd small house shape (a chamfered corner 3 m wide) is still a rectangle, never a flat odd polygon
    odd = Polygon([(0, 0), (12, 0), (12, 7), (9, 10), (0, 10)])
    kind, geo = houses.shape(odd)
    assert kind == "rects", kind
    # 4. a big irregular building: flat roof, square corners, slanted wall kept straight (few vertices)
    B = Polygon([(0, 0), (60, 0), (60, 15), (45, 35), (0, 35)])
    kind, g = houses.shape(B)
    assert kind == "flat" and len(g.exterior.coords) <= 8 and abs(g.area - B.area) / B.area < 0.05

    # 5. doors: the road runs along y = -10 (south): the front door is in the middle of the south wall;
    # a driveway touching the east wall gets a garage door there, at the driveway
    rect = np.array([[0, 0], [12, 0], [12, 8], [0, 8]], float)            # counter-clockwise
    road = cKDTree(np.column_stack([np.linspace(-50, 50, 101), np.full(101, -10.0)]))
    dw = box(12.2, 1.0, 20.0, 4.5)
    sp = houses.door_specials(rect, road, [dw], ground_at=lambda x, y: 100.0)
    print("specials:", sp)
    assert sp.get(0) and list(sp[0].values()) == ["door"], sp          # wall 0 = south (outward normal -y)
    nb0 = int(round(12 / 3.4))
    assert list(sp[0].keys())[0] in ((nb0 - 1) // 2, nb0 // 2)
    assert sp.get(1) and list(sp[1].values())[0][0] == "garage", sp     # wall 1 = east
    assert list(sp[1].keys())[0] == 0 and list(sp[1].values())[0][1] == 100.0     # bay nearest the driveway
    # a bay hidden behind a wing gets no door
    wing = np.array([[3, -6], [9, -6], [9, 0.5], [3, 0.5]], float)
    sp2 = houses.door_specials(rect, road, [], others=[wing])
    assert all(k != 1 for k in sp2.get(0, {})), sp2
    # 6. steps in front of the door, facing out
    st = houses.steps(rect, {0: {1: "door"}}, floor0=100.6, base=100.0)
    assert len(st) == 2
    V = np.vstack([s[0] for s in st])
    assert V[:, 1].min() < -0.5 and V[:, 1].max() <= 0.01 and abs(V[:, 2].max() - 100.58) < 0.03

    # 7. facade: the door bay and the garage bay use their own texture rows; no foundation under the garage
    from app.textures import ROW_D, ROW_G, row_v

    a = rect
    V, F, UV = meshes._facade(a, np.roll(a, -1, axis=0), 99.4, 100.3, 103.1, specials={0: {1: "door"},
                                                                                      1: {0: ("garage", 100.0)}},
                              ground=100.0)
    vd = row_v(ROW_D)
    vg = row_v(ROW_G)
    assert np.isclose(UV[:, 1], vd[0]).any() and np.isclose(UV[:, 1], vg[0]).any()
    q = V.reshape(-1, 4, 3)
    uvq = UV.reshape(-1, 4, 2)
    garage = [k for k in range(len(q)) if np.isclose(uvq[k, :, 1], vg[0]).any()]
    assert len(garage) == 1 and abs(q[garage[0], :, 2].min() - 100.0) < 1e-6

    # 8. building types from OSM building=* (and shop tags), untagged ones by size
    K = houses.building_kind
    assert K({"tags": {"building": "garage"}}, 30) == "garage" and K({"tags": {"building": "garages"}}, 120) == "garage"
    assert K({"tags": {"building": "retail"}}, 600) == "commercial"
    assert K({"tags": {"building": "yes", "shop": "bakery"}}, 200) == "commercial"
    assert K({"tags": {"building": "warehouse"}}, 2000) == "industrial"
    assert K({"tags": {"building": "apartments", "building:levels": "4"}}, 300) == "block"
    assert K({"tags": {"building": "house"}}, 120) == "house" and K({"tags": {}}, 140) == "house"
    assert K({"tags": {}}, 20) == "garage" and K({"tags": {}}, 2500) == "commercial"

    # 9. a detached garage without a driveway: its doors on the wall facing the road (south, wall 0)
    g = np.array([[0, 0], [9, 0], [9, 6], [0, 6]], float)
    sp = houses.garage_specials(g, road, [], n_doors=2, ground_at=lambda x, y: 100.0)
    assert list(sp) == [0] and len(sp[0]) == 2 and all(v[0] == "garage" for v in sp[0].values()), sp
    assert houses.garage_specials(g, road, [], n_doors=-1)[0].keys() == {0, 1, 2}

    # 10. porch / door canopy: a shed roof under the eave, sloping outwards, posts and a deck for a porch,
    # the steps moved out in front of the deck
    sp = {0: {1: "door"}}
    roofs, trims, offs = houses.door_extras(rect, sp, 100.3, 100.0, porch=True, eave=102.9)
    assert len(roofs) == 1 and offs == {(0, 1): houses.PORCH_DEPTH}
    Vr = roofs[0][0]
    assert Vr[:, 2].max() <= 102.9 - 0.2 + 1e-9 and Vr[:, 2].min() >= 100.3 + 2.25 - 1e-9
    n = np.cross(Vr[1] - Vr[0], Vr[2] - Vr[0])
    assert n[2] > 0 and n[1] < 0, n                        # faces up and out (wall 0 faces -y)
    assert sum(1 for V, F, UV in trims if np.ptp(V[:, 2]) > 2.0) >= 2      # posts
    assert min(V[:, 1].min() for V, F, UV in trims) < -houses.PORCH_DEPTH + 0.01    # out to the porch edge
    st = houses.steps(rect, sp, 100.3, 100.0, offs)
    assert min(V[:, 1].min() for V, F, UV in st) < -houses.PORCH_DEPTH - 0.5        # in front of the deck
    roofs, trims, offs = houses.door_extras(rect, sp, 100.3, 100.0, porch=False, eave=102.9)
    assert len(roofs) == 1 and not offs and np.ptp(roofs[0][0][:, 0]) < 2.0          # a small canopy

    # 11. shops: shop fronts on the street wall, office windows above, a cornice on the parapet
    from app.textures import CORNICE_M, ROW_C, ROW_K, ROW_S
    from shapely.geometry import Polygon as _P

    shop = np.array([[0, 0], [20, 0], [20, 14], [0, 14]], float)
    levels, top = meshes._block_levels({"tags": {"building": "retail", "building:levels": "2"}}, _P(shop),
                                       "commercial", 100.15)
    assert len(levels) == 2 and abs(top - (100.15 + 4.0 + 3.2)) < 1e-6
    street = meshes._street_walls(shop, road)
    assert street == {0}, street
    V, F, UV = meshes._block_facade(shop, 99.4, 100.15, levels, top, "commercial", street, door_wall=0)
    q, uvq = V.reshape(-1, 4, 3), UV.reshape(-1, 4, 2)
    south = [k for k in range(len(q)) if np.allclose(q[k, :, 1], 0)]
    vs, vc, vk = row_v(ROW_S), row_v(ROW_C), row_v(ROW_K)
    assert any(np.isclose(uvq[k, :, 1], vs[0]).any() for k in south)            # shop fronts on the street
    north = [k for k in range(len(q)) if np.allclose(q[k, :, 1], 14)]
    assert not any(np.isclose(uvq[k, :, 1], vs[0]).any() for k in north)        # not at the back
    assert any(np.isclose(uvq[k, :, 1], vc[0]).any() for k in north)
    assert any(np.isclose(uvq[k, :, 1], vk[0]).any() and abs(q[k, :, 2].max() - (top + CORNICE_M)) < 1e-6
               for k in range(len(q)))
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

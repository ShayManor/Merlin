import numpy as np
from eval.drone import obstacles as ob

ROUTE = [[0.0, 0.75, 0.0], [4.0, 0.75, 0.0], [4.0, 0.75, 3.0]]   # 7 m L-shaped route

def test_usable_templates_filters_by_height():
    bb = {"tall": (0.7, 1.9, 0.7), "sofa": (2.1, 0.80, 0.96), "rug": (1.0, 0.018, 1.0)}
    keep = ob.usable_templates(bb)
    assert "tall" in keep and "sofa" in keep and "rug" not in keep

def test_route_points_count_and_altitude():
    pts = ob.route_points(ROUTE, 3, seed=0)
    assert pts.shape == (3, 3)
    assert np.allclose(pts[:, 1], 0.75)          # stay at route altitude

def test_route_points_avoid_start_and_goal():
    pts = ob.route_points(ROUTE, 3, seed=0, edge_margin=1.0, lateral_jitter=0.0)
    a, b = np.array(ROUTE[0]), np.array(ROUTE[-1])
    for p in pts:
        assert np.linalg.norm(p - a) > 0.99
        assert np.linalg.norm(p - b) > 0.99

def test_route_points_deterministic_in_seed():
    assert np.allclose(ob.route_points(ROUTE, 2, seed=7), ob.route_points(ROUTE, 2, seed=7))
    assert not np.allclose(ob.route_points(ROUTE, 2, seed=7), ob.route_points(ROUTE, 2, seed=8))

def test_straddles_alt():
    assert ob.straddles_alt(0.0, 1.9, 0.75)      # floor-standing tall object
    assert not ob.straddles_alt(0.0, 0.6, 0.75)  # too short: drone flies over
    assert not ob.straddles_alt(1.2, 2.0, 0.75)  # starts above the drone

def test_blocks_route_true_on_route_false_when_far():
    bbox = (0.7, 1.9, 0.7)
    assert ob.blocks_route(ROUTE, [2.0, 0.75, 0.0], bbox)
    assert not ob.blocks_route(ROUTE, [2.0, 0.75, 5.0], bbox)

def test_floor_lift_handles_wall_mounted_origin():
    assert abs(ob.floor_lift(0.0)) < 1e-9        # already floor-origin
    assert abs(ob.floor_lift(-1.066) - 1.066) < 1e-9   # wall cabinet must be lifted

def test_densify_matches_one_metre_spacing():
    d = ob.densify([[0.0, 0.75, 0.0], [3.0, 0.75, 0.0]], spacing=1.0)
    assert len(d) == 4                                  # start + 3 subgoals
    assert np.allclose(d[-1], [3.0, 0.75, 0.0])

def test_covers_waypoint_detects_buried_goal():
    wps = ob.densify(ROUTE, 1.0)
    bbox = (0.52, 1.49, 0.49)
    assert ob.covers_waypoint(wps[2], bbox, wps)         # sitting exactly on a waypoint
    far = np.array(wps[2]) + np.array([0.0, 0.0, 3.0])
    assert not ob.covers_waypoint(far, bbox, wps)

def test_shift_clear_of_waypoints_frees_a_buried_waypoint():
    wps = ob.densify(ROUTE, 1.0)
    bbox = (0.52, 1.49, 0.49)
    p = np.array(wps[2], float)
    assert ob.covers_waypoint(p, bbox, wps)
    q = ob.shift_clear_of_waypoints(p, np.array([0.0, 0.0, 1.0]), bbox, wps)
    assert q is not None
    assert not ob.covers_waypoint(q, bbox, wps)
    # still close enough to the route that the drone must react to it
    assert ob.blocks_route(ROUTE, q, bbox)

def test_shift_clear_works_for_a_large_obstacle():
    wps = ob.densify(ROUTE, 1.0)
    sofa = (2.14, 0.80, 0.96)
    q = ob.shift_clear_of_waypoints(np.array(wps[2], float), np.array([0.0, 0.0, 1.0]),
                                    sofa, wps)
    assert q is not None and not ob.covers_waypoint(q, sofa, wps)


def test_usable_templates_include_filters_by_substring():
    bb = {"frl_apartment_chair_05": (0.41, 0.90, 0.51),
          "frl_apartment_umbrella": (0.35, 0.96, 0.14),
          "frl_apartment_wall_cabinet_01": (0.49, 1.96, 0.38)}
    assert set(ob.usable_templates(bb)) == set(bb)
    assert ob.usable_templates(bb, include=["chair"]) == ["frl_apartment_chair_05"]
    assert set(ob.usable_templates(bb, include=["chair", "umbrella"])) == {
        "frl_apartment_chair_05", "frl_apartment_umbrella"}
    assert ob.usable_templates(bb, include=["nothing_matches"]) == []


def test_usable_templates_include_still_respects_height_and_width():
    bb = {"frl_apartment_chair_05": (0.41, 0.90, 0.51),
          "frl_apartment_chair_tiny": (0.41, 0.30, 0.51)}
    assert ob.usable_templates(bb, include=["chair"]) == ["frl_apartment_chair_05"]
    assert ob.usable_templates(bb, include=["chair"], min_width=0.9) == []

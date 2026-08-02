"""Closed-loop planner behaviour against analytic geometry (no habitat / no PX4).

These are the cases the real C4 episodes hit: an obstacle straddling flight altitude that
the route walks straight into. With perfect depth the drone must get around it; blind it
must not.
"""
import numpy as np
import pytest

from eval.drone.simnav import Boxes, fly

ALT = 0.75
ROUTE = [np.array([0.0, ALT, 0.0]), np.array([3.0, ALT, 0.0]), np.array([6.0, ALT, 0.0])]
# a wide cabinet between the first two waypoints: 1.6 m wide, 0.4 m deep, 1.2 m tall.
# It sits BETWEEN waypoints, never on one -- an obstacle swallowing a waypoint is
# unwinnable for every arm, which is what obstacles.shift_clear_of_waypoints prevents.
CABINET = Boxes([([1.5, 0.6, 0.0], [0.2, 0.6, 0.8])])


def test_depth_sees_the_cabinet_ahead():
    from eval.drone.simnav import look_at_R
    R = look_at_R([1.0, 0, 0])
    f = 48 / (2 * np.tan(np.deg2rad(45.0)))
    K = np.array([[f, 0, 24.0], [0, f, 24.0], [0, 0, 1.0]])
    d = CABINET.depth(np.array([0.0, ALT, 0.0]), R, K)
    assert abs(d[24, 24] - 1.3) < 0.15          # front face at x=1.3
    assert d[24, 0] >= 7.0                      # nothing at the image edge


def test_clear_corridor_reaches_goal():
    r = fly(Boxes([([0.0, -5.0, 0.0], [0.1, 0.1, 0.1])]), ROUTE)
    assert r["ended"] == "goal"


def test_blind_arm_crashes_into_the_cabinet():
    r = fly(CABINET, ROUTE, blind=True)
    assert r["ended"] == "collision"


def test_perception_arm_gets_around_the_cabinet():
    r = fly(CABINET, ROUTE)
    assert r["ended"] == "goal", r["ended"]
    assert not CABINET.hit(r["path"][-1], 0.18)


def test_greedy_planner_stalls_without_escape():
    """Documents why escape exists: the same episode times out when it is disabled."""
    r = fly(CABINET, ROUTE, cfg={"escape": False})
    assert r["ended"] == "timeout"


@pytest.mark.parametrize("off", [-0.4, 0.0, 0.4])
def test_gets_around_offset_obstacles(off):
    b = Boxes([([1.5, 0.6, off], [0.25, 0.6, 0.7])])
    r = fly(b, ROUTE)
    assert r["ended"] == "goal", (off, r["ended"])


def test_three_obstacles_in_series():
    b = Boxes([([1.3, 0.6, -0.3], [0.2, 0.6, 0.6]),
               ([3.6, 0.6, 0.4], [0.2, 0.6, 0.6]),
               ([4.9, 0.6, -0.2], [0.2, 0.6, 0.6])])
    r = fly(b, ROUTE, cfg={"timeout": 120.0})
    assert r["ended"] == "goal", r["ended"]


def test_skips_a_subgoal_buried_in_an_obstacle():
    """An obstacle placed ON a waypoint must not deadlock the route: the planner cannot
    reach the subgoal, so the loop drops it and steers at the next one."""
    b = Boxes([([3.0, 0.6, 0.0], [0.3, 0.6, 0.6])])      # swallows ROUTE[1]
    r = fly(b, ROUTE)
    assert r["ended"] == "goal", r["ended"]


def test_will_not_fly_into_unobserved_space():
    """The camera sees 90 deg forward; everything else is unknown, not empty.

    Without this the planner treats never-observed voxels as free and flies sideways into
    geometry it has never seen -- measured in habitat as collisions where the nearest MAPPED
    voxel was still 0.26-0.52 m away, even with perfect depth.
    """
    from eval.drone.mapper import VoxelMap
    from eval.drone.planner import Planner3D
    vmap = VoxelMap(voxel=0.10, min_hits=1)
    cur = np.array([0.0, ALT, 0.0])
    # nothing observed at all: every direction is unknown, so the planner must not commit
    p_known = Planner3D(step=0.4, safety=0.30, require_known=True)
    tgt, blocked = p_known.next_setpoint(vmap, cur, np.array([4.0, ALT, 0.0]))
    assert blocked and np.allclose(tgt, cur)
    # the same map with the check off looks completely free
    p_blind = Planner3D(step=0.4, safety=0.30, require_known=False)
    tgt, blocked = p_blind.next_setpoint(vmap, cur, np.array([4.0, ALT, 0.0]))
    assert not blocked and tgt[0] > cur[0]


def test_observed_space_opens_up_after_a_keyframe():
    """One keyframe carves the forward cone, which is what lets the drone start moving."""
    r = fly(CABINET, ROUTE)
    assert r["ended"] == "goal"
    assert r["path"].shape[0] > 5          # it did move, rather than freezing at the start

import numpy as np

from eval.drone.gen_nearfield import BODY_R, central_depth, forward_of, push_in


def look_z():
    """Camera matrix (right/up/back) for a camera looking along -Z."""
    return np.eye(3)


class FakeClient:
    """A wall perpendicular to -Z at z = wall_z. Depth at a camera position is the distance
    to it, so pushing the camera forward must reduce the reported depth by the same amount."""

    def __init__(self, wall_z=-3.0, solid_from=None):
        self.wall_z = wall_z
        self.solid_from = solid_from
        self.renders = []

    def render(self, p, R):
        d = abs(float(p[2]) - self.wall_z)
        self.renders.append(np.asarray(p, float).copy())
        return {"depth": np.full((40, 40), d, np.float32),
                "rgb": np.zeros((40, 40, 3), np.uint8)}

    def collide(self, p, radius):
        hit = self.solid_from is not None and float(p[2]) <= self.solid_from
        return {"hit": hit}


def test_forward_is_minus_z_of_the_camera_matrix():
    assert np.allclose(forward_of(np.eye(3)), [0, 0, -1])


def test_central_depth_ignores_a_stray_near_edge_pixel():
    d = np.full((40, 40), 2.0, np.float32)
    d[0, 0] = 0.05
    assert central_depth(d) == 2.0


def test_central_depth_returns_none_when_the_frame_is_mostly_invalid():
    assert central_depth(np.zeros((40, 40), np.float32)) is None


def test_push_in_places_the_surface_at_the_requested_distance():
    c = FakeClient(wall_z=-3.0)
    out = push_in(c, np.zeros(3), look_z(), target=0.35, max_step=5.0)
    assert out is not None
    assert float(np.median(out["depth"])) == np.float32(0.35)
    assert c.renders[-1][2] == -2.65     # moved 2.65 m toward the wall


def test_push_in_refuses_a_pose_the_drone_body_could_not_occupy():
    c = FakeClient(wall_z=-3.0, solid_from=-2.0)
    assert push_in(c, np.zeros(3), look_z(), target=0.35, max_step=5.0) is None


def test_push_in_declines_when_the_surface_is_already_closer_than_target():
    c = FakeClient(wall_z=-0.2)
    assert push_in(c, np.zeros(3), look_z(), target=0.35, max_step=5.0) is None


def test_push_in_declines_when_nothing_is_in_range():
    c = FakeClient(wall_z=-40.0)
    assert push_in(c, np.zeros(3), look_z(), target=0.35, max_step=5.0) is None


def test_max_step_bounds_the_move_and_then_the_frame_is_rejected_as_not_near():
    """A capped step leaves the surface far away, so the near-fraction filter must drop it
    rather than silently writing a mid-range frame into the near-field set."""
    c = FakeClient(wall_z=-10.0)
    assert push_in(c, np.zeros(3), look_z(), target=0.35, max_step=0.5) is None


def test_body_radius_is_the_one_used_for_the_collision_query():
    seen = {}
    c = FakeClient(wall_z=-3.0)
    c.collide = lambda p, radius: seen.setdefault("r", radius) and {"hit": False}
    push_in(c, np.zeros(3), look_z(), target=0.35, max_step=5.0)
    assert seen["r"] == BODY_R

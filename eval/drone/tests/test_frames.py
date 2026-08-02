import numpy as np
from eval.drone.frames import (ned_to_hab_pos, hab_to_ned_pos, R_NED2HAB,
                               quat_to_R, ned_quat_to_hab_R)

def test_pos_roundtrip():
    p = np.array([3.0, -2.0, -5.0])  # N=3, E=-2, D=-5 (5 m up)
    h = ned_to_hab_pos(p)
    # up (negative D) must map to +Y in habitat
    assert h[1] == 5.0
    assert np.allclose(hab_to_ned_pos(h), p)

def test_basis_is_rotation():
    assert np.allclose(R_NED2HAB @ R_NED2HAB.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R_NED2HAB), 1.0)

def test_identity_orientation_faces_forward():
    # FRD body aligned with NED (identity quat) -> camera forward (world -Z_hab? we check
    # the mapped forward axis is horizontal, i.e. zero vertical component).
    R = ned_quat_to_hab_R(np.array([0.0, 0.0, 0.0, 1.0]))
    fwd = R @ np.array([0.0, 0.0, -1.0])  # camera looks down local -Z
    assert abs(fwd[1]) < 1e-9  # level flight -> forward is horizontal


def test_backprojection_lands_in_front_of_the_camera():
    """The map must place geometry where the camera is looking.

    render() is driven with a Habitat camera matrix (looks down local -Z, +Y up) while
    mapper.backproject() returns the optical convention (+Z forward, +Y down). Handing the
    Habitat matrix straight to integrate() reflects every point through the camera -- behind
    it and upside down -- which is silent: the map fills up, the planner reacts to it, and
    nothing errors. Measured on the real renderer: 0% of points landed in front.
    """
    import numpy as np
    from eval.drone.frames import EpisodeFrame, HAB_CAM_TO_OPTICAL
    for yaw in (0.0, 0.7, -2.1, 3.0):
        ep = EpisodeFrame([1.0, 0.75, 2.0], yaw)
        R_hab = ep.ned_quat_to_hab_R(np.array([0.0, 0.0, 0.0, 1.0]))
        fwd = -R_hab[:, 2]                       # Habitat camera forward
        R_opt = R_hab @ HAB_CAM_TO_OPTICAL
        # a pixel at the image centre with depth d back-projects to (0, 0, d) in optical
        p = R_opt @ np.array([0.0, 0.0, 2.5])
        assert float(p @ fwd) > 2.49, "centre pixel must land ahead, at its measured depth"
        # and a pixel below the centre must map below the camera, not above
        below = R_opt @ np.array([0.0, 1.0, 2.5])
        assert below[1] < 0.0, "image-down must be world-down"

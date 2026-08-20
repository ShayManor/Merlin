"""Geometry tests for the VO front end. No cv2 here: the helpers under test are the ones
that decide whether a chained monocular trajectory is metrically self-consistent, and they
are pure numpy so they run on a machine without OpenCV.
"""
import numpy as np

from eval.vo import R_to_quat, chain, intrinsics, relative_scale
from eval.drone.frames import quat_to_R


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def test_intrinsics_match_a_90_degree_camera():
    K = intrinsics(504, 504, 90.0)
    assert abs(K[0, 0] - 252.0) < 1e-6      # f = (w/2)/tan(45) = w/2
    assert abs(K[0, 2] - 252.0) < 1e-6


def test_R_to_quat_roundtrips_through_quat_to_R():
    """The quaternion must come back in frames.quat_to_R's convention, or every VO pose is
    silently rotated when run_c2 converts it back."""
    for a in (0.0, 0.4, 2.0, 3.0):
        for R in (_Rz(a), np.array([[1., 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])):
            assert np.allclose(quat_to_R(R_to_quat(R)), R, atol=1e-8)


def test_chain_recovers_a_known_camera_path():
    """Per-pair (R, t) map frame k into frame k+1; chained centres must land on the truth."""
    centres_true = [np.zeros(3), np.array([1., 0, 0]), np.array([1., 1., 0]), np.array([0., 1., 0])]
    rots_true = [_Rz(0.0), _Rz(0.3), _Rz(0.6), _Rz(0.9)]      # cam -> frame0
    rels = []
    for k in range(3):
        R_k = rots_true[k].T                                   # frame0 -> cam_k
        R_k1 = rots_true[k + 1].T
        R_rel = R_k1 @ R_k.T
        t_rel = R_k1 @ (centres_true[k] - centres_true[k + 1])
        rels.append((R_rel, t_rel))
    C, rots = chain(rels)
    assert np.allclose(C, np.array(centres_true), atol=1e-9)
    for got, want in zip(rots, rots_true):
        assert np.allclose(got, want, atol=1e-9)


def test_relative_scale_is_the_median_depth_ratio():
    prev = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]  # >= min_pts, or it falls back
    curr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]       # current pair is half the size
    assert abs(relative_scale(prev, curr) - 2.0) < 1e-9


def test_relative_scale_ignores_outliers():
    """Triangulation near the epipole produces heavy-tailed depths; a mean would follow them."""
    prev = [2.0] * 12 + [1e6]
    curr = [1.0] * 12 + [1.0]
    assert abs(relative_scale(prev, curr) - 2.0) < 1e-6


def test_relative_scale_falls_back_to_unity_without_overlap():
    """Too little overlap must degrade to constant speed, not to an arbitrary number."""
    assert relative_scale([1.0, 2.0], [1.0, 2.0]) == 1.0
    assert relative_scale([], []) == 1.0


def test_chain_with_unit_translations_gives_equal_segments():
    """The failure mode scale propagation exists to prevent: unscaled chaining makes every
    segment the same length regardless of the true motion."""
    rels = [(np.eye(3), np.array([0., 0., -1.])) for _ in range(4)]
    C, _ = chain(rels)
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    assert np.allclose(seg, seg[0])

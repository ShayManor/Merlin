import numpy as np
from eval.drone import noise as nz

def _img():
    return np.full((32, 32, 3), 120, np.uint8)

def test_rgb_noise_zero_sigma_is_identity():
    im = _img()
    assert np.array_equal(nz.add_rgb_noise(im, 0.0, np.random.default_rng(0)), im)

def test_rgb_noise_changes_pixels_and_stays_uint8():
    out = nz.add_rgb_noise(_img(), 8.0, np.random.default_rng(0))
    assert out.dtype == np.uint8
    assert not np.array_equal(out, _img())
    assert out.min() >= 0 and out.max() <= 255

def test_rgb_noise_clips_at_bounds():
    dark = np.zeros((8, 8, 3), np.uint8)
    out = nz.add_rgb_noise(dark, 50.0, np.random.default_rng(1))
    assert out.min() >= 0 and out.max() <= 255

def test_motion_blur_zero_speed_is_identity():
    im = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    assert np.array_equal(nz.motion_blur(im, 0.0), im)

def test_motion_blur_smooths_an_edge():
    im = np.zeros((16, 16, 3), np.uint8)
    im[:, 8:] = 255
    out = nz.motion_blur(im, 2.0)
    col = out[8, :, 0].astype(int)
    assert len(set(col.tolist())) > 2          # intermediate values appeared

def test_perturb_pose_zero_sigma_is_identity():
    p, R = np.array([1.0, 2.0, 3.0]), np.eye(3)
    p2, R2 = nz.perturb_pose(p, R, 0.0, 0.0, np.random.default_rng(0))
    assert np.allclose(p, p2) and np.allclose(R, R2)

def test_perturb_pose_stays_a_rotation():
    p2, R2 = nz.perturb_pose(np.zeros(3), np.eye(3), 0.05, 2.0, np.random.default_rng(0))
    assert np.allclose(R2 @ R2.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R2) - 1.0) < 1e-9


def test_random_depth_is_smooth_and_in_range():
    """The uninformative-depth control must produce a map of the same character as a real
    one -- coherent surfaces in the same range -- or it degenerates into flying blind."""
    import numpy as np
    from eval.drone.noise import random_depth
    d = random_depth((64, 64), np.random.default_rng(0))
    assert d.shape == (64, 64)
    assert 0.3 <= d.min() and d.max() <= 3.0
    # neighbouring pixels differ far less than the overall spread: this is a surface, not
    # per-pixel speckle that the occupancy threshold would simply discard
    assert np.abs(np.diff(d, axis=1)).mean() < 0.1 * (d.max() - d.min())


def test_random_depth_is_deterministic_in_seed():
    import numpy as np
    from eval.drone.noise import random_depth
    a = random_depth((32, 32), np.random.default_rng(3))
    b = random_depth((32, 32), np.random.default_rng(3))
    c = random_depth((32, 32), np.random.default_rng(4))
    assert np.allclose(a, b) and not np.allclose(a, c)

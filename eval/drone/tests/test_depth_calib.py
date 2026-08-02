import numpy as np
import pytest

from eval.drone.depth_calib import R_HI, R_LO, DepthCalib


def fill(tab, pred, ratios, shape=(28, 28), reps=40):
    """Accumulate frames where every pixel predicts `pred` and gt/pred cycles through
    `ratios`."""
    p = np.full(shape, float(pred))
    for _ in range(reps):
        for r in ratios:
            tab.accumulate(p, p * r)


def test_identity_when_prediction_is_perfect():
    t = DepthCalib(grid=4)
    fill(t, 1.0, [1.0])
    d = np.full((28, 28), 1.0, np.float32)
    assert np.allclose(t.apply(d, q=0.2), 1.0, atol=0.03)


def test_low_quantile_pulls_a_noisy_cell_in_and_leaves_a_tight_one_alone():
    """The whole point: the correction keys on the SPREAD of gt/pred, not its centre. Two
    cells with the same median ratio must be corrected differently when one is noisy."""
    t = DepthCalib(grid=4)
    fill(t, 0.4, [0.7, 0.85, 1.0, 1.15, 1.3])     # noisy near-field cell, median 1.0
    fill(t, 2.0, [0.98, 0.99, 1.0, 1.01, 1.02])   # tight far cell, median 1.0
    r = t.ratio_map(q=0.2, min_count=10)
    near = r[:, :, np.searchsorted(t.pred_edges, 0.4) - 1].mean()
    far = r[:, :, np.searchsorted(t.pred_edges, 2.0) - 1].mean()
    assert near < 0.9, near
    assert far > 0.97, far


def test_median_quantile_reproduces_the_centre():
    t = DepthCalib(grid=4)
    fill(t, 1.0, [0.7, 0.85, 0.9, 0.95, 1.0])     # median 0.9
    r = t.ratio_map(q=0.5, min_count=10)[:, :, np.searchsorted(t.pred_edges, 1.0) - 1]
    assert abs(float(r.mean()) - 0.9) < 0.05


def test_correction_never_pushes_a_surface_away():
    """A ratio above 1 would move mapped geometry further from the drone, which is the
    direction that caused the measured collisions. It must be clamped out."""
    t = DepthCalib(grid=4)
    fill(t, 1.0, [1.4, 1.5, 1.6])
    d = np.full((28, 28), 1.0, np.float32)
    assert (t.apply(d, q=0.2, min_count=10) <= 1.0 + 1e-6).all()
    assert (t.ratio_map(q=0.2, min_count=10) <= R_HI).all()


def test_correction_is_clamped_below():
    t = DepthCalib(grid=4)
    fill(t, 1.0, [0.4, 0.41, 0.42])
    assert (t.ratio_map(q=0.2, min_count=10) >= R_LO - 1e-9).all()


def test_far_surfaces_are_not_dragged_in_by_the_relative_clamp():
    """R_LO is relative, so on its own it lets a 2.5 m wall move in by over a metre. That is a
    corridor-narrowing global margin change, which is the thing already known to fail."""
    t = DepthCalib(grid=4)
    fill(t, 2.5, [0.5])
    d = np.full((28, 28), 2.5, np.float32)
    assert float(t.apply(d, q=0.2, min_count=10).mean()) == pytest.approx(2.15, abs=0.02)
    assert float(t.apply(d, q=0.2, min_count=10, max_pull=0).mean()) < 1.6


def test_near_surfaces_are_still_correctable_by_a_large_fraction():
    t = DepthCalib(grid=4)
    fill(t, 0.4, [0.5])
    d = np.full((28, 28), 0.4, np.float32)
    assert float(t.apply(d, q=0.2, min_count=10).mean()) == pytest.approx(0.4 * R_LO, abs=0.01)


def test_sparse_cells_fall_back_to_identity_not_to_noise():
    t = DepthCalib(grid=4)
    p = np.full((28, 28), 1.0)
    t.accumulate(p, p * 0.6)                      # 784 samples, one pred bin only
    r = t.ratio_map(q=0.2, min_count=10_000_000)  # nothing clears the threshold
    assert np.allclose(r, 1.0)


def test_pooled_fallback_used_when_a_block_is_empty_but_the_bin_is_not():
    t = DepthCalib(grid=4)
    p = np.zeros((28, 28))
    p[:7] = 1.0                                   # only the top row of blocks sees anything
    g = p * 0.7
    for _ in range(50):
        t.accumulate(p, g)
    j = np.searchsorted(t.pred_edges, 1.0) - 1
    r = t.ratio_map(q=0.2, min_count=200)[:, :, j]
    assert abs(float(r.mean()) - 0.7) < 0.05      # empty blocks inherit the pooled ratio


def test_apply_is_continuous_across_a_predicted_depth_bin_edge():
    """A step lookup would inject a fake depth discontinuity at every bin edge, and the
    mapper turns depth discontinuities into voxels."""
    t = DepthCalib(grid=4)
    for pred, ratio in [(0.3, 0.7), (0.5, 0.8), (0.9, 0.9), (1.6, 1.0), (3.0, 1.0)]:
        fill(t, pred, [ratio], reps=60)
    xs = np.linspace(0.2, 3.0, 400, dtype=np.float32)
    out = t.apply(np.tile(xs, (28, 1)), q=0.5, min_count=10)[0]
    assert np.abs(np.diff(out)).max() < 0.03      # a step lookup measures 0.11 here
    assert (np.diff(out) > -0.02).all()           # still monotone in depth


def test_apply_preserves_invalid_pixels():
    t = DepthCalib(grid=4)
    fill(t, 1.0, [0.8])
    d = np.full((28, 28), 1.0, np.float32)
    d[0, 0] = np.nan
    d[0, 1] = 0.0
    out = t.apply(d, q=0.2, min_count=10)
    assert np.isnan(out[0, 0]) and out[0, 1] == 0.0
    assert out[1, 1] < 1.0


def test_blocks_are_resolution_independent():
    t = DepthCalib(grid=4)
    fill(t, 1.0, [0.8], shape=(28, 28))
    small = t.apply(np.full((28, 28), 1.0, np.float32), q=0.2, min_count=10)
    large = t.apply(np.full((56, 56), 1.0, np.float32), q=0.2, min_count=10)
    assert abs(float(small.mean()) - float(large.mean())) < 1e-3


def test_save_load_round_trip(tmp_path):
    t = DepthCalib(grid=4)
    fill(t, 1.0, [0.7, 0.9, 1.1])
    p = tmp_path / "calib.npz"
    t.save(str(p))
    t2 = DepthCalib.load(str(p))
    d = np.full((28, 28), 1.0, np.float32)
    assert np.array_equal(t.counts, t2.counts)
    assert np.allclose(t.apply(d, q=0.2, min_count=10), t2.apply(d, q=0.2, min_count=10))


def test_shape_mismatch_is_an_error():
    t = DepthCalib(grid=4)
    with pytest.raises(ValueError):
        t.accumulate(np.ones((8, 8)), np.ones((8, 9)))


def test_spread_reports_the_pixel_axis():
    """If the table has collapsed to a global rescale the per-block IQR is ~0, and that has
    to be visible rather than assumed away."""
    t = DepthCalib(grid=4)
    p = np.full((28, 28), 1.0)
    g = np.full((28, 28), 1.0)
    g[:14] = 0.7                                  # top half of the image reads long
    for _ in range(60):
        t.accumulate(p, g)
    j = np.searchsorted(t.pred_edges, 1.0) - 1
    row = [r for r in t.spread(q=0.2, min_count=100) if r[0] <= 1.0 < r[1]][0]
    assert row[2] > 0
    assert row[4] > 0.2, row                      # blocks genuinely disagree
    flat = DepthCalib(grid=4)
    fill(flat, 1.0, [0.7])
    assert [r for r in flat.spread(q=0.2, min_count=100) if r[0] <= 1.0 < r[1]][0][4] < 0.02
    assert j >= 0

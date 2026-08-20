"""Contract and behaviour tests for the alternative occupancy representations.

Two kinds of test here. The contract tests pin the drop-in interface the planner relies on, so
a new map cannot silently fail to be substitutable. The behaviour tests pin the one property
each method exists for, measured against a synthetic wall with the student's own measured
error profile: a displaced, noisy near-field surface.
"""
import numpy as np
import pytest

from eval.drone.maps import (DEFAULT_SIGMA, LogOddsMap, NanoMapLocal, RiskInflatedMap,
                             build_map, sigma_lookup)
from eval.drone.mapper import VoxelMap

RES, HFOV = 48, 90.0
F = RES / (2 * np.tan(np.deg2rad(HFOV) / 2))
K = np.array([[F, 0, RES / 2], [0, F, RES / 2], [0, 0, 1.0]])
R = np.eye(3)                      # optical +Z forward, camera at origin looking down +Z
ALL = ["voxel", "logodds", "nanomap", "risk"]


def wall_depth(dist, bias=0.0, noise=0.0, seed=0):
    """A fronto-parallel wall at `dist`, optionally displaced and noisy like the student."""
    rng = np.random.default_rng(seed)
    d = np.full((RES, RES), dist + bias, np.float32)
    if noise:
        d = d + rng.normal(0, noise, d.shape).astype(np.float32)
    return np.clip(d, 0.05, None)


def feed(m, dist, n=6, **kw):
    for i in range(n):
        m.integrate(wall_depth(dist, seed=i, **kw), K, R, np.zeros(3), max_range=8.0)
    return m


# --- contract ----------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL)
def test_drop_in_contract(name):
    m = build_map(name, voxel=0.1, min_hits=2)
    feed(m, 1.5)
    q = np.array([[0.0, 0.0, 1.5], [0.0, 0.0, 0.4]])
    assert m.any_occupied_within(q, 0.3).shape == (2,)
    assert m.unknown(q).shape == (2,)
    assert np.isfinite(m.nearest_obstacle_dist(np.zeros(3)))
    assert m.occupied_points().ndim == 2
    assert m._occ_codes().ndim == 1
    assert m.voxel == 0.1


@pytest.mark.parametrize("name", ALL)
def test_empty_map_blocks_nothing(name):
    m = build_map(name, voxel=0.1)
    q = np.array([[0.0, 0.0, 1.0]])
    assert not m.any_occupied_within(q, 0.5).any()
    assert m.unknown(q).all()


@pytest.mark.parametrize("name", ALL)
def test_sees_a_clean_wall(name):
    m = feed(build_map(name, voxel=0.1, min_hits=2), 1.5)
    assert m.any_occupied_within(np.array([[0.0, 0.0, 1.5]]), 0.25).all()
    assert not m.any_occupied_within(np.array([[0.0, 0.0, 0.5]]), 0.25).any()


# --- behaviour ---------------------------------------------------------------------

def test_sigma_lookup_is_piecewise_and_covers_the_range():
    f = sigma_lookup()
    b, s = f(np.array([0.1, 0.3, 1.0, 2.0, 20.0]))
    assert s.shape == (5,)
    assert (s > 0).all()
    # Measured for student_nf: sigma rises monotonically with range, 0.163 -> 0.475. The
    # near-field fine-tune made the near field the MOST reliable band, which is the opposite
    # of the pre-fine-tune table in CLAUDE.md.
    mid = np.array([r[3] for r in DEFAULT_SIGMA])
    assert (np.diff(mid) > 0).all(), f"sensor model is no longer monotone: {mid}"


def test_sigma_bias_flips_sign_with_range():
    """Near surfaces are pushed away, far ones pulled in. A single global offset correction
    would therefore be wrong at one end or the other, which is why the bump is de-biased
    per range bin rather than by a constant."""
    f = sigma_lookup()
    b, _ = f(np.array([0.3, 3.0]))
    assert b[0] > 0 > b[1]


def test_logodds_bump_recovers_a_displaced_surface():
    """The measured failure: the near face is placed too far, so a hard deposit puts the map's
    only evidence behind the true surface and the planner's block fires late. Spreading the
    deposit over +-k*sigma should still mark the true face."""
    true_d, bias = 0.45, 0.25            # surface at 0.45 m, predicted at 0.70 m
    q = np.array([[0.0, 0.0, true_d]])
    hard = feed(VoxelMap(voxel=0.1, min_hits=2), true_d, bias=bias)
    soft = feed(LogOddsMap(voxel=0.1, k_sigma=2.0, l_thresh=0.4), true_d, bias=bias)
    assert not hard.any_occupied_within(q, 0.05).any(), "fixture no longer displaces the wall"
    assert soft.any_occupied_within(q, 0.05).any(), "bump failed to cover the true surface"


def test_logodds_downweights_the_far_field():
    """The whole point of inverse-variance weighting: an obstacle first seen at 2 m, where
    sigma is 0.407, must not reach the confidence of the same surface seen at 0.35 m, where it
    is 0.163. This is the sighting range at which obstacles actually enter the map (1.62-1.98 m
    measured), so it is the case that decides whether the map lands inflated or deflated.
    l_max is raised past saturation or this measures the cap instead of the weight."""
    near = feed(LogOddsMap(voxel=0.1, l_thresh=99, l_max=1e6), 0.35, n=4)
    far = feed(LogOddsMap(voxel=0.1, l_thresh=99, l_max=1e6), 2.0, n=4)
    assert max(far._l.values()) < max(near._l.values())


def test_logodds_threshold_is_monotone_conservatism_knob():
    m = feed(LogOddsMap(voxel=0.1, l_thresh=0.2), 1.2, noise=0.15)
    counts = []
    for t in (0.2, 1.0, 2.0, 3.5):
        m.l_thresh = t
        m._codes_cache = None
        counts.append(len(m._occ_codes()))
    assert counts == sorted(counts, reverse=True), f"not monotone: {counts}"


def test_nanomap_does_not_accumulate_noise_across_frames():
    """The point of not fusing: a run of noisy frames must not grow the occupied set the way
    a counting map does, because each query is answered from one view."""
    nm = feed(NanoMapLocal(voxel=0.1, window=12), 1.2, n=12, noise=0.25)
    vm = feed(VoxelMap(voxel=0.1, min_hits=2), 1.2, n=12, noise=0.25)
    assert len(nm._occ_codes()) < len(vm._occ_codes())


def test_nanomap_window_forgets_old_frames():
    m = NanoMapLocal(voxel=0.1, window=3)
    feed(m, 1.0, n=5)
    assert len(m.frames) == 3


def test_nanomap_agree_is_a_conservatism_knob():
    q = np.stack([np.zeros(40), np.zeros(40), np.linspace(0.6, 1.6, 40)], 1)
    lo = feed(NanoMapLocal(voxel=0.1, window=8, agree=1), 1.2, n=8, noise=0.3)
    hi = feed(NanoMapLocal(voxel=0.1, window=8, agree=6), 1.2, n=8, noise=0.3)
    assert hi.any_occupied_within(q, 0.2).sum() <= lo.any_occupied_within(q, 0.2).sum()


def test_risk_inflation_grows_with_risk_k():
    # The wall is at 1.3 m but the log-odds bump already spreads it to ~0.94 m, so the query
    # sits at 0.6 m: outside the bump at radius 0.15, inside it once 3 sigma of margin is added.
    q = np.array([[0.0, 0.0, 0.6]])
    zero = feed(RiskInflatedMap(voxel=0.1, risk_k=0.0, l_thresh=0.4), 1.3)
    wide = feed(RiskInflatedMap(voxel=0.1, risk_k=3.0, l_thresh=0.4), 1.3)
    assert not zero.any_occupied_within(q, 0.15).any()
    assert wide.any_occupied_within(q, 0.15).any()


def test_risk_reduces_to_logodds_at_zero_risk():
    """k_sigma must be matched explicitly: RiskInflatedMap defaults it to 0 so the uncertainty
    is charged once, as clearance, rather than once as a bump and again as a margin."""
    q = np.stack([np.zeros(30), np.zeros(30), np.linspace(0.5, 2.0, 30)], 1)
    a = feed(RiskInflatedMap(voxel=0.1, risk_k=0.0, k_sigma=2.0, l_thresh=0.4), 1.3)
    b = feed(LogOddsMap(voxel=0.1, k_sigma=2.0, l_thresh=0.4), 1.3)
    assert (a.any_occupied_within(q, 0.2) == b.any_occupied_within(q, 0.2)).all()


def test_build_map_filters_unknown_kwargs():
    m = build_map("voxel", voxel=0.2, risk_k=9.9, agree=3)   # extras must not raise
    assert isinstance(m, VoxelMap) and m.voxel == 0.2


def test_default_sigma_table_is_ordered_and_contiguous():
    for a, b in zip(DEFAULT_SIGMA, DEFAULT_SIGMA[1:]):
        assert a[1] == b[0], "bins must tile the range without gaps"
        assert a[3] > 0 and b[3] > 0

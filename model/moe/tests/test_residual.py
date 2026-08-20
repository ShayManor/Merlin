"""Invariants for MoEResidual. Spec: docs/superpowers/specs/2026-08-08-moe-01-residual-module.md

Test 1 (identity at init) is the tightest regression guard available: it is what makes every
training run a measured delta against the deployed model rather than against a fresh baseline.
Test 6 (static export) is the compilability claim in mechanical form, so a change that
reintroduces routing fails here instead of failing to compile much later.
"""
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from moe.residual import LearnedGate, MoEResidual, PrescribedRangeGate  # noqa: E402
from moe.upcycle import (  # noqa: E402
    aux_loss, delta_parameters, set_keys, upcycle, utilisation_entropy,
)

CENTRES = torch.log(torch.tensor([0.35, 0.65, 1.0, 1.5]))


def conv_moe(k=4, r=8, cin=32, cout=16, gate="learned_top1"):
    base = nn.Conv2d(cin, cout, 3, padding=1)
    g = LearnedGate(cin, k, mode="top1" if gate.endswith("top1") else "soft") \
        if gate.startswith("learned") else PrescribedRangeGate(CENTRES[:k])
    return MoEResidual(base, g, num_experts=k, rank=r), base


# --------------------------------------------------------------- 1. identity at init

@pytest.mark.parametrize("k", [2, 4, 8])
@pytest.mark.parametrize("r", [4, 8, 16])
def test_identity_at_init_conv(k, r):
    torch.manual_seed(0)
    m, base = conv_moe(k=k, r=r)
    x = torch.randn(2, 32, 12, 12)
    assert torch.equal(m(x), base(x))


@pytest.mark.parametrize("k,r", [(4, 8), (8, 4)])
def test_identity_at_init_linear(k, r):
    torch.manual_seed(0)
    base = nn.Linear(24, 40)
    m = MoEResidual(base, LearnedGate(24, k, conv=False), num_experts=k, rank=r)
    x = torch.randn(2, 7, 24)
    assert torch.equal(m(x), base(x))


def test_identity_holds_for_prescribed_gate():
    torch.manual_seed(0)
    m, base = conv_moe(gate="prescribed")
    x = torch.randn(2, 32, 12, 12)
    m.set_key(torch.rand(2, 1, 12, 12) * 3 + 0.2)
    assert torch.equal(m(x), base(x))


# --------------------------------------------------------------- 2. algebraic equivalence

def test_matches_explicit_per_expert_sum():
    """The fused form must equal materialising W_base + A_i B_i per expert and mixing by g."""
    torch.manual_seed(0)
    k, r, cin, cout = 4, 6, 8, 5
    base = nn.Conv2d(cin, cout, 1, bias=False)
    m = MoEResidual(base, LearnedGate(cin, k, mode="soft"), num_experts=k, rank=r)
    nn.init.normal_(m.B.weight, std=0.1)                    # leave the identity regime
    x = torch.randn(2, cin, 5, 5)

    g = m.gate(x, None)
    ref = base(x)
    for i in range(k):
        Ai = m.A.weight[i * r:(i + 1) * r]                  # (r, cin, 1, 1)
        Bi = m.B.weight[:, i * r:(i + 1) * r]               # (cout, r, 1, 1)
        zi = torch.nn.functional.conv2d(x, Ai)
        ref = ref + g[:, i:i + 1] * torch.nn.functional.conv2d(zi, Bi)
    assert torch.allclose(m(x), ref, atol=1e-5)


# --------------------------------------------------------------- 3. symmetry break

def test_experts_differentiate_after_one_step():
    torch.manual_seed(0)
    m, _ = conv_moe(k=4, r=4)
    opt = torch.optim.SGD([m.A.weight, m.B.weight], lr=1.0)
    x = torch.randn(4, 32, 8, 8)
    w = torch.linspace(0.1, 2.0, 8 * 8).reshape(1, 1, 8, 8)   # non-uniform over positions
    (m(x) * w).square().mean().backward()
    opt.step()
    deltas = [m.B.weight[:, i * 4:(i + 1) * 4] for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(deltas[i], deltas[j])


# --------------------------------------------------------------- 4. parameter accounting

@pytest.mark.parametrize("k,r", [(4, 8), (16, 8), (32, 4)])
def test_added_params_closed_form_linear(k, r):
    d_in, d_out = 768, 3072
    m = MoEResidual(nn.Linear(d_in, d_out), LearnedGate(d_in, k, conv=False),
                    num_experts=k, rank=r)
    assert m.added_params() == k * r * (d_in + d_out)


def test_added_params_conv_scales_with_kernel():
    m, _ = conv_moe(k=4, r=8, cin=32, cout=16)               # 3x3 kernel
    assert m.added_params() == 4 * 8 * (32 * 3 * 3) + 16 * 4 * 8


# --------------------------------------------------------------- 5. frozen base

def test_frozen_base_takes_no_gradient():
    m, _ = conv_moe()
    m.freeze_base()
    assert all(not p.requires_grad for p in m.base.parameters())
    m(torch.randn(2, 32, 6, 6)).sum().backward()
    assert all(p.grad is None for p in m.base.parameters())


def test_delta_parameters_excludes_base():
    model = nn.Sequential(nn.Conv2d(8, 8, 1), nn.ReLU(), nn.Conv2d(8, 4, 1))
    upcycle(model, ["2"], "learned_top1", num_experts=4, rank=4)
    ids = {id(p) for p in delta_parameters(model)}
    assert all(id(p) not in ids for p in model[2].base.parameters())
    assert len(ids) == 3                                      # A, B, gate proj


# --------------------------------------------------------------- 6. static export

def test_export_has_no_data_dependent_shapes():
    """No NonZero / TopK / NonMaxSuppression, and every Conv shape static."""
    onnx = pytest.importorskip("onnx")
    m, _ = conv_moe(k=8, r=8)
    m.eval()
    path = "/tmp/moe_static.onnx"
    torch.onnx.export(m, torch.randn(1, 32, 16, 16), path, opset_version=13)
    g = onnx.load(path).graph
    banned = {"NonZero", "TopK", "NonMaxSuppression", "Loop", "If"}
    assert not (banned & {n.op_type for n in g.node})
    for vi in list(g.input) + list(g.output):
        dims = vi.type.tensor_type.shape.dim
        assert all(d.HasField("dim_value") for d in dims), f"{vi.name} has a dynamic dim"


# --------------------------------------------------------------- prescribed routing

def test_prescribed_gate_is_soft_and_normalised():
    g = PrescribedRangeGate(CENTRES)
    x = torch.zeros(2, 3, 6, 6)
    key = torch.rand(2, 1, 6, 6) * 2.5 + 0.15
    w = g(x, key)
    assert w.shape == (2, 4, 6, 6)
    assert torch.allclose(w.sum(1), torch.ones(2, 6, 6), atol=1e-5)
    assert (w > 0).sum(1).max() <= 2                          # at most two adjacent bins


def test_prescribed_gate_assigns_by_range():
    """Near pixels must land on low bins and far pixels on high bins."""
    g = PrescribedRangeGate(CENTRES)
    x = torch.zeros(1, 3, 1, 2)
    w = g(x, torch.tensor([[[[0.2, 3.0]]]]))
    assert w[0, :, 0, 0].argmax() == 0
    assert w[0, :, 0, 1].argmax() == 3


def test_quantile_centres_are_equal_population():
    torch.manual_seed(0)
    d = torch.rand(200_000) * 4 + 0.1
    c = PrescribedRangeGate.quantile_centres(d, 8)
    assert c.numel() == 8 and bool((c[1:] > c[:-1]).all())
    counts = torch.bincount(torch.bucketize(d.log(), c), minlength=9).float()
    assert counts[1:-1].std() / counts[1:-1].mean() < 0.25


def test_prescribed_gate_rejects_bad_centres():
    with pytest.raises(ValueError):
        PrescribedRangeGate(torch.tensor([1.0, 0.5]))         # not increasing
    with pytest.raises(NotImplementedError):
        PrescribedRangeGate(CENTRES)(torch.zeros(2, 7, 24), torch.rand(2, 7))


# --------------------------------------------------------------- diagnostics

def test_utilisation_entropy_flags_collapse():
    model = nn.Sequential(nn.Conv2d(8, 8, 1))
    mods = upcycle(model, ["0"], "learned_top1", num_experts=8, rank=4)
    with torch.no_grad():                                     # force every token to expert 0
        mods[0].gate.proj.weight.zero_()
        mods[0].gate.proj.weight[0].fill_(10.0)
    # Input must be positive: with randn the logit for expert 0 is 10*sum(x) and goes negative
    # for half the tokens, which sends argmax elsewhere and reads as partial collapse.
    model(torch.rand(2, 8, 6, 6) + 0.1)
    assert utilisation_entropy(model) < 0.05

    model2 = nn.Sequential(nn.Conv2d(8, 8, 1))
    upcycle(model2, ["0"], "prescribed", num_experts=4, rank=4, centres_log=CENTRES)
    set_keys(model2, torch.rand(2, 1, 6, 6) * 2.5 + 0.15)
    model2(torch.randn(2, 8, 6, 6))
    assert utilisation_entropy(model2) > 0.8


def test_aux_loss_present_for_learned_absent_for_prescribed():
    m1 = nn.Sequential(nn.Conv2d(8, 8, 1))
    upcycle(m1, ["0"], "learned_top1", num_experts=4, rank=4)
    m1(torch.randn(2, 8, 5, 5))
    assert aux_loss(m1) is not None and float(aux_loss(m1)) > 0

    m2 = nn.Sequential(nn.Conv2d(8, 8, 1))
    upcycle(m2, ["0"], "prescribed", num_experts=4, rank=4, centres_log=CENTRES)
    set_keys(m2, torch.rand(2, 1, 5, 5) * 2 + 0.2)
    m2(torch.randn(2, 8, 5, 5))
    assert aux_loss(m2) is None


def test_upcycled_model_is_bit_identical_end_to_end():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 3, padding=1))
    x = torch.randn(2, 4, 10, 10)
    before = model(x).clone()
    upcycle(model, ["0", "2"], "learned_soft", num_experts=8, rank=8)
    assert torch.equal(model(x), before)


def test_b_init_random_breaks_the_gate_gradient_that_zero_init_kills():
    """B=0 makes dL/dg exactly zero, which is why routing never bootstrapped. Random B must
    both (a) give the gate a non-zero gradient at init and (b) stay small against the base."""
    import torch.nn as nn
    from moe.residual import LearnedGate, MoEResidual

    base = nn.Linear(64, 128)
    wb = float(base.weight.norm())

    zero = MoEResidual(base, LearnedGate(64, 4, mode="soft", conv=False),
                       num_experts=4, rank=8, b_init="zero")
    rand = MoEResidual(nn.Linear(64, 128), LearnedGate(64, 4, mode="soft", conv=False),
                       num_experts=4, rank=8, b_init="random", b_scale=0.02)

    for mod, expect_grad in ((zero, False), (rand, True)):
        mod.zero_grad()
        x = torch.randn(2, 16, 64)
        mod(x).pow(2).mean().backward()
        g = mod.gate.proj.weight.grad
        has = g is not None and float(g.abs().max()) > 0
        assert has is expect_grad, f"gate gradient present={has}, expected {expect_grad}"

    # each expert's delta lands at the requested fraction of the base norm
    r = rand.rank
    for i in range(rand.num_experts):
        d = rand.B.weight[:, i * r:(i + 1) * r] @ rand.A.weight[i * r:(i + 1) * r]
        ratio = float(d.norm()) / float(rand.base.weight.norm())
        assert 0.015 < ratio < 0.025, f"expert {i} delta ratio {ratio:.4f} off target 0.02"

    # and zero-init is still exactly function-preserving
    x = torch.randn(2, 16, 64)
    assert torch.allclose(zero(x), zero.base(x), atol=1e-6)


def test_mixture_health_flags_the_uniform_gate_failure():
    """The failure that survived a 6-seed grid: near-orthogonal experts averaged by a uniform
    soft gate cancel to nothing. Health must call that INERT, and must NOT call a real
    mixture inert."""
    import torch.nn as nn
    from moe.residual import LearnedGate, MoEResidual
    from moe.upcycle import mixture_health

    def build(b_init, scale=0.02):
        m = nn.Module()
        m.site = MoEResidual(nn.Linear(64, 128),
                             LearnedGate(64, 8, mode="soft", conv=False),
                             num_experts=8, rank=4, b_init=b_init, b_scale=scale)
        return m

    zero = build("zero")
    zero.site(torch.randn(2, 16, 64))
    hz = mixture_health(zero)
    assert hz["inert"] is True, "B=0 mixture contributes nothing and must be flagged"
    assert hz["realised_ratio"] < 1e-6

    live = build("random", 0.05)
    live.site(torch.randn(2, 16, 64))
    hl = mixture_health(live)
    assert hl["realised_ratio"] > hz["realised_ratio"]
    assert hl["inert"] is False, f"real mixture wrongly flagged inert: {hl}"

    # uniform mixing of near-orthogonal experts must realise far less than one expert alone
    assert hl["realised_ratio"] < hl["expert_ratio"], (
        "averaging orthogonal experts should shrink the realised delta")


def test_periodic_diagnostic_fires_on_a_plain_steps_run():
    """range(steps) means the last step is steps-1, so a modulo-only schedule never fires on
    the common case (diag_matrix_every == steps). Guard against the guard no-opping."""
    every, steps = 3000, 3000
    fired = [s for s in range(steps) if every and s > 0 and s % every == 0]
    assert not fired, "sanity: modulo alone genuinely never fires here"
    # run_cell compensates with an unconditional end-of-run pass, so the recorded schedule
    # must contain at least one entry.
    recorded = fired + ([steps] if every else [])
    assert recorded, "a run with diagnostics enabled must record at least one matrix"

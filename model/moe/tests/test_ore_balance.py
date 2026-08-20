"""Load-balancing regression tests.

Every ORE run so far reported "K=128" while routing to ~3 experts, and nothing failed. These
tests exist so that cannot happen silently again: they assert the bias steers SELECTION only,
that it actually moves load toward uniform, and that the diagnostics see a collapse.
"""
import torch
import torch.nn as nn

from model.moe.ore import OREResidual, ore_z_loss


def make(k=8, balance="aux", **kw):
    torch.manual_seed(0)
    return OREResidual(nn.Linear(64, 64), num_experts=k, rank=4, n_layers=2, balance=balance, **kw)


def test_bias_does_not_leak_into_the_gate_value():
    """p_sel must come from the UNBIASED softmax, or balancing pressure enters the task loss."""
    m = make(balance="bias")
    x = torch.randn(200, 64)
    with torch.no_grad():
        probs = m.gate(x).softmax(-1)
        m.sel_bias.copy_(torch.randn(m.k) * 5.0)     # a bias big enough to change every choice
        idx, p_sel = m._route(x)
    assert torch.allclose(p_sel, probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1))
    assert (idx == probs.argmax(-1)).float().mean() < 0.9, "bias never changed the selection"


def test_aux_mode_ignores_the_bias():
    m = make(balance="aux")
    x = torch.randn(128, 64)
    with torch.no_grad():
        base = m._route(x)[0]
        m.sel_bias.copy_(torch.randn(m.k) * 5.0)
        assert torch.equal(m._route(x)[0], base)


def test_bias_step_is_scaled_by_the_logit_spread():
    """A fixed absolute step is a no-op when logits are large -- balancing that silently does
    nothing is the exact failure this whole plan exists to fix."""
    m = make(balance="bias")
    x = torch.randn(256, 64)
    def step(mult):                                  # let the scale EMA settle before comparing
        mm = make(balance="bias")
        with torch.no_grad():
            mm.gate.weight.mul_(mult)
            for _ in range(60):
                mm._route(x)
            mm.update_bias()
        return float(mm.sel_bias.abs().max())
    assert step(100.0) > 20 * step(1.0)


def test_bias_update_pushes_load_toward_uniform():
    """The whole point. A router with a hard-collapsed init must spread out under the update."""
    m = make(k=8, balance="bias")
    mu = torch.randn(64)
    x = torch.randn(512, 64) + mu * 0.5           # diverse tokens, so experts CAN split them
    with torch.no_grad():                            # one expert aligned with the data mean wins
        m.gate.weight.normal_(0, 0.05)               # everything; the rest stay distinguishable
        m.gate.weight[0] = mu
    first = None
    for _ in range(1200):
        with torch.no_grad():
            m._route(x)
        if first is None:
            first = m.load_stats()["effective_experts"]
        m.update_bias()
    with torch.no_grad():
        m._route(x)
    assert first < 1.5, f"expected a collapsed start, got {first}"
    assert m.load_stats()["effective_experts"] > 6.0, m.load_stats()


def test_tok_count_accumulates_then_resets():
    """Grad accumulation only helps if counts survive the micro-steps."""
    m = make(balance="bias")
    x = torch.randn(100, 64)
    with torch.no_grad():
        m._route(x); m._route(x); m._route(x)
    assert float(m.tok_count.sum()) == 300
    m.update_bias()
    assert float(m.tok_count.sum()) == 0


def test_load_stats_detects_collapse_and_uniformity():
    m = make(k=16)
    m.utilisation = torch.zeros(16); m.utilisation[3] = 1.0
    s = m.load_stats()
    assert abs(s["effective_experts"] - 1.0) < 1e-3 and s["dead"] == 15 and s["cv"] > 3.0
    m.utilisation = torch.full((16,), 1 / 16)
    s = m.load_stats()
    assert abs(s["effective_experts"] - 16.0) < 1e-3 and s["cv"] < 1e-5


def test_z_loss_present_only_when_requested():
    x = torch.randn(64, 64)
    for mode, want in (("aux", False), ("bias", False), ("bias+z", True), ("z", True)):
        m = make(balance=mode)
        with torch.no_grad():
            m._route(x)
        assert (m.z_loss is not None) == want, mode
    m = make(balance="bias+z")
    with torch.no_grad():
        m._route(x)
    assert ore_z_loss(nn.Sequential(m)) is not None


def test_z_loss_penalises_large_logits():
    m = make(balance="z")
    x = torch.randn(64, 64)
    with torch.no_grad():
        m._route(x)
        small = float(m.z_loss)
        m.gate.weight.mul_(50.0)
        m._route(x)
    assert float(m.z_loss) > small * 10


def test_hash_routing_is_balanced_by_construction():
    """A fixed partition cannot collapse. This is the control the learned arms all failed."""
    m = make(k=32, route_mode="hash")
    out = m(torch.randn(2, 731, 64))
    assert out.shape == (2, 731, 64)
    s = m.load_stats()
    assert s["effective_experts"] > 31.5 and s["dead"] == 0 and s["cv"] < 0.05, s
    assert m.aux_loss is None, "a fixed partition needs no balancing loss"


def test_key_routing_bins_by_depth():
    centres = torch.linspace(-1.6, 1.6, 8).tolist()          # log-metres
    m = make(k=8, route_mode="key", centres_log=centres)
    key = torch.full((2, 1, 27, 27), 0.25)                   # log(0.25) = -1.39 -> bin 0 or 1
    m.set_key(key)
    idx = m._fixed_idx((2, 731, 64), key.device)      # AAT prepends 2 special tokens
    assert idx.unique().numel() == 1 and idx[0].item() <= 1
    m.set_key(torch.full((2, 1, 27, 27), 4.0))               # log(4) = 1.39 -> top bin
    assert m._fixed_idx((2, 731, 64), key.device)[-1].item() >= 6


def test_fixed_routing_trains_no_gate():
    m = make(k=8, route_mode="hash")
    from model.moe.ore import ore_parameters
    assert not any(p is m.gate.weight for p in ore_parameters(nn.Sequential(m)))


def test_fused_butterfly_matches_eager():
    """The fused kernel is the deployed rotation path, so it must equal the reference it
    replaces. GPU-only; the kernel has no CPU fallback."""
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from model.moe.ore import ButterflyStack
    from model.moe.triton_moe import fused_butterfly
    for dim, n_l in ((512, 1), (512, 2), (2048, 2)):   # butterfly needs pow-2 dims
        torch.manual_seed(0)
        st = ButterflyStack(16, dim, n_l, "cuda", torch.float16)
        torch.nn.init.normal_(st.angle, std=0.5)
        x = torch.randn(300, dim, device="cuda", dtype=torch.float16)
        idx = torch.randint(0, 16, (300,), device="cuda")
        with torch.enable_grad():          # grad mode forces the eager reference branch
            ref = st(x, idx).detach()
        with torch.no_grad():              # no-grad selects the fused kernel inside the module
            got = st(x, idx)
        rel = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
        assert rel < 3e-3, f"dim={dim} n_l={n_l} rel={rel}"


def test_fused_butterfly_handles_noncontiguous_input():
    """OREResidual rotates a SLICE of the activation, so the kernel must never assume contiguity.
    A stride mismatch between X and Y wrote out of bounds and killed a 25k-step run at startup."""
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from model.moe.ore import ButterflyStack
    st = ButterflyStack(8, 512, 2, "cuda", torch.float16)
    torch.nn.init.normal_(st.angle, std=0.5)
    wide = torch.randn(200, 768, device="cuda", dtype=torch.float16)
    sl = wide[:, :512]                                  # non-contiguous view, stride 768
    assert sl.stride(0) == 768
    idx = torch.randint(0, 8, (200,), device="cuda")
    with torch.enable_grad():
        ref = st(sl, idx).detach()
    with torch.no_grad():
        got = st(sl, idx)
    rel = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert rel < 3e-3, rel

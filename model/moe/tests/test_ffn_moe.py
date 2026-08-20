#!/usr/bin/env python3
"""CPU unit tests for FFNMoE. Cheap enough for a login node; the GPU gates need the real model.

These check the SEMANTICS the arm's claim rests on: function-preserving init at any top-k,
council masking that actually restricts a frame, active-vs-total accounting, and a dispatch path
that reaches every expert with gradient. `gate_ffn_moe.py` re-checks all of it on the real
student against real data, plus throughput.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from moe.ffn_moe import FFNMoE, ffn_parameters, upcycle_ffn  # noqa: E402


class Mlp(nn.Module):
    """Stand-in with the same fc1/act/fc2 surface as the AAT block's MLP."""

    def __init__(self, d=32, h=64):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d=32, h=64):
        super().__init__()
        self.mlp = Mlp(d, h)

    def forward(self, x):
        return self.mlp(x)


def _x(b=3, t=7, d=32):
    torch.manual_seed(0)
    return torch.randn(b, t, d)


def test_function_preserving_init():
    """Identical copies + renormalised gate values == the dense parent, at any top-k."""
    x = _x()
    for k, tk, cm in [(8, 1, 0), (8, 2, 0), (8, 4, 0), (8, 2, 3), (1, 1, 0), (2, 2, 2)]:
        blk = Block()
        ref = blk.mlp(x)
        m = FFNMoE(blk.mlp, num_experts=k, top_k=tk, council_m=cm)
        d = (m(x) - ref).abs().max()
        assert d < 1e-5, f"K={k} top_k={tk} council={cm}: max diff {d:.2e}"


def test_council_restricts_the_frame():
    """A council of M leaves at most M experts used per frame, and M >= top_k."""
    x = _x(b=4, t=64)
    blk = Block()
    m = FFNMoE(blk.mlp, num_experts=8, top_k=2, council_m=3, router_norm=False)
    nn.init.normal_(m.gate.weight, std=1.0)          # a real partition, not the near-uniform init
    flat = x.reshape(-1, m.d_in)
    idx, w = m._route_topk(flat, x.shape)
    per_frame = idx.reshape(x.shape[0], -1, m.top_k)
    for f in range(x.shape[0]):
        used = per_frame[f].unique()
        assert used.numel() <= 3, f"frame {f} used {used.numel()} experts under a 3-council"
        assert torch.isin(used, m.council[f].nonzero().squeeze(-1)).all()
    assert torch.allclose(w.sum(-1), torch.ones(w.shape[0]), atol=1e-5)


def test_free_dispatch_can_exceed_council_width():
    """Without a council the same router is free to use more than 3 experts per frame."""
    x = _x(b=4, t=64)
    blk = Block()
    m = FFNMoE(blk.mlp, num_experts=8, top_k=2, council_m=0)
    nn.init.normal_(m.gate.weight, std=1.0)
    idx, _ = m._route_topk(x.reshape(-1, m.d_in), x.shape)
    assert idx.reshape(x.shape[0], -1, 2)[0].unique().numel() > 3


def test_accounting():
    blk = Block(d=32, h=64)
    per_expert = sum(p.numel() for p in blk.mlp.parameters())
    m = FFNMoE(blk.mlp, num_experts=8, top_k=2)
    g = m.gate_params()
    assert m.expert_params() == per_expert
    assert m.total_params() == 8 * per_expert + g
    assert m.active_params() == 2 * per_expert + g
    # the claim of the arm: active trainable == the dense fine-tune at top-1
    assert FFNMoE(blk.mlp, num_experts=8, top_k=1).active_params() - g == per_expert


def test_every_expert_gets_gradient_and_they_diverge():
    x = _x(b=8, t=32)
    blk = Block()
    m = FFNMoE(blk.mlp, num_experts=4, top_k=1)
    nn.init.normal_(m.gate.weight, std=1.0)
    opt = torch.optim.AdamW(m.parameters_to_train(), lr=1e-2)
    tgt = torch.randn_like(m(x))
    for _ in range(20):
        opt.zero_grad()
        loss = (m(x) - tgt).pow(2).mean() + 0.1 * m.aux_loss
        loss.backward()
        for i, e in enumerate(m.experts):
            assert e.fc1.weight.grad is not None and e.fc1.weight.grad.norm() > 0, \
                f"expert {i} got no gradient: the dispatch path is detached"
        opt.step()
    assert abs(m.expert_divergence()["expert_cos"]) < 0.999, "experts stayed tied"


def test_aux_is_one_at_uniform_and_larger_when_collapsed():
    """Switch statistic, so aux_coef stays K-independent as upcycle.aux_loss assumes."""
    x = _x(b=4, t=128)
    blk = Block()
    m = FFNMoE(blk.mlp, num_experts=8, top_k=1)
    with torch.no_grad():
        m.gate.weight.zero_()
        m.gate.bias.zero_()
    m._route_topk(x.reshape(-1, m.d_in), x.shape)
    assert abs(float(m.aux_loss) - 1.0) < 0.2
    with torch.no_grad():
        m.gate.bias[0] = 50.0
    m._route_topk(x.reshape(-1, m.d_in), x.shape)
    assert float(m.aux_loss) > 7.0


def test_topk_counts_every_slot():
    x = _x(b=2, t=64)
    blk = Block()
    m = FFNMoE(blk.mlp, num_experts=4, top_k=2)
    nn.init.normal_(m.gate.weight, std=1.0)
    m.tok_count.zero_()
    m._route_topk(x.reshape(-1, m.d_in), x.shape)
    assert abs(float(m.tok_count.sum()) - x.shape[0] * x.shape[1] * 2) < 1e-3


def test_upcycle_wraps_modules_not_linears():
    model = nn.ModuleDict({"b": Block()})
    mods = upcycle_ffn(model, ["b.mlp"], num_experts=4, top_k=1)
    assert isinstance(model["b"].mlp, FFNMoE) and len(mods) == 1
    assert len(ffn_parameters(model)) == 4 * 4 + 2         # 4 experts x 4 tensors, + gate w/b
    try:
        upcycle_ffn(nn.ModuleDict({"b": Block()}), ["b.mlp.fc1"], num_experts=2)
    except TypeError:
        return
    raise AssertionError("wrapping a Linear should raise")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:                                  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("ALL UNIT TESTS PASS" if not fails else f"{fails} UNIT TEST FAILURES")
    sys.exit(1 if fails else 0)

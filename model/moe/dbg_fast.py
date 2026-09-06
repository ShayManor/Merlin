"""Localise where the fused inference path departs from the eager one."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from moe.ore import OREResidual, enable_fast_inference                       # noqa: E402

D_IN, D_OUT, M = 768, 3072, 729
K, R = 32, 128


def site():
    torch.manual_seed(0)
    base = nn.Linear(D_IN, D_OUT, device="cuda", dtype=torch.float16)
    o = OREResidual(base, num_experts=K, rank=R, n_layers=2, block_m=16).cuda().eval()
    with torch.no_grad():
        o.B.normal_(0, 0.02)
        o.A.normal_(0, 0.02)
        o.theta.angle.normal_(0, 0.4)
        o.phi.angle.normal_(0, 0.4)
        o.gate.weight.normal_(0, 0.05)
    return o


o = site()
x = torch.randn(1, M, D_IN, device="cuda", dtype=torch.float16)
flat = x.reshape(-1, D_IN)
with torch.no_grad():
    idx_ref, p_ref = o._route(flat, x.shape)
    ref = o(x)
    enable_fast_inference(o)
    got = o(x)
    buf = o._fbuf

d = (got.float() - ref.float()).abs().reshape(-1, D_OUT)
row = d.max(1).values
bad = (row > 1e-2).nonzero().flatten()
print(f"rows differing >1e-2: {bad.numel()} / {M}   max row err {row.max():.4f}")
print(f"idx mismatches: {(buf['idx'].long() != idx_ref).sum().item()}")
print(f"psel max diff : {(buf['psel'].float() - p_ref.float()).abs().max():.5f}")
if bad.numel():
    b = bad[:8]
    print("bad rows      :", b.tolist())
    print("  idx fast/ref:", buf["idx"][b].tolist(), idx_ref[b].tolist())
    print("  row err     :", [round(v, 4) for v in row[b].tolist()])
    print("  rows also idx-mismatched:",
          int((buf["idx"].long() != idx_ref)[bad].sum()))

# is the residual or the oriented term responsible?
with torch.no_grad():
    o.fast = False
    orient = o._oriented(flat, idx_ref)
    resid = o._residual_sparse(flat, idx_ref) * p_ref.unsqueeze(-1)
    o.fast = True
    o(x)
    st, be = buf["stok"], buf["be"]
    from moe.triton_moe import fused_lowrank, fused_rot                      # noqa: E402
    r_fast = fused_lowrank(flat, o.A, o.B, st, be, o.bm)
    xi = fused_rot(flat, o.theta.angle, o.rot_in_dim, st, be, o.bm)
    h = torch.nn.functional.linear(xi, o.base.weight)
    o_fast = fused_rot(h, o.phi.angle, o.rot_out_dim, st, be, o.bm, bias=o.base.bias)


def rep(name, a, b):
    dd = (a.float() - b.float()).abs()
    print(f"{name:>22}  max {dd.max():.4f}   rows>1e-2 {int((dd.max(1).values > 1e-2).sum())}")


rep("oriented", o_fast, orient)
rep("residual (unscaled)", r_fast, resid / p_ref.unsqueeze(-1).clamp_min(1e-6))

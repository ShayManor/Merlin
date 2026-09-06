#!/usr/bin/env python3
"""What does a REAL MoE cost on the Jetson? Dummy weights, real shapes.

Context: the range-routed experiment computed every expert densely and gated them per pixel,
which is not an MoE in the sense that matters -- it has no sparse activation, so it costs 2.5x
a dense update instead of matching it. This measures the variants directly on-device rather
than predicting them from a roofline.

Variants, all at the deployed site (Conv2d 128->128 k3 at 378x378), batch 1:

  dense          the conv as-is. The baseline every MoE has to beat.
  moe_allK       K full-rank experts, ALL computed, soft gate, weighted sum.
                 The honest cost of "no dispatch". Should be ~Kx dense.
  moe_top1       K full-rank experts, top-1 per pixel, gather -> per-expert conv -> scatter.
                 A REAL MoE: ~1x dense FLOPs, Kx the weights. This is the variant whose whole
                 premise is that sparse activation buys capacity for free.
  lowrank_ens    what the experiment actually ran: rank-r A/B pair plus a router head.

Reports measured latency against analytic MACs, so the gap between the two is visible: on a
compute-bound device top-1 should approach dense, on a bandwidth-bound one it cannot.
"""
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

H = W = 378
CIN = COUT = 128


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(CIN, COUT, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class MoEAllK(nn.Module):
    """Every expert computed for every pixel, then mixed. No dispatch, no saving."""

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.experts = nn.ModuleList([nn.Conv2d(CIN, COUT, 3, padding=1) for _ in range(k)])
        self.gate = nn.Conv2d(CIN, k, 1)

    def forward(self, x):
        g = self.gate(x).softmax(1)
        return sum(g[:, i:i + 1] * e(x) for i, e in enumerate(self.experts))


class MoETop1(nn.Module):
    """A real sparse MoE: each pixel goes to exactly one expert.

    Implemented as gather -> per-expert 1x1 GEMM on that expert's pixels -> scatter. The 3x3
    receptive field is materialised first with unfold, because gathering scattered pixels
    destroys spatial locality and a conv can no longer be applied directly. That unfold is
    itself part of the true cost of sparse dispatch and is deliberately not hidden.
    """

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(k, COUT, CIN * 9) * 0.02)
        self.b = nn.Parameter(torch.zeros(k, COUT))
        self.gate = nn.Conv2d(CIN, k, 1)

    def forward(self, x):
        B = x.shape[0]
        idx = self.gate(x).argmax(1).reshape(-1)                      # (B*H*W,)
        cols = F.unfold(x, 3, padding=1).transpose(1, 2).reshape(-1, CIN * 9)
        out = torch.empty(cols.shape[0], COUT, device=x.device, dtype=x.dtype)
        for i in range(self.k):
            sel = (idx == i).nonzero(as_tuple=True)[0]                # data-dependent shape
            if sel.numel():
                out[sel] = cols[sel] @ self.w[i].t() + self.b[i]
        return out.reshape(B, H, W, COUT).permute(0, 3, 1, 2).contiguous()


class LowRankEns(nn.Module):
    """What the experiment ran: frozen base + rank-r experts + router head."""

    def __init__(self, k, r, hidden=64):
        super().__init__()
        self.base = nn.Conv2d(CIN, COUT, 3, padding=1)
        self.A = nn.Conv2d(CIN, k * r, 3, padding=1, bias=False)
        self.B = nn.Conv2d(k * r, COUT, 1, bias=False)
        self.router = nn.Sequential(nn.Conv2d(CIN, hidden, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(hidden, k, 1))
        self.k, self.r = k, r

    def forward(self, x):
        g = self.router(x).softmax(1).repeat_interleave(self.r, 1)
        return self.base(x) + self.B(self.A(x) * g)


def macs(name, k, r, hidden=64):
    px = H * W
    c = lambda i, o, kk: px * o * i * kk * kk
    if name == "dense":
        return c(CIN, COUT, 3)
    if name == "moe_allK":
        return k * c(CIN, COUT, 3) + c(CIN, k, 1)
    if name == "moe_top1":
        return c(CIN, COUT, 3) + c(CIN, k, 1)          # each pixel touches ONE expert
    if name == "lowrank_ens":
        return (c(CIN, COUT, 3) + c(CIN, k * r, 3) + c(k * r, COUT, 1)
                + c(CIN, hidden, 3) + c(hidden, k, 1))
    raise ValueError(name)


@torch.no_grad()
def bench(m, x, iters=50, warmup=10):
    for _ in range(warmup):
        m(x)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        m(x)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rank", type=int, default=29)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="/home/evc/merlin/moe_edge.json")
    args = ap.parse_args()

    dev = "cuda"
    dt = torch.float16 if args.dtype == "fp16" else torch.float32
    x = torch.randn(1, CIN, H, W, device=dev, dtype=dt)

    builds = {"dense": Dense(), "moe_allK": MoEAllK(args.k),
              "moe_top1": MoETop1(args.k), "lowrank_ens": LowRankEns(args.k, args.rank)}
    res, base_ms = {}, None
    print(f"site Conv2d({CIN}->{COUT},k3) @ {H}x{W}, batch 1, K={args.k}, rank={args.rank}, "
          f"{args.dtype}\n")
    print(f"{'variant':<15}{'ms':>9}{'vs dense':>10}{'GMAC':>9}{'vs dense':>10}"
          f"{'params':>11}{'eff TOPS':>10}")
    for name, m in builds.items():
        m = m.to(dev).to(dt).eval()
        try:
            ms = bench(m, x, args.iters)
        except Exception as ex:
            print(f"{name:<15}  FAILED: {type(ex).__name__}: {str(ex)[:60]}")
            continue
        g = macs(name, args.k, args.rank) / 1e9
        p = sum(q.numel() for q in m.parameters())
        if name == "dense":
            base_ms, base_g = ms, g
        tops = 2 * g / (ms / 1e3) / 1e3
        print(f"{name:<15}{ms:>9.2f}{ms / base_ms:>10.2f}{g:>9.2f}"
              f"{g / base_g:>10.2f}{p:>11,}{tops:>10.2f}")
        res[name] = {"ms": ms, "gmac": g, "params": p, "eff_tops": tops}
        del m
        torch.cuda.empty_cache()

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    if "moe_top1" in res and "dense" in res:
        print("\nThe question a real MoE lives or dies on: moe_top1 does ~1x dense FLOPs with "
              f"{args.k}x the weights.")
        print(f"Measured, it takes {res['moe_top1']['ms'] / res['dense']['ms']:.2f}x dense "
              "latency. Anything far above 1x means dispatch overhead, not arithmetic, is the "
              "cost on this device.")


if __name__ == "__main__":
    main()

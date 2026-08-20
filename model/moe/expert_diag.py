#!/usr/bin/env python3
"""Did the K=32 experts differentiate, or is the mixture arithmetically one delta?

The matched-latency test came back null (Rbig vs Dwide, p=0.49). Two very different things
produce that null and they need separating:

  (i)  the experts specialised, and specialisation simply does not help here; or
  (ii) the experts never differentiated, so the "mixture" is one averaged correction wearing
       32 hats, and the null says nothing about conditional compute at all.

Three independent readings, cheapest first:

  WEIGHTS   expert i's delta is exactly B[:, i*r:(i+1)*r] @ A[i*r:(i+1)*r, :]. Pairwise cosine
            similarity across the 32 deltas needs no forward pass. Near 1.0 => (ii).
  GATE      per-expert utilisation, and the joint distribution of the argmax expert against the
            token's true depth bin. A gate that ignores range cannot produce range experts.
  MATRIX    force every token to one expert and measure per-bin error, K x bins. A strong
            diagonal-ish structure (expert i best on some bin) => (i). Flat rows => (ii).

This is the sibling of expert_matrix.py, which covers K=4 prescribed routing at the conv site.
Kept separate because that one trains its own model and assumes conv layout; this one loads a
trained checkpoint and works on the AAT token sites where the deployed mixture actually lives.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from run_cell import (  # noqa: E402
    BINS, load_depth, load_model, load_pairs, predict_depth,
)
from moe.upcycle import moe_modules, upcycle  # noqa: E402


class ForcedGate(nn.Module):
    """Sends every token to one expert, whatever the layout."""

    def __init__(self, num_experts, idx, eaxis):
        super().__init__()
        self.num_experts, self.idx, self.eaxis = num_experts, idx, eaxis
        self.aux_loss = self.utilisation = None

    def forward(self, x, key=None):
        ax = self.eaxis % x.dim()
        shape = list(x.shape)
        shape[ax] = self.num_experts
        g = torch.zeros(shape, device=x.device, dtype=x.dtype)
        g.index_fill_(ax, torch.tensor([self.idx], device=x.device), 1.0)
        return g


def expert_deltas(mod):
    """(K, out, in) stack of the effective per-expert weight deltas."""
    r, k = mod.rank, mod.num_experts
    A = mod.A.weight.detach().float()                 # (K*r, in)   [Linear]
    B = mod.B.weight.detach().float()                 # (out, K*r)
    return torch.stack([B[:, i * r:(i + 1) * r] @ A[i * r:(i + 1) * r, :] for i in range(k)])


def weight_report(mods, out):
    print("=" * 74)
    print("1. WEIGHTS: did the experts learn different corrections?")
    print("=" * 74)
    sims, norms = [], []
    for mod in mods:
        if not isinstance(mod.A, nn.Linear):
            continue
        W = expert_deltas(mod)                        # (K, out, in)
        f = W.reshape(W.shape[0], -1)
        n = f.norm(dim=1)
        fn = f / n.clamp_min(1e-12).unsqueeze(1)
        C = fn @ fn.t()
        k = C.shape[0]
        off = (C.sum() - C.diag().sum()) / (k * (k - 1))
        sims.append(float(off))
        norms.append(n.cpu().numpy())
    nz = np.concatenate([x for x in norms]) if norms else np.array([])
    print(f"  sites analysed: {len(sims)}")
    print(f"  mean pairwise cosine between expert deltas: {np.mean(sims):+.4f} "
          f"(min {np.min(sims):+.4f}, max {np.max(sims):+.4f})")
    print(f"  expert delta Frobenius norms: mean {nz.mean():.5f}  min {nz.min():.5f}  "
          f"max {nz.max():.5f}  (max/min {nz.max()/max(nz.min(),1e-12):.1f}x)")
    print("  reading: ~+1.0 => experts are the same correction (mixture == one delta);")
    print("           ~0.0 => genuinely different corrections.")
    out["cosine_mean"] = float(np.mean(sims))
    out["cosine_per_site"] = sims
    out["delta_norms"] = {"mean": float(nz.mean()), "min": float(nz.min()),
                          "max": float(nz.max())}


@torch.no_grad()
def gate_report(model, mods, pairs, size, dev, n, out):
    print("\n" + "=" * 74)
    print("2. GATE: is routing used at all, and does it track metric range?")
    print("=" * 74)
    probe = mods[0]
    cap = {}

    def hook(_m, _i, o):
        cap["g"] = o.detach().float()
    h = probe.gate.register_forward_hook(hook)

    util = torch.zeros(probe.num_experts, dtype=torch.float64)
    joint = np.zeros((probe.num_experts, len(BINS)), dtype=np.float64)
    frames = 0
    for r, d in pairs[:n]:
        z = predict_depth(model, [r], size, dev)[0]
        if "g" not in cap:
            continue
        g = cap["g"]                                   # (B, N, K) for Linear sites
        if g.dim() != 3:
            continue
        util += g.mean(dim=(0, 1)).double().cpu()
        frames += 1
        gt = torch.from_numpy(load_depth(d, *z.shape)).float()
        side = int(round(g.shape[1] ** 0.5))
        if side * side != g.shape[1]:
            continue                                   # not a square patch grid; skip mapping
        gp = torch.nn.functional.adaptive_avg_pool2d(gt.unsqueeze(0).unsqueeze(0),
                                                     (side, side)).reshape(-1)
        top = g[0].argmax(-1).cpu()
        for j, (lo, hi) in enumerate(BINS):
            m = (gp >= lo) & (gp < hi)
            if m.any():
                idx, cnt = torch.unique(top[m], return_counts=True)
                joint[idx.numpy(), j] += cnt.numpy()
    h.remove()
    if not frames:
        print("  could not capture gate outputs"); return

    util = (util / frames).numpy()
    live = int((util > 0.01 / probe.num_experts).sum())
    print(f"  frames {frames}, experts {probe.num_experts}")
    print(f"  utilisation: min {util.min():.4f}  max {util.max():.4f}  "
          f"uniform would be {1/probe.num_experts:.4f}")
    print(f"  experts receiving >1% of uniform share: {live}/{probe.num_experts}")
    ent = -(util * np.log(util + 1e-12)).sum() / np.log(probe.num_experts)
    print(f"  normalised routing entropy: {ent:.4f}   (1.0 = perfectly uniform, 0 = collapsed)")

    if joint.sum():
        P = joint / joint.sum()
        pe, pb = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
        mi = float((P * np.log((P + 1e-12) / (pe * pb + 1e-12))).sum())
        print(f"  mutual information I(expert ; range bin) = {mi:.4f} nats "
              f"(max possible {np.log(min(probe.num_experts, len(BINS))):.4f})")
        print("  reading: MI ~0 => routing is blind to range, so no range specialisation is "
              "possible.")
        out["mi_expert_range"] = mi
    out["utilisation"] = util.tolist()
    out["routing_entropy"] = float(ent)


@torch.no_grad()
def matrix_report(model, mods, pairs, size, dev, n, out):
    print("\n" + "=" * 74)
    print("3. FORCED-EXPERT MATRIX: per-bin relative error, expert x range bin")
    print("=" * 74)
    k = mods[0].num_experts
    real = [(m, m.gate) for m in mods]

    def score():
        acc = [[] for _ in BINS]
        for r, d in pairs[:n]:
            z = predict_depth(model, [r], size, dev)[0]
            gt = torch.from_numpy(load_depth(d, *z.shape)).float().to(dev)
            v = torch.isfinite(z) & torch.isfinite(gt) & (gt > 0.1) & (gt < 10) & (z > 0.05)
            if v.sum() < 200:
                continue
            pa = z * torch.median(gt[v] / z[v])
            for j, (lo, hi) in enumerate(BINS):
                b = v & (gt >= lo) & (gt < hi)
                if b.sum() > 100:
                    acc[j].append(float(((pa[b] - gt[b]).abs() / gt[b]).mean()))
        return [float(np.mean(a)) if a else float("nan") for a in acc]

    rows = {"real gate": score()}
    for i in range(k):
        for m, _ in real:
            m.gate = ForcedGate(m.num_experts, i, m.eaxis).to(dev)
        rows[f"expert {i}"] = score()
        print(f"  ...expert {i} done", flush=True)
    for m, g in real:
        m.gate = g                                     # restore

    hdr = "".join(f"{f'{lo}-{hi}':>10}" for lo, hi in BINS)
    print(f"\n  {'arm':<12}{hdr}{'mean':>10}")
    for name, v in rows.items():
        mu = float(np.nanmean(v))
        print(f"  {name:<12}" + "".join(f"{x:>10.4f}" for x in v) + f"{mu:>10.4f}")

    ex = np.array([rows[f"expert {i}"] for i in range(k)])
    best = np.nanargmin(ex, axis=0)
    print(f"\n  best expert per bin: {list(best)}")
    print(f"  distinct winners: {len(set(best.tolist()))} of {len(BINS)} bins")
    spread = np.nanmax(ex, axis=0) - np.nanmin(ex, axis=0)
    print(f"  per-bin spread across experts: "
          f"{['%.4f' % s for s in spread]}  (mean {np.nanmean(spread):.4f})")
    print("  reading: if the spread is ~0 the experts are interchangeable; if each bin has a")
    print("  different winner with real spread, they specialised by range.")
    out["matrix"] = {k2: v for k2, v in rows.items()}
    out["best_expert_per_bin"] = best.tolist()
    out["per_bin_spread"] = spread.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/data/ckpt/rbig_s2_trained.pt")
    ap.add_argument("--init", default="/data/ckpt/student_hab_nav.pt")
    ap.add_argument("--val-root", default="/data/nf2/val")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--gate-n", type=int, default=60)
    ap.add_argument("--matrix-n", type=int, default=40)
    ap.add_argument("--skip-matrix", action="store_true")
    ap.add_argument("--out", default="/data/results/rmoe/expert_diag.json")
    args = ap.parse_args()

    dev = "cuda"
    size = (args.res // 14) * 14
    val = load_pairs(args.val_root, 1)
    # Everything --save-ckpt writes is tensors plus primitives, so this needs no unpickling.
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    print(f"[diag] {args.ckpt}: arm={ck['arm']} K={ck['k']} r={ck['rank']} "
          f"seed={ck['seed']} sites={len(ck['sites'])}")

    model = load_model(args.init, dev, aat_depth=ck.get("aat_depth", 8))
    upcycle(model, ck["sites"], {"L_soft": "learned_soft", "L_top1": "learned_top1"}[ck["arm"]],
            num_experts=ck["k"], rank=ck["rank"], freeze=False)
    model.to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mods = moe_modules(model)
    print(f"[diag] {len(mods)} MoE sites live")

    out = {"ckpt": args.ckpt, "k": ck["k"], "rank": ck["rank"]}
    weight_report(mods, out)
    gate_report(model, mods, val, size, dev, args.gate_n, out)
    if not args.skip_matrix:
        matrix_report(model, mods, val, size, dev, args.matrix_n, out)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

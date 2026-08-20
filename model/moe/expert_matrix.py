#!/usr/bin/env python3
"""Did the experts actually differentiate? Force every expert onto every bin and compare.

Trains a K=4 prescribed-routing model, then evaluates it 5 times: once with the real gate, and
once per expert with the gate overridden to send EVERY pixel to that one expert. The result is
a 4x4 matrix of relative error, expert i against range bin j.

Reading it:
  - Strong diagonal (expert i best on bin i) => the experts specialised by range.
  - Flat rows, all experts alike on every bin => they converged to the same correction, and the
    mixture is arithmetically one delta. That is the explanation for P tying the dense
    fine-tune control, and it would mean the loss (range-blind) never asked them to differ.

Read-only with respect to run_cell.py: the screen driver spawns a fresh interpreter per cell,
so editing that file mid-flight would change the running experiment.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from run_cell import (  # noqa: E402
    SITE, batch_key, bin_edges_from_centres, centres_from_edges, fit_bin_shares, fit_centres,
    fit_key_calibration, load_depth, load_model, load_pairs, pixel_weights, precompute_keys,
    predict_depth, si_loss,
)
from moe.upcycle import delta_parameters, moe_modules, upcycle  # noqa: E402


class ForcedGate(nn.Module):
    """Sends every token to one expert, regardless of range.

    Layout-aware: MoEResidual puts the expert axis at 1 for Conv2d (B,K,H,W) and at -1 for
    Linear (B,N,K). Hardcoding the conv layout silently builds a wrong-shaped gate at a Linear
    site instead of failing, so the axis follows the wrapped module.
    """

    def __init__(self, num_experts, idx, conv=True):
        super().__init__()
        self.num_experts, self.idx = num_experts, idx
        self.eaxis = 1 if conv else -1
        self.aux_loss = self.utilisation = None

    def forward(self, x, key=None):
        ax = self.eaxis % x.dim()
        shape = list(x.shape)
        shape[ax] = self.num_experts
        g = torch.zeros(shape, device=x.device, dtype=x.dtype)
        sel = [slice(None)] * x.dim()
        sel[ax] = self.idx
        g[tuple(sel)] = 1.0
        return g


@torch.no_grad()
def per_bin_rel(model, pairs, arm, cache, size, dev, edges, n, cal=None):
    acc = [[] for _ in range(len(edges) - 1)]
    for r, d in pairs[:n]:
        z = predict_depth(model, [r], size, dev,
                          key=batch_key(arm, cache, [r], [d], (size, size), dev, cal))[0]
        gt = torch.from_numpy(load_depth(d, *z.shape)).float().to(dev)
        v = torch.isfinite(z) & torch.isfinite(gt) & (gt > 0.1) & (gt < 10) & (z > 0.05)
        if v.sum() < 200:
            continue
        pa = z * torch.median(gt[v] / z[v])
        for j in range(len(edges) - 1):
            b = v & (gt >= edges[j]) & (gt < edges[j + 1])
            if b.sum() > 100:
                acc[j].append(float(((pa[b] - gt[b]).abs() / gt[b]).mean()))
    return [float(np.mean(a)) if a else float("nan") for a in acc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-root", default="/data/nf2/train")
    ap.add_argument("--val-root", default="/data/nf2/val")
    ap.add_argument("--init", default="/data/ckpt/student_hab_nav.pt")
    ap.add_argument("--keys", default="/data/keys_habnav")
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--loss", choices=["uniform", "binbal"], default="uniform")
    ap.add_argument("--calibrate-key", action="store_true")
    ap.add_argument("--bin-edges", default=None)
    ap.add_argument("--sites", default=None,
                    help="comma-separated module paths; default is the single conv SITE")
    ap.add_argument("--gate", default="prescribed",
                    choices=["prescribed", "learned_top1", "learned_soft"],
                    help="routing trained before the forced-gate sweep. Use learned_top1 to ask "
                         "whether a LEARNED router's experts specialised by range.")
    ap.add_argument("--out", default="/data/results/rmoe/expert_matrix.json")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    size = (args.res // 14) * 14
    train, val = load_pairs(args.train_root, 1), load_pairs(args.val_root, 1)

    model = load_model(args.init, dev)
    precompute_keys(model, train + val, args.keys, size, dev)
    cal = fit_key_calibration(args.keys, train, (size, size), dev) if args.calibrate_key else None
    if args.bin_edges:
        centres = centres_from_edges([float(x) for x in args.bin_edges.split(",")])
    else:
        centres = fit_centres("P", args.keys, train, (size, size), dev, args.k)
    edges = bin_edges_from_centres(centres)
    print(f"[mx] K={args.k} bin edges (m): "
          f"{[round(float(e), 3) for e in edges[1:-1]]}", flush=True)

    sites = [s.strip() for s in args.sites.split(",")] if args.sites else [SITE]
    mods = upcycle(model, sites, args.gate, num_experts=args.k, rank=args.rank,
                   centres_log=centres, freeze=True)
    print(f"[mx] {len(mods)} site(s), gate={args.gate}, "
          f"added={sum(m.added_params() for m in mods)}", flush=True)
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    params = delta_parameters(model)
    for p in params:
        p.requires_grad_(True)

    shares = fit_bin_shares(train, edges, (size, size)) if args.loss == "binbal" else None
    if shares is not None:
        print(f"[mx] bin-balanced loss, shares={[round(float(x),3) for x in shares]}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        idx = rng.integers(0, len(train), args.batch)
        paths = [train[i][0] for i in idx]
        deps = [train[i][1] for i in idx]
        key = batch_key("P", args.keys, paths, deps, (size, size), dev, cal)
        pz = predict_depth(model, paths, size, dev, key=key, batch=args.batch)
        gts = torch.stack([torch.from_numpy(load_depth(d, *pz.shape[1:])).float()
                           for d in deps], 0).to(dev)
        pw = pixel_weights(gts, edges, shares, dev) if shares is not None else None
        loss = si_loss(pz, gts, pw)
        if loss is None:
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"[mx] step {step} loss {loss.item():.4f} "
                  f"({(time.time() - t0) / max(step, 1):.2f}s/step)", flush=True)

    all_mods = moe_modules(model)
    mod = all_mods[0]
    real_gates = [m.gate for m in all_mods]
    model.eval()

    res = {"k": args.k, "rank": args.rank, "steps": args.steps, "loss": args.loss,
           "calibrated_key": bool(cal), "explicit_edges": args.bin_edges,
           "bin_edges_m": [float(e) for e in edges[1:-1]],
           "centres_m": [float(np.exp(c)) for c in centres]}

    res["routed"] = per_bin_rel(model, val, "P", args.keys, size, dev, edges, args.eval_n)
    print(f"[mx] routed (real gate): {[round(v, 4) for v in res['routed']]}", flush=True)

    matrix = []
    for i in range(args.k):
        # every wrapped site must be forced together; forcing only the first leaves the rest
        # routing normally and the row measures a mixture, not expert i
        for m in all_mods:
            m.gate = ForcedGate(args.k, i, conv=m.is_conv).to(dev)
        row = per_bin_rel(model, val, "P", args.keys, size, dev, edges, args.eval_n, cal)
        matrix.append(row)
        print(f"[mx] forced expert {i}: {[round(v, 4) for v in row]}", flush=True)
    for m, g in zip(all_mods, real_gates):
        m.gate = g
    res["matrix"] = matrix

    # How different are the experts, really? If the spread across experts within a bin is tiny,
    # they learned the same correction and the mixture collapses to a single delta.
    m = np.array(matrix)
    res["spread_within_bin"] = [float(np.nanmax(m[:, j]) - np.nanmin(m[:, j]))
                                for j in range(args.k)]
    res["diagonal_is_best"] = [bool(np.nanargmin(m[:, j]) == j) for j in range(args.k)]
    B = mod.B.weight.detach()
    r = args.rank
    blocks = [B[:, i * r:(i + 1) * r].flatten() for i in range(args.k)]
    cos = [[float(torch.nn.functional.cosine_similarity(a, b, dim=0)) for b in blocks]
           for a in blocks]
    res["delta_cosine"] = cos
    off = [cos[i][j] for i in range(args.k) for j in range(args.k) if i != j]
    res["delta_cosine_offdiag_mean"] = float(np.mean(off))

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\n[mx] spread within bin: {[round(s, 4) for s in res['spread_within_bin']]}")
    print(f"[mx] diagonal best per bin: {res['diagonal_is_best']}")
    print(f"[mx] mean off-diagonal cosine between expert deltas: "
          f"{res['delta_cosine_offdiag_mean']:.4f}  (1.0 = identical experts)")
    print(f"[mx] wrote {args.out}")


if __name__ == "__main__":
    main()

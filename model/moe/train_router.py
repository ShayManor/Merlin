#!/usr/bin/env python3
"""Train a GT-supervised bin classifier to replace the depth-keyed router.

The prescribed gate has no parameters: it thresholds the model's own predicted depth, which
carries ~47% relative error in the near field. Measured routing accuracy against true bins is
48.5%, near chance, so experts never receive range-partitioned data. This head is asked a much
easier question -- "which of 4 range bins is this pixel in" -- supervised by GT bins at
training time only. Inference uses image features alone, so unlike the ground-truth-keyed
oracle arm it leaks nothing.

Reports the keyed-router baseline with per-IMAGE statistics. A binomial test over ~28M pixels
would return p ~ 0, but pixels within a frame are massively correlated, so the effective sample
size is the number of frames, not pixels.
"""
import argparse
import json
import os
import sys

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from run_cell import (  # noqa: E402
    SITE, apply_calibration, fit_key_calibration, gt_key, load_key, load_model, load_pairs,
    make_views,
)


class RouterHead(nn.Module):
    """Per-pixel range-bin classifier on the site's input features."""

    def __init__(self, in_ch, k, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, k, 1))

    def forward(self, x):
        return self.net(x)


def site_module(model):
    m = model
    for p in SITE.split("."):
        m = m[int(p)] if p.isdigit() and not hasattr(m, p) else getattr(m, p)
    return m


def bins_of(depth, edges, dev):
    return torch.bucketize(depth, torch.tensor(edges, device=dev))


def per_image_stats(accs, chances):
    a, c = np.array(accs), np.array(chances)
    d = a - c
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    return a.mean(), a.std(ddof=1), c.mean(), t, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--edges", default="0.5,0.8,1.2")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ordinal", action="store_true",
                    help="soft targets over ADJACENT bins instead of one-hot. Depth bins are "
                         "ordered and the gate blends adjacent bins, so plain cross-entropy "
                         "penalises off-by-one exactly as hard as off-by-three while the "
                         "mixture barely cares. 92.7% of errors are off-by-one.")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-root", default="/data/nf2/train")
    ap.add_argument("--val-root", default="/data/nf2/val")
    ap.add_argument("--init", default="/data/ckpt/student_hab_nav.pt")
    ap.add_argument("--keys", default="/data/keys_habnav")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--out", default="/data/results/rmoe/router.json")
    ap.add_argument("--save", default="/data/ckpt/router_head.pt")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the keyed-router baseline, which needs the precomputed key "
                         "cache; the baseline is already measured at 0.440")
    args = ap.parse_args()

    dev, size = "cuda", (args.res // 14) * 14
    edges = [float(x) for x in args.edges.split(",")]
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train, val = load_pairs(args.train_root, 1), load_pairs(args.val_root, 1)
    model = load_model(args.init, dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    feats = {}
    site_module(model).register_forward_pre_hook(lambda m, i: feats.__setitem__("x", i[0]))

    # ---- baseline: the depth-keyed router, per IMAGE
    accs, chances = [], []
    if not args.skip_baseline:
        cal = fit_key_calibration(args.keys, train, (size, size), dev)
        with torch.no_grad():
            for r, d in val[:args.eval_n]:
                key = apply_calibration(load_key(args.keys, r, dev), cal)
                gt = gt_key(d, (size, size), dev)
                m = (torch.isfinite(gt) & (gt > 0.1) & (gt < 8)
                     & torch.isfinite(key) & (key > 0.05))
                if m.sum() < 1000:
                    continue
                tb, rb = bins_of(gt[m], edges, dev), bins_of(key[m], edges, dev)
                accs.append(float((tb == rb).float().mean()))
                chances.append(float(torch.bincount(tb, minlength=args.k).max() / tb.numel()))
    if accs:
        b_acc, b_sd, b_ch, b_t, b_n = per_image_stats(accs, chances)
        print(f"[router] KEYED baseline over {b_n} images: acc {b_acc:.3f} (sd {b_sd:.3f}) "
              f"vs per-image majority-class {b_ch:.3f}   paired t={b_t:+.2f}", flush=True)
    else:
        b_acc = b_sd = b_ch = float("nan")
        print("[router] keyed baseline skipped (previously measured 0.440)", flush=True)

    centres_m = [float(np.sqrt(a * b)) for a, b in
                 zip([0.2] + edges, edges + [3.0])]        # bin centres for the ordinal target
    head = RouterHead(site_module(model).in_channels, args.k, args.hidden).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    print(f"[router] head params {sum(p.numel() for p in head.parameters()):,}", flush=True)

    for step in range(args.steps):
        idx = rng.integers(0, len(train), args.batch)
        paths = [train[i][0] for i in idx]
        view = make_views(paths, size, dev)
        with torch.no_grad():
            model(view, memory_efficient_inference=True, minibatch_size=args.batch)
        x = feats["x"]
        gts = torch.stack([gt_key(train[i][1], x.shape[-2:], dev) for i in idx], 0)
        tb = bins_of(gts, edges, dev)
        valid = torch.isfinite(gts) & (gts > 0.1) & (gts < 8)
        logits = head(x.float())
        if args.ordinal:
            # soft target: mass split between the two bins straddling this depth, by log-distance
            cen = torch.tensor(centres_m, device=dev).log()
            v = gts.clamp_min(1e-3).log().unsqueeze(1)
            hi = torch.bucketize(v, cen).clamp(1, args.k - 1)
            lo = hi - 1
            w = ((v - cen[lo]) / (cen[hi] - cen[lo]).clamp_min(1e-6)).clamp(0, 1)
            tgt = torch.zeros_like(logits)
            tgt.scatter_(1, lo, 1 - w); tgt.scatter_add_(1, hi, w)
            loss = -(tgt * logits.log_softmax(1)).sum(1)[valid].mean()
        else:
            loss = nn.functional.cross_entropy(logits, tb.clamp(0, args.k - 1),
                                               reduction="none")[valid].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"[router] step {step} loss {loss.item():.4f}", flush=True)

    # ---- learned router accuracy, same per-image protocol
    head.eval()
    l_accs, C = [], np.zeros((args.k, args.k), dtype=np.int64)
    with torch.no_grad():
        for r, d in val[:args.eval_n]:
            view = make_views([r], size, dev)
            model(view, memory_efficient_inference=True, minibatch_size=1)
            x = feats["x"]
            gt = gt_key(d, x.shape[-2:], dev)[None]
            m = torch.isfinite(gt) & (gt > 0.1) & (gt < 8)
            if m.sum() < 1000:
                continue
            pb = head(x.float()).argmax(1)
            tb = bins_of(gt, edges, dev)
            l_accs.append(float((pb[m] == tb[m]).float().mean()))
            for a in range(args.k):
                for b in range(args.k):
                    C[a, b] += int(((tb[m] == a) & (pb[m] == b)).sum())
    l_acc, l_sd, _, _, l_n = per_image_stats(l_accs, chances[:len(l_accs)] or l_accs)
    if accs:
        d = np.array(l_accs) - np.array(accs[:len(l_accs)])
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    else:
        d, t = np.array([float("nan")]), float("nan")
    print(f"\n[router] LEARNED over {l_n} images: acc {l_acc:.3f} (sd {l_sd:.3f})")
    print(f"[router] learned - keyed = {d.mean():+.3f}   paired t={t:+.2f} over {len(d)} images")
    lbl = [f"<{edges[0]}"] + [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)] + [f">{edges[-1]}"]
    print("\nrows = TRUE bin, cols = PREDICTED bin (learned router)")
    print("true/pred".ljust(12) + "".join(f"{l:>10}" for l in lbl))
    for a in range(args.k):
        tot = max(C[a].sum(), 1)
        print(f"{lbl[a]:<12}" + "".join(f"{100*C[a,b]/tot:>9.1f}%" for b in range(args.k)))

    torch.save({"state_dict": head.state_dict(), "edges": edges, "k": args.k}, args.save)
    json.dump({"keyed_acc": b_acc, "keyed_sd": b_sd, "chance": b_ch,
               "learned_acc": l_acc, "learned_sd": l_sd, "delta": float(d.mean()),
               "paired_t": float(t), "n_images": int(len(d)),
               "confusion": C.tolist(), "edges": edges}, open(args.out, "w"), indent=2)
    print(f"\n[router] saved {args.save}")


if __name__ == "__main__":
    main()

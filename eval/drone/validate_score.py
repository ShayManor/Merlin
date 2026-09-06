#!/usr/bin/env python3
"""Does the closed-form score predict the measured (source, destination) ranking? -- RECIRC-01.

The source paper picks its layer pair with a full O(L^2) grid of evaluations and lists
"optimal layers appear domain or problem dependent" as a limitation. `model/recirc/score.py`
claims all 190 pairs can be ranked from ONE forward and ONE backward:

    S(s,d) = <g_d, alpha * (f(z_s) - z_d)> / (1 - G),   G = alpha * gamma * rho(J_{d->s})

This script tests that claim against the exhaustive grid produced by `pair_grid.py`. That
validation is only affordable because the student is small; at 12B the grid does not exist,
which is exactly why the predictor would be worth having.

Reports Spearman and Pearson between predicted and measured, with and without the (1-G)
stability denominator, plus top-k overlap -- the quantity that actually matters, since the
predictor is used to pick a handful of sites to test rather than to estimate effect sizes.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model"))

from eval.drone.perception import Perception  # noqa: E402
from recirc.recirc import tap_list  # noqa: E402
from recirc.score import all_gains, score_pairs  # noqa: E402


def make_forward(perc, rgb, res):
    """A grad-enabled forward returning the metric z-depth tensor for one RGB frame.

    Mirrors Perception._forward but keeps the autograd graph (and skips
    memory_efficient_inference, which is an inference-only path)."""
    import cv2
    cv2.imwrite(perc._tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    s = (res // 14) * 14
    v = perc._load_images([perc._tmp], resize_mode="square", size=s)
    v[0]["img"] = v[0]["img"].to(perc.device)

    def fwd():
        pr = perc.model(v, minibatch_size=1)[0]
        return pr["pts3d_cam"][0, ..., 2]

    return fwd


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--grid", required=True, help="pair_grid.json to validate against")
    ap.add_argument("--metric", default="rel_0.5-0.8",
                    help="measured key to predict; the 0.2-0.5 m bin is too underpowered")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--center", type=float, default=0.9)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--n_calib", type=int, default=6, help="frame PAIRS to average the score over")
    ap.add_argument("--gain_iters", type=int, default=4, help="0 to skip the (1-G) term")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    perc = Perception(args.ckpt, device=args.device)
    labels = [l for l, _ in tap_list(perc.model)]

    files = sorted(glob.glob(os.path.join(args.seqs, "*.npz")))
    calib = []
    for f in files:
        rgb = np.load(f)["rgb"]
        if len(rgb) >= 3:
            calib.append((rgb[len(rgb) // 2 - 1], rgb[len(rgb) // 2]))
        if len(calib) >= args.n_calib:
            break
    print(f"{len(calib)} calibration frame pairs", flush=True)

    gains = {}
    if args.gain_iters:
        fwd = make_forward(perc, calib[0][1], args.res)
        with torch.no_grad():
            gains = all_gains(perc.model, fwd, alpha=args.alpha, iters=args.gain_iters)
        print(f"loop gains: min {min(gains.values()):.3f} max {max(gains.values()):.3f}, "
              f"{sum(1 for g in gains.values() if g >= 1.0)} pairs supercritical", flush=True)

    acc = {}
    for i, (prev_rgb, cur_rgb) in enumerate(calib):
        f_prev = make_forward(perc, prev_rgb, args.res)
        f_cur = make_forward(perc, cur_rgb, args.res)
        # Self-consistency loss: no GT needed, and it is the quantity recirculation could
        # plausibly improve -- agreement between the current prediction and the previous
        # frame's. Using GT here would tune the predictor on the eval signal.
        with torch.no_grad():
            target = f_prev().detach()
        rows = score_pairs(perc.model, f_prev, f_cur,
                           lambda o: (o - target).abs().mean(),
                           alpha=args.alpha, gains=gains)
        for r in rows:
            acc.setdefault((r["source"], r["dest"]), []).append((r["first_order"], r["score"]))
        print(f"  calib {i+1}/{len(calib)}", flush=True)

    pred = {k: (float(np.mean([v[0] for v in vs])), float(np.mean([v[1] for v in vs])))
            for k, vs in acc.items()}

    grid = json.load(open(args.grid))
    meas = {}
    for p in grid["pairs"]:
        m = p["paired"].get(args.metric) or p["paired"].get("whole_absrel")
        if m:
            meas[(p["source"], p["dest"])] = m["mean"]
    common = [k for k in meas if k in pred]
    print(f"\n{len(common)} pairs in both grid and prediction", flush=True)

    y = np.array([meas[k] for k in common])
    x1 = np.array([pred[k][0] for k in common])
    x2 = np.array([pred[k][1] for k in common])
    out = {"metric": args.metric, "n": len(common),
           "spearman_first_order": spearman(x1, y), "pearson_first_order": float(np.corrcoef(x1, y)[0, 1]),
           "spearman_with_gain": spearman(x2, y), "pearson_with_gain": float(np.corrcoef(x2, y)[0, 1])}
    print(f"predicting measured `{args.metric}` (both negative = better)")
    print(f"  first order only : spearman {out['spearman_first_order']:+.3f}  "
          f"pearson {out['pearson_first_order']:+.3f}")
    print(f"  with (1-G) term  : spearman {out['spearman_with_gain']:+.3f}  "
          f"pearson {out['pearson_with_gain']:+.3f}")

    for k in (5, 10, 20):
        best_pred = set(sorted(common, key=lambda c: pred[c][1])[:k])
        best_meas = set(sorted(common, key=lambda c: meas[c])[:k])
        out[f"top{k}_overlap"] = len(best_pred & best_meas)
        print(f"  top-{k} overlap: {len(best_pred & best_meas)}/{k} "
              f"(chance {k*k/len(common):.1f})")
    top = sorted(common, key=lambda c: pred[c][1])[:8]
    print("\n  predicted best 8 (measured delta in brackets):")
    for c in top:
        print(f"    {c[0]:>7}->{c[1]:<7} score {pred[c][1]:+.4e}  [measured {meas[c]:+.4f}]")

    if args.out:
        json.dump({"summary": out,
                   "pairs": [{"source": k[0], "dest": k[1], "first_order": pred[k][0],
                              "score": pred[k][1], "measured": meas[k]} for k in common]},
                  open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

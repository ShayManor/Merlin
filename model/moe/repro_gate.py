#!/usr/bin/env python3
"""Reproduce the published depth table before generating any new number.

Spec: docs/superpowers/specs/2026-08-08-rmoe-01-corrected-rerun.md section 2.

The previous screen generated results on an unvalidated harness and only afterwards discovered
it had the wrong init and the wrong validation split. This gate exists so that cannot recur:
if the two published checkpoints do not reproduce their published per-bin numbers on this
harness, nothing downstream runs.

Reference (docs/merlin-c4-handoff.md, 300 held-out frames, per-frame scale fit):

  true range   student_hab_nav (base)        student_nf (fine-tuned)
  0.2-0.5      0.196 abs / 0.466 rel / +0.193   0.067 / 0.171 / +0.035
  0.5-0.8      0.167 / 0.260 / +0.060           0.078 / 0.123 / +0.020
  0.8-1.2      0.249 / 0.268 / -0.007           0.072 / 0.076 / -0.009
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from run_cell import evaluate, load_model, load_pairs  # noqa: E402

PUBLISHED = {
    "student_hab_nav": {"0.2-0.5": (0.196, 0.466, +0.193),
                        "0.5-0.8": (0.167, 0.260, +0.060),
                        "0.8-1.2": (0.249, 0.268, -0.007)},
    "student_nf":      {"0.2-0.5": (0.067, 0.171, +0.035),
                        "0.5-0.8": (0.078, 0.123, +0.020),
                        "0.8-1.2": (0.072, 0.076, -0.009)},
}
TOL = 0.15          # relative tolerance on abs and rel; bias is reported, not gated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", default="/data/nf2/val")
    ap.add_argument("--ckpt-dir", default="/data/ckpt")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--out", default="/data/results/rmoe/repro_gate.json")
    args = ap.parse_args()

    dev, size = "cuda", (args.res // 14) * 14
    val = load_pairs(args.val_root, 1)
    print(f"[gate] val frames: {len(val)} from {args.val_root}", flush=True)

    res, verdict = {}, "PASS"
    for name, ref in PUBLISHED.items():
        model = load_model(os.path.join(args.ckpt_dir, f"{name}.pt"), dev)
        ev = evaluate(model, val, "D", None, size, dev, args.eval_n)
        res[name] = ev
        print(f"\n{name}   whole_absrel {ev['whole_absrel']:.4f}   n={ev['n']}")
        print(f"{'bin':<10}{'abs':>18}{'rel':>18}{'bias':>18}")
        for b, (pa, pr, pb) in ref.items():
            g = ev["bins"][b]
            if not g["n"]:
                print(f"{b:<10}{'NO SUPPORT':>18}")
                verdict = "FAIL"
                continue
            da = abs(g["abs"] - pa) / pa
            dr = abs(g["rel"] - pr) / pr
            ok = da <= TOL and dr <= TOL
            verdict = verdict if ok else "FAIL"
            print(f"{b:<10}"
                  f"{g['abs']:>9.3f} ({pa:.3f}){'':>1}"
                  f"{g['rel']:>9.3f} ({pr:.3f}){'':>1}"
                  f"{g['bias']:>+9.3f} ({pb:+.3f})"
                  f"   {'ok' if ok else 'MISS'}")
        del model

    print(f"\nGATE: {verdict}   (tolerance {TOL:.0%} relative on abs and rel)")
    if verdict == "FAIL":
        print("Do not proceed. Find the harness difference: render settings, scale fit, or the\n"
              "valid-pixel mask. Generating new numbers on an unvalidated harness is exactly\n"
              "what produced the previous screen's wrong conclusions.")
    json.dump({"verdict": verdict, "tolerance": TOL, "results": res,
               "published": {k: {b: list(v) for b, v in r.items()}
                             for k, r in PUBLISHED.items()}},
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Corrected rerun driver: the K=4 grid.

Spec: docs/superpowers/specs/2026-08-08-rmoe-01-corrected-rerun.md section 4.

Two variables, loss weighting and step budget, everything else fixed. Arm F (dense fine-tune of
the same conv) appears in every cell so "do experts beat just training the layer" is always
answerable. Resumable: a cell whose JSON exists is skipped.

Order is by decision value. The bin-balanced 3000-step cells are the actual hypothesis, so they
fly first; if the box dies early those are the ones worth having.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (loss, steps) in decision order; each runs both arms.
CONFIGS = [("binbal", 3000), ("uniform", 3000), ("binbal", 12000), ("uniform", 12000)]
ARMS = ["P", "F"]
SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/results/rmoe/grid")
    ap.add_argument("--train-root", default="/data/nf2/train")
    ap.add_argument("--val-root", default="/data/nf2/val")
    ap.add_argument("--init", default="/data/ckpt/student_hab_nav.pt")
    ap.add_argument("--keys", default="/data/keys_habnav")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs = [(loss, steps, arm, seed)
            for loss, steps in CONFIGS for arm in ARMS for seed in SEEDS]

    for loss, steps, arm, seed in jobs:
        tag = f"{arm}_{loss}_{steps}_s{seed}"
        path = os.path.join(args.out, f"{tag}.json")
        if os.path.exists(path):
            print(f"[grid] skip {tag}", flush=True)
            continue
        cmd = [sys.executable, os.path.join(HERE, "run_cell.py"),
               "--arm", arm, "--seed", str(seed), "--steps", str(steps),
               "--loss", loss, "--batch", str(args.batch), "--stride", "1",
               "--train-root", args.train_root, "--val-root", args.val_root,
               "--init", args.init, "--keys", args.keys,
               "--eval-n", str(args.eval_n), "--out", path]
        if arm != "F":
            cmd += ["--k", str(args.k), "--rank", str(args.rank)]
        print(f"[grid] {tag}", flush=True)
        if args.dry:
            continue
        t = time.time()
        if subprocess.run(cmd).returncode != 0:
            print(f"[grid] FAILED {tag}", flush=True)
            continue
        print(f"[grid] {tag} took {(time.time() - t) / 60:.1f} min", flush=True)

    done = sorted(f for f in os.listdir(args.out) if f.endswith(".json"))
    print(f"\n[grid] {len(done)}/{len(jobs)} cells", flush=True)
    for f in done:
        r = json.load(open(os.path.join(args.out, f)))
        e = r["final_eval"]
        tel = (r.get("delta_telemetry") or [{}])[-1]
        print(f"  {f[:-5]:<26} whole={e['whole_absrel']:.4f} "
              f"near_rel={e['bins']['0.2-0.5']['rel']:.4f} "
              f"|B|={tel.get('b_norm', float('nan')):.3f} "
              f"cos={tel.get('cos_mean', float('nan')):.3f}", flush=True)


if __name__ == "__main__":
    main()

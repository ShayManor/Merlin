#!/usr/bin/env python3
"""Phase A screen driver. Spec: docs/superpowers/specs/2026-08-08-rmoe-00-crossing-curves.md

18 cells: L_top1 and P at K in {8, 32}, O at K=32, D, three seeds each. Resumable: a cell whose
output JSON already exists is skipped, so a crash costs one cell rather than the grid.

Cells are ordered by decision value. If the box dies early, the ones already flown are the ones
worth having: K=32 first, because that is where the crossing must appear.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (arm, K) in decision-value order. K=32 decides the hypothesis; K=8 anchors it; D is the floor.
CELLS = [("D", 0), ("M", 0), ("F", 0), ("L_top1", 32), ("P", 32), ("O", 32),
         ("L_top1", 8), ("P", 8)]
# Arm M is the deployed MERLIN checkpoint, so it uses its own init.
ARM_INIT = {"M": "/data/ckpt/student_nf.pt"}
SEEDS = [0, 1, 2]
# D and M do no training, so extra seeds are identical; one run each is enough.
SINGLE_SEED = {"D", "M"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/results/rmoe/screen")
    ap.add_argument("--train-root", default="/data/nf/train")
    ap.add_argument("--val-root", default="/data/nf/val")
    ap.add_argument("--init", default="/data/ckpt/student_hab.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--keys", default="/data/keys_hab")
    ap.add_argument("--eval-n", type=int, default=105)
    ap.add_argument("--only", default=None, help="run one cell: ARM:K:SEED")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs = [(a, k, s) for a, k in CELLS
            for s in ([0] if a in SINGLE_SEED else SEEDS)]
    if args.only:
        a, k, s = args.only.split(":")
        jobs = [(a, int(k), int(s))]

    for arm, k, seed in jobs:
        tag = f"{arm}_K{k}_s{seed}"
        path = os.path.join(args.out, f"{tag}.json")
        if os.path.exists(path):
            print(f"[screen] skip {tag} (done)", flush=True)
            continue
        cmd = [sys.executable, os.path.join(HERE, "run_cell.py"),
               "--arm", arm, "--seed", str(seed), "--steps", str(args.steps),
               "--train-root", args.train_root, "--val-root", args.val_root,
               "--init", ARM_INIT.get(arm, args.init), "--out", path,
               "--stride", str(args.stride), "--keys", args.keys,
               "--eval-n", str(args.eval_n)]
        if arm not in ("D", "M", "F"):
            cmd += ["--k", str(k)]
        print(f"[screen] {tag}: {' '.join(cmd)}", flush=True)
        if args.dry:
            continue
        t = time.time()
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print(f"[screen] FAILED {tag} rc={p.returncode}", flush=True)
            continue
        print(f"[screen] {tag} took {(time.time()-t)/60:.1f} min", flush=True)

    done = sorted(f for f in os.listdir(args.out) if f.endswith(".json"))
    print(f"[screen] complete: {len(done)}/{len(jobs)} cells", flush=True)
    for f in done:
        r = json.load(open(os.path.join(args.out, f)))
        e = r["final_eval"]
        nf = e["bins"]["0.2-0.5"]
        print(f"  {f[:-5]:<16} whole={e['whole_absrel']:.4f} "
              f"near_rel={nf['rel']:.4f} near_bias={nf['bias']:+.4f} "
              f"fnear={e['false_near']:.4f} bf1={e['boundary_f1']:.3f} "
              f"ent={r.get('utilisation_entropy', float('nan')):.3f}", flush=True)


if __name__ == "__main__":
    main()

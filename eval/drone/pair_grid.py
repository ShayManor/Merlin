#!/usr/bin/env python3
"""Exhaustive (source, destination) sweep for recirculation -- the RECIRC-01 ground truth.

Everything measured so far used 1 of 190 ordered layer pairs, chosen by transposing the
source paper's Gemma heuristic. That is the weakest part of the design: the paper's own sweeps
show the landscape has structure and only some sites work, and it lists "optimal layers appear
domain or problem dependent" as a limitation.

MERLIN is small enough to just run the grid. 20 taps (12 encoder + 8 AAT blocks, all width 768)
give 190 ordered pairs with source deeper than destination. At a reduced frame budget this is a
few hours on CPU, which no one can do for a 12B model -- and it is exactly the ground truth the
closed-form scorer in model/recirc/score.py has to be validated against.

Ranks on whole-frame AbsRel and the 0.5-0.8 m bin, which are the metrics with enough sequences
to be stable at this budget; the 0.2-0.5 m bin holds too few sequences to rank on (see the
RECIRC-00 spec).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model"))

from eval.drone.eval_recirc import BINS, agg, run_arm, seq_metrics  # noqa: E402
from eval.drone.perception import Perception  # noqa: E402
from recirc.recirc import Recirculator, tap_list  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--center", type=float, default=0.9)
    ap.add_argument("--norm", default="token")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_seqs", type=int, default=10)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    perc = Perception(args.ckpt, device=args.device)
    labels = [l for l, _ in tap_list(perc.model)]
    pairs = [(s, d) for di, d in enumerate(labels) for s in labels[di + 1:]]
    print(f"{len(labels)} taps, {len(pairs)} ordered pairs", flush=True)

    files = sorted(glob.glob(os.path.join(args.seqs, "*.npz")))[: args.max_seqs]
    seqs = []
    for f in files:
        z = np.load(f)
        seqs.append((os.path.basename(f)[:-4],
                     z["rgb"][:: args.stride][: args.max_frames],
                     z["dep"][:: args.stride][: args.max_frames]))
    n_frames = sum(len(s[1]) for s in seqs)
    print(f"{len(seqs)} sequences, {n_frames} frames, "
          f"{len(pairs) * n_frames} forwards to go", flush=True)

    keys = (["whole_absrel", "scale_spread"]
            + [f"rel_{a}-{b}" for a, b in BINS] + [f"bias_{a}-{b}" for a, b in BINS])

    base_rows = run_arm(perc, seqs, args.res)
    base = agg(base_rows, keys)
    base_by_seq = {r["seq"]: r for r in base_rows}
    print(f"base: whole {base['whole_absrel']:.4f}  rel58 {base['rel_0.5-0.8']:.4f}  "
          f"spread {base['scale_spread']:.4f}", flush=True)

    def paired_delta(rows, key):
        """Per-sequence difference against base, which is the statistic that matters: the arms
        share sequences and between-scene variance swamps the effect otherwise."""
        d = [r[key] - base_by_seq[r["seq"]][key] for r in rows
             if r["seq"] in base_by_seq and np.isfinite(r[key])
             and np.isfinite(base_by_seq[r["seq"]][key])]
        if len(d) < 3:
            return None
        d = np.array(d)
        se = d.std(ddof=1) / np.sqrt(len(d))
        return {"n": len(d), "mean": float(d.mean()), "sem": float(se),
                "t": float(d.mean() / se) if se > 0 else 0.0,
                "better": int((d < 0).sum())}

    out = {"meta": {"alpha": args.alpha, "center": args.center, "norm": args.norm,
                    "stride": args.stride, "n_seq": len(seqs), "n_frames": n_frames,
                    "ckpt": args.ckpt, "seqs": args.seqs},
           "base": base, "pairs": []}

    for i, (src, dst) in enumerate(pairs):
        rec = Recirculator(perc.model, src, dst, alpha=args.alpha,
                           norm=args.norm, mode="prev", center=args.center)
        rows = run_arm(perc, seqs, args.res, rec=rec)
        rec.remove()
        rec_out = {"source": src, "dest": dst,
                   "agg": {k: agg(rows, keys)[k] for k in ("whole_absrel", "rel_0.5-0.8",
                                                           "rel_0.2-0.5", "scale_spread")},
                   "paired": {k: paired_delta(rows, k)
                              for k in ("whole_absrel", "rel_0.5-0.8", "scale_spread")}}
        out["pairs"].append(rec_out)
        pw = rec_out["paired"]["whole_absrel"]
        p58 = rec_out["paired"]["rel_0.5-0.8"]
        print(f"[{i+1:3d}/{len(pairs)}] {src:>7}->{dst:<7} "
              f"whole {rec_out['agg']['whole_absrel']:.4f} "
              f"(d {pw['mean']:+.4f} t {pw['t']:+.2f})  "
              f"rel58 d {p58['mean']:+.4f} t {p58['t']:+.2f}" if pw and p58 else
              f"[{i+1:3d}/{len(pairs)}] {src:>7}->{dst:<7} (insufficient)", flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    ranked = sorted([p for p in out["pairs"] if p["paired"]["whole_absrel"]],
                    key=lambda p: p["paired"]["whole_absrel"]["mean"])
    print("\n=== best 10 by paired whole-frame AbsRel ===", flush=True)
    for p in ranked[:10]:
        w, r = p["paired"]["whole_absrel"], p["paired"]["rel_0.5-0.8"]
        print(f"  {p['source']:>7}->{p['dest']:<7} whole {w['mean']:+.4f} t {w['t']:+.2f} "
              f"{w['better']}/{w['n']}   rel58 {r['mean']:+.4f} t {r['t']:+.2f}", flush=True)
    print("=== worst 5 ===", flush=True)
    for p in ranked[-5:]:
        w = p["paired"]["whole_absrel"]
        print(f"  {p['source']:>7}->{p['dest']:<7} whole {w['mean']:+.4f} t {w['t']:+.2f}",
              flush=True)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

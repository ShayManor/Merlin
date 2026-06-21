#!/usr/bin/env python3
"""Aggregate + PAIRED analysis of closed-loop rows. Arms share a fixed episode set
(same ep_idx), so the right test for M1's subtle effect is paired: per-episode
differences, bootstrap CI on the mean difference. Reports per-arm means with 95% CIs
and pairwise paired deltas (B - A) for requested arm pairs."""
import argparse
import json

import numpy as np

METRICS = ["success", "spl", "collisions_per_m", "energy_per_m", "fwd_perr_m",
           "mean_stale_m", "mean_res", "mean_speed", "path_len"]


def boot_mean(xs, n=5000, seed=0):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if xs.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = rng.choice(xs, (n, xs.size), replace=True).mean(1)
    return float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_delta(rows_a, rows_b, metric, n=5000, seed=0):
    """mean(b - a) over shared ep_idx, with paired-bootstrap 95% CI."""
    a = {r["ep_idx"]: r[metric] for r in rows_a}
    b = {r["ep_idx"]: r[metric] for r in rows_b}
    keys = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
    d = np.array([b[k] - a[k] for k in keys], float)
    if d.size == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n, d.size), replace=True).mean(1)
    return {"mean_delta": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5))], "n_pairs": int(d.size),
            "frac_b_better": float((d < 0).mean()) if metric in
            ("collisions_per_m", "fwd_perr_m", "energy_per_m") else float((d > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B pairs for paired delta")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.rows) if l.strip()]
    arms = sorted({r["arm"] for r in rows})

    print("=== per-arm (mean [95% CI]) ===")
    for a in arms:
        rr = [r for r in rows if r["arm"] == a]
        line = f"{a} (n={len(rr)}): "
        for m in ("success", "spl", "collisions_per_m", "fwd_perr_m", "mean_speed"):
            mn, lo, hi = boot_mean([r[m] for r in rr])
            line += f"{m}={mn:.3f}[{lo:.3f},{hi:.3f}] "
        print(line)

    for pair in args.pairs:
        A, B = pair.split(":")
        ra = [r for r in rows if r["arm"] == A]
        rb = [r for r in rows if r["arm"] == B]
        print(f"\n=== paired delta  {B} - {A} ===")
        for m in ("success", "spl", "collisions_per_m", "fwd_perr_m"):
            d = paired_delta(ra, rb, m)
            if d:
                sig = "" if (d["ci"][0] <= 0 <= d["ci"][1]) else "  *"
                print(f"  {m}: {d['mean_delta']:+.4f}  CI[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]"
                      f"  n={d['n_pairs']}  B_better={d['frac_b_better']:.2f}{sig}")


if __name__ == "__main__":
    main()

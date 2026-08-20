#!/usr/bin/env python3
"""Crossover figure: does removing the navmesh safety net (no_slide) make perception
fidelity matter? Plots success and collisions/m for the student (baseline) vs the GT oracle
under the sliding planner vs the no-slide (hard-stop) planner."""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140})


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def boot(xs, n=2000, seed=0):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if xs.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    bs = [rng.choice(xs, xs.size, replace=True).mean() for _ in range(n)]
    return float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def arm_metric(rows, arm_sub, metric):
    return boot([r[metric] for r in rows if arm_sub in r["arm"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True, help="sim_m1 rows (sliding planner)")
    ap.add_argument("--noslide", required=True, help="sim_noslide rows (hard-stop)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sl, ns = load(args.slide), load(args.noslide)

    conds = [("sliding", sl), ("no-slide", ns)]
    arms = [("baseline", "distilled student", "#1f77b4"), ("gt", "GT oracle", "#2ca02c")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    x = np.arange(len(conds)); w = 0.36
    for j, (metric, title) in enumerate([("success", "success rate"),
                                         ("collisions_per_m", "collisions / m")]):
        ax = axes[j]
        for i, (sub, lab, col) in enumerate(arms):
            ys, errlo, errhi = [], [], []
            for _, rows in conds:
                m, lo, hi = arm_metric(rows, sub + "_", metric)
                ys.append(m); errlo.append(m - lo); errhi.append(hi - m)
            ax.bar(x + (i - 0.5) * w, ys, w, label=lab, color=col)
            ax.errorbar(x + (i - 0.5) * w, ys, yerr=[errlo, errhi], fmt="none",
                        color="k", capsize=3, lw=1)
        ax.set_xticks(x); ax.set_xticklabels([c[0] for c in conds])
        ax.set_title(title)
        if j == 0:
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

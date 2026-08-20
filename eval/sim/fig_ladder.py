#!/usr/bin/env python3
"""Depth-quality ladder figure: navigation success and collisions/m as depth improves
from the OOD original student -> domain-adapted student -> GT oracle, under the sliding
vs no-slide collision model. Shows whether perception fidelity changes navigation once
the perception is actually usable."""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140})

# depth quality (held-out apt_5 median abs_rel) for the x-axis ordering
LADDER = [("orig", "orig\n(0.33)"), ("adapted", "adapted\n(0.10)"), ("gt", "GT\n(0.0)")]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def boot(xs, n=2000, seed=0):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if xs.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    bs = [rng.choice(xs, xs.size, replace=True).mean() for _ in range(n)]
    return float(xs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def series(rows, metric):
    ys, lo, hi = [], [], []
    for sub, _ in LADDER:
        m, l, h = boot([r[metric] for r in rows if r["arm"].split("_")[0] == sub])
        ys.append(m); lo.append(m - l); hi.append(h - m)
    return ys, [lo, hi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", required=True)
    ap.add_argument("--noslide", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sl, ns = load(args.slide), load(args.noslide)
    x = np.arange(len(LADDER))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for j, (metric, title) in enumerate([("success", "success rate"),
                                         ("collisions_per_m", "collisions / m")]):
        ax = axes[j]
        for rows, lab, col in [(sl, "sliding", "#1f77b4"), (ns, "no-slide", "#d62728")]:
            ys, err = series(rows, metric)
            ax.errorbar(x, ys, yerr=err, fmt="-o", color=col, label=lab, lw=1.7,
                        ms=5, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in LADDER])
        ax.set_title(title); ax.set_xlabel("depth quality (median abs_rel)")
        if j == 0:
            ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

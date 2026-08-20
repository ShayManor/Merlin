#!/usr/bin/env python3
"""C1 Pareto figure: 230M distilled student vs 1.23B MapAnything teacher, depth error per TUM
sequence (378px, scale-aligned gt abs_rel, 40 frames/seq). The student (5.3x smaller, deployed at
20 FPS / 10W on a $249 Nano) tracks the teacher within ~1-7% on 7/8 sequences; fr1_room is the one
capacity-limited outlier (cluttered scene). Clean/minimal."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "swapc")
os.makedirs(ROOT, exist_ok=True)

seqs = ["fr1_xyz", "fr1_plant", "fr1_360", "fr1_room", "fr2_xyz", "fr2_desk", "fr3_office", "fr3_nostruct"]
teacher = [0.0521, 0.0988, 0.0556, 0.0530, 0.0570, 0.0701, 0.0563, 0.0199]
student = [0.0516, 0.0999, 0.0596, 0.0966, 0.0587, 0.0709, 0.0579, 0.0193]

x = np.arange(len(seqs)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.bar(x - w/2, teacher, w, label="teacher (MapAnything, 1.23B)", color="#8d99ae")
ax.bar(x + w/2, student, w, label="student (MERLIN, 230M, deployed)", color="#2e86ab")
ax.set_xticks(x); ax.set_xticklabels(seqs, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("depth error (scale-aligned abs_rel)")
ax.set_title("C1: 5.3× smaller student tracks the teacher (within ~6% median)", fontsize=10.5)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
# flag the one outlier
ax.annotate("capacity limit\n(one cluttered scene)", xy=(3+w/2, 0.097), xytext=(3.3, 0.115),
            fontsize=7.5, color="#d1495b", arrowprops=dict(arrowstyle="->", color="#d1495b", lw=1.2))
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "c1_pareto.png"), dpi=130)
print("saved", os.path.join(ROOT, "c1_pareto.png"))

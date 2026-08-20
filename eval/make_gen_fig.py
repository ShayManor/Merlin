#!/usr/bin/env python3
"""Generalization figure: deployed student depth error across TUM cameras/scenes (378px, scale-
aligned gt abs_rel, ~40 frames/seq). The model was distilled on fr1_xyz/fr1_plant; cross-camera
fr2/fr3 were never trained. Finding: no train-vs-held-out gap -- cross-camera seqs are as good or
better; the variation is SCENE-driven (foliage/clutter hard even in a training scene), not domain
overfitting. Clean/minimal."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "swapc")
os.makedirs(ROOT, exist_ok=True)

# (label, abs_rel, category)  category: train / heldout(fr1) / cross-camera
D = [
    ("fr1_xyz", 0.0516, "train"), ("fr1_plant", 0.0999, "train"),
    ("fr1_360", 0.0596, "held-out\nscene"), ("fr1_room", 0.0966, "held-out\nscene"),
    ("fr2_xyz", 0.0587, "cross-\ncamera"), ("fr2_desk", 0.0709, "cross-\ncamera"),
    ("fr3_office", 0.0579, "cross-\ncamera"), ("fr3_nostruct", 0.0193, "cross-\ncamera"),
]
col = {"train": "#8d99ae", "held-out\nscene": "#2e86ab", "cross-\ncamera": "#b9770e"}
fig, ax = plt.subplots(figsize=(6.6, 3.7))
xs = range(len(D))
ax.bar(xs, [d[1] for d in D], color=[col[d[2]] for d in D], width=0.7)
ax.set_xticks(list(xs)); ax.set_xticklabels([d[0] for d in D], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("depth error (scale-aligned abs_rel)")
ax.set_title("Generalizes across cameras & scenes (no train/held-out gap)", fontsize=10.5)
import numpy as np
med = float(np.median([d[1] for d in D]))
ax.axhline(med, ls="--", color="0.5", lw=1); ax.text(7.4, med+0.002, f"median {med:.3f}", fontsize=7.5, color="0.4", ha="right")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#8d99ae", "#2e86ab", "#b9770e"]]
ax.legend(handles, ["train (fr1)", "held-out scene", "cross-camera (fr2/fr3)"], fontsize=8, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.0, -0.34, "variation is scene-driven (foliage/clutter hard even in a training scene), not domain overfitting",
        transform=ax.transAxes, fontsize=7, color="0.45")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "generalization.png"), dpi=130)
print("saved", os.path.join(ROOT, "generalization.png"))

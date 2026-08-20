#!/usr/bin/env python3
"""Capacity-vs-training ablation on the fr1_room failure scene (378px, scale-aligned gt abs_rel).
Tests whether the hard-scene capacity limit is closed by more PARAMETERS or more TRAINING.
Controlled: 8L(230M) vs 12L(258M) trained identically (8000 steps, same regenerated cache).
Finding: capacity is ~flat (0.168->0.158, doesn't close the gap), but the SAME 8L arch with more
training (12000 steps, original cache = the deployed v3) reaches 0.097 -- so the failure is
training/data-bound, not parameter-count-bound. Clean/minimal."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "swapc")
os.makedirs(ROOT, exist_ok=True)

# (label, fr1_room scale-aligned abs_rel, color) -- controlled decomposition (same new cache
# except the deployed bar): capacity ~flat, steps = big lever, data = small residual.
B = [
    ("8L 230M\n8k steps", 0.168, "#8d99ae"),
    ("12L 258M\n8k steps", 0.158, "#b9770e"),
    ("8L 230M\n12k steps", 0.117, "#5fa8d3"),
    ("8L 12k steps\n+orig data\n(deployed)", 0.097, "#2e86ab"),
    ("teacher\n1.23B", 0.053, "#cfcfcf"),
]
fig, ax = plt.subplots(figsize=(7.0, 3.9))
xs = range(len(B))
bars = ax.bar(xs, [b[1] for b in B], color=[b[2] for b in B], width=0.66)
for b, (lab, v, c) in zip(bars, B):
    ax.text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(list(xs)); ax.set_xticklabels([b[0] for b in B], fontsize=7.5)
ax.set_ylabel("fr1_room depth error (scale-aligned abs_rel)")
ax.set_title("Hard-scene failure is training-bound, not capacity-bound", fontsize=10.5)
ax.set_ylim(0, 0.20)
ax.spines[["top", "right"]].set_visible(False)
ax.annotate("+params\n~flat", xy=(0.5, 0.182), fontsize=7, color="0.45", ha="center")
ax.annotate("+steps\n−30%", xy=(1.5, 0.150), fontsize=8, color="#2e86ab", ha="center", fontweight="bold")
ax.annotate("", xy=(2, 0.125), xytext=(1, 0.165), arrowprops=dict(arrowstyle="->", color="#2e86ab", lw=1.6))
ax.annotate("+data\n−17%", xy=(2.5, 0.112), fontsize=7.5, color="0.45", ha="center")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "ablation_capacity.png"), dpi=130)
print("saved", os.path.join(ROOT, "ablation_capacity.png"))

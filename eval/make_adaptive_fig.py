#!/usr/bin/env python3
"""Adaptive-compute figure: token-merge robustness + stacked energy ladder.

Left: depth error vs encoder token keep-ratio (4 saliency criteria, fr1_room) ->
token merging is near-free; a feature-norm criterion is best, RGB-gradient worst.
Right: measured model-slice energy/frame ladder -> inter-frame gating x intra-frame
token-merge stack to ~6-10x. All numbers measured on the Orin Nano.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
G = os.path.join(HERE, "..", "results", "gate")
td = json.load(open(os.path.join(G, "tokdrop_fr1room.json")))

C = {"random": "#999999", "rgb": "#b2182b", "featnorm": "#1b7837", "geom": "#2166ac"}
LB = {"random": "random", "rgb": "RGB-gradient (LiteVGGT)", "featnorm": "feature-norm (ToMe)", "geom": "geometry-gradient"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.6))

# --- Left: abs_rel vs keep ratio ---
keeps = td["keeps"]
for k in ["random", "featnorm", "geom", "rgb"]:
    y = [td["abs_rel"][k][str(kp)] for kp in keeps]
    axL.plot(keeps, y, "-o", color=C[k], lw=2.2 if k == "random" else 1.6, ms=3.6, label=LB[k])
axL.invert_xaxis()
axL.set_xlabel("encoder token keep-ratio")
axL.set_ylabel("depth error (abs_rel)")
axL.set_title("Token merging is near-free", fontsize=11)
axL.legend(frameon=False, fontsize=8, loc="upper left")
axL.spines[["top", "right"]].set_visible(False)
axL.annotate("drop 75% of tokens\nfor +0.3%", xy=(0.25, td["abs_rel"]["random"]["0.25"]),
             xytext=(0.45, 0.176), fontsize=8, color="#555",
             arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))

# --- Right: measured model-slice energy ladder ---
# full=0.57 J/frame (GPU rail). token-merge keep0.35 ~= 1.5x full-model (encoder 2.4x, ~part of fwd).
# gating h=0.8 -> 0.027+(1-0.8)*0.54 = 0.135 J (4.2x). stacked ~= /1.5.
labels = ["full", "+ token-merge\n(keep 0.35)", "+ gating\n(h=0.8)", "stacked"]
energy = [0.57, 0.57 / 1.5, 0.135, 0.135 / 1.5]
mult = [0.57 / e for e in energy]
cols = ["#cccccc", "#7fbf7b", "#7fbf7b", "#1b7837"]
x = np.arange(4)
axR.bar(x, energy, color=cols, width=0.62)
for i, (e, m) in enumerate(zip(energy, mult)):
    axR.text(i, e + 0.012, f"{m:.1f}x", ha="center", fontsize=9,
             fontweight="bold" if i == 3 else "normal")
axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8.5)
axR.set_ylabel("model-slice energy / frame (J)")
axR.set_title("Stacked FPS/W (measured on Orin Nano)", fontsize=11)
axR.set_ylim(0, 0.66)
axR.spines[["top", "right"]].set_visible(False)
axR.annotate("~6x conservative\n~10x at h=0.9/keep0.25", xy=(3, energy[3]),
             xytext=(1.55, 0.42), fontsize=8, color="#555",
             arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))

fig.tight_layout()
out = os.path.join(G, "adaptive_fig.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)

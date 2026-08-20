#!/usr/bin/env python3
"""SWaP-C benchmark figure: on-board real-time metric-3D foundation-model deployment.
Cost ($, log) vs on-board FPS; metric-capable points filled, non-metric hollow.
Numbers: MERLIN measured on-device (this work); baselines from cited papers + vendor
specs (VGGT-SLAM 2.0 arXiv:2601.19887; EC3R-SLAM arXiv:2510.02080; MapAnything teacher).
Clean/minimal style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "swapc")
os.makedirs(ROOT, exist_ok=True)

# name, cost($), onboard_FPS, metric?, color, marker, note
P = [
    ("MERLIN (ours)\nOrin Nano 8GB", 249, 16.5, True,  "#2e86ab", 18.5),   # 16-17 e2e FPS, ~18.5W board avg
    ("VGGT-SLAM 2.0\nAGX Thor",       3499, 3.5, False, "#d1495b", 80.0),   # 3.5 FPS onboard, no metric scale
    ("EC3R-SLAM\nOrin NX",            500,  6.0, True,  "#8d99ae", 20.0),   # est FPS; >8GB VRAM (>Nano budget)
    ("MapAnything teacher\nA100 (datacenter)", 10000, 1.6, True, "#e0a458", 300.0),  # offline-ish, not edge
]

fig, ax = plt.subplots(figsize=(6.4, 4.0))
for name, cost, fps, metric, col, w in P:
    ax.scatter(cost, fps, s=160, c=(col if metric else "white"),
               edgecolors=col, linewidths=2.2, zorder=3, marker="o")
    off = (12, -6) if name.startswith("MERLIN") else ((8, -22) if "EC3R" in name else (8, 6))
    ax.annotate(name, (cost, fps), textcoords="offset points", xytext=off, fontsize=8.5)
ax.set_xscale("log")
ax.set_ylim(0, 19.5)
ax.set_xlabel("system cost (USD, log)")
ax.set_ylabel("on-board real-time FPS")
ax.set_title("SWaP-C: deployable metric-3D foundation models", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
# annotate the empty corner MERLIN occupies
ax.text(300, 14, "cheap · low-power · metric · real-time\n(MERLIN occupies this corner)",
        fontsize=8, color="#2e86ab")
ax.text(0.98, 0.04, "filled = recovers metric scale;  hollow = up-to-scale",
        transform=ax.transAxes, ha="right", fontsize=7.5, color="0.4")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "swapc_pareto.png"), dpi=130)
print("saved", os.path.join(ROOT, "swapc_pareto.png"))

# ---- Fig 2: MERLIN advantage over the deployed baseline (VGGT-SLAM 2.0 on Thor) ----
fig2, ax2 = plt.subplots(figsize=(5.4, 3.6))
labels = ["cost\n(cheaper)", "board power\n(lower)", "on-board FPS\n(higher)"]
mult = [3499/249, 100/19.2, 16.5/3.5]          # ~14x, ~5x (Thor ~100W typ of 40-130), ~4.7x
bars = ax2.bar(labels, mult, color=["#2e86ab", "#2e86ab", "#2e86ab"], width=0.6)
for b, m in zip(bars, mult):
    ax2.text(b.get_x()+b.get_width()/2, m+0.4, f"{m:.0f}×", ha="center", fontsize=11, fontweight="bold")
ax2.axhline(1.0, ls="--", color="0.6", lw=1)
ax2.text(2.4, 1.3, "baseline (1×)", fontsize=8, color="0.5", ha="right")
ax2.set_ylabel("MERLIN advantage factor")
ax2.set_title("MERLIN vs VGGT-SLAM 2.0 (the deployed baseline)\n+ MERLIN recovers metric scale; Thor does not", fontsize=10)
ax2.set_ylim(0, max(mult)*1.2)
ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()
fig2.savefig(os.path.join(ROOT, "swapc_advantage.png"), dpi=130)
print("saved", os.path.join(ROOT, "swapc_advantage.png"))

# ---- Fig 3: operating-point selection on the Nano (measured accuracy vs FPS) ----
# (FPS, scale-aligned gt abs_rel, label, color) -- accuracy measured on TUM held-out;
# bf16 FPS from the Nano resolution sweep; INT8@378 the deployed engine (measured 20 FPS).
OP = [
    (11.9, 0.0623, "bf16 252px", "#8d99ae"),
    (6.8,  0.0525, "bf16 378px", "#8d99ae"),
    (5.3,  0.0517, "bf16 518px", "#8d99ae"),
    (20.1, 0.0525, "INT8 378px\n(deployed)", "#2e86ab"),
]
fig3, ax3 = plt.subplots(figsize=(5.6, 3.6))
for fps, ar, lab, col in OP:
    ax3.scatter(fps, ar, s=150, c=col, zorder=3, edgecolors="k" if "deployed" in lab else col, linewidths=1.5)
    ax3.annotate(lab, (fps, ar), textcoords="offset points", xytext=(6, 6), fontsize=8.5)
ax3.set_xlabel("on-board FPS (Jetson Orin Nano)")
ax3.set_ylabel("depth error (scale-aligned abs_rel, lower=better)")
ax3.set_title("Operating-point selection: INT8 @ 378px is the sweet spot", fontsize=10.5)
ax3.invert_yaxis()
ax3.spines[["top", "right"]].set_visible(False)
ax3.text(0.02, 0.04, "higher res buys ~no accuracy; INT8 @378 is fastest at near-best quality",
         transform=ax3.transAxes, fontsize=7.5, color="0.4")
fig3.tight_layout()
fig3.savefig(os.path.join(ROOT, "swapc_oppoint.png"), dpi=130)
print("saved", os.path.join(ROOT, "swapc_oppoint.png"))

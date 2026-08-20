#!/usr/bin/env python3
"""End-to-end gating Pareto through the REAL student (eval/real_gate.py output).

skip-rate vs output depth error for metric / appearance / fixed-rate gates, with the
measured model-slice energy multiplier on the top axis. Shows content-driven gating
(esp. metric) sustains high skip at <= baseline error where appearance/fixed-rate do not.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt

G = os.path.join(os.path.dirname(__file__), "..", "results", "gate")
C = {"metric": "#1b7837", "appear": "#b2182b", "fixedrate": "#999999"}
LB = {"metric": "metric gate (ToF)", "appear": "appearance gate", "fixedrate": "fixed-rate"}

def mult(h): return 0.567 / (0.027 + (1 - h) * 0.54)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
for ax, seq, title in [(axes[0], "realgate_v3_fr1room.json", "fr1_room"),
                       (axes[1], "realgate_v3_fr2desk.json", "fr2_desk (held-out cam)")]:
    d = json.load(open(os.path.join(G, seq)))
    base = min(p[1] for p in d["pareto"]["metric"] if p[0] < 0.15)
    for g in ["metric", "appear", "fixedrate"]:
        pts = sorted(d["pareto"][g]); h = [p[0] for p in pts]; e = [p[1] for p in pts]
        ax.plot(h, e, "-o", color=C[g], lw=1.8, ms=4, label=LB[g])
    ax.axhline(base, ls="--", color="#bbbbbb", lw=1, zorder=0)
    ax.text(0.02, base + 0.0008, "per-frame baseline (v3)", fontsize=7.5, color="#888")
    ax.set_xlabel("skip-rate  h"); ax.set_title(title, fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim())
    ticks = [0.0, 0.5, 0.8, 0.9]; ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{mult(t):.1f}x" for t in ticks], fontsize=8)
    ax2.spines[["right"]].set_visible(False)
axes[0].set_ylabel("output depth error (abs_rel)")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
axes[1].text(0.5, 1.16, "model-slice energy reduction (measured)", transform=axes[1].transAxes,
             ha="center", fontsize=8.5, color="#444")
fig.suptitle("Gated refresh on the deployed (v3) model: gates tie on smooth motion", fontsize=10.5, y=1.02)
fig.tight_layout()
out = os.path.join(G, "realgate_fig.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)

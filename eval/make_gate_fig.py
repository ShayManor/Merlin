#!/usr/bin/env python3
"""Figure for EXP-G: metric (ToF) gate vs appearance/fixed-rate gate.

Left: skip-rate vs incurred depth-error Pareto (the headline -- metric gate skips
more frames at equal error). Right: staleness-prediction quality (Spearman of each
gate signal vs true reuse error), averaged across sequences.

Reads results/gate/gate_*.json (scp'd from the Jetson). Minimal styling.
"""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
JDIR = os.path.join(HERE, "..", "results", "gate")
files = sorted(glob.glob(os.path.join(JDIR, "gate_*.json")))
data = [json.load(open(f)) for f in files]
assert data, f"no json in {JDIR}"

C = {"tof": "#1b7837", "appear": "#b2182b", "fixedrate": "#999999", "oracle": "#2166ac"}
LB = {"tof": "ToF metric gate", "appear": "appearance gate",
      "fixedrate": "fixed-rate", "oracle": "oracle (lower bound)"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.6))

# --- Left: Pareto, averaged across sequences ---
for key in ["oracle", "tof", "appear", "fixedrate"]:
    curves = np.array([[p[1] for p in d[f"pareto_{key}"]] for d in data])
    fracs = [p[0] for p in data[0][f"pareto_{key}"]]
    mean = curves.mean(0)
    style = "--" if key == "oracle" else "-"
    axL.plot(fracs, mean, style, color=C[key], lw=2.2 if key == "tof" else 1.6,
             marker="o", ms=3.5, label=LB[key])
axL.set_xlabel("skip-rate  h  (fraction of frames reusing the map)")
axL.set_ylabel("depth error on skipped frames (abs_rel)")
axL.set_title("Compute-skip Pareto", fontsize=11)
axL.legend(frameon=False, fontsize=8.5, loc="upper left")
axL.spines[["top", "right"]].set_visible(False)
# annotate the equal-quality skip-rate gap at the energy-relevant band
axL.axhline(0.029, color="#cccccc", lw=0.8, zorder=0)

# --- Right: staleness-prediction quality (Spearman) ---
seqs = [d["seq"].replace("rgbd_dataset_", "") for d in data]
tof = [d["spearman_tof_vs_trueerr"] for d in data]
app = [d["spearman_appear_vs_trueerr"] for d in data]
fix = [d["spearman_lag_vs_trueerr"] for d in data]
x = np.arange(len(seqs)); w = 0.26
axR.bar(x - w, tof, w, color=C["tof"], label="ToF metric")
axR.bar(x, app, w, color=C["appear"], label="appearance")
axR.bar(x + w, fix, w, color=C["fixedrate"], label="fixed-rate")
axR.set_xticks(x); axR.set_xticklabels(seqs, fontsize=8.5, rotation=12)
axR.set_ylabel("staleness prediction\nSpearman(signal, true error)")
axR.set_title("Which gate knows when reuse breaks", fontsize=11)
axR.legend(frameon=False, fontsize=8.5)
axR.spines[["top", "right"]].set_visible(False)
axR.set_ylim(0, 1)

fig.tight_layout()
out = os.path.join(JDIR, "gate_fig.png")
fig.savefig(out, dpi=160, bbox_inches="tight")
print("wrote", out)

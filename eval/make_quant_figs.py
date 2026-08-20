#!/usr/bin/env python3
"""Figures for the role-bimodal quantization study. Clean/minimal style.
Reads results/quant/qd_all.json (aggregated experiment outputs)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "results", "quant")
D = json.load(open(os.path.join(ROOT, "qd_all.json")))

# palette
C_ENC, C_AAT, C_RAIL, C_SCALE, C_GEOM = "#d1495b", "#2e86ab", "#8d99ae", "#e0a458", "#2e86ab"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "axes.grid": False})

def runs(name): return {r["tag"]: r for r in D[name]["runs"]}

# ---- Fig 1: role-bimodal component sensitivity @3-bit (the headline) ----
def fig_rolebimodal():
    c = runs("qd_comp3.json")
    comps = [("encoder", "DINOv2 encoder\n(pretrained, 86.6M)", C_ENC),
             ("info_sharing", "3D-reasoning AAT\n(distilled, 56.6M)", C_AAT),
             ("scale_rail", "scale/pose heads\n(distilled, 2.0M)", C_RAIL)]
    keys = {"encoder": "only_encoder", "info_sharing": "only_info_sharing", "scale_rail": "only_scale_rail"}
    vals = [c[keys[k]]["geom_absrel_vs_ref"] for k, _, _ in comps]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar([l for _, l, _ in comps], vals, color=[cc for *_, cc in comps], width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.0006, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_ylabel("geometry error from 3-bit quant\n(abs_rel vs fp32)")
    ax.set_title("Quantization fragility is role-bimodal", fontsize=12)
    ax.set_ylim(0, max(vals)*1.25)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig1_rolebimodal.png")); plt.close()

# ---- Fig 2: bit-width sweep, scale vs geometry decomposition ----
def fig_bitsweep():
    u = D["qd_uniform.json"]["runs"]
    bits = [r["bits"] for r in u]; sd = [r["scale_drift_vs_ref"] for r in u]; gm = [r["geom_absrel_vs_ref"] for r in u]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(bits, gm, "o-", color=C_GEOM, label="relative-geometry error")
    ax.plot(bits, sd, "s--", color=C_ENC, label="global-scale drift")
    ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("weight bit-width (uniform)"); ax.set_ylabel("error vs fp32 (log)")
    ax.axvspan(2.4, 1.6, color="0.92"); ax.text(2.0, gm[-1]*0.5, "2-bit\ncliff", ha="center", fontsize=9, color="0.4")
    ax.set_title("Geometry degrades first; scale only collapses at the cliff", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig2_bitsweep.png")); plt.close()

# ---- Fig 3: weight-outlier dose-response (mechanism) ----
def fig_mechanism():
    ws = D["qd_stats.json"]["weight_stats"]
    # raw DINOv2 measured separately (M-test-1): in_spread 4.60
    labels = ["raw DINOv2\n(pretrained)", "encoder\n(light distill)", "3D-reasoning AAT\n(distilled from scratch)"]
    spreads = [4.60, ws["encoder"]["in_spread"], ws["info_sharing"]["in_spread"]]
    cols = [C_ENC, C_SCALE, C_AAT]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(labels, spreads, color=cols, width=0.6)
    for b, v in zip(bars, spreads):
        ax.text(b.get_x()+b.get_width()/2, v+0.07, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("input-channel weight-outlier spread\n(max / median)")
    ax.set_title("More distillation -> fewer outliers -> more quantizable", fontsize=11)
    ax.annotate("", xy=(2, 1.7), xytext=(0, 4.7), arrowprops=dict(arrowstyle="->", color="0.5", lw=1.3))
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig3_mechanism.png")); plt.close()

# ---- Fig 4: asymmetric-allocation Pareto ----
def fig_pareto():
    r = runs("qd_recipe_v3.json")
    # tag, label, color, marker, (dx,dy) label offset in points
    pts = [("uniform_w8", "uniform W8", C_RAIL, "o", (-10, 8)),
           ("uniform_w4", "uniform W4", C_RAIL, "o", (6, 6)),
           ("uniform_w3", "uniform W3", C_RAIL, "o", (6, 4)),
           ("uniform_w3_g64", "W3 + group-quant", C_AAT, "D", (8, -2)),
           ("enc4_rest3", "asym enc4/rest3", C_ENC, "s", (6, -14)),
           ("enc6_rest3", "asym enc6/rest3", C_ENC, "s", (6, 4)),
           ("enc8_rest3", "asym enc8/rest3", C_ENC, "s", (6, 4))]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for tag, lab, col, mk, off in pts:
        if tag not in r: continue
        x, y = r[tag]["avg_bits"], r[tag]["geom_absrel_vs_ref"]
        ax.scatter(x, y, color=col, marker=mk, s=70, zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, fontsize=8.5)
    ax.set_xlim(2.4, 8.8)
    ax.set_xlabel("average weight bit-width"); ax.set_ylabel("geometry error vs fp32")
    ax.set_yscale("log")
    ax.set_title("Characterization-guided allocation dominates uniform", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig4_pareto.png")); plt.close()

# ---- Fig 5: activation quant W vs A ----
def fig_actquant():
    a = runs("qd_actquant_v3.json")
    order = [("W8A8_all","W8A8"),("W4A8_all","W4A8"),("W3A8_all","W3A8"),
             ("W8A4_all","W8A4"),("W4A4_all","W4A4")]
    labs=[l for _,l in order]; vals=[a[t]["geom_absrel_vs_ref"] for t,_ in order]
    cols=[C_AAT if "A8" in l else C_ENC for l in labs]
    fig, ax = plt.subplots(figsize=(5.6,3.6))
    bars=ax.bar(labs, vals, color=cols, width=0.62)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, v+0.002, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("geometry error vs fp32"); ax.set_title("Activations bind harder than weights (A4 = cliff)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig5_actquant.png")); plt.close()

# ---- Fig 6: QAD closes the encoder gap (the method) ----
def fig_qad():
    # encoder-only 3-bit quantization error (geom vs each model's own fp32)
    labels = ["original\n(PTQ)", "QAD v1\n(fixed-STE)", "QAD v2\n(LSQ+anchor)"]
    vals = [0.0228, 0.0136, 0.0000]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(labels, vals, color=[C_ENC, C_SCALE, C_AAT], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.0006, f"{v:.4f}", ha="center", fontsize=10)
    ax.axhline(0.0010, ls="--", color="0.5", lw=1)
    ax.text(2.35, 0.0016, "distilled-AAT\nrobustness floor", fontsize=8, color="0.4", ha="right")
    ax.set_ylabel("encoder 3-bit quantization error\n(geom abs_rel vs fp32)")
    ax.set_title("QAD makes the pretrained encoder 3-bit-native", fontsize=11)
    ax.set_ylim(0, 0.026)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig6_qad.png")); plt.close()

# ---- Fig 7: teacher vs student, same block opposite quantizability ----
def fig_teacher():
    import numpy as np
    groups = ["image encoder", "3D-reasoning /\ninfo-sharing"]
    teacher = [0.0516, 0.0121]   # pretrained
    student = [0.0228, 0.0010]   # distilled
    x = np.arange(len(groups)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    b1 = ax.bar(x - w/2, teacher, w, color=C_ENC, label="teacher 1.23B (pretrained)")
    b2 = ax.bar(x + w/2, student, w, color=C_AAT, label="distilled student (distilled)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.0012, f"{b.get_height():.4f}",
                    ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("3-bit quantization error (geom vs fp32)")
    ax.set_title("Distillation makes the same block more quantizable", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 0.062)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig7_teacher.png")); plt.close()

# ---- Fig 8: IMU-anchored quantization (scale is recoverable) ----
def fig_imu():
    bits = [8, 6, 5, 4, 3, 2]
    S_naive = [0.074, 0.0733, 0.074, 0.0747, 0.0758, 1.255]
    S_imu   = [0.0539, 0.0529, 0.0524, 0.0578, 0.0536, 0.1502]
    T_naive = [0.0789, 0.08, 0.0806, 0.0824, 0.1187, 0.5776]
    T_imu   = [0.054, 0.0538, 0.0542, 0.0553, 0.0568, 0.1736]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for ax, (nm, nv, iv) in zip(axes, [("distilled student (230M)", S_naive, S_imu),
                                       ("teacher (1.23B)", T_naive, T_imu)]):
        ax.plot(bits, nv, "o-", color=C_ENC, label="naive metric output")
        ax.plot(bits, iv, "s--", color=C_AAT, label="IMU-recovered (scale re-anchored)")
        ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel("weight bit-width"); ax.set_title(nm, fontsize=10)
    axes[0].set_ylabel("metric depth error (abs_rel, log)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("An IMU recovers quantization-induced scale drift -> aggressive quant becomes usable",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig8_imu.png")); plt.close()

# ---- Fig 9: IMU-anchoring generalizes to pruning (compression-general) ----
def fig_prune():
    sp = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    naive = [0.072, 0.064, 0.089, 0.164, 0.205, 0.194]
    imu   = [0.051, 0.049, 0.056, 0.109, 0.138, 0.137]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot(sp, naive, "o-", color=C_ENC, label="naive metric output")
    ax.plot(sp, imu, "s--", color=C_AAT, label="IMU-recovered (scale re-anchored)")
    ax.axvspan(0.78, 0.97, color="0.92")
    ax.text(0.87, 0.16, "geometry\nbreaks", ha="center", fontsize=8, color="0.4")
    ax.set_xlabel("magnitude-pruning sparsity"); ax.set_ylabel("metric depth error (abs_rel)")
    ax.set_title("IMU recovery generalizes from quantization to pruning", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig9_prune.png")); plt.close()

# ---- Fig 10: scale-free QAD makes 2-bit viable (the method capstone) ----
def fig_scalefree():
    labels = ["naive\n(no IMU)", "PTQ\n+ IMU", "scale-dep\nQAD + IMU", "scale-free\nQAD + IMU"]
    vals = [1.255, 0.150, 0.0544, 0.0493]
    cols = [C_ENC, C_SCALE, C_RAIL, C_AAT]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(labels, vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*1.12, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.axhline(0.054, ls="--", color="0.5", lw=1)
    ax.text(3.4, 0.058, "fp-quality\ngeometry floor", fontsize=8, color="0.4", ha="right")
    ax.set_ylabel("2-bit IMU-recovered error (abs_rel, log)")
    ax.set_title("Scale-free QAD makes 2-bit metric-3D viable for VI", fontsize=10.5)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "fig10_scalefree.png")); plt.close()

if __name__ == "__main__":
    fig_rolebimodal(); fig_bitsweep(); fig_mechanism(); fig_pareto(); fig_actquant(); fig_qad(); fig_teacher(); fig_imu(); fig_prune(); fig_scalefree()
    print("figures written to", ROOT)

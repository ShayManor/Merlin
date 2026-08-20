#!/usr/bin/env python3
"""Accuracy at equal measured MILLISECONDS on the Nano, not at equal parameters.

Every arm-vs-arm verdict in this thread so far is matched-parameter, and under that rule the
mixture's extra experts are free. On the deployed device they are not: routing, sorting and
capacity padding all cost time, and the dense controls can spend that time on width. This joins
the measured on-device latency to the step-147,000 accuracy table and asks the only question the
deployment story cares about.

Accuracy source: log.md L-ORE-95, step 147,000, near-field 0.2-0.5 m rel is the primary metric,
run-to-run scatter 0.0087. Latency source: bench_ffn_moe_latency.py on the Orin Nano at
MAXN_SUPER. Nothing here re-derives a number; if the two disagree the table wins and this file
is stale.
"""
import argparse
import glob
import json
import os

# arm -> (trainable, active/token, whole, near, unseen). Step 147,000.
ACC = {
    "dense":     (37.8e6, 37.8e6, 0.0927, 0.1156, 0.0705),
    "widen165":  (39.8e6, 39.8e6, 0.0873, 0.1185, 0.0700),
    "U8_top2":  (302.2e6, 75.6e6, 0.0862, 0.1323, 0.0691),
    "U2_top1":   (75.6e6, 37.8e6, 0.0939, 0.1351, 0.0696),
    "U8_top1":  (302.2e6, 37.8e6, 0.0957, 0.1457, 0.0725),
}
# widen168 is widen165 with the hidden dim padded 3237 -> 3240. Not trained: three extra hidden
# units cannot move a depth metric, but the alignment they buy halves the latency, so the row is
# carried with widen165's accuracy and flagged.
INFERRED_ACC = {"widen168": "widen165"}
SCATTER = 0.0087
BLOCKS = 8
# Whole-model references, both measured on this board. A site ratio badly overstates the frame
# penalty: the AAT FFN is only ~4.2% of the fp16 model, so scaling has to go through the frame.
FP16_MODEL_MS = 169.49          # capacity.json base:8, fp16 PyTorch, 378 px (size_moe.MS_FP16_TORCH)
INT8_MODEL_MS = 49.66           # merlin_v2_int8.engine, deployed INT8-TRT, 20.1 FPS
INT8_FFN_MS = 2.1               # the 8 AAT FFN blocks' share of it (L-ORE-18)


def derive(row, dense_site, trt=None):
    """Frame-level cost, in the accounting form L-ORE-18 established.

    fp16 is a DELTA against the measured fp16 frame time, which cancels the fact that the 169.49
    base was recorded when the dense site read 0.888 ms rather than this run's 1.09.

    The INT8 columns are EXTRAPOLATIONS and are labelled as such. Two deployment routes exist and
    they are NOT interchangeable:

      trt      the mixture stays inside the engine, using the one-hot static dispatch that
               actually builds. Scaled by the RATIO of standalone TRT engines, never by their
               absolute times: a standalone dense FFN engine measures 1.017 ms and eight of them
               would be 8.1 ms, against the deployed engine's whole 2.1 ms FFN share, because a
               fused whole-model engine does not pay per-layer I/O. Substituting absolutes would
               inflate every mixture row by ~4x.
      hybrid   the mixture runs as a fp16 Triton kernel OUTSIDE the engine. Here the absolute
               PyTorch wall time IS the cost, because those blocks genuinely leave TensorRT.
    """
    site = row.get("site_ms")
    if site is None or dense_site is None:
        return
    row["frame_ms_fp16_derived"] = FP16_MODEL_MS + BLOCKS * (site - dense_site)
    if row["arm"].startswith("U"):
        row["frame_ms_int8_hybrid"] = INT8_MODEL_MS - INT8_FFN_MS + BLOCKS * site
        if trt is None:
            row["int8_basis"] = ("no TRT engine measured for this arm: trt_ffn_moe.py covers "
                                 "top-1 dispatch only, and top-2 doubles the dispatch slots")
        if trt:
            row["trt_ratio_vs_dense"] = trt["ratio"]
            row["frame_ms_int8_trt"] = INT8_MODEL_MS - INT8_FFN_MS + INT8_FFN_MS * trt["ratio"]
    else:
        row["frame_ms_int8_trt"] = (INT8_MODEL_MS - INT8_FFN_MS
                                    + INT8_FFN_MS * row["dh"] / 3072)
        row["int8_basis"] = ("dense, fully INT8-TRT; only the FFN share scales with width. The "
                             "+165 arm's fp16 misalignment penalty is a cuBLAS artifact and is "
                             "NOT carried into INT8, where TRT picks its own kernels -- unmeasured")
    best = row.get("frame_ms_int8_trt") or row.get("frame_ms_int8_hybrid")
    for k_, v in (("fps_int8_trt", row.get("frame_ms_int8_trt")),
                  ("fps_int8_hybrid", row.get("frame_ms_int8_hybrid"))):
        if v:
            row[k_] = 1000 / v
    row["frame_ms_int8_best"] = min(x for x in (row.get("frame_ms_int8_trt"),
                                                row.get("frame_ms_int8_hybrid")) if x)
    row["fps_int8_best"] = 1000 / row["frame_ms_int8_best"]


def load_trt(root, cap_factor="1.5"):
    """Best TRT dispatch ratio vs a dense FFN engine, both measured standalone at INT8.

    cap_factor 1.5 is the smallest sweep point that is DROP-FREE and bit-exact against reference
    top-1 on this data. cf 1.0 is cheaper and is not a valid arm: it drops 3.6% of tokens.
    """
    out = {}
    for f in glob.glob(os.path.join(root, "**", "trt_k*.json"), recursive=True):
        d = _load(f)
        if not d:
            continue
        by = {r["variant"]: r for r in d["rows"]}
        dn = by.get("dense", {}).get("int8", {}).get("ms")
        mx = by.get(f"caponehot_cf{cap_factor}", {}).get("int8", {}).get("ms")
        if dn and mx:
            out[d["k"]] = {"dense_int8_ms": dn, "moe_int8_ms": mx, "ratio": mx / dn,
                           "cap_factor": float(cap_factor),
                           "slots": by[f"caponehot_cf{cap_factor}"]["slots"]}
    return out


K_OF = {"U8_top1": 8, "U8_top2": 8, "U2_top1": 2}
TOPK_OF = {"U8_top1": 1, "U8_top2": 2, "U2_top1": 1}
DH = {"dense": 3072, "widen165": 3237, "widen168": 3240, "widen5200": 8272,
      "U8_top1": 3072, "U8_top2": 3072, "U2_top1": 3072}
LABEL = {"dense": "dense FT", "widen165": "dense widen +165 (as trained, ld unaligned)",
         "widen168": "dense widen +168 (ld aligned)", "U8_top1": "U8 top-1",
         "U8_top2": "U8 top-2", "U2_top1": "U2 top-1"}


def _load(f):
    """Skip a truncated or empty json instead of taking the whole analysis down. A crashed board
    leaves 0-byte results behind: trt_k2.json printed "wrote" and landed as 0 bytes."""
    try:
        return json.load(open(f))
    except (json.JSONDecodeError, OSError):
        print(f"  [skip] {os.path.basename(f)} is empty or truncated")
        return None


def collect(root):
    """Prefer a measured whole-model frame time; fall back to the site cost scaled by 8 blocks."""
    model, site = {}, {}
    for f in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        d = _load(f)
        if not d:
            continue
        for r in d.get("model", []):
            model[r["arm"]] = r
        for r in d.get("site", []):
            ms = r.get("graph_ms") or r["ms"]
            # keep the best dispatch implementation per arm: the fused kernel is the deployable
            # one, and the block_m sweep lands several rows for the same arm
            if r["arm"] not in site or ms < site[r["arm"]]["ms_best"]:
                site[r["arm"]] = {**r, "ms_best": ms}
    return model, site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/rmoe/latency")
    ap.add_argument("--fig", default="results/rmoe/latency/matched_latency.png")
    ap.add_argument("--out", default="results/rmoe/latency/matched_latency.json")
    args = ap.parse_args()

    model, site = collect(args.root)
    trt = load_trt(args.root)
    if not site and not model:
        raise SystemExit(f"no latency json under {args.root}")
    dense_site = site["dense"]["ms_best"] if "dense" in site else None

    rows = []
    for arm in ("dense", "widen168", "widen165", "U2_top1", "U8_top1", "U8_top2"):
        acc = ACC.get(arm) or ACC[INFERRED_ACC[arm]]
        r = {"arm": arm, "label": LABEL[arm], "trainable_M": acc[0] / 1e6,
             "active_M": acc[1] / 1e6, "whole": acc[2], "near": acc[3], "unseen": acc[4],
             "accuracy_inferred": arm in INFERRED_ACC}
        r["dh"] = DH[arm]
        if arm in site:
            r["site_ms"] = site[arm]["ms_best"]
            r["site_vs_dense"] = site[arm]["ms_best"] / dense_site if dense_site else None
        # trt_ffn_moe.py measures TOP-1 dispatch only. A top-2 arm doubles the dispatch slots
        # and has no measured engine, so it must not inherit top-1's ratio.
        r["top_k"] = TOPK_OF.get(arm, 1)
        derive(r, dense_site, trt.get(K_OF.get(arm)) if r["top_k"] == 1 else None)
        if arm in model:
            r["frame_ms"] = model[arm]["ms"]
            r["fps"] = model[arm]["fps"]
            r["params_M"] = model[arm].get("params_M")
        rows.append(r)

    key = "frame_ms" if any("frame_ms" in r for r in rows) else "site_ms"
    base = next((r[key] for r in rows if r["arm"] == "dense" and key in r), None)
    print(f"\naccuracy at equal measured ms ({'whole model, one frame' if key == 'frame_ms' else 'one AAT site'}, fp16, Orin Nano)\n")
    print(f"{'arm':>44} {'act/tok':>8} {'site ms':>8} {'fp16 frm':>9} {'int8 TRT':>9} "
          f"{'int8 hyb':>9} {'best FPS':>9} {'whole':>7} {'near':>7} {'unseen':>7}")
    for r in rows:
        ms = r.get(key)
        print(f"{r['label']:>44} {r['active_M']:>7.1f}M "
              f"{(ms if ms else float('nan')):>8.3f} "
              f"{r.get('frame_ms_fp16_derived', float('nan')):>9.1f} "
              f"{r.get('frame_ms_int8_trt', float('nan')):>9.1f} "
              f"{r.get('frame_ms_int8_hybrid', float('nan')):>9.1f} "
              f"{r.get('fps_int8_best', float('nan')):>9.1f} "
              f"{r['whole']:>7.4f} {r['near']:>7.4f} {r['unseen']:>7.4f}"
              + ("  *acc inf" if r["accuracy_inferred"] else ""))
    if trt:
        for k_, v in sorted(trt.items()):
            print(f"\nTRT dispatch, K={k_}: one-hot static top-1 at cap factor {v['cap_factor']} "
                  f"({v['slots']} slots, drop-free) costs {v['moe_int8_ms']:.3f} ms against a dense "
                  f"FFN engine's {v['dense_int8_ms']:.3f} ms = {v['ratio']:.2f}x, INT8, standalone.")
    print("\nFrame columns are DERIVED, not measured end to end.\n"
          "  fp16 frm  delta on the measured 169.49 ms fp16 frame.\n"
          "  int8 TRT  mixture stays in the engine; the deployed 2.1 ms FFN share scaled by the\n"
          "            standalone TRT ratio above. Ratio, never absolutes: 8 standalone dense FFN\n"
          "            engines are 8.1 ms against the fused engine's whole 2.1 ms share.\n"
          "  int8 hyb  mixture runs as fp16 Triton OUTSIDE the engine, so its PyTorch wall time is\n"
          "            substituted directly for the 2.1 ms.")

    # Verdict, strictly: a mixture is dominated when some dense arm is cheaper AND no worse on
    # ALL THREE metrics, with no tolerance applied. Deliberately not "within scatter": the only
    # measured scatters are near 0.0087 and unseen 0.0025 (log.md), there is none for whole, and
    # borrowing near's would hand the dense arms a 0.0087 margin they have not earned. Where a
    # mixture survives, the margin it survives by and what it pays in ms are printed instead.
    dense_rows = [d for d in rows if not d["arm"].startswith("U") and key in d]
    verdict = {}
    for r in rows:
        if not r["arm"].startswith("U") or key not in r:
            continue
        beaten = [d for d in dense_rows if d[key] < r[key]
                  and all(d[m] <= r[m] for m in ("whole", "near", "unseen"))]
        cheapest = min((d for d in dense_rows if d[key] < r[key]), key=lambda d: d[key],
                       default=None)
        v = {"dominated_by": [d["label"] for d in beaten]}
        print(f"\n{r['label']} @ {r[key]:.3f} ms")
        if beaten:
            print(f"  DOMINATED by {', '.join(d['label'] for d in beaten)} "
                  f"(cheaper and no worse on whole, near and unseen)")
        else:
            print("  not strictly dominated. What it wins, and what it costs:")
            for d in dense_rows:
                if d[key] >= r[key]:
                    continue
                wins = {m: d[m] - r[m] for m in ("whole", "near", "unseen") if r[m] < d[m]}
                loses = {m: r[m] - d[m] for m in ("whole", "near", "unseen") if r[m] > d[m]}
                v[d["arm"]] = {"ms_premium": r[key] / d[key], "better_by": wins,
                               "worse_by": loses}
                print(f"    vs {d['label']:<44} {r[key]/d[key]:.2f}x the ms | "
                      + " | ".join(f"{m} {-x:+.4f}" for m, x in wins.items())
                      + (" | " if wins and loses else "")
                      + " | ".join(f"{m} {x:+.4f}" for m, x in loses.items()))
        verdict[r["arm"]] = v
    dominated = {k: v["dominated_by"] for k, v in verdict.items()}

    json.dump({"metric_key": key, "scatter_near": SCATTER, "rows": rows,
               "dominated": dominated, "verdict": verdict}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")

    try:
        make_fig(rows, key, args.fig)
        print(f"wrote {args.fig}")
    except ImportError:
        print("matplotlib unavailable, skipped the figure")


def make_fig(rows, key, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, metric, title in zip(axes, ("near", "whole"),
                                 ("near-field 0.2-0.5 m rel", "whole-frame absrel")):
        for r in rows:
            if key not in r:
                continue
            mix = r["arm"].startswith("U")
            ax.scatter(r[key], r[metric], s=46, zorder=3,
                       color="#c2453a" if mix else "#2b6cb0",
                       marker="o" if mix else "s")
            ax.annotate(r["label"].split(" (")[0], (r[key], r[metric]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                        color="#c2453a" if mix else "#2b6cb0")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("measured ms on Orin Nano, fp16")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("error (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

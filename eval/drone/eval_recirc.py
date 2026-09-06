#!/usr/bin/env python3
"""RECIRC-00 go/no-go: does cross-frame recirculation help a FROZEN MERLIN student?

Runs three arms over cached sequential trajectories (eval/drone/gen_traj.py):

  base     unmodified student
  prev     inject frame t-1's z_s at layer d (the mechanism)
  shuffle  inject a foreign frame's z_s, resampled every frame (the control)

`shuffle` is not optional. If `prev` beats `base` and `shuffle` also beats `base`, the gain is
generic activation perturbation, not temporal state, and the result is dead.

Metrics follow the repo rule: bin by TRUE range, never report the frame mean alone. Scale is
fitted per-arm per-sequence (median GT/pred) so no arm can win on scale alone; scale SPREAD
across the sequence is reported separately as the temporal-consistency metric.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model"))

from eval.drone.perception import Perception  # noqa: E402
from recirc.recirc import Recirculator, tap_list  # noqa: E402

BINS = [(0.2, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 2.5)]


def seq_metrics(gts, preds):
    """One sequence -> dict. `preds` are up-to-scale; scale is fitted here."""
    ratios = []
    for gt, pr in zip(gts, preds):
        v = np.isfinite(gt) & (gt > 0.1) & (gt < 10) & (pr > 0.05)
        if v.sum() > 200:
            ratios.append(float(np.median(gt[v] / pr[v])))
    if len(ratios) < 4:
        return None
    scale = float(np.median(ratios))
    out = {"scale": scale, "scale_spread": float(np.std(ratios)), "n_frames": len(ratios)}
    whole, per_bin = [], {f"{a}-{b}": ([], []) for a, b in BINS}
    for gt, pr in zip(gts, preds):
        p = pr * scale
        v = np.isfinite(gt) & (gt > 0.1) & (gt < 10) & (p > 0.05)
        if v.sum() < 200:
            continue
        whole.append(float(np.mean(np.abs(p[v] - gt[v]) / gt[v])))
        for a, b in BINS:
            m = v & (gt >= a) & (gt < b)
            if m.sum() > 50:
                rel, bias = per_bin[f"{a}-{b}"]
                rel.append(float(np.mean(np.abs(p[m] - gt[m]) / gt[m])))
                bias.append(float(np.mean(p[m] - gt[m])))
    out["whole_absrel"] = float(np.mean(whole)) if whole else float("nan")
    for k, (rel, bias) in per_bin.items():
        out[f"rel_{k}"] = float(np.mean(rel)) if rel else float("nan")
        out[f"bias_{k}"] = float(np.mean(bias)) if bias else float("nan")
    return out


def run_arm(perc, seqs, res, rec=None, pool=None, rng=None, keep_preds=False):
    """Returns per-sequence metric dicts, and the raw predictions when asked."""
    rows, kept = [], []
    for si, (tag, rgb, dep) in enumerate(seqs):
        if rec is not None:
            rec.reset()
        preds = []
        for i in range(len(rgb)):
            if rec is not None and rec.mode == "shuffle":
                # frame 0 is the warm-up in `prev` mode; keep the arms comparable by not
                # injecting into it here either.
                rec.set_state(pool[rng.integers(len(pool))] if (pool and i > 0) else None)
            preds.append(perc._raw_depth(rgb[i], res))
        if keep_preds:
            kept.append(preds)
        m = seq_metrics([d.astype(np.float32) for d in dep], preds)
        if m:
            m["seq"] = tag
            rows.append(m)
    return (rows, kept) if keep_preds else rows


def ema_preds(preds, w):
    """Output-space temporal smoothing of the depth predictions. Post-processing, so it costs
    no extra forwards and can be stacked on top of any arm."""
    acc, out = None, []
    for p in preds:
        acc = p if acc is None else (1.0 - w) * p + w * acc
        out.append(acc)
    return out


def rows_from(seqs, preds_list):
    rows = []
    for (tag, rgb, dep), preds in zip(seqs, preds_list):
        m = seq_metrics([d.astype(np.float32) for d in dep], preds)
        if m:
            m["seq"] = tag
            rows.append(m)
    return rows


def agg(rows, keys):
    out = {"n_seq": len(rows)}
    for k in keys:
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        out[k] = float(np.mean(v)) if v else float("nan")
        out[k + "_sem"] = float(np.std(v) / max(1, np.sqrt(len(v)))) if v else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seqs", required=True, help="dir of npz from gen_traj.py")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--source", default="aat.5")
    ap.add_argument("--dest", default="enc.9")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.25])
    ap.add_argument("--norm", default="token", choices=["token", "global", "none"])
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1,
                    help="subsample frames to widen the camera baseline between consecutive "
                         "model invocations; the redundancy of frame t-1 is the thing being "
                         "varied, and striding holds scene content fixed while doing it")
    ap.add_argument("--pool", type=int, default=48, help="foreign activations for shuffle")
    ap.add_argument("--ema", nargs="+", type=float, default=[],
                    help="output-space depth EMA weights, scored as a control. If a trivial "
                         "temporal smoothing of the OUTPUT matches recirculation, then "
                         "recirculation is just smoothing and buys nothing in feature space.")
    ap.add_argument("--list_taps", action="store_true")
    ap.add_argument("--center", type=float, default=0.0,
                    help="EMA decay for mean-centred injection; 0 = raw Eq. 1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    perc = Perception(args.ckpt, device=args.device)
    labels = [l for l, _ in tap_list(perc.model)]
    print("TAPS:", labels, flush=True)
    if args.list_taps:
        return

    files = sorted(glob.glob(os.path.join(args.seqs, "*.npz")))
    if args.max_seqs:
        files = files[: args.max_seqs]
    seqs = []
    for f in files:
        z = np.load(f)
        seqs.append((os.path.basename(f)[:-4], z["rgb"][::args.stride], z["dep"][::args.stride]))
    print(f"{len(seqs)} sequences, {sum(len(s[1]) for s in seqs)} frames", flush=True)

    keys = (["whole_absrel", "scale_spread"]
            + [f"rel_{a}-{b}" for a, b in BINS] + [f"bias_{a}-{b}" for a, b in BINS])
    results = {}

    per_seq = {}
    base_rows, base_preds = run_arm(perc, seqs, args.res, keep_preds=True)
    results["base"] = agg(base_rows, keys)
    per_seq["base"] = base_rows

    # Output-space EMA over the SAME baseline predictions: no extra forwards, and the honest
    # comparison for any claim that recirculation improves temporal behaviour.
    for w in args.ema:
        tag = f"ema_w{w}"
        rows = rows_from(seqs, [ema_preds(p, w) for p in base_preds])
        results[tag] = agg(rows, keys)
        per_seq[tag] = rows
        print(f"{tag}: whole {results[tag]['whole_absrel']:.4f} "
              f"rel58 {results[tag]['rel_0.5-0.8']:.4f} "
              f"spread {results[tag]['scale_spread']:.4f}", flush=True)
    print("base:", json.dumps({k: round(v, 4) for k, v in results["base"].items()
                               if not k.endswith("_sem")}), flush=True)

    # Foreign-activation pool for the shuffle control: z_s captured from frames spread across
    # the dataset, so the injected state is as time-varying as the real one but temporally wrong.
    rng = np.random.default_rng(0)
    pool = []
    probe = Recirculator(perc.model, args.source, args.dest, alpha=0.0,
                         norm=args.norm, mode="prev", center=args.center)
    for si in range(0, len(seqs), max(1, len(seqs) // 8)):
        rgb = seqs[si][1]
        for i in range(0, len(rgb), max(1, len(rgb) // 6)):
            perc._raw_depth(rgb[i], args.res)
            if probe.last is not None:
                pool.append(probe.last.clone())
            if len(pool) >= args.pool:
                break
        if len(pool) >= args.pool:
            break
    probe.remove()
    print(f"shuffle pool: {len(pool)} activations", flush=True)

    for alpha in args.alphas:
        for mode in ("prev", "shuffle"):
            rec = Recirculator(perc.model, args.source, args.dest, alpha=alpha,
                               norm=args.norm, mode=mode, center=args.center)
            rows, arm_preds = run_arm(perc, seqs, args.res, rec=rec, pool=pool, rng=rng,
                                      keep_preds=True)
            rec.remove()
            tag = f"{mode}_a{alpha}"
            results[tag] = agg(rows, keys)
            per_seq[tag] = rows
            # Guard against silently-inert hooks and against a block being invoked more than
            # once per frame: expect one injection per frame after each sequence's warm-up.
            expect = sum(len(s[1]) for s in seqs) - len(seqs)
            results[tag]["n_injected"] = rec.n_injected
            if rec.n_injected != expect:
                print(f"  WARNING {tag}: {rec.n_injected} injections, expected {expect}",
                      flush=True)
            # Stack output-EMA on top of recirculation. If the two are orthogonal -- feature
            # mixing helping the mid field, output smoothing helping frame-to-frame variance --
            # the combination should beat both. Free: EMA is post-processing.
            for w in args.ema:
                ctag = f"{tag}+ema{w}"
                crows = rows_from(seqs, [ema_preds(p, w) for p in arm_preds])
                results[ctag] = agg(crows, keys)
                per_seq[ctag] = crows
                print(f"{ctag}: whole {results[ctag]['whole_absrel']:.4f} "
                      f"rel58 {results[ctag]['rel_0.5-0.8']:.4f} "
                      f"spread {results[ctag]['scale_spread']:.4f}", flush=True)
            d_near = results[tag]["rel_0.2-0.5"] - results["base"]["rel_0.2-0.5"]
            d_whole = results[tag]["whole_absrel"] - results["base"]["whole_absrel"]
            print(f"{tag}: whole {results[tag]['whole_absrel']:.4f} ({d_whole:+.4f})  "
                  f"near_rel {results[tag]['rel_0.2-0.5']:.4f} ({d_near:+.4f})  "
                  f"near_bias {results[tag]['bias_0.2-0.5']:+.4f}  "
                  f"spread {results[tag]['scale_spread']:.4f}", flush=True)

    # Per-sequence rows, not just aggregates: every arm sees the IDENTICAL sequences, so the
    # meaningful statistic is the paired per-sequence difference. Between-scene variance in
    # near-field error is large enough (unpaired SEM ~0.04) to hide a real 0.036 effect.
    meta = {"source": args.source, "dest": args.dest, "norm": args.norm,
            "center": args.center, "stride": args.stride,
            "ckpt": args.ckpt, "n_seq": len(seqs)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "results": results, "per_seq": per_seq}, f, indent=2)
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

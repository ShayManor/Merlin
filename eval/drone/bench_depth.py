#!/usr/bin/env python3
"""Depth error binned by TRUE range, for one or more checkpoints, on a held-out frame set.

Binning by true range is the whole point. A whole-frame absrel of 0.208 hides a near field
that is 0.64 relative error, and the near field is the regime a collision happens in. Every
bin is reported so a near-field gain that was paid for out of the 1-2.5 m band -- the band
the voxel map is actually built from -- shows up instead of hiding inside the mean.

Scale is re-fitted per frame (median gt/pred), so this measures depth SHAPE and is not
confounded by the metric-scale question the IMU chain owns.
"""
import argparse
import glob
import os

import numpy as np

from eval.drone.perception import Perception

BINS = [(0.2, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 2.5), (2.5, 4.0)]


def load_pairs(root, limit=0):
    rgbs = sorted(glob.glob(os.path.join(root, "rgb_*.png")))
    if limit:
        rgbs = rgbs[::max(1, len(rgbs) // limit)][:limit]
    return [(r, r.replace("rgb_", "dep_").replace(".png", ".npy")) for r in rgbs]


def bench(ckpt, pairs, res, aat_depth=None, global_scale=False, moments=None):
    """moments: optional dict filled with per-bin (n, sum, sumsq) of the SIGNED residual.

    The per-bin median absolute error below is the right statistic for reporting depth
    quality, but an occupancy fusion needs the spread, and it needs it at the scale the
    episode actually flies at. global_scale fits one median scale over the whole frame set
    instead of per frame, which is the flown condition: nav_loop holds scale fixed for an
    episode while a per-frame refit silently removes the very error the map accumulates.
    """
    import cv2
    perc = Perception(ckpt, aat_depth=aat_depth)
    acc = {b: [] for b in BINS}
    frame_absrel = []
    gscale = None
    if global_scale:
        rs = []
        for rp, dp in pairs[::max(1, len(pairs) // 60)]:
            rgb = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
            pred = perc._raw_depth(rgb, res)
            gt = np.load(dp).astype(np.float32)
            if gt.shape != pred.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            v = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
            if v.sum() > 200:
                rs.append(float(np.median(gt[v] / pred[v])))
        gscale = float(np.median(rs)) if rs else 1.0
    for rp, dp in pairs:
        rgb = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
        pred = perc._raw_depth(rgb, res)
        gt = np.load(dp).astype(np.float32)
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        v = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
        if v.sum() < 200:
            continue
        pa = pred * (gscale if gscale is not None else float(np.median(gt[v] / pred[v])))
        frame_absrel.append(float(np.median(np.abs(pa[v] - gt[v]) / gt[v])))
        for lo, hi in BINS:
            b = v & (gt >= lo) & (gt < hi)
            if b.sum() > 100:
                acc[(lo, hi)].append((float(np.median(np.abs(pa[b] - gt[b]))),
                                      float(np.median(np.abs(pa[b] - gt[b]) / gt[b])),
                                      float(np.median((pa - gt)[b]))))
            if moments is not None and b.any():
                r = (pa - gt)[b].astype(np.float64)
                m = moments.setdefault((lo, hi), [0, 0.0, 0.0])
                m[0] += int(r.size); m[1] += float(r.sum()); m[2] += float((r * r).sum())
    return acc, frame_absrel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path pairs, or paths")
    ap.add_argument("--val", required=True)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--aat_depth", type=int, default=None)
    ap.add_argument("--global_scale", action="store_true",
                    help="fit one scale over the whole set instead of per frame (the flown "
                         "condition, since an episode holds scale fixed)")
    ap.add_argument("--sigma_out", default=None,
                    help="write the per-bin signed-residual mean and std as JSON, for use as "
                         "the inverse sensor model in eval/drone/maps.py")
    args = ap.parse_args()

    pairs = load_pairs(args.val, args.limit)
    print(f"{len(pairs)} frames from {args.val}", flush=True)
    results = {}
    moments = {} if args.sigma_out else None
    for spec in args.ckpts:
        name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        results[name] = bench(path, pairs, args.res, args.aat_depth,
                              global_scale=args.global_scale, moments=moments)
        print(f"  benched {name}", flush=True)
    if args.sigma_out:
        import json
        tab = []
        for (lo, hi), (n, s, ss) in sorted(moments.items()):
            mu = s / max(n, 1)
            tab.append({"lo": lo, "hi": hi, "n": n, "bias": round(mu, 4),
                        "sigma": round(float(np.sqrt(max(ss / max(n, 1) - mu * mu, 0.0))), 4)})
        json.dump(tab, open(args.sigma_out, "w"), indent=1)
        print("\nsensor model written to", args.sigma_out)
        for r in tab:
            print("  %.1f-%.1f m  bias %+.3f  sigma %.3f  (n=%d)"
                  % (r["lo"], r["hi"], r["bias"], r["sigma"], r["n"]))

    print(f"\n{'true range':>12s} " + " ".join(f"{n:>26s}" for n in results))
    print(f"{'':>12s} " + " ".join(f"{'abs':>8s}{'rel':>9s}{'bias':>9s}" for _ in results))
    for b in BINS:
        row = f"{b[0]:5.1f}-{b[1]:4.1f} "
        for name in results:
            a = results[name][0][b]
            if not a:
                row += f"{'-':>26s}"
                continue
            m = np.median(np.array(a), 0)
            row += f"{m[0]:8.3f}{m[1]:9.3f}{m[2]:+9.3f}"
        print(row)
    print("\nwhole-frame absrel (the statistic that hides the above): "
          + "  ".join(f"{n} {np.median(results[n][1]):.3f}" for n in results))


if __name__ == "__main__":
    main()

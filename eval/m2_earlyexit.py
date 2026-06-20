#!/usr/bin/env python3
"""M2 early-exit operating points: accuracy + latency at AAT compute depth K.

Unlike resolution scaling (shallow: coarse=smoother=lower error), early-exit varies
COMPUTE at FIXED resolution -> no smoothness confound -> a genuine accuracy/latency
frontier. The student is deep-supervision trained at depths {6,8}. This measures
scale-aligned absrel vs GT and the per-frame GPU latency at each K, so the M2
deadline controller has real operating points to choose between.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "model", "backbones"))
sys.path.insert(0, os.path.join(HERE, "geometry"))
from data import tum_pairs, load_gt_depth   # noqa: E402
from student import build_student            # noqa: E402


def sa_absrel(gt, pred):
    v = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
    if v.sum() < 200:
        return np.nan
    r = np.median(gt[v] / pred[v])
    return float(np.mean(np.abs(gt[v] - r * pred[v]) / gt[v]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/student_earlyexit.pt")
    ap.add_argument("--seq", default="/workspace/data/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--depths", default="6,8")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    dev = "cuda"
    from mapanything.utils.image import load_images
    m = build_student(size="base", aat_depth=8, device="cpu")
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"])
    m = m.to(dev).eval()

    pairs = tum_pairs(args.seq)
    step = max(1, len(pairs) // args.n); pairs = pairs[::step][:args.n]
    s = (args.res // 14) * 14
    depths = [int(k) for k in args.depths.split(",")]

    out = {}
    for K in depths:
        m.info_sharing.depth = K
        accs = []
        # warmup + latency
        v0 = load_images([pairs[0][0]], resize_mode="square", size=s); v0[0]["img"] = v0[0]["img"].to(dev)
        for _ in range(3):
            with torch.no_grad():
                m(v0, memory_efficient_inference=True, minibatch_size=1)
        torch.cuda.synchronize(); t = time.time(); ITERS = 15
        for _ in range(ITERS):
            with torch.no_grad():
                m(v0, memory_efficient_inference=True, minibatch_size=1)
        torch.cuda.synchronize(); lat = (time.time() - t) / ITERS * 1000
        for rp, dp in pairs:
            v = load_images([rp], resize_mode="square", size=s); v[0]["img"] = v[0]["img"].to(dev)
            with torch.no_grad():
                sd = m(v, memory_efficient_inference=True, minibatch_size=1)[0]["pts3d_cam"][0, ..., 2].float().cpu().numpy()
            gt = load_gt_depth(dp, sd.shape[0], sd.shape[1])
            accs.append(sa_absrel(gt, sd))
        out[K] = {"absrel": round(float(np.nanmean(accs)), 4), "latency_ms_a40": round(lat, 1)}
        print(f"K={K}: absrel={out[K]['absrel']} latency(A40)={out[K]['latency_ms_a40']}ms", flush=True)
    # frontier check: deeper should be MORE accurate (and slower)
    ks = sorted(depths)
    print(f"frontier: K{ks[0]} absrel {out[ks[0]]['absrel']} -> K{ks[-1]} absrel {out[ks[-1]]['absrel']} "
          f"({'STEEP/clean' if out[ks[-1]]['absrel'] < out[ks[0]]['absrel'] else 'flat/inverted'})", flush=True)
    import json
    json.dump(out, open(args.ckpt.replace(".pt", "_ee.json"), "w"), indent=2)
    print("=== EARLYEXIT_EVAL_DONE ===", flush=True)


if __name__ == "__main__":
    main()

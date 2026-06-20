#!/usr/bin/env python3
"""Robust M1 evaluation: compare distilled checkpoints on navigation-relevant
metrics over a large held-out set (the in-training 24-frame eval is too noisy).

For each checkpoint, run the student over N cached held-out frames and report,
averaged: fidelity-vs-teacher, near/far-field abs_rel, per-column obstacle-range
MAE, obstacle IoU, and (if GT available) absolute depth/scale. The baseline vs
nav1 delta is the M1 misalignment result.

Eval is teacher-free: cached teacher depth_z is the reference for fidelity/nav,
TUM depth PNG is the GT. Runs on the pod (A40) after the runs finish.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "model", "backbones"))
sys.path.insert(0, os.path.join(HERE, "geometry"))
sys.path.insert(0, HERE)
from student import build_student            # noqa: E402
from data import load_gt_depth               # noqa: E402
from metrics import depth_metrics, scale_error  # noqa: E402
from nav_metrics import nav_metrics           # noqa: E402


def load_frames(cache_dir, name_filter, n):
    idx = json.load(open(os.path.join(cache_dir, "index.json")))
    files = [v["file"] for k, v in sorted(idx.items(), key=lambda x: int(x[0]))
             if name_filter in v["rgb"]]
    # even stride across the sequence for coverage
    if len(files) > n:
        step = len(files) / n
        files = [files[int(i * step)] for i in range(n)]
    return [torch.load(f, map_location="cpu", weights_only=False) for f in files]


@torch.no_grad()
def eval_ckpt(ckpt_path, frames, dev):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_student(size=ck.get("size", "base"), aat_depth=ck.get("aat_depth", 8), device="cpu")
    model.load_state_dict(ck["state_dict"]); model = model.to(dev).eval()
    agg = {"fid": [], "gt_absrel": [], "gt_scale": [],
           "near_absrel": [], "far_absrel": [], "col_range_mae": [], "obstacle_iou": []}
    for b in frames:
        img = b["img"].float().unsqueeze(0).to(dev)
        H = W = img.shape[-1]
        view = [dict(img=img, true_shape=np.array([[H, W]], np.int32), idx=0,
                     instance=["0"], data_norm_type=["dinov2"])]
        pr = model(view, memory_efficient_inference=True, minibatch_size=1)[0]
        sz = pr["pts3d_cam"][0, ..., 2].float().cpu().numpy()
        tz = b["depth_z"].float().numpy()
        v = np.isfinite(sz) & np.isfinite(tz) & (tz > 0.1) & (sz > 0.05)
        agg["fid"].append(float(np.mean(np.abs(tz[v] - sz[v]) / tz[v])) if v.sum() else np.nan)
        nm = nav_metrics(tz, sz)
        for k in ("near_absrel", "far_absrel", "col_range_mae", "obstacle_iou"):
            if np.isfinite(nm[k]):
                agg[k].append(nm[k])
        gt = load_gt_depth(b["depth_path"], H, W)
        agg["gt_absrel"].append(depth_metrics(gt, sz)["abs_rel"]); agg["gt_scale"].append(scale_error(gt, sz))
    del model; torch.cuda.empty_cache()
    return {k: round(float(np.nanmean(v)), 4) for k, v in agg.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/root/cache/sz378")
    ap.add_argument("--ckpts", nargs="+", required=True, help="tag=path ...")
    ap.add_argument("--seqs", nargs="+", default=["freiburg1_room", "freiburg3_long_office_household"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="/workspace/ckpt/m1_compare.json")
    args = ap.parse_args()
    dev = "cuda"
    runs = dict(c.split("=", 1) for c in args.ckpts)

    report = {}
    for seq in args.seqs:
        frames = load_frames(args.cache, seq, args.n)
        print(f"\n=== {seq} ({len(frames)} frames) ===", flush=True)
        report[seq] = {}
        for tag, path in runs.items():
            m = eval_ckpt(path, frames, dev)
            report[seq][tag] = m
            print(f"  {tag:10s} {m}", flush=True)
        b, n1 = report[seq].get("baseline"), report[seq].get("nav1")
        if b and n1:
            print(f"  M1 delta (nav1-baseline): near={round(n1['near_absrel']-b['near_absrel'],4)} "
                  f"far={round(n1['far_absrel']-b['far_absrel'],4)} "
                  f"col_mae={round(n1['col_range_mae']-b['col_range_mae'],4)} "
                  f"iou={round(n1['obstacle_iou']-b['obstacle_iou'],4)}", flush=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

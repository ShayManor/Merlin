#!/usr/bin/env python3
"""Probe in-sim perception quality: near/far depth abs_rel of each student vs the sim GT
depth, on photorealistic Habitat frames. Answers whether M1's offline near-field gain
survives the Habitat domain shift -- if baseline vs nav1 differ here the way they do
offline, the closed-loop nav-null is a real transfer gap (gain present, doesn't help nav),
not a domain-shift confound that erased the gain.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "backbones"))
from sim_client import SimClient            # noqa: E402
from run_experiments import spawn_server    # noqa: E402


def absrel(gt, pred, lo, hi):
    import cv2
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    v = np.isfinite(gt) & np.isfinite(pred) & (gt > 0.1) & (gt < 10) & (pred > 0.05)
    if v.sum() < 200:
        return np.nan
    s = np.median(gt[v] / pred[v])
    pred = pred * s
    band = v & (gt >= lo) & (gt < hi)
    if band.sum() < 100:
        return np.nan
    return float(np.mean(np.abs(gt[band] - pred[band]) / gt[band]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=40, help="poses per scene")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--env-name", default="/root/conda_envs/habenv")
    ap.add_argument("--conda-sh", default="/workspace/miniconda3/etc/profile.d/conda.sh")
    ap.add_argument("--port", type=int, default=5750)
    ap.add_argument("--out", default="/workspace/ckpt/sim_probe.json")
    args = ap.parse_args()
    ckpts = dict(c.split("=", 1) for c in args.ckpts)

    from perception import Perception
    server = spawn_server(args.port, args.conda_sh, args.env_name)
    client = SimClient(args.port)
    frames = []  # (rgb, gt)
    try:
        for si, scene in enumerate(args.scenes):
            for k in range(args.frames):
                ep = client.sample(scene, 1.0, 8.0, 7000 + 1000 * si + k, args.dataset)
                if not ep:
                    continue
                obs = client.reset(scene, ep["start"], ep["yaw"], ep["goal"], args.dataset)
                frames.append((obs["rgb"].copy(), obs["depth"].copy()))
            print(f"[{scene}] {len(frames)} frames", flush=True)
    finally:
        client.close(); server.terminate()

    report = {}
    for tag, path in ckpts.items():
        p = Perception(path, device="cuda")
        near, far = [], []
        for rgb, gt in frames:
            p.scale = 1.0
            pred = p._raw_depth(rgb, args.res)
            near.append(absrel(gt, pred, 0.1, 2.0))
            far.append(absrel(gt, pred, 2.0, 10.0))
        report[tag] = {"near_absrel": round(float(np.nanmean(near)), 4),
                       "far_absrel": round(float(np.nanmean(far)), 4),
                       "n": int(np.sum(np.isfinite(near)))}
        print(tag, report[tag], flush=True)
        del p
        import torch; torch.cuda.empty_cache()
    json.dump(report, open(args.out, "w"), indent=2)
    print("=== PROBE_DONE ===", flush=True)


if __name__ == "__main__":
    main()

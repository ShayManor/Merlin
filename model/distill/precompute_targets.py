#!/usr/bin/env python3
"""Precompute MapAnything teacher targets for the whole frame set, once.

Online distillation re-runs the fp32 ViT-G teacher every step on a fixed frame
set (~10x redundant). Instead, cache per-frame {preprocessed img, depth_along_ray,
ray_directions, metric_scaling_factor, teacher depth_z} as fp16. Training then
runs student-only and reads targets from RAM -- ~4x faster per step and reused
across every distillation variant (baseline, M1 nav, M2 exits, Scal3R).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import gather_frames  # noqa: E402
from mapanything.models import MapAnything  # noqa: E402
from mapanything.utils.image import load_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-roots", nargs="+", default=["/workspace/data/tum"])
    ap.add_argument("--size", type=int, default=378)
    ap.add_argument("--out", default="/workspace/cache")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    cache = os.path.join(args.out, f"sz{args.size}")
    os.makedirs(cache, exist_ok=True)
    dev = "cuda"

    frames = gather_frames(args.data_roots, stride=args.stride, with_depth=True)
    print(f"[precompute] {len(frames)} frames -> {cache}", flush=True)
    index = {}

    print("loading teacher fp32 ...", flush=True)
    teacher = MapAnything.from_pretrained("facebook/map-anything-apache").to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    done = 0
    for start in range(0, len(frames), args.batch):
        chunk = frames[start:start + args.batch]
        out_paths = [os.path.join(cache, f"{start+i:05d}.pt") for i in range(len(chunk))]
        if all(os.path.exists(o) for o in out_paths):
            done += len(chunk)
            for i, (rp, dp) in enumerate(chunk):
                index[start + i] = {"rgb": rp, "depth": dp, "file": out_paths[i]}
            continue
        imgs, shapes = [], []
        for rp, _ in chunk:
            v = load_images([rp], resize_mode="square", size=args.size)[0]
            imgs.append(v["img"]); shapes.append(v["true_shape"])
        img = torch.cat(imgs, 0).to(dev)
        B = img.shape[0]
        view = [dict(img=img, true_shape=np.concatenate(shapes, 0), idx=0,
                     instance=[str(i) for i in range(B)], data_norm_type=["dinov2"] * B)]
        with torch.inference_mode():
            pr = teacher(view, memory_efficient_inference=True, minibatch_size=B)[0]
        dar = pr["depth_along_ray"].float().cpu()          # (B,H,W,1)
        ray = pr["ray_directions"].float().cpu()           # (B,H,W,3)
        dz = pr["pts3d_cam"][..., 2].float().cpu()         # (B,H,W) teacher metric z
        msf = pr.get("metric_scaling_factor")
        msf = msf.float().cpu() if msf is not None else None
        for i, (rp, dp) in enumerate(chunk):
            torch.save({"img": img[i].half().cpu(),
                        "depth": dar[i].half(), "ray": ray[i].half(),
                        "depth_z": dz[i].half(),
                        "scale": (msf[i].half() if msf is not None else None),
                        "rgb": rp, "depth_path": dp}, out_paths[i])
            index[start + i] = {"rgb": rp, "depth": dp, "file": out_paths[i]}
        done += len(chunk)
        if start % (args.batch * 20) == 0:
            print(f"  {done}/{len(frames)}", flush=True)
    with open(os.path.join(cache, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"=== PRECOMPUTE_DONE {done} frames at sz{args.size} ===", flush=True)


if __name__ == "__main__":
    main()

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


def robust_save(payload, path, tries=4):
    """Atomic, retrying save -- the network/overlay FS occasionally short-writes
    torch.save's zip stream (inline_container pos mismatch). Write to a tmp file
    then os.replace; retry on failure."""
    import time as _t
    for k in range(tries):
        tmp = f"{path}.tmp{k}"
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
            return True
        except Exception as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            if k == tries - 1:
                print(f"  [save-fail] {os.path.basename(path)}: {e}", flush=True)
                return False
            _t.sleep(0.5)


class FrameSet(torch.utils.data.Dataset):
    """Parallel image decode/resize (the precompute bottleneck) via DataLoader workers."""
    def __init__(self, frames, size):
        self.frames = frames; self.size = size

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        rp, dp = self.frames[i]
        v = load_images([rp], resize_mode="square", size=self.size)[0]
        return i, v["img"][0], v["true_shape"][0], rp, dp


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
    index = {str(i): {"rgb": rp, "depth": dp, "file": os.path.join(cache, f"{i:05d}.pt")}
             for i, (rp, dp) in enumerate(frames)}

    print("loading teacher fp32 ...", flush=True)
    teacher = MapAnything.from_pretrained("facebook/map-anything-apache").to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    loader = torch.utils.data.DataLoader(
        FrameSet(frames, args.size), batch_size=args.batch, num_workers=8,
        shuffle=False, drop_last=False, prefetch_factor=2, persistent_workers=False)

    done = 0
    for idxs, imgs, shapes, rps, dps in loader:
        out_paths = [os.path.join(cache, f"{int(j):05d}.pt") for j in idxs]
        done += len(idxs)
        if all(os.path.exists(o) for o in out_paths):
            if done % (args.batch * 50) < args.batch:
                print(f"  {done}/{len(frames)} (skip)", flush=True)
            continue
        img = imgs.to(dev, non_blocking=True)  # (B,3,H,W)
        B = img.shape[0]
        view = [dict(img=img, true_shape=shapes.numpy(), idx=0,
                     instance=[str(i) for i in range(B)], data_norm_type=["dinov2"] * B)]
        with torch.inference_mode():
            pr = teacher(view, memory_efficient_inference=True, minibatch_size=B)[0]
        dar = pr["depth_along_ray"].float().half().cpu()
        ray = pr["ray_directions"].float().half().cpu()
        dz = pr["pts3d_cam"][..., 2].float().half().cpu()
        msf = pr.get("metric_scaling_factor")
        msf = msf.float().half().cpu() if msf is not None else None
        imgh = img.half().cpu()
        for i in range(B):
            if os.path.exists(out_paths[i]):
                continue
            robust_save({"img": imgh[i], "depth": dar[i], "ray": ray[i], "depth_z": dz[i],
                         "scale": (msf[i] if msf is not None else None),
                         "rgb": rps[i], "depth_path": dps[i]}, out_paths[i])
        if done % (args.batch * 20) < args.batch:
            print(f"  {done}/{len(frames)}", flush=True)
    with open(os.path.join(cache, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"=== PRECOMPUTE_DONE {done} frames at sz{args.size} ===", flush=True)


if __name__ == "__main__":
    main()

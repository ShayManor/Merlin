#!/usr/bin/env python3
"""Benchmark smart quantization of the MERLIN model on the Jetson.

For each scheme (bf16 baseline, int8, int4_int8), measures:
  - latency / FPS (steady state)
  - peak GPU memory
  - depth fidelity vs the bf16 baseline (abs_rel, delta1, scale offset)

Fidelity-vs-baseline isolates the quantization error (no GT needed). Run with a
real image; depth content matters here (unlike pure latency).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "geometry"))


def load_teacher(hf_id, device):
    from teacher_loader import load_teacher_bf16
    return load_teacher_bf16(hf_id, device=device)


def make_views(image, res, device):
    from mapanything.utils.image import load_images
    if res in (512, 518):
        views = load_images([image], resize_mode="fixed_mapping", resolution_set=res)
    else:
        s = (res // 14) * 14
        views = load_images([image], resize_mode="square", size=s)
    for v in views:
        v["img"] = v["img"].to(device)  # fp32; autocast handles bf16
    return views


def infer_depth(model, views):
    with torch.no_grad():
        out = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                          use_amp=True, amp_dtype="bf16", apply_mask=False, mask_edges=False)
    p = out[0]
    return p["depth_z"].float().squeeze().cpu().numpy(), p.get("metric_scaling_factor")


def bench_latency(model, views, warmup, iters):
    for _ in range(warmup):
        infer_depth(model, views)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for _ in range(iters):
        torch.cuda.synchronize(); t = time.time()
        infer_depth(model, views); torch.cuda.synchronize()
        lat.append((time.time() - t) * 1000)
    return np.array(lat), torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="facebook/map-anything-apache")
    ap.add_argument("--image", required=True)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--schemes", default="bf16,int8,int4_int8")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    from metrics import depth_metrics, scale_error
    from quantize import quantize_model, count_linears

    dev = "cuda"
    results = []
    ref_depth = None
    for scheme in args.schemes.split(","):
        # Reload fresh each scheme (quantization is in-place and irreversible).
        torch.cuda.empty_cache()
        model = load_teacher(args.hf_id, dev)
        if scheme != "bf16":
            print(f"[quant] applying {scheme} ... linears={count_linears(model)}")
            quantize_model(model, scheme=scheme)
        views = make_views(args.image, args.res, dev)

        depth, scale = infer_depth(model, views)
        lat, peak = bench_latency(model, views, args.warmup, args.iters)
        r = {
            "scheme": scheme, "res": args.res,
            "latency_ms": {"mean": round(float(lat.mean()), 1), "median": round(float(np.median(lat)), 1)},
            "fps": round(1000.0 / float(lat.mean()), 2),
            "peak_gpu_mem_gb": round(peak, 2),
            "metric_scale": round(float(scale.flatten()[0]), 4) if scale is not None else None,
        }
        if ref_depth is None and scheme == "bf16":
            ref_depth = depth
        if ref_depth is not None and scheme != "bf16":
            m = depth_metrics(ref_depth, depth)
            r["fidelity_vs_bf16"] = {"abs_rel": round(m["abs_rel"], 4), "delta1": round(m["delta1"], 4),
                                     "rmse": round(m["rmse"], 4), "scale_offset": round(scale_error(ref_depth, depth), 4)}
        results.append(r)
        print(json.dumps(r, indent=2))
        del model
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()

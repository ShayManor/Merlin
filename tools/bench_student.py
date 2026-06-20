#!/usr/bin/env python3
"""Benchmark a distilled MERLIN student on the Jetson across quant schemes.

For each scheme (bf16 baseline, int8, int4_int8) reports steady-state latency/FPS,
peak GPU memory, board power, and depth fidelity vs the bf16 student (isolates the
quantization error). This is the deployment table for claims C1/C5: a TRAINED
student, quantized, on the real Nano.

Usage (on the Jetson):
  python tools/bench_student.py --ckpt student_baseline.pt --image test.jpg \
      --res 378 --schemes bf16,int8,int4_int8 --report
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "model", "backbones"))
sys.path.insert(0, os.path.join(HERE, "..", "model", "quant"))
sys.path.insert(0, os.path.join(HERE, "..", "eval", "geometry"))

from profile_floor import TegrastatsSampler, make_views  # noqa: E402


def build_and_load(ckpt_path, device, dtype=torch.bfloat16):
    from student import build_student
    import teacher as T
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_student(size=ck.get("size", "base"), aat_depth=ck.get("aat_depth", 8), device="cpu")
    model.load_state_dict(ck["state_dict"])
    model = model.to(dtype).eval()
    T.add_dtype_casting_hooks(model)
    T._upcast_geometry_postproc()
    T.patch_linalg_cpu_fallback()
    return model.to(device)


def infer_depth(model, views):
    with torch.no_grad():
        out = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                          use_amp=False, apply_mask=False, mask_edges=False)[0]
    return out["depth_z"].float().squeeze().cpu().numpy(), out.get("metric_scaling_factor")


def bench(model, views, warmup, iters):
    for _ in range(warmup):
        infer_depth(model, views)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    lat = []
    with TegrastatsSampler() as power:
        for _ in range(iters):
            torch.cuda.synchronize(); t = time.time()
            infer_depth(model, views); torch.cuda.synchronize()
            lat.append((time.time() - t) * 1000)
    return np.array(lat), torch.cuda.max_memory_allocated() / 1e9, power.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--schemes", default="bf16,int8,int4_int8")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    from metrics import depth_metrics, scale_error
    from quantize import quantize_model, count_linears
    dev = "cuda"

    results = []
    ref = None
    for scheme in args.schemes.split(","):
        torch.cuda.empty_cache()
        model = build_and_load(args.ckpt, dev)
        if scheme != "bf16":
            print(f"[quant] {scheme} linears={count_linears(model)}", flush=True)
            quantize_model(model, scheme=scheme)
        views = make_views(args.image, args.res, device=dev)
        depth, scale = infer_depth(model, views)
        lat, peak, power = bench(model, views, args.warmup, args.iters)
        r = {"scheme": scheme, "res": args.res,
             "latency_ms": {"mean": round(float(lat.mean()), 1), "median": round(float(np.median(lat)), 1)},
             "fps": round(1000.0 / float(lat.mean()), 2),
             "peak_gpu_mem_gb": round(peak, 2),
             "metric_scale": round(float(scale.flatten()[0]), 4) if scale is not None else None,
             "power": power}
        if scheme == "bf16":
            ref = depth
        else:
            m = depth_metrics(ref, depth)
            r["fidelity_vs_bf16"] = {"abs_rel": round(m["abs_rel"], 4), "delta1": round(m["delta1"], 4),
                                     "scale_offset": round(scale_error(ref, depth), 4)}
        results.append(r); print(json.dumps(r, indent=2), flush=True)
        del model
    if args.json_out:
        json.dump(results, open(args.json_out, "w"), indent=2)
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()

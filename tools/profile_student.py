#!/usr/bin/env python3
"""Profile the distilled MERLIN student on the Jetson Orin Nano (PyTorch bf16).

Loads a student checkpoint (state_dict from distill_cache.py), runs monocular RGB
inference with the same bf16 recipe as the teacher, and reports steady-state
latency / FPS / peak memory / board power at a sweep of resolutions. This is the
deployment floor for the TRAINED model (Phase 0 measured the untrained arch).

Usage (on the Jetson):
  python tools/profile_student.py --ckpt student_baseline.pt --image test.jpg \
      --res 252,378,518 --report
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

from profile_floor import TegrastatsSampler, make_views  # noqa: E402


def load_student(ckpt_path, device, dtype=torch.bfloat16):
    from student import build_student
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    size = ck.get("size", "base"); depth = ck.get("aat_depth", 8)
    model = build_student(size=size, aat_depth=depth, device="cpu")
    model.load_state_dict(ck["state_dict"])
    model = model.to(dtype).eval()
    import teacher as T
    T.add_dtype_casting_hooks(model)
    T._upcast_geometry_postproc()
    T.patch_linalg_cpu_fallback()
    return model.to(device)


def run(model, views):
    with torch.no_grad():
        return model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                           use_amp=False, apply_mask=False, mask_edges=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--res", default="252,378,518")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    dev = "cuda"

    t0 = time.time()
    model = load_student(args.ckpt, dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"[load] student {n/1e6:.1f}M in {time.time()-t0:.1f}s", flush=True)

    results = []
    for res in [int(r) for r in str(args.res).split(",")]:
        views = make_views(args.image, res, device=dev)
        # bf16 input handled by hooks; keep img fp32 per teacher recipe
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup):
            run(model, views)
        torch.cuda.synchronize()
        out = run(model, views)
        dz = out[0]["depth_z"].float().flatten()
        sc = out[0].get("metric_scaling_factor")
        lat = []
        with TegrastatsSampler() as power:
            for _ in range(args.iters):
                torch.cuda.synchronize(); t = time.time()
                run(model, views); torch.cuda.synchronize()
                lat.append((time.time() - t) * 1000)
        lat = np.array(lat)
        r = {"res": res, "img_shape": tuple(views[0]["img"].shape), "params_M": round(n / 1e6, 1),
             "depth_z_m": {"min": round(float(dz.min()), 3), "median": round(float(dz.median()), 3),
                           "max": round(float(dz.max()), 3)},
             "metric_scale": round(float(sc.flatten()[0]), 4) if sc is not None else None,
             "latency_ms": {"mean": round(float(lat.mean()), 1), "median": round(float(np.median(lat)), 1),
                            "p95": round(float(np.percentile(lat, 95)), 1)},
             "fps_mean": round(1000.0 / float(lat.mean()), 2),
             "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
             "power": power.summary()}
        results.append(r)
        if args.report:
            print(json.dumps(r, indent=2), flush=True)
    if args.json_out:
        json.dump(results, open(args.json_out, "w"), indent=2)
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()

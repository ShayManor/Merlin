#!/usr/bin/env python3
"""Phase-0 floor profiler for the MERLIN model node on the Jetson Orin Nano.

Measures per-keyframe latency, FPS, peak GPU memory, and board power for
monocular RGB inference. Works with the MapAnything teacher (PyTorch) now; a
TensorRT engine backend is added once the student engine exists.

Power is read from tegrastats (VDD_IN = total board, plus SoC/GPU rails when
exposed). Latency is reported at steady state after warmup, not cold.

Usage:
  python tools/profile_floor.py --backend teacher --res 518 --iters 30 --report
  python tools/profile_floor.py --backend teacher --res 384 --iters 30 --report
"""
import argparse
import json
import re
import subprocess
import time
from contextlib import contextmanager

import numpy as np
import torch


# ---- power sampling via tegrastats -----------------------------------------
class TegrastatsSampler:
    """Background tegrastats reader. Parses power rails in mW."""

    RAIL_RE = re.compile(r"(VDD_IN|VDD_CPU_GPU_CV|VDD_SOC|POM_5V_IN|VDD_GPU_SOC)\s+(\d+)mW")

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.proc = None
        self.samples = {}

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except FileNotFoundError:
            self.proc = None
        return self

    def _drain(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            out, _ = self.proc.communicate(timeout=3)
        except Exception:
            self.proc.kill()
            out = ""
        for line in out.splitlines():
            for rail, mw in self.RAIL_RE.findall(line):
                self.samples.setdefault(rail, []).append(int(mw))

    def __exit__(self, *a):
        self._drain()

    def summary(self):
        out = {}
        for rail, vals in self.samples.items():
            if vals:
                out[rail] = {"mean_w": round(np.mean(vals) / 1000, 2),
                             "peak_w": round(np.max(vals) / 1000, 2),
                             "n": len(vals)}
        return out


@contextmanager
def cuda_timer():
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield lambda: (end.record(), torch.cuda.synchronize(), start.elapsed_time(end))[-1]
    end.record()
    torch.cuda.synchronize()


PATCH = 14


def make_views(image, res, device="cuda"):
    """Single-view input from a real image at a patch-aligned square resolution.

    The DINOv2 encoder requires both dims divisible by the patch size (14).
    We snap `res` down to a multiple of 14 and use square resize so token count
    is deterministic: (res/14)^2 tokens. Latency is content-independent.
    """
    from mapanything.utils.image import load_images
    if res in (512, 518):
        views = load_images([image], resize_mode="fixed_mapping", resolution_set=res)
    else:
        s = (res // PATCH) * PATCH
        views = load_images([image], resize_mode="square", size=s)
    for v in views:
        v["img"] = v["img"].to(device)  # fp32; autocast handles bf16
    return views


def load_teacher(hf_id, device, dtype):
    # Memory-safe streaming bf16 loader: the naive from_pretrained().to(cuda)
    # peaks ~8.5GB and OOM-crashes the 8GB Nano. See model/backbones/teacher.py.
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from teacher_loader import load_teacher_bf16
    return load_teacher_bf16(hf_id, device=device)


def run_teacher(model, views, amp_str):
    # Weights are uniform bf16, so run pure bf16 (use_amp=False); autocast assumes
    # fp32 weights and would mismatch. Input is cast to bf16 in make_views.
    with torch.no_grad():
        return model.infer(
            views, memory_efficient_inference=True, minibatch_size=1,
            use_amp=True, amp_dtype="bf16", apply_mask=False, mask_edges=False,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["teacher"], default="teacher")
    ap.add_argument("--hf-id", default="facebook/map-anything-apache")
    ap.add_argument("--image", required=True)
    ap.add_argument("--res", default="518", help="comma-separated resolutions to sweep")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda"

    print(f"[load] backend={args.backend} res={args.res} amp={args.amp_dtype}")
    t0 = time.time()
    model = load_teacher(args.hf_id, device, torch.float32)
    print(f"[load] model ready in {time.time()-t0:.1f}s")
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[load] params = {nparams/1e6:.1f}M")

    results = []
    for res in [int(r) for r in str(args.res).split(",")]:
        views = make_views(args.image, res, device=device)
        shape = tuple(views[0]["img"].shape)
        print(f"\n[res {res}] img_shape={shape}")

        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup):
            run_teacher(model, views, args.amp_dtype)
        torch.cuda.synchronize()

        # depth sanity at this resolution
        out = run_teacher(model, views, args.amp_dtype)
        dz = out[0]["depth_z"].float().flatten()
        sc = out[0].get("metric_scaling_factor", None)

        lat_ms = []
        with TegrastatsSampler() as power:
            for _ in range(args.iters):
                torch.cuda.synchronize()
                t = time.time()
                run_teacher(model, views, args.amp_dtype)
                torch.cuda.synchronize()
                lat_ms.append((time.time() - t) * 1000)

        lat = np.array(lat_ms)
        r = {
            "backend": args.backend, "res": res, "img_shape": shape, "amp": args.amp_dtype,
            "params_M": round(nparams / 1e6, 1),
            "depth_z_m": {"min": round(float(dz.min()), 3), "median": round(float(dz.median()), 3),
                          "max": round(float(dz.max()), 3)},
            "metric_scale": round(float(sc.flatten()[0]), 4) if sc is not None else None,
            "latency_ms": {"mean": round(float(lat.mean()), 1), "median": round(float(np.median(lat)), 1),
                           "p95": round(float(np.percentile(lat, 95)), 1), "min": round(float(lat.min()), 1)},
            "fps_mean": round(1000.0 / float(lat.mean()), 2),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "power": power.summary(),
        }
        results.append(r)
        if args.report:
            print(json.dumps(r, indent=2))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {args.json_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Smoke test: load the MapAnything teacher and run one mono inference.

Loads in bf16 (converted on CPU before moving to GPU) so the 8GB Jetson never
holds the 4.23GB fp32 copy on-device. Reports memory at each stage, sane metric
depth, and a first latency number.
"""
import argparse
import time

import torch

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def gb(x):
    return round(x / 1e9, 2)


def cuda_mem():
    return f"alloc={gb(torch.cuda.memory_allocated())} reserved={gb(torch.cuda.memory_reserved())} peak={gb(torch.cuda.max_memory_allocated())}"


def make_views(image, res, device):
    if res in (512, 518):
        views = load_images([image], resize_mode="fixed_mapping", resolution_set=res)
    else:
        views = load_images([image], resize_mode="longest_side", size=res)
    for v in views:
        v["img"] = v["img"].to(device)
    return views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="facebook/map-anything-apache")
    ap.add_argument("--image", required=True)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--amp", default="bf16")
    args = ap.parse_args()

    dev = "cuda"
    t0 = time.time()
    # Load to CPU (fp32), convert to bf16 on CPU, THEN move to GPU. The GPU never
    # holds the full fp32 copy.
    model = MapAnything.from_pretrained(args.hf_id)
    model = model.to(torch.bfloat16).eval()
    print(f"[load] cpu bf16 in {time.time()-t0:.1f}s  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    model = model.to(dev)
    torch.cuda.synchronize()
    print(f"[load] on gpu  mem: {cuda_mem()}")

    views = make_views(args.image, args.res, dev)
    print(f"[input] res={args.res} img_shape={tuple(views[0]['img'].shape)}")

    def run():
        with torch.no_grad():
            return model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                               use_amp=True, amp_dtype=args.amp, apply_mask=False, mask_edges=False)

    torch.cuda.reset_peak_memory_stats()
    out = run()
    pred = out[0]
    print(f"[output] keys={sorted(pred.keys())}")
    dz = pred["depth_z"].float().flatten()
    print(f"[depth_z m] min={dz.min():.3f} median={dz.median():.3f} max={dz.max():.3f}")
    scale = pred.get("metric_scaling_factor", None)
    if scale is not None:
        print(f"[metric_scale] {scale.flatten().tolist()}")
    print(f"[mem after infer] {cuda_mem()}")

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    import statistics as st
    lat = []
    for _ in range(args.iters):
        torch.cuda.synchronize(); t = time.time()
        run(); torch.cuda.synchronize()
        lat.append((time.time() - t) * 1000)
    print(f"[latency ms] mean={st.mean(lat):.1f} median={st.median(lat):.1f} min={min(lat):.1f}")
    print(f"[fps] {1000/st.mean(lat):.2f}")
    print(f"[peak_gpu_mem GB] {gb(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Where is the Jetson's actual capacity ceiling for the student?

The deployed 230M student runs at ~40 ms and ~1.2 GB, which is nowhere near the 6.5 GB the
model node is allowed. So the binding constraint is latency, not memory, and the interesting
question for a mixture is not "does it fit" but "at a fixed latency budget, does capacity buy
more accuracy as dense width/depth or as conditional experts".

This measures the frontier: for each student size, params / latency / peak memory at the
deployment resolution. N is then the largest config that still clears the C5 floor (>=5 FPS,
200 ms) and, more usefully, the 20 FPS point the INT8 engine currently holds.

fp16 torch here, not the TRT engine, so absolute numbers run high; the SHAPE of the frontier
and the ratios between configs are what this is for. Compare configs to each other, never a
number here to a published INT8 figure.
"""
import argparse
import json
import time

import torch


AAT_FFN = [f"info_sharing.self_attention_blocks.{i}.mlp.{f}"
           for i in range(8) for f in ("fc1", "fc2")]


def measure(size, aat_depth, res, iters, warmup, dev="cuda", ders=None, widen=0):
    import sys, os
    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(root, "..", "model", "distill"))
    sys.path.insert(0, os.path.join(root, "..", "model", "backbones"))
    sys.path.insert(0, os.path.join(root, "..", "model"))
    from student import build_student
    from teacher import _upcast_geometry_postproc, mono_infer, patch_linalg_cpu_fallback

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # bf16 on CPU BEFORE moving to cuda. An fp32 .to("cuda") peaks past the 8 GB of unified
    # memory and hard-reboots this board; build_student does the cast in the right order.
    model = build_student(size=size, aat_depth=aat_depth, device=dev,
                          dtype=torch.bfloat16).eval()
    if widen:
        sys.path.insert(0, os.path.join(root, "..", "model", "moe"))
        from run_cell import widen_ffn
        widen_ffn(model, widen)
        model = model.to(dev).to(torch.bfloat16).eval()
    if ders:
        from moe.upcycle import upcycle
        k, rank = ders
        upcycle(model, AAT_FFN, "learned_soft", num_experts=k, rank=rank, freeze=False)
        model = model.to(dev).to(torch.bfloat16).eval()
    params = sum(p.numel() for p in model.parameters())
    # build_student registers the casting hooks; the other two patches this board needs are
    # global, so apply them here (both are idempotent).
    _upcast_geometry_postproc()
    patch_linalg_cpu_fallback()

    views = [{"img": torch.rand(1, 3, res, res, device=dev), "data_norm_type": ["dinov2"]}]
    for _ in range(warmup):
        mono_infer(model, views)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mono_infer(model, views)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3

    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()
    return {"params": params, "ms": ms, "fps": 1e3 / ms, "peak_gb": peak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--configs", default="base:8,base:12,base:16,base:24,large:8,large:12")
    ap.add_argument("--out", default="/home/evc/merlin/capacity.json")
    args = ap.parse_args()

    print(f"student capacity frontier @ {args.res}px, bf16, batch 1")
    print("added M/ms = added params per ms of added latency, i.e. how efficiently each way "
          "of\nspending the frame budget converts into capacity. Higher is better.\n")
    print(f"{'config':<16}{'params':>13}{'added':>12}{'ms':>8}{'FPS':>7}"
          f"{'peak GB':>9}{'added M/ms':>12}")
    # Everything is scored against the FIRST config listed, which must be the deployed
    # base:8. "added" is capacity over that baseline however it was bought, dense or expert.
    rows, base_ms, base_params = {}, None, None
    for spec in args.configs.split(","):
        parts = spec.split(":")
        size, depth = parts[0], int(parts[1])
        widen = int(parts[2][1:]) if len(parts) == 3 and parts[2].startswith("w") else 0
        ders = (int(parts[2]), int(parts[3])) if len(parts) == 4 else None
        try:
            r = measure(size, depth, args.res, args.iters, args.warmup, ders=ders, widen=widen)
        except torch.cuda.OutOfMemoryError:
            print(f"{spec:<16}  OOM")
            torch.cuda.empty_cache()
            continue
        except Exception as ex:
            print(f"{spec:<16}  FAILED {type(ex).__name__}: {str(ex)[:50]}")
            continue
        if base_ms is None:
            base_ms, base_params = r["ms"], r["params"]
        r["added"] = r["params"] - base_params
        d_ms = r["ms"] - base_ms
        eff = (r["added"] / 1e6 / d_ms) if (r["added"] and d_ms > 0.5) else float("nan")
        print(f"{spec:<16}{r['params']:>13,}{r['added']:>12,}{r['ms']:>8.1f}{r['fps']:>7.2f}"
              f"{r['peak_gb']:>9.2f}{eff:>12.2f}")
        rows[spec] = r

    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Unattended full MERLIN Phase-0 benchmark on the Jetson.

Produces one JSON with, for the teacher (fp32 + autocast bf16) and quantized
variants: latency / FPS / peak GPU mem / board power / depth sanity / depth
fidelity vs the fp32 reference; plus the ~300M student's latency (untrained --
architecture speed ceiling, which answers the >=5 FPS Phase-0 crux).

Each teacher variant reloads fp32 (torchao quant is in-place + irreversible).
Writes partial results after every variant so a late failure keeps earlier data.
Run in the background; expect ~30-45 min (slow swap-bound loads).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model", "backbones"))
sys.path.insert(0, os.path.join(HERE, "..", "model", "quant"))
sys.path.insert(0, os.path.join(HERE, "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "geometry"))


def tegrastats_power(seconds_done):
    pass  # power sampled inline below


class Power:
    import re as _re
    RAIL = _re.compile(r"(VDD_IN|VDD_CPU_GPU_CV|VDD_SOC)\s+(\d+)mW")

    def __init__(self):
        import subprocess
        self.s = {}
        try:
            self.p = subprocess.Popen(["tegrastats", "--interval", "100"],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            self.p = None

    def stop(self):
        if not self.p:
            return {}
        self.p.terminate()
        try:
            out, _ = self.p.communicate(timeout=3)
        except Exception:
            self.p.kill(); out = ""
        for line in out.splitlines():
            for rail, mw in self.RAIL.findall(line):
                self.s.setdefault(rail, []).append(int(mw))
        return {r: {"mean_w": round(np.mean(v) / 1000, 2), "peak_w": round(np.max(v) / 1000, 2)}
                for r, v in self.s.items() if v}


def make_views(image, res, device):
    from mapanything.utils.image import load_images
    if res in (512, 518):
        v = load_images([image], resize_mode="fixed_mapping", resolution_set=res)
    else:
        s = (res // 14) * 14
        v = load_images([image], resize_mode="square", size=s)
    for x in v:
        x["img"] = x["img"].to(device)  # fp32 input; hooks cast for the net, postproc stays fp32
    return v


def infer(model, views):
    # pure bf16 (no autocast); dtype-casting hooks keep weighted modules consistent
    with torch.no_grad():
        out = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                          use_amp=False, apply_mask=False, mask_edges=False)
    p = out[0]
    return p["depth_z"].float().squeeze().cpu().numpy(), p.get("metric_scaling_factor")


def bench(model, views, warmup, iters):
    for _ in range(warmup):
        infer(model, views)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    pw = Power()
    lat = []
    for _ in range(iters):
        torch.cuda.synchronize(); t = time.time()
        infer(model, views); torch.cuda.synchronize()
        lat.append((time.time() - t) * 1000)
    power = pw.stop()
    lat = np.array(lat)
    return {"latency_ms": {"mean": round(float(lat.mean()), 1), "median": round(float(np.median(lat)), 1),
                           "p95": round(float(np.percentile(lat, 95)), 1)},
            "fps": round(1000.0 / float(lat.mean()), 2),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "power": power}


def save(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/merlin/benchmark_all.json"))
    ap.add_argument("--res", default="252,378")
    ap.add_argument("--quant-res", type=int, default=378)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    from teacher import load_teacher
    from quantize import quantize_model, count_linears
    from metrics import depth_metrics, scale_error
    from student import build_student, param_report

    dev = "cuda"
    res_list = [int(r) for r in args.res.split(",")]
    results = {"meta": {"image": args.image, "res_list": res_list, "quant_res": args.quant_res}, "teacher": [], "student": []}
    ref_depth = {}

    # ---- teacher fp32 floor sweep ----
    print("=== loading teacher fp32 (slow) ===", flush=True)
    t0 = time.time()
    model = load_teacher(device=dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[load] {time.time()-t0:.0f}s params={nparams/1e6:.1f}M", flush=True)
    for res in res_list:
        try:
            views = make_views(args.image, res, dev)
            depth, scale = infer(model, views)
            ref_depth[res] = depth
            b = bench(model, views, args.warmup, args.iters)
            r = {"variant": "fp32_autocast_bf16", "res": res, "params_M": round(nparams / 1e6, 1),
                 "depth_z_m": {"min": round(float(np.nanmin(depth)), 3), "median": round(float(np.nanmedian(depth)), 3),
                               "max": round(float(np.nanmax(depth)), 3)},
                 "metric_scale": round(float(scale.flatten()[0]), 4) if scale is not None else None, **b}
            results["teacher"].append(r); save(results, args.out)
            print(json.dumps(r), flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results["teacher"].append({"variant": "fp32", "res": res, "error": str(e)[:200]}); save(results, args.out)
    del model; torch.cuda.empty_cache()

    # ---- quantized teacher variants (reload fp32 each) ----
    for scheme in ["int8", "int4_int8"]:
        try:
            print(f"=== loading teacher for {scheme} ===", flush=True)
            model = load_teacher(device=dev)
            print(f"[quant] {scheme} linears={count_linears(model)}", flush=True)
            quantize_model(model, scheme=scheme)
            views = make_views(args.image, args.quant_res, dev)
            depth, scale = infer(model, views)
            b = bench(model, views, args.warmup, args.iters)
            r = {"variant": scheme, "res": args.quant_res, **b,
                 "metric_scale": round(float(scale.flatten()[0]), 4) if scale is not None else None}
            if args.quant_res in ref_depth:
                m = depth_metrics(ref_depth[args.quant_res], depth)
                r["fidelity_vs_fp32"] = {"abs_rel": round(m["abs_rel"], 4), "delta1": round(m["delta1"], 4),
                                         "scale_offset": round(scale_error(ref_depth[args.quant_res], depth), 4)}
            results["teacher"].append(r); save(results, args.out)
            print(json.dumps(r), flush=True)
            del model; torch.cuda.empty_cache()
        except Exception as e:
            import traceback; traceback.print_exc()
            results["teacher"].append({"variant": scheme, "error": str(e)[:200]}); save(results, args.out)
            torch.cuda.empty_cache()

    # ---- student (untrained) latency: architecture speed ceiling ----
    try:
        print("=== building student ~300M ===", flush=True)
        student = build_student("base", 8, device=dev)  # fp32 + autocast, like the teacher
        rep = param_report(student)
        print("[student params]", rep, flush=True)
        for res in res_list:
            try:
                views = make_views(args.image, res, dev)
                infer(student, views)  # correctness/shape (untrained output is meaningless)
                b = bench(student, views, args.warmup, args.iters)
                results["student"].append({"res": res, "params": rep, **b}); save(results, args.out)
                print(json.dumps({"student_res": res, **b}), flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                results["student"].append({"res": res, "error": str(e)[:200]}); save(results, args.out)
    except Exception as e:
        import traceback; traceback.print_exc()
        results["student"].append({"error": str(e)[:200]}); save(results, args.out)

    save(results, args.out)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()

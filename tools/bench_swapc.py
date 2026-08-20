#!/usr/bin/env python3
"""On-device SWaP-C benchmark: every candidate measured the same way, on the same board.

The existing SWaP-C figure carries MERLIN as the only measured point and takes its baselines
from papers and vendor specs. That is fine for a Pareto sketch and not fine for a benchmark
claim, because the baselines were measured on other hardware, at other resolutions, with other
power methodologies. This runs each candidate on THIS Orin Nano at the same input resolution,
samples the same INA3221 rails through tegrastats, and reports the same four numbers.

Reported per candidate:
  fps            steady-state, after warmup (the Nano throttles, so cold numbers flatter)
  latency_ms     median and p95, since a keyframe pipeline cares about the tail
  power_w        VDD_IN mean over the measured window, board-level not module-level
  energy_j_kf    power / fps, the number that decides battery life per keyframe
  peak_mem_gb    against the ~6.5 GB the model node is allowed

A candidate that cannot run is recorded with its failure, not omitted: "does not fit in 8 GB"
is a result, and silently dropping it would overstate what the tier can do.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Tegrastats:
    """Board power via tegrastats. VDD_IN is the whole-board rail, which is what a battery
    actually sees; the module rails understate the deployed cost."""

    import re
    RAIL = re.compile(r"(VDD_IN|VDD_CPU_GPU_CV|VDD_SOC)\s+(\d+)mW")

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.proc = None
        self.dgpu = False
        self.samples = {}

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True)
        except FileNotFoundError:
            # Not a Tegra board. Fall back to nvidia-smi so the datacenter tier is measured
            # rather than cited, but keep the rail named differently: nvidia-smi reports the
            # BOARD power of the discrete card, which is not comparable to VDD_IN on a module
            # and must not be pooled with it.
            try:
                self.proc = subprocess.Popen(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                     "-l", "1"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                self.dgpu = True
            except FileNotFoundError:
                self.proc = None
        return self

    def read(self):
        if self.proc is None or self.proc.stdout is None:
            return
        import select
        while select.select([self.proc.stdout], [], [], 0)[0]:
            line = self.proc.stdout.readline()
            if not line:
                break
            if self.dgpu:
                try:
                    self.samples.setdefault("GPU_BOARD", []).append(float(line.strip()))
                except ValueError:
                    pass
                continue
            for rail, mw in self.RAIL.findall(line):
                self.samples.setdefault(rail, []).append(int(mw) / 1000.0)

    def __exit__(self, *a):
        self.read()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def summary(self):
        return {k: {"mean_w": round(float(np.mean(v)), 2), "peak_w": round(float(np.max(v)), 2),
                    "n": len(v)}
                for k, v in self.samples.items() if v}


def bench_engine(path, n_iter=200, warmup=50, sampler=None):
    """TensorRT engine throughput with real device-side timing."""
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    def ck(r):
        """CUDA python bindings return (err,), (err, x) or (err, x, y) depending on the call,
        so unwrap by arity rather than assuming one value."""
        if not isinstance(r, tuple):
            if int(r) != 0:
                raise RuntimeError(f"CUDA error {r}")
            return None
        if int(r[0]) != 0:
            raise RuntimeError(f"CUDA error {r[0]}")
        return r[1] if len(r) == 2 else (r[1:] if len(r) > 2 else None)

    logger = trt.Logger(trt.Logger.ERROR)
    rt = trt.Runtime(logger)
    with open(path, "rb") as f:
        eng = rt.deserialize_cuda_engine(f.read())
    if eng is None:
        raise RuntimeError("failed to deserialize engine")
    ctx = eng.create_execution_context()
    stream = ck(cudart.cudaStreamCreate())
    bufs, keep = {}, []
    for i in range(eng.num_io_tensors):
        name = eng.get_tensor_name(i)
        shape = ctx.get_tensor_shape(name)
        dt = trt.nptype(eng.get_tensor_dtype(name))
        nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize
        ptr = ck(cudart.cudaMalloc(nbytes))
        keep.append(ptr)
        bufs[name] = ptr
        ctx.set_tensor_address(name, ptr)
    for _ in range(warmup):
        ctx.execute_async_v3(stream)
    ck(cudart.cudaStreamSynchronize(stream))
    if sampler is not None:
        sampler.read()
        sampler.samples.clear()      # power window = the timed loop, not engine load
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        ctx.execute_async_v3(stream)
        ck(cudart.cudaStreamSynchronize(stream))
        ts.append((time.perf_counter() - t0) * 1000.0)
    for p in keep:
        cudart.cudaFree(p)
    ts = np.array(ts)
    free_b, total_b = ck(cudart.cudaMemGetInfo())
    return {"latency_ms": {"median": round(float(np.median(ts)), 2),
                           "p95": round(float(np.percentile(ts, 95)), 2)},
            "fps": round(1000.0 / float(np.median(ts)), 2),
            "peak_mem_gb": round((total_b - free_b) / 1e9, 2)}


def bench_torch(fn, n_iter=40, warmup=10, sampler=None):
    """Steady-state torch latency. Warmup matters more here than on a desktop: the Nano
    clocks up under load and a cold measurement reads high.

    `sampler` is cleared after warmup so the power window covers only the timed loop. Without
    that the mean is diluted by model load, which is idle time -- measured, it made a 0.50 GB
    model read 3.7 W against a 0.17 GB one at 6.2 W purely because the larger one took longer
    to load. Energy per keyframe is meaningless if the window is not the compute.
    """
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if sampler is not None:
        sampler.read()
        sampler.samples.clear()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.array(ts)
    return {"latency_ms": {"median": round(float(np.median(ts)), 2),
                           "p95": round(float(np.percentile(ts, 95)), 2)},
            "fps": round(1000.0 / float(np.median(ts)), 2),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


def measure(name, runner, meta, out, seconds=12.0):
    """Run `runner` under power sampling and record the result, or the failure."""
    print(f"--- {name}", flush=True)
    rec = {"name": name, **meta}
    try:
        with Tegrastats() as tg:
            time.sleep(1.0)
            tg.read()
            r = runner(tg)
            tg.read()
            time.sleep(0.4)
        rec.update(r)
        p = tg.summary()
        rec["power"] = p
        vin = p.get("VDD_IN", {}).get("mean_w") or p.get("GPU_BOARD", {}).get("mean_w")
        if vin and r.get("fps"):
            rec["power_w"] = vin
            rec["energy_j_kf"] = round(vin / r["fps"], 3)
        print("    %6.2f FPS  %6.1f ms  %5.2f GB  %5.1f W  %5.3f J/kf"
              % (r.get("fps", 0), r.get("latency_ms", {}).get("median", 0),
                 r.get("peak_mem_gb", 0), vin or 0, rec.get("energy_j_kf", 0)), flush=True)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        print("    FAILED:", rec["error"], flush=True)
    out.append(rec)
    return rec


def bench_hf_depth(hf_id, res, n_iter=40, sampler=None):
    """A transformers depth model, for an independent point on the same board.

    MERLIN's SWaP-C figure currently compares it only against its own teacher and against
    numbers lifted from other papers on other hardware. One externally-authored model measured
    here, at the same resolution and on the same rails, is what makes the tier a benchmark.

    The comparison is throughput-and-power only: these models emit depth, not depth plus
    relative pose, so the figure has to mark the capability gap rather than let FPS imply
    equivalence.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(hf_id)
    model = AutoModelForDepthEstimation.from_pretrained(hf_id).to("cuda").eval()
    px = torch.rand(1, 3, res, res, device="cuda")

    def once():
        with torch.no_grad():
            model(pixel_values=px)

    r = bench_torch(once, n_iter=n_iter, warmup=10, sampler=sampler)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    del model
    torch.cuda.empty_cache()
    r["params_M"] = round(n_params, 1)
    return r


def probe_capacity(hf_id, resolutions, views, budget_gb=6.5):
    """Where does a full-size model stop fitting on this board?

    The SWaP-C claim is that top-of-line reconstruction is undeployable at this tier, and that
    deserves a measurement rather than an assertion. Walk resolution and view count upward and
    record the first configuration that fails, with the exception it failed on. `budget_gb` is
    the model node's actual allowance (~6.5 of the 8 GB), so a config that only fits by using
    memory the rest of the system needs is recorded as over budget, not as a pass.
    """
    import torch
    from mapanything.models import MapAnything

    model = MapAnything.from_pretrained(hf_id).to("cuda").eval()
    rows = []
    for res in resolutions:
        for nv in views:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            rec = {"res": res, "views": nv}
            try:
                vs = [{"img": torch.rand(1, 3, res, res, device="cuda"),
                       "data_norm_type": ["dinov2"]} for _ in range(nv)]
                t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model.infer(vs, memory_efficient_inference=True)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1e9
                rec.update({"ok": True, "peak_mem_gb": round(peak, 2),
                            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                            "over_budget": bool(peak > budget_gb)})
            except Exception as e:
                rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:160]})
            rows.append(rec)
            print("    res %-4d views %-2d %s" % (
                res, nv, ("%.2f GB %7.0f ms%s" % (rec.get("peak_mem_gb", 0),
                                                  rec.get("latency_ms", 0),
                                                  "  OVER BUDGET" if rec.get("over_budget") else ""))
                if rec["ok"] else "FAILED: " + rec["error"][:70]), flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="*", default=[], help="name=path.engine")
    ap.add_argument("--teacher", default=None, help="run the MapAnything teacher at --res")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--out", default="swapc_onboard.json")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--teacher_iters", type=int, default=5,
                    help="timed iterations for the teacher; a handful is enough to show the "
                         "cost and each one is expensive")
    ap.add_argument("--hf_depth", nargs="*", default=[],
                    help="transformers depth models to benchmark on this board, as hf ids")
    ap.add_argument("--probe", default=None,
                    help="hf id: walk resolution/view count to find where it stops fitting")
    ap.add_argument("--probe_res", nargs="+", type=int, default=[252, 378, 518])
    ap.add_argument("--probe_views", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--budget_gb", type=float, default=6.5,
                    help="memory the model node is actually allowed of the 8 GB")
    args = ap.parse_args()

    results = []
    for spec in args.engines:
        name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        if not os.path.exists(path):
            results.append({"name": name, "error": "engine not found"})
            print(f"--- {name}\n    FAILED: engine not found", flush=True)
            continue
        measure(name, lambda tg, p=path: bench_engine(p, args.iters, sampler=tg),
                {"kind": "tensorrt", "res": args.res, "path": path}, results)

    if args.teacher:
        import torch
        from mapanything.models import MapAnything

        def run_teacher(tg=None):
            # A 1.2 B fp32 checkpoint is ~4.9 GB, and the default load path holds the state
            # dict and the model at once on a board with 7.6 GB total. Measured: that drives
            # 6.5 GB into swap at 73-84 MB/s with 1% user CPU, so any timing taken then
            # describes the swap device rather than the accelerator. low_cpu_mem_usage avoids
            # the duplicate where the loader supports it.
            try:
                model = MapAnything.from_pretrained(args.teacher, low_cpu_mem_usage=True)
            except TypeError:
                model = MapAnything.from_pretrained(args.teacher)
            model = model.to("cuda").eval()
            img = torch.rand(1, 3, args.res, args.res, device="cuda")
            views = [{"img": img, "data_norm_type": ["dinov2"]}]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                r = bench_torch(lambda: model.infer(views, memory_efficient_inference=True),
                                n_iter=args.teacher_iters,
                                warmup=max(1, args.teacher_iters // 2), sampler=tg)
            r["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
            del model
            torch.cuda.empty_cache()
            return r

        measure(f"teacher {args.teacher}", run_teacher,
                {"kind": "torch-bf16", "res": args.res}, results)

    for hf in args.hf_depth:
        measure(hf, lambda tg, h=hf: bench_hf_depth(h, args.res, sampler=tg),
                {"kind": "torch-hf", "res": args.res, "outputs": "depth only"}, results)

    if args.probe:
        print(f"--- capacity probe: {args.probe}", flush=True)
        try:
            results.append({"name": f"probe {args.probe}", "kind": "capacity",
                            "budget_gb": args.budget_gb,
                            "grid": probe_capacity(args.probe, args.probe_res, args.probe_views,
                                                   args.budget_gb)})
        except Exception as e:
            results.append({"name": f"probe {args.probe}", "kind": "capacity",
                            "error": f"{type(e).__name__}: {e}"[:300]})
            print("    probe FAILED:", results[-1]["error"], flush=True)

    json.dump(results, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)
    print("%-28s %8s %9s %8s %8s %9s" % ("candidate", "FPS", "lat(ms)", "mem(GB)", "power(W)", "J/kf"))
    for r in results:
        if "error" in r:
            print("%-28s %s" % (r["name"][:28], "DID NOT RUN: " + r["error"][:40]))
            continue
        print("%-28s %8.2f %9.1f %8.2f %8.1f %9.3f"
              % (r["name"][:28], r.get("fps", 0), r.get("latency_ms", {}).get("median", 0),
                 r.get("peak_mem_gb", 0), r.get("power_w", 0), r.get("energy_j_kf", 0)))


if __name__ == "__main__":
    main()

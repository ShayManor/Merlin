#!/usr/bin/env python3
"""Gate 0 for the MoE direction: is the vanilla-MoE weight-traffic penalty real on this board?

Measures achieved INT8 GEMM throughput at MERLIN's deployed AAT FFN shapes and sustained device
bandwidth, then predicts dense / vanilla-MoE / residual-MoE block times from those two measured
numbers. Decision rule lives in docs/superpowers/specs/2026-08-08-moe-00-gate0-roofline.md; the
cost accounting here is unit-tested in tools/tests/test_bench_roofline.py.

Run on the Jetson. Engines are device-specific, so building this anywhere else is meaningless.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

D = 768          # AAT model dim
DH = 3072        # AAT FFN hidden dim
TOKENS = 729     # 27x27 patches at 378px, patch 14
K = 4            # experts
R = 8            # delta rank
WBYTES = 1       # INT8 weight

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


# ---------------------------------------------------------------- cost accounting

def ffn_gemms():
    """The two GEMMs in one AAT FFN block, as (M, K, N)."""
    return [(TOKENS, D, DH), (TOKENS, DH, D)]


def variant_cost(variant, k=K, r=R):
    """Weight bytes and FLOPs for one FFN block."""
    base = 2 * D * DH                      # W1 + W2
    delta = 2 * k * r * (D + DH)           # k experts, rank r, both matrices
    if variant == "dense":
        wparams, fparams = base, base
    elif variant == "vanilla":
        # k full expert copies are read every pass; top-1 means each token does one expert
        wparams, fparams = k * base, base
    elif variant == "residual":
        # base plus low-rank deltas; all k branches computed for every token
        wparams = fparams = base + delta
    else:
        raise ValueError(variant)
    return wparams * WBYTES, 2 * TOKENS * fparams


def predict_time_s(flops, wbytes, tops, bw):
    """Roofline block time: whichever of math and weight traffic dominates."""
    return max(flops / tops, wbytes / bw)


def penalty(tops, bw, k=K, r=R):
    """Predicted vanilla-MoE and residual-MoE block times relative to dense."""
    out = {}
    for v in ("dense", "vanilla", "residual"):
        b, f = variant_cost(v, k, r)
        t = predict_time_s(f, b, tops, bw)
        out[v] = {
            "weight_bytes": b,
            "flops": f,
            "intensity_flop_per_byte": f / b,
            "time_us": t * 1e6,
            "bound_by": "memory" if b / bw > f / tops else "compute",
        }
    d = out["dense"]["time_us"]
    out["vanilla_penalty"] = out["vanilla"]["time_us"] / d
    out["residual_penalty"] = out["residual"]["time_us"] / d
    return out


def decide(vanilla_penalty_by_mode):
    """Spec 00 decision rule. Returns one of proceed / kill / measure_directly."""
    best = max(vanilla_penalty_by_mode.values())
    if best >= 1.25:
        return "proceed"
    if all(p < 1.10 for p in vanilla_penalty_by_mode.values()):
        return "kill"
    return "measure_directly"


# ---------------------------------------------------------------- synthetic GEMM

def make_qdq_matmul_onnx(m, k, n, path):
    """Single INT8 MatMul via explicit QuantizeLinear/DequantizeLinear on input and weight."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def scale(name, v):
        return numpy_helper.from_array(np.array(v, dtype=np.float32), name)

    def zp(name):
        return numpy_helper.from_array(np.array(0, dtype=np.int8), name)

    w = np.random.randn(k, n).astype(np.float32) * 0.02
    inits = [
        numpy_helper.from_array(w, "W"),
        scale("sx", 0.02), zp("zx"), scale("sw", 0.001), zp("zw"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["X", "sx", "zx"], ["Xq"]),
        helper.make_node("DequantizeLinear", ["Xq", "sx", "zx"], ["Xdq"]),
        helper.make_node("QuantizeLinear", ["W", "sw", "zw"], ["Wq"]),
        helper.make_node("DequantizeLinear", ["Wq", "sw", "zw"], ["Wdq"]),
        helper.make_node("MatMul", ["Xdq", "Wdq"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes, f"gemm_{m}x{k}x{n}",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [m, k])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [m, n])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    return path


def run_trtexec(onnx_path, iterations=500, profile=False, extra=()):
    """Time an engine. Layer profiling is OFF during timing: --profilingVerbosity=detailed
    with --dumpLayerInfo instruments every layer and inflated these shapes by ~68% when both
    were left on, which is measurement overhead rather than board behaviour."""
    cmd = [
        TRTEXEC, f"--onnx={onnx_path}", "--best",
        f"--iterations={iterations}", "--warmUp=2000", "--duration=10",
        "--noDataTransfers", "--useSpinWait",
    ]
    if profile:
        cmd += ["--profilingVerbosity=detailed", "--dumpLayerInfo"]
    cmd.extend(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = p.stdout + p.stderr
    m = re.search(r"GPU Compute Time:.*?median = ([\d.]+) ms", log)
    if not m:
        return None, log
    return float(m.group(1)) / 1e3, log


def layer_precision(log):
    """Which precisions TensorRT actually selected, from the dumped layer info."""
    found = set()
    for tag in ("Int8", "Half", "Float", "INT8", "FP16", "FP32"):
        if re.search(rf'"[Pp]recision"\s*:\s*"{tag}"', log) or f'precision: {tag}' in log:
            found.add(tag.upper().replace("HALF", "FP16").replace("FLOAT", "FP32"))
    return sorted(found) or ["unknown"]


def measure_gemms(workdir, iterations):
    """Achieved throughput at each deployed FFN GEMM shape."""
    out = []
    for (m, k, n) in ffn_gemms():
        path = os.path.join(workdir, f"gemm_{m}_{k}_{n}.onnx")
        make_qdq_matmul_onnx(m, k, n, path)
        t, log = run_trtexec(path, iterations)
        flops = 2 * m * k * n
        rec = {"shape": [m, k, n], "flops": flops}
        if t is None:
            rec["error"] = log[-2000:]
        else:
            # Separate instrumented pass, timing discarded, only to record what TRT selected.
            _, plog = run_trtexec(path, iterations=10, profile=True)
            rec.update(time_us=t * 1e6, tops=flops / t / 1e12,
                       tactic=(re.search(r"TacticName: (\S+)", plog) or [None, "unknown"])[1],
                       int8_gemm=bool(re.search(r"LayerType: gemm.*?Int8", plog, re.S)))
        out.append(rec)
    return out


# ---------------------------------------------------------------- bandwidth

def measure_bandwidth(mb=512, reps=50):
    """Sustained device bandwidth: streaming read, and read+write copy."""
    import torch
    n = mb * 1024 * 1024
    x = torch.empty(n, dtype=torch.int8, device="cuda")
    y = torch.empty_like(x)

    def timed(fn, warmup=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / 1e3 / reps

    t_copy = timed(lambda: y.copy_(x))
    # fp32 reduction: a well-optimised read-only streaming kernel. An int8->int64 sum measures
    # type-conversion cost, not the memory system, so it is not a usable proxy here.
    f = torch.empty(n // 4, dtype=torch.float32, device="cuda")
    t_read = timed(lambda: torch.sum(f))
    copy_gbs = 2 * n / t_copy / 1e9
    read_gbs = (n // 4) * 4 / t_read / 1e9
    return {
        "copy_gbs": copy_gbs,               # read + write
        "read_f32_gbs": read_gbs,           # streaming read
        # Weight fetch proxy. Taking the best observed rate is the conservative choice: higher
        # bandwidth makes the vanilla arm less memory-bound and so weakens the method claim.
        "achieved_gbs": max(copy_gbs, read_gbs),
        "bytes": n,
    }


# ---------------------------------------------------------------- routing assumption

def all_experts_touched(trials=2000, k=K, tokens=TOKENS):
    """Fraction of forward passes in which every expert receives at least one token.

    An untrained top-1 router over 729 tokens is the best case for the vanilla arm's weight
    traffic. A trained router can only lower this by collapsing, which would also defeat the
    point of having experts, so this bounds the assumption rather than settling it.
    """
    import torch
    g = torch.Generator().manual_seed(0)
    hits = 0
    for _ in range(trials):
        a = torch.randint(0, k, (tokens,), generator=g)
        hits += int(torch.bincount(a, minlength=k).min() > 0)
    return hits / trials


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gate0.json")
    ap.add_argument("--mode-label", default="unknown", help="power mode this run was taken in")
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--skip-gemm", action="store_true")
    args = ap.parse_args()

    res = {"mode": args.mode_label, "config": {"D": D, "DH": DH, "TOKENS": TOKENS, "K": K, "R": R}}

    if not args.skip_gemm:
        with tempfile.TemporaryDirectory() as wd:
            res["gemms"] = measure_gemms(wd, args.iterations)

    res["bandwidth"] = measure_bandwidth()
    res["all_experts_touched"] = all_experts_touched()

    ok = [g for g in res.get("gemms", []) if "tops" in g]
    if ok:
        # Both GEMMs make up one block; achieved throughput is the block aggregate.
        tot_f = sum(g["flops"] for g in ok)
        tot_t = sum(g["time_us"] for g in ok) / 1e6
        tops = tot_f / tot_t / 1e12
        bw = res["bandwidth"]["achieved_gbs"] * 1e9
        res["achieved_tops"] = tops
        res["achieved_gbs"] = bw / 1e9
        res["roofline"] = penalty(tops * 1e12, bw)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

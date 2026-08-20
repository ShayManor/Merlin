#!/usr/bin/env python3
"""What does the FFNMoE arm actually COST on the Nano? Real shapes, dummy weights.

`bench_ore_latency.py` measures ORE, whose expert is a rank-r residual on a shared base. The
arms that were trained and flown (U8 top-1/top-2, U2 top-1) are FFNMoE: K full copies of the
AAT MLP, routed per token. Nothing has ever timed those on-device, so every claim about them is
matched-PARAMETER. Matched-latency is the bar the deployment story needs and this is the
measurement it is missing.

Two levels, both fp16:
  site     one AAT MLP, 768 -> 3072 -> 768 over 729 tokens. Isolates dispatch from everything.
  model    the whole distilled student on one 378x378 frame, 8 AAT sites wrapped. The number
           that actually decides accuracy-per-ms.

Two dispatch implementations, because they are 3x apart and only one of them is a fair floor:
  loop     FFNMoE.forward as trained -- a Python loop over K experts, boolean mask, nonzero,
           index_add. K kernel launches per site whether or not the expert is used.
  fused    sort tokens by expert once, two grouped GEMMs (triton_moe._grouped_gemm), combine.
           top_k > 1 replicates each token into top_k dispatch slots, so J = M*top_k rows go
           through the same kernel and the weighted sum happens on the scatter.

Dense controls at the same site: F=3072 (dense FT, arm L150_ft) and the two widened arms,
+165 hidden (matched-active, 39.8M) and +5200 (+63.9M).
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "backbones"))

import triton  # noqa: E402

from moe.triton_moe import _grouped_gemm, align_block_size, padded_capacity  # noqa: E402

D, DH, N_TOK = 768, 3072, 729
SITES = [f"info_sharing.self_attention_blocks.{b}.mlp" for b in range(8)]


# ------------------------------------------------------------------ fused dispatch

class FusedMoETopK(nn.Module):
    """Top-k MoE FFN over K experts, sorted dispatch, two grouped GEMMs.

    The top-1 case scatters straight out of the second GEMM (the renormalised gate weight is
    exactly 1.0, so there is nothing to combine). For top_k > 1 the second GEMM writes in SLOT
    order and a masked index_add does the weighted combine, which is one elementwise pass over
    (n_slots, d) instead of a second sort.
    """

    def __init__(self, k, top_k, d=D, dh=DH, block_m=64, device="cuda", dtype=torch.float16):
        super().__init__()
        self.k, self.top_k, self.d, self.dh = k, top_k, d, dh
        self.block_m = block_m if k > 4 else 32
        self.gate = nn.Linear(d, k, device=device, dtype=dtype)
        self.w1 = nn.Parameter(torch.randn(k, d, dh, device=device, dtype=dtype) * d ** -0.5)
        self.w2 = nn.Parameter(torch.randn(k, dh, d, device=device, dtype=dtype) * dh ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(k, dh, device=device, dtype=dtype))
        self.b2 = nn.Parameter(torch.zeros(k, d, device=device, dtype=dtype))

    @torch.no_grad()
    def load_from(self, moe):
        """Copy an FFNMoE's gate and expert MLPs in, for the parity check."""
        self.gate.weight.copy_(moe.gate.weight)
        self.gate.bias.copy_(moe.gate.bias)
        for i, e in enumerate(moe.experts):
            self.w1[i].copy_(e.fc1.weight.t())
            self.b1[i].copy_(e.fc1.bias)
            self.w2[i].copy_(e.fc2.weight.t())
            self.b2[i].copy_(e.fc2.bias)
        return self

    def route(self, flat):
        probs = self.gate(flat).softmax(-1)
        if self.top_k == 1:
            return probs.argmax(-1).to(torch.int32).reshape(-1, 1), None
        idx = probs.topk(self.top_k, dim=-1).indices
        w = probs.gather(-1, idx)
        return idx.to(torch.int32), w / w.sum(-1, keepdim=True).clamp_min(1e-9)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.d)
        m = flat.shape[0]
        idx, w = self.route(flat)
        j = m * self.top_k
        eidx = idx.reshape(-1)
        n_slots = padded_capacity(j, self.k, self.block_m)
        stok, be = align_block_size(eidx, self.k, self.block_m, n_slots)
        n_blocks = n_slots // self.block_m

        # slot -> source TOKEN. Padding slots carry the sentinel j, which maps to m and is what
        # the kernel tests against, so the same tensor serves as gather index and padding mask.
        gat = torch.where(stok < j, stok // self.top_k, m).to(torch.int32)
        mid = torch.empty((n_slots, self.dh), device=flat.device, dtype=flat.dtype)
        scatter = self.top_k == 1
        out = (torch.empty_like(flat) if scatter
               else torch.zeros((n_slots, self.d), device=flat.device, dtype=flat.dtype))

        grid = lambda meta: (n_blocks * triton.cdiv(self.dh, meta["BLOCK_N"]),)
        _grouped_gemm[grid](
            flat, self.w1, self.b1, mid, gat, be, m, self.dh, self.d,
            flat.stride(0), flat.stride(1),
            self.w1.stride(0), self.w1.stride(1), self.w1.stride(2),
            mid.stride(0), mid.stride(1), self.b1.stride(0),
            GATHER_A=True, SCATTER_C=False, ACT_GELU=True, BLOCK_M=self.block_m)
        grid = lambda meta: (n_blocks * triton.cdiv(self.d, meta["BLOCK_N"]),)
        _grouped_gemm[grid](
            mid, self.w2, self.b2, out, gat, be, m, self.d, self.dh,
            mid.stride(0), mid.stride(1),
            self.w2.stride(0), self.w2.stride(1), self.w2.stride(2),
            out.stride(0), out.stride(1), self.b2.stride(0),
            GATHER_A=False, SCATTER_C=scatter, ACT_GELU=False, BLOCK_M=self.block_m)
        if scatter:
            return out.reshape(shape)

        valid = stok < j
        src = stok.clamp(max=j - 1).long()
        ws = torch.where(valid, w.reshape(-1)[src], torch.zeros((), dtype=w.dtype, device=w.device))
        tok = torch.where(valid, src // self.top_k, torch.zeros_like(src))
        y = torch.zeros((m, self.d), device=flat.device, dtype=flat.dtype)
        y.index_add_(0, tok, out * ws.unsqueeze(-1))
        return y.reshape(shape)


class DenseMLP(nn.Module):
    """The AAT MLP as the student ships it, so `loop` and `fused` share one dense floor."""

    def __init__(self, d=D, dh=DH, device="cuda", dtype=torch.float16):
        super().__init__()
        self.fc1 = nn.Linear(d, dh, device=device, dtype=dtype)
        self.fc2 = nn.Linear(dh, d, device=device, dtype=dtype)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ------------------------------------------------------------------ timing

def bench(fn, iters=50, warm=20, cool=0.0):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    if cool:
        time.sleep(cool)          # let the board settle between back-to-back saturating bursts
    return a.elapsed_time(b) / iters


_KEEP_GRAPHS = []
HEARTBEAT = os.environ.get("MOE_HEARTBEAT")


def beat(what):
    """Name the thing about to run, on disk, fsynced, before running it.

    Three hangs have left no kernel evidence and no clue which variant was executing. The last
    line of this file is that clue, so it is written BEFORE the work and forced to the platter;
    a buffered heartbeat tells you nothing about the crash that ate the buffer.
    """
    if not HEARTBEAT:
        return
    with open(HEARTBEAT, "a") as fh:
        fh.write(f"{time.time():.1f} {what}\n")
        fh.flush()
        os.fsync(fh.fileno())


def bench_graph(mod, x, iters=50):
    """Same forward, CUDA-graph captured. Returns None if capture fails.

    Sorted dispatch is ~10 kernels over 729- and 8-element tensors, so eager it is pure launch
    latency on an edge GPU and the measurement describes the CPU, not the device. align_block_size
    is deliberately sync-free for exactly this reason (triton_moe, 2026-08-11). Warm up on a side
    stream first or triton autotunes DURING capture and the graph records the wrong kernel.
    """
    beat("  graph capture")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            mod(x)
    torch.cuda.current_stream().wait_stream(s)
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            mod(x)
    except Exception as e:
        print(f"    [graph] capture failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return None
    ms = bench(g.replay, iters=iters)
    # The graph is deliberately NOT deleted, and it is deliberately parked in a module-level
    # list. Two site runs that kept their graphs alive completed; every run since `del g` was
    # added has taken the board off the network mid-cell, always inside this bench. Freeing a
    # just-replayed graph's private memory pool is the only thing that changed. Leaking it costs
    # a bounded pool in a process that exits after one cell, which is the cheaper side to be
    # wrong on -- a crash costs a power cycle.
    _KEEP_GRAPHS.append(g)
    torch.cuda.synchronize()
    return ms


def check_parity(args, dev, dt):
    """`fused` must compute what `loop` computes, or its latency describes a different model.

    fp16 with a 3072-deep reduction, so the bar is relative: the two paths accumulate the same
    dot products in a different order. A real dispatch bug (wrong expert, dropped token, missing
    renormalisation) is O(1) here, not O(1e-3).
    """
    from moe.ffn_moe import FFNMoE

    x = torch.randn(1, N_TOK, D, device=dev, dtype=dt)
    out = []
    with torch.no_grad():
        for tag, k, tk in args.cells:
            mlp = DenseMLP(device=dev, dtype=dt)
            moe = FFNMoE(mlp, num_experts=k, top_k=tk, balance="aux").to(dev)
            # upcycle_ffn makes the experts EXACT copies, which is function-preserving by
            # design -- and it would make this check pass for any dispatch at all, including
            # one that sent every token to expert 0. Separate them first.
            for e in moe.experts:
                e.fc1.weight.add_(torch.randn_like(e.fc1.weight) * 0.05)
                e.fc2.weight.add_(torch.randn_like(e.fc2.weight) * 0.05)
            moe.eval()
            f = FusedMoETopK(k, tk, block_m=args.block_m, device=dev, dtype=dt).load_from(moe)
            a, b = moe(x), f(x)
            rel = float((a - b).abs().max() / a.abs().max())
            ok = rel < 5e-3
            out.append({"arm": tag, "rel_max": rel, "pass": ok})
            print(f"[check] {tag:>10} fused-vs-loop rel max {rel:.2e} {'PASS' if ok else 'FAIL'}",
                  flush=True)
            del mlp, moe, f
            torch.cuda.empty_cache()
    return out


def bench_site(args, dev, dt, flush=None):
    from moe.ffn_moe import FFNMoE

    x = torch.randn(1, N_TOK, D, device=dev, dtype=dt)
    rows = []
    if flush is None:
        flush = lambda _rows: None

    with torch.no_grad():
        # widen168 is widen165 rounded up to a multiple of 8. The trained arm used 165, and an
        # odd hidden dim drops cuBLAS off the tensor-core path, so the two are reported
        # separately rather than one standing in for the other.
        for tag, dh in (("dense", DH), ("widen165", DH + 165), ("widen168", DH + 168),
                        ("widen5200", DH + 5200)):
            beat(f"site dense {tag} dh={dh}")
            mlp = DenseMLP(dh=dh, device=dev, dtype=dt)
            rows.append({"arm": tag, "impl": "dense", "k": 1, "top_k": 1, "dh": dh,
                         "ms": bench(lambda: mlp(x)),
                         "graph_ms": bench_graph(mlp, x) if args.graph else None})
            flush(rows)
            del mlp
            torch.cuda.empty_cache()
        dense_ms = rows[0]["ms"]

        for tag, k, tk in args.cells:
            mlp = DenseMLP(device=dev, dtype=dt)
            moe = FFNMoE(mlp, num_experts=k, top_k=tk, balance="aux").to(dev).to(dt)
            moe.eval()
            rows.append({"arm": tag, "impl": "loop", "k": k, "top_k": tk, "dh": DH,
                         "ms": bench(lambda: moe(x))})
            # router in isolation: the part that exists whatever the dispatch looks like
            flat = x.reshape(-1, D)
            rows[-1]["router_ms"] = bench(lambda: moe._route_topk(flat, x.shape))
            flush(rows)
            del mlp, moe
            torch.cuda.empty_cache()

            # block_m is the whole story for top-1: every expert pads its token run up to a
            # multiple of it, so 729 real tokens become up to 1792 GEMM rows at bm=128 and the
            # padded rows cost full compute. Sweeping it here rather than inheriting
            # triton_moe's bm=64 default, which was tuned at K=4..64 on a different shape.
            for bm in args.block_ms:
                beat(f"site fused {tag} k={k} top_k={tk} bm={bm} graph={args.graph}")
                f = FusedMoETopK(k, tk, block_m=bm, device=dev, dtype=dt)
                def _disp(f=f, bm=bm):
                    idx, _ = f.route(flat)
                    jj = N_TOK * tk
                    ns = padded_capacity(jj, k, f.block_m)
                    st, _b = align_block_size(idx.reshape(-1), k, f.block_m, ns)
                    return torch.where(st < jj, st // tk, N_TOK)
                rows.append({"arm": tag, "impl": f"fused bm={f.block_m}", "k": k, "top_k": tk,
                             "dh": DH, "block_m": f.block_m, "ms": bench(lambda: f(x)),
                             "router_ms": bench(lambda: f.route(flat)),
                             "dispatch_ms": bench(_disp),
                             "graph_ms": bench_graph(f, x) if args.graph else None,
                             "slots": padded_capacity(N_TOK * tk, k, f.block_m)})
                flush(rows)
                del f
                torch.cuda.empty_cache()

    for r in rows:
        r["vs_dense"] = r["ms"] / dense_ms
    return rows, dense_ms


# ------------------------------------------------------------------ whole model

def _get(model, path):
    m = model
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() and not hasattr(m, p) else getattr(m, p)
    return m


def _set(model, path, new):
    parent, _, leaf = path.rpartition(".")
    obj = _get(model, parent) if parent else model
    setattr(obj, leaf, new)


def view(size, dev, dt):
    """The image stays fp32 on purpose. The geometric encoders and LayerNorms leak fp32, which
    is why build_student(dtype=...) installs casting hooks; handing them an fp16 image instead
    breaks the RGB postproc. Same recipe as the working teacher path."""
    import numpy as np
    img = torch.randn(1, 3, size, size, device=dev, dtype=torch.float32)
    return [dict(img=img, true_shape=np.array([[size, size]]), idx=0, instance=["0"],
                 data_norm_type=["dinov2"])]


def bench_model(args, dev, dt):
    from student import build_student
    from moe.ffn_moe import FFNMoE

    size = (args.res // 14) * 14
    v = view(size, dev, dt)
    rows = []

    def run(model):
        with torch.no_grad():
            fn = lambda: model(v, memory_efficient_inference=True, minibatch_size=1)
            ms = bench(fn, iters=args.model_iters, warm=5)
        return ms, torch.cuda.max_memory_allocated() / 1e9

    plan = ([("dense", 0, 0, "dense")]
            + [(t, k, tk, i) for t, k, tk in args.cells for i in ("loop", "fused")]
            + [("widen165", 0, 0, "widen165"), ("widen168", 0, 0, "widen168"),
               ("widen5200", 0, 0, "widen5200")])
    if args.arms:
        plan = [p for p in plan if f"{p[0]}:{p[3]}" in args.arms]
    for tag, k, tk, impl in plan:
        torch.cuda.reset_peak_memory_stats()
        model = build_student(size="base", aat_depth=8, device="cpu", dtype=dt)
        if impl == "loop":
            from moe.ffn_moe import upcycle_ffn
            upcycle_ffn(model, SITES, num_experts=k, top_k=tk, balance="aux")
        elif impl == "fused":
            for p in SITES:
                _set(model, p, FusedMoETopK(k, tk, block_m=args.block_m, device="cpu", dtype=dt))
        elif impl.startswith("widen"):
            import moe.run_cell as rc
            rc.widen_ffn(model, int(impl[5:]))
            model = model.to(dt)   # widen_ffn builds fresh fp32 Linears with no casting hook
        model = model.to(dev).eval()
        n = sum(p.numel() for p in model.parameters())
        ms, gb = run(model)
        rows.append({"arm": tag, "impl": impl, "k": k, "top_k": tk, "ms": ms,
                     "fps": 1000 / ms, "params_M": n / 1e6, "peak_gb": gb})
        print(f"  {tag:>10} {impl:>7}  {ms:8.2f} ms  {1000/ms:6.2f} FPS  "
              f"{n/1e6:7.1f} M  peak {gb:.2f} GB  "
              f"resident {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
        del model
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="both", choices=["site", "model", "both"])
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--block-ms", type=int, nargs="+", default=[64],
                    help="one value per PROCESS is the safe way to sweep: each distinct "
                         "BLOCK_M recompiles the full 18-config autotune space")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--model-iters", type=int, default=20)
    ap.add_argument("--no-graph", dest="graph", action="store_false",
                    help="skip CUDA-graph capture; eager numbers only")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="model level: restrict to these 'tag:impl' cells, one process each")
    ap.add_argument("--out", default="/home/evc/merlin/ffn_moe_latency.json")
    args = ap.parse_args()
    # (label, K, top_k) -- exactly the trained arms, plus K=1 as the FFNMoE-equals-dense check
    args.cells = [("U8_top1", 8, 1), ("U8_top2", 8, 2), ("U2_top1", 2, 1)]

    dev, dt = "cuda", torch.float16
    torch.manual_seed(0)
    res = {"device": torch.cuda.get_device_name(0), "dtype": "fp16",
           "block_m": args.block_m, "res": args.res}

    if args.level in ("site", "both"):
        print("\n== parity: fused dispatch vs the trained loop ==", flush=True)
        res["parity"] = check_parity(args, dev, dt)
        if any(not c["pass"] for c in res["parity"]):
            print("PARITY FAILED -- fused latency does not describe the trained model")
            return 1
        print(f"\n== site: {D}->{DH}->{D}, {N_TOK} tokens, fp16 ==", flush=True)
        def _flush(rs):
            """Write and fsync after every row. Three board hangs have landed inside this bench,
            and one of them ate a result the process had already printed as written."""
            res["site"] = rs
            with open(args.out, "w") as fh:
                json.dump(res, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())

        rows, dense_ms = bench_site(args, dev, dt, flush=_flush)
        res["site"] = rows
        res["site_dense_ms"] = dense_ms
        print(f"{'arm':>10} {'impl':>14} {'ms':>8} {'xdense':>7} {'graph':>8} {'slots':>6} {'router':>8} "
              f"{'disp':>8}")
        for r in rows:
            print(f"{r['arm']:>10} {r['impl']:>14} {r['ms']:>8.3f} {r['vs_dense']:>6.2f}x "
                  f"{(r.get('graph_ms') or float('nan')):>8.3f} {r.get('slots', 0):>6d} "
                  f"{r.get('router_ms', float('nan')):>8.3f} "
                  f"{r.get('dispatch_ms', float('nan')):>8.3f}")

    if args.level in ("model", "both"):
        print(f"\n== whole student, {args.res}px, fp16, batch 1 ==", flush=True)
        res["model"] = bench_model(args, dev, dt)

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)

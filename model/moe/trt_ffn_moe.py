#!/usr/bin/env python3
"""Can a routed FFN reach an INT8 TensorRT engine at all, and at what cost?

The MoE plan lists this as unstarted work and assumes the answer is a custom IPluginV3, because
true top-1 dispatch has a data-dependent output shape (expert i gets however many tokens the
router gave it). Every deployed MERLIN number is INT8-TRT, so "no engine" means the mixture is
not deployable and the honest fallback is fp16 PyTorch on-device.

Four formulations of the SAME top-1 routed FFN, in increasing order of how much they ask of the
parser. The question each answers is "does this build, and what does it cost":

  dense      the unrouted MLP. The control, and the latency floor.
  masked     compute all K experts, weight by the router, sum. Fully static, definitely builds,
             and costs ~K x dense. This is the price of refusing to dispatch.
  capacity   fixed capacity C per expert, so every shape is a host constant: one-hot, cumsum for
             the rank within an expert, scatter into (K, C, d), batched GEMM, gather back.
             Tokens past C are DROPPED, which is an accuracy cost, not a latency one, and is
             reported as drop_frac. If this builds, no plugin is needed.
  caponehot  the same fixed capacity, but the scatter and gather written as 0/1 matmuls. Tests
             whether the capacity failure is a property of ONE OPERATOR or of static dispatch.
  nonzero    the textbook implementation: torch.nonzero per expert. Data-dependent shapes, and
             the formulation the plugin would exist to support. Expected to fail; measured
             rather than assumed.

Latency comes from trtexec, which fabricates dynamic ranges under --int8. That makes the INT8
ACCURACY meaningless and the INT8 LATENCY representative, which is the only thing asked here.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

D, DH, N_TOK = 768, 3072, 729


class DenseMLP(nn.Module):
    def __init__(self, d=D, dh=DH):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, dh), nn.Linear(dh, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Experts(nn.Module):
    """K expert MLPs held as stacked (K, d, dh) / (K, dh, d) tensors, so every formulation
    below reads the same weights and any latency difference is dispatch, not layout."""

    def __init__(self, k, d=D, dh=DH):
        super().__init__()
        self.k, self.d, self.dh = k, d, dh
        self.gate = nn.Linear(d, k)
        self.w1 = nn.Parameter(torch.randn(k, d, dh) * d ** -0.5)
        self.w2 = nn.Parameter(torch.randn(k, dh, d) * dh ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(k, 1, dh))
        self.b2 = nn.Parameter(torch.zeros(k, 1, d))


class MaskedMoE(_Experts):
    """All K experts computed, router-weighted sum. No dispatch, so no dynamic shape."""

    def forward(self, x):
        flat = x.reshape(-1, self.d)
        p = self.gate(flat).softmax(-1)                              # (M, K)
        h = F.gelu(torch.matmul(flat.unsqueeze(0), self.w1) + self.b1)   # (K, M, dh)
        y = torch.matmul(h, self.w2) + self.b2                        # (K, M, d)
        return (p.t().unsqueeze(-1) * y).sum(0).reshape(x.shape)


class CapacityMoE(_Experts):
    """Top-1 with a fixed per-expert capacity, which is what makes every shape static.

    `pos` is the token's rank inside its own expert's queue, from a cumulative sum over the
    one-hot assignment. That is the only place the data-dependence lives, and a cumsum has a
    static shape, so the whole dispatch reduces to one scatter and one gather at constant size.
    The overflow row (index K*C) absorbs every token past capacity and is discarded, so a
    dropped token contributes zero rather than corrupting another token's slot.
    """

    def __init__(self, k, cap_factor=2.0, m=N_TOK, **kw):
        super().__init__(k, **kw)
        self.c = int(-(-m // k) * cap_factor)
        self.m = m

    def forward(self, x):
        flat = x.reshape(-1, self.d)
        m, k, c = self.m, self.k, self.c
        idx = self.gate(flat).argmax(-1)                             # (M,)
        oh = F.one_hot(idx, k).to(flat.dtype)                        # (M, K)
        pos = (oh.cumsum(0) * oh).sum(-1) - 1                        # (M,) rank within expert
        keep = (pos < c).to(flat.dtype)                              # (M,)
        slot = (idx * c + pos.clamp(0, c - 1).long()) * keep.long() + (k * c) * (1 - keep.long())

        buf = torch.zeros(k * c + 1, self.d, dtype=flat.dtype, device=flat.device)
        buf = buf.index_copy(0, slot, flat)
        h = F.gelu(torch.bmm(buf[:k * c].reshape(k, c, self.d), self.w1) + self.b1)
        y = (torch.bmm(h, self.w2) + self.b2).reshape(k * c, self.d)
        y = torch.cat([y, torch.zeros(1, self.d, dtype=y.dtype, device=y.device)], 0)
        return (y[slot] * keep.unsqueeze(-1)).reshape(x.shape)

    @torch.no_grad()
    def drop_frac(self, x):
        flat = x.reshape(-1, self.d)
        idx = self.gate(flat).argmax(-1)
        oh = F.one_hot(idx, self.k).to(flat.dtype)
        pos = (oh.cumsum(0) * oh).sum(-1) - 1
        return float((pos >= self.c).float().mean())


class CapacityOneHotMoE(CapacityMoE):
    """Same fixed-capacity top-1, with the scatter and the gather replaced by matmuls.

    TRT 10.3 rejects the scatter form with "an IScatterLayer cannot be used to compute a shape
    tensor", so the question of whether STATIC top-1 dispatch is expressible at all is not
    actually settled by that failure -- it is a statement about one operator. The permutation a
    scatter performs is a 0/1 matrix, and A.t() @ x moves tokens into slots exactly as
    index_copy does, using an operator every backend supports.

    The cost is two extra (M x K*C x d) matmuls, which is the honest price of refusing to
    dispatch dynamically. It buys the whole engine.
    """

    def forward(self, x):
        flat = x.reshape(-1, self.d)
        k, c = self.k, self.c
        idx = self.gate(flat).argmax(-1)
        oh = F.one_hot(idx, k).to(flat.dtype)
        pos = (oh.cumsum(0) * oh).sum(-1) - 1
        keep = (pos < c).to(flat.dtype)
        slot = (idx * c + pos.clamp(0, c - 1).long()) * keep.long() + (k * c) * (1 - keep.long())

        # (M, K*C): row t is one-hot on the slot token t occupies. The overflow column is sliced
        # off, so a dropped token has an all-zero row and contributes nothing in either direction.
        a = F.one_hot(slot, k * c + 1)[:, :k * c].to(flat.dtype)
        xin = (a.t() @ flat).reshape(k, c, self.d)
        h = F.gelu(torch.bmm(xin, self.w1) + self.b1)
        y = (torch.bmm(h, self.w2) + self.b2).reshape(k * c, self.d)
        return (a @ y).reshape(x.shape)


class NonZeroMoE(_Experts):
    """The data-dependent formulation a plugin would exist to support."""

    def forward(self, x):
        flat = x.reshape(-1, self.d)
        idx = self.gate(flat).argmax(-1)
        out = torch.zeros_like(flat)
        for i in range(self.k):
            sel = (idx == i).nonzero(as_tuple=True)[0]
            xi = flat.index_select(0, sel)
            h = F.gelu(xi @ self.w1[i] + self.b1[i])
            out = out.index_copy(0, sel, h @ self.w2[i] + self.b2[i])
        return out.reshape(x.shape)


# ------------------------------------------------------------------ export + build

def export(mod, x, path, opset=17):
    mod.eval()
    torch.onnx.export(mod, (x,), path, input_names=["x"], output_names=["y"],
                      opset_version=opset, dynamo=False)
    return path


def _trtexec():
    for c in (os.environ.get("TRTEXEC"), "/usr/src/tensorrt/bin/trtexec",
              shutil.which("trtexec")):
        if c and os.path.exists(c):
            return c
    return None


TRTEXEC = _trtexec()


def build(onnx, precision, workspace=2048):
    """Returns (ok, median_gpu_ms, message). Latency is trtexec's median GPU compute time."""
    if TRTEXEC is None:
        return False, None, "trtexec not found (set $TRTEXEC)"
    flags = {"fp16": ["--fp16"], "int8": ["--int8", "--fp16"], "fp32": []}[precision]
    cmd = [TRTEXEC, f"--onnx={onnx}", f"--memPoolSize=workspace:{workspace}",
           "--avgRuns=100", "--iterations=200", "--noDataTransfers", "--useSpinWait"] + flags
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return False, None, "trtexec timed out after 1800 s"
    log = p.stdout + p.stderr
    if p.returncode != 0:
        err = [l for l in log.splitlines() if "rror" in l or "not supported" in l.lower()]
        return False, None, " | ".join(err[-3:])[:300] or f"exit {p.returncode}"
    mm = re.search(r"GPU Compute Time:.*?median = ([\d.]+) ms", log)
    return True, (float(mm.group(1)) if mm else None), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cap-factors", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--precisions", nargs="+", default=["fp16", "int8"])
    ap.add_argument("--work", default="/home/evc/merlin/trt_moe")
    ap.add_argument("--out", default="/home/evc/merlin/trt_ffn_moe.json")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    torch.manual_seed(0)

    x = torch.randn(1, N_TOK, D)
    variants = [("dense", DenseMLP()), ("masked", MaskedMoE(args.k))]
    for cf in args.cap_factors:
        variants.append((f"capacity_cf{cf}", CapacityMoE(args.k, cap_factor=cf)))
        variants.append((f"caponehot_cf{cf}", CapacityOneHotMoE(args.k, cap_factor=cf)))
    variants.append(("nonzero", NonZeroMoE(args.k)))

    # Every routed variant reads ONE set of expert weights, so a latency difference between them
    # is dispatch and nothing else, and `capacity` can be checked against the reference top-1.
    ref = NonZeroMoE(args.k)
    for _, mod in variants:
        if isinstance(mod, _Experts):
            mod.load_state_dict(ref.state_dict())
    with torch.no_grad():
        ref_y = ref(x)

    rows = []
    for name, mod in variants:
        rec = {"variant": name, "k": args.k if name != "dense" else 1}
        if isinstance(mod, CapacityMoE):
            rec["capacity"] = mod.c
            rec["slots"] = mod.k * mod.c
            rec["drop_frac_untrained_gate"] = mod.drop_frac(x)
            # Exact whenever nothing is dropped, so this separates "capacity too small" from
            # "dispatch is wrong". Drop here is measured on an UNTRAINED gate, which routes
            # near-uniformly; a trained router does not, and the capacity factor a real
            # deployment needs is k * max_share of its busiest expert.
            with torch.no_grad():
                rec["rel_vs_top1"] = float((mod(x) - ref_y).abs().max() / ref_y.abs().max())
            print(f"{name:>16}  C={mod.c} slots={mod.k * mod.c} "
                  f"drop={rec['drop_frac_untrained_gate']:.4f} "
                  f"rel-vs-top1={rec['rel_vs_top1']:.2e}", flush=True)
        onnx = os.path.join(args.work, f"{name}.onnx")
        try:
            export(mod, x, onnx)
            rec["onnx"] = "ok"
        except Exception as e:
            rec["onnx"] = f"{type(e).__name__}: {str(e)[:200]}"
            rows.append(rec)
            print(f"{name:>16}  ONNX EXPORT FAILED  {rec['onnx']}", flush=True)
            continue
        for prec in args.precisions:
            ok, ms, msg = build(onnx, prec)
            rec[prec] = {"built": ok, "ms": ms, "msg": msg}
            print(f"{name:>16}  {prec:>5}  "
                  + (f"{ms:8.3f} ms" if ok and ms else f"BUILD FAILED  {msg}"), flush=True)
        rows.append(rec)

    json.dump({"k": args.k, "tokens": N_TOK, "d": D, "dh": DH, "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)

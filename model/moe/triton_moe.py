#!/usr/bin/env python3
"""Fused grouped-GEMM top-1 MoE for the AAT FFN site, in Triton.

The measured problem: bench_moe_aat.py's MoETop1 does gather -> per-expert GEMM -> scatter in a
Python loop, so K experts cost K kernel launches over tiny badly-shaped matmuls. At K=64 that
path measured 32.3 ms/block against a 6.7 ms bandwidth roofline (90 GB/s measured, fp16).

This replaces the loop with two grouped GEMMs. Tokens are sorted by expert and padded to a
multiple of BLOCK_M, so every BLOCK_M-row tile belongs to exactly one expert and one kernel
launch covers all of them. The intermediate stays in sorted layout, so the whole FFN costs one
gather (first GEMM reads x through the sort) and one scatter (second GEMM writes out through
it), not two of each.

Semantics match MoETop1 exactly: hard top-1 by argmax with no gate weighting, exact erf GELU,
bias on both layers.
"""
import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------ token sorting

def padded_capacity(m, num_experts, block_m):
    """Host-side upper bound on padded slots: each expert wastes at most block_m-1."""
    return -(-(m + num_experts * (block_m - 1)) // block_m) * block_m


def align_block_size(expert_idx, num_experts, block_m, n_slots):
    """Group tokens by expert, padding each group to a multiple of block_m.

    Returns (sorted_tokens, block_experts):
      sorted_tokens  (n_slots,)   token id per slot, or M for a padding slot
      block_experts  (n_blocks,)  which expert owns each BLOCK_M tile

    Deliberately free of any device->host sync -- and that is load-bearing, not a nicety. Reading
    the true padded total back to the host to size the launch costs a full pipeline stall every
    forward, which measured ~1.9 ms and dominated the kernel it was feeding. Instead the caller
    sizes buffers to the worst case and the kernel skips all-padding tiles. Sync-freedom is ALSO
    what makes this CUDA-graph capturable, which is worth 7.8x on the routing path (Orin Nano,
    2026-08-11): the chain below is ~8 tiny kernels over 128- and 729-element tensors, so on an
    edge GPU it is pure launch latency, 60-120 us apiece.
    """
    m = expert_idx.numel()
    dev = expert_idx.device

    order = torch.argsort(expert_idx, stable=True)          # token ids, grouped by expert
    grouped = expert_idx[order].long()
    # scatter_add, NOT bincount. torch.bincount syncs device->host to size its output, which broke
    # the sync-free promise this docstring makes and made CUDA-graph capture fail outright
    # ("Cannot copy between CPU and CUDA tensors during capture"). Measured on an Orin Nano:
    # scatter_add is 1.21x faster eager, and capture takes this whole function 1.553 -> 0.099 ms.
    li = expert_idx.long()
    counts = torch.zeros(num_experts, device=dev, dtype=torch.int64).scatter_add_(
        0, li, torch.ones_like(li))
    padded = ((counts + block_m - 1) // block_m) * block_m

    dst_start = torch.cumsum(padded, 0) - padded            # each expert's slot region
    src_start = torch.cumsum(counts, 0) - counts            # each expert's run inside `order`
    rank = torch.arange(m, device=dev) - src_start[grouped]
    dst = dst_start[grouped] + rank

    sorted_tokens = torch.full((n_slots,), m, dtype=torch.int32, device=dev)
    sorted_tokens[dst] = order.to(torch.int32)

    # block -> owning expert, without repeat_interleave (whose output length is device-side)
    blocks = torch.arange(n_slots // block_m, device=dev)
    block_experts = (torch.searchsorted(dst_start // block_m, blocks, right=True) - 1
                     ).to(torch.int32)
    return sorted_tokens, block_experts


# ------------------------------------------------------------------ kernel

def _configs():
    return [
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
        for bn in (32, 64, 128)
        for bk in (32, 64)
        for w, s in ((4, 3), (4, 4), (8, 3))
    ]


@triton.autotune(configs=_configs(), key=["N", "K"])
@triton.jit
def _grouped_gemm(
    a_ptr, w_ptr, b_ptr, c_ptr,
    sorted_ptr, expert_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_we, stride_wk, stride_wn,
    stride_cm, stride_cn,
    stride_be,
    GATHER_A: tl.constexpr, SCATTER_C: tl.constexpr, ACT_GELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid // n_tiles
    pid_n = pid % n_tiles

    slots = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_ptr + slots)
    valid = tok < M                                  # padding slots carry the sentinel M
    if tl.min(tok) >= M:                             # tile is entirely padding, nothing to do
        return
    e = tl.load(expert_ptr + pid_m)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # first GEMM reads x through the sort; second reads the intermediate, already sorted
    a_row = tok if GATHER_A else slots
    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + e * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        krem = K - k0 * BLOCK_K
        a = tl.load(a_ptrs, mask=valid[:, None] & (offs_k[None, :] < krem), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < krem) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, w, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    acc += tl.load(b_ptr + e * stride_be + offs_n, mask=offs_n < N, other=0.0)[None, :].to(tl.float32)
    if ACT_GELU:
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865475))

    c_row = tok if SCATTER_C else slots
    c_ptrs = c_ptr + c_row[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty),
             mask=valid[:, None] & (offs_n[None, :] < N))


# ------------------------------------------------------------------ module

class FusedMoETop1(torch.nn.Module):
    """Top-1 MoE FFN (D -> DH -> D) over K experts, two grouped GEMMs.

    block_m wants to be LARGE, which is the opposite of what padding waste suggests. Every
    M-tile re-reads its expert's full weight slice, so weight traffic tracks the block count,
    n_blocks ~ M/block_m + K, and the padding a big tile wastes costs only compute, which is
    not the binding resource here. Measured across K=4..64, block_m=16 was worst everywhere and
    64 was best or within noise of best; 32 wins only at K=4 where there are few experts.
    """

    def __init__(self, k, d, dh, block_m=None, device="cuda", dtype=torch.float16):
        super().__init__()
        self.k, self.d, self.dh = k, d, dh
        self.block_m = block_m if block_m is not None else (32 if k <= 4 else 64)
        self.gate = torch.nn.Linear(d, k).to(device=device, dtype=dtype)
        # (E, in, out) so the reduction dim is the middle one, matching the kernel's strides
        self.w1 = torch.nn.Parameter(torch.empty(k, d, dh, device=device, dtype=dtype))
        self.w2 = torch.nn.Parameter(torch.empty(k, dh, d, device=device, dtype=dtype))
        self.b1 = torch.nn.Parameter(torch.empty(k, dh, device=device, dtype=dtype))
        self.b2 = torch.nn.Parameter(torch.empty(k, d, device=device, dtype=dtype))
        for t, fan in ((self.w1, d), (self.w2, dh)):
            torch.nn.init.uniform_(t, -fan ** -0.5, fan ** -0.5)
        for t, fan in ((self.b1, d), (self.b2, dh)):
            torch.nn.init.uniform_(t, -fan ** -0.5, fan ** -0.5)

    def load_from(self, experts, gate):
        """Copy weights out of a list of nn.Linear-based DenseFFNs, for parity checking."""
        with torch.no_grad():
            self.gate.weight.copy_(gate.weight); self.gate.bias.copy_(gate.bias)
            for i, e in enumerate(experts):
                self.w1[i].copy_(e.w1.weight.t()); self.b1[i].copy_(e.w1.bias)
                self.w2[i].copy_(e.w2.weight.t()); self.b2[i].copy_(e.w2.bias)
        return self

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.d)
        m = flat.shape[0]

        idx = self.gate(flat).argmax(-1).to(torch.int32)
        n_slots = padded_capacity(m, self.k, self.block_m)
        sorted_tokens, block_experts = align_block_size(idx, self.k, self.block_m, n_slots)
        n_blocks = n_slots // self.block_m

        mid = torch.empty((n_slots, self.dh), device=flat.device, dtype=flat.dtype)
        out = torch.empty_like(flat)

        grid = lambda meta: (n_blocks * triton.cdiv(self.dh, meta["BLOCK_N"]),)
        _grouped_gemm[grid](
            flat, self.w1, self.b1, mid,
            sorted_tokens, block_experts,
            m, self.dh, self.d,
            flat.stride(0), flat.stride(1),
            self.w1.stride(0), self.w1.stride(1), self.w1.stride(2),
            mid.stride(0), mid.stride(1),
            self.b1.stride(0),
            GATHER_A=True, SCATTER_C=False, ACT_GELU=True, BLOCK_M=self.block_m,
        )
        grid = lambda meta: (n_blocks * triton.cdiv(self.d, meta["BLOCK_N"]),)
        _grouped_gemm[grid](
            mid, self.w2, self.b2, out,
            sorted_tokens, block_experts,
            m, self.d, self.dh,
            mid.stride(0), mid.stride(1),
            self.w2.stride(0), self.w2.stride(1), self.w2.stride(2),
            out.stride(0), out.stride(1),
            self.b2.stride(0),
            GATHER_A=False, SCATTER_C=True, ACT_GELU=False, BLOCK_M=self.block_m,
        )
        return out.reshape(shape)


# ------------------------------------------------------------------ fused butterfly

@triton.jit
def _butterfly_kernel(X, ANG, IDX, Y, M, DIM,
                      stride_xm, stride_xd, stride_am, stride_al, stride_ad,
                      N_L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_G: tl.constexpr):
    """n_l layers of per-expert Givens rotations, applied entirely in registers.

    The eager version materialises cos, sin, four products and a stack PER LAYER, so it makes ~8
    passes over the activation and costs 3.4 ms/block on an Orin Nano -- 4x the dense FFN it
    decorates -- and it does NOT shrink under a CUDA graph, because it is memory traffic, not
    launch latency. Here the activation is read once and written once.

    Layer l pairs element i with i+2^l, so every pairing across all n_l layers stays inside an
    aligned group of G = 2^n_l contiguous elements, held in explicit registers v0..v3. Written
    flat on purpose: Triton rewrites in-kernel Python lists into tuples (so `.append` fails to
    trace) and rejects nested `def`s, so neither a list nor a helper survives tracing.
    n_l <= 2 covers the deployed configuration.
    """
    pm = tl.program_id(0)
    pg = tl.program_id(1)
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    d0 = (pg * BLOCK_G + tl.arange(0, BLOCK_G)) * (1 << N_L)
    mm = rm < M
    e = tl.load(IDX + rm, mask=mm, other=0).to(tl.int32)
    xb = X + rm[:, None] * stride_xm
    ab = ANG + e[:, None] * stride_am

    d = d0[None, :]
    m0 = mm[:, None] & (d < DIM)
    v0 = tl.load(xb + d * stride_xd, mask=m0, other=0.0).to(tl.float32)
    m1 = mm[:, None] & (d + 1 < DIM)
    v1 = tl.load(xb + (d + 1) * stride_xd, mask=m1, other=0.0).to(tl.float32)

    # layer 0, stride 1: angle index of element i is i//2
    a = tl.load(ab + (d // 2) * stride_ad, mask=m0, other=0.0).to(tl.float32)
    c, sn = tl.cos(a), tl.sin(a)
    v0, v1 = v0 * c - v1 * sn, v0 * sn + v1 * c

    if N_L > 1:
        m2 = mm[:, None] & (d + 2 < DIM)
        m3 = mm[:, None] & (d + 3 < DIM)
        v2 = tl.load(xb + (d + 2) * stride_xd, mask=m2, other=0.0).to(tl.float32)
        v3 = tl.load(xb + (d + 3) * stride_xd, mask=m3, other=0.0).to(tl.float32)
        a = tl.load(ab + ((d + 2) // 2) * stride_ad, mask=m2, other=0.0).to(tl.float32)
        c, sn = tl.cos(a), tl.sin(a)
        v2, v3 = v2 * c - v3 * sn, v2 * sn + v3 * c
        # layer 1, stride 2: angle index of element i is (i//4)*2 + (i%2)
        a = tl.load(ab + stride_al + ((d // 4) * 2 + (d % 2)) * stride_ad,
                    mask=m0, other=0.0).to(tl.float32)
        c, sn = tl.cos(a), tl.sin(a)
        v0, v2 = v0 * c - v2 * sn, v0 * sn + v2 * c
        a = tl.load(ab + stride_al + (((d + 1) // 4) * 2 + ((d + 1) % 2)) * stride_ad,
                    mask=m1, other=0.0).to(tl.float32)
        c, sn = tl.cos(a), tl.sin(a)
        v1, v3 = v1 * c - v3 * sn, v1 * sn + v3 * c
        tl.store(Y + rm[:, None] * stride_xm + (d + 2) * stride_xd,
                 v2.to(Y.dtype.element_ty), mask=m2)
        tl.store(Y + rm[:, None] * stride_xm + (d + 3) * stride_xd,
                 v3.to(Y.dtype.element_ty), mask=m3)

    tl.store(Y + rm[:, None] * stride_xm + d * stride_xd, v0.to(Y.dtype.element_ty), mask=m0)
    tl.store(Y + rm[:, None] * stride_xm + (d + 1) * stride_xd, v1.to(Y.dtype.element_ty), mask=m1)


def fused_butterfly(x, angle, idx, block_m=32, block_g=32):
    """x (M, dim) -> rotated (M, dim), using angle (K, n_l, dim/2) selected by idx (M,)."""
    # CONTIGUOUS, always. OREResidual applies the rotation to a SLICE -- xi[:, :rot_in_dim] of a
    # d_in-wide tensor -- so x.stride(0) is 768 while empty_like() hands back a contiguous 512-wide
    # buffer with stride 512. The kernel used X's strides for both, so it wrote past the end of Y:
    # an illegal memory access that only appears on non-contiguous input, which is why contiguous
    # unit tests passed on the Jetson and the first real run died in router calibration.
    x = x.contiguous()
    m, dim = x.shape
    n_l = angle.shape[1]
    if not 1 <= n_l <= 2:
        raise ValueError(f"fused_butterfly supports n_l in 1..2, got {n_l}")
    g = 1 << n_l
    if dim % g:
        raise ValueError(f"dim {dim} must be a multiple of the group size {g}")
    y = torch.empty_like(x)
    assert y.stride() == x.stride(), "kernel indexes Y with X's strides"
    grid = (-(-m // block_m), -(-(dim // g) // block_g))
    _butterfly_kernel[grid](x, angle, idx.to(torch.int32), y, m, dim,
                            x.stride(0), x.stride(1),
                            angle.stride(0), angle.stride(1), angle.stride(2),
                            N_L=n_l, BLOCK_M=block_m, BLOCK_G=block_g)
    return y


@triton.jit
def _butterfly_sorted_kernel(X, ANG, STOK, BE, Y, M, DIM,
                             stride_xm, stride_xd, stride_am, stride_al, stride_ad,
                             N_L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_G: tl.constexpr):
    """Butterfly over tokens ALREADY GROUPED BY EXPERT, so the angle row is loaded once.

    The unsorted kernel gathers each token's own expert row, so a 32-token program can touch 32
    different rows and the reads scatter -- measured 0.510 ms/block against a 0.187 ms roofline,
    2.7x off, with the gather as the only plausible cause. Here every program covers ONE expert
    (BE gives the block's expert, STOK the token ids), so the angles are a single contiguous read
    broadcast across the block. Reuses the grouping align_block_size already computes for the
    grouped GEMM, so the sort costs nothing extra.
    """
    pm = tl.program_id(0)
    pg = tl.program_id(1)
    e = tl.load(BE + pm).to(tl.int32)                     # ONE expert for this whole block
    slot = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(STOK + slot).to(tl.int32)
    live = tok < M
    d0 = (pg * BLOCK_G + tl.arange(0, BLOCK_G)) * (1 << N_L)
    d = d0[None, :]
    xb = X + tok[:, None] * stride_xm
    ab = ANG + e * stride_am                              # scalar base: one row, not a gather

    m0 = live[:, None] & (d < DIM)
    m1 = live[:, None] & (d + 1 < DIM)
    v0 = tl.load(xb + d * stride_xd, mask=m0, other=0.0).to(tl.float32)
    v1 = tl.load(xb + (d + 1) * stride_xd, mask=m1, other=0.0).to(tl.float32)
    a = tl.load(ab + (d // 2) * stride_ad, mask=(d < DIM), other=0.0).to(tl.float32)
    c, sn = tl.cos(a), tl.sin(a)
    v0, v1 = v0 * c - v1 * sn, v0 * sn + v1 * c

    if N_L > 1:
        m2 = live[:, None] & (d + 2 < DIM)
        m3 = live[:, None] & (d + 3 < DIM)
        v2 = tl.load(xb + (d + 2) * stride_xd, mask=m2, other=0.0).to(tl.float32)
        v3 = tl.load(xb + (d + 3) * stride_xd, mask=m3, other=0.0).to(tl.float32)
        a = tl.load(ab + ((d + 2) // 2) * stride_ad, mask=(d + 2 < DIM), other=0.0).to(tl.float32)
        c, sn = tl.cos(a), tl.sin(a)
        v2, v3 = v2 * c - v3 * sn, v2 * sn + v3 * c
        a = tl.load(ab + stride_al + ((d // 4) * 2 + (d % 2)) * stride_ad,
                    mask=(d < DIM), other=0.0).to(tl.float32)
        c, sn = tl.cos(a), tl.sin(a)
        v0, v2 = v0 * c - v2 * sn, v0 * sn + v2 * c
        a = tl.load(ab + stride_al + (((d + 1) // 4) * 2 + ((d + 1) % 2)) * stride_ad,
                    mask=(d + 1 < DIM), other=0.0).to(tl.float32)
        c, sn = tl.cos(a), tl.sin(a)
        v1, v3 = v1 * c - v3 * sn, v1 * sn + v3 * c
        tl.store(Y + tok[:, None] * stride_xm + (d + 2) * stride_xd,
                 v2.to(Y.dtype.element_ty), mask=m2)
        tl.store(Y + tok[:, None] * stride_xm + (d + 3) * stride_xd,
                 v3.to(Y.dtype.element_ty), mask=m3)

    tl.store(Y + tok[:, None] * stride_xm + d * stride_xd, v0.to(Y.dtype.element_ty), mask=m0)
    tl.store(Y + tok[:, None] * stride_xm + (d + 1) * stride_xd, v1.to(Y.dtype.element_ty), mask=m1)


def fused_butterfly_sorted(x, angle, sorted_tok, block_experts, block_m, block_g=32):
    """Butterfly using the expert grouping align_block_size already produced."""
    x = x.contiguous()
    m, dim = x.shape
    n_l = angle.shape[1]
    if not 1 <= n_l <= 2:
        raise ValueError(f"supports n_l in 1..2, got {n_l}")
    y = torch.empty_like(x)
    grid = (sorted_tok.numel() // block_m, -(-(dim // (1 << n_l)) // block_g))
    _butterfly_sorted_kernel[grid](x, angle, sorted_tok, block_experts, y, m, dim,
                                   x.stride(0), x.stride(1),
                                   angle.stride(0), angle.stride(1), angle.stride(2),
                                   N_L=n_l, BLOCK_M=block_m, BLOCK_G=block_g)
    return y


# ------------------------------------------------------------------ fused low-rank residual

def _lr_configs():
    return [
        triton.Config({"BLOCK_K": bk, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bk in (64, 128)
        for bn in (64, 128, 256)
        for w, s in ((4, 3), (8, 3), (8, 4))
    ]


@triton.autotune(configs=_lr_configs(), key=["D_IN", "R", "D_OUT", "EPILOGUE"])
@triton.jit
def _lowrank_kernel(X, A, B, Y, STOK, BE, H, ANG, PSEL, BIAS, M, D_IN, R, D_OUT, ROT,
                    stride_xm, stride_xd, stride_ae, stride_ad, stride_ar,
                    stride_be, stride_br, stride_bn, stride_ym, stride_yn,
                    stride_hm, stride_hn, stride_am, stride_al, stride_ad2,
                    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr,
                    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
                    EPILOGUE: tl.constexpr):
    """y = (x @ A_e) @ B_e for one expert-homogeneous tile, both GEMMs in one launch.

    Run as two grouped GEMMs the rank-r intermediate goes out to memory and comes straight back,
    and the first GEMM is badly shaped: N = r = 128 is a single output tile, so each program does
    almost no work and the A-side measured 0.200 ms flat across every block size, 6.5x off its
    roofline and unresponsive to tiling. Here z stays in registers while the program walks the
    output tiles, so the A-side work is amortised over the B-side's traffic instead of paying its
    own launch. Requires r <= BLOCK_R, which is why the host falls back for large ranks.
    """
    pm = tl.program_id(0)
    slot = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(STOK + slot)
    live = tok < M
    if tl.min(tok) >= M:
        return
    e = tl.load(BE + pm)

    offs_r = tl.arange(0, BLOCK_R)
    rmask = offs_r < R
    offs_k = tl.arange(0, BLOCK_K)
    xp = X + tok[:, None] * stride_xm + offs_k[None, :] * stride_xd
    ap = A + e * stride_ae + offs_k[:, None] * stride_ad + offs_r[None, :] * stride_ar
    z = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(D_IN, BLOCK_K)):
        krem = D_IN - k0 * BLOCK_K
        a = tl.load(xp, mask=live[:, None] & (offs_k[None, :] < krem), other=0.0)
        w = tl.load(ap, mask=(offs_k[:, None] < krem) & rmask[None, :], other=0.0)
        z = tl.dot(a, w, z)
        xp += BLOCK_K * stride_xd
        ap += BLOCK_K * stride_ad
    zh = z.to(Y.dtype.element_ty)

    bp = B + e * stride_be + offs_r[:, None] * stride_br
    if EPILOGUE:
        # Phi mixes output columns inside aligned groups of 4, which a (BLOCK_M, BLOCK_N) register
        # tile cannot shuffle. Splitting the output tile into the four residue classes mod 4 makes
        # every pairing an elementwise op between two accumulators instead: load B (and the
        # substrate output H) in four strided slices and keep four accumulators. That is what lets
        # the rotation, the bias, the p_sel scaling and the residual add ride the GEMM's own
        # epilogue rather than costing three more passes over a (729, 3072) tensor.
        p = tl.load(PSEL + tok, mask=live, other=0.0).to(tl.float32)[:, None]
        hb = H + tok[:, None] * stride_hm
        yb = Y + tok[:, None] * stride_ym
        ab = ANG + e * stride_am
        for n0 in range(0, tl.cdiv(D_OUT, BLOCK_N)):
            on = n0 * BLOCK_N + tl.arange(0, BLOCK_N // 4) * 4
            n0m, n1m = on < D_OUT, on + 1 < D_OUT
            n2m, n3m = on + 2 < D_OUT, on + 3 < D_OUT
            u0 = tl.dot(zh, tl.load(bp + on[None, :] * stride_bn,
                                    mask=rmask[:, None] & n0m[None, :], other=0.0))
            u1 = tl.dot(zh, tl.load(bp + (on + 1)[None, :] * stride_bn,
                                    mask=rmask[:, None] & n1m[None, :], other=0.0))
            u2 = tl.dot(zh, tl.load(bp + (on + 2)[None, :] * stride_bn,
                                    mask=rmask[:, None] & n2m[None, :], other=0.0))
            u3 = tl.dot(zh, tl.load(bp + (on + 3)[None, :] * stride_bn,
                                    mask=rmask[:, None] & n3m[None, :], other=0.0))
            d = on[None, :]
            v0 = tl.load(hb + d * stride_hn, mask=live[:, None] & n0m[None, :],
                         other=0.0).to(tl.float32)
            v1 = tl.load(hb + (d + 1) * stride_hn, mask=live[:, None] & n1m[None, :],
                         other=0.0).to(tl.float32)
            v2 = tl.load(hb + (d + 2) * stride_hn, mask=live[:, None] & n2m[None, :],
                         other=0.0).to(tl.float32)
            v3 = tl.load(hb + (d + 3) * stride_hn, mask=live[:, None] & n3m[None, :],
                         other=0.0).to(tl.float32)
            # angle 0 past ROT -> cos 1, sin 0, so the un-rotated tail rides the same code
            aa = tl.load(ab + (d // 2) * stride_ad2, mask=(d < ROT), other=0.0).to(tl.float32)
            c, sn = tl.cos(aa), tl.sin(aa)
            v0, v1 = v0 * c - v1 * sn, v0 * sn + v1 * c
            aa = tl.load(ab + ((d + 2) // 2) * stride_ad2, mask=(d + 2 < ROT),
                         other=0.0).to(tl.float32)
            c, sn = tl.cos(aa), tl.sin(aa)
            v2, v3 = v2 * c - v3 * sn, v2 * sn + v3 * c
            aa = tl.load(ab + stride_al + ((d // 4) * 2 + (d % 2)) * stride_ad2,
                         mask=(d < ROT), other=0.0).to(tl.float32)
            c, sn = tl.cos(aa), tl.sin(aa)
            v0, v2 = v0 * c - v2 * sn, v0 * sn + v2 * c
            aa = tl.load(ab + stride_al + (((d + 1) // 4) * 2 + ((d + 1) % 2)) * stride_ad2,
                         mask=(d + 1 < ROT), other=0.0).to(tl.float32)
            c, sn = tl.cos(aa), tl.sin(aa)
            v1, v3 = v1 * c - v3 * sn, v1 * sn + v3 * c
            v0 += p * u0 + tl.load(BIAS + d, mask=n0m[None, :], other=0.0).to(tl.float32)
            v1 += p * u1 + tl.load(BIAS + d + 1, mask=n1m[None, :], other=0.0).to(tl.float32)
            v2 += p * u2 + tl.load(BIAS + d + 2, mask=n2m[None, :], other=0.0).to(tl.float32)
            v3 += p * u3 + tl.load(BIAS + d + 3, mask=n3m[None, :], other=0.0).to(tl.float32)
            tl.store(yb + d * stride_yn, v0.to(Y.dtype.element_ty),
                     mask=live[:, None] & n0m[None, :])
            tl.store(yb + (d + 1) * stride_yn, v1.to(Y.dtype.element_ty),
                     mask=live[:, None] & n1m[None, :])
            tl.store(yb + (d + 2) * stride_yn, v2.to(Y.dtype.element_ty),
                     mask=live[:, None] & n2m[None, :])
            tl.store(yb + (d + 3) * stride_yn, v3.to(Y.dtype.element_ty),
                     mask=live[:, None] & n3m[None, :])
    else:
        for n0 in range(0, tl.cdiv(D_OUT, BLOCK_N)):
            offs_n = n0 * BLOCK_N + tl.arange(0, BLOCK_N)
            nmask = offs_n < D_OUT
            b = tl.load(bp + offs_n[None, :] * stride_bn,
                        mask=rmask[:, None] & nmask[None, :], other=0.0)
            acc = tl.dot(zh, b)
            tl.store(Y + tok[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                     acc.to(Y.dtype.element_ty), mask=live[:, None] & nmask[None, :])


def fused_lowrank(x, a, b, sorted_tok, block_experts, block_m, out=None,
                  h=None, angle=None, rot_dim=0, psel=None, bias=None):
    """Top-1 low-rank residual B_e A_e x, one launch. Returns None if the rank does not fit.

    Pass h/angle/psel/bias to fold the whole oriented epilogue in as well, computing
    Phi_e(h) + bias + p_sel * B_e A_e x and never writing the residual to memory at all.
    """
    m, d_in = x.shape
    k, _, r = a.shape
    d_out = b.shape[2]
    block_r = triton.next_power_of_2(r)
    if block_r > 256 or r < 16:
        return None
    epi = h is not None
    if epi and (angle.shape[1] != 2 or d_out % 4 or bias is None):
        return None                                  # the four-accumulator split assumes n_l=2
    y = torch.empty((m, d_out), device=x.device, dtype=x.dtype) if out is None else out
    _lowrank_kernel[(sorted_tok.numel() // block_m,)](
        x, a, b, y, sorted_tok, block_experts, h, angle, psel, bias, m, d_in, r, d_out, rot_dim,
        x.stride(0), x.stride(1), a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2), y.stride(0), y.stride(1),
        h.stride(0) if epi else 0, h.stride(1) if epi else 0,
        angle.stride(0) if epi else 0, angle.stride(1) if epi else 0,
        angle.stride(2) if epi else 0,
        BLOCK_M=block_m, BLOCK_R=block_r, EPILOGUE=epi)
    return y


# ------------------------------------------------------------------ fused routing

@triton.jit
def _route_gate_kernel(RF, GW, GB, IDX, PSEL, HIST, M, DR, K,
                       stride_rm, stride_rd, stride_gd, stride_gk,
                       BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
    """Gate GEMM, softmax denominator, argmax and expert histogram, in ONE launch.

    The eager equivalent is a Linear, a softmax, an argmax, a gather and a scatter over a
    (729, K) logit table -- five launches over tensors small enough that each is pure overhead.
    Only the winner's probability is ever used, and for the argmax winner that is exactly
    1/sum(exp(l - max)), so the softmax never has to be materialised.
    """
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = rm < M
    ok = tl.arange(0, BLOCK_K)
    km = ok < K
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for d0 in range(0, tl.cdiv(DR, BLOCK_D)):
        od = d0 * BLOCK_D + tl.arange(0, BLOCK_D)
        dm = od < DR
        a = tl.load(RF + rm[:, None] * stride_rm + od[None, :] * stride_rd,
                    mask=mm[:, None] & dm[None, :], other=0.0)
        w = tl.load(GW + od[:, None] * stride_gd + ok[None, :] * stride_gk,
                    mask=dm[:, None] & km[None, :], other=0.0)
        acc = tl.dot(a, w, acc)
    acc += tl.load(GB + ok, mask=km, other=0.0)[None, :].to(tl.float32)
    acc = tl.where(km[None, :], acc, float("-inf"))
    idx = tl.argmax(acc, 1).to(tl.int32)
    psel = 1.0 / tl.sum(tl.exp(acc - tl.max(acc, 1)[:, None]), 1)
    tl.store(IDX + rm, idx, mask=mm)
    tl.store(PSEL + rm, psel.to(PSEL.dtype.element_ty), mask=mm)
    tl.atomic_add(HIST + idx, tl.full((BLOCK_M,), 1, tl.int32), mask=mm)


@triton.jit
def _route_prefix_kernel(HIST, CURSOR, BE, STOK, K, NB, NS, M,
                         BM: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_B: tl.constexpr):
    """One program: pad each expert's count up to BM, prefix-sum, publish the block->expert map.

    CURSOR starts at each expert's first slot, so the placement kernel can claim slots with a
    plain atomic increment and never needs a rank computed on the host. HIST is zeroed here, on
    the read, so a frame costs no separate memset.
    """
    ok = tl.arange(0, BLOCK_K)
    km = ok < K
    c = tl.load(HIST + ok, mask=km, other=0)
    padded = ((c + BM - 1) // BM) * BM
    start = tl.cumsum(padded, 0) - padded
    tl.store(CURSOR + ok, start, mask=km)
    tl.store(HIST + ok, tl.zeros_like(c), mask=km)
    sb = start // BM
    for b0 in range(0, tl.cdiv(NB, BLOCK_B)):
        ob = b0 * BLOCK_B + tl.arange(0, BLOCK_B)
        # the LAST expert whose region starts at or before this block, matching
        # searchsorted(start // BM, blocks, right=True) - 1 for zero-width (unused) experts
        e = tl.sum(tl.where(km[None, :] & (sb[None, :] <= ob[:, None]), 1, 0), 1) - 1
        tl.store(BE + ob, e.to(tl.int32), mask=ob < NB)
    for s0 in range(0, tl.cdiv(NS, BLOCK_B)):
        os_ = s0 * BLOCK_B + tl.arange(0, BLOCK_B)
        tl.store(STOK + os_, tl.full((BLOCK_B,), M, tl.int32), mask=os_ < NS)


@triton.jit
def _route_place_kernel(IDX, CURSOR, STOK, M, BLOCK_M: tl.constexpr):
    """Claim a slot per token by atomic increment. Order within an expert is arbitrary, which
    is fine: results are scattered back by token id, so the block output is unaffected."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = rm < M
    e = tl.load(IDX + rm, mask=mm, other=0)
    slot = tl.atomic_add(CURSOR + e, tl.full((BLOCK_M,), 1, tl.int32), mask=mm)
    tl.store(STOK + slot, rm.to(tl.int32), mask=mm)


def fused_route(rf, gate_wt, gate_b, buf, block_m):
    """gate -> idx, p_sel, sorted slots and block->expert map, in three launches.

    Replaces the ~13-launch eager chain (Linear, softmax, argmax, gather, argsort, scatter_add,
    two cumsums, arange, searchsorted, ...). On an Orin Nano each of those is 60-120 us eager and
    still ~18 us inside a CUDA graph, over tensors of 128 and 729 elements: pure dispatch cost.

    `buf` holds the persistent tensors (idx, psel, hist, cursor, stok, be); reusing them keeps
    the path allocation-free and CUDA-graph capturable. `gate_wt` is the gate weight TRANSPOSED
    to (d_route, K) so the kernel reads it coalesced.
    """
    m, dr = rf.shape
    k = gate_b.numel()
    bk = max(16, triton.next_power_of_2(k))
    _route_gate_kernel[(triton.cdiv(m, 64),)](
        rf, gate_wt, gate_b, buf["idx"], buf["psel"], buf["hist"], m, dr, k,
        rf.stride(0), rf.stride(1), gate_wt.stride(0), gate_wt.stride(1),
        BLOCK_M=64, BLOCK_K=bk, BLOCK_D=64, num_warps=4)
    ns = buf["stok"].numel()
    _route_prefix_kernel[(1,)](
        buf["hist"], buf["cursor"], buf["be"], buf["stok"], k, buf["be"].numel(), ns, m,
        BM=block_m, BLOCK_K=bk, BLOCK_B=256, num_warps=4)
    _route_place_kernel[(triton.cdiv(m, 128),)](
        buf["idx"], buf["cursor"], buf["stok"], m, BLOCK_M=128, num_warps=4)
    return buf["idx"], buf["psel"], buf["stok"], buf["be"]


# ------------------------------------------------------------------ fused rotate + epilogue

@triton.jit
def _rot_full_kernel(X, ANG, STOK, BE, RES, PSEL, BIAS, Y, M, DIM, ROT,
                     stride_xm, stride_xd, stride_am, stride_al, stride_ad,
                     stride_rm, stride_rd,
                     N_L: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_G: tl.constexpr,
                     HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr, ACT_GELU: tl.constexpr):
    """Rotate the pow2 prefix of a FULL-WIDTH activation and fold the epilogue into the pass.

    Two things this removes. (1) `_oriented` rotated a slice and then `torch.cat`-ed the tail
    back on, so the activation was read and written twice to rotate part of it; here columns past
    ROT simply take a zero angle, which is the identity, and ride the same pass. (2) the bias
    add, the p_sel scaling, the residual add and the GELU were four more elementwise passes over
    a (729, 3072) tensor; folded in, they cost nothing beyond the residual's own read.

    Tokens arrive grouped by expert (STOK/BE from fused_route), so the angle row is one
    contiguous broadcast read per program rather than a per-token gather.
    """
    pm = tl.program_id(0)
    pg = tl.program_id(1)
    e = tl.load(BE + pm).to(tl.int32)
    slot = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(STOK + slot).to(tl.int32)
    live = tok < M
    if tl.min(tok) >= M:
        return
    d = ((pg * BLOCK_G + tl.arange(0, BLOCK_G)) * (1 << N_L))[None, :]
    xb = X + tok[:, None] * stride_xm
    ab = ANG + e * stride_am

    m0 = live[:, None] & (d < DIM)
    m1 = live[:, None] & (d + 1 < DIM)
    v0 = tl.load(xb + d * stride_xd, mask=m0, other=0.0).to(tl.float32)
    v1 = tl.load(xb + (d + 1) * stride_xd, mask=m1, other=0.0).to(tl.float32)
    # angle 0 outside the rotated prefix -> cos 1, sin 0 -> the tail passes through untouched
    a = tl.load(ab + (d // 2) * stride_ad, mask=(d < ROT), other=0.0).to(tl.float32)
    c, sn = tl.cos(a), tl.sin(a)
    v0, v1 = v0 * c - v1 * sn, v0 * sn + v1 * c

    if N_L > 1:
        m2 = live[:, None] & (d + 2 < DIM)
        m3 = live[:, None] & (d + 3 < DIM)
        v2 = tl.load(xb + (d + 2) * stride_xd, mask=m2, other=0.0).to(tl.float32)
        v3 = tl.load(xb + (d + 3) * stride_xd, mask=m3, other=0.0).to(tl.float32)
        a = tl.load(ab + ((d + 2) // 2) * stride_ad, mask=(d + 2 < ROT), other=0.0).to(tl.float32)
        c, sn = tl.cos(a), tl.sin(a)
        v2, v3 = v2 * c - v3 * sn, v2 * sn + v3 * c
        a = tl.load(ab + stride_al + ((d // 4) * 2 + (d % 2)) * stride_ad,
                    mask=(d < ROT), other=0.0).to(tl.float32)
        c, sn = tl.cos(a), tl.sin(a)
        v0, v2 = v0 * c - v2 * sn, v0 * sn + v2 * c
        a = tl.load(ab + stride_al + (((d + 1) // 4) * 2 + ((d + 1) % 2)) * stride_ad,
                    mask=(d + 1 < ROT), other=0.0).to(tl.float32)
        c, sn = tl.cos(a), tl.sin(a)
        v1, v3 = v1 * c - v3 * sn, v1 * sn + v3 * c

    if HAS_BIAS:
        v0 += tl.load(BIAS + d, mask=(d < DIM), other=0.0).to(tl.float32)
        v1 += tl.load(BIAS + d + 1, mask=(d + 1 < DIM), other=0.0).to(tl.float32)
        if N_L > 1:
            v2 += tl.load(BIAS + d + 2, mask=(d + 2 < DIM), other=0.0).to(tl.float32)
            v3 += tl.load(BIAS + d + 3, mask=(d + 3 < DIM), other=0.0).to(tl.float32)
    if HAS_RES:
        p = tl.load(PSEL + tok, mask=live, other=0.0).to(tl.float32)[:, None]
        rb = RES + tok[:, None] * stride_rm
        v0 += p * tl.load(rb + d * stride_rd, mask=m0, other=0.0).to(tl.float32)
        v1 += p * tl.load(rb + (d + 1) * stride_rd, mask=m1, other=0.0).to(tl.float32)
        if N_L > 1:
            v2 += p * tl.load(rb + (d + 2) * stride_rd, mask=m2, other=0.0).to(tl.float32)
            v3 += p * tl.load(rb + (d + 3) * stride_rd, mask=m3, other=0.0).to(tl.float32)
    if ACT_GELU:
        v0 = 0.5 * v0 * (1.0 + tl.math.erf(v0 * 0.7071067811865475))
        v1 = 0.5 * v1 * (1.0 + tl.math.erf(v1 * 0.7071067811865475))
        if N_L > 1:
            v2 = 0.5 * v2 * (1.0 + tl.math.erf(v2 * 0.7071067811865475))
            v3 = 0.5 * v3 * (1.0 + tl.math.erf(v3 * 0.7071067811865475))

    yb = Y + tok[:, None] * stride_xm
    tl.store(yb + d * stride_xd, v0.to(Y.dtype.element_ty), mask=m0)
    tl.store(yb + (d + 1) * stride_xd, v1.to(Y.dtype.element_ty), mask=m1)
    if N_L > 1:
        tl.store(yb + (d + 2) * stride_xd, v2.to(Y.dtype.element_ty), mask=m2)
        tl.store(yb + (d + 3) * stride_xd, v3.to(Y.dtype.element_ty), mask=m3)


def fused_rot(x, angle, rot_dim, sorted_tok, block_experts, block_m, out=None,
              res=None, psel=None, bias=None, gelu=False, block_g=32):
    """Rotate x[:, :rot_dim] per expert and fold bias / p_sel*res / GELU into the same pass."""
    x = x.contiguous()
    m, dim = x.shape
    n_l = angle.shape[1]
    if not 1 <= n_l <= 2:
        raise ValueError(f"supports n_l in 1..2, got {n_l}")
    if dim % (1 << n_l):
        raise ValueError(f"dim {dim} must be a multiple of the group size {1 << n_l}")
    y = torch.empty_like(x) if out is None else out
    assert y.stride() == x.stride(), "kernel indexes Y with X's strides"
    grid = (sorted_tok.numel() // block_m, -(-(dim // (1 << n_l)) // block_g))
    _rot_full_kernel[grid](
        x, angle, sorted_tok, block_experts, res, psel, bias, y, m, dim, rot_dim,
        x.stride(0), x.stride(1), angle.stride(0), angle.stride(1), angle.stride(2),
        res.stride(0) if res is not None else 0, res.stride(1) if res is not None else 0,
        N_L=n_l, BLOCK_M=block_m, BLOCK_G=block_g,
        HAS_BIAS=bias is not None, HAS_RES=res is not None, ACT_GELU=gelu)
    return y

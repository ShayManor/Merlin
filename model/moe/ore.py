#!/usr/bin/env python3
"""ORE — Oriented Residual Experts. The method.

Each of K experts is a VIEW of one shared frozen weight W, not a copy of it:

    y_i(x) = Phi_i · W · Theta_i^T · x  +  B_i · A_i · x           (+ bias)
             \____ oriented substrate ____/   \___ residual ___/

Theta_i, Phi_i are butterfly-parameterised orthogonal maps applied to ACTIVATIONS, so W_i is
never materialised and an expert costs O(n_l*(d_in+d_out)) to apply. A_i, B_i are rank r.

Why both terms: Phi W Theta^T has the SAME SINGULAR VALUES as W. Rotation can reorient the row
and column spaces but cannot rescale the spectrum, so rotation-only experts live on a
norm-preserving orbit of one matrix -- they differ in direction but not in what they amplify.
The low-rank residual is exactly the term that changes the spectrum, cheaply but at limited
rank. Degenerate cases: n_l=0 -> DeRS; r=0 -> ButterflyViT (arXiv:2603.06746); K=1 -> LoRA.
Those are ablations of this module, not separate implementations.

Cost, per site: storage |W| + K*(n_l + r)*(d_in+d_out); compute |W| + (n_l + r)*(d_in+d_out).
The substrate is paid once in both, so total/active grows linearly in K at constant compute.

Init is function-preserving: Theta = Phi = I (angles 0) and B = 0, so the wrapped module is
bit-identical to its dense parent at step 0. Same convention as widen_ffn and DeRS.
"""
import math

import numpy as np
import torch
import torch.nn as nn

try:
    import triton
    from .triton_moe import (_grouped_gemm, align_block_size, fused_butterfly, fused_lowrank,
                             fused_rot, fused_route, padded_capacity)
    _HAVE_TRITON = True
except Exception:                                    # CPU / no-triton: reference path only
    _HAVE_TRITON = False


# ------------------------------------------------------------------ butterfly

class ButterflyStack(nn.Module):
    """K independent butterfly-orthogonal maps on R^dim, n_l layers of paired Givens rotations.

    Layer l pairs element j with j+2^l; viewing (M, dim/(2s), 2, s) puts those pairs adjacent.
    dim/2 angles per layer per expert, so the whole stack is n_l*dim/2 params -- the same count
    as a rank-n_l residual, but it re-orients W at FULL rank instead of adding a rank-n_l term.
    """

    def __init__(self, k, dim, n_l, device=None, dtype=None):
        super().__init__()
        if dim & (dim - 1):
            raise ValueError(f"butterfly needs a power-of-two dim, got {dim}")
        self.k, self.dim, self.n_l = k, dim, n_l
        self.angle = nn.Parameter(torch.zeros(k, n_l, dim // 2, device=device, dtype=dtype))

    def forward(self, x, eidx):
        if self.n_l == 0:
            return x
        # Fused path: read the activation once, rotate in registers, write once. The eager path
        # below materialises cos, sin, four products and a stack PER LAYER -- ~8 passes over the
        # activation -- which measured 3.4 ms/block on an Orin Nano, 4x the dense FFN, and does
        # NOT shrink under a CUDA graph because it is traffic, not launch latency. Fused it is
        # 0.51 ms/block, 8-20x faster, matching the eager result to 8e-4 relative (fp16 rounding).
        # Inference only: the kernel has no autograd backward, exactly as for _residual_sparse.
        if _HAVE_TRITON and x.is_cuda and not torch.is_grad_enabled() and 1 <= self.n_l <= 2:
            return fused_butterfly(x, self.angle, eidx)
        a = self.angle[eidx]                                   # (M, n_l, dim/2)
        for l in range(self.n_l):
            s = 1 << l
            nb = self.dim // (2 * s)
            v = x.reshape(-1, nb, 2, s)
            ang = a[:, l].reshape(-1, nb, s)
            c, sn = torch.cos(ang), torch.sin(ang)
            x0, x1 = v[:, :, 0], v[:, :, 1]
            x = torch.stack([x0 * c - x1 * sn, x0 * sn + x1 * c], 2).reshape(-1, self.dim)
        return x


def _pow2_floor(n):
    return 1 << (n.bit_length() - 1)


# ------------------------------------------------------------------ the module

class OREResidual(nn.Module):
    """Wrap an nn.Linear as K oriented-residual experts with hard top-1 routing."""

    def __init__(self, base, num_experts=64, rank=32, n_layers=2, block_m=64,
                 b_init="random", b_scale=0.01, balance="aux", route_mode="learned",
                 centres_log=None, router_norm=False, route_dim=0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"ORE wraps nn.Linear, got {type(base).__name__}")
        d_in, d_out = base.in_features, base.out_features
        dev, dt = base.weight.device, base.weight.dtype

        self.base = base
        self.k, self.r, self.n_l, self.bm = num_experts, rank, n_layers, block_m
        self.d_in, self.d_out = d_in, d_out

        # butterfly needs power-of-two dims; AAT is 768/3072 so rotate the largest pow2 prefix
        # and pass the tail through unchanged rather than padding the activation.
        self.rot_in_dim = _pow2_floor(d_in)
        self.rot_out_dim = _pow2_floor(d_out)
        self.theta = ButterflyStack(num_experts, self.rot_in_dim, n_layers, dev, dt)
        self.phi = ButterflyStack(num_experts, self.rot_out_dim, n_layers, dev, dt)

        # route_dim > 0 routes on an EXTERNAL frozen feature rather than this site's own input.
        # MEASURED motivation: site 0's routing input is a function of the frozen encoder and
        # frozen attention, so its partition never goes stale and it holds ~11 of 32 experts,
        # while deeper sites -- whose inputs carry the accumulated expert residuals -- average
        # 2.4. Routing every site on a frozen feature gives every site site-0's stability.
        self.route_dim = route_dim
        gate_in = route_dim or d_in
        # The gate carries a bias so a k-means calibration can be installed EXACTLY:
        # argmax(c_i.z - ||c_i||^2/2) is the nearest centroid to z. See calibrate_router.
        self.gate = nn.Linear(gate_in, num_experts, bias=True, device=dev, dtype=dt)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)
        # MEASURED ROOT CAUSE of the routing collapse (2026-08-10). AAT FFN tokens sit in a tight
        # cone around a large mean: ||mu|| 27-28 against a token-to-token spread of only 1.5-6.2,
        # and the centred cloud has effective rank ~9 of 768. A linear gate on the RAW activation
        # therefore returns near-identical logits for every token, and a fresh gate already routes
        # 2924 real tokens to 1.0-1.5 experts of 128 with 122-127 dead -- BEFORE ANY TRAINING. No
        # balancing loss can fix an input the gate cannot tell apart. Centring recovers ~20
        # experts, which is exactly what the aux-trained router reached, and whitening ~30.
        self.router_norm = router_norm
        self.register_buffer("in_mean", torch.zeros(gate_in, device=dev, dtype=torch.float32))
        self.register_buffer("in_std", torch.ones(gate_in, device=dev, dtype=torch.float32))

        # (E, in, out) layout: reduction dim in the middle, matching the grouped-GEMM strides
        self.A = nn.Parameter(torch.empty(num_experts, d_in, rank, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(num_experts, rank, d_out, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))       # symmetry break
        self.b_init, self.b_scale = b_init, b_scale
        if b_init == "random":
            # B=0 is the LoRA/DeRS convention but it is the wrong one when a ROUTER must learn:
            # the delta is exactly zero, so dL/dA and dL/dgate are both zero at init and only B
            # receives signal. This repo already found that for DeRS (--b-init random). Scale each
            # expert's delta to b_scale*||W|| so the gate has gradient from step 0 without the
            # init perturbing the pretrained substrate.
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
            with torch.no_grad():
                wn = base.weight.detach().float().norm()
                dn = torch.einsum("edr,erf->edf", self.A.float(), self.B.float()).norm(dim=(1, 2))
                self.B.mul_((b_scale * wn / dn.clamp_min(1e-8)).view(-1, 1, 1).to(dt))
        elif b_init != "zero":
            raise ValueError(f"b_init must be zero or random, got {b_init!r}")

        # ---- load balancing state
        # DeepSeek-V3 (arXiv:2412.19437) form: a per-expert bias added to the routing logits for
        # top-k SELECTION ONLY, never to the gate value, so it steers load without injecting an
        # interference gradient into the task objective. It is driven by an EMA of expert load,
        # which is what makes it usable here: the Switch aux statistic estimates f_i from one
        # batch, and at 2916 tokens over 128 experts that estimate is far too noisy to correct a
        # collapse. An EMA accumulates across steps instead.
        self.register_buffer("sel_bias", torch.zeros(num_experts, device=dev, dtype=torch.float32))
        self.register_buffer("load_ema", torch.full((num_experts,), 1.0 / num_experts,
                                                    device=dev, dtype=torch.float32))
        self.register_buffer("tok_count", torch.zeros(num_experts, device=dev, dtype=torch.float32))
        self.register_buffer("logit_scale", torch.ones((), device=dev, dtype=torch.float32))
        self.balance = balance
        self.route_mode = route_mode
        if route_mode == "key":
            if centres_log is None or len(centres_log) != num_experts:
                raise ValueError(f"route_mode='key' needs {num_experts} bin centres")
            self.register_buffer("centres", torch.as_tensor(centres_log, dtype=torch.float32,
                                                            device=dev))
        elif route_mode not in ("learned", "hash", "forced"):
            raise ValueError(f"unknown route_mode {route_mode!r}")
        self.aux_loss = None
        self.z_loss = None
        self.utilisation = None

    # -------------------------------------------------- accounting (asserted in tests)
    def expert_params(self):
        rot = self.n_l * (self.rot_in_dim // 2 + self.rot_out_dim // 2)
        return rot + self.r * (self.d_in + self.d_out)

    def total_params(self):
        base = self.base.weight.numel() + (self.base.bias.numel() if self.base.bias is not None else 0)
        return base + self.k * self.expert_params() + self.gate.weight.numel()

    def active_params(self):
        base = self.base.weight.numel() + (self.base.bias.numel() if self.base.bias is not None else 0)
        return base + self.expert_params() + self.gate.weight.numel()

    def N(self):
        """Params carried per param computed. The quantity the method lives or dies by."""
        return self.total_params() / self.active_params()

    # -------------------------------------------------- routing
    def _route_input(self, flat, shape=None):
        """The tensor the gate reads: this site's own input, or the broadcast frozen feature."""
        if not self.route_dim:
            return flat
        rf = getattr(self, "_route_feat", None)
        if rf is None:
            raise RuntimeError("route_dim set but no frozen routing feature was broadcast")
        # Token counts differ along the AAT stack: the frozen feature carries 731 per frame
        # (27x27 patches + 2 prepended scale/pose tokens) while some sites see 730. Align on
        # the TAIL, which is the patch grid, since the specials are prepended.
        if shape is not None and rf.dim() >= 2:
            b, t = shape[0], int(np.prod(shape[1:-1]))
            # The windowed C2 path can hand back a feature whose row count is not a multiple of b
            # (observed 5841 rows against b=8, i.e. 8*730+1), so the per-frame reshape below is not
            # always possible. Fall back to aligning on the tail of the flattened feature, which is
            # the same rule the per-frame branch applies, and report it once so the layout that
            # caused it is visible rather than silently absorbed.
            if rf.numel() % (b * self.route_dim):
                if not getattr(OREResidual, "_warned_rows", False):
                    OREResidual._warned_rows = True
                    print(f"[ore] routing feature {tuple(rf.shape)} = "
                          f"{rf.numel() // self.route_dim} rows does not divide b={b} "
                          f"(site shape={tuple(shape)}, t={t}); aligning on the flat tail",
                          flush=True)
                rin = rf.reshape(-1, self.route_dim)
                n = flat.shape[0]
                return rin[-n:] if rin.shape[0] >= n else torch.nn.functional.pad(
                    rin, (0, 0, n - rin.shape[0], 0))
            rf2 = rf.reshape(b, -1, self.route_dim)
            if rf2.shape[1] != t:
                rf2 = rf2[:, -t:] if rf2.shape[1] > t else torch.nn.functional.pad(
                    rf2, (0, 0, t - rf2.shape[1], 0))
            rf = rf2
        rin = rf.reshape(-1, self.route_dim)
        if rin.shape[0] != flat.shape[0]:
            raise RuntimeError(f"routing feature has {rin.shape[0]} rows, site sees "
                               f"{flat.shape[0]}")
        return rin

    def _route(self, flat, shape=None):
        """Returns (idx, p_sel). p_sel is the chosen expert's router probability.

        The residual is scaled by p_sel (Switch Transformer form). Without it the router gets NO
        TASK GRADIENT: argmax is non-differentiable, so the only signal reaching the gate is the
        load-balancing term and routing optimises for balance rather than for depth. Scaling the
        residual -- not the oriented substrate -- keeps the base contribution at full strength.
        """
        rin = self._route_input(flat, shape)
        logits = self.gate(self._norm_in(rin))
        probs = logits.softmax(-1)
        # SELECTION uses the balancing bias; the GATE VALUE never does. Mixing the bias into
        # p_sel would inject a balance gradient straight into the task loss, which is exactly the
        # interference DeepSeek-V3 removes.
        idx = (logits + self.sel_bias if self.balance.startswith("bias") else probs).argmax(-1)
        p_sel = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        # Switch load balance, so capacity is not wasted on dead experts
        hard = torch.zeros_like(probs).scatter_(-1, idx.unsqueeze(-1), 1.0)
        f, P = hard.mean(0), probs.mean(0)
        self.aux_loss = self.k * (f * P).sum()
        # ST-MoE (arXiv:2202.08906) router z-loss: penalise large logits so the softmax cannot
        # saturate into near-deterministic routing.
        self.z_loss = logits.float().logsumexp(-1).pow(2).mean() if "z" in self.balance else None
        # Counts ACCUMULATE across micro-steps, so balance is judged on the effective batch
        # rather than on one 729-token frame. run_cell resets them after each optimiser step.
        self.tok_count += hard.detach().sum(0).float()
        # The bias step is measured in units of the logit spread. A fixed absolute gamma is a
        # silent no-op whenever the router's logits are large -- the bias never closes the gap and
        # balancing looks implemented while doing nothing. Tracking the scale makes gamma
        # dimensionless and transferable across sites and training stages.
        self.logit_scale.mul_(0.9).add_(logits.detach().float().std().clamp_min(1e-6), alpha=0.1)
        self.utilisation = f.detach()
        return idx, p_sel

    @torch.no_grad()
    def update_bias(self, gamma=0.005, ema=0.99):
        """DeepSeek-V3 bias update, called once per OPTIMISER step (not per micro-step).

        b_i moves against the sign of the load error, so overloaded experts become slightly less
        attractive at selection time. Driven by an EMA rather than the current batch: at 22.8
        tokens/expert a single batch cannot estimate load at all.

        gamma is small and ema long on purpose. sign() is a bang-bang controller, so an
        aggressive step OVERSHOOTS and oscillates: measured on a collapsed 8-expert router,
        gamma 0.05/ema 0.9 settles at 3.3 effective experts while 0.005/ema 0.99 reaches 7.9.
        """
        tot = self.tok_count.sum()
        if tot <= 0:
            return
        self.load_ema.mul_(ema).add_(self.tok_count / tot, alpha=1 - ema)
        self.sel_bias -= gamma * self.logit_scale * torch.sign(self.load_ema - self.load_ema.mean())
        self.tok_count.zero_()

    @torch.no_grad()
    def load_stats(self):
        """Per-expert load diagnostics. K^H is the number of experts actually in use.

        Prefers tok_count, which ACCUMULATES across micro-batches, over utilisation, which holds
        only the last one. With gradient accumulation those differ by the accumulation factor: at
        accum 8 a step routes 32 frames but utilisation still describes 4, so reading it
        understates the diversity the optimiser actually saw.
        """
        u = self.tok_count if float(self.tok_count.sum()) > 0 else self.utilisation
        if u is None:
            return None
        p = (u / u.sum().clamp_min(1e-9)).clamp_min(1e-12)
        h = float(-(p * p.log()).sum() / math.log(self.k)) if self.k > 1 else 1.0
        return {"effective_experts": float(self.k ** h), "entropy": h,
                "cv": float(u.std(unbiased=False) / u.mean().clamp_min(1e-9)),
                "dead": int((u <= 0).sum()), "max_share": float(u.max()),
                "hist": [round(v, 5) for v in u.tolist()]}

    def _norm_in(self, flat):
        """Router input only. The substrate and the residual still see the raw activation."""
        if not self.router_norm:
            return flat
        return ((flat.float() - self.in_mean) / self.in_std).to(flat.dtype)

    @torch.no_grad()
    def calibrate_router(self, x, iters=25, seed=0, warm=False, ema=1.0):
        """Whiten the router input and put one gate row on each k-means centroid.

        Whitening alone lifts routing from ~1 expert to ~30, because the centred cloud is
        anisotropic (top eigenvalue holds 27-29% of the variance). k-means goes further, to
        104-107 of 128 with 0-1 dead, so the tokens ARE separable into K groups -- a random
        linear argmax is simply the wrong partitioner for a rank-9 cloud. Installing the
        centroids costs one pass over data and starts training where a random gate cannot reach.
        """
        x = x.float()
        old_mean, old_std = self.in_mean.clone(), self.in_std.clone()
        # ema < 1 blends toward the new statistics instead of replacing them. A hard re-partition
        # is disruptive even with cluster identity preserved, because the whitening shifts every
        # logit at once: measured, a cold refit took unseen-scene error 0.1807 -> 0.5713 and a
        # warm one 0.1696 -> 0.3355. Small, frequent blends track the drift without a jump.
        self.in_mean.mul_(1 - ema).add_(x.mean(0), alpha=ema)
        self.in_std.mul_(1 - ema).add_(x.std(0).clamp_min(1e-6), alpha=ema)
        z = (x - self.in_mean) / self.in_std
        if warm:
            # WARM START from the existing centroids, mapped through the OLD whitening back to raw
            # space and forward through the NEW one. A cold refit re-seeds k-means from random
            # tokens, so cluster i afterwards has no relation to cluster i before: every expert
            # keeps the specialisation it learned and inherits a DIFFERENT cluster. Measured cost
            # of getting this wrong: one hard refit at step 2000 took the unseen-scene error from
            # 0.1807 to 0.5713. Warm starting tracks the drift while preserving cluster identity.
            c = ((self.gate.weight.float() * old_std + old_mean) - self.in_mean) / self.in_std
            c = c.contiguous()
        else:
            g = torch.Generator(device=z.device).manual_seed(seed)
            c = z[torch.randperm(len(z), generator=g, device=z.device)[:self.k]].clone()
        for _ in range(iters):
            a = torch.cdist(z, c).argmin(1)
            for j in range(self.k):
                m = a == j
                if m.any():
                    c[j] = z[m].mean(0)
        if warm and ema < 1.0:
            c = (1 - ema) * self.gate.weight.float() + ema * c
        self.gate.weight.copy_(c.to(self.gate.weight.dtype))
        self.gate.bias.copy_((-0.5 * (c * c).sum(1)).to(self.gate.bias.dtype))
        return int((torch.bincount(torch.cdist(z, c).argmin(1), minlength=self.k) == 0).sum())

    def _oriented(self, flat, idx):
        """Phi_i · W · Theta_i^T · x, with bias applied after rotation so init is exact."""
        xi = flat
        if self.n_l:
            head, tail = xi[:, :self.rot_in_dim], xi[:, self.rot_in_dim:]
            xi = torch.cat([self.theta(head, idx), tail], 1) if tail.numel() else self.theta(head, idx)
        h = torch.nn.functional.linear(xi, self.base.weight)   # bias added after Phi
        if self.n_l:
            head, tail = h[:, :self.rot_out_dim], h[:, self.rot_out_dim:]
            h = torch.cat([self.phi(head, idx), tail], 1) if tail.numel() else self.phi(head, idx)
        if self.base.bias is not None:
            h = h + self.base.bias
        return h

    def _residual_dense(self, flat, idx):
        """Reference: compute every expert's residual and mask. Numerically == the sparse path.

        O(M*E*(d*r + r*F)) -- only used to CHECK the other two paths. Do not train through it.
        """
        one = torch.zeros(flat.shape[0], self.k, device=flat.device, dtype=flat.dtype)
        one.scatter_(-1, idx.unsqueeze(-1), 1.0)
        z = torch.einsum("md,edr->mer", flat, self.A)          # (M, E, r)
        z = z * one.unsqueeze(-1)
        return torch.einsum("mer,erf->mf", z, self.B)

    def _residual_gather(self, flat, idx):
        """Differentiable AND sparse in compute: gather the chosen expert's A,B per token.

        The masked reference computes all E experts and throws E-1 away, which at K=512 is 512x
        the necessary work and measured 1.02 s/step. This does exactly the active expert's work
        (M*(d*r + r*F)) and is a plain composition of indexing and bmm, so autograd handles it.
        Costs gather traffic instead: A[idx] is (M, d, r) and B[idx] is (M, r, F).
        """
        z = torch.bmm(flat.unsqueeze(1), self.A[idx])          # (M, 1, r)
        return torch.bmm(z, self.B[idx]).squeeze(1)            # (M, d_out)

    def _residual_sparse(self, flat, idx, st=None, be=None):
        m = flat.shape[0]
        ns = padded_capacity(m, self.k, self.bm)
        if st is None:
            st, be = align_block_size(idx.to(torch.int32), self.k, self.bm, ns)
        ns = st.numel()
        nb = ns // self.bm
        if getattr(self, "_zb", None) is None or self._zb.device != flat.device:
            # cached: allocating a (K, N) zero bias per forward measured as a flat ~1.5 ms
            self._zb = torch.zeros(self.k, max(self.r, self.d_out),
                                   device=flat.device, dtype=flat.dtype)
        zero_b = self._zb
        z = torch.empty((ns, self.r), device=flat.device, dtype=flat.dtype)
        y = torch.zeros((m, self.d_out), device=flat.device, dtype=flat.dtype)
        for a, w, out, N, K, gath, scat in (
                (flat, self.A, z, self.r, self.d_in, True, False),
                (z, self.B, y, self.d_out, self.r, False, True)):
            grid = lambda meta: (nb * triton.cdiv(N, meta["BLOCK_N"]),)
            _grouped_gemm[grid](
                a, w, zero_b, out, st, be, m, N, K,
                a.stride(0), a.stride(1), w.stride(0), w.stride(1), w.stride(2),
                out.stride(0), out.stride(1), zero_b.stride(0),
                GATHER_A=gath, SCATTER_C=scat, ACT_GELU=False, BLOCK_M=self.bm)
        return y

    # -------------------------------------------------- fused inference path
    def _fast_gate(self):
        """Gate weight transposed to (d_route, K), with router normalisation folded in.

        ((x - mu)/s) W^T + b == x (W/s)^T + (b - (mu/s) W^T), so normalisation costs nothing at
        inference. Cached on the parameter versions, since deployment freezes them.
        """
        w, b = self.gate.weight, self.gate.bias
        key = (w._version, b._version, self.in_mean._version, self.in_std._version)
        if getattr(self, "_fg_key", None) != key:
            if self.router_norm:
                s = self.in_std.to(w.dtype)
                w = w / s
                b = b - (self.in_mean.to(w.dtype) / s) @ w.t()
            self._fg = (w.t().contiguous(), b.contiguous())
            self._fg_key = key
        return self._fg

    def _fast_buf(self, m, device, dtype):
        """Persistent routing buffers. Reused so the path allocates nothing and can be captured
        into a CUDA graph, which is worth 2x on a Nano at these tensor sizes."""
        key = (m, str(device), dtype)
        if getattr(self, "_fbuf", None) is None or self._fbuf["key"] != key:
            ns = padded_capacity(m, self.k, self.bm)
            z = lambda n, dt: torch.zeros(n, dtype=dt, device=device)
            self._fbuf = {"key": key, "idx": z(m, torch.int32), "psel": z(m, dtype),
                          "hist": z(self.k, torch.int32), "cursor": z(self.k, torch.int32),
                          "stok": z(ns, torch.int32), "be": z(ns // self.bm, torch.int32)}
        return self._fbuf

    # Folding the oriented epilogue into the residual GEMM saves a 4.5 MB round trip but forces
    # the output tile into four strided accumulators; whether that pays depends on block_m, so it
    # stays switchable and is decided by measurement, not by the traffic argument alone.
    fuse_epilogue = True

    def can_fast(self):
        """Whether the fused path implements exactly what this configuration asks for."""
        return (_HAVE_TRITON and self.route_mode == "learned" and 1 <= self.n_l <= 2
                and not self.balance.startswith("bias"))

    def _forward_fast(self, x):
        """Same function as forward(), in 7 launches instead of ~30.

        What the eager path spends its time on, measured on an Orin Nano, is not arithmetic: the
        router is 13 launches over 128- and 729-element tensors, and `_oriented` makes six extra
        passes over a (729, 3072) activation to rotate a prefix, re-concatenate the tail, add a
        bias and add the scaled residual. Here routing is three launches and every elementwise
        pass is folded into the two rotation kernels.

        Bookkeeping that only training reads -- aux_loss, z_loss, utilisation, tok_count -- is not
        maintained here, which is why this path is opt-in via enable_fast_inference().
        """
        shape = x.shape
        flat = x.reshape(-1, self.d_in)
        gw_t, gb = self._fast_gate()
        buf = self._fast_buf(flat.shape[0], flat.device, flat.dtype)
        idx, psel, st, be = fused_route(self._route_input(flat, shape), gw_t, gb, buf, self.bm)
        xi = fused_rot(flat, self.theta.angle, self.rot_in_dim, st, be, self.bm)
        h = torch.nn.functional.linear(xi, self.base.weight)
        # One kernel for the residual AND the whole oriented epilogue, so the residual is never
        # written to memory: at 3072 wide that alone was a 4.5 MB store and a 4.5 MB load.
        out = None
        if self.fuse_epilogue:
            out = fused_lowrank(flat, self.A, self.B, st, be, self.bm, h=h, angle=self.phi.angle,
                                rot_dim=self.rot_out_dim, psel=psel, bias=self.base.bias)
        if out is None:                                  # rank or shape outside the fused kernel
            res = fused_lowrank(flat, self.A, self.B, st, be, self.bm)
            if res is None:
                res = self._residual_sparse(flat, idx, st, be)
            # NOT in place on h: Triton marks tensor arguments noalias, so a kernel whose loads
            # all precede its stores may still have them reordered when X and Y are one buffer.
            out = fused_rot(h, self.phi.angle, self.rot_out_dim, st, be, self.bm,
                            res=res, psel=psel, bias=self.base.bias)
        return out.reshape(*shape[:-1], self.d_out)

    def set_key(self, key):
        """Per-token routing key for route_mode='key'. (B,1,H,W) metric depth, as arms P/O."""
        self._key = key

    def _fixed_idx(self, shape, device):
        """Routing that cannot collapse, because no gate chooses it.

        'hash'  token position scrambled mod K (Hash Layers, arXiv:2106.04426). Balance is
                structural: expert i gets floor(T/K) or ceil(T/K) tokens of EVERY frame.
        'key'   nearest equal-population quantile centre in log-depth, i.e. arms P/O carried
                over to this module. Also balanced by construction, since the centres are
                equal-population on the training set.

        Both exist to separate "the router is broken" from "conditional capacity does not help
        here". Every learned mechanism measured on 2026-08-10 collapsed, so neither hypothesis
        could be tested against the other.
        """
        b, t = shape[0], int(np.prod(shape[1:-1]))
        if self.route_mode == "forced":       # diagnostic: every token to one expert
            return torch.full((b * t,), int(self._forced), dtype=torch.long, device=device)
        if self.route_mode == "hash":
            pos = torch.arange(t, device=device)
            return ((pos * 31 + 17) % self.k).repeat(b)
        key = getattr(self, "_key", None)
        if key is None:
            raise RuntimeError("route_mode='key' needs set_keys(model, key) before the forward")
        # The AAT sequence is 731, not 729: two special tokens (scale/pose) are PREPENDED to the
        # 27x27 patch grid. They have no depth, so they take the frame's median bin -- 2 of 731
        # tokens, so the choice is immaterial even if the convention is the other way round.
        s = int(float(t) ** 0.5)
        k = torch.nn.functional.interpolate(key.float(), size=(s, s), mode="nearest")
        lg = k.reshape(b, -1).clamp_min(1e-3).log()
        idx = (lg.unsqueeze(-1) - self.centres.view(1, 1, -1)).abs().argmin(-1)
        if t > s * s:
            pad = idx.median(dim=1, keepdim=True).values.expand(b, t - s * s)
            idx = torch.cat([pad, idx], 1)
        return idx.reshape(-1)

    def forward(self, x, sparse=None):
        shape = x.shape
        flat = x.reshape(-1, self.d_in)
        if getattr(self, "fast", False) and flat.is_cuda and not torch.is_grad_enabled():
            return self._forward_fast(x)
        if self.route_mode == "learned":
            idx, p_sel = self._route(flat, shape)
        else:
            idx = self._fixed_idx(shape, flat.device)
            p_sel = torch.ones(flat.shape[0], device=flat.device, dtype=flat.dtype)
            self.aux_loss = self.z_loss = None
            hard = torch.zeros(flat.shape[0], self.k, device=flat.device)
            self.utilisation = hard.scatter_(-1, idx.unsqueeze(-1), 1.0).mean(0)
        if sparse is None:
            # TRITON KERNELS HAVE NO AUTOGRAD BACKWARD. Routing training through the sparse path
            # silently detaches A and B -- their .grad comes back None, 1.0B of 1.04B trainable
            # params never move, and the run reports a plausible-looking null instead of an
            # error. Sparse is an INFERENCE optimisation; training uses the gather path, which is
            # differentiable and does only the ACTIVE expert's work.
            sparse = _HAVE_TRITON and flat.is_cuda and not torch.is_grad_enabled()
        if sparse:
            res = self._residual_sparse(flat, idx)
        else:
            # MEASURED: masked 1.02 s/step vs gather 1.40 s/step at K=512. The gather looks
            # cheaper in FLOPs (only the active expert) but materialises A[idx] (M,d,r) and
            # B[idx] (M,r,F) -- 179 MB per site, 2.9 GB per forward -- so it is memory-bound and
            # loses. Consequence worth stating in the paper: ORE inference is O(1) in K, but
            # TRAINING is O(K), because the backward needs every expert.
            res = self._residual_dense(flat, idx)
        res = res * p_sel.unsqueeze(-1).to(res.dtype)          # router's task gradient path
        return (self._oriented(flat, idx) + res).reshape(*shape[:-1], self.d_out)

    def freeze_base(self):
        self.base.requires_grad_(False)
        return self


def enable_fast_inference(model, on=True):
    """Switch every ORE site to the fused inference path. Returns the sites that took it.

    Opt-in, not automatic: the fused path drops the router bookkeeping (aux_loss, z_loss,
    utilisation, tok_count) that analysis scripts read under no_grad, so turning it on silently
    would change what those scripts measure rather than only how fast they run.
    """
    took = []
    for m in ore_modules(model):
        m.fast = bool(on) and m.can_fast()
        if m.fast:
            took.append(m)
    return took


def set_route_feature(model, feat):
    """Broadcast one frozen feature to every ORE site's router."""
    for m in ore_modules(model):
        if m.route_dim:
            m._route_feat = feat


def ore_modules(model):
    return [m for m in model.modules() if isinstance(m, OREResidual)]


def ore_z_loss(model):
    """Mean router z-loss over ORE sites, or None if no site is using it."""
    zs = [m.z_loss for m in ore_modules(model) if m.z_loss is not None]
    return sum(zs) / len(zs) if zs else None


def ore_parameters(model):
    """Trainable set: rotations, residuals, gates. Never the frozen substrate."""
    ps = []
    for m in ore_modules(model):
        ps += [m.A, m.B]
        if m.route_mode == "learned":          # a fixed partition has no gate to train
            ps += [m.gate.weight, m.gate.bias]
        if m.n_l:
            ps += [m.theta.angle, m.phi.angle]
    return ps


def upcycle_ore(model, target_paths, num_experts=64, rank=32, n_layers=2, freeze=True,
                balance="aux", route_mode="learned", centres_log=None, router_norm=False,
                b_init="random", b_scale=0.01, route_dim=0):
    from .upcycle import _get, _set
    wrapped = []
    for path in target_paths:
        base = _get(model, path)
        mod = OREResidual(base, num_experts, rank, n_layers, b_init=b_init, b_scale=b_scale,
                          balance=balance,
                          route_mode=route_mode, centres_log=centres_log,
                          router_norm=router_norm, route_dim=route_dim)
        if freeze:
            mod.freeze_base()
        _set(model, path, mod)
        wrapped.append(mod)
    return wrapped

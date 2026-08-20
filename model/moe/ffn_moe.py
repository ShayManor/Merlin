#!/usr/bin/env python3
"""Full-FFN sparse upcycling: K trainable COPIES of a block's MLP, routed per token.

The control ORE could not run. An ORE expert is a rank-r residual on a FROZEN shared substrate,
so active TRAINABLE adaptation capacity is 0.21x dense and the governing prior art is
LoRA-Learns-Less (2405.09673). Here every expert is a full copy of the pretrained MLP and all of
them train, so active trainable params EQUAL the dense fine-tune by construction and the
comparison the sparse-upcycling literature makes (Komatsuzaki et al., 2212.05055) is finally the
one being run. See docs/superpowers/specs/2026-08-14-fcr-00-upcycling-control.md.

    y_t = sum_{i in S_t} w_{t,i} * MLP_i(x_t),   S_t = top-k of the router,
                                                w = softmax over the SELECTED logits

Renormalising w over the selected experts is what makes init exactly function-preserving for any
k: the experts are identical copies at step 0, so sum_i w_i * MLP(x) = MLP(x). Switch's
un-normalised p_sel does not have that property and would start every arm at a different
function.

Subclasses OREResidual for its ROUTER only. Inherited on purpose, each because it is a fix this
repo paid for: _route_input/route_dim (frozen-feature routing, L-ORE-32), _norm_in +
calibrate_router (a fresh linear gate routes 2924 AAT tokens to 1.0-1.5 experts of 128 BEFORE
training, L-ORE-14), update_bias/sel_bias (DeepSeek-V3 balancing), load_stats (feeds run_cell's
collapse-abort gate). __init__ does NOT call super().__init__: that one allocates a butterfly
stack and A/B against an nn.Linear base, none of which exists here. ore.py is deliberately left
untouched -- it still supplies the ORE baseline row.
"""
import copy

import torch
import torch.nn as nn

from .ore import OREResidual


class FFNMoE(OREResidual):
    """K full copies of one MLP module, top-k routed, all trainable. No shared substrate."""

    def __init__(self, mlp, num_experts=8, top_k=1, council_m=0, balance="aux",
                 route_mode="learned", router_norm=False, route_dim=0):
        nn.Module.__init__(self)
        fc1, fc2 = mlp.fc1, mlp.fc2
        dev, dt = fc1.weight.device, fc1.weight.dtype
        if top_k > num_experts:
            raise ValueError(f"top_k {top_k} > K {num_experts}")
        if council_m and council_m < top_k:
            raise ValueError(f"council_m {council_m} < top_k {top_k}: the council would starve "
                             f"token routing")
        self.k, self.top_k, self.council_m = num_experts, top_k, council_m
        self.d_in, self.d_out = fc1.in_features, fc2.out_features
        self.r, self.n_l = 0, 0                    # no residual, no rotation: this is the control

        # EXACT copies, no noise. Symmetry is broken by the k-means router partition, which
        # already separates tokens at step 0; exact copies are what buy function-preserving init.
        # Recorded alternative if experts stay tied: per-expert N(0, 0.01*||W||).
        self.experts = nn.ModuleList([copy.deepcopy(mlp) for _ in range(num_experts)])

        # ---- router state, mirroring OREResidual.__init__ so every inherited method works
        self.route_dim = route_dim
        gate_in = route_dim or self.d_in
        self.gate = nn.Linear(gate_in, num_experts, bias=True, device=dev, dtype=dt)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)
        self.router_norm = router_norm
        self.register_buffer("in_mean", torch.zeros(gate_in, device=dev, dtype=torch.float32))
        self.register_buffer("in_std", torch.ones(gate_in, device=dev, dtype=torch.float32))
        self.register_buffer("sel_bias", torch.zeros(num_experts, device=dev, dtype=torch.float32))
        self.register_buffer("load_ema", torch.full((num_experts,), 1.0 / num_experts,
                                                    device=dev, dtype=torch.float32))
        self.register_buffer("tok_count", torch.zeros(num_experts, device=dev, dtype=torch.float32))
        self.register_buffer("logit_scale", torch.ones((), device=dev, dtype=torch.float32))
        self.balance = balance
        if route_mode != "learned":
            raise ValueError(f"FFNMoE routes learned only, got {route_mode!r}")
        self.route_mode = route_mode
        self.aux_loss = None
        self.z_loss = None
        self.utilisation = None

    # -------------------------------------------------- accounting (asserted by the gates)
    def expert_params(self):
        return sum(p.numel() for p in self.experts[0].parameters())

    def gate_params(self):
        return self.gate.weight.numel() + self.gate.bias.numel()

    def total_params(self):
        return self.k * self.expert_params() + self.gate_params()

    def active_params(self):
        return self.top_k * self.expert_params() + self.gate_params()

    def parameters_to_train(self):
        return list(self.experts.parameters()) + [self.gate.weight, self.gate.bias]

    def freeze_base(self):
        raise RuntimeError("FFNMoE has no shared base to freeze; every expert trains")

    # -------------------------------------------------- routing
    def _council(self, logits, shape):
        """Frame-level dispatch: one expert council per frame, tokens choose inside it.

        The gate is LINEAR, so mean_t gate(z_t) == gate(mean_t z_t) exactly; pooling the logits is
        the pooled-representation council of report.md with no extra work. Masking happens BEFORE
        the softmax so the token weights renormalise within the council, which keeps the
        function-preserving init property.
        """
        b = shape[0]
        m = logits.shape[0] // b
        if m * b != logits.shape[0]:
            raise RuntimeError(f"{logits.shape[0]} tokens do not divide batch {b}")
        per_frame = logits.reshape(b, m, self.k).mean(1)                       # (b, K)
        keep = torch.zeros_like(per_frame, dtype=torch.bool).scatter_(
            -1, per_frame.topk(self.council_m, dim=-1).indices, True)          # (b, K)
        self.council = keep.detach()
        return logits.masked_fill(~keep.repeat_interleave(m, 0), float("-inf"))

    def _route_topk(self, flat, shape=None):
        """(idx, w) with idx (M, top_k) and w renormalised over the selected experts.

        Same bookkeeping as OREResidual._route: selection may use sel_bias, the gate VALUE never
        does (DeepSeek-V3), counts accumulate across micro-steps so balance is judged on the
        effective batch, and aux is the Switch term with f measured over dispatch slots.
        """
        rin = self._route_input(flat, shape)
        logits = self.gate(self._norm_in(rin))
        if self.council_m:
            logits = self._council(logits, shape)
        probs = logits.softmax(-1)
        sel = logits + self.sel_bias if self.balance.startswith("bias") else probs
        idx = sel.topk(self.top_k, dim=-1).indices                             # (M, top_k)
        w = probs.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        hard = torch.zeros_like(probs).scatter_(-1, idx, 1.0 / self.top_k)
        f, P = hard.mean(0), probs.mean(0)
        self.aux_loss = self.k * (f * P).sum()
        self.z_loss = (logits.float().logsumexp(-1).pow(2).mean()
                       if "z" in self.balance else None)
        self.tok_count += hard.detach().sum(0).float() * self.top_k
        self.logit_scale.mul_(0.9).add_(logits.detach().float()[logits.isfinite()].std()
                                        .clamp_min(1e-6), alpha=0.1)
        self.utilisation = f.detach()
        return idx, w

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.d_in)
        idx, w = self._route_topk(flat, shape)
        out = torch.zeros(flat.shape[0], self.d_out, device=flat.device, dtype=flat.dtype)
        for i, ex in enumerate(self.experts):
            hit = idx == i                                                     # (M, top_k)
            tok = hit.any(-1).nonzero(as_tuple=True)[0]
            if tok.numel() == 0:
                continue
            # A token can select an expert once at most, so summing the masked weights picks the
            # right slot's weight without a second gather.
            wi = (w * hit).sum(-1)[tok]
            out.index_add_(0, tok, ex(flat[tok]) * wi.unsqueeze(-1))
        return out.reshape(*shape[:-1], self.d_out)

    # -------------------------------------------------- diagnostics
    @torch.no_grad()
    def expert_divergence(self, sample=4096):
        """Have the copies actually separated? Subsampled: full fc1 weights are 2.4M each.

        cos == 1.000 after training means the experts are still the dense parent and any accuracy
        number describes the dense model, which is the FFN analogue of upcycle.INERT_MIXTURE.
        """
        idx = torch.arange(0, min(sample, self.experts[0].fc1.weight.numel()))
        f = torch.stack([e.fc1.weight.detach().float().reshape(-1)[idx] for e in self.experts])
        f = f - f.mean(0, keepdim=True)
        n = f.norm(dim=1).clamp_min(1e-12)
        c = (f / n.unsqueeze(1)) @ (f / n.unsqueeze(1)).t()
        k = self.k
        off = (c.sum() - c.diag().sum()) / max(k * (k - 1), 1)
        return {"expert_cos": float(off) if k > 1 else float("nan"),
                "expert_spread": float(n.mean())}


def ffn_modules(model):
    return [m for m in model.modules() if isinstance(m, FFNMoE)]


def upcycle_ffn(model, target_paths, num_experts=8, top_k=1, council_m=0, balance="aux",
                router_norm=False, route_dim=0):
    """Replace each target MLP module with an FFNMoE. Paths are module paths, not Linears."""
    from .upcycle import _get, _set
    wrapped = []
    for path in target_paths:
        mlp = _get(model, path)
        if not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2")):
            raise TypeError(f"{path} is a {type(mlp).__name__}, expected an MLP with fc1/fc2. "
                            f"Pass the mlp module (…blocks.N.mlp), not its Linears.")
        mod = FFNMoE(mlp, num_experts=num_experts, top_k=top_k, council_m=council_m,
                     balance=balance, router_norm=router_norm, route_dim=route_dim)
        _set(model, path, mod)
        wrapped.append(mod)
    return wrapped


def ffn_parameters(model):
    ps = []
    for m in ffn_modules(model):
        ps += m.parameters_to_train()
    return ps

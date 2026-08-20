"""Low-rank residual experts with pluggable routing.

Spec: docs/superpowers/specs/2026-08-08-moe-01-residual-module.md and
      docs/superpowers/specs/2026-08-08-rmoe-00-crossing-curves.md

Wraps an existing Linear or Conv2d as a K-expert mixture without replacing the weight:

    y = base(x) + sum_i g_i(x) * B_i(A_i(x))

A is randomly initialised, B is ZERO initialised, so every delta is exactly zero at
construction and the wrapped module is bit-identical to its dense parent. Router symmetry still
breaks because A differs per expert (the LoRA construction).

All K experts are always computed. There is no dispatch, so shapes are static. The trick that
makes this cheap regardless of K: g_i is a scalar per position and B_i is linear, so
g_i * B_i(z_i) == B_i(g_i * z_i). Scaling before the projection lets all K experts share one
fused A and one fused B, i.e. two ops total instead of 2K.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ routers

class LearnedGate(nn.Module):
    """Router that must discover the split from data. Arm L.

    mode="soft"  : softmax over K, every expert receives gradient (the stronger baseline).
    mode="top1"  : hard top-1 with a straight-through estimator, the standard sparse-MoE
                   formulation the expert-count literature saturates under.
    """

    def __init__(self, in_ch, num_experts, mode="top1", conv=True):
        super().__init__()
        self.num_experts = num_experts
        self.mode = mode
        self.conv = conv
        self.eaxis = 1 if conv else -1        # where the expert axis lands for this layout
        self.proj = nn.Conv2d(in_ch, num_experts, 1, bias=False) if conv \
            else nn.Linear(in_ch, num_experts, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.aux_loss = None          # load-balancing loss, collected by the trainer
        self.utilisation = None       # per-expert token share, for the entropy diagnostic

    def forward(self, x, key=None):
        ax = self.eaxis % x.dim()
        logits = self.proj(x)                                   # experts on `ax`
        probs = logits.softmax(dim=ax)

        # Load balance (Switch Transformer form): K * sum_i f_i * P_i, minimised at uniform.
        dims = [d for d in range(probs.dim()) if d != ax]
        P = probs.mean(dim=dims)
        hard = torch.zeros_like(probs).scatter_(ax, probs.argmax(ax, keepdim=True), 1.0)
        f = hard.mean(dim=dims)
        self.aux_loss = self.num_experts * (f * P).sum()
        self.utilisation = f.detach()
        # ARGMAX share (above) and SOFT mass (below) can point opposite ways, and for a soft
        # gate it is the soft mass that determines the output. A soft gate at uniform mass is
        # degenerate -- it mixes every expert equally, which is not routing at all -- yet its
        # argmax share can look healthily concentrated. Recording both; mixture_health() reads
        # this one.
        self.soft_mass = P.detach()

        if self.mode == "soft":
            return probs
        # top-1, straight-through so the router still receives gradient
        return hard + probs - probs.detach()


class PrescribedRangeGate(nn.Module):
    """Router prescribed by metric range. Arm P (predicted key) and arm O (GT key).

    Soft assignment between the two adjacent bin centres, linear in log-depth. Hard assignment
    would put a discontinuity in the output at every bin edge, which shows up as seams in the
    depth map and phantom structure in the occupancy map.

    Bin centres are frozen at construction. They are computed once from the training set as
    equal-population quantiles, which is what makes collapse impossible by construction rather
    than merely discouraged: every expert is guaranteed tokens.
    """

    def __init__(self, centres_log):
        super().__init__()
        c = torch.as_tensor(centres_log, dtype=torch.float32)
        if c.dim() != 1 or c.numel() < 2 or not bool((c[1:] > c[:-1]).all()):
            raise ValueError("bin centres must be 1-D, length >= 2, strictly increasing")
        self.register_buffer("centres", c)
        self.num_experts = c.numel()
        self.eaxis = 1                # spatial (B, K, H, W) only; see forward()
        self.aux_loss = None          # prescribed routing needs no balancing loss
        self.utilisation = None

    @staticmethod
    def quantile_centres(depths, num_experts, dmin=0.05):
        """Equal-population bin centres in log-depth, from a sample of training depths."""
        d = torch.as_tensor(depths, dtype=torch.float32).flatten()
        d = d[torch.isfinite(d) & (d > dmin)]
        qs = (torch.arange(num_experts, dtype=torch.float32) + 0.5) / num_experts
        return torch.quantile(d.log(), qs)

    def forward(self, x, key=None):
        if key is None:
            raise ValueError("PrescribedRangeGate needs a depth key")
        if x.dim() != 4:
            # Per-token sites would need the key gathered per patch, which is a different and
            # currently unneeded mapping. Fail loudly rather than assume a layout.
            raise NotImplementedError(
                "PrescribedRangeGate supports spatial (B, C, H, W) sites only; "
                f"got {x.dim()}-D. The DPT head site is spatial.")
        if key.dim() == x.dim() - 1:
            key = key.unsqueeze(1)
        if key.shape[-2:] != x.shape[-2:]:
            key = F.interpolate(key, size=x.shape[-2:], mode="nearest")

        v = key.clamp_min(1e-3).log()
        c = self.centres
        hi = torch.bucketize(v, c).clamp(1, c.numel() - 1)      # index of upper centre
        lo = hi - 1
        c_lo, c_hi = c[lo], c[hi]
        w_hi = ((v - c_lo) / (c_hi - c_lo).clamp_min(1e-6)).clamp(0, 1)

        g = torch.zeros(v.shape[0], c.numel(), *v.shape[2:], device=v.device, dtype=x.dtype)
        g.scatter_(1, lo, (1 - w_hi).to(g.dtype))
        g.scatter_add_(1, hi, w_hi.to(g.dtype))
        dims = [0] + list(range(2, g.dim()))
        self.utilisation = g.mean(dim=dims).detach()
        return g


class PretrainedRouterGate(nn.Module):
    """Gate driven by a GT-supervised bin classifier instead of thresholded predicted depth.

    The depth-keyed gate routed at 44.0% against a 54.3% per-image majority baseline, i.e.
    significantly WORSE than a constant assignment (paired t = -7.04 over 200 images), so the
    experts never received range-partitioned data. This head reads the same features the site
    already consumes and routes at 63.3% (t = +14.89). It sees no ground truth at inference.
    """

    def __init__(self, head, num_experts, temperature=1.0, freeze=True):
        super().__init__()
        self.head, self.num_experts, self.temperature = head, num_experts, temperature
        self.eaxis = 1
        self.aux_loss = self.utilisation = None
        # A frozen gate must stay out of the trainable set, or the router's 74k parameters are
        # counted against the expert budget and the capacity match to the dense control breaks.
        self.frozen_gate = bool(freeze)
        if freeze:
            for p in self.head.parameters():
                p.requires_grad_(False)

    def forward(self, x, key=None):
        g = (self.head(x.float()) / self.temperature).softmax(dim=1).to(x.dtype)
        dims = [d for d in range(g.dim()) if d != 1]
        self.utilisation = g.mean(dim=dims).detach()
        return g


# ------------------------------------------------------------------ the module

class MoEResidual(nn.Module):
    def __init__(self, base, gate, num_experts=4, rank=8, b_init="zero", b_scale=0.01):
        super().__init__()
        if isinstance(base, nn.Conv2d):
            in_ch, out_ch, conv = base.in_channels, base.out_channels, True
        elif isinstance(base, nn.Linear):
            in_ch, out_ch, conv = base.in_features, base.out_features, False
        else:
            raise TypeError(f"cannot wrap {type(base).__name__}")

        self.base, self.gate = base, gate
        self.num_experts, self.rank, self.is_conv = num_experts, rank, conv
        self.eaxis = 1 if conv else -1
        if getattr(gate, "num_experts", num_experts) != num_experts:
            raise ValueError(f"gate has {gate.num_experts} experts, module has {num_experts}")
        kr = num_experts * rank

        if conv:
            self.A = nn.Conv2d(in_ch, kr, base.kernel_size, stride=base.stride,
                               padding=base.padding, dilation=base.dilation, bias=False)
            self.B = nn.Conv2d(kr, out_ch, 1, bias=False)
        else:
            self.A = nn.Linear(in_ch, kr, bias=False)
            self.B = nn.Linear(kr, out_ch, bias=False)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        if b_init == "zero":
            nn.init.zeros_(self.B.weight)      # identity at init, and the LoRA symmetry break
        elif b_init == "random":
            # B=0 is the LoRA/DeRS convention, but it makes dL/dg EXACTLY zero at init: the
            # gate cannot feel a routing decision that changes nothing, so it never bootstraps
            # and the load-balance term parks it at uniform. MoE upcycling instead starts
            # experts non-degenerate. Scale each delta to b_scale * ||W_base|| so the gate gets
            # signal from step 0 without the init perturbation swamping the pretrained weights.
            nn.init.kaiming_uniform_(self.B.weight, a=math.sqrt(5))
            self._rescale_b(base, b_scale)
        else:
            raise ValueError(f"b_init must be zero or random, got {b_init!r}")
        self.b_init, self.b_scale = b_init, b_scale
        self._key = None

    @torch.no_grad()
    def _rescale_b(self, base, target):
        """Set each expert's ||B_i A_i||_F to target * ||W_base||_F.

        Done per expert rather than globally so no expert starts dominant. For Conv2d the delta
        is not a plain matrix product, so the norm is bounded by ||B_i|| ||A_i|| instead; that
        over-estimates, which errs toward a SMALLER perturbation.
        """
        wb = float(base.weight.detach().float().norm())
        r = self.rank
        for i in range(self.num_experts):
            bs = (slice(None), slice(i * r, (i + 1) * r))
            Bi = self.B.weight[bs].float()
            Ai = self.A.weight[i * r:(i + 1) * r].float()
            cur = float((Bi @ Ai).norm()) if not self.is_conv else \
                float(Bi.norm()) * float(Ai.norm())
            if cur > 1e-12:
                self.B.weight[bs] *= target * wb / cur

    def set_key(self, key):
        """Routing key for prescribed gates, set before the forward pass."""
        self._key = key

    def added_params(self):
        """Expert parameters only, excluding the gate. Equals K*r*(d_in + d_out) for Linear."""
        return self.A.weight.numel() + self.B.weight.numel()

    def freeze_base(self):
        self.base.requires_grad_(False)
        return self

    def forward(self, x):
        g = self.gate(x, self._key)                             # experts on self.eaxis
        z = self.A(x)                                           # K*r on self.eaxis
        # g_i * B_i(z_i) == B_i(g_i * z_i): scale before projecting so one fused B serves all K,
        # which keeps the cost two ops regardless of K.
        g = g.repeat_interleave(self.rank, dim=self.eaxis)
        return self.base(x) + self.B(z * g.to(z.dtype))

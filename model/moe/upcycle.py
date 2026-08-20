"""Swap named modules of a built model for MoEResidual wrappers, and drive them.

Target paths are explicit rather than pattern-matched, so what was wrapped is recorded in the
checkpoint and reproducible.
"""
import numpy as np
import torch

from .residual import LearnedGate, MoEResidual, PrescribedRangeGate, PretrainedRouterGate


def _get(model, path):
    m = model
    for p in path.split("."):
        m = m[int(p)] if p.isdigit() and not hasattr(m, p) else getattr(m, p)
    return m


def _set(model, path, new):
    parent, _, leaf = path.rpartition(".")
    obj = _get(model, parent) if parent else model
    if leaf.isdigit() and not hasattr(obj, leaf):
        obj[int(leaf)] = new
    else:
        setattr(obj, leaf, new)


def make_gate(kind, in_ch, num_experts, centres_log=None, conv=True, router=None):
    if kind in ("learned_top1", "learned_soft"):
        return LearnedGate(in_ch, num_experts,
                           mode="top1" if kind.endswith("top1") else "soft", conv=conv)
    if kind == "prescribed":
        if centres_log is None:
            raise ValueError("prescribed routing needs frozen bin centres")
        if len(centres_log) != num_experts:
            raise ValueError(f"{len(centres_log)} centres for {num_experts} experts")
        return PrescribedRangeGate(centres_log)
    if kind == "router":
        if router is None:
            raise ValueError("router gate needs a pretrained head")
        return PretrainedRouterGate(router, num_experts)
    raise ValueError(f"unknown gate kind {kind!r}")


def upcycle(model, target_paths, kind, num_experts=4, rank=8, centres_log=None,
            freeze=True, router=None, b_init="zero", b_scale=0.01):
    """Wrap each target path in place. Returns the list of wrappers, in order."""
    wrapped = []
    for path in target_paths:
        base = _get(model, path)
        conv = isinstance(base, torch.nn.Conv2d)
        in_ch = base.in_channels if conv else base.in_features
        gate = make_gate(kind, in_ch, num_experts, centres_log, conv=conv, router=router)
        mod = MoEResidual(base, gate, num_experts=num_experts, rank=rank,
                          b_init=b_init, b_scale=b_scale)
        if freeze:
            mod.freeze_base()
        _set(model, path, mod)
        wrapped.append(mod)
    return wrapped


def moe_modules(model):
    return [m for m in model.modules() if isinstance(m, MoEResidual)]


# A mixture whose realised delta is below this fraction of the base weight norm is INERT: it
# cannot change the output, so any accuracy result from it measures the base model, not the
# mixture. Set from measurement -- the K=32 r=32 run that produced a null sat at 0.00048.
INERT_RATIO = 0.005


def mixture_health(model):
    """Is the mixture actually doing anything? Weights + last forward's gate stats, no data.

    Exists because a null result was once attributed to "the task is not capacity-bound" when
    the real cause was that the experts perturbed the base weights by 0.05% and could not have
    changed anything. Three numbers separate those cases:

      realised_ratio  ||sum_i mass_i * B_i A_i|| / ||W_base||. THE number. Near zero => inert.
      expert_ratio    mean ||B_i A_i|| / ||W_base||. Large while realised is small => the
                      experts exist but the gate is averaging them into cancellation.
      soft_entropy    normalised entropy of the gate's SOFT mass. 1.0 => uniform mixing, i.e.
                      no routing; 0.0 => collapsed to one expert. Both extremes are degenerate.
    """
    import math

    mods = moe_modules(model)
    if not mods:
        return _ore_health(model)
    cos, er, rr, ents = [], [], [], []
    for m in mods:
        r, k = m.rank, m.num_experts
        A = m.A.weight.detach().float().reshape(k * r, -1)
        B = m.B.weight.detach().float().reshape(-1, k * r)
        W = torch.stack([B[:, i * r:(i + 1) * r] @ A[i * r:(i + 1) * r] for i in range(k)])
        wb = float(m.base.weight.detach().float().norm())
        if wb < 1e-12:
            continue
        f = W.reshape(k, -1)
        n = f.norm(dim=1)
        er.append(float(n.mean()) / wb)
        mass = getattr(m.gate, "soft_mass", None)
        w = (mass.detach().float().to(W.device) if mass is not None and mass.numel() == k
             else torch.full((k,), 1.0 / k, device=W.device))
        rr.append(float((W * w.reshape(k, 1, 1)).sum(0).norm()) / wb)
        fn = f / n.clamp_min(1e-12).unsqueeze(1)
        C = fn @ fn.t()
        cos.append(float((C.sum() - C.diag().sum()) / (k * (k - 1))))
        p = (w / w.sum().clamp_min(1e-12)).cpu().numpy()
        ents.append(float(-(p * np.log(p + 1e-12)).sum() / math.log(k)))
    if not rr:
        return None
    out = {"realised_ratio": float(np.mean(rr)), "expert_ratio": float(np.mean(er)),
           "cos_mean": float(np.mean(cos)), "soft_entropy": float(np.mean(ents))}
    out["inert"] = bool(out["realised_ratio"] < INERT_RATIO)
    return out


def _ore_health(model, sample=24, sites=4):
    """Same three numbers for OREResidual, whose deltas live in (E, d_in, r) parameters.

    `mixture_health` returned None for every ORE run because it only knows MoEResidual, so the
    repo's own inert-mixture safeguard was silently off for the whole method. Experts and sites
    are SUBSAMPLED: K=128 full deltas at 768x3072 would be ~1.2 GB.
    """
    import math

    from .ore import ore_modules
    # FFNMoE subclasses OREResidual for its router and owns no A/B, so it lands here and would
    # raise. Its health check is expert_divergence(), reported through delta_telemetry.
    mods = [m for m in ore_modules(model) if hasattr(m, "A")]
    if not mods:
        return None
    cos, er, rr, ents = [], [], [], []
    for m in mods[:sites]:
        k = min(sample, m.k)
        sel = torch.randperm(m.k, device=m.A.device)[:k]
        W = torch.einsum("edr,erf->edf", m.A.detach().float()[sel], m.B.detach().float()[sel])
        wb = float(m.base.weight.detach().float().norm())
        if wb < 1e-12:
            continue
        f = W.reshape(k, -1)
        n = f.norm(dim=1)
        er.append(float(n.mean()) / wb)
        u = m.utilisation
        w = (u.detach().float()[sel] if u is not None and u.numel() == m.k
             else torch.full((k,), 1.0 / k, device=W.device))
        w = w / w.sum().clamp_min(1e-12)
        rr.append(float((W * w.reshape(k, 1, 1)).sum(0).norm()) / wb)
        fn = f / n.clamp_min(1e-12).unsqueeze(1)
        C = fn @ fn.t()
        cos.append(float((C.sum() - C.diag().sum()) / (k * (k - 1))) if k > 1 else float("nan"))
        p = w.cpu().numpy()
        ents.append(float(-(p * np.log(p + 1e-12)).sum() / math.log(k)) if k > 1 else 1.0)
    if not rr:
        return None
    out = {"realised_ratio": float(np.mean(rr)), "expert_ratio": float(np.mean(er)),
           "cos_mean": float(np.mean(cos)), "soft_entropy": float(np.mean(ents))}
    out["inert"] = bool(out["realised_ratio"] < INERT_RATIO)
    st = mods[0].load_stats()
    if st:
        out.update({k: v for k, v in st.items() if k != "hist"})
    return out


def health_banner(h):
    """One line, plus a loud block when the mixture cannot possibly matter."""
    if h is None:
        return ""
    line = (f"[health] realised {h['realised_ratio']*100:.3f}% of base | per-expert "
            f"{h['expert_ratio']*100:.3f}% | cos {h['cos_mean']:+.3f} | "
            f"soft-entropy {h['soft_entropy']:.3f}")
    if "effective_experts" in h:
        line += (f" | USING {h['effective_experts']:.1f} experts, cv {h['cv']:.2f}, "
                 f"{h['dead']} dead, top {h['max_share']*100:.1f}%")
    if not h["inert"]:
        return line
    return line + (
        "\n" + "!" * 78 +
        f"\n!! INERT MIXTURE: experts move the weights by {h['realised_ratio']*100:.3f}%, "
        f"under the {INERT_RATIO*100:.1f}% floor.\n"
        "!! Any accuracy number from this run describes the BASE model, not the mixture.\n"
        "!! Do NOT interpret a null from this run as 'capacity does not help'.\n"
        + ("!! Cause is likely the gate: soft-entropy %.3f (1.0 = uniform mixing, which\n"
           "!! cancels near-orthogonal experts by ~sqrt(K); 0.0 = collapsed).\n"
           % h["soft_entropy"]) +
        "!" * 78)


def delta_parameters(model):
    """The trainable set: expert deltas and gates, never the frozen base."""
    ps = []
    for m in moe_modules(model):
        ps += [m.A.weight, m.B.weight]
        if not getattr(m.gate, "frozen_gate", False):
            ps += [p for p in m.gate.parameters()]
    return ps


def set_keys(model, key):
    """Broadcast the routing key to every wrapper before a forward pass."""
    from .ore import ore_modules
    for m in moe_modules(model) + ore_modules(model):
        m.set_key(key)


def _gated_modules(model):
    """MoEResidual keeps its router in .gate; OREResidual owns aux_loss/utilisation directly.
    Both must be collected or the ORE arm trains with NO load balancing and can collapse onto a
    handful of experts -- which would look like "the method does not work" rather than a bug.
    """
    from .ore import OREResidual
    out = [(m, m.gate) for m in moe_modules(model)]
    out += [(m, m) for m in model.modules() if isinstance(m, OREResidual)]
    return out


def aux_loss(model):
    """MEAN load-balancing loss over gated sites. Zero for prescribed routing.

    Mean, not sum: the Switch term is K*sum(f_i*P_i), which is 1.0 at uniform but up to K when
    collapsed. Summed over 16 sites at K=512 that reaches ~675 and swamps a task loss of ~0.4,
    so the coefficient would have to be retuned for every K. Averaging makes aux_coef
    K-independent: ~0.4 of pressure while collapsed, ~0.01 once balanced.
    """
    vals = [g.aux_loss for _, g in _gated_modules(model) if getattr(g, "aux_loss", None) is not None]
    if not vals:
        return None
    total = vals[0]
    for v in vals[1:]:
        total = total + v
    return total / len(vals)


def utilisation_entropy(model):
    """Mean normalised entropy of expert usage. 1.0 is uniform, 0.0 is fully collapsed.

    This is the mechanism check: the prediction is that learned routing's entropy falls as K
    grows while prescribed routing stays flat by construction.
    """
    import math
    vals = []
    for _, g in _gated_modules(model):
        u = getattr(g, "utilisation", None)
        if u is None:
            continue
        p = (u / u.sum().clamp_min(1e-9)).clamp_min(1e-9)
        vals.append(float(-(p * p.log()).sum() / math.log(len(p))))
    return sum(vals) / len(vals) if vals else float("nan")

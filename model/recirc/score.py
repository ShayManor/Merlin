#!/usr/bin/env python3
"""Closed-form scoring of recirculation (source, destination) pairs -- RECIRC-01.

The source paper (arXiv 2608.17981) picks its layer pair by a full grid sweep over all (s,d)
within 12 layers, ranked on tuning-set perplexity, and lists "optimal layers appear domain or
problem dependent" as a limitation. That sweep is O(L^2) full evaluations. This module scores
every pair from ONE forward and ONE backward pass instead.

Derivation. Injection at destination d perturbs the residual stream by

    dz_d = alpha * (f(z_s^{t-1}) - z_d^t)                                   (Eq. 1, beta=1-alpha)

so to first order the change in any downstream scalar L is <g_d, dz_d> with g_d = dL/dz_d.
One backward pass yields g_d at EVERY d at once, and one forward yields z_s at every s, so all
L^2 pairs reduce to inner products.

Because the injection recurs every frame, the effect compounds. The perturbation returns to d
next frame scaled by the loop gain G = alpha * gamma * rho(J_{d->s}), giving a geometric series
and the full predictor

    S(s,d) = <g_d, alpha * (f(z_s) - z_d)> / (1 - G)

Numerator: the injection must carry information the destination lacks and the head must be
sensitive to it. Denominator: the loop must be subcritical, and the score diverges exactly at
the stability boundary G -> 1. Negative S predicts loss reduction.

Validity: first order in alpha. The paper operates at alpha 0.07-0.15 with a convex mixture,
which is the regime where this holds; check `linearity_residual` before trusting a ranking.

MERLIN is small enough (20 taps, 190 ordered pairs) to run the exhaustive grid and MEASURE the
ranking this predicts. That validation is the contribution; a 12B model cannot afford it.
"""
import torch

from model.recirc.recirc import loop_gain, tap_list


def _norm_match(src, dst, mode="token"):
    if mode == "token":
        return src * (dst.norm(dim=-1, keepdim=True) / src.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    if mode == "global":
        return src * (dst.norm() / src.norm().clamp_min(1e-6))
    return src


def collect(model, forward_fn, retain_grad=False):
    """Run `forward_fn` once, returning {label: residual-stream tensor} at every tap.

    With `retain_grad`, the tensors are kept in the autograd graph and `.grad` is populated by
    a subsequent backward, which is what gives g_d at every destination from one pass.
    """
    out, hooks = {}, []

    def mk(label):
        def hook(mod, inp, o):
            t = o[0] if isinstance(o, tuple) else o
            if retain_grad:
                t.retain_grad()
                out[label] = t
            else:
                out[label] = t.detach()
            return o
        return hook

    for label, mod in tap_list(model):
        hooks.append(mod.register_forward_hook(mk(label)))
    try:
        res = forward_fn()
    finally:
        for h in hooks:
            h.remove()
    return out, res


def score_pairs(model, forward_prev, forward_cur, loss_fn, alpha=0.10, norm="token",
                gains=None, max_pairs=None):
    """Score every ordered (source, destination) pair from one forward + one backward.

    forward_prev  callable running frame t-1; its tap activations supply z_s
    forward_cur   callable running frame t, returning whatever loss_fn consumes
    loss_fn       maps forward_cur's output to a scalar to differentiate
    gains         optional {(s,d): G} from `loop_gain`; omitted pairs use G=0 (no compounding)

    Returns a list of dicts sorted by predicted improvement (most negative S first).
    """
    labels = [l for l, _ in tap_list(model)]

    with torch.no_grad():
        prev_z, _ = collect(model, forward_prev)

    cur_z, out = collect(model, forward_cur, retain_grad=True)
    loss = loss_fn(out)
    model.zero_grad(set_to_none=True)
    loss.backward()

    rows = []
    for di, d in enumerate(labels):
        zd, gd = cur_z[d], cur_z[d].grad
        if gd is None:
            continue
        for s in labels[di + 1:]:
            zs = prev_z[s]
            n = min(zs.shape[1], zd.shape[1])
            src = _norm_match(zs[:, :n].to(zd.dtype), zd[:, :n].detach(), norm)
            dz = alpha * (src - zd[:, :n].detach())
            first = float((gd[:, :n] * dz).sum())
            G = float((gains or {}).get((s, d), 0.0))
            rows.append({"source": s, "dest": d, "first_order": first, "G": G,
                         "score": first / max(1e-3, 1.0 - G)})
            if max_pairs and len(rows) >= max_pairs:
                break
    rows.sort(key=lambda r: r["score"])
    return rows


def all_gains(model, forward_fn, alpha=0.10, norm="token", pairs=None, iters=8):
    """Loop gain G for each pair. O(L^2) cheap forwards, no evaluation and no autograd."""
    labels = [l for l, _ in tap_list(model)]
    pairs = pairs or [(s, d) for di, d in enumerate(labels) for s in labels[di + 1:]]
    return {(s, d): loop_gain(model, forward_fn, s, d, alpha, norm, iters)["G"]
            for s, d in pairs}


def linearity_residual(model, forward_cur, loss_fn, source, dest, alpha, prev_state,
                       norm="token"):
    """How well does the first-order term predict the ACTUAL loss change at this alpha?

    Returns (predicted, actual). The predictor is only usable where these track; if they
    diverge the mixture is outside the linear regime and the ranking should not be trusted.
    """
    from model.recirc.recirc import Recirculator

    with torch.no_grad():
        base = float(loss_fn(forward_cur()))
    cur_z, out = collect(model, forward_cur, retain_grad=True)
    loss = loss_fn(out)
    model.zero_grad(set_to_none=True)
    loss.backward()
    zd, gd = cur_z[dest], cur_z[dest].grad
    n = min(prev_state.shape[1], zd.shape[1])
    src = _norm_match(prev_state[:, :n].to(zd.dtype), zd[:, :n].detach(), norm)
    pred = float((gd[:, :n] * (alpha * (src - zd[:, :n].detach()))).sum())

    rec = Recirculator(model, source, dest, alpha=alpha, norm=norm, mode="shuffle")
    rec.set_state(prev_state)
    with torch.no_grad():
        actual = float(loss_fn(forward_cur())) - base
    rec.remove()
    return pred, actual

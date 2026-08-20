#!/usr/bin/env python3
"""One cell of the crossing-curves grid.

Spec: docs/superpowers/specs/2026-08-08-rmoe-00-crossing-curves.md

Site is `dpt_regressor_head.conv2.0` (Conv2d 128->128 k3), the final conv block before the
prediction layer `conv2.2` (128->6), which is MoE3D's location.

Arms:
  D       dense student, untrained (the init reference)
  M       MERLIN as deployed: student_nf.pt, the global near-field fine-tune, untrained here.
          The bar the experts have to clear, evaluated identically to every other arm.
  F       dense fine-tune of the SAME site, no experts (the control that separates "experts"
          from "you trained this layer"; 147k params vs K=8's 82k, so it is not starved)
  L_top1  learned top-1 router with load-balancing loss
  L_soft  learned soft router (stronger baseline; cannot starve experts)
  P       prescribed range bins, keyed on the FROZEN base model's own predicted depth
  O       prescribed range bins, keyed on ground-truth depth (oracle, not deployable)
  R       gated by a GT-SUPERVISED bin classifier (model/moe/train_router.py). Routes at 63.3%
          against the depth-keyed gate's 44.0%, which was itself below a constant guess.

Everything except the deltas and the gate is frozen, so every run starts bit-identical to the
init checkpoint and is a measured delta against it.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "backbones"))

from data import gather_frames, load_gt_depth, make_views          # noqa: E402
from student import build_student                                   # noqa: E402
from moe.residual import PrescribedRangeGate                        # noqa: E402
from moe.ore import OREResidual, ore_modules, ore_z_loss, set_route_feature  # noqa: E402
from moe.upcycle import (  # noqa: E402
    aux_loss, delta_parameters, health_banner, mixture_health, moe_modules, set_keys, upcycle,
    utilisation_entropy,
)

SITE = "dpt_regressor_head.conv2.0"

BINS = [(0.2, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 1.8), (1.8, 2.5)]
BOUNDARY_THRESH = 0.10        # |grad log-depth| defining a boundary pixel; fixed before any arm
FALSE_NEAR_MARGIN = 0.15      # metres a prediction must sit IN FRONT of truth to count as phantom


# ------------------------------------------------------------------ model

def load_pairs(root, stride=1):
    """(rgb, depth) pairs from either a TUM sequence root or a Habitat render directory.

    Habitat renders (rgb_%06d.png + dep_%06d.npy, written by eval/drone/gen_nearfield.py) are
    the only source with dense sub-0.5 m ground truth: every structured-light indoor dataset
    has a ~0.5 m minimum range, so TUM/NYUv2/ScanNet cannot see the near-field regime at all.
    """
    rgbs = sorted(glob.glob(os.path.join(root, "rgb_*.png")))
    if rgbs:
        pairs = [(r, r.replace("rgb_", "dep_").replace(".png", ".npy")) for r in rgbs]
        # Drop pairs whose depth is missing. A partially copied directory otherwise kills a
        # 25,000-step run at its first eval, hours in -- which is exactly what happened when six
        # ord/val depths failed to transfer.
        pairs = [p for p in pairs if os.path.exists(p[1])]
        return [p for i, p in enumerate(pairs) if i % stride == 0]
    return gather_frames([root], stride=stride, with_depth=True)


def load_depth(path, H, W):
    if path.endswith(".npy"):
        import cv2
        d = np.load(path).astype(np.float32)
        if d.shape != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        return d
    return load_gt_depth(path, H, W)


SRC_DEPTH = 8                                    # what student_hab_nav was distilled at


def load_model(init, dev, aat_depth=SRC_DEPTH):
    """Load the distilled student, optionally DEEPENED by identity blocks.

    The dense counterpart to adding experts is adding layers, so the two have to start from
    the same function or the comparison is rigged. DeRS-upcycling is function-preserving at
    init (B_i = 0); this makes depth-upcycling function-preserving too.

    New blocks are APPENDED after the last source block and their LayerScale gammas zeroed,
    which makes each one exactly the identity (x + 0 * attn(norm(x)) == x). Appending rather
    than interleaving is what keeps it exact: info_sharing.indices = [3, 5] taps intermediate
    AAT features for the DPT head, so inserting anywhere before layer 5 would move what those
    taps see. The cost is that the added depth only refines the final representation and not
    the tapped ones.
    """
    m = build_student(size="base", aat_depth=aat_depth, device="cpu")
    ck = torch.load(init, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"] if "state_dict" in ck else ck)
    if aat_depth > SRC_DEPTH:
        pre = "info_sharing.self_attention_blocks"
        for j in range(SRC_DEPTH, aat_depth):
            src = j % SRC_DEPTH                   # cycle the source blocks for a varied init
            for k in [k for k in sd if k.startswith(f"{pre}.{src}.")]:
                tail = k[len(f"{pre}.{src}."):]
                sd[f"{pre}.{j}.{tail}"] = sd[k].clone()
            for g in ("ls1.gamma", "ls2.gamma"):
                sd[f"{pre}.{j}.{g}"] = torch.zeros_like(sd[f"{pre}.{src}.{g}"])
        print(f"[cell] depth-upcycled {SRC_DEPTH} -> {aat_depth} AAT blocks, "
              f"{aat_depth - SRC_DEPTH} identity-initialised", flush=True)
    m.load_state_dict(sd)
    return m.to(dev)


def widen_ffn(model, extra):
    """Grow the AAT FFN hidden dim by `extra`, function-preservingly.

    The depth-upcycled arm can only add capacity AFTER the last source block, because
    info_sharing.indices taps layers 3 and 5 for the DPT head. DeRS has no such restriction:
    it edits all 16 FFN Linears in place. That makes depth a confounded control -- it differs
    from the mixture in WHERE the capacity sits as well as whether it is conditional. Widening
    puts dense capacity at exactly the mixture's sites, so the only remaining difference is
    conditionality.

    New fc1 rows are copied from existing ones with noise (identical rows would stay tied
    under a shared gradient); new fc2 columns are ZERO, so the block's output is unchanged at
    init and the model still starts as base:8.
    """
    import torch.nn as nn
    blocks = model.info_sharing.self_attention_blocks
    for blk in blocks:
        fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
        hid, din = fc1.weight.shape
        dout = fc2.weight.shape[0]
        n1 = nn.Linear(din, hid + extra, bias=fc1.bias is not None)
        n2 = nn.Linear(hid + extra, dout, bias=fc2.bias is not None)
        with torch.no_grad():
            src = torch.randint(0, hid, (extra,))
            n1.weight.copy_(torch.cat([fc1.weight, fc1.weight[src] * 1.0], 0))
            n1.weight[hid:] += torch.randn_like(n1.weight[hid:]) * 0.01
            if fc1.bias is not None:
                n1.bias.copy_(torch.cat([fc1.bias, fc1.bias[src]], 0))
            n2.weight.zero_()
            n2.weight[:, :hid].copy_(fc2.weight)
            if fc2.bias is not None:
                n2.bias.copy_(fc2.bias)
        blk.mlp.fc1, blk.mlp.fc2 = n1, n2
    print(f"[cell] widened {len(blocks)} AAT FFNs by {extra} hidden units "
          f"(fc2 zero-init, function-preserving)", flush=True)
    return model


@torch.no_grad()
def forced_expert_matrix(model, pairs, size, dev, n):
    """Per-bin error with every token forced onto each expert in turn.

    Run DURING training, not after a grid has been flown. If the spread across experts is ~0
    the experts are interchangeable or too small to matter, and the run's accuracy number says
    nothing about the mixture. That failure once survived a whole 6-seed grid undetected.
    """
    mods = moe_modules(model)
    ore = [m for m in ore_modules(model) if hasattr(m, "A")]   # FFNMoE has no forced-expert path
    if not mods and ore:
        return forced_expert_matrix_ore(model, pairs, size, dev, n)
    k = mods[0].num_experts
    saved = [m.gate for m in mods]
    model.eval()

    def score():
        acc = [[] for _ in BINS]
        for r, d in pairs[:n]:
            z = predict_depth(model, [r], size, dev)[0]
            gt = torch.from_numpy(load_depth(d, *z.shape)).float().to(dev)
            v = torch.isfinite(z) & torch.isfinite(gt) & (gt > 0.1) & (gt < 10) & (z > 0.05)
            if v.sum() < 200:
                continue
            pa = z * torch.median(gt[v] / z[v])
            for j, (lo, hi) in enumerate(BINS):
                b = v & (gt >= lo) & (gt < hi)
                if b.sum() > 100:
                    acc[j].append(float(((pa[b] - gt[b]).abs() / gt[b]).mean()))
        return [float(np.mean(a)) if a else float("nan") for a in acc]

    real = score()
    rows = []
    for i in range(k):
        for m in mods:
            m.gate = _ForcedGate(m.num_experts, i, m.eaxis).to(dev)
        rows.append(score())
    for m, g in zip(mods, saved):
        m.gate = g
    ex = np.array(rows)
    spread = float(np.nanmean(np.nanmax(ex, 0) - np.nanmin(ex, 0)))
    winners = len(set(np.nanargmin(ex, 0).tolist()))
    print(f"[cell] [matrix] expert spread {spread:.5f} across bins, {winners}/{len(BINS)} "
          f"distinct winners, real-gate mean {float(np.nanmean(real)):.4f}", flush=True)
    if spread < 1e-3:
        print("[cell] !! experts are INTERCHANGEABLE (spread ~0): forcing any single expert "
              "reproduces the real gate. The mixture is not routing.", flush=True)
    return {"real": real, "experts": rows, "spread": spread, "distinct_winners": winners}


@torch.no_grad()
def forced_expert_matrix_ore(model, pairs, size, dev, n, max_experts=16):
    """Same diagnostic for OREResidual, which routes internally rather than through a .gate module.

    Forces every token onto expert i by overriding route_mode, then measures per-bin error. If the
    spread across experts is ~0 the experts are interchangeable and the mixture is not doing
    per-token work, whatever the accuracy number says. Experts are SUBSAMPLED at large K: the cost
    is one full val pass per expert.
    """
    ore = ore_modules(model)
    if not ore:
        return {}
    k = ore[0].k
    sel = list(range(k)) if k <= max_experts else \
        [int(round(i * (k - 1) / (max_experts - 1))) for i in range(max_experts)]
    saved = [(m.route_mode, getattr(m, "_forced", None)) for m in ore]
    model.eval()

    def score():
        acc = [[] for _ in BINS]
        for r, d in pairs[:n]:
            z = predict_depth(model, [r], size, dev)[0]
            gt = torch.from_numpy(load_depth(d, *z.shape)).float().to(dev)
            v = torch.isfinite(z) & torch.isfinite(gt) & (gt > 0.1) & (gt < 10) & (z > 0.05)
            if v.sum() < 200:
                continue
            pa = z * torch.median(gt[v] / z[v])
            for j, (lo, hi) in enumerate(BINS):
                b = v & (gt >= lo) & (gt < hi)
                if b.sum() > 100:
                    acc[j].append(float(((pa[b] - gt[b]).abs() / gt[b]).mean()))
        return [float(np.mean(a)) if a else float("nan") for a in acc]

    real = score()
    rows = []
    for i in sel:
        for m in ore:
            m.route_mode, m._forced = "forced", i
        rows.append(score())
    for m, (rm, fo) in zip(ore, saved):
        m.route_mode, m._forced = rm, fo
    ex = np.array(rows)
    spread = float(np.nanmean(np.nanmax(ex, 0) - np.nanmin(ex, 0)))
    winners = len(set(np.nanargmin(ex, 0).tolist()))
    print(f"[cell] [matrix] expert spread {spread:.5f} across bins, {winners}/{len(BINS)} "
          f"distinct winners over {len(sel)} of {k} experts, real-gate mean "
          f"{float(np.nanmean(real)):.4f}", flush=True)
    if spread < 1e-3:
        print("[cell] !! experts are INTERCHANGEABLE (spread ~0): forcing any single expert "
              "reproduces the real gate. The mixture is not routing.", flush=True)
    return {"real": real, "experts": rows, "which": sel, "spread": spread,
            "distinct_winners": winners}


class _ForcedGate(torch.nn.Module):
    """Sends every token to one expert, in whatever layout the site uses."""

    def __init__(self, num_experts, idx, eaxis):
        super().__init__()
        self.num_experts, self.idx, self.eaxis = num_experts, idx, eaxis
        self.aux_loss = self.utilisation = self.soft_mass = None

    def forward(self, x, key=None):
        ax = self.eaxis % x.dim()
        shape = list(x.shape)
        shape[ax] = self.num_experts
        g = torch.zeros(shape, device=x.device, dtype=x.dtype)
        g.index_fill_(ax, torch.tensor([self.idx], device=x.device), 1.0)
        return g


def predict_depth(model, paths, size, dev, key=None, batch=1):
    view = make_views(paths, size, dev)
    if key is not None:
        set_keys(model, key)
    pr = model(view, memory_efficient_inference=True, minibatch_size=batch)[0]
    return pr["pts3d_cam"][..., 2].float()


# ------------------------------------------------------------------ routing keys

def key_path(cache, rgb):
    return os.path.join(cache, os.path.abspath(rgb).replace("/", "_") + ".npy")


@torch.no_grad()
def precompute_keys(model, pairs, cache, size, dev):
    """The frozen base model's own depth prediction, cached once per image.

    This is the arm-P routing key. It is deterministic because the trunk is frozen, and caching
    it keeps training single-pass. Deployment would need a cheap key instead; that is explicitly
    out of scope for this experiment.
    """
    os.makedirs(cache, exist_ok=True)
    model.eval()
    todo = [(r, d) for r, d in pairs if not os.path.exists(key_path(cache, r))]
    for i, (r, _) in enumerate(todo):
        z = predict_depth(model, [r], size, dev)[0]
        np.save(key_path(cache, r), z.cpu().numpy().astype(np.float16))
        if i % 200 == 0:
            print(f"[keys] {i}/{len(todo)}", flush=True)
    return len(todo)


def load_key(cache, rgb, dev):
    return torch.from_numpy(np.load(key_path(cache, rgb)).astype(np.float32)).to(dev)


def gt_key(dep, hw, dev):
    return torch.from_numpy(load_depth(dep, *hw)).float().to(dev)


def batch_key(arm, cache, paths, deps, hw, dev, cal=None):
    if arm == "P":
        k = torch.stack([apply_calibration(load_key(cache, r, dev), cal) for r in paths], 0)
    elif arm == "O":
        k = torch.stack([gt_key(d, hw, dev) for d in deps], 0)
    else:
        return None
    return k.unsqueeze(1)                       # (B,1,H,W)


# ------------------------------------------------------------------ bin edges

def fit_key_calibration(cache, pairs, hw, dev, sample=300, nq=33):
    """Monotone quantile map from PREDICTED depth to TRUE depth.

    The raw key is biased about 2x high (training-set quartiles: predicted 1.16/1.84/2.71
    against true 0.59/0.90/1.54). Binning it directly puts every boundary far beyond the near
    field, so the gate partitions "what the model believes about range" rather than range, and
    62.7% of truly-near pixels land in one expert. This maps predicted percentiles onto true
    percentiles so the key is an unbiased range estimate before it is binned.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pairs), min(sample, len(pairs)), replace=False)
    pv, tv = [], []
    for i in idx:
        r, d = pairs[i]
        pk = load_key(cache, r, dev).flatten()
        gt = gt_key(d, hw, dev).flatten()
        m = torch.isfinite(pk) & torch.isfinite(gt) & (pk > 0.1) & (gt > 0.1) & (gt < 10)
        if m.sum() < 500:
            continue
        j = torch.randint(0, int(m.sum()), (2000,), device=pk.device)
        pv.append(pk[m][j].cpu()); tv.append(gt[m][j].cpu())
    P, T = torch.cat(pv).numpy(), torch.cat(tv).numpy()
    qs = np.linspace(1, 99, nq)
    return np.percentile(P, qs), np.percentile(T, qs)


def apply_calibration(key, cal):
    """Piecewise-linear quantile map in log space, applied to a predicted-depth key."""
    if cal is None:
        return key
    pq, tq = cal
    x = key.clamp_min(1e-3).log()
    xp = torch.as_tensor(np.log(pq), dtype=x.dtype, device=x.device)
    fp = torch.as_tensor(np.log(tq), dtype=x.dtype, device=x.device)
    i = torch.bucketize(x, xp).clamp(1, len(xp) - 1)
    x0, x1, y0, y1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    t = ((x - x0) / (x1 - x0).clamp_min(1e-6)).clamp(0, 1)
    return (y0 + t * (y1 - y0)).exp()


def centres_from_edges(edges_m, lo=0.2, hi=3.0):
    """Explicit bin boundaries -> gate centres (geometric midpoints), in log-depth."""
    e = [lo] + list(edges_m) + [hi]
    return torch.tensor([float(np.log(np.sqrt(e[i] * e[i + 1]))) for i in range(len(e) - 1)])


def fit_centres(arm, cache, pairs, hw, dev, k, sample=400):
    """Equal-population centres in log-depth, frozen from the TRAINING set only."""
    vals = []
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pairs), min(sample, len(pairs)), replace=False)
    for i in idx:
        r, d = pairs[i]
        v = load_key(cache, r, dev) if arm == "P" else gt_key(d, hw, dev)
        v = v[torch.isfinite(v) & (v > 0.1) & (v < 10)]
        if v.numel():
            vals.append(v[torch.randint(0, v.numel(), (2000,), device=v.device)])
    return PrescribedRangeGate.quantile_centres(torch.cat(vals).cpu(), k)


# ------------------------------------------------------------------ loss and eval

def si_loss(pred, gt, pix_w=None, anchor_shift=False, metric=False):
    """Scale-invariant log-L1. `pix_w` optionally reweights pixels (bin-balanced objective).

    `anchor_shift` fits the per-image shift over ALL valid pixels instead of the weighted set.
    Required whenever `pix_w` masks pixels out entirely: otherwise the scale is estimated from
    the surviving subset only, nothing constrains the global scale, and the model drifts. Eval
    refits scale on the whole frame, so a bin-masked shift is a train/eval mismatch that made
    bin specialists score worse than the untrained model on their own bin.
    """
    v = torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0.1) & (gt < 8.0) & (pred > 0.05)
    if v.sum() < 200:
        return None
    diff = pred.clamp_min(0.05).log() - gt.clamp_min(0.05).log()
    w = v.float() if pix_w is None else v.float() * pix_w
    if w.sum() < 200:
        return None
    if metric:
        # METRIC: no per-image shift, so absolute scale is supervised. The default path fits and
        # removes the shift, which makes the objective (and the eval that mirrors it) blind to
        # scale -- fine for relative depth, but it cannot support a metric claim or feed the
        # occupancy map, which consumes metres.
        return (w * diff.abs()).sum() / w.sum().clamp_min(1e-6)
    sw = v.float() if anchor_shift else w
    shift = (sw * diff).sum() / sw.sum().clamp_min(1e-6)
    return (w * (diff - shift).abs()).sum() / w.sum().clamp_min(1e-6)


def bin_edges_from_centres(centres):
    """Bin boundaries are midpoints between adjacent gate centres, in log-depth."""
    c = np.asarray(centres, dtype=float)
    mids = (c[:-1] + c[1:]) / 2
    return np.exp(np.concatenate([[-np.inf], mids, [np.inf]]))


def fit_bin_shares(pairs, edges, hw, sample=300):
    """Share of valid TRUE-depth pixels falling in each routing bin, over the training set.

    Under a uniform objective the expert owning the sparsest bin receives a proportional share
    of the gradient and so has almost no pressure to be good at its own job. These shares are
    what the bin-balanced loss divides out.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pairs), min(sample, len(pairs)), replace=False)
    tot = np.zeros(len(edges) - 1)
    for i in idx:
        d = load_depth(pairs[i][1], *hw)
        d = d[np.isfinite(d) & (d > 0.1) & (d < 8.0)]
        if d.size:
            tot += np.histogram(d, bins=edges)[0]
    return tot / max(tot.sum(), 1)


def pixel_weights(gt, edges, shares, dev):
    """1 / share(bin of this pixel), normalised to mean 1 over the batch."""
    e = torch.as_tensor(edges[1:-1], dtype=torch.float32, device=dev)
    inv = torch.as_tensor(1.0 / np.clip(shares, 1e-6, None), dtype=torch.float32, device=dev)
    w = inv[torch.bucketize(gt, e)]
    return w / w.mean().clamp_min(1e-6)


@torch.no_grad()
def calibrate_routers(model, mods, pairs, size, dev, n_frames, kmeans=False, seed=0, warm=False,
                      ema=1.0):
    """One forward pass over real frames to fit each site's router statistics.

    Has to run on the WRAPPED model: the statistics of block j's input depend on what blocks
    0..j-1 do, and every one of them is now an ORE site.
    """
    # A fixed-partition run still needs the ceiling, but 'key' routing would raise here: the
    # calibration forward carries no routing key. Route by hash for the duration of the pass.
    saved = [m.route_mode for m in mods]
    for m in mods:
        if m.route_mode == "key":
            m.route_mode = "hash"
    acts, hs = {i: [] for i in range(len(mods))}, []
    for i, m in enumerate(mods):
        if m.route_dim:
            hs.append(m.register_forward_pre_hook(
                lambda mod, inp, i=i: acts[i].append(
                    mod._route_feat.detach().reshape(-1, mod.route_dim))))
        else:
            hs.append(m.register_forward_pre_hook(
                lambda mod, inp, i=i: acts[i].append(
                    inp[0].detach().reshape(-1, inp[0].shape[-1]))))
    model.eval()
    for s in range(0, min(n_frames, len(pairs)), 4):
        predict_depth(model, [p[0] for p in pairs[s:s + 4]], size, dev, batch=4)
    for h in hs:
        h.remove()
    dead, ceiling = [], None
    for i, m in enumerate(mods):
        x = torch.cat(acts[i], 0)
        acts[i] = None
        if i == 0:
            ceiling = route_ceiling(x, m.k, batch_tokens=x.shape[0] * 4 // n_frames)
        if kmeans:
            dead.append(m.calibrate_router(x, seed=seed, warm=warm, ema=ema))
        else:
            m.in_mean.mul_(1 - ema).add_(x.float().mean(0), alpha=ema)
            m.in_std.mul_(1 - ema).add_(x.float().std(0).clamp_min(1e-6), alpha=ema)
    for m, rm in zip(mods, saved):
        m.route_mode = rm
    model.train()
    torch.cuda.empty_cache()
    print(f"[cell] router calibrated on {min(n_frames, len(pairs))} frames, "
          f"achievable effective experts at this batch size {ceiling:.1f}"
          + (f", kmeans dead/site: min {min(dead)} max {max(dead)}" if dead else ""), flush=True)
    return ceiling


@torch.no_grad()
def route_ceiling(x, k, batch_tokens, iters=20):
    """How many experts COULD a good partition use on ONE batch of these activations?

    The abort gate cannot be a fixed fraction of K. Measured: a single frame's 729 tokens reach
    only 13-14 of 128 experts and a 4-frame batch ~30, because the token cloud has effective rank
    ~9 and 75% of its variance is within-frame. A K/4 gate would therefore abort every K=128 run
    at batch 4 no matter how healthy it was. k-means on the real activations gives the reference:
    the best a partition can do at this batch size, which is what a run should be judged against.
    """
    x = x.float()
    z = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
    c = z[torch.randperm(len(z), device=z.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(z, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = z[m].mean(0)
    # CONTIGUOUS, not a random draw: tokens sampled across 64 frames are far more diverse than
    # one real batch (108.6 vs ~30 effective experts at K=128), which would set the gate above
    # anything a run could reach.
    sub = z[:max(batch_tokens, 1)]
    cnt = torch.bincount(torch.cdist(sub, c).argmin(1), minlength=k).float()
    p = (cnt / cnt.sum()).clamp_min(1e-12)
    return float(k ** float(-(p * p.log()).sum() / np.log(k)))


def delta_telemetry(model):
    """Magnitude and pairwise cosine of the expert deltas. Turns "the deltas barely moved"
    into a measurement rather than an inference from the cosine alone."""
    import torch.nn.functional as F
    from moe.ffn_moe import ffn_modules
    out = {}
    for m in ffn_modules(model):          # arm U: the copies, not a low-rank delta
        out = m.expert_divergence()
    for m in moe_modules(model):
        r, k = m.rank, m.num_experts
        B = m.B.weight.detach()
        blocks = [B[:, i * r:(i + 1) * r].flatten() for i in range(k)]
        cos = [float(F.cosine_similarity(blocks[i], blocks[j], dim=0))
               for i in range(k) for j in range(k) if i < j]
        out = {"b_norm": float(B.norm()), "a_norm": float(m.A.weight.detach().norm()),
               "cos_mean": float(np.mean(cos)) if cos else float("nan")}
    return out


def boundary_f1(pred, gt, valid):
    def edges(z):
        gy = (z[1:, :-1] - z[:-1, :-1]).abs()
        gx = (z[:-1, 1:] - z[:-1, :-1]).abs()
        return (gy + gx) > BOUNDARY_THRESH
    lp, lg = pred.clamp_min(0.05).log(), gt.clamp_min(0.05).log()
    m = valid[:-1, :-1]
    ep, eg = edges(lp) & m, edges(lg) & m
    tp = float((ep & eg).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / float(ep.sum()), tp / float(eg.sum())
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def evaluate(model, pairs, arm, cache, size, dev, n=200, cal=None, metric=False):
    model.eval()
    acc = {f"{lo}-{hi}": {"abs": [], "rel": [], "bias": []} for lo, hi in BINS}
    whole, fnear, bf1 = [], [], []
    for r, d in pairs[:n]:
        z = predict_depth(model, [r], size, dev,
                          key=batch_key(arm, cache, [r], [d], (size, size), dev, cal))[0]
        gt = torch.from_numpy(load_depth(d, *z.shape)).float().to(dev)
        v = torch.isfinite(z) & torch.isfinite(gt) & (gt > 0.1) & (gt < 10) & (z > 0.05)
        if v.sum() < 200:
            continue
        # metric=True reports depth AS PREDICTED, in metres. The default rescales per frame to
        # GT, which is the repo convention for relative depth but means scale error is invisible.
        pa = z if metric else z * torch.median(gt[v] / z[v])
        whole.append(float(torch.median((gt[v] - pa[v]).abs() / gt[v])))
        # Phantom near surfaces are what block a clear route. This is a per-frame stand-in for
        # the offline map gate (diag_depth clear-route blockage), which needs a voxel map built
        # over a whole trajectory rather than a single frame.
        fnear.append(float(((pa < gt - FALSE_NEAR_MARGIN) & v).float().sum() / v.float().sum()))
        bf1.append(boundary_f1(pa, gt, v))
        for lo, hi in BINS:
            b = v & (gt >= lo) & (gt < hi)
            if b.sum() > 100:
                e = pa[b] - gt[b]                              # +ve = predicted FARTHER than truth
                a = acc[f"{lo}-{hi}"]
                # MEDIAN, matching eval/drone/bench_depth.py exactly. Using the mean here put
                # every published cell 1.3-2.3x high on an identical model; the repo's depth
                # convention is median per frame, then median across frames.
                a["abs"].append(float(e.abs().median()))
                a["rel"].append(float((e.abs() / gt[b]).median()))
                a["bias"].append(float(e.median()))
    model.train()
    out = {"whole_absrel": float(np.nanmedian(whole)) if whole else float("nan"),
           "false_near": float(np.nanmean(fnear)) if fnear else float("nan"),
           "boundary_f1": float(np.nanmean(bf1)) if bf1 else float("nan"),
           "n": len(whole), "bins": {}}
    for k, a in acc.items():
        out["bins"][k] = {m: (float(np.nanmedian(vv)) if vv else float("nan"))
                          for m, vv in a.items()} | {"n": len(a["abs"])}
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["D", "M", "F", "L_top1", "L_soft", "P", "O", "R", "ORE", "U"])
    ap.add_argument("--top-k", type=int, default=1,
                    help="arm U: experts per token. Gate weights are renormalised over the "
                         "selected experts, so init is function-preserving at any top-k.")
    ap.add_argument("--council-m", type=int, default=0,
                    help="arm U: experts pre-selected PER FRAME before token routing. 0 is free "
                         "token dispatch; M>=top_k restricts every token in a frame to the same "
                         "M experts, which is the frame-coherent dispatch of report.md.")
    ap.add_argument("--n-layers", type=int, default=2,
                    help="ORE butterfly layers per rotation. 0 makes ORE degenerate to DeRS.")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", action="store_true",
                    help="supervise and report ABSOLUTE depth: no per-image shift in the loss, "
                         "no median rescale at eval. Default is scale-invariant/relative.")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--val2-root", default="", help="second held-out set, unseen SCENES")
    ap.add_argument("--val2-stride", type=int, default=10)
    ap.add_argument("--val-stride", type=int, default=0,
                    help="val subsampling, independent of --stride. 0 keeps the legacy stride*3.")
    ap.add_argument("--gate-lr", type=float, default=None,
                    help="separate lr for the router. 0 freezes it at its calibrated init.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--aux-coef", type=float, default=0.01)
    ap.add_argument("--loss", choices=["uniform", "binbal"], default="uniform")
    ap.add_argument("--router-ckpt", default="/data/ckpt/router_head.pt")
    ap.add_argument("--aat-depth", type=int, default=SRC_DEPTH,
                    help="deepen the AAT stack with identity blocks before training. The "
                         "matched-LATENCY dense rival to experts: on the Jetson base:12 costs "
                         "179.5 ms against base:8 + DeRS at ~176 ms, for 14x more added params.")
    ap.add_argument("--diag-matrix-every", type=int, default=3000,
                    help="every N steps, force each expert on in turn and report per-bin error. "
                         "Catches an inert or non-routing mixture DURING the run instead of "
                         "after a whole grid has been flown against it. 0 disables.")
    ap.add_argument("--diag-matrix-n", type=int, default=12,
                    help="frames per forced-expert pass; K passes are run, so keep it small.")
    ap.add_argument("--b-init", choices=["zero", "random"], default="zero",
                    help="expert B init. zero is the LoRA/DeRS convention and is exactly "
                         "function-preserving, but it zeroes the gate's gradient so routing "
                         "never bootstraps. random starts experts non-degenerate.")
    ap.add_argument("--b-scale", type=float, default=0.01,
                    help="with --b-init random, each expert delta is scaled to this fraction "
                         "of the base weight norm.")
    ap.add_argument("--save-ckpt", default="",
                    help="write trained weights here. Needed for post-hoc diagnostics like the "
                         "forced-expert matrix; runs otherwise emit only JSON metrics.")
    ap.add_argument("--resume", default="",
                    help="continue from a checkpoint of THIS arm, loaded after upcycling so ORE "
                         "expert tensors match. Weights only, no optimiser state. Needed because "
                         "ORE K=32 fits ~166k steps in a 12 h wall against a 250k target.")
    ap.add_argument("--snapshot-steps", default="",
                    help="comma-separated steps at which to also write a step-tagged copy of "
                         "--save-ckpt. The plain --save-ckpt path is overwritten every eval, so "
                         "without this only the last step survives and C2/C4 cannot be run at a "
                         "fixed set of steps. Snapshots land on eval boundaries only.")
    ap.add_argument("--ffn-widen", type=int, default=0,
                    help="grow each AAT FFN hidden dim by this much. Dense capacity at the "
                         "mixture's own sites, so it isolates conditionality from placement.")
    ap.add_argument("--unfreeze-all", action="store_true",
                    help="train the WHOLE model, optionally with experts on top. F's ceiling "
                         "is a full-model fine-tune, which is what the deployed checkpoint is; "
                         "freezing the base caps both arms far below it.")
    ap.add_argument("--sites", default=None,
                    help="comma-separated module paths to wrap or fine-tune. Defaults to the "
                         "DPT conv. The AAT FFN blocks are the site where DeRS economics "
                         "actually work: deltas are ~5%% of the base GEMM there against ~90%% "
                         "at the full-resolution conv.")
    ap.add_argument("--calibrate-key", action="store_true",
                    help="quantile-map the predicted-depth key onto the true-depth "
                         "distribution before binning, so the gate partitions range rather "
                         "than the model's biased belief about range")
    ap.add_argument("--bin-edges", default=None,
                    help="explicit bin boundaries in metres, e.g. 0.5,0.8,1.2. Overrides "
                         "equal-population quantiles, which spend their resolution where the "
                         "specialists show no headroom")
    ap.add_argument("--bin-only", type=int, default=-1,
                    help="index into BINS. Masks the loss to that range bin only, giving the "
                         "layer its FULL capacity for a single regime. This is the upper bound "
                         "on what range specialisation can buy: a real MoE expert gets 1/K of "
                         "the parameters and shares the layer, so it cannot beat this.")
    ap.add_argument("--bins-k", type=int, default=4,
                    help="routing bins used by the bin-balanced loss; kept fixed "
                         "across arms so F and P optimise the same objective")
    ap.add_argument("--train-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--init", required=True)
    ap.add_argument("--keys", default="/data/keys")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-batches per optimiser step; effective batch = batch*accum. "
                         "Needed so tokens/expert is not 22.8 at K=128 (see TODO_ore_routing).")
    ap.add_argument("--router-norm", action="store_true",
                    help="whiten the router input from calibration statistics. The raw AAT "
                         "activation has ||mean|| 4-19x the token-to-token spread, so a linear "
                         "gate sees the same vector for every token and collapses at init.")
    ap.add_argument("--router-init", default="none", choices=("none", "kmeans"),
                    help="kmeans: put one gate row on each centroid of the calibration "
                         "activations (implies --router-norm)")
    ap.add_argument("--recalib-every", type=int, default=0,
                    help="re-fit router whitening + centroids every N steps, on CURRENT "
                         "activations. 0 disables. The activation distribution drifts as the "
                         "experts train, so a one-shot calibration goes stale.")
    ap.add_argument("--route-dim", type=int, default=0,
                    help="route every site on the frozen AAT input of this width instead of the "
                         "site's own (drifting) activation. 0 keeps local routing.")
    ap.add_argument("--recalib-ema", type=float, default=1.0,
                    help="blend rate toward the refit centroids/statistics. 1.0 replaces them "
                         "outright (disruptive); small values track drift continuously.")
    ap.add_argument("--router-cal-frames", type=int, default=64,
                    help="frames used for the router calibration pass")
    ap.add_argument("--route-mode", default="learned",
                    choices=("learned", "hash", "key", "key_gt"),
                    help="how tokens pick experts. hash/key are FIXED partitions: balanced by "
                         "construction, so they separate 'the router collapsed' from "
                         "'conditional capacity does not help here'. key=predicted depth "
                         "(deployable, arm P), key_gt=GT depth (oracle upper bound, arm O).")
    ap.add_argument("--balance", default="aux", choices=("aux", "bias", "bias+z", "z", "none"),
                    help="load-balancing mechanism for ORE")
    ap.add_argument("--bias-gamma", type=float, default=0.005,
                    help="DeepSeek-V3 selection-bias step, in units of the router logit spread")
    ap.add_argument("--z-coef", type=float, default=1e-3, help="ST-MoE router z-loss coefficient")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="also evaluate every N steps and record the curve, so validation can "
                         "be plotted against training rather than inferred from two endpoints")
    ap.add_argument("--precompute-only", action="store_true")
    args = ap.parse_args()

    sites = [x.strip() for x in args.sites.split(",")] if args.sites else [SITE]
    snap_steps = {int(x) for x in args.snapshot_steps.split(",") if x.strip()}
    if snap_steps and any(s % args.eval_every for s in snap_steps):
        raise SystemExit(f"--snapshot-steps {sorted(snap_steps)} must all be multiples of "
                         f"--eval-every {args.eval_every}; snapshots only land on eval boundaries")
    dev = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    size = (args.res // 14) * 14

    train = load_pairs(args.train_root, args.stride)
    # val stride is INDEPENDENT of train stride. It used to be stride*3, which is harmless while
    # every cell runs stride 1, but silently gives each cell a DIFFERENT val set the moment train
    # stride varies -- 129 / 43 / 19 frames at stride 1 / 3 / 7, so a data-scaling sweep would
    # compare arms on different, and increasingly noisy, held-out sets.
    val = load_pairs(args.val_root, args.val_stride if args.val_stride else args.stride * 3)
    # A SECOND held-out set from data the training union never touched at all. nearfield/val is
    # unseen frames from seen scenes; ord/ is a scene family absent from data/all/train entirely,
    # so it is the honest generalisation check. Both are evaluated at every eval.
    val2 = load_pairs(args.val2_root, args.val2_stride) if args.val2_root else []
    print(f"[cell] arm={args.arm} K={args.k} r={args.rank} seed={args.seed} "
          f"train={len(train)} val={len(val)}", flush=True)

    model = load_model(args.init, dev, aat_depth=args.aat_depth)
    if args.ffn_widen:
        widen_ffn(model, args.ffn_widen)
        model.to(dev)

    route_ceiling_val = None
    keyarm = {"key": "P", "key_gt": "O"}.get(args.route_mode) if args.arm == "ORE" else None
    if args.arm in ("P",) or keyarm == "P" or args.precompute_only:
        n = precompute_keys(model, train + val, args.keys, size, dev)
        print(f"[cell] precomputed {n} keys", flush=True)
        if args.precompute_only:
            return

    cal = None
    router_edges = None
    if args.calibrate_key and (args.arm == "P" or keyarm == "P"):
        cal = fit_key_calibration(args.keys, train, (size, size), dev)
        print(f"[cell] key calibrated: predicted quartiles "
              f"{[round(float(np.percentile(cal[0], q)), 2) for q in (25, 50, 75)]} -> true "
              f"{[round(float(np.percentile(cal[1], q)), 2) for q in (25, 50, 75)]}", flush=True)

    centres = None
    if args.arm in ("P", "O") or keyarm:
        if args.bin_edges:
            edges_m = [float(x) for x in args.bin_edges.split(",")]
            if len(edges_m) != args.k - 1:
                raise ValueError(f"{len(edges_m)} edges for K={args.k}; need {args.k - 1}")
            centres = centres_from_edges(edges_m)
        else:
            centres = fit_centres(keyarm or args.arm, args.keys, train, (size, size), dev, args.k)
        print(f"[cell] bin centres (m): "
              f"{[round(float(np.exp(c)), 3) for c in centres]}", flush=True)

    if args.arm == "M":
        params = []                        # evaluate the deployed model as-is
    elif args.arm == "F":
        if args.unfreeze_all:
            params = list(model.parameters())
            for p in params:
                p.requires_grad_(True)
            print(f"[cell] FULL-model dense fine-tune, "
                  f"trainable={sum(p.numel() for p in params):,}", flush=True)
        else:
            for p in model.parameters():
                p.requires_grad_(False)
            # honour --sites: the dense control has to train the SAME modules the mixture
            # wraps, or it is a different experiment at a different site
            params = []
            for path in sites:
                site = model
                for part in path.split("."):
                    site = site[int(part)] if part.isdigit() and not hasattr(site, part) \
                        else getattr(site, part)
                params += list(site.parameters())
            for p in params:
                p.requires_grad_(True)
            print(f"[cell] {len(sites)} site(s) dense-finetune trainable="
                  f"{sum(p.numel() for p in params)}", flush=True)
    elif args.arm == "U":
        # THE CONTROL ORE COULD NOT RUN: K full COPIES of each site's MLP, all trainable, so
        # active trainable params equal the dense fine-tune and total is K x. See
        # docs/superpowers/specs/2026-08-14-fcr-00-upcycling-control.md.
        from moe.ffn_moe import ffn_parameters, upcycle_ffn
        mods = upcycle_ffn(model, sites, num_experts=args.k, top_k=args.top_k,
                           council_m=args.council_m, balance=args.balance,
                           router_norm=args.router_norm or args.router_init == "kmeans",
                           route_dim=args.route_dim)
        if args.route_dim:
            blk0 = model.info_sharing.self_attention_blocks[0]
            blk0.register_forward_pre_hook(
                lambda m, i: set_route_feature(model, i[0].detach()))
        model.to(dev)
        route_ceiling_val = calibrate_routers(
            model, mods, train, size, dev, args.router_cal_frames,
            kmeans=args.router_init == "kmeans", seed=args.seed)
        for p in model.parameters():
            p.requires_grad_(False)
        params = ffn_parameters(model)
        for p in params:
            p.requires_grad_(True)
        tot = sum(m.total_params() for m in mods)
        act = sum(m.active_params() for m in mods)
        rest = sum(p.numel() for p in model.parameters()) - tot
        res_extra = {"ore_total": tot, "ore_active": act, "N_site": tot / act,
                     "model_total": rest + tot, "model_active": rest + act,
                     "N_model": (rest + tot) / (rest + act),
                     "expert_params": mods[0].expert_params()}
        print(f"[cell] UPCYCLE {len(mods)} site(s) K={args.k} top_k={args.top_k} "
              f"council={args.council_m} | trainable={sum(p.numel() for p in params):,} "
              f"active/token={act:,} | model total={(rest+tot)/1e6:.0f}M "
              f"N={(rest+tot)/(rest+act):.2f}", flush=True)
    elif args.arm == "ORE":
        # THE METHOD: shared frozen substrate + per-expert butterfly orientation + rank-r
        # residual, hard top-1. total >> active, unlike every other arm here.
        from moe.ore import ore_parameters, upcycle_ore
        mods = upcycle_ore(model, sites, num_experts=args.k, rank=args.rank, balance=args.balance,
                           n_layers=args.n_layers, freeze=True, centres_log=centres,
                           route_mode="key" if keyarm else args.route_mode,
                           router_norm=args.router_norm or args.router_init == "kmeans",
                           b_init=args.b_init, b_scale=args.b_scale,
                           route_dim=args.route_dim)
        if args.route_dim:
            # Capture the AAT stack's INPUT once per forward and hand it to every site's router.
            # That input is a function of the frozen encoder only, so it never drifts -- which is
            # precisely why site 0 keeps ~11 of 32 experts alive while deeper sites fall to 2.4.
            blk0 = model.info_sharing.self_attention_blocks[0]
            blk0.register_forward_pre_hook(
                lambda m, i: set_route_feature(model, i[0].detach()))
            print(f"[cell] routing on the FROZEN AAT input ({args.route_dim}-d), all sites",
                  flush=True)
        model.to(dev)
        # ALWAYS calibrate: even when no statistics are installed, the pass measures the
        # achievable-experts ceiling the abort gate is judged against.
        route_ceiling_val = calibrate_routers(
            model, mods, train, size, dev, args.router_cal_frames,
            kmeans=args.router_init == "kmeans", seed=args.seed)
        for p in model.parameters():
            p.requires_grad_(False)
        params = ore_parameters(model)
        if args.unfreeze_all:
            # also adapt the wrapped substrates: this is FT + experts, the config a practitioner
            # would ship, and the one that must beat plain FT for the method to be worth anything
            for m in mods:
                params += [m.base.weight] + ([m.base.bias] if m.base.bias is not None else [])
        for p in params:
            p.requires_grad_(True)
        tot = sum(m.total_params() for m in mods)
        act = sum(m.active_params() for m in mods)
        # N must be reported at MODEL level. The site-local ratio ignores the always-active rest
        # of the network and overstates N by ~4x (19.5 vs 5.1 at K=512), which would be a
        # flattering number to quote by accident.
        wrapped_base = sum(m.base.weight.numel() +
                           (m.base.bias.numel() if m.base.bias is not None else 0) for m in mods)
        rest = sum(p.numel() for p in model.parameters()) - tot
        model_tot, model_act = rest + tot, rest + act
        res_extra = {"ore_total": tot, "ore_active": act, "N_site": tot / act,
                     "model_total": model_tot, "model_active": model_act,
                     "N_model": model_tot / model_act}
        print(f"[cell] ORE {len(mods)} site(s) K={args.k} r={args.rank} n_l={args.n_layers} "
              f"| model total={model_tot/1e6:.0f}M active={model_act/1e6:.0f}M "
              f"N={model_tot/model_act:.2f} (site-local N={tot/act:.1f}) "
              f"| trainable={sum(p.numel() for p in params):,}", flush=True)
        if model_tot / model_act <= 1.0 and args.k > 1:
            # K=1 is DELIBERATELY dense -- it is the LoRA degenerate case and the only true
            # matched-active control for ORE (same active params, same structure, no mixture).
            # The guard exists to catch a config that should be sparse and silently is not, so
            # it must not block the one baseline that is dense on purpose.
            raise SystemExit("[cell] ABORT: model-level N <= 1 with K>1, this config is dense.")
    elif args.arm != "D":
        kind = {"L_top1": "learned_top1", "L_soft": "learned_soft",
                "P": "prescribed", "O": "prescribed", "R": "router"}[args.arm]
        router = router_edges = None
        if args.arm == "R":
            from train_router import RouterHead
            rc = torch.load(args.router_ckpt, map_location="cpu", weights_only=False)
            site = model
            for part in SITE.split("."):
                site = site[int(part)] if part.isdigit() and not hasattr(site, part) \
                    else getattr(site, part)
            router = RouterHead(site.in_channels, rc["k"])
            router.load_state_dict(rc["state_dict"])
            router_edges = rc["edges"]
            print(f"[cell] router gate from {args.router_ckpt}, edges {rc['edges']}", flush=True)
        mods = upcycle(model, sites, kind, num_experts=args.k, rank=args.rank,
                       centres_log=centres, freeze=not args.unfreeze_all, router=router,
                       b_init=args.b_init, b_scale=args.b_scale)
        model.to(dev)                          # wrappers are built on CPU, move them across
        if args.unfreeze_all:
            params = [p for p in model.parameters()]
            for p in params:
                p.requires_grad_(True)
            print(f"[cell] FULL-model fine-tune PLUS experts, "
                  f"trainable={sum(p.numel() for p in params):,}", flush=True)
        else:
            for p in model.parameters():
                p.requires_grad_(False)
            params = delta_parameters(model)
            for p in params:
                p.requires_grad_(True)
        print(f"[cell] {len(sites)} site(s) added="
              f"{sum(m.added_params() for m in mods)} trainable="
              f"{sum(p.numel() for p in params)}", flush=True)
    else:
        params = []

    res = {"route_ceiling": route_ceiling_val,
           "arm": args.arm, "k": args.k, "rank": args.rank, "seed": args.seed,
           "router_edges": router_edges if args.arm == "R" else None,
           "calibrated_key": bool(cal),
           "key_calibration": ({"pred_q": [float(x) for x in cal[0]],
                                "true_q": [float(x) for x in cal[1]]} if cal else None),
           "bin_edges_m": ([float(x) for x in args.bin_edges.split(",")]
                           if args.bin_edges else None),
           "sites": sites, "steps": args.steps, "lr": args.lr,
           "centres_log": [float(c) for c in centres] if centres is not None else None,
           "boundary_thresh": BOUNDARY_THRESH, "train_n": len(train), "val_n": len(val)}
    if args.arm in ("ORE", "U"):
        # the parameter audit was computed and then dropped on the floor; arm U's whole claim is
        # "active trainable params EQUAL dense", so it has to survive into the JSON
        res.update(res_extra)
        res["top_k"], res["council_m"] = args.top_k, args.council_m

    edges = shares = None
    bin_mask = None
    if args.bin_only >= 0:
        lo, hi = BINS[args.bin_only]
        bin_mask = (lo, hi)
        res["bin_only"] = [lo, hi]
        print(f"[cell] specialist: loss masked to {lo}-{hi} m only", flush=True)
    if args.loss == "binbal":
        bc = centres if centres is not None else fit_centres(
            "O", args.keys, train, (size, size), dev, args.bins_k)
        edges = bin_edges_from_centres(bc)
        shares = fit_bin_shares(train, edges, (size, size))
        res["loss_bin_edges_m"] = [float(e) for e in edges[1:-1]]
        res["loss_bin_shares"] = [float(x) for x in shares]
        print(f"[cell] bin-balanced loss, shares="
              f"{[round(float(x), 3) for x in shares]}", flush=True)
    res["loss"] = args.loss

    if args.resume:
        # --init loads the BASE student before any surgery, so it cannot carry an ORE checkpoint:
        # the expert tensors are not keys of the dense parent and the strict load fails. Loading
        # here instead, after upcycling, is what lets a run continue past the 12 h wall (ORE K=32
        # runs 0.26 s/step, i.e. ~166k steps per job against a 250k target).
        # NOTE this resumes WEIGHTS ONLY. The optimiser state is not restored, so expect a
        # transient at the join and do not read the first few evals as a continuation of the
        # previous curve. Apply it to every arm being compared, or the comparison is not matched.
        rk = torch.load(args.resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(rk["state_dict"], strict=False)
        if missing or unexpected:
            raise SystemExit(f"--resume mismatch: {len(missing)} missing, {len(unexpected)} "
                             f"unexpected; first missing {missing[:3]}, first unexpected "
                             f"{unexpected[:3]}")
        model = model.to(dev)
        print(f"[cell] RESUMED weights from {args.resume} "
              f"(step {rk.get('step')}); optimiser state NOT restored", flush=True)

    res["init_eval"] = evaluate(model, val, keyarm or args.arm, args.keys, size, dev, args.eval_n, cal,
                                metric=args.metric)
    print(f"[cell] init: {json.dumps(res['init_eval']['bins']['0.2-0.5'])} "
          f"whole={res['init_eval']['whole_absrel']:.4f}", flush=True)

    if args.arm in ("D", "M"):                             # no training, just the eval
        res["final_eval"] = res["init_eval"]
        json.dump(res, open(args.out, "w"), indent=2)
        return

    # The gate gets its own learning rate. MEASURED: at lr 3e-5 a k-means-calibrated router holds
    # entropy 0.774, but at the TUNED 3e-4 it falls to 0.469 (K=32) and 0.346 (K=128) -- the gate
    # moves fast enough to undo the calibration, and the mixture loses to its own K=1 ablation.
    # --gate-lr 0 freezes the gate entirely, i.e. fixed feature-space routing that cannot collapse.
    _ore = ore_modules(model) if args.arm in ("ORE", "U") else []
    gate_ps = [p for m in _ore for p in (m.gate.weight, m.gate.bias) if p.requires_grad]
    if args.gate_lr is not None and gate_ps:
        gid = {id(p) for p in gate_ps}
        rest = [p for p in params if id(p) not in gid]
        if args.gate_lr == 0:
            for p in gate_ps:
                p.requires_grad_(False)
            groups = [{"params": rest, "lr": args.lr}]
            print("[cell] gate FROZEN at its calibrated init", flush=True)
        else:
            groups = [{"params": rest, "lr": args.lr},
                      {"params": gate_ps, "lr": args.gate_lr}]
            print(f"[cell] gate lr {args.gate_lr} vs body {args.lr}", flush=True)
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    ent, tel, curve, lo_win, matrices = [], [], [], [], []
    t0 = time.time()
    ore = ore_modules(model) if args.arm in ("ORE", "U") else []   # FFNMoE is an OREResidual
    task_win, load_trace = [], []
    for step in range(args.steps):
        # Gradient accumulation. The point is not memory, it is TOKENS PER EXPERT: at batch 4 a
        # step routes 2916 tokens, so at K=128 each expert sees 22.8 and ~125 of them see zero.
        # Routing counts accumulate across the micro-steps (OREResidual.tok_count), so balance is
        # judged on the effective batch. See docs/TODO_ore_routing.md.
        opt.zero_grad()
        lv = tv = 0.0
        nmicro = 0
        for _ in range(args.accum):
            idx = rng.integers(0, len(train), args.batch)
            paths = [train[i][0] for i in idx]
            deps = [train[i][1] for i in idx]
            key = batch_key(keyarm or args.arm, args.keys, paths, deps, (size, size), dev, cal)
            pz = predict_depth(model, paths, size, dev, key=key, batch=args.batch)
            gts = torch.stack([torch.from_numpy(load_depth(d, *pz.shape[1:])).float()
                               for d in deps], 0).to(dev)
            pw = pixel_weights(gts, edges, shares, dev) if shares is not None else None
            if bin_mask is not None:
                m = ((gts >= bin_mask[0]) & (gts < bin_mask[1])).float()
                pw = m if pw is None else pw * m
            task = si_loss(pz, gts, pw, anchor_shift=bin_mask is not None, metric=args.metric)
            if task is None:
                continue
            loss = task
            a = aux_loss(model)
            if a is not None:
                loss = loss + args.aux_coef * a
            z = ore_z_loss(model)
            if z is not None:
                loss = loss + args.z_coef * z
            (loss / args.accum).backward()
            lv += float(loss.item()); tv += float(task.item()); nmicro += 1
        if not nmicro:
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        for m in ore:                       # bias update is per OPTIMISER step, and it zeroes
            m.update_bias(args.bias_gamma)  # tok_count, so it must run after the accum loop
        # RE-CALIBRATE the router. MEASURED: freezing the gate does NOT freeze the routing.
        # The gate is calibrated on the activation distribution at step 0, but the experts change
        # that distribution as they train, so the partition goes stale under it -- a FROZEN gate
        # still collapsed from 8.5 effective experts of 32 at step 0 to 2.2 by step 1000. Re-fitting
        # the whitening statistics and the k-means centroids on CURRENT activations tracks the
        # drift instead of fighting it.
        if args.recalib_every and step > 0 and step % args.recalib_every == 0:
            calibrate_routers(model, _ore, train, size, dev, args.router_cal_frames,
                              kmeans=args.router_init == "kmeans", seed=args.seed, warm=True,
                              ema=args.recalib_ema)
            model.train()
        ent.append(utilisation_entropy(model))
        lo_win.append(lv / nmicro)
        # Task loss WITHOUT the aux/z terms. `loss_trace` mixes them in and aux is None for the
        # dense arms, which made every cross-arm train-loss comparison in L-ORE-11 invalid.
        task_win.append(tv / nmicro)
        if ore and (step % 50 == 0 or step == args.steps - 1):
            st = ore[0].load_stats()
            if st:
                # Per-site too: the collapse is monotone in DEPTH, not uniform. Site 0 routes on a
                # static input (frozen encoder + frozen attention) and stays healthy at ~11
                # effective experts, while the mean over all 16 sites implies ~2.4 at the deeper
                # ones, whose inputs carry the accumulated residuals. A single site's number, or
                # the mean, hides that.
                per_site = [m.load_stats()["effective_experts"] for m in ore]
                load_trace.append({"step": step, "per_site_eff": [round(v, 2) for v in per_site],
                                   **{k: v for k, v in st.items() if k != "hist"}})
                # Fail LOUD. A collapsed router still returns a plausible depth number, which is
                # how 8 runs of K_effective~3 were reported as "K=128".
                # SMOOTHED, not a single batch. Site-0 utilisation swings 5.9-22.1 effective
                # experts between 4-frame batches (91 tokens per expert at balance, so which four
                # frames you drew matters), and a threshold tested against one draw is eventually
                # crossed by chance. That is not hypothetical: this gate aborted five healthy
                # 25,000-step cells at step 2800 on single-batch readings of 3.2-3.3.
                floor = 0.35 * (res.get("route_ceiling") or args.k / 4)
                recent = [r["effective_experts"] for r in load_trace[-10:]]
                smoothed = sum(recent) / len(recent)
                if step >= 1000 and len(recent) >= 10 and smoothed < floor:
                    res["ABORT"] = (f"router collapsed: {smoothed:.1f} effective experts "
                                    f"(mean of last 10 checks) at step {step}, under 35% of the "
                                    f"{floor / 0.35:.1f} a k-means partition achieves here")
                    res["load_trace"] = load_trace
                    json.dump(res, open(args.out, "w"), indent=2)
                    print("[cell] ABORT " + res["ABORT"], flush=True)
                    return
        if step % 500 == 0 and args.arm not in ("D", "M", "F"):
            tel.append({"step": step, **delta_telemetry(model)})
        if args.eval_every and step > 0 and step % args.eval_every == 0:
            ev = evaluate(model, val, keyarm or args.arm, args.keys, size, dev, args.eval_n, cal,
                          metric=args.metric)
            bins = [ev["bins"][f"{a}-{b}"]["rel"] for a, b in BINS
                    if ev["bins"][f"{a}-{b}"]["n"]]
            h = mixture_health(model)
            curve.append({"step": step, "loss": float(np.mean(lo_win[-args.eval_every:])),
                          "primary": float(np.mean(bins)),
                          "whole": ev["whole_absrel"],
                          "near_rel": ev["bins"]["0.2-0.5"]["rel"],
                          "health": h,
                          "load": (ore[0].load_stats() if ore else None),
                          "unseen": (evaluate(model, val2, keyarm or args.arm, args.keys, size,
                                              dev, args.eval_n, cal, metric=args.metric)
                                     ["whole_absrel"] if val2 else None)})
            c = curve[-1]
            print(f"[cell] eval@{step} whole {c['whole']:.4f} near {c['near_rel']:.4f}"
                  + (f" unseen {c['unseen']:.4f}" if c.get("unseen") else "")
                  + f" loss {c['loss']:.4f}", flush=True)
            if h:
                print("[cell] " + health_banner(h), flush=True)
            # Checkpoint EVERY eval too, overwriting. A run that produces only curves cannot feed
            # C2 (metric scale) or C4 (closed-loop nav), which need weights -- and a 100,000-step
            # run that is killed at 90% would otherwise leave nothing to evaluate.
            if args.save_ckpt:
                ck = {"state_dict": model.state_dict(), "arm": args.arm, "k": args.k,
                      "rank": args.rank, "sites": sites, "seed": args.seed,
                      "step": step, "route_dim": args.route_dim,
                      "aat_depth": args.aat_depth, "ffn_widen": args.ffn_widen,
                      "top_k": args.top_k, "council_m": args.council_m}
                torch.save(ck, args.save_ckpt)
                if step in snap_steps:
                    torch.save(ck, args.save_ckpt[:-3] + f"_step{step}.pt"
                               if args.save_ckpt.endswith(".pt")
                               else args.save_ckpt + f"_step{step}")
            # Partial dump EVERY eval. A 25,000-step run is ~11 h; writing only at the end means a
            # kill, a preemption or simply not being finished yet yields nothing at all.
            json.dump({**res, "curve": curve, "partial_at_step": step,
                       "loss_trace": [float(np.mean(lo_win[i:i + 100]))
                                      for i in range(0, len(lo_win), 100)],
                       "task_trace": [float(np.mean(task_win[i:i + 100]))
                                      for i in range(0, len(task_win), 100)],
                       "load_trace": load_trace}, open(args.out + ".partial", "w"), indent=2)
        if (args.diag_matrix_every and step > 0 and step % args.diag_matrix_every == 0
                and (moe_modules(model) or ore_modules(model))):
            mx = forced_expert_matrix(model, val, size, dev, args.diag_matrix_n)
            matrices.append({"step": step, **mx})
            model.train()
        if step % 250 == 0:
            print(f"[cell] step {step} loss {loss.item():.4f} ent {ent[-1]:.3f} "
                  f"({(time.time()-t0)/max(step,1):.2f}s/step)", flush=True)

    if args.arm not in ("D", "M", "F"):
        tel.append({"step": args.steps, **delta_telemetry(model)})
    res["delta_telemetry"] = tel
    res["curve"] = curve
    res["loss_trace"] = [float(np.mean(lo_win[i:i + 100])) for i in range(0, len(lo_win), 100)]
    res["task_trace"] = [float(np.mean(task_win[i:i + 100])) for i in range(0, len(task_win), 100)]
    res["load_trace"] = load_trace
    res["load_stats"] = ore[0].load_stats() if ore else None
    res["sec_per_step"] = (time.time() - t0) / args.steps
    res["utilisation_entropy"] = float(np.mean(ent[-200:]))
    if val2:
        res["unseen_eval"] = evaluate(model, val2, keyarm or args.arm, args.keys, size, dev,
                                      args.eval_n, cal, metric=args.metric)
    res["final_eval"] = evaluate(model, val, keyarm or args.arm, args.keys, size, dev, args.eval_n, cal,
                                 metric=args.metric)
    # ALWAYS end on a matrix, never only on the modulo: the loop is range(steps), so a
    # 3000-step run's last step is 2999 and `step % 3000 == 0` fires never. A guard that
    # silently no-ops is worse than no guard, which is the exact failure it exists to catch.
    if args.diag_matrix_every and (moe_modules(model) or ore_modules(model)):
        matrices.append({"step": args.steps,
                         **forced_expert_matrix(model, val, size, dev, args.diag_matrix_n)})
    res["expert_matrices"] = matrices
    res["health"] = mixture_health(model)
    if res["health"]:
        print("[cell] " + health_banner(res["health"]), flush=True)
        res["INERT"] = res["health"]["inert"]
    if args.save_ckpt:
        # route_dim and step must match the per-eval save above: omitting route_dim here made every
        # completed run's untagged checkpoint unloadable by the C4 harness, which rebuilt the gate
        # at the site's own width instead of the frozen-feature width.
        torch.save({"state_dict": model.state_dict(), "arm": args.arm, "k": args.k,
                    "rank": args.rank, "sites": sites, "seed": args.seed,
                    "step": args.steps, "route_dim": args.route_dim,
                    "aat_depth": args.aat_depth, "ffn_widen": args.ffn_widen,
                    "top_k": args.top_k, "council_m": args.council_m},
                   args.save_ckpt)
        print(f"[cell] saved weights -> {args.save_ckpt}", flush=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[cell] DONE {args.out} whole={res['final_eval']['whole_absrel']:.4f} "
          f"ent={res['utilisation_entropy']:.3f} {res['sec_per_step']:.2f}s/step", flush=True)


if __name__ == "__main__":
    main()

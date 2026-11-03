#!/usr/bin/env python3
"""Router labels and fine-tuning for GateRoute (model/gateroute/gateroute.py).

TRAINING FRAMES. TUM fr1_desk, fr1_xyz, fr1_360, fr3_long_office_household, every 2nd associated
rgb/depth pair (the frames train_conf.py used for the Co-Me predictor), disjoint from every eval
set (NYUv2 test, ETH3D, TUM fr1_room / fr2_desk). Same deployed preprocessing (data.preprocess).

LABELS (--stage labels), one value per token of the 27x27 grid at 378, from GT depth:
  planner  fraction of the token's samples that are DECISIVE for the 0.45 m collision test, exactly
           eval/gateroute/planner_metric.patch_labels (the frozen planner-metric definition).
  deperr   the dense student's own per-token AbsRel against GT after per-image median alignment
           (the depth-error control: same recipe, label swapped).

FINE-TUNING (--stage train). Start: student_v2.pt (the dense_378 baseline checkpoint, see
student_io.py). Teacher: the frozen dense student on the same frame (MonoCore outputs).
  loss = model/distill/train.distill_loss (log-L1 + 0.5 SI on depth along ray, 2x log-L1 on the
         metric scale, 0.5x ray cosine: the loss the student was distilled with)
       + w_router * sum over routed blocks of Co-Me's pairwise-rank loss (train_conf.rank_loss)
         between that block's router scores and the label (learned routers only).
Router inputs are detached (gateroute.py), so the label trains only the routers and the distillation
loss trains only the student. Every arm (planner / deperr / norm / random) trains the same student
parameters with the same data order, steps and schedule; the parameter counts are logged.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "distill"))
from eval.gateroute.data import DATA_ROOT, preprocess, _tum_assoc  # noqa: E402
from eval.gateroute import planner_metric as pm  # noqa: E402
from model.gateroute.student_io import load_student, dense_core  # noqa: E402
from model.gateroute.gateroute import GateRouteCore, save_gateroute, ENC_SITES, AAT_SITES  # noqa: E402
from model.gateroute.train_conf import TRAIN_SEQS, rank_loss  # noqa: E402

RES = 378
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def train_pairs(stride=2):
    out = []
    for s in TRAIN_SEQS:
        seq = os.path.join(DATA_ROOT, s)
        out += [(os.path.join(seq, r), os.path.join(seq, d)) for r, d in _tum_assoc(seq)[::stride]]
    return out


@torch.no_grad()
def labels(out):
    from model.gateroute.come import depth_z
    pairs = train_pairs()
    core = dense_core(load_student(), RES)
    img8, lab_p, lab_e, okf = [], [], [], []
    for i, (rp, dp) in enumerate(pairs):
        rgb = np.asarray(Image.open(rp).convert("RGB"))
        x, (y0, x0, side) = preprocess(rgb, RES)
        img8.append(torch.from_numpy(np.clip((x.transpose(1, 2, 0) * STD + MEAN) * 255 + 0.5, 0, 255)
                                     .astype(np.uint8)))
        d = np.asarray(Image.open(dp)).astype(np.float64) / 5000.0
        gt = d[y0:y0 + side, x0:x0 + side]
        valid = (gt > 0.1) & (gt <= 10.0)
        K = pm.intrinsics("tum", rp, d.shape)
        lp, ok = pm.patch_labels(gt, valid, K, (y0, x0, side))
        pr = depth_z(*core(torch.from_numpy(x)[None].cuda()))
        pr = F.interpolate(pr[None, None], size=(side, side), mode="bilinear", align_corners=False)[0, 0]
        pr = pr.cpu().numpy().astype(np.float64)
        v = valid & (pr > 1e-3)
        pr = pr * np.median(gt[v] / pr[v])
        are = np.where(v, np.abs(pr - gt) / np.maximum(gt, 1e-6), 0.0)
        e = F.adaptive_avg_pool2d(torch.from_numpy(are)[None, None], pm.PATCH_GRID).flatten()
        n = F.adaptive_avg_pool2d(torch.from_numpy(v.astype(np.float64))[None, None], pm.PATCH_GRID).flatten()
        lab_p.append(lp)
        lab_e.append((e / n.clamp_min(1e-6)).float())
        okf.append(ok)
        if i % 250 == 0:
            print(i, len(pairs), f"planner label>0 frac {float((lp > 0).float().mean()):.3f}", flush=True)
    lp, le = torch.stack(lab_p), torch.stack(lab_e)
    torch.save({"pairs": pairs, "img": torch.stack(img8), "planner": lp, "deperr": le, "valid_frac": torch.stack(okf)},
               out)
    pos = (lp > 0).float().mean(1)
    print(f"{len(pairs)} frames; planner label>0 per frame: mean {pos.mean():.3f}, median {pos.median():.3f}, "
          f"frames with none {(pos == 0).float().mean():.3f}; Spearman-free check: mean planner label on the "
          f"top-25% deperr tokens {float(lp.gather(1, le.topk(182, 1).indices).mean()):.3f} vs all "
          f"{float(lp.mean()):.3f}", flush=True)


@torch.no_grad()
def labels_shards(shard_dir, out):
    """Planner labels for prep_broad.py shards (no deperr label: those frames serve the planner arm).
    Images stay as JPEG files on disk; the cache stores their paths."""
    imgs, lab, okf, names = [], [], [], []
    for f in sorted(x for x in os.listdir(shard_dir) if x.endswith(".npz")):
        z = np.load(os.path.join(shard_dir, f))
        for jpg, g, name, (H, W) in zip(z["jpg"], z["gdep"], z["names"], z["shape"]):
            s = min(H, W)
            crop = ((H - s) // 2, (W - s) // 2, s)
            rows, cols = pm.grid_pixels(crop)
            gt = np.zeros((s, s))
            gt[rows, cols] = g.astype(np.float64).ravel()      # only grid samples are ever read
            K = pm.intrinsics("nyu" if name.startswith("nyu") else "tum", str(name), (H, W))
            lp, ok = pm.patch_labels(gt, gt > 0, K, crop)
            imgs.append(os.path.join(os.path.abspath(shard_dir), str(jpg)))
            lab.append(lp)
            okf.append(ok)
            names.append(str(name))
        print(f, len(names), flush=True)
    lp = torch.stack(lab)
    torch.save({"pairs": [(n, None) for n in names], "img": imgs, "planner": lp,
                "valid_frac": torch.stack(okf)}, out)
    print(f"{len(names)} frames; planner label>0 per frame mean {float((lp > 0).float().mean()):.3f}", flush=True)


def _sim_one(pair):
    g = np.load(pair[1]).astype(np.float64)
    lp, ok = pm.patch_labels(g, (g > 0) & (g <= 10.0), pm.intrinsics("sim", pair[0], g.shape), (0, 0, g.shape[0]),
                             dev="cpu")
    return lp.numpy(), ok.numpy()               # numpy: torch tensors cross Pool via fds and exhaust them


def labels_sim(dirs, out):
    """Planner labels for drone-sim renders (eval/drone/gen_nearfield.py: rgb_%06d.png + dep_%06d.npy at
    378 x 378). The sets share frames through symlinks, so frames are deduplicated by real path; each
    frame is named sim_<set>_<split>/<file> so --val_seq can hold out a split."""
    import glob
    from multiprocessing import Pool
    torch.set_num_threads(1)
    seen, pairs, names = set(), [], []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "rgb_*.png"))):
            r = os.path.realpath(f)
            if r not in seen:
                seen.add(r)
                pairs.append((r, os.path.realpath(f[:-len(".png")].replace("rgb_", "dep_") + ".npy")))
                parts = r.split("/")
                names.append(f"sim_{parts[-3]}_{parts[-2]}/{parts[-1]}")
    print(f"{len(pairs)} unique frames from {len(dirs)} dirs", flush=True)
    with Pool(max(1, len(os.sched_getaffinity(0)) - 2)) as p:
        res = p.map(_sim_one, pairs, chunksize=64)
    lp = torch.from_numpy(np.stack([r[0] for r in res]))
    torch.save({"pairs": [(n, None) for n in names], "img": [r for r, _ in pairs], "planner": lp,
                "valid_frac": torch.from_numpy(np.stack([r[1] for r in res]))}, out)
    print(f"{len(names)} frames; planner label>0 per frame mean {float((lp > 0).float().mean()):.3f}", flush=True)


def seq_of(path):
    """Sequence id of a label-cache frame: the TUM sequence directory, nyu_train, or sim_<set>_<split>."""
    return next((p for p in str(path).split("/") if p.startswith(("rgbd_dataset_", "sim_")) or p == "nyu_train"),
                "?")


def gt_grids(cs):
    """GT depth (m, 0 = invalid) at the 108 x 108 planner-metric grid for every label-cache frame, in cache
    order: broad shards store it, TUM pairs and sim renders are sampled from their depth files."""
    shard = {}
    out = []
    for c in cs:
        for (name, dp), im in zip(c["pairs"], c["img"]):
            if isinstance(im, str) and "broad_shards" in im:
                d = os.path.dirname(os.path.dirname(im))
                if d not in shard:
                    for f in (x for x in os.listdir(d) if x.endswith(".npz")):
                        z = np.load(os.path.join(d, f))
                        shard.update({os.path.join(d, str(j)): g for j, g in zip(z["jpg"], z["gdep"])})
                out.append(shard[im].astype(np.float16))
                continue
            if isinstance(im, str):                      # sim render: 378 x 378 metric depth .npy
                g = np.load(im[:-len(".png")].replace("rgb_", "dep_") + ".npy").astype(np.float64)
                g = np.where((g > 0) & (g <= 10.0), g, 0)
            else:                                        # TUM pair
                g = np.asarray(Image.open(dp)).astype(np.float64) / 5000.0
                g = np.where((g > 0.1) & (g <= 10.0), g, 0)
            H, W = g.shape
            s = min(H, W)
            rows, cols = pm.grid_pixels(((H - s) // 2, (W - s) // 2, s))
            out.append(g[(H - s) // 2:][:s, (W - s) // 2:][rows, cols].reshape(pm.GRID, pm.GRID).astype(np.float16))
    return torch.from_numpy(np.stack(out))


@torch.no_grad()
def teacher_grids(cs, out, batch=8):
    """MapAnything teacher metric z-depth at the 108 x 108 planner-metric grid for every label-cache frame,
    in cache order: the 378 input crop resized to 518, bf16 teacher (model/backbones/teacher.py)."""
    from model.backbones.teacher import load_teacher
    teacher = load_teacher().eval()
    imgs = [im for c in cs for im in c["img"]]
    res = []
    for j in range(0, len(imgs), batch):
        x8 = torch.stack([torch.from_numpy(np.asarray(Image.open(im).convert("RGB"))) if isinstance(im, str) else im
                          for im in imgs[j:j + batch]]).cuda()
        x = F.interpolate(to_input(x8), size=(518, 518), mode="bicubic", antialias=True, align_corners=False)
        B = len(x)
        view = [dict(img=x, true_shape=np.array([[518, 518]] * B), idx=0, instance=[str(i) for i in range(B)],
                     data_norm_type=["dinov2"] * B)]
        z = teacher(view, memory_efficient_inference=True, minibatch_size=B)[0]["pts3d_cam"][..., 2].float()
        res.append(F.interpolate(z[:, None], size=(pm.GRID, pm.GRID), mode="bilinear", align_corners=False)[:, 0]
                   .clamp_min(0).half().cpu())
        if j % (batch * 250) == 0:
            print(j, len(imgs), f"median z {float(z.median()):.3f}", flush=True)
    torch.save(torch.cat(res), out)
    print("saved", out, len(imgs), flush=True)


def gt_loss(s, g, y, G, w_planner, w_metric=0.0):
    """Median-free scale-invariant log-L1 against the GT grid g (B,108,108), samples weighted
    1 + w_planner on tokens with planner label > 0 (y (B, G*G)); w_metric adds the per-image log scale error."""
    z = (s["ray"][..., 2] * s["depth"] * s["scale"].reshape(-1, 1, 1))[:, None]
    p = F.interpolate(z, size=g.shape[-1], mode="bilinear", align_corners=False)[:, 0]
    v = (g > 0).float()
    w = v * (1 + w_planner * F.interpolate((y > 0).float().reshape(-1, 1, G, G), size=g.shape[-1])[:, 0])
    d = torch.log(p.clamp_min(1e-3)) - torch.log(g.clamp_min(1e-3))
    ws = w.sum((1, 2)).clamp_min(1e-6)
    shift = (w * d).sum((1, 2)) / ws
    per = (w * (d - shift[:, None, None]).abs()).sum((1, 2)) / ws + w_metric * shift.abs()
    ok = (v.sum((1, 2)) > 100).float()
    return (per * ok).sum() / ok.sum().clamp_min(1)


def clear_loss(s, g, void=False):
    """L1 (m) between predicted and GT clearance fields on sim frames (378 crop, 90 deg hfov). The clearance
    along an in-view direction d is the planner's free(d) (planner_metric.entry) made continuous in depth:
    soft min (2 cm temperature) over grid points of t - sqrt(SAFETY^2 - rho^2)_+ + (rho - SAFETY)_+, capped
    at MAX_RANGE and negative when the camera is inside a safety ball. Points that miss the safety ball still
    pull on it, so a missed obstacle has a gradient. Prediction median-aligned
    to GT (scale factor detached), as the primary planner metric. void keeps predicted points on GT-void pixels
    (no surface, e.g. the unrendered ceiling) in the predicted field, so near depth there costs like a phantom."""
    K, crop = pm.intrinsics("sim", "", (RES, RES)), (0, 0, RES)
    D = torch.as_tensor(pm.frustum_dirs(K, crop), dtype=torch.float32, device=g.device)
    rows, cols = (torch.as_tensor(v, dtype=torch.float32, device=g.device) for v in pm.grid_pixels(crop))
    z = (s["ray"][..., 2] * s["depth"] * s["scale"].reshape(-1, 1, 1))[:, None].float()
    p = F.interpolate(z, size=g.shape[-1], mode="bilinear", align_corners=False)[:, 0]
    out = []
    for pi, gi in zip(p.flatten(1), g.flatten(1)):
        ok = (gi > pm.MIN_Z) & (gi < pm.MAX_RANGE)
        if ok.sum() < 100:
            continue
        pi = pi * (gi[ok] / pi[ok].detach().clamp_min(1e-3)).median().detach()

        def field(zz, keep):
            P = pm.points(zz[keep], rows[keep], cols[keep], K, crop)
            t = P @ D.T
            rho = ((P * P).sum(1, keepdim=True) - t * t).clamp_min(1e-8).sqrt()
            c = t - (pm.SAFETY ** 2 - rho ** 2).clamp_min(0).sqrt() + (rho - pm.SAFETY).clamp_min(0)
            return (-0.02 * torch.logsumexp(-c / 0.02, 0)).clamp(max=pm.MAX_RANGE)   # soft min over points
        cg = field(gi, ok)
        cp = field(pi, (pi > pm.MIN_Z) & (pi < pm.MAX_RANGE) & ((gi > 0) | void))
        m = torch.minimum(cp, cg) < pm.MAX_RANGE
        if m.any():
            out.append((cp[m] - cg[m]).abs().mean())
    return torch.stack(out).mean() if out else p.sum() * 0


def to_input(img8):
    x = (img8.float() / 255.0 - torch.from_numpy(MEAN).to(img8.device)) / torch.from_numpy(STD).to(img8.device)
    return x.permute(0, 3, 1, 2).contiguous()


def save_preview(x, g, path, n=5):
    """Up to n training samples as the student sees them (after augmentation) beside their GT grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = min(n, len(x))
    img = (x[:n].permute(0, 2, 3, 1).float().cpu().numpy() * STD + MEAN).clip(0, 1)
    fig, axs = plt.subplots(n, 2 if g is not None else 1, figsize=(5, 2.5 * n), squeeze=False)
    for i in range(n):
        axs[i, 0].imshow(img[i])
        if g is not None:
            axs[i, 1].imshow(np.ma.masked_equal(g[i].cpu().numpy(), 0), cmap="turbo_r", vmin=0, vmax=4)   # void blank
    for ax in axs.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close(fig)


def unpack(dense, scale):
    return {"depth": dense[:, 3].float(), "ray": dense[:, :3].permute(0, 2, 3, 1).float(), "scale": scale.float()}


def train(a):
    from train import distill_loss
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)
    cs = [torch.load(p, weights_only=False) for p in a.labels]
    imgs = [im for c in cs for im in c["img"]]          # uint8 (378,378,3) tensors or JPEG paths
    lab = torch.cat([c[a.label] for c in cs]) if a.label != "none" else None
    seqs = np.array([seq_of(p[0]) for c in cs for p in c["pairs"]])
    n = len(imgs)
    gtg = (torch.load(a.gt_source) if a.gt_source else gt_grids(cs)) if a.w_gt or a.w_clear else None
    if a.val_seq:                                        # hold out whole sequences
        is_val = np.array([any(v in s for v in a.val_seq) for s in seqs])
        val_i, tr_i = np.where(is_val)[0], np.where(~is_val)[0]
        val_i = val_i[np.random.RandomState(0).permutation(len(val_i))[:150]]
    else:
        perm = np.random.RandomState(0).permutation(n)   # split fixed across seeds and arms
        val_i, tr_i = perm[:n // 20], perm[n // 20:]
    print(f"frames: train {len(tr_i)} val {len(val_i)}; train seqs {sorted(set(seqs[tr_i]))}; "
          f"val seqs {sorted(set(seqs[val_i]))}", flush=True)
    res, G = a.res, a.res // 14                         # images are cached as 378 crops; resized below
    if lab is not None and G != 27:                      # regrid the 27 x 27 token labels to G x G
        lab = F.adaptive_avg_pool2d(lab.float().reshape(-1, 1, 27, 27), G).reshape(-1, G * G)
    tres = a.teacher_res or res
    teacher = dense_core(load_student(), tres)
    for p in teacher.parameters():
        p.requires_grad_(False)
    m = load_student()
    core = GateRouteCore(m, (res, res), a.k, a.enc_sites, a.aat_sites, a.router, a.seed, a.kv_full,
                         a.heavy, a.k_heavy, a.heavy_router, a.router_arch, a.coverage).cuda()
    core.eval()                                          # no dropout / drop-path; grads still flow
    if a.train_scope == "routed":                        # only the routed blocks' own weights move
        keep = tuple(f"encoder.model.blocks.{i}." for i in a.enc_sites) + \
            tuple(f"info_sharing.self_attention_blocks.{i}." for i in a.aat_sites)
        for name, p in m.named_parameters():
            p.requires_grad_(name.startswith(keep))
    stu_p = [p for p in m.parameters() if p.requires_grad]
    rt_p = list(core.routers.parameters())
    hv_p = list(core.heavy.parameters())
    n_stu, n_rt = sum(p.numel() for p in stu_p), sum(p.numel() for p in rt_p)
    print(f"trainable: student {n_stu / 1e6:.2f}M, routers {n_rt} params; k={a.k} router={a.router} "
          f"label={a.label} enc={a.enc_sites} aat={a.aat_sites}; heavy {a.heavy} x k={a.k_heavy} "
          f"({a.heavy_router}, {sum(p.numel() for p in hv_p) / 1e6:.2f}M)", flush=True)
    groups = [{"params": stu_p, "lr": a.lr}] + ([{"params": rt_p, "lr": a.lr_router}] if rt_p else []) + \
        ([{"params": hv_p, "lr": a.lr_heavy}] if hv_p else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    k_eval = min(a.k, G * G)

    def img8(i):
        im = imgs[i]
        return torch.from_numpy(np.asarray(Image.open(im).convert("RGB"))) if isinstance(im, str) else im

    def batch(ix, flip, aug=False):
        x8 = torch.stack([img8(i) for i in ix]).cuda()
        g = gtg[ix].cuda().float() if gtg is not None else None
        if aug and a.void_aug:                           # unrendered-ceiling wedge: black RGB, GT void (C4 scenes have no ceiling)
            r, c = torch.arange(RES, device="cuda")[:, None], torch.arange(RES, device="cuda")[None]
            gr, gc = (torch.as_tensor(v, device="cuda") for v in pm.grid_pixels((0, 0, RES)))
            for b in np.where(rng.rand(len(ix)) < a.void_aug)[0]:
                h0, k, c0, w = (rng.uniform(0.15, 0.45) * RES, rng.uniform(0.5, 1.5), rng.uniform(0.3, 0.7) * RES,
                                rng.uniform(0.05, 0.3) * RES)   # flat-bottomed trapezoid, as the flight void
                vm = r < h0 - k * ((c - c0).abs() - w).clamp_min(0)
                x8[b][vm] = 0
                if g is not None:
                    g[b][vm[gr, gc].reshape(pm.GRID, pm.GRID)] = 0
        x0 = to_input(x8)
        x = x0
        if res != RES:
            x = F.interpolate(x0, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
        y = lab[ix].cuda() if lab is not None else None
        if flip:
            x, x0 = x.flip(-1), x0.flip(-1)
            if y is not None:
                y = y.reshape(-1, G, G).flip(-1).reshape(-1, G * G)
            if g is not None:
                g = g.flip(-1)
        if tres == res:
            xt = x
        else:                                            # teacher input from the 378 crop, not the student input
            xt = x0 if tres == RES else F.interpolate(x0, size=(tres, tres), mode="bilinear", antialias=True,
                                                      align_corners=False)
        return x, y, g, xt

    def teach(xt):
        t = unpack(*teacher(xt))
        if tres != res:                                  # teacher outputs resized to the student grid
            t["depth"] = F.interpolate(t["depth"][:, None], size=(res, res), mode="bilinear", align_corners=False)[:, 0]
            t["ray"] = F.interpolate(t["ray"].permute(0, 3, 1, 2), size=(res, res), mode="bilinear",
                                     align_corners=False).permute(0, 2, 3, 1)
        return t

    def router_terms(y):
        """Rank loss (learned routers only) and, for every router kind, the lift of the top-k picks:
        P(label > 0 | picked) / P(label > 0)."""
        if y is None:
            return torch.zeros((), device="cuda"), {}
        losses, prec = [], {}
        for name, s in core.last_scores.items():
            if (a.router if name != "heavy" else a.heavy_router) == "learned":
                losses.append(rank_loss(s.float(), y))
            top = s.topk(k_eval if name != "heavy" else min(a.k_heavy, G * G), 1).indices
            prec[name] = float((y.gather(1, top) > 0).float().mean() / (y > 0).float().mean().clamp_min(1e-6))
        return (torch.stack(losses).mean() if losses else torch.zeros((), device="cuda")), prec

    @torch.no_grad()
    def evaluate():
        tot, rl, lift = [], [], []
        for j in range(0, len(val_i), a.batch):
            ix = val_i[j:j + a.batch]
            x, y, _, xt = batch(ix, False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t = teach(xt)
                s = unpack(*core(x))
            loss, _ = distill_loss(s, t)
            r, prec = router_terms(y)
            tot.append(float(loss))
            rl.append(float(r))
            if prec:
                lift.append(np.mean(list(prec.values())))
        return {"distill": float(np.mean(tot)), "router_rank": float(np.mean(rl)),
                "router_lift": float(np.mean(lift)) if lift else None}

    log = {"args": vars(a), "trainable_student": n_stu, "trainable_router": n_rt, "n_train": len(tr_i),
           "n_val": len(val_i), "val": {0: evaluate()}}
    print("val@0", log["val"][0], flush=True)
    t0 = time.time()
    pw = None
    if a.nyu_frac is not None:                           # NYU frames get nyu_frac of the sampling mass
        is_nyu = np.array(["nyu" in str(cs_p[0]) for c in cs for cs_p in c["pairs"]])[tr_i]
        pw = np.where(is_nyu, a.nyu_frac / is_nyu.sum(), (1 - a.nyu_frac) / (~is_nyu).sum())
        print(f"sampling: {is_nyu.sum()} NYU / {(~is_nyu).sum()} other train frames, nyu_frac {a.nyu_frac}", flush=True)
    for step in range(1, a.steps + 1):
        ix = rng.choice(tr_i, a.batch, replace=False, p=pw)
        x, y, g, xt = batch(ix, rng.rand() < 0.5, True)
        if step == 1:                                    # look at what the student trains on
            save_preview(x, g, a.out.replace(".pt", "_samples.png"))
            if a.preview_only:
                return
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t = teach(xt)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = unpack(*core(x))
        dl, parts = distill_loss(s, t)
        rl, _ = router_terms(y)
        loss = a.w_distill * dl + a.w_router * rl
        if a.w_gt:
            yg = y[:, torch.randperm(G * G, device=y.device)] if a.gt_random else y   # same mass, random places
            lg = gt_loss(s, g, yg, G, a.gt_planner, a.gt_metric)
            loss = loss + a.w_gt * lg
        if a.w_clear:
            lc = clear_loss(s, g, bool(a.clear_void))
            loss = loss + a.w_clear * lc
        if a.w_dense:                                    # same weights run with routing off, same teacher
            core.k = G * G
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sd = unpack(*core(x))
            core.k = a.k
            dd, _ = distill_loss(sd, t)
            loss = loss + a.w_dense * dd
            if a.w_gt:
                loss = loss + a.w_dense * a.w_gt * gt_loss(sd, g, yg, G, a.gt_planner, a.gt_metric)
        log.setdefault("train", []).append((step, float(dl), float(rl)) + ((float(dd),) if a.w_dense else ())
                                           + ((float(lg),) if a.w_gt else ()) + ((float(lc),) if a.w_clear else ()))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(stu_p + hv_p, 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step} distill {float(dl):.4f} router {float(rl):.4f} "
                  f"{f'gt {float(lg):.4f} ' if a.w_gt else ''}{f'clear {float(lc):.4f} ' if a.w_clear else ''}{parts} "
                  f"({(time.time() - t0) / step:.2f}s/it)", flush=True)
        stop = a.max_hours and time.time() - t0 > a.max_hours * 3600
        if step % a.eval_every == 0 or step == a.steps or stop:
            log["val"][step] = evaluate()
            print(f"val@{step}", log["val"][step], flush=True)
        if stop:                                         # walltime guard: save before the job is killed
            print(f"max_hours reached at step {step}", flush=True)
            break
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_gateroute(a.out, core, {"label": a.label, "seed": a.seed, "train_log": log})
    json.dump(log, open(a.out.replace(".pt", ".json"), "w"), indent=1)
    print("saved", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["labels", "labels_shards", "labels_sim", "teacher_grids", "train"],
                    required=True)
    ap.add_argument("--labels", nargs="+", default=[os.path.expanduser("~/gateroute/gr_labels.pt")])
    ap.add_argument("--shards", default="", help="prep_broad.py shard dir (labels_shards)")
    ap.add_argument("--sim_dirs", nargs="*", default=[], help="gen_nearfield.py output dirs (labels_sim)")
    ap.add_argument("--val_seq", nargs="*", default=[], help="hold out these sequences for validation")
    ap.add_argument("--label", default="planner", choices=["planner", "deperr", "none"])
    ap.add_argument("--router", default="learned", choices=["learned", "norm", "random"])
    ap.add_argument("--router_arch", default="linear", choices=["linear", "ctx"], help="learned router: per-token linear or 3x3-context conv")
    ap.add_argument("--coverage", type=float, default=0.0, help="selection penalty per block that already updated a token")
    ap.add_argument("--k", type=int, default=182)
    ap.add_argument("--res", type=int, default=RES, help="input side; the 378 crops are resized to it")
    ap.add_argument("--teacher_res", type=int, default=0, help="teacher input side (0: same as --res)")
    ap.add_argument("--kv_full", action="store_true", help="unpicked tokens still give keys / values")
    ap.add_argument("--enc_sites", type=int, nargs="*", default=list(ENC_SITES))
    ap.add_argument("--aat_sites", type=int, nargs="*", default=list(AAT_SITES))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr_router", type=float, default=1e-3)
    ap.add_argument("--w_router", type=float, default=1.0)
    ap.add_argument("--w_dense", type=float, default=0.0,
                    help="weight of a routing-off distillation pass that keeps the shared weights dense-calibrated")
    ap.add_argument("--w_gt", type=float, default=0.0, help="weight of the GT-depth loss (gt_loss)")
    ap.add_argument("--gt_planner", type=float, default=0.0,
                    help="extra GT-loss weight on samples of tokens with planner label > 0")
    ap.add_argument("--gt_random", action="store_true",
                    help="control: gt_planner weights on randomly permuted tokens (same count per image)")
    ap.add_argument("--gt_source", default="", help="teacher_grids output to use as the GT grid (else real GT)")
    ap.add_argument("--gt_metric", type=float, default=0.0, help="GT-loss weight on the per-image log scale error")
    ap.add_argument("--w_clear", type=float, default=0.0, help="weight of the GT clearance-field loss (clear_loss)")
    ap.add_argument("--clear_void", type=int, default=0, help="1: clear_loss also counts predicted points on GT-void pixels")
    ap.add_argument("--preview_only", type=int, default=0, help="1: write <out>_samples.png at step 1 and exit")
    ap.add_argument("--void_aug", type=float, default=0.0, help="share of train frames given a black GT-void ceiling wedge")
    ap.add_argument("--w_distill",type=float, default=1.0, help="weight of the distillation from the dense student")
    ap.add_argument("--nyu_frac", type=float, default=None, help="share of sampled frames from NYU (default: uniform)")
    ap.add_argument("--grids_out", default="", help="teacher_grids output path")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--max_hours", type=float, default=0.0, help="stop and save after this much training time")
    ap.add_argument("--train_scope", default="all", choices=["all", "routed"])
    ap.add_argument("--heavy", type=int, default=0, help="extra blocks after the AAT stack on the top-k_heavy tokens")
    ap.add_argument("--k_heavy", type=int, default=132)
    ap.add_argument("--heavy_router", default="learned", choices=["learned", "random"])
    ap.add_argument("--lr_heavy", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.stage == "labels":
        labels(a.labels[0])
    elif a.stage == "labels_shards":
        labels_shards(a.shards, a.labels[0])
    elif a.stage == "labels_sim":
        labels_sim(a.sim_dirs, a.labels[0])
    elif a.stage == "teacher_grids":
        teacher_grids([torch.load(p, weights_only=False) for p in a.labels], a.grids_out)
    else:
        train(a)


if __name__ == "__main__":
    main()

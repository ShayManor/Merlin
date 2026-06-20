#!/usr/bin/env python3
"""Fast cache-based distillation: student-only training on precomputed teacher
targets (see precompute_targets.py). ~4x faster per step than online, and the
target cache is reused across every variant (baseline / M1 nav / M2 exits).

Method hook: --nav-weight a  blends navigation-relevance weighting into the depth
loss (M1). a=0 == uniform reconstruction (the ablation baseline).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "backbones"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "eval", "geometry"))
from student import build_student, param_report          # noqa: E402
from distill_online import distill_loss                   # noqa: E402
from data import load_gt_depth                            # noqa: E402
from metrics import depth_metrics, scale_error            # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "..", "eval"))
from nav_metrics import nav_metrics                        # noqa: E402


def load_cache(cache_dir, val_name):
    idx = json.load(open(os.path.join(cache_dir, "index.json")))
    train, val = [], []
    for k in sorted(idx, key=lambda x: int(x)):
        rec = idx[k]
        (val if val_name in rec["rgb"] else train).append(rec["file"])
    print(f"[cache] {len(train)} train, {len(val)} val (val={val_name})", flush=True)
    # load all targets into CPU RAM
    def loadall(files):
        out = []
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            out.append(d)
        return out
    return loadall(train), loadall(val)


def make_view(batch, dev):
    img = torch.stack([b["img"].float() for b in batch], 0).to(dev)
    B = img.shape[0]
    H = W = img.shape[-1]
    ts = np.array([[H, W]] * B, np.int32)
    return [dict(img=img, true_shape=ts, idx=0,
                 instance=[str(i) for i in range(B)], data_norm_type=["dinov2"] * B)]


def targets(batch, dev):
    return {"depth": torch.stack([b["depth"].float() for b in batch], 0).to(dev),
            "ray": torch.stack([b["ray"].float() for b in batch], 0).to(dev),
            "scale": (torch.stack([b["scale"].float() for b in batch], 0).to(dev)
                      if batch[0].get("scale") is not None else None)}


def prep_batch(batch, dev, augment=False, multi_res=None):
    """Build (view, targets) with consistent per-sample augmentation. The cache
    stores fixed preprocessed frames, so without aug the student memorizes (~12
    epochs). H-flip is geometry-correct (flip img/depth/ray on W, negate ray-x);
    color jitter is photometric-only. Targets stay aligned with the flipped image.

    multi_res: if a list of resolutions, pick one per batch and re-decode imgs +
    resize targets to it (trains the student to be accurate at all operating points
    -> a genuine accuracy/latency frontier for the M2 deadline-elastic controller)."""
    R = None
    if multi_res:
        R = int(random.choice(multi_res)); R = (R // 14) * 14
    imgs, depths, rays = [], [], []
    for b in batch:
        if R is not None and R != int(b["img"].shape[-1]):
            from data import make_views  # re-decode raw image at R (dinov2-normalized)
            img = make_views([b["rgb"]], R, "cpu")[0]["img"][0].float()
            depth = F.interpolate(b["depth"].float().permute(2, 0, 1)[None], (R, R),
                                  mode="nearest")[0].permute(1, 2, 0)
            ray = F.interpolate(b["ray"].float().permute(2, 0, 1)[None], (R, R),
                                mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            ray = ray / ray.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        else:
            img = b["img"].float(); depth = b["depth"].float(); ray = b["ray"].float()
        if augment:
            if torch.rand(1).item() < 0.5:                      # horizontal flip
                img = torch.flip(img, [2])
                depth = torch.flip(depth, [1])
                ray = torch.flip(ray, [1]).clone(); ray[..., 0] = -ray[..., 0]
            # photometric (on the dinov2-normalized image): contrast/brightness/noise
            img = img * (0.85 + 0.3 * torch.rand(1).item()) + (0.2 * torch.rand(1).item() - 0.1)
            img = img + 0.02 * torch.randn_like(img)
        imgs.append(img); depths.append(depth); rays.append(ray)
    img = torch.stack(imgs, 0).to(dev)
    B = img.shape[0]; H = W = img.shape[-1]
    view = [dict(img=img, true_shape=np.array([[H, W]] * B, np.int32), idx=0,
                 instance=[str(i) for i in range(B)], data_norm_type=["dinov2"] * B)]
    tgt = {"depth": torch.stack(depths, 0).to(dev), "ray": torch.stack(rays, 0).to(dev),
           "scale": (torch.stack([b["scale"].float() for b in batch], 0).to(dev)
                     if batch[0].get("scale") is not None else None)}
    return view, tgt


def extract(pr):
    return {"depth": pr["depth_along_ray"].float(), "ray": pr["ray_directions"].float(),
            "scale": (pr.get("metric_scaling_factor").float()
                      if pr.get("metric_scaling_factor") is not None else None),
            "ptscam": pr.get("pts3d_cam")}


@torch.no_grad()
def evaluate(student, val, dev, n=24):
    student.eval()
    fid, gtm, sce = [], [], []
    nav = {"near_absrel": [], "far_absrel": [], "col_range_mae": [], "obstacle_iou": []}
    vstep = max(1, len(val) // n)  # stride across the whole held-out trajectory (not the easy first frames)
    for b in val[::vstep][:n]:
        view = make_view([b], dev)
        sp = extract(student(view, memory_efficient_inference=True, minibatch_size=1)[0])
        sz = sp["ptscam"][0, ..., 2].float().cpu().numpy()
        tz = b["depth_z"].float().numpy()
        H, W = sz.shape
        v = np.isfinite(sz) & np.isfinite(tz) & (tz > 0.1) & (sz > 0.05)
        fid.append(float(np.mean(np.abs(tz[v] - sz[v]) / tz[v])) if v.sum() else np.nan)
        gt = load_gt_depth(b["depth_path"], H, W)
        gtm.append(depth_metrics(gt, sz)["abs_rel"]); sce.append(scale_error(gt, sz))
        nm = nav_metrics(tz, sz)  # nav fidelity vs teacher (what control consumes)
        for k in nav:
            if np.isfinite(nm[k]):
                nav[k].append(nm[k])
    student.train()
    out = {"fid_absrel_vs_teacher": round(float(np.nanmean(fid)), 4),
           "gt_absrel": round(float(np.nanmean(gtm)), 4),
           "gt_scale_err": round(float(np.nanmean(sce)), 4)}
    out.update({f"nav_{k}": round(float(np.mean(vs)), 4) for k, vs in nav.items() if vs})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/cache/sz378")
    ap.add_argument("--val-name", default="freiburg1_room")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--nav-weight", type=float, default=0.0)
    ap.add_argument("--colmin", type=float, default=0.0, help="M1-v2: per-column-min (obstacle-range) loss weight")
    ap.add_argument("--encoder-lr-mult", type=float, default=1.0)
    ap.add_argument("--augment", action="store_true", help="h-flip + color jitter (anti-overfit)")
    ap.add_argument("--multi-res", default="", help="comma res list, e.g. 252,378,518 (M2 anytime)")
    ap.add_argument("--multi-exit", default="", help="comma AAT depths, e.g. 6,8 (M2 early-exit deep supervision)")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="/workspace/ckpt/student.pt")
    ap.add_argument("--init", default="")
    ap.add_argument("--tag", default="cache")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0); random.seed(0)

    train, val = load_cache(args.cache, args.val_name)
    student = build_student(size="base", aat_depth=8, device=dev)
    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location=dev, weights_only=False)["state_dict"]
        student.load_state_dict(sd); print(f"[init] loaded {args.init}", flush=True)
    print("[student]", param_report(student), flush=True)
    if args.freeze_encoder:
        for n, p in student.named_parameters():
            if n.startswith("encoder."):
                p.requires_grad_(False)
    student.train()
    # Optional: lower LR for the pretrained DINOv2 encoder (distillation stability).
    enc = [p for n, p in student.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    rest = [p for n, p in student.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    trainable = enc + rest
    if args.encoder_lr_mult != 1.0:
        groups = [{"params": enc, "lr": args.lr * args.encoder_lr_mult}, {"params": rest, "lr": args.lr}]
        print(f"[opt] encoder lr x{args.encoder_lr_mult}", flush=True)
    else:
        groups = [{"params": trainable, "lr": args.lr}]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup)) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, max(0, s - args.warmup) / max(1, args.steps - args.warmup))))))

    log = {"args": vars(args), "steps": [], "eval": []}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = random.sample(train, args.batch)
        mres = [int(r) for r in args.multi_res.split(",")] if args.multi_res else None
        view, tgt = prep_batch(batch, dev, augment=args.augment, multi_res=mres)
        if args.multi_exit:  # early-exit deep supervision: random AAT depth per batch
            student.info_sharing.depth = random.choice([int(k) for k in args.multi_exit.split(",")])
        sp = extract(student(view, memory_efficient_inference=True, minibatch_size=args.batch)[0])
        loss, parts = distill_loss(sp, tgt, nav_alpha=args.nav_weight, colmin_w=args.colmin)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); sched.step()
        if step % 25 == 0 or step == 1:
            it = (time.time() - t0) / step
            print(f"[{step}/{args.steps}] loss={float(loss):.4f} {parts} "
                  f"lr={sched.get_last_lr()[0]:.2e} {it:.2f}s/it", flush=True)
            log["steps"].append({"step": step, "loss": float(loss), **parts})
        if step % args.eval_every == 0 or step == args.steps:
            if args.multi_exit:
                student.info_sharing.depth = max(int(k) for k in args.multi_exit.split(","))  # eval at full depth
            ev = evaluate(student, val, dev); ev["step"] = step
            print(f"  >>> EVAL {ev}", flush=True)
            log["eval"].append(ev)
            torch.save({"state_dict": student.state_dict(), "size": "base",
                        "aat_depth": 8, "step": step, "tag": args.tag}, args.out)
            json.dump(log, open(args.out.replace(".pt", f"_{args.tag}.json"), "w"), indent=2)
    print(f"=== DONE {args.tag} === {args.out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Online distillation: MapAnything teacher -> ~230M student, on a real GPU.

Teacher (fp32, frozen) and student co-resident; each step runs both on a batch of
monocular frames and trains the student to match the teacher's metric outputs.
This is the A40 path (the two-phase disk-cache trainer in train.py existed only
for the 8GB Jetson). Targets: depth-along-ray, ray directions, metric scale.

Method hooks:
  --nav-weight a  : blend a navigation-relevance weighting into the depth loss
                    (M1). a=0 reproduces uniform reconstruction L2 (the ablation).
                    Near-field (small teacher depth) + lower image rows weighted up.

Eval: every --eval-every steps, report student-vs-teacher fidelity and, on a
held-out TUM sequence, metric depth vs GT (abs_rel, delta1, scale error).
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

from data import gather_frames, make_views, load_gt_depth  # noqa: E402
from student import build_student, param_report             # noqa: E402


def nav_weight_map(teacher_depth, alpha, near_m=3.0):
    """Per-pixel navigation-relevance weight (M1).

    Emphasizes near-field geometry (depth < near_m) and the lower image region
    (the robot's drivable frustum). alpha blends uniform(=1) <-> nav weighting.
    teacher_depth: (B,H,W,1). Returns (B,H,W,1) weights, mean-normalized to ~1.
    """
    if alpha <= 0:
        return torch.ones_like(teacher_depth)
    B, H, W, _ = teacher_depth.shape
    d = teacher_depth.clamp_min(0.1)
    # Aggressive near-field focus: inverse-square in depth (~50x dynamic range from
    # 0.5m to 5m) so gradients concentrate on the obstacle zone. A soft near_m/(d+near_m)
    # weight on top of an already depth-relative log-L1 loss was too gentle to reallocate
    # (M1 delta ~0.002). No (1-alpha) floor at alpha=1 -> pure nav weighting.
    near = (1.0 / (d + 0.3)) ** 2                          # steep near emphasis
    rows = torch.linspace(0.15, 1.0, H, device=d.device).view(1, H, 1, 1)  # drivable frustum
    w = near * rows.expand(B, H, W, 1)
    w = w / w.mean().clamp_min(1e-6)                       # keep loss scale stable
    return (1 - alpha) + alpha * w


def colmin_loss(s_depth, t_depth, row_lo=0.4, eps=1e-6):
    """M1-v2: log-L1 on the per-column MINIMUM depth over drivable (lower) rows -- the
    nearest-obstacle range the local planner actually consumes (2D laser-scan proxy).
    M1's near-field MEAN weighting improved mean near abs_rel but worsened this min-range
    (and collisions); this targets the right statistic directly."""
    sd = s_depth.squeeze(-1) if s_depth.dim() == 4 else s_depth
    td = t_depth.squeeze(-1) if t_depth.dim() == 4 else t_depth
    r0 = int(sd.shape[1] * row_lo)
    s_min = sd[:, r0:, :].clamp_min(eps).amin(dim=1)       # (B, W) nearest range per column
    t_min = td[:, r0:, :].clamp_min(eps).amin(dim=1)
    return (s_min.log() - t_min.log()).abs().mean()


def distill_loss(s, t, w_depth=1.0, w_scale=2.0, w_ray=0.5, nav_alpha=0.0, colmin_w=0.0, eps=1e-6):
    sd = s["depth"].clamp_min(eps).log()
    td = t["depth"].clamp_min(eps).log()
    wmap = nav_weight_map(t["depth"], nav_alpha)
    diff = (sd - td).abs()
    log_l1 = (diff * wmap).sum() / wmap.sum().clamp_min(eps)
    si = ((sd - td).flatten(1).var(dim=1) + 1e-8).sqrt().mean()  # per-sample SI
    loss = w_depth * (log_l1 + 0.5 * si)
    parts = {"depth_logL1": round(float(log_l1), 4), "depth_si": round(float(si), 4)}
    if colmin_w > 0:
        cm = colmin_loss(s["depth"], t["depth"])
        loss = loss + colmin_w * cm
        parts["colmin"] = round(float(cm), 4)
    if s.get("scale") is not None and t.get("scale") is not None:
        ls = (s["scale"].clamp_min(eps).log() - t["scale"].clamp_min(eps).log()).abs().mean()
        loss = loss + w_scale * ls
        parts["scale_logL1"] = round(float(ls), 4)
    if s.get("ray") is not None and t.get("ray") is not None:
        ray = (1 - F.cosine_similarity(s["ray"], t["ray"], dim=-1)).mean()
        loss = loss + w_ray * ray
        parts["ray_cos"] = round(float(ray), 4)
    return loss, parts


def extract(pr):
    return {"depth": pr["depth_along_ray"].float(),
            "ray": pr["ray_directions"].float(),
            "scale": (pr.get("metric_scaling_factor").float()
                      if pr.get("metric_scaling_factor") is not None else None),
            "ptscam": pr.get("pts3d_cam")}


@torch.no_grad()
def evaluate(student, teacher, val_pairs, size, dev, n=16):
    from metrics import depth_metrics, scale_error, fidelity_vs_teacher
    student.eval()
    fid, gtm, sce = [], [], []
    for rp, dp in val_pairs[:n]:
        views = make_views([rp], size, dev)
        sp = extract(student(views, memory_efficient_inference=True, minibatch_size=1)[0])
        tp = extract(teacher(views, memory_efficient_inference=True, minibatch_size=1)[0])
        sz = sp["ptscam"][0, ..., 2].float().cpu().numpy()
        tz = tp["ptscam"][0, ..., 2].float().cpu().numpy()
        H, W = sz.shape
        gt = load_gt_depth(dp, H, W)
        fid.append(fidelity_vs_teacher(tz, sz)["abs_rel"])
        m = depth_metrics(gt, sz)
        gtm.append(m["abs_rel"]); sce.append(scale_error(gt, sz))
    student.train()
    return {"fid_absrel_vs_teacher": round(float(np.nanmean(fid)), 4),
            "gt_absrel": round(float(np.nanmean(gtm)), 4),
            "gt_scale_err": round(float(np.nanmean(sce)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-roots", nargs="+", default=["/workspace/data/tum"])
    ap.add_argument("--val-seq", default="/workspace/data/tum/rgbd_dataset_freiburg1_room")
    ap.add_argument("--size", type=int, default=392)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--nav-weight", type=float, default=0.0)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="/workspace/ckpt/student.pt")
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = "cuda"
    torch.manual_seed(0); random.seed(0)

    from teacher import load_teacher  # reuse loader, but force fp32 on A40
    print("=== loading teacher (fp32) ===", flush=True)
    from mapanything.models import MapAnything
    teacher = MapAnything.from_pretrained("facebook/map-anything-apache").to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("=== building student ===", flush=True)
    student = build_student(size="base", aat_depth=8, device=dev)
    print("[student]", param_report(student), flush=True)
    if args.freeze_encoder:
        for n, p in student.named_parameters():
            if n.startswith("encoder."):
                p.requires_grad_(False)
    student.train()

    frames = gather_frames(args.data_roots, stride=args.stride)
    # hold out the val sequence from training
    val_name = os.path.basename(args.val_seq)
    frames = [f for f in frames if val_name not in f]
    from data import tum_pairs
    val_pairs = tum_pairs(args.val_seq)
    print(f"[data] {len(frames)} train frames, {len(val_pairs)} val pairs", flush=True)

    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup)) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, max(0, s - args.warmup) / max(1, args.steps - args.warmup))))))

    log = {"args": vars(args), "steps": [], "eval": []}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        paths = random.sample(frames, args.batch)
        views = make_views(paths, args.size, dev)
        with torch.no_grad():
            tp = extract(teacher(views, memory_efficient_inference=True, minibatch_size=args.batch)[0])
        sp = extract(student(views, memory_efficient_inference=True, minibatch_size=args.batch)[0])
        loss, parts = distill_loss(sp, tp, nav_alpha=args.nav_weight)
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
            ev = evaluate(student, teacher, val_pairs, args.size, dev)
            ev["step"] = step
            print(f"  >>> EVAL {ev}", flush=True)
            log["eval"].append(ev)
            torch.save({"state_dict": student.state_dict(), "size": "base",
                        "aat_depth": 8, "step": step, "tag": args.tag}, args.out)
            with open(args.out.replace(".pt", f"_{args.tag}.json"), "w") as f:
                json.dump(log, f, indent=2)
    print(f"=== DONE {args.tag} === saved {args.out}", flush=True)


if __name__ == "__main__":
    main()

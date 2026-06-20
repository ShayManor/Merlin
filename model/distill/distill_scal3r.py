#!/usr/bin/env python3
"""Online distillation from the Scal3R teacher -> MERLIN student (task 6).

Scal3R outputs per-frame depth (VGGT DPT head). We distill the same ~230M student
(reused from the MapAnything work) to match Scal3R's depth with a SCALE-INVARIANT
log-L1 loss (robust to whether Scal3R depth is metric or up-to-scale -- VGGT-family
is typically up-to-scale, so absolute matching would be wrong).

Teacher and student see the SAME frames at the same patch-aligned square res, but
different preprocessing: Scal3R wants raw RGB in [0,1]; the student wants the
dinov2-normalized tensor from load_images. Both run on the A40 (teacher ~5GB).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "backbones"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "eval", "geometry"))
from data import gather_frames, load_gt_depth          # noqa: E402
from student import build_student, param_report         # noqa: E402
from scal3r_teacher import load_scal3r, scal3r_depth     # noqa: E402
from metrics import depth_metrics, scale_error           # noqa: E402


def teacher_in(path, res, dev):
    s = (res // 14) * 14
    im = Image.open(path).convert("RGB").resize((s, s))
    x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None, None] / 255.0  # (1,1,3,H,W)
    return x.to(dev)


def student_views(path, res, dev):
    from mapanything.utils.image import load_images
    s = (res // 14) * 14
    v = load_images([path], resize_mode="square", size=s)
    for x in v:
        x["img"] = x["img"].to(dev)
    return v


def si_logl1(pred, target, eps=1e-3):
    """Scale-invariant log-L1: align by per-sample mean log-offset, then L1."""
    lp = pred.clamp_min(eps).log()
    lt = target.clamp_min(eps).log()
    diff = lp - lt
    diff = diff - diff.flatten(1).mean(1).view(-1, 1, 1)   # remove global scale
    return diff.abs().mean()


@torch.no_grad()
def evaluate(student, teacher, frames, res, dev, n=20):
    student.eval()
    fid, gtm = [], []
    for rp, dp in frames[:n]:
        tv = teacher_in(rp, res, dev)
        td = scal3r_depth(teacher, tv)[0].float().squeeze().cpu().numpy()
        sv = student_views(rp, res, dev)
        pr = student(sv, memory_efficient_inference=True, minibatch_size=1)[0]
        sd = pr["pts3d_cam"][0, ..., 2].float().cpu().numpy()
        H, W = sd.shape
        if td.shape != (H, W):
            td = np.asarray(Image.fromarray(td).resize((W, H)))
        v = np.isfinite(td) & np.isfinite(sd) & (td > 0.1) & (sd > 0.05)
        if v.sum() > 100:
            r = np.median(td[v] / sd[v])
            fid.append(float(np.mean(np.abs(td[v] - r * sd[v]) / td[v])))  # scale-aligned
        gt = load_gt_depth(dp, H, W)
        gtm.append(depth_metrics(gt, sd)["abs_rel"])
    student.train()
    return {"fid_si_vs_scal3r": round(float(np.nanmean(fid)), 4),
            "gt_absrel": round(float(np.nanmean(gtm)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-roots", nargs="+", default=["/workspace/data/tum"])
    ap.add_argument("--val-name", default="freiburg1_room")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default="/workspace/ckpt/student_scal3r.pt")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = "cuda"
    import random
    torch.manual_seed(0); random.seed(0)

    print("=== loading Scal3R teacher ===", flush=True)
    teacher = load_scal3r(dev)
    print("=== building student ===", flush=True)
    student = build_student(size="base", aat_depth=8, device=dev)
    print("[student]", param_report(student), flush=True)
    student.train()

    frames = gather_frames(args.data_roots, stride=args.stride, with_depth=True)
    val = [f for f in frames if args.val_name in f[0]]
    train = [f for f in frames if args.val_name not in f[0]]
    print(f"[data] {len(train)} train, {len(val)} val", flush=True)

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, args.warmup)) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, max(0, s - args.warmup) / max(1, args.steps - args.warmup))))))

    import json
    log = {"args": vars(args), "eval": []}
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = random.sample(train, args.batch)
        loss = 0.0
        opt.zero_grad(set_to_none=True)
        for rp, _ in batch:
            with torch.no_grad():
                td = scal3r_depth(teacher, teacher_in(rp, args.res, dev))[0].float()  # (1,1,H,W)->? keep (H,W)
                td = td.squeeze()
            sv = student_views(rp, args.res, dev)
            pr = student(sv, memory_efficient_inference=True, minibatch_size=1)[0]
            sd = pr["pts3d_cam"][0, ..., 2].float()
            if td.shape != sd.shape:
                td = F.interpolate(td[None, None], size=sd.shape, mode="bilinear", align_corners=False)[0, 0]
            l = si_logl1(sd[None], td[None])
            l.backward()
            loss += float(l)
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()
        if step % 25 == 0 or step == 1:
            print(f"[{step}/{args.steps}] loss={loss/args.batch:.4f} {(time.time()-t0)/step:.2f}s/it", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(student, teacher, val, args.res, dev); ev["step"] = step
            print(f"  >>> EVAL {ev}", flush=True)
            log["eval"].append(ev)
            torch.save({"state_dict": student.state_dict(), "size": "base", "aat_depth": 8,
                        "step": step, "teacher": "scal3r"}, args.out)
            json.dump(log, open(args.out.replace(".pt", ".json"), "w"), indent=2)
    print("=== SCAL3R_DISTILL_DONE ===", flush=True)


if __name__ == "__main__":
    main()

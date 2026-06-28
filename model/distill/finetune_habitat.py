#!/usr/bin/env python3
"""Domain-adapt the distilled student to the Habitat render distribution by supervised
fine-tuning on rendered (RGB, GT-depth) pairs. The TUM-distilled student degrades OOD on
Habitat (median abs_rel ~0.40, corr 0.15); this closes that gap so the closed-loop
perception-fidelity question is asked with USABLE perception.

--nav alpha re-creates the M1 objective IN-DOMAIN (inverse-square near-field + lower-row
weighting) so M1 vs uniform can be re-tested with good Habitat depth.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "backbones"))
from student import build_student   # noqa: E402
from data import make_views         # noqa: E402


def load_split(root):
    rgbs = sorted(glob.glob(os.path.join(root, "rgb_*.png")))
    return [(r, r.replace("rgb_", "dep_").replace(".png", ".npy")) for r in rgbs]


def gt_to(pred_shape, dep_path):
    import cv2
    d = np.load(dep_path).astype(np.float32)
    if d.shape != pred_shape:
        d = cv2.resize(d, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(d)


def nav_weight(gt, alpha):
    if alpha <= 0:
        return torch.ones_like(gt)
    wd = 1.0 / (gt.clamp(0.1, 10.0) + 0.3) ** 2          # inverse-square: near obstacles
    H = gt.shape[-2]
    rows = torch.linspace(0.3, 1.7, H, device=gt.device).view(1, H, 1)  # lower rows up-weighted
    w = wd * rows
    w = w / (w.mean(dim=(-1, -2), keepdim=True) + 1e-6)
    return (1 - alpha) + alpha * w


def loss_fn(pred_z, gt, alpha):
    v = torch.isfinite(pred_z) & torch.isfinite(gt) & (gt > 0.1) & (gt < 8.0) & (pred_z > 0.05)
    if v.sum() < 200:
        return None
    logp = torch.log(pred_z.clamp_min(0.05)); logg = torch.log(gt.clamp_min(0.05))
    diff = (logp - logg)
    w = nav_weight(gt, alpha) * v.float()
    shift = (w * diff).sum() / w.sum()                  # weighted log-scale alignment (SI)
    return (w * (diff - shift).abs()).sum() / w.sum()


@torch.no_grad()
def evaluate(student, val, dev, res, n=40):
    student.eval()
    mr, corr = [], []
    for rp, dp in val[:n]:
        view = make_views([rp], (res // 14) * 14, dev)
        pr = student(view, memory_efficient_inference=True, minibatch_size=1)[0]
        pz = pr["pts3d_cam"][0, ..., 2].float()
        gt = gt_to(pz.shape, dp).to(dev)
        v = torch.isfinite(pz) & torch.isfinite(gt) & (gt > 0.1) & (gt < 10) & (pz > 0.05)
        if v.sum() < 200:
            continue
        s = torch.median(gt[v] / pz[v]); pa = pz * s
        mr.append(float(torch.median(torch.abs(gt[v] - pa[v]) / gt[v])))
        gv, pv = gt[v].cpu().numpy(), pz[v].cpu().numpy()
        if gv.std() > 1e-6 and pv.std() > 1e-6:
            corr.append(float(np.corrcoef(gv, pv)[0, 1]))
    student.train()
    return {"median_absrel": round(float(np.nanmean(mr)), 4),
            "pearson": round(float(np.nanmean(corr)), 4) if corr else float("nan"),
            "n": len(mr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--init", default="/workspace/ckpt/student_v2.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--nav", type=float, default=0.0, help="M1 nav-weight alpha (in-domain)")
    ap.add_argument("--eval-every", type=int, default=500)
    args = ap.parse_args()
    dev = "cuda"
    student = build_student(size="base", aat_depth=8, device="cpu")
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    student.load_state_dict(ck["state_dict"]); student = student.to(dev).train()
    train, val = load_split(args.train), load_split(args.val)
    print(f"[ft] train {len(train)} val {len(val)}  init {os.path.basename(args.init)} nav={args.nav}", flush=True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    s = (args.res // 14) * 14
    print("[ft] init eval:", evaluate(student, val, dev, args.res), flush=True)
    for step in range(args.steps):
        idx = rng.integers(0, len(train), args.batch)
        paths = [train[i][0] for i in idx]
        view = make_views(paths, s, dev)
        pr = student(view, memory_efficient_inference=True, minibatch_size=args.batch)[0]
        pz = pr["pts3d_cam"][..., 2].float()
        gts = torch.stack([gt_to(pz.shape[1:], train[i][1]) for i in idx], 0).to(dev)
        loss = loss_fn(pz, gts, args.nav)
        if loss is None:
            continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"[ft] step {step} loss {loss.item():.4f}", flush=True)
        if step > 0 and step % args.eval_every == 0:
            print(f"[ft] step {step} eval:", evaluate(student, val, dev, args.res), flush=True)
    print("[ft] final eval:", evaluate(student, val, dev, args.res), flush=True)
    torch.save({"state_dict": student.state_dict(), "size": "base", "aat_depth": 8}, args.out)
    print(f"=== FT_DONE {args.out} ===", flush=True)


if __name__ == "__main__":
    main()

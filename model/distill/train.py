#!/usr/bin/env python3
"""Output-level distillation: MapAnything teacher -> ~300M student (MERLIN).

Two phases so the teacher (4.23GB fp32) and student never sit on the 8GB GPU at
once:
  1. precompute: run the teacher on the image set, cache metric targets to disk,
     free the teacher.
  2. train: build the student, train it to match the cached targets.

Losses (claude.md priorities): metric depth (scale-invariant + log-L1), the
metric scale factor (weighted highest -- the hard-to-compress metric head), and
ray directions (cosine).

This is the recipe + a runnable on-device proof (loss should drop). Reaching
"within 10-15% of teacher" needs many GPU-hours on a real NVIDIA GPU and the
full multi-dataset corpus -- out of scope for the 8GB Jetson. Use --max-steps
small for the proof.
"""
import argparse
import glob
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backbones"))


def _views(image, res, device, dtype=None):
    from mapanything.utils.image import load_images
    s = (res // 14) * 14
    v = load_images([image], resize_mode="square", size=s)
    for x in v:
        x["img"] = x["img"].to(device)
        if dtype is not None:
            x["img"] = x["img"].to(dtype)
    return v


def precompute_targets(paths, res, cache_dir, device):
    # Use forward() (raw head outputs), NOT infer(): infer is @inference_mode
    # (no grad) and its geometry postproc hits an unsupported cuSOLVER symbol on
    # the Jetson. forward() gives the differentiable raw targets directly.
    from teacher import load_teacher
    os.makedirs(cache_dir, exist_ok=True)
    teacher = load_teacher(device=device)
    for i, p in enumerate(paths):
        out_path = os.path.join(cache_dir, f"{i:04d}.pt")
        if os.path.exists(out_path):
            continue
        v = _views(p, res, device)
        with torch.no_grad():
            pr = teacher(v, memory_efficient_inference=True, minibatch_size=1)[0]
        torch.save({"path": p,
                    "depth": pr["depth_along_ray"].float().cpu(),
                    "ray": pr["ray_directions"].float().cpu(),
                    "scale": (pr.get("metric_scaling_factor").float().cpu()
                              if pr.get("metric_scaling_factor") is not None else None)},
                   out_path)
        print(f"[precompute {i+1}/{len(paths)}] {os.path.basename(p)}", flush=True)
    del teacher
    torch.cuda.empty_cache()


def distill_loss(s, t, w_depth=1.0, w_scale=2.0, w_ray=0.5, eps=1e-6):
    sd = s["depth"].clamp_min(eps).log()
    td = t["depth"].to(sd.device).clamp_min(eps).log()
    log_l1 = (sd - td).abs().mean()
    si = ((sd - td).var() + 1e-8).sqrt()
    loss = w_depth * (log_l1 + 0.5 * si)
    parts = {"depth_logL1": round(float(log_l1), 4), "depth_si": round(float(si), 4)}
    if s["scale"] is not None and t["scale"] is not None:
        ls = (s["scale"].clamp_min(eps).log() - t["scale"].to(s["scale"].device).clamp_min(eps).log()).abs().mean()
        loss = loss + w_scale * ls
        parts["scale_logL1"] = round(float(ls), 4)
    if s["ray"] is not None and t["ray"] is not None:
        ray = (1 - F.cosine_similarity(s["ray"], t["ray"].to(s["ray"].device), dim=-1)).mean()
        loss = loss + w_ray * ray
        parts["ray_cos"] = round(float(ray), 4)
    return loss, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--size", default="base")
    ap.add_argument("--aat-depth", type=int, default=8)
    ap.add_argument("--res", type=int, default=252)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/merlin/distill_targets"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.expanduser("~/merlin/student_distilled.pt"))
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(os.path.join(args.images, "*.png")))
    assert paths, f"no images in {args.images}"
    dev = args.device

    print("=== phase 1: precompute teacher targets ===", flush=True)
    precompute_targets(paths, args.res, args.cache_dir, dev)

    print("=== phase 2: train student ===", flush=True)
    sys.path.insert(0, HERE)
    from student import build_student, param_report
    student = build_student(args.size, args.aat_depth, device=dev)
    print("[student]", param_report(student), flush=True)
    if args.freeze_encoder:
        for n, p in student.named_parameters():
            if n.startswith("encoder."):
                p.requires_grad_(False)
    trainable = [p for p in student.parameters() if p.requires_grad]
    print(f"[trainable] {sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    targets = sorted(glob.glob(os.path.join(args.cache_dir, "*.pt")))
    step = 0
    t0 = time.time()
    losses = []
    while step < args.max_steps:
        for tf in targets:
            if step >= args.max_steps:
                break
            tgt = torch.load(tf, map_location=dev, weights_only=True)
            v = _views(tgt["path"], args.res, dev)
            pr = student(v, memory_efficient_inference=True, minibatch_size=1)[0]  # forward(): differentiable
            s = {"depth": pr["depth_along_ray"].float(), "ray": pr["ray_directions"].float(),
                 "scale": pr.get("metric_scaling_factor")}
            loss, parts = distill_loss(s, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1
            losses.append(float(loss))
            if step % 5 == 0 or step == 1:
                print(f"[step {step}] loss={float(loss):.4f} {parts} ({(time.time()-t0)/step:.1f}s/it)", flush=True)

    torch.save({"state_dict": student.state_dict(), "size": args.size, "aat_depth": args.aat_depth,
                "loss_curve": losses}, args.out)
    print(f"[saved] {args.out}  first_loss={losses[0]:.4f} last_loss={losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()

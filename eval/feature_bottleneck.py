#!/usr/bin/env python3
"""Geometry-aware feature bottleneck (the pivot wedge).

For SPLIT inference of a metric-3D model across two edge SoCs, the intermediate
feature at the split crosses a bandwidth-limited link. Feature-compression / bottleneck
methods (FrankenSplit, BottleFit) optimize for a RECOGNITION scalar and tolerate lossy
features. For a METRIC-3D model the silent failure is global SCALE drift, not a class-prob
drop. This probes the rate-distortion of the encoder->reasoning split feature: compress it
at varying rates and measure the scale-vs-geometry distortion of the metric depth output.
Question: is SCALE the binding constraint for FEATURE compression (as it is, at the cliff,
for weight compression)? What min rate preserves metric depth?
"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "geometry"))
from student import build_student          # noqa: E402
from data import make_views, tum_pairs, load_gt_depth  # noqa: E402
from quant_decomp import decompose         # noqa: E402


def q_feature(x, bits):
    """Per-channel (last-dim) symmetric quant of a feature tensor -> dequant."""
    if bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    s = amax / qmax
    return torch.clamp(torch.round(x / s), -qmax, qmax) * s


def lowrank_feature(x, k):
    """Keep top-k principal directions of the feature (per forward) -> rate = k dims."""
    if k <= 0:
        return x
    sh = x.shape
    X = x.reshape(-1, sh[-1]).float()
    mu = X.mean(0, keepdim=True)
    Xc = X - mu
    try:
        U, S, V = torch.linalg.svd(Xc, full_matrices=False)
    except Exception:
        return x
    Vk = V[:k]
    Xr = (Xc @ Vk.T) @ Vk + mu
    return Xr.reshape(sh).to(x.dtype)


class Bottleneck:
    def __init__(self, mode, level):
        self.mode, self.level, self.enabled = mode, level, True
    def hook(self, mod, args):
        if not self.enabled:
            return None
        out = []
        for a in args:
            if torch.is_tensor(a) and a.is_floating_point() and a.dim() >= 2:
                out.append(q_feature(a, self.level) if self.mode == "quant"
                           else lowrank_feature(a, self.level))
            else:
                out.append(a)
        return tuple(out)


@torch.no_grad()
def infer(model, frames, dev, res):
    model.eval(); s = (res // 14) * 14; out = []
    for rp, dp in frames:
        v = make_views([rp], s, dev)
        pr = model(v, memory_efficient_inference=True, minibatch_size=1)[0]
        out.append(pr["pts3d_cam"][0, ..., 2].float().cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/student_v3.pt")
    ap.add_argument("--tum-root", default="/workspace/data/tum")
    ap.add_argument("--seqs", default="rgbd_dataset_freiburg1_xyz,rgbd_dataset_freiburg1_plant")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--split", default="info_sharing", help="module whose INPUT is the wire feature")
    ap.add_argument("--out", default="/root/feat_bottleneck.json")
    args = ap.parse_args()
    dev = "cuda"

    model = build_student(size="base", aat_depth=8, device="cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"]); model = model.to(dev)

    # find the split module + its feature dim
    split_mod = dict(model.named_modules())[args.split]
    bn = Bottleneck("quant", 16)
    split_mod.register_forward_pre_hook(bn.hook)

    frames = []
    for sq in args.seqs.split(","):
        frames += tum_pairs(os.path.join(args.tum_root, sq))[::args.stride]
    frames = frames[:args.limit]

    bn.enabled = False
    ref = infer(model, frames, dev, args.res)
    shapes = [r.shape for r in ref]
    import cv2
    gt = [load_gt_depth(dp, sh[0], sh[1]) for (rp, dp), sh in zip(frames, shapes)]
    bn.enabled = True
    print(f"[feat-bottleneck] split={args.split} n={len(frames)} (ref=uncompressed feature)")

    results = {"meta": {"split": args.split, "n": len(frames)}, "runs": []}
    # quant rate sweep (bits per channel) + low-rank sweep (dims kept)
    configs = [("quant", b) for b in (8, 6, 4, 3, 2)] + [("lowrank", k) for k in (512, 256, 128, 64, 32, 16)]
    for mode, lvl in configs:
        bn.mode, bn.level = mode, lvl
        q = infer(model, frames, dev, args.res)
        d = decompose(ref, q, gt)
        d.update({"mode": mode, "level": lvl})
        results["runs"].append(d)
        print(f"[{mode}] level={lvl:4}  scale_drift={d['scale_drift_vs_ref']}  geom={d['geom_absrel_vs_ref']}"
              f"  scale_ratio={d.get('scale_ratio_qref')}  gt_absrel={d['gt_absrel']}  gt_imu={d['gt_absrel_imu']}",
              flush=True)
    import json
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()

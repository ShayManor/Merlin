#!/usr/bin/env python3
"""Robust in-sim depth quality for one or more checkpoints on a (optionally held-out) scene:
median abs_rel (scale-aligned) + Pearson correlation with GT. Used to verify domain
adaptation actually fixed the OOD depth before re-running the closed-loop nulls."""
import argparse
import sys

import numpy as np

sys.path.insert(0, "eval/sim")
sys.path.insert(0, "model/distill")
sys.path.insert(0, "model/backbones")
from sim_client import SimClient            # noqa: E402
from run_experiments import spawn_server    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="tag=path ...")
    ap.add_argument("--scene", default="apt_5")
    ap.add_argument("--dataset", default="/root/habitat-data/replica_cad/replicaCAD.scene_dataset_config.json")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--port", type=int, default=5798)
    args = ap.parse_args()
    import cv2  # noqa
    from perception import Perception
    srv = spawn_server(args.port, "/workspace/miniconda3/etc/profile.d/conda.sh",
                       "/root/conda_envs/habenv")
    cl = SimClient(args.port)
    frames = []
    try:
        for k in range(args.frames):
            ep = cl.sample(args.scene, 1.0, 10.0, 12345 + k, args.dataset)
            if ep:
                o = cl.reset(args.scene, ep["start"], float(k * 0.7), ep["goal"], args.dataset)
                frames.append((o["rgb"].copy(), o["depth"].copy()))
    finally:
        cl.close(); srv.terminate()

    for c in args.ckpts:
        tag, path = c.split("=", 1)
        p = Perception(path, device="cuda")
        mr, nr, corr = [], [], []
        for rgb, gt in frames:
            p.scale = 1.0
            pred = p._raw_depth(rgb, args.res)
            if pred.shape != gt.shape:
                g = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                g = gt
            v = np.isfinite(g) & np.isfinite(pred) & (g > 0.1) & (g < 10) & (pred > 0.05)
            if v.sum() < 200:
                continue
            s = np.median(g[v] / pred[v]); pa = pred * s
            mr.append(np.median(np.abs(g[v] - pa[v]) / g[v]))
            nb = v & (g < 2.0)
            if nb.sum() > 100:
                nr.append(np.median(np.abs(g[nb] - pa[nb]) / g[nb]))
            if pred[v].std() > 1e-6:
                corr.append(np.corrcoef(g[v], pred[v])[0, 1])
        print(f"{tag:16s} median_absrel={np.mean(mr):.3f} near={np.mean(nr):.3f} "
              f"corr={np.nanmean(corr):.3f} (scene={args.scene}, n={len(mr)})", flush=True)
        del p
        import torch; torch.cuda.empty_cache()
    print("=== DEPTH_CHECK_DONE ===", flush=True)


if __name__ == "__main__":
    main()

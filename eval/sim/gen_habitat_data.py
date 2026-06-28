#!/usr/bin/env python3
"""Render a (RGB, GT-depth) dataset from Habitat navigable poses, for domain-adapting the
distilled student to the Habitat render distribution (it is trained on TUM and degrades OOD).
Train scenes and a HELD-OUT val scene are kept separate so adaptation is measured honestly.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "model", "backbones"))
from sim_client import SimClient            # noqa: E402
from run_experiments import spawn_server    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--per-scene", type=int, default=700)
    ap.add_argument("--yaws", type=int, default=4, help="random yaws per navigable point")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--env-name", default="/root/conda_envs/habenv")
    ap.add_argument("--conda-sh", default="/workspace/miniconda3/etc/profile.d/conda.sh")
    ap.add_argument("--port", type=int, default=5795)
    args = ap.parse_args()
    import cv2
    os.makedirs(args.out, exist_ok=True)
    server = spawn_server(args.port, args.conda_sh, args.env_name)
    client = SimClient(args.port)
    n = 0
    try:
        for si, scene in enumerate(args.scenes):
            pts = max(1, args.per_scene // args.yaws)
            for k in range(pts):
                ep = client.sample(scene, 0.5, 12.0, args.seed0 + 1000 * si + k, args.dataset)
                if not ep:
                    continue
                for j in range(args.yaws):
                    yaw = float((j / args.yaws) * 2 * np.pi + 0.37 * (k % 7))
                    obs = client.reset(scene, ep["start"], yaw, ep["goal"], args.dataset)
                    d = obs["depth"]
                    if not np.isfinite(d).any() or (d > 0.1).mean() < 0.3:
                        continue
                    cv2.imwrite(os.path.join(args.out, f"rgb_{n:06d}.png"),
                                cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR))
                    np.save(os.path.join(args.out, f"dep_{n:06d}.npy"), d.astype(np.float16))
                    n += 1
            print(f"[{scene}] total {n}", flush=True)
    finally:
        client.close(); server.terminate()
    print(f"=== GEN_DONE n={n} out={args.out} ===", flush=True)


if __name__ == "__main__":
    main()

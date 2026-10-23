#!/usr/bin/env python3
"""Render a NEAR-FIELD (RGB, GT-depth) training set from the drone renderer.

The student's depth is accurate past 1 m and collapses inside it (0.2-0.5 m: 0.64 relative
error, and the error is noise rather than a correctable offset, so no post-hoc calibration
recovers it -- five have been tried). The reason is visible in the training distribution: a
navmesh-sampled pose stands `nav_radius` from every surface and looks down a corridor, so
almost no training pixel is ever closer than a metre. The model is not failing at 0.3 m, it
was never shown 0.3 m.

This generator produces those poses deliberately. From a navigable point it renders, reads
how far the surface ahead actually is, and pushes the camera forward until that surface sits
`--target` metres away, rejecting the pose if the drone body would be in contact. Obstacles
are injected first, so a share of the near surfaces are free-standing objects (the thing the
drone hits) rather than walls.

`--mix` keeps a fraction of ordinary route poses. Training on near frames alone would trade
the near field for the 1-2.5 m band, and 1-2.5 m is where the map is actually built.

Output format matches model/distill/finetune_habitat.py: rgb_%06d.png + dep_%06d.npy.
"""
import argparse
import os
import sys
import time

import numpy as np

from eval.drone.sim_client import SimClient

# route_poses/spawn live in diag_depth, which pulls in the student (and torch). Imported
# lazily so this module -- which never runs the model -- stays importable without it.

PORT = 5601 + int(os.environ.get("MERLIN_PX4_INSTANCE", 0))

# Fraction of the frame that must be closer than NEAR_M for a pose to count as near-field.
NEAR_M, NEAR_FRAC = 1.0, 0.20
BODY_R = 0.18   # drone body radius; a pose inside this of geometry is not a pose it can fly


def forward_of(R):
    """Habitat camera matrix columns are right/up/back, so it looks along -Z."""
    return -np.asarray(R, float)[:, 2]


def central_depth(d, frac=0.35):
    """Median GT depth over the middle of the frame: what the camera is looking AT, robust to
    a stray near pixel at the image edge (which a min would chase)."""
    H, W = d.shape
    h, w = int(H * frac / 2), int(W * frac / 2)
    c = d[H // 2 - h:H // 2 + h, W // 2 - w:W // 2 + w]
    c = c[np.isfinite(c) & (c > 0.05)]
    return float(np.median(c)) if c.size > 50 else None


def push_in(client, p, R, target, max_step, body_clear=BODY_R):
    """Move the camera along its own view direction until the surface ahead is `target` m
    away. Returns the new render, or None if the pose is unusable."""
    out = client.render(p, R)
    d0 = central_depth(out["depth"])
    if d0 is None or d0 < target or d0 > 6.0:
        return None
    step = min(d0 - target, max_step)
    q = np.asarray(p, float) + forward_of(R) * step
    if client.collide([float(x) for x in q], body_clear)["hit"]:
        return None
    out = client.render(q, R)
    d = out["depth"]
    m = np.isfinite(d) & (d > 0.05)
    if m.mean() < 0.5 or (d[m] < NEAR_M).mean() < NEAR_FRAC:
        return None
    return out


def write(out_dir, n, rgb, depth):
    import cv2
    cv2.imwrite(os.path.join(out_dir, f"rgb_{n:06d}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(out_dir, f"dep_{n:06d}.npy"), depth.astype(np.float16))


def generate(client, args, scenes, out_dir):
    from eval.drone.diag_depth import route_poses
    os.makedirs(out_dir, exist_ok=True)
    n, near, tried = 0, 0, 0
    for scene in scenes:
        # nav_radius is deliberately small here: the sampler keeps a start that far from every
        # surface, so the default 0.35 would put a floor under how close a pose can ever be.
        client.reset(scene, args.dataset, args.nav_radius)
        for k in range(args.episodes):
            epi = client.sample(scene, args.min_geo, args.max_geo, seed=args.seed0 + k,
                                alt=args.alt, dataset=args.dataset)
            if epi is None:
                continue
            wps = epi.get("waypoints_hab") or [epi["start_hab"], epi["goal_hab"]]
            client.clear_obstacles()
            if args.n_obs:
                client.place_obstacles(wps, args.n_obs, seed=args.seed0 + k, alt=args.alt,
                                       intrude=0.9, min_width=0.35, block_margin=-0.10,
                                       jitter=0.35, include=args.obs_include or None)
            poses = route_poses(wps, args.spacing)
            for i, (p, R) in enumerate(poses):
                if args.mix and (i % args.mix) == 0:
                    out = client.render(p, R)          # ordinary route pose, keeps mid-range
                    write(out_dir, n, out["rgb"], out["depth"]); n += 1
                for t in args.targets:
                    tried += 1
                    out = push_in(client, p, R, t, args.max_step, args.body_clear)
                    if out is None:
                        continue
                    write(out_dir, n, out["rgb"], out["depth"]); n += 1
                    near += 1
                if n >= args.limit:
                    break
            if n >= args.limit:
                break
        print(f"  {os.path.basename(scene)}: {n} frames ({near} near, "
              f"{near/max(tried,1):.0%} of near attempts kept)", flush=True)
        if n >= args.limit:
            break
    return n, near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--val_scenes", nargs="+", default=[])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--alt", type=float, default=0.75)
    ap.add_argument("--nav_radius", type=float, default=0.12)
    ap.add_argument("--min_geo", type=float, default=2.0)
    ap.add_argument("--max_geo", type=float, default=14.0)
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--n_obs", type=int, default=2)
    ap.add_argument("--obs_include", nargs="+", default=[],
                    help="restrict injected TRAINING obstacles to handles containing these "
                         "substrings, e.g. chair. The chair recovers only 0.56 of its depth "
                         "step vs 0.76-0.97 for other objects, so thin strut structure has "
                         "to be over-represented on purpose")
    ap.add_argument("--body_clear", type=float, default=BODY_R,
                    help="reject a pose with geometry inside this radius. Defaults to the drone "
                         "body radius, i.e. only poses it could actually fly. Lower it to render "
                         "a CONTACT subset: those frames are closer than the drone can go, but "
                         "the band is otherwise unsupervised, and an unsupervised band is exactly "
                         "what the predicted-range bias measures.")
    ap.add_argument("--targets", nargs="+", type=float, default=[0.35, 0.55, 0.8],
                    help="distances (m) to place the surface ahead at")
    ap.add_argument("--max_step", type=float, default=3.0)
    ap.add_argument("--mix", type=int, default=4,
                    help="also keep every Nth ordinary route pose; 0 = near frames only")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--val_limit", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=7000)
    args = ap.parse_args()

    from eval.drone.diag_depth import spawn
    proc = spawn(PORT, args.res, 90.0)
    try:
        time.sleep(12)
        client = SimClient(PORT)
        n, near = generate(client, args, args.scenes, os.path.join(args.out, "train"))
        print(f"train: {n} frames ({near} near)", flush=True)
        if args.val_scenes:
            args.limit = args.val_limit
            m, mn = generate(client, args, args.val_scenes, os.path.join(args.out, "val"))
            print(f"val: {m} frames ({mn} near)", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=20)
    print(f"=== GEN_NF_DONE {args.out} ===", flush=True)


if __name__ == "__main__":
    main()

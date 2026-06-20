"""Scal3R teacher adapter for MERLIN distillation (alternate backbone, task 6).

Scal3R (arXiv:2604.08542, CVPR'26 Highlight) is VGGT + test-time training for
ultra-long sequences. For per-frame distillation targets we want the BASE VGGT
forward (aggregator -> DPT depth), bypassing the TTT/offload backend pipeline.

Clean teacher path (mono, S=1):
  output_dict, patch_start_idx = model.agg_regator(images)   # images [B,S,3,H,W] in [0,1]
  tokens_list = [output_dict[i] for i in model.agg_regator.intermediate_layer_idx]
  depth = model.dpt_decoder(tokens_list, images, patch_start_idx)  # [B,S,1,H,W] (+conf)

Run on the A40 (5GB checkpoint, ViT-L embed_dim 1024). The TTT layers (global,
idx 14/17/20/23) adapt over a sequence; for single-frame targets they are inert
or can be disabled via global_use_ttt:false in the config.
"""
import os
import sys

import torch

SCAL3R_ROOT = "/workspace/Scal3R"
CONFIG = "configs/models/scal3r.yaml"


def load_scal3r(device="cuda", config=CONFIG):
    sys.path.insert(0, SCAL3R_ROOT)
    cwd = os.getcwd()
    os.chdir(SCAL3R_ROOT)  # config paths are relative to repo root
    try:
        from scal3r.models.scal3r import build_sampler_from_config
        sampler, dataset_cfg = build_sampler_from_config(config, torch.device(device))
    finally:
        os.chdir(cwd)
    sampler.eval()
    return sampler


@torch.no_grad()
def scal3r_depth(model, images):
    """images: (B, S, 3, H, W) in [0,1]. Returns depth (B,S,1,H,W) and the raw
    aggregator tokens (for feature distillation if wanted)."""
    out = model.agg_regator(images)
    tokens, patch_start_idx = out if isinstance(out, tuple) else (out, 0)
    # DPTHead indexes aggregated_tokens_list[layer_idx] by ABSOLUTE layer index
    # (e.g. 14/17/20/23), so pass the full tokens container (dict or list), NOT a sub-list.
    dpt = model.dpt_decoder(tokens, images, patch_start_idx)
    depth = dpt[0] if isinstance(dpt, (tuple, list)) else dpt
    return depth, tokens, patch_start_idx


if __name__ == "__main__":
    import argparse, glob, time
    import numpy as np
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="/workspace/data/tum/rgbd_dataset_freiburg1_desk")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    print("loading Scal3R ...", flush=True)
    model = load_scal3r(dev)
    print(f"loaded in {time.time()-t0:.1f}s, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    rgb = sorted(glob.glob(f"{args.seq}/rgb/*.png"))[:args.n]
    s = (args.res // 14) * 14
    for p in rgb:
        im = Image.open(p).convert("RGB").resize((s, s))
        x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None, None] / 255.0  # (1,1,3,H,W)
        x = x.to(dev)
        depth, toks, psi = scal3r_depth(model, x)
        d = depth.float().squeeze().cpu().numpy()
        print(f"{os.path.basename(p)}: depth shape={depth.shape} range[{d.min():.2f},{np.median(d):.2f},{d.max():.2f}] "
              f"ntok_layers={len(toks)} patch_start_idx={psi}", flush=True)
    print("SCAL3R_TEACHER_OK", flush=True)

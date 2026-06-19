"""~300M student for MERLIN, derived from the MapAnything teacher config.

The student keeps the teacher's structure (DINOv2 image encoder + alternating-
attention info-sharing + DPT/pose/scale heads) but shrinks the two parts that
dominate params and memory traffic:
  - encoder: ViT-Giant (dim 1536, ~280M) -> ViT-Base (dim 768, ~86M)
  - info-sharing: 16 layers @ 1536 -> 8 layers @ 768 (~260M -> ~40M)
The pose/scale/metric head structure is preserved -- per claude.md the metric
head is the hard part to compress, so we keep it intact and lean on distillation
(and higher quant precision) there rather than shrinking it.

The encoder is initialized from pretrained DINOv2 ViT-B (good init); the rest is
trained by distillation from the teacher.
"""
import copy
import json
import os

import torch

from mapanything.models import MapAnything

# ViT-Base intermediate-feature indices for an 8-layer info-sharing stack.
# Must be < depth. DPT uses encoder feats + these AAT feats.
STUDENT_AAT_INDICES = [3, 5]


def _teacher_config(hf_id="facebook/map-anything-apache"):
    from huggingface_hub import snapshot_download
    snap = snapshot_download(hf_id, allow_patterns=["*.json"])
    with open(os.path.join(snap, "config.json")) as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def student_config(size="base", aat_depth=8, hf_id="facebook/map-anything-apache"):
    """Build a shrunk config dict for the student."""
    cfg = copy.deepcopy(_teacher_config(hf_id))
    dims = {"small": 384, "base": 768, "large": 1024}
    heads = {"small": 6, "base": 12, "large": 16}
    dim, nh = dims[size], heads[size]

    enc = cfg["encoder_config"]
    enc["size"] = size
    enc.pop("keep_first_n_layers", None)  # use all layers of the smaller ViT

    isc = cfg["info_sharing_config"]["module_args"]
    isc["dim"] = dim
    isc["num_heads"] = nh
    isc["depth"] = aat_depth
    isc["indices"] = STUDENT_AAT_INDICES
    isc["size"] = f"{aat_depth}_layers"
    isc["name"] = f"aat_{aat_depth}_layers_vit{size[0]}_dim_ifr"

    # Lighter MLP in info-sharing (swiglu -> standard mlp) to save params.
    cfg["info_sharing_mlp_layer_str"] = "mlp"
    return cfg


def build_student(size="base", aat_depth=8, device="cuda", hf_id="facebook/map-anything-apache", dtype=None):
    cfg = student_config(size, aat_depth, hf_id)
    model = MapAnything(**cfg)
    if dtype is not None:
        model = model.to(dtype)
        # same dtype-mismatch fix as the teacher (geometric encoders on fp32 inputs)
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backbones"))
        from teacher import add_dtype_casting_hooks
        add_dtype_casting_hooks(model)
    return model.to(device)


def param_report(model):
    groups = {"encoder": 0, "info_sharing": 0, "heads": 0, "geometric": 0, "other": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        if name.startswith("encoder."):
            groups["encoder"] += n
        elif "info_sharing" in name:
            groups["info_sharing"] += n
        elif any(h in name for h in ("head", "adaptor", "scale_token", "dpt")):
            groups["heads"] += n
        elif any(g in name for g in ("ray_dirs_encoder", "depth_encoder", "cam_rot", "cam_trans", "depth_scale", "scale_encoder")):
            groups["geometric"] += n
        else:
            groups["other"] += n
    total = sum(groups.values())
    return {"total_M": round(total / 1e6, 1), **{k: round(v / 1e6, 1) for k, v in groups.items()}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base")
    ap.add_argument("--aat-depth", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    m = build_student(args.size, args.aat_depth, device=args.device)
    print(json.dumps(param_report(m), indent=2))

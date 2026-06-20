#!/usr/bin/env python3
"""Attempt to export the MERLIN student's mono compute-core to ONNX (for a TRT
INT8 engine). Wraps encoder -> geometric fusion -> info_sharing -> dense+scale
heads into a plain tensor-in/tensor-out module. Geometry postproc (rays->pointmap)
stays in Python outside the engine.

Runs on CPU (does not touch the training GPU). Reports the first blocking op so we
know whether a TRT path is viable or needs op replacement (flash-attn/LaCT/custom
pos-enc are the likely walls on the AAT stack).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "distill"))
sys.path.insert(0, os.path.join(HERE, "..", "backbones"))
from student import build_student  # noqa: E402


class MonoCore(nn.Module):
    """img (1,3,H,W) -> (dense_value, scale_value). Replicates MapAnything.forward
    for a single images-only view, up to the raw head outputs."""

    def __init__(self, m, hw):
        super().__init__()
        self.m = m
        self.hw = hw  # (H, W)

    def forward(self, img):
        from uniception.models.info_sharing.base import MultiViewTransformerInput
        views = [{"img": img, "data_norm_type": ["dinov2"],
                  "true_shape": torch.tensor([[self.hw[0], self.hw[1]]])}]
        feats, regs = self.m._encode_n_views(views)
        feats = self.m._encode_and_fuse_optional_geometric_inputs(views, feats)
        isi = MultiViewTransformerInput(
            features=feats, additional_input_tokens_per_view=regs,
            additional_input_tokens=self.m.scale_token.unsqueeze(0))
        final, inter = self.m.info_sharing(isi)
        dhi = [torch.cat(feats, 0),
               torch.cat(inter[0].features, 0),
               torch.cat(inter[1].features, 0),
               torch.cat(final.features, 0)]
        dense = self.m.downstream_dense_head(dhi, self.hw)
        from uniception.models.prediction_heads.base import PredictionHeadTokenInput, AdaptorInput
        sh = self.m.scale_head(PredictionHeadTokenInput(last_feature=final.additional_token_features))
        sc = self.m.scale_adaptor(AdaptorInput(adaptor_feature=sh.decoded_channels, output_shape_hw=self.hw))
        return dense.value, sc.value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--res", type=int, default=378)
    ap.add_argument("--out", default="/workspace/merlin_student_core.onnx")
    args = ap.parse_args()
    dev = "cpu"
    m = build_student(size="base", aat_depth=8, device=dev).eval()
    if args.ckpt and os.path.exists(args.ckpt):
        m.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False)["state_dict"])
    H = W = args.res
    core = MonoCore(m, (H, W)).eval()
    img = torch.randn(1, 3, H, W)

    print("=== sanity forward ===", flush=True)
    with torch.no_grad():
        dv, sv = core(img)
    print("dense", tuple(dv.shape), "scale", tuple(sv.shape), flush=True)

    print("=== onnx export (opset 17) ===", flush=True)
    with torch.no_grad():
        torch.onnx.export(core, (img,), args.out, opset_version=17,
                          input_names=["img"], output_names=["dense", "scale"],
                          dynamo=False)
    print(f"[exported] {args.out}", flush=True)


if __name__ == "__main__":
    main()

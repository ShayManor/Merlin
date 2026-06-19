"""Smart post-training quantization for MERLIN, via torchao.

Strategy (claude.md): INT8 broadly, INT4 on the heavy encoder + info-sharing
linears (the bulk of the params and memory traffic), but keep the metric / scale
/ pose heads at higher precision -- the metric head is the part that loses
accuracy under aggressive quantization.

This is weight-only PTQ, which runs directly on the Jetson GPU in PyTorch (no
ONNX/TensorRT export needed) and is the reliable first quantization path. A
TensorRT INT8 engine is a separate, higher-effort export.
"""
import sys
import types

import torch


def _stub_missing_inductor():
    """torchao 0.13 eagerly imports torch._inductor.kernel.flex_attention, which
    the Jetson torch 2.11 build lacks. That path (prototype int8-SDPA inductor
    lowering) is unused for weight-only PTQ, so stub it to let torchao import."""
    name = "torch._inductor.kernel.flex_attention"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        m = types.ModuleType(name)
        m.construct_strides = lambda *a, **k: None
        m.maybe_realize = lambda *a, **k: None
        sys.modules[name] = m


# Name fragments for modules whose precision we protect (metric-critical).
PROTECTED = ("scale_head", "pose_head", "scale_token", "scale_adaptor", "pose_adaptor")
# Name fragments for the heavy bulk we push to INT4.
HEAVY = ("encoder", "info_sharing")


def _is_protected(name):
    return any(p in name for p in PROTECTED)


def _is_heavy(name):
    return any(h in name for h in HEAVY)


def quantize_model(model, scheme="int8", protect_heads=True):
    """Quantize Linear layers in-place.

    scheme:
      "int8"      - INT8 weight-only everywhere (except protected heads)
      "int4_int8" - INT4 weight-only on encoder+info_sharing, INT8 elsewhere
      "int8_dyn"  - INT8 dynamic activation + INT8 weight (heavier compute path)
    """
    _stub_missing_inductor()
    from torchao.quantization import (
        quantize_,
        Int8WeightOnlyConfig,
        Int4WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
    )

    def make_filter(want_heavy=None, group=128):
        def f(mod, fqn):
            if not isinstance(mod, torch.nn.Linear):
                return False
            if protect_heads and _is_protected(fqn):
                return False
            # torchao grouped quant needs in_features % group == 0 and a real 2D
            # weight; tiny/odd linears (e.g. 1-dim projections) break the reshape.
            out_f, in_f = mod.weight.shape
            if in_f % group != 0 or out_f < 8:
                return False
            if want_heavy is None:
                return True
            return _is_heavy(fqn) if want_heavy else not _is_heavy(fqn)
        return f

    if scheme == "int8":
        quantize_(model, Int8WeightOnlyConfig(), filter_fn=make_filter())
    elif scheme == "int8_dyn":
        quantize_(model, Int8DynamicActivationInt8WeightConfig(), filter_fn=make_filter())
    elif scheme == "int4_int8":
        # INT4 needs a group size that divides in_features; 128 is the torchao default.
        quantize_(model, Int4WeightOnlyConfig(group_size=128), filter_fn=make_filter(want_heavy=True))
        quantize_(model, Int8WeightOnlyConfig(), filter_fn=make_filter(want_heavy=False))
    else:
        raise ValueError(f"unknown scheme {scheme}")
    return model


def count_linears(model):
    n_total = n_protected = n_heavy = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            n_total += 1
            if _is_protected(name):
                n_protected += 1
            elif _is_heavy(name):
                n_heavy += 1
    return {"linear_total": n_total, "protected": n_protected, "heavy_int4_candidates": n_heavy}

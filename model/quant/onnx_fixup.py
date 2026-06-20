#!/usr/bin/env python3
"""Sanitize an ONNX graph for TensorRT: replace non-finite (inf) constants with a
finite fp16-safe magnitude. TRT's activation importer rejects beta=inf (e.g. a
Clip(0,+inf) unbounded clamp), failing the whole parse. Replacing +/-inf with
+/-65504 is numerically harmless (it's an unbounded upper clip) and lets TRT build.
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper

FP16_MAX = 65504.0


def sanitize(model):
    n_fixed = 0
    # initializers
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, posinf=FP16_MAX, neginf=-FP16_MAX, nan=0.0)
            init.CopyFrom(numpy_helper.from_array(arr, init.name)); n_fixed += 1
    # Constant node tensor attributes
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.t.ByteSize() and attr.name in ("value",):
                arr = numpy_helper.to_array(attr.t)
                if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                    arr = np.nan_to_num(arr, posinf=FP16_MAX, neginf=-FP16_MAX, nan=0.0)
                    attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name)); n_fixed += 1
    return n_fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = onnx.load(args.inp)
    n = sanitize(m)
    onnx.checker.check_model(m)
    onnx.save(m, args.out)
    print(f"[fixup] replaced inf in {n} tensors -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Make a fake-quantized student checkpoint (weights snapped to a W{bits} per-output-
channel grid) that loads via build_student + load_state_dict and runs in the closed-loop
nav harness unchanged. Used to test IMU-anchored quantization in closed loop: the harness's
per-episode scale calibration is the IMU stand-in, so a W3/W2 student navigating tests
whether the geometry-bound quantization budget (paper §4.2) holds at the robot level.
"""
import argparse, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model", "distill"))
from student import build_student          # noqa: E402


def fakequant_channel(w, bits):
    if bits >= 16:
        return w
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / qmax
    return torch.clamp(torch.round(w / scale), -qmax, qmax) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/ckpt/student_hab.pt")
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = build_student(size="base", aat_depth=8, device="cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    # snap every Linear weight to the W{bits} grid (matches the decomposition's quant set)
    lin_names = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    sd = model.state_dict()
    nq = 0
    for k in list(sd.keys()):
        mod = k[:-len(".weight")] if k.endswith(".weight") else None
        if mod in lin_names and sd[k].dim() == 2:
            sd[k] = fakequant_channel(sd[k].float(), args.bits).to(sd[k].dtype)
            nq += 1
    out = {"state_dict": sd, "size": "base", "aat_depth": 8, "wbits": args.bits}
    local = "/root/" + os.path.basename(args.out)
    for _ in range(4):
        try:
            torch.save(out, local + ".tmp"); os.replace(local + ".tmp", local); break
        except Exception as e:
            print("save retry", e, flush=True)
    print(f"=== QCKPT_DONE {local} bits={args.bits} quantized_linears={nq} ===", flush=True)


if __name__ == "__main__":
    main()

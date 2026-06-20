# MERLIN

**Metric Edge Reconstruction for Lightweight Indoor Navigation.**

A feed-forward metric 3D reconstruction foundation model, distilled and quantized to run
live on a Jetson Orin Nano 8GB as a dedicated ROS2 inference node. A single monocular RGB
stream becomes a live metric 3D map; an IMU anchors scale and bounds drift; the map feeds
Nav2 for closed-loop indoor navigation. All compute on the robot.

This README summarizes the implementation and results to date.

## Hardware

| Role | Device |
|---|---|
| Inference | Jetson Orin Nano 8GB (67 INT8 TOPS, 102 GB/s, JetPack 6.2, TRT 10.3) |
| Training | RunPod GPU node (A40 48GB class) |
| Companion | Rubik Pi 3 (QCS6490) on the rover |

## Result summary

| Claim | Target | Result | Status |
|---|---|---|---|
| C1 distill fidelity | within 10-15% of teacher | 0.21 abs_rel vs teacher (held-out) | close (see note) |
| C2 metric scale | <5% (mono+IMU) | scale-aligned shape error ~5%; raw mono scale off ~30% (IMU's job) | on track |
| C5 real-time | >=5 FPS, ~10-15 W | 6.8 FPS @378 bf16 / ~16-17 FPS TRT-INT8, 8.5 W GPU rail, 0.68 GB | MET |

### Pipeline (MapAnything backbone)
- **Teacher**: `facebook/map-anything-apache`, 1.23B params (DINOv2 ViT-G + 16-layer
  alternating-attention + DPT/pose/scale heads).
- **Student**: 230M (DINOv2 ViT-B 86.6M + 8-layer AAT 56.7M + heads 24.8M + geometric
  encoders 62.2M). Encoder from pretrained DINOv2; rest trained by output distillation.
- **Distillation (C1)**: online + precomputed-target distillation on the A40 over 10 TUM
  RGB-D sequences (9050 frames). Metric depth-along-ray (scale-invariant + log-L1), ray
  cosine, and the metric-scale factor (weighted highest, the hard-to-compress head).
  - Honest finding: an early run reported 0.16 fidelity but that was the easy start of the
    held-out trajectory; the robust strided eval was 0.34. Root cause: the fixed-image cache
    had no augmentation, so ~12 epochs memorized the frames. Fixed with geometry-correct
    augmentation (h-flip with ray-x negation + photometric jitter) + diverse data + honest
    strided eval -> **0.34 -> 0.21** held-out, train/test gap 7x -> 2.7x.
  - The residual ~21% is largely the 230M-vs-1.23B capacity gap imposed by the Nano. This
    is MERLIN's core tension and the motivation for M1.

### Deployment and quantization (C5)
Trained student on the Jetson, full resolution sweep:

| res | bf16 FPS | latency | peak mem | GPU-rail |
|---|---|---|---|---|
| 252 | 11.9 | 84 ms | 0.57 GB | 7.4 W |
| 378 | 6.8 | 147 ms | 0.68 GB | 8.5 W |
| 518 | 5.3 | 189 ms | 0.76 GB | 8.8 W |

TensorRT engines (student compute-core, 378px), built on the Nano:

| backend | GPU compute | throughput | engine | speedup |
|---|---|---|---|---|
| PyTorch bf16 (full infer) | 147 ms | 6.8 FPS | - | 1x |
| TRT FP16 core | 67.8 ms | 14.4 qps | 477 MB | 2.2x |
| TRT INT8 (--best) core | 51.5 ms | 19.1 qps | 380 MB | 2.8x |

Adding the cheap Python geometry postproc gives **~16-17 FPS @378 on a $249 8GB Nano**.
ONNX export required two fixes: monkeypatch SDPA to explicit matmul+softmax (torch 2.4
exporter mistranslates the scale attr), and sanitize inf constants (a Clip(0,+inf) fails
TRT's setBeta finite check). torchao weight-only INT8 in PyTorch is near-lossless
(abs_rel 0.001 vs bf16) but gives no latency win on sm_87 without TensorRT.

## Methods (the paper contribution)

### M1: navigation-aware budget allocation
Distill under a navigation-relevance weighting (near-field obstacle depth + drivable
frustum rows) instead of uniform reconstruction error. The misalignment thesis: the
baseline spends capacity matching the teacher everywhere, including far-field detail
control never consumes, and ends up worse in the near field than the far field.

Clean ablation (alpha=0 uniform vs alpha=1 nav-weighted, same arch/data/steps, matched
150-frame held-out eval):

| metric | baseline | M1 | delta |
|---|---|---|---|
| near_absrel (obstacles) | 0.239 | 0.221 | **-7.4%** |
| far_absrel (background) | 0.171 | 0.184 | +8% (traded) |
| col_range_mae (planner scan) | 0.208 | 0.201 | -3.2% |
| obstacle_iou (free-space) | 0.702 | 0.716 | +2.1% |

All navigation-relevant metrics improve; only far-field (control never consumes it)
degrades. The mechanism strength matters: a soft near-weight gave a null delta (-0.002);
an aggressive inverse-square near-weight gives -0.018 (7x). Dose-response: far_absrel is
monotonic in alpha (0.171 -> 0.179 -> 0.184), a dose-dependent budget trade.

### M2: deadline-elastic anytime reconstruction
The student exposes multiple operating points; a per-frame controller picks one to hold
the navigation deadline (deadline from camera-motion velocity). Two axes evaluated:
- **Resolution (252/378/518)**: the adaptive controller Pareto-dominates every fixed point
  (matches best hit-rate, beats best accuracy 2-8%), but the frontier is shallow. Found
  experimentally (incl. a multi-res-trained student) that this is intrinsic: scale-aligned
  depth error favors coarser/smoother predictions, so lower res is both faster and
  lower-error. Not a training artifact.
- **Early-exit by AAT depth (K=6 vs K=8), fixed resolution**: the axis without the
  smoothness confound. Deep-supervision trained (random K per batch). Result: K=6 absrel
  0.1372, K=8 absrel 0.1366 (clean-direction but 0.4% for +2 layers). On-device Jetson
  latency: K=6 6.78 FPS (147.5 ms) vs K=8 6.67 FPS (149.9 ms), only 2.4 ms apart -- the AAT
  is a small slice of total compute (encoder + DPT head dominate), so early-exit is flat on
  latency too. Frontier is flat on BOTH accuracy and latency, on BOTH A40 and Jetson.

**Synthesis.** Both M2 axes are flat for this student: lower resolution is faster AND
lower-error, and K=6 is ~as accurate as K=8 and faster. The distilled student SATURATES on
indoor near-field depth. So M2's deadline-elastic benefit is limited (cheap operating
points are already near-optimal), and this reinforces M1: the impactful lever is WHERE the
saturated capacity is spent (M1 nav-aware allocation), not HOW MUCH (M2 compute/resolution).
M1 is the primary methods contribution; M2 is characterized with an honest negative-ish
result that motivates M1.

### Scal3R (alternate backbone)
Scal3R (CVPR'26 Highlight, VGGT + test-time training) is a harder distillation target: its
forward is deeply coupled to the TTT pipeline (5 fixes to get a clean per-frame teacher:
DPT absolute-layer indexing, ttt_order required, TTT cache must stay non-None, update=False
for speed). Distilled the same 230M student to it (warm-started, since a cold-start fresh
student has negative depth_z that the scale-invariant loss clamp zeros out): **fid_si 0.10**
(matches Scal3R depth shape within ~10%). Shows the recipe generalizes across backbones.

## Repo layout
```
model/backbones/   teacher.py (MapAnything bf16 recipe), scal3r_teacher.py
model/distill/     student.py, distill_cache.py (aug, multi-res, multi-exit), precompute_targets.py, distill_scal3r.py
model/quant/       quantize.py (torchao INT8/INT4), export_onnx.py, onnx_fixup.py
eval/              metrics.py, nav_metrics.py, eval_nav_compare.py (M1), m2_deadline.py, m2_earlyexit.py
tools/             profile_floor.py, profile_student.py, bench_student.py, upload_hf.py
```

## Artifacts
- Code: `github.com/ShayManor/Merlin`
- Distilled student + TRT-ready ONNX: `huggingface.co/ShayManor/merlin-mapanything-student`

## Reproduce (intended)
```bash
# A40: precompute teacher targets + distill
python model/distill/precompute_targets.py --size 378
python model/distill/distill_cache.py --augment --tag baseline
# Jetson: pull student, build INT8 engine, benchmark
python tools/bench_student.py --ckpt student.pt --image test.jpg --schemes bf16,int8
```

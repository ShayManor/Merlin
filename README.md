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
| C1 distill fidelity | within 10-15% of teacher | **0.178** abs_rel vs teacher held-out (0.21 -> 0.178 via encoder-lr-mult + longer) | close; capacity-bound |
| C2 metric scale | <5% (mono+IMU) | **0.4-1.3%** via decoupled VI-scale + per-device calib (held-out); needs a production VIO front end (lightweight VO insufficient) | MET (decoupled) |
| C3 drift | <1-2% over 100 m, no backend | mechanism validated in sim: 9-axis IMU bounds drift to ~1.5-2% (= C2 scale floor) -- BUT needs a trusted yaw reference (indoor-mag risk); rover-pending | characterized |
| C4 closed-loop nav | >=80-90% success, N>=6 scenes | sim: distilled student navigates ~75% from mono, 0 collisions (5 ReplicaCAD apts); real rover pending | partial (sim) |
| C5 real-time | >=5 FPS, ~10-15 W | **16-17 FPS** TRT-INT8 @378 (6.8 bf16), ~9.8 W compute rail at STEADY STATE (sustained, not cold), 0.68 GB | MET |
| M1 nav-aware allocation | improve navigation | depth proxies improve but NAV is null/worse across two planners; perception trades success for safety | honest negative |
| M2 deadline-elastic | hold deadline, beat fixed | null: student saturates; staleness absorbed; run the cheapest op point | honest negative |

Headline: the system works (deployable mono metric-depth edge node + decoupled metric scale);
the "smart methods" (M1/M2) do not improve navigation -- the distilled student already saturates
the perception needs of reactive indoor nav, so the lever is the planner/mapping stack, not
perception fidelity. See the per-claim sections below for the evidence and honest caveats.

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
    strided eval -> **0.34 -> 0.21** held-out, train/test gap 7x -> 2.7x. A later run with a
    lower encoder LR (0.3x) + a longer 12k-step schedule further improved this to **0.178**
    (the scorecard number); the residual is the 230M-vs-1.23B capacity gap.
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

Steady-state re-validation (tegrastats measured DURING sustained inference, not cold). FPS
reproduces; power is reported per RAIL to avoid confusion (the bf16 sweep "GPU-rail" column
above is total-board VDD_IN, not the compute rail):
- **INT8 @378 (deployed)**: 20.2 qps / 49 ms; compute rail (VDD_CPU_GPU_CV) **~9.8 W**, total
  board (VDD_IN) ~18.3 W. INT8 saturates the GPU at 20 qps, so its rails sit higher than bf16's.
- **bf16 @378 (PyTorch baseline)**: 6.57 FPS; compute rail ~5.0 W, total board ~8.5 W (matches
  the table's "GPU-rail" 8.5 W = VDD_IN). bf16 sweep compute rails: 252/378/518 -> 3.2/5.0/5.9 W.
The deployed model slice (INT8 compute rail ~9.8 W) is within the 10-15 W target; energy/inference
favors INT8 (9.8 W / 20 qps = 0.49 J vs bf16 5.0 W / 6.57 = 0.76 J). All steady-state, not cold.

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

These depth PROXIES improve under M1. But a navigation-safety eval (turn each depth into
the local planner's forward-corridor go/stop decision, n=400, vs GT) tells a different and
more honest story:

| variant | collision rate (global scale) | collision (near/IMU scale) | corridor range-MAE |
|---|---|---|---|
| baseline (uniform) | 0.100 | 0.135 | 0.106 |
| M1 (mean near-weight) | 0.143 | 0.152 | 0.113 |
| M1-v2 (per-column-min loss) | 0.109 | 0.144 | 0.105 |

M1 is WORSE on collisions under both scale assumptions. Diagnosis: collision avoidance
depends on the nearest-obstacle RANGE (per-column minimum), but M1 optimizes the MEAN
near-field depth; improving the mean worsens the min. M1-v2 targets the min directly and
recovers to baseline (and marginally improves range-MAE) but does NOT beat uniform.

Honest conclusion: neither task-aware variant beats uniform distillation on navigation
safety. The compact student SATURATES (see M2), so uniform distillation already sits near
the navigation-relevant ceiling; reallocating the objective hurts (M1) or ties (M1-v2). The
real lever is not the distillation objective. A closed-loop navigation sim is the next step
to confirm this dynamically; the depth-proxy "wins" do not transfer.

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

**Dynamic test (latency-coupled staleness sim).** The proper M2 test replays a trajectory
where the planner acts on perception delayed by the op-point latency (slow op point -> stale
map). Result REFUTES M2: the slowest/most-accurate op point (518) gives the fewest collisions
at every agent speed (x1: 0.097, x8: 0.127), beating the adaptive controller (x8: 0.187).
Indoor latency-staleness (<=0.55m even at ~2.9 m/s) is negligible vs the accuracy gain, so
trading accuracy for freshness backfires. Correct policy: run the most accurate model that
fits a generous deadline, not adaptive switching.

**Synthesis (both methods negative).** The distilled student saturates; uniform distillation
sits near the navigation ceiling. M1 (reallocate the objective) does not beat uniform on
collisions; M2 (anytime elasticity) does not beat the fixed best op point. The honest
contribution here is a SYSTEM + a careful characterization: a deployable INT8 metric-recon
node at 16-17 FPS on an 8GB Nano, plus the finding that task-aware distillation objectives
and anytime elasticity do not improve indoor navigation for a saturated compact student
(with the mechanisms: saturation, the mean-vs-nearest-range objective trap, and negligible
indoor staleness). This was confirmed at scale in closed loop (next section).

### Closed-loop validation (Habitat) — the load-bearing result
A latency-coupled photorealistic sim (Habitat-sim, ReplicaCAD apartments; reactive
VFH/proportional-braking planner; sim time advanced by the measured Jetson per-op-point
latency so a slow map goes stale) tested M1 and M2 in actual navigation. Code in `eval/sim/`.
- **System works (positive, first end-to-end MERLIN demo):** the distilled 230M student drives
  the planner to ~75% goal success with zero collisions from a single mono RGB stream across
  5 apartments.
- **M1 null (well-powered):** nav-aware vs uniform, n=119 paired episodes, success delta
  -0.009 [-0.06, +0.04] — statistically indistinguishable on success / SPL / collisions.
- **M2 null:** success identical to three decimals (0.874) across fixed_252/378/518/adaptive;
  staleness is real and ordered (up to ~83 mm at 2 m/s) but absorbed by proportional braking.
- **Headline (counterintuitive):** a perfect-depth ORACLE navigates **44 points worse** than
  the student (0.32 vs 0.75 success). Navigation here is **planner-bound, not
  perception-fidelity-bound** — a reactive planner brakes on any perceived obstacle, so sharper
  depth (M1) or fresher maps (M2) or even an oracle do not help, and over-precise depth stalls
  in clutter. The distilled student already CLEARS the perception bar for reactive indoor nav.
- **SWaP-C payoff:** since resolution does not buy success, success-per-watt is maximized by the
  CHEAPEST op point — **run fixed_252 always** (~2.7x less energy/m, same success). The honest
  paper framing: how much perception fidelity does reactive indoor nav need? The student already
  clears it; the lever is the planner/mapping stack. (Caveats: per-episode GT scale align as IMU
  stand-in.)
- **Robust to a smarter planner (addresses the obvious objection):** a second planner -- a GLOBAL
  occupancy-mapping + A* planner that plans on the PERCEIVED depth map (no navmesh guide,
  `eval/sim/map_planner.py`) -- gives the SAME verdict. 3 apartments, 45 paired episodes: student
  baseline 0.705, M1 0.667 (CIs overlap -> M1 still null), perfect-depth oracle 0.378 (still
  worst). So neither finding is a reactive-planner artifact: M1 is null and perfect depth HURTS
  under both a reactive and a map-building planner. Mechanism: perfect depth over-clutters the
  occupancy grid with fine/thin obstacles and stalls A*; the student's smoother depth yields a
  cleaner, more navigable map (0 collisions everywhere -- the navmesh handles physical avoidance).
  Robust to the occupancy threshold: a stricter grid (3 hits to mark a cell occupied vs 1) helps
  the oracle only modestly (0.378 -> 0.433) and the student still wins (0.700), so over-cluttering
  is part of the mechanism but not all of it -- perfect depth is genuinely worse, not a single
  threshold artifact.
- **No-slide / real collisions -- a SUCCESS/SAFETY TRADEOFF (the key refinement):** the
  "collisions rare by construction" caveat (the navmesh slides the robot along obstacles) was
  probed with a no-slide mode (`try_step_no_sliding`, robot stops at obstacles). With the MAPPING
  planner it floors (no stuck-recovery -> all arms incl. the oracle near 0). With the REACTIVE
  planner + navmesh guide (n=214) it is informative and reveals a tradeoff: the student is
  SUCCESSFUL but UNSAFE (success 0.79, 1.41 collisions/m -- it under-sees and barges through),
  while perfect depth is SAFE but over-cautious (success 0.39, 0.00 collisions/m); M1 does not
  improve either axis (0.79 / 1.44). So "perception fidelity does not matter" is incomplete:
  under real collisions, fidelity TRADES success for safety -- perfect depth is far safer, just
  less successful, and the distilled student's "smoother is better" advantage is really a
  success-for-safety trade. The deployment implication (cheap op point, smoother depth) holds
  only where the navmesh/controller guarantees physical safety; a safety-critical deployment
  wants the fidelity (or a planner that brakes earlier). M1's specific allocation still doesn't
  help.

### Architecture finding: decouple pose from depth
A visual-inertial metric-scale estimator (TUM accelerometer + the student's own camera poses)
was built and tested. It fails: the student's pose output is too noisy (~22% VO translation
error; rotations too noisy to bring the accelerometer into a consistent gravity frame, so a
fixed extrinsic cannot be calibrated). The compact distilled student is an excellent DENSE
METRIC DEPTH model but a poor EGO-MOTION estimator. Implication: the "one model does pose +
depth + scale" premise is not supported; pose/scale/drift should come from a decoupled front
end (classical VO or tight IMU integration), with the student serving as the dense-depth node.

This is validated, not just hypothesized: re-running the SAME linear VI scale solver with
GT-quality poses (stand-in for a classical VO front end) recovers metric scale to **5.5-12.6%
across four TUM sequences** (median ~6-8%), with clean gravity calibration (|g| 9.69-9.79).
The residual is a consistent underestimate (characteristic MEMS dynamic-motion attenuation).
A one-time per-device accelerometer scale-factor calibration (k fit on one sequence) closes
it on HELD-OUT sequences: calibrating on freiburg1_xyz and applying to freiburg1_room and
_desk gives **1.3% and 0.4%** scale error (freiburg2_desk, own-device, 0.8%). **C2 (<5%
metric scale) is MET** through the DECOUPLED design (student dense depth + VO/IMU pose+scale),
not the single-model premise. Robustness to a realistic VO front end: injecting rotation noise, C2 scale error stays under
5% up to ~3 deg of rotation error (0 deg 1.3%, 1 deg 1.6%, 2 deg 2.4%, 5 deg 5.4%) -- and
mono VO (ORB-SLAM-grade) has well under 1-2 deg relative-rotation error over short windows.
So the module works with a deployable VO front end, not just perfect rotations. This is the
validated positive contribution alongside the deployable dense-depth node.

Caveat (keeps the claim honest): a LIGHTWEIGHT VO is not enough. A naive OpenCV essential-
matrix VO gives 19-64 deg relative-rotation error on these sequences (small baseline +
appearance change), far above the ~3 deg the scale module tolerates. So C2 requires a
PRODUCTION VIO front end (ORB-SLAM3 / VINS with bundle adjustment and map tracking, which
routinely achieve <0.5 deg) -- a standard but real integration. End-to-end validation with
ORB-SLAM3 rotations is the concrete next engineering step; the module + robustness results
establish that it will hold given that front end.

### C3: does the IMU bound drift without a global backend? (mechanism validated in sim)
The decoupled odometry integrates relative poses, so error accumulates; the C3 claim is the
IMU bounds it. A CPU-only characterization on TUM (`eval/c3_drift.py`): simulate the VI
odometry (rotation random-walk at rate sigma + the C2 scale error) and measure ATE vs path
length with vs without a 9-axis MPU-9250 keeping orientation bounded (accel -> roll/pitch,
mag -> yaw). A full 9-axis IMU bounds position drift to a FLAT ~1.5-2% floor regardless of the
VO rotation-drift rate (sigma 1/3 deg/keyframe -> 9-axis 1.5-2.0%, no-IMU 1.9-4.0%); that
~1.5-2% floor IS the C2 scale error. BUT the critical caveat: the magnetometer is doing the
work. A gyro+accel-only variant (realistic INDOOR, where the mag is unreliable -- steel,
motors distort it) does NOT bound drift to the floor: yaw drifts and ATE grows to ~2.3-4.1%
(at sigma=3), sometimes as bad as no-IMU. Because indoor motion is mostly horizontal, YAW
drift dominates (a yaw error rotates the whole trajectory off), so bounding roll/pitch (accel)
helps little -- you need a trusted YAW reference (mag, loop closure, or visual heading
consistency) for the bounded-drift claim. So C3 is supported by the mechanism ONLY with a
reliable yaw reference; the indoor-magnetometer risk is real. Constraint-compatible fix (no
mag, no global backend): a Manhattan-world / vanishing-point heading from the visual input.
Validated CPU-only on TUM -- the dominant horizontal line direction has std ~7 deg (mean 0.6,
no accumulation), so an indoor VP heading bounds yaw without a magnetometer (a proper
VP+RANSAC estimator would tighten the ~7 deg). This is the recommended indoor yaw reference.
Caveat: simulation (GT
trajectory + synthetic sensor noise + synthetic VO drift, short ~15 m paths); real-rover
validation (real sensor noise, real VO/mag, 100 m+) is the C3/C4 hardware step.

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

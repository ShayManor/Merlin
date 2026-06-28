# MERLIN

**Metric Edge Reconstruction for Lightweight Indoor Navigation** — a 1.23B metric-3D
foundation model distilled to 230M and quantized to run live on a $249 Jetson Orin Nano 8GB,
turning one monocular RGB stream into metric depth + pose for closed-loop indoor navigation.

- Model + ONNX: [huggingface.co/ShayManor/merlin-mapanything-student](https://huggingface.co/ShayManor/merlin-mapanything-student)

## Core findings

| | Result |
|---|---|
| **Deploys** | 16-17 FPS TRT-INT8 @378px, ~9.8 W, 0.68 GB on an 8GB Nano. |
| **Fidelity** | 0.178 abs_rel vs the teacher; the residual is the 230M-vs-1.23B capacity gap. |
| **Metric scale** | 0.4-1.3% error via a *decoupled* VI front end (the student is a poor ego-motion estimator). |
| **Drift** | A 9-axis IMU bounds drift to ~1.5-2%, but indoors needs a visual yaw reference, not the magnetometer. |
| **Navigation** | Reactive indoor nav is planner-bound, not perception-bound: the student saturates the perception bar (a perfect-depth oracle navigates *worse*), so nav-aware distillation and anytime elasticity both buy nothing — run the cheapest op point. |

## Run
```bash
# A40: precompute teacher targets + distill the student
python model/distill/precompute_targets.py --size 378
python model/distill/distill_cache.py --augment --tag baseline
# Jetson: pull student, build INT8 engine, benchmark
python tools/bench_student.py --ckpt student.pt --image test.jpg --schemes bf16,int8
```

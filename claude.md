# MERLIN

**Metric Edge Reconstruction for Lightweight Indoor Navigation.**

Context for agentic coding tools. Read this before working in the repo. If a change would violate anything under "Hard constraints," stop and flag it instead of working around it.

## What this is

A feed-forward metric 3D reconstruction foundation model, distilled and quantized to run live on a **Jetson Orin Nano 8GB** as a dedicated ROS2 inference node. It turns a single monocular RGB stream into a live metric 3D map, with an IMU anchoring scale and bounding drift, and feeds that map to Nav2 for closed-loop autonomous indoor navigation. All compute is on the robot. No pre-built map, no wheel encoders, no base station.

The bet: existing 3D-foundation reconstruction is either accurate-but-undeployable (datacenter GPUs) or deployable-but-impoverished (classical mono SLAM). MERLIN is the first point that is metric, dense, and deployable. The only comparable real-time deployment (VGGT-SLAM 2.0) needs a Jetson AGX Thor, ~14x the cost and ~5-13x the power of the Nano.

## Status

Research project, pre-implementation. Currently at **Phase 0** (profiling the floor on real hardware). Nothing in "Repo layout" or "Build and run" is built yet; those describe the intended structure. Target venue ICRA 2027 (deadline ~Sep 15 2026). Fallback IROS 2027 (~Mar 2027) if closed-loop autonomy slips.

## Hardware (fixed)

| Component | Spec |
|---|---|
| Inference compute | Jetson Orin Nano 8GB. 67 INT8 TOPS, 102 GB/s, shared LPDDR5. Runs the model node only. |
| Companion compute | Rubik Pi 3 (QCS6490), on the rover. Runs everything except the model. |
| Chassis | WAVE ROVER skid-steer, 18650 pack. No wheel encoders. |
| Camera | Monocular USB webcam (global shutter preferred). |
| IMU | MPU-9250 class. |
| Middleware | ROS2 Jazzy, CycloneDDS, distributed across Jetson + Rubik Pi. |

## System architecture (the core design rule)

The Jetson runs **one node: the model**. Headless, no display, minimal OS footprint. This dedicates ~6.5 GB of the 8, plus the full GPU and thermal budget, to inference. This is the single decision that makes 8 GB viable. Do not add other nodes to the Jetson.

Everything else runs on the Rubik Pi companion: camera driver (or it streams to the Jetson), IMU fusion, scale and drift correction, mapping, Nav2, control.

**Intended node split (open, but this is the working plan):**
- Jetson model node: image in, metric depth + relative pose out. Pure inference.
- Rubik Pi: IMU-fused scale and drift correction, map assembly, Nav2, control.

**What crosses the wire (important):** never publish raw dense point clouds over DDS. A 640x480 depth lifts to ~300k points, multiple MB/frame, and saturates the link. Publish compact depth + pose, or occupancy deltas. Dense map at keyframe rate; pose on a separate high-rate channel so the planner always has fresh state between keyframes. Tune DDS QoS (best-effort for high-rate pose, reliable for keyframes). Clock sync across the two boards matters for closed-loop.

## The model

- **Primary backbone:** MapAnything (arXiv:2509.13414, Apache-2.0). Metric, monocular-capable, modular.
- **Alternate:** Scal3R (arXiv:2604.08542), if the TTT-state + pose-estimation angle is wanted.
- **Compression:** distill to a small student (~100-300M params), then INT8 with INT4 on the heavy layers, exported through TensorRT. The metric / global-state head is the part that is hardest to compress without losing metric accuracy; that head is the focus of the distillation recipe and is the transferable method contribution.

## Hard constraints (do not violate)

- Model node fits in ~6.5 GB and runs **model-only** on the Jetson.
- Keyframe-rate dense mapping **>=5 FPS** at usable quality; model slice **~10-15 W**.
- Metric scale is recovered (mono + IMU). Relative-only output is a failure.
- Drift bounded over 100 m+ traversals **without** a global bundle-adjustment backend and **without** wheel encoders.
- No raw dense clouds on the wire.
- "No external compute" means all compute is physically on the rover. The companion is the Rubik Pi, not a laptop. Do not introduce a base-station dependency.

## Repo layout (proposed, not yet built)

```
merlin/
  ros2_ws/src/
    merlin_model/        # Jetson model node (inference only)
    merlin_bringup/      # launch files, DDS/QoS profiles, multi-machine config
    merlin_map/          # Rubik Pi: scale+drift fusion, map assembly
    merlin_nav/          # Nav2 params, costmap from MERLIN map
  model/
    distill/             # student training, teacher = full backbone
    quant/               # INT8/INT4, TensorRT engine build
    backbones/           # MapAnything / Scal3R adapters
  eval/
    geometry/            # ScanNet++/TUM/7-Scenes/ETH3D metrics
    nav/                 # closed-loop success/SPL/collision
    swapc/               # power, latency, FPS, energy/meter, $ benchmark
  tools/
    profile_floor.py     # Phase 0 harness: per-keyframe latency on the Nano
```

## Build and run (intended workflow)

```bash
# Workspace (per machine)
cd ros2_ws && colcon build && source install/setup.bash

# Build the TensorRT engine on the Jetson (not on x86; engines are device-specific)
python model/quant/build_engine.py --backbone mapanything --precision int8

# Phase 0: floor profiling, model-only, headless, on the real Nano
python tools/profile_floor.py --engine merlin_int8.engine --res 384 --report

# Run the model node on the Jetson
ros2 run merlin_model node --engine merlin_int8.engine

# Run map + nav on the Rubik Pi
ros2 launch merlin_bringup companion.launch.py
```

Set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and a shared `ROS_DOMAIN_ID` on both boards. Keep a CycloneDDS XML profile in `merlin_bringup` for the interface and QoS.

## Evaluation targets (claims to validate)

- **C1 (Pareto, headline):** within ~10-15% of full-precision backbone accuracy at ~14x lower cost and ~5-13x lower power than the Thor deployment.
- **C2 (metric):** absolute scale error <5% from mono + IMU.
- **C3 (drift):** ATE <1-2% of path length over 100 m+, no encoders, no global backend.
- **C4 (closed-loop):** nav success >=80-90% in N>=6 unseen indoor scenes from the live onboard map only.
- **C5 (real-time):** keyframe-rate >=5 FPS, model slice ~10-15 W. Crux risk, settled by Phase 0.

Datasets: ScanNet++ v2, TUM RGB-D, 7-Scenes, ETH3D (offline geometry); own rover logs and live runs (closed-loop). Baselines: full-precision backbone on workstation (ceiling), VGGT-SLAM 2.0 on Thor (deployed real-time), ORB-SLAM3 mono and RTAB-Map mono (classical), MASt3R-SLAM (learned reference).

## Known gotchas

- flash-attn and LaCT custom kernels are painful on Jetson aarch64 / SM87. Budget time, or replace with Jetson-compatible attention.
- The Nano is bandwidth-bound (102 GB/s), so per-keyframe latency is dominated by memory traffic, not FLOPs. Quantization and small activations matter more than raw TOPS.
- TensorRT engines are device-specific. Build on the Nano, not on a dev x86 box.
- Thermal: sustained inference on the Nano will throttle without a fan and MAXN/Super mode. Profile at steady state, not cold.
- Scale ambiguity is the silent failure mode. Verify metric output against a known baseline early.

## Standing instructions for the agent

- **Phase 0 before features.** The whole project hinges on the floor number. Do not build downstream nodes until profiling shows >=5 FPS keyframe-rate at usable quality. If it fails: smaller student, lower resolution, INT4, before anything else.
- Keep the Jetson model node lean. If a task tempts you to add a second node there, it belongs on the companion.
- Never publish raw dense clouds.
- Match the existing code style. Surgical edits over rewrites; show what changes and where. Concise, technically precise. No em dashes in prose or comments.
- When citing a paper, use a verified arXiv ID. Do not invent references.

## Positioning (so you understand the framing)

The TTT-linearization method (tttLRM, ZipMap, TTT3R, VGG-T3, Scal3R) is saturated; do not pitch "we made it efficient via TTT." VGGT-family edge SLAM (VGGT-SLAM 2.0, Flash-Mono, LeanGate) is saturated; do not pitch "first edge SLAM." MERLIN's defensible contributions are the metric-head compression recipe, the IMU scale+drift module, the SWaP-C benchmark, and the real-world testbed data. Frame any competing "foundation-model-on-robot" preprint as a baseline to beat, not a scoop.

Networking:
 Jetson ↔ Rubik ping working over the ethernet cable

  ┌───────────────┬──────────────────────────────┬───────────────────────────────┐
  │               │   Jetson (evc@100.93.187.32) │   Rubik Pi evc5@100.93.187.32 │
  ├───────────────┼──────────────────────────────┼───────────────────────────────┤
  │ SSH           │ evc@100.93.187.32 (host evc) │ evc@evc5 (host ubuntu)        │
  ├───────────────┼──────────────────────────────┼───────────────────────────────┤
  │ Eth interface │ enP8p1s0                     │ enxf074e47f5013 (USB adapter) │
  ├───────────────┼──────────────────────────────┼───────────────────────────────┤
  │ Link IP       │ 192.168.50.1/24              │ 192.168.50.2/24               │
  ├───────────────┼──────────────────────────────┼───────────────────────────────┤
  │ NM profile    │ ros-eth                      │ enxf074e47f5013               │
  └───────────────┴──────────────────────────────┴───────────────────────────────┘
  Password is 123456 for both
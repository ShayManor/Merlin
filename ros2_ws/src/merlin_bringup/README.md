# merlin_bringup

Bringup + DDS/QoS + the **C4 closed-loop nav** runbook. This is the turnkey path from "rover assembled"
to "C4 numbers in the paper". The node split is fixed (see top-level `CLAUDE.md`):

- **Jetson Orin Nano** — runs ONE node: the model. Image in → metric depth + relative pose out. Headless.
- **Rubik Pi 3 (companion)** — IMU fusion (scale C2 / drift C3), map assembly, Nav2, control.

All compute is on the rover. **Never publish raw dense point clouds over the link** — publish compact
depth + pose, or occupancy deltas (a 640×480 depth lifts to ~300k points / multiple MB per frame and
saturates the link).

## 1. Network + DDS (both boards)

Wired link (from `CLAUDE.md`): Jetson `enP8p1s0` `192.168.50.1/24` ↔ Rubik `enxf074e47f5013` `192.168.50.2/24`.

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=17                                  # same on both boards
export CYCLONEDDS_URI=file://$PWD/config/cyclonedds.xml
# pick the right interface per board:
export CYCLONEDDS_IFACE=enP8p1s0        # on the Jetson
# export CYCLONEDDS_IFACE=enxf074e47f5013   # on the Rubik Pi
```

Sanity: `ros2 multicast receive` is not used (point-to-point, peers are static in `cyclonedds.xml`);
verify cross-board discovery with `ros2 topic list` from each side after launch.

## 2. Topics + QoS (the wire contract)

| Topic | Producer | Rate | QoS | Notes |
|---|---|---|---|---|
| `/merlin/pose` | Jetson | high (per-frame) | **best-effort**, depth 5 | rel pose; planner always has fresh state |
| `/merlin/depth_keyframe` | Jetson | keyframe (~5 FPS) | **reliable**, depth 2 | compact metric depth, NOT raw cloud |
| `/merlin/scale` | Rubik (C2) | on update | reliable, transient_local | IMU-recovered metric scale |
| `/map` / costmap deltas | Rubik | keyframe | reliable | assembled occupancy for Nav2 |
| `/cmd_vel` | Rubik (Nav2) | control | reliable | skid-steer command |

Clock sync across the two boards matters for closed-loop; run chrony/PTP or `ros2 run` a shared time
source before episodes.

## 3. Run (intended)

```bash
# per machine, once
cd ros2_ws && colcon build && source install/setup.bash

# Jetson: the model node (engine built on-device; see model/quant/build_engine.py)
ros2 run merlin_model node --engine merlin_int8.engine          # [node TODO: not yet built]

# Rubik Pi: companion (map + C2/C3 fusion + Nav2 + control)
ros2 launch merlin_bringup companion.launch.py                  # [launch TODO: not yet built]
```

The `merlin_model` node and `companion.launch.py` are **not yet implemented** — they are the
remaining systems engineering for the rover. The DDS profile (`config/cyclonedds.xml`), the QoS
contract above, and the C4 evaluation below are ready now.

## 4. C4 closed-loop evaluation (ready now)

During each navigation goal, the companion (or a rosbag post-processor) writes ONE episode JSON per
goal into a run directory, in the schema documented at the top of `eval/nav/c4_eval.py`:

```json
{ "scene": "lab_room_3", "start": [0,0], "goal": [4.0,2.0], "shortest_path_m": 4.8,
  "trajectory": [[t,x,y,yaw], ...], "collisions": [[t,x,y], ...] }
```

Then compute the paper's C4 metrics (success rate, SPL, soft-SPL, collisions/m, path efficiency,
per-scene):

```bash
python3 eval/nav/c4_eval.py --episodes runs/*.json --out results/nav/c4.json
# verify the harness logic with no hardware:
python3 eval/nav/c4_eval.py --self-test
```

Target (C4): nav success ≥80–90% across N≥6 unseen indoor scenes from the live onboard map only.
The aggregate JSON drops straight into the paper's pending C4 row (§4 / §7).

## What is ready vs pending

- **Ready now:** DDS/QoS profile, the wire contract, the C4 metrics harness (`eval/nav/c4_eval.py`,
  self-test passes), the episode logging schema.
- **Pending (rover systems work):** the `merlin_model` ROS2 node wrapper around the INT8 engine, the
  `companion.launch.py` (map assembly + C2/C3 fusion + Nav2 params + control), and the episode logger.

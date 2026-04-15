# Physical AI Challenge 2026

## SO101 Autonomous Pick-and-Place: Complete Process and Training Guide

This repository now serves as a full execution reference for the workflow you followed in `Context/Plan-1.md`:

- **Core challenge stack:** SO101 + ROS 2 Jazzy + Gazebo Harmonic + MoveIt 2 (Dockerized).
- **Learning and policy stack:** MuJoCo + YOLOv8 + ACT + DenseFusion prototype in `macos_pipeline/`.
- **Strategy:** simulation-first baseline, then robustness, then learning policy integration.

This README is intentionally deep and operational. It explains the entire pick-and-place process, policy layers, training paths, and how each part maps to files in this repo.

---

## 1) Current Project State (What Is Already Here)

### Implemented and runnable now

1. **SO101 simulation bring-up**
   - Launches Gazebo + ros2_control + MoveIt + RViz using:
   - `ros2 launch so101_moveit simulated_robot.launch.py`

2. **Planning-scene setup utilities**
   - `add_scene_objects`: adds ground, pick table, place table, and three cubes.
   - `insert_obstacle`: adds/removes dynamic cylindrical obstacles.

3. **Autonomous simulation loop (MuJoCo path)**
   - `macos_pipeline/run_autonomous.py` executes detect -> plan -> execute -> verify cycles.
   - `macos_pipeline/autonomous_runner.py` provides structured multi-cycle autonomous control.

4. **Perception and policy modules (training-ready interfaces)**
   - YOLO detector wrapper with analytic fallback (`macos_pipeline/perception/yolo_cube_detector.py`).
   - ACT policy wrapper with stub and model mode (`macos_pipeline/policy/act_policy.py`).

5. **Data and training tools**
   - Episode data collection and conversion to YOLO format (`macos_pipeline/collect_training_data.py`).
   - Synthetic dataset generation from MuJoCo (`macos_pipeline/mujoco_demo/generate_dataset.py`).
   - DenseFusion-style prototype training/inference script (`densefusion_pick_place.py`).

### Important design note

The **SO101 ROS stack** is the challenge-faithful environment. The **macOS pipeline** is a practical learning/policy sandbox for fast iteration on perception and imitation policy ideas.

---

## 2) System Architecture (From Plan to Execution)

### High-level data flow (target architecture)

1. RGB-D sensing provides image + depth + intrinsics.
2. Perception detects object(s), estimates 3D pose.
3. Policy selects the next target and grasp/place strategy.
4. MoveIt plans collision-aware trajectories.
5. ros2_control executes joint/gripper commands.
6. Verification checks success/failure and loops.

### What this repository currently uses

- **SO101 stack:** robust simulation + planning + scene management.
- **Policy/perception experimentation:** MuJoCo loop with pluggable YOLO and ACT.
- **Training interfaces:** data collection, YOLO train path, ACT model-loading path, DenseFusion prototype.

---

## 3) Repository Map (Operational)

### Root

- `docker-compose.yml` and `docker-compose.windows.yml`: container startup profiles.
- `Dockerfile`: build recipe for ROS Jazzy + MoveIt + ros_gz stack.
- `runbook.md`: detailed canonical runtime instructions.
- `Context/Plan-1.md`: canonical architecture/research plan you followed.

### SO101 ROS2 workspace

- `so101_ws/src/so101_description/`: URDF/Xacro, Gazebo world, ROS-GZ bridge config.
- `so101_ws/src/so101_controller/`: ros2_control controller configuration and launch.
- `so101_ws/src/so101_moveit/`: MoveIt launch + planning-scene utility scripts.
- `so101_ws/src/so101_moveit/config/so101_robot.srdf`: semantic model (planning groups, end-effector).

### Learning / policy pipeline

- `macos_pipeline/perception/`: cube detection and synthetic-data helpers.
- `macos_pipeline/policy/`: ACT policy wrapper and stubs.
- `macos_pipeline/mujoco_demo/`: local simulation demo, recording, dataset generation.
- `macos_pipeline/collect_training_data.py`: rollout data capture for model training.
- `densefusion_pick_place.py`: DenseFusion-style prototype model/training script.

---

## 4) SO101 Challenge Stack: Start and Run (Docker)

## Prerequisites

1. Git
2. Docker Desktop or Docker Engine
3. GUI forwarding support:
   - Linux / Windows 11 WSL2: X11 forwarding via host display.
   - Windows 10: VcXsrv recommended.

## Bring up container

```bash
git clone https://github.com/vishal-finch/physical-ai-challenge-2026.git
cd physical-ai-challenge-2026
```

Linux / WSL2:

```bash
xhost +local:root
docker compose up -d
```

Windows 10 with VcXsrv:

```bash
docker compose -f docker-compose.windows.yml up -d
```

Enter container:

```bash
docker exec -it so101_hackathon bash
```

Inside container, launch full stack:

```bash
source /opt/ros/jazzy/setup.bash
source /so101_ws/install/setup.bash

ros2 launch so101_moveit simulated_robot.launch.py
```

In a second container terminal, add scene:

```bash
docker exec -it so101_hackathon bash
source /opt/ros/jazzy/setup.bash
source /so101_ws/install/setup.bash

ros2 run so101_moveit add_scene_objects
```

Optional dynamic obstacle test:

```bash
ros2 run so101_moveit insert_obstacle --x 0.3 --y -0.2 --z 0.5 --radius 0.04 --height 0.25
ros2 run so101_moveit insert_obstacle --name obstacle --remove
```

---

## 5) Deep Pick-and-Place Process (Step by Step)

This section captures the exact process logic from Plan-1, adapted to this codebase.

### Step 0: World and robot initialization

1. Launch simulation and MoveIt.
2. Spawn scene objects (tables + cubes) into planning scene.
3. Verify joint states and controllers are alive.

Recommended checks:

```bash
ros2 topic echo /joint_states --once
ros2 control list_controllers
```

### Step 1: Perception input

Target signal chain:

- RGB frame
- Depth frame
- Camera intrinsics
- TF chain camera -> base_link

Current status in this repo:

- The SO101 environment and bridge are ready.
- Camera perception node integration is part of the next layer to attach (as described in `Context/Plan-1.md`).
- Policy/perception experimentation is actively implemented in `macos_pipeline/`.

### Step 2: 2D detection to 3D grasp point

Core projection equations (camera model):

- X = (u - cx) * d / fx
- Y = (v - cy) * d / fy
- Z = d

Then transform from camera frame to `base_link` and generate grasp pose.

Best practice followed in Plan-1:

1. Use robust depth from a center ROI median (not a single depth pixel).
2. Reject invalid depth (zero/noisy ranges).
3. Keep grasp approach top-down for baseline stability.

### Step 3: Policy decision

Four policy levels (same as Plan-1):

1. Rule-based state machine (fastest baseline)
2. Rule-based + improved grasp scoring
3. ACT behavior cloning
4. Diffusion/VLA advanced policies

In this repository:

- `macos_pipeline/policy/act_policy.py` supports:
  - `mode='stub'` for deterministic baseline chunks
  - `mode='model'` for trained checkpoint loading

### Step 4: Motion planning

Use MoveIt for:

1. Pre-grasp move (collision aware)
2. Transfer path (collision aware)
3. Place path

Use Cartesian descent/lift for:

1. Final approach to object
2. Initial lift after grasp

### Step 5: Grasp/attach and transport

Operational sequence:

1. Move to pre-grasp above object.
2. Descend to grasp height.
3. Close gripper (or simulated attach in MuJoCo path).
4. Lift vertically.
5. Move to place target.
6. Descend and release.
7. Return to ready/home.

### Step 6: Verify outcome and recover

For each cycle:

1. Check object distance to target threshold.
2. Mark success/failure.
3. If failed: retry with adjusted approach, confidence threshold, or replan.

The autonomous scripts already compute cycle success and aggregate stats.

---

## 6) Policy and Training Workflows

## A) Autonomous run loop (MuJoCo policy path)

```bash
cd macos_pipeline

# Optional environment setup
./setup_and_run.sh

# Sanity test
python test_autonomous.py

# Run autonomous cycles (stub policy fallback available)
python run_autonomous.py --cycles 10
```

Use trained checkpoints when available:

```bash
python run_autonomous.py \
  --cycles 20 \
  --yolo-model models/yolov8_cube.pt \
  --act-model models/act_policy \
  --headless
```

## B) Collect training data from autonomous rollouts

```bash
cd macos_pipeline
python collect_training_data.py --episodes 200 --output datasets/collected --to-yolo
```

Outputs:

- `datasets/collected/images/`
- `datasets/collected/annotations/`
- `datasets/collected/summary.json`
- optional `datasets/yolo_dataset/`

## C) Train YOLOv8 detector

Generate synthetic set from MuJoCo (alternative path):

```bash
cd macos_pipeline/mujoco_demo
python generate_dataset.py --num 1000 --out ../datasets/synthetic_cube
```

Train using detector helper:

```bash
cd ../
python perception/yolo_cube_detector.py --train datasets/synthetic_cube/data.yaml --epochs 100
```

## D) Train ACT policy

Current code supports model loading and inference API integration. For full ACT training, use LeRobot or your ACT training framework and then point `--act-model` to your exported checkpoint directory.

Typical LeRobot flow (from Plan-1 approach):

```bash
pip install lerobot

# Record demonstrations
python -m lerobot.record --repo-id my_so101_pick_place --num-episodes 100

# Train ACT
python -m lerobot.train \
  --policy-path lerobot/configs/policy/act.yaml \
  --dataset-repo-id my_so101_pick_place \
  --output-dir outputs/train/so101_act
```

Then run this repository in `mode='model'` by passing:

- `--act-model outputs/train/so101_act/...`

## E) DenseFusion prototype training

```bash
# Train
python densefusion_pick_place.py --mode train --epochs 20 --num_samples 2000 --checkpoint_dir ./checkpoints

# Inference demo
python densefusion_pick_place.py --mode infer --checkpoint_dir ./checkpoints
```

Note: this is a lightweight prototype implementation for experimentation and pipeline integration, not the original full DenseFusion benchmark code.

---

## 7) Evaluation Gates (Aligned with Plan-1)

| Phase | Target metric | Pass gate |
|---|---|---|
| Baseline | Pick-place success | >= 70% (7/10) |
| Robustness | Generalized positions/obstacles | >= 85% |
| Learning (ACT) | Match or exceed robust baseline | >= baseline |
| Advanced | Diffusion/VLA stretch | optional research target |

Track minimum metrics every run:

1. Success rate
2. Cycle time
3. Planning failures
4. Perception confidence and miss rate
5. Collision/near-collision events

---

## 8) Critical Frame and Planning Notes

1. In `add_scene_objects.py`, objects are published in `base_link` frame.
2. In `insert_obstacle.py`, obstacles are published in `world` frame.
3. Do not mix these frames without TF conversion.
4. Use semantic planning group exactly as configured in SRDF (`so101_arm` in this repo).

---

## 9) Troubleshooting

## Container does not start

```bash
docker compose logs --tail=200
docker ps -a
```

## GUI does not show (Gazebo/RViz)

Linux/WSL2:

```bash
xhost +local:root
echo $DISPLAY
```

Windows 10:

- Start VcXsrv with access control disabled.
- Use `docker-compose.windows.yml`.

## ROS environment not sourced

```bash
source /opt/ros/jazzy/setup.bash
source /so101_ws/install/setup.bash
```

## MuJoCo model not found in macOS pipeline

Ensure SO101 XML exists at:

- `macos_pipeline/third_party/SO-ARM100/Simulation/SO101/so101_new_calib.xml`

## Missing Python packages in macOS pipeline

```bash
cd macos_pipeline
python -m pip install -r requirements.txt
python run_pipeline.py --test-imports
```

---

## 10) Sim2Real Readiness Checklist

Before moving to real SO101:

1. Camera intrinsics/extrinsics calibrated
2. TF chain validated end-to-end
3. Workspace limits clamped
4. Speed/acceleration reduced for first trials
5. Collision scene matches physical setup
6. E-stop and supervision ready

---

## 11) Recommended Execution Order (Practical)

1. Start SO101 docker stack and verify stable simulation.
2. Validate planning scene and obstacle insertion.
3. Run autonomous MuJoCo loop to validate policy/perception interfaces.
4. Collect training rollouts.
5. Train YOLO and ACT checkpoints.
6. Re-run with trained checkpoints and compare metrics.
7. Integrate trained perception/policy into SO101 ROS stack.

---

## 12) Key Files to Read First

1. `Context/Plan-1.md`
2. `runbook.md`
3. `so101_ws/src/so101_moveit/launch/simulated_robot.launch.py`
4. `so101_ws/src/so101_moveit/so101_moveit/add_scene_objects.py`
5. `macos_pipeline/run_autonomous.py`
6. `macos_pipeline/collect_training_data.py`
7. `macos_pipeline/policy/act_policy.py`
8. `macos_pipeline/perception/yolo_cube_detector.py`
9. `densefusion_pick_place.py`

---

This README is designed as the canonical operations guide for your current challenge implementation trajectory: baseline SO101 reliability first, then policy learning depth, then advanced autonomy.

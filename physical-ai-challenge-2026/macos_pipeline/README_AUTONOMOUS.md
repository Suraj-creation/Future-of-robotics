# Autonomous Pick-and-Place Simulation for SO101 Robot Arm

A fully autonomous robotic manipulation system using **MuJoCo**, **YOLOv8** for perception, and **ACT (Action Chunking Transformer)** for policy learning.

## Features

- **Fully Autonomous**: No human intervention required - detects cubes, plans actions, and executes pick-and-place automatically
- **SO101 Robot Arm**: Simulated using MuJoCo physics engine with accurate SO101 kinematics
- **YOLOv8 Perception**: Real-time cube detection with option to train on synthetic data
- **ACT Policy**: Action Chunking Transformer for smooth, coordinated motion planning
- **Terminal-based**: Runs entirely in terminal with progress logging

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install mujoco numpy ultralytics opencv-python pillow
```

### Run Simulation

```bash
# Navigate to macos_pipeline directory
cd /Users/udbhavkulkarni/Downloads/Future-of-robotics-main/physical-ai-challenge-2026/macos_pipeline

# Option 1: Use the launcher script (recommended)
./launch_simulation.sh --cycles 10

# Option 2: Run directly with Python
python run_autonomous.py --cycles 10

# Option 3: Use the full autonomous runner
python autonomous_runner.py --cycles 10
```

### Command Line Options

```bash
# Run with custom number of cycles
python run_autonomous.py --cycles 20

# Run headless (no visualization, faster)
python run_autonomous.py --cycles 10 --headless

# Run with trained YOLOv8 model
python run_autonomous.py --yolo-model path/to/yolov8_cube.pt

# Run with trained ACT policy
python run_autonomous.py --act-model path/to/act_model

# Combine all options
python run_autonomous.py \
    --cycles 50 \
    --yolo-model models/yolov8_cube.pt \
    --act-model models/act_policy \
    --headless
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   MuJoCo     │    │    YOLOv8    │    │     ACT      │    │
│  │  Simulation  │◄──►│  Perception  │───►│    Policy    │    │
│  │              │    │              │    │              │    │
│  │  • SO101 Arm │    │  • Detect    │    │  • Plan      │    │
│  │  • Cube      │    │  • Track     │    │  • Execute   │    │
│  │  • Target    │    │  • Estimate  │    │  • Adapt     │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│          ▲                                        │            │
│          │                                        │            │
│          └────────────────────────────────────────┘            │
│                    Action Commands                               │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    CONTROL LOOP                          │    │
│  │  1. Detect cube position (YOLOv8/analytic)            │    │
│  │  2. Plan pick-and-place actions (ACT)                  │    │
│  │  3. Execute actions in simulation                       │    │
│  │  4. Verify success and repeat                           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## File Structure

```
macos_pipeline/
├── autonomous_runner.py          # Full-featured autonomous runner
├── run_autonomous.py            # Terminal-friendly version
├── launch_simulation.sh         # Bash launcher script
│
├── perception/
│   ├── yolo_cube_detector.py   # YOLOv8 cube detection
│   └── analytic_pose.py         # Analytic detection fallback
│
├── policy/
│   └── act_policy.py            # ACT policy implementation
│
├── mujoco_demo/
│   ├── runner.py                # Demo runner
│   ├── scene_builder.py         # Scene construction
│   └── record_demo.py           # Demo recording
│
└── third_party/
    └── SO-ARM100/
        └── Simulation/SO101/
            └── so101_new_calib.xml  # SO101 robot model
```

## Training Custom Models

### Train YOLOv8 on Synthetic Data

```python
# Generate synthetic training data
from perception.yolo_cube_detector import generate_synthetic_training_data

generate_synthetic_training_data(
    output_dir='datasets/synthetic_cube',
    num_samples=1000
)

# Train YOLOv8
from perception.yolo_cube_detector import CubeDetector

detector = CubeDetector()
detector.train(
    data_yaml='datasets/synthetic_cube/data.yaml',
    epochs=100
)
```

### Train ACT Policy

Use the LeRobot training pipeline:

```bash
# Record demonstrations
python mujoco_demo/record_demo.py --out datasets/demo

# Train ACT (using LeRobot)
# See: https://github.com/huggingface/lerobot
```

## API Usage

### Perception Module

```python
from perception.yolo_cube_detector import CubeDetector

# Create detector
detector = CubeDetector(model_path='path/to/yolov8_cube.pt')

# Detect cube in image
position, confidence = detector.detect(frame=image)
print(f"Cube at {position} with confidence {confidence}")
```

### Policy Module

```python
from policy.act_policy import ACTPolicy

# Create policy
policy = ACTPolicy(mode='stub')  # or 'model' with trained checkpoint

# Plan actions
cube_pose = (0.25, 0.0, 0.02)
target_pose = (0.0, 0.25, 0.02)
actions = policy.predict(cube_pose, target_pose)

for action in actions:
    print(f"Action: {action['name']}")
    print(f"  Controls: {action['ctrls']}")
```

### Autonomous Controller

```python
from autonomous_runner import AutonomousPickPlace

# Create controller
controller = AutonomousPickPlace(
    yolo_model='path/to/yolov8.pt',
    act_model='path/to/act_model',
    visualize=True
)

# Run autonomous operation
controller.run(num_cycles=10)
```

## Simulation Output

```
======================================================================
  SO101 ROBOT ARM - AUTONOMOUS PICK & PLACE SIMULATION
  Using YOLOv8 + ACT (Action Chunking Transformer)
======================================================================

Initializing...
  YOLO model: analytic (stub)
  ACT model: stub policy
  Cycles: 10

Found 6 actuators
Actuators: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

Starting autonomous operation...
Press Ctrl+C to stop

----------------------------------------------------------------------
[10:23:45] Cycle   1 | DETECT       | Cube at (0.251, -0.043, 0.020)
[10:23:45] Cycle   1 | PERCEPTION   | Detected (0.249, -0.045) conf=0.95
[10:23:46] Cycle   1 | PLAN         | Generated 9 action chunks
[10:23:46] Cycle   1 | EXECUTE      | move_home
[10:23:47] Cycle   1 | EXECUTE      | approach
[10:23:48] Cycle   1 | EXECUTE      | lower
[10:23:48] Cycle   1 | GRIPPER      | Grasping cube
[10:23:49] Cycle   1 | EXECUTE      | lift
[10:23:50] Cycle   1 | EXECUTE      | move_deliver
[10:23:51] Cycle   1 | EXECUTE      | place
[10:23:51] Cycle   1 | GRIPPER      | Releasing cube
[10:23:52] Cycle   1 | EXECUTE      | return_home
[10:23:53] Cycle   1 | COMPLETE     | SUCCESS | dist=0.012m | time=8.2s
----------------------------------------------------------------------
...

======================================================================
FINAL RESULTS
======================================================================
Total cycles: 10
Successful: 9
Success rate: 90.0%
Average cycle time: 8.5s
Fastest cycle: 7.8s
Slowest cycle: 9.2s
======================================================================
```

## Troubleshooting

### MuJoCo Not Found
```bash
pip install mujoco
```

### YOLOv8 Not Found
```bash
pip install ultralytics
```

### SO101 Model Not Found
Ensure the SO-ARM100 repository is cloned:
```bash
cd third_party
git clone https://github.com/huggingface/SO-ARM100.git
```

### Display Issues (macOS)
The simulation runs headless by default. For visualization:
```bash
python run_autonomous.py  # No headless flag
```

## References

- **ACT**: [Action Chunking with Transformers](https://arxiv.org/abs/2305.00465)
- **YOLOv8**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **SO101**: [SO-ARM100](https://github.com/huggingface/SO-ARM100)
- **MuJoCo**: [MuJoCo Physics Engine](https://mujoco.org/)
- **LeRobot**: [HuggingFace LeRobot](https://github.com/huggingface/lerobot)

## License

This project is for the Physical AI Hackathon 2026.

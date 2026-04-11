# 🏆 Physical AI Hackathon 2026 — Elite Competitor's Battle Plan
> **GPU: 20GB + 24GB VRAM | Full Pipeline | SOTA Models | Win-Optimized**

---

## ⚡ TLDR — The Winning Stack

```
YOUR DISRUPTION EDGE
├── Task 1 (Pick & Place)   → GR00T N1.6  fine-tuned, 24GB GPU
├── Task 2 (Charger Plug)   → Pi0.5       fine-tuned via openpi, 24GB GPU  
├── Task 3 (Liquid Pour)    → Pi0.5       same model, language-conditioned
├── Task 4 (Humanoid Walk)  → GR00T N1.6  + MuJoCo + MediaPipe pipeline
└── Final Round             → Pi0.5       unified language-driven policy
```

Most teams will use ACT or SmolVLA. **You will use Pi0.5 + GR00T N1.6 — the most capable open models on Earth for manipulation as of 2026.** With 24GB VRAM, you can run them. With the right data strategy, you need only 20–40 demos per task.

---

## Table of Contents

1. [Competitive Threat Model — Where Others Will Fail](#1-competitive-threat-model)
2. [Your Full Tech Stack](#2-your-full-tech-stack)
3. [GPU Allocation Strategy](#3-gpu-allocation-strategy)
4. [Model Selection Deep Dive](#4-model-selection-deep-dive)
5. [The Data Strategy (Your Biggest Weapon)](#5-the-data-strategy)
6. [All Datasets You Must Download](#6-all-datasets-you-must-download)
7. [Full Pipeline — Task by Task](#7-full-pipeline--task-by-task)
   - [Task 1: Object Pick & Place](#task-1-object-pick--place)
   - [Task 2: Charger Plugging](#task-2-charger-plugging)
   - [Task 3: Liquid Pouring](#task-3-liquid-pouring)
   - [Task 4: Dynamic Humanoid Walking](#task-4-dynamic-humanoid-walking)
   - [Final Round: Unified Multi-Task](#final-round-unified-multi-task)
8. [Environment Setup — Complete Commands](#8-environment-setup--complete-commands)
9. [Training Pipeline — Exact Commands](#9-training-pipeline--exact-commands)
10. [Evaluation & Iteration Loop](#10-evaluation--iteration-loop)
11. [Advanced Tricks That Win Competitions](#11-advanced-tricks-that-win-competitions)
12. [Complete Resource Directory](#12-complete-resource-directory)

---

## 1. Competitive Threat Model

### Where Other Teams Will Get Stuck

| Team Type | Their Approach | Why They Fail |
|---|---|---|
| **Beginner teams** | Basic ACT, train from scratch | 50+ demos per task, overfit easily, no generalization |
| **Average teams** | SmolVLA fine-tune | Good baseline but limited precision on plugging/pouring |
| **Strong teams** | Pi0 (v1) | Misses open-world generalization that Pi0.5 adds |
| **You** | Pi0.5 + GR00T N1.6 + smart data | Best-in-class models + domain randomization data = win |

### Your Unfair Advantages

1. **Foundation models with 10,000+ hours of pre-training** — you start from a robot that already understands manipulation physics deeply
2. **GR00T N1.6's FLARE objective** trains on internet videos too, not just robot data
3. **Pi0.5's open-world generalization** — handles position/object variation robustly
4. **Data multiplier via MimicGen** — generate 1000 demos from 5 human teleoperated demos automatically
5. **Dual-GPU strategy** — 20GB for fast ACT/SmolVLA iteration, 24GB for final Pi0.5/GR00T training

---

## 2. Your Full Tech Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                      COMPLETE TECH STACK                             │
│                                                                       │
│  SIMULATION LAYER          FRAMEWORK LAYER         MODEL LAYER       │
│  ┌─────────────┐          ┌──────────────┐        ┌─────────────┐   │
│  │  MuJoCo 3.x │          │  LeRobot     │        │  GR00T N1.6 │   │
│  │  SO-101 URDF│◄────────►│  v0.4+       │◄──────►│  Pi0.5      │   │
│  │  Humanoid   │          │  (PyTorch)   │        │  ACT        │   │
│  │  URDF       │          └──────┬───────┘        └─────────────┘   │
│  └─────────────┘                 │                                    │
│                            DATA LAYER                                 │
│          ┌────────────────────────────────────────┐                  │
│          │  Your Sim Demos + Community Datasets +  │                  │
│          │  MimicGen Augmentation + Domain Rand    │                  │
│          └────────────────────────────────────────┘                  │
│                                                                       │
│  PERCEPTION        HUMANOID LAYER          MONITORING                │
│  ┌──────────┐     ┌─────────────────┐     ┌──────────────────┐      │
│  │ YOLOv11  │     │ MediaPipe Pose  │     │ Weights & Biases │      │
│  │ + SAM2   │     │ Joint Mapping   │     │ + Rerun          │      │
│  └──────────┘     └─────────────────┘     └──────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why These Specific Tools?

| Tool | Why It's In Your Stack |
|---|---|
| **GR00T N1.6** | Latest NVIDIA model (Dec 2025), has 32-layer DiT vs N1.5's 16-layer. Officially tested on SO-101. |
| **Pi0.5** | Open-world generalization > Pi0. Now natively in LeRobot v0.4. Handles novel object positions. |
| **ACT** | Fast iteration baseline. Train in 2 hours, validate pipeline, then switch to Pi0.5 |
| **MimicGen** | Generate 1000 high-quality demos from 5 human demos. Used by NVIDIA itself for GR00T training. |
| **YOLOv11 + SAM2** | Object detection + segmentation for robust perception pipeline |
| **MediaPipe** | Specified in hackathon problem statement for Task 4 |
| **Rerun** | Official LeRobot visualization tool for debugging joint trajectories |

---

## 3. GPU Allocation Strategy

You have **two GPUs: 20GB + 24GB VRAM**. Use them optimally:

### 24GB GPU (Primary — for Foundation Models)
```
CUDA_VISIBLE_DEVICES=0  # or whichever is your 24GB card
├── GR00T N1.6 fine-tuning   (requires ~24GB with diffusion head tuned)
├── Pi0.5 fine-tuning         (requires 16-20GB, fits with bs=8)
└── Final inference           (heavy model, needs 24GB)
```

### 20GB GPU (Secondary — for Fast Iteration)
```
CUDA_VISIBLE_DEVICES=1  # 20GB card
├── ACT fine-tuning           (needs only 6-8GB, very fast)
├── SmolVLA fine-tuning       (needs 8-10GB)
├── Pipeline debugging        (quick eval loops)
└── Perception models         (YOLOv11, SAM2)
```

### Memory Optimization Flags

**For GR00T on 20GB (if needed):**
```bash
# Add this flag to reduce VRAM by freezing diffusion model
python scripts/gr00t_finetune.py \
  --no-tune_diffusion_model \   # saves ~5GB VRAM
  --batch-size 8 \
  --lora-rank 16               # LoRA fine-tuning saves memory
```

**For Pi0.5 on 20GB:**
```bash
lerobot-train \
  --policy.type=pi05 \
  --batch_size=4 \             # reduce from default 8
  --policy.device=cuda \
  --gradient_accumulation_steps=2  # simulate larger batch
```

---

## 4. Model Selection Deep Dive

### The Model Hierarchy (Ranked for This Hackathon)

```
CAPABILITY TIER
                    Pi0.5          GR00T N1.6
    ████████████  [BEST FOR]     [BEST FOR]
    Foundation    Multi-task     Manipulation
    VLA           Final round    + Humanoid
         ↓
    SmolVLA       GR00T N1.5
    ████████      ██████████
    Good general  Good, but N1.6
    purpose       supersedes it
         ↓
    ACT / Diffusion Policy
    ████
    Great for baseline/fast iteration
```

### 🔴 GR00T N1.6 — Your Weapon for Tasks 1, 2, 3, 4

**What's new in N1.6 vs N1.5:**
- 2x larger DiT (32 layers vs 16 layers) → better motion quality
- Internal NVIDIA Cosmos-Reason-2B VLM backbone → better language grounding
- Flexible image resolution — no padding artifacts
- Trained on bimanual + semi-humanoid + full humanoid data
- Officially tested on SO-100/SO-101

**GitHub:** https://github.com/NVIDIA/Isaac-GR00T
**HuggingFace:** https://huggingface.co/nvidia/GR00T-N1.5-3B (N1.6 coming to same repo)

**VRAM Requirements:**
| Mode | VRAM |
|---|---|
| Full fine-tune (diffusion head ON) | ~24-25GB |
| Fine-tune with `--no-tune_diffusion_model` | ~16-18GB |
| LoRA fine-tune (`--lora-rank 16`) | ~12-14GB |
| Inference only | ~8-10GB |

---

### 🔵 Pi0.5 (π₀.₅) — Your Weapon for Final Round

**What makes Pi0.5 different:**
- Built for **open-world generalization**, not just a memorized task
- Predicts relative joint offsets by default (more stable training)
- Now natively in **LeRobot v0.4** — no extra setup needed
- Trained on 10,000+ hours of cross-robot data
- Language conditioning actually works (unlike Pi0 fine-tunes which often ignore prompts)

**HuggingFace:** https://huggingface.co/lerobot/pi05
**Docs:** https://huggingface.co/docs/lerobot/pi05
**OpenPI GitHub:** https://github.com/Physical-Intelligence/openpi

**Fine-tune via LeRobot (easiest way):**
```bash
lerobot-train \
  --dataset.repo_id=YOUR_USER/so101_multitask \
  --policy.type=pi05 \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --batch_size=8 \
  --steps=30000 \
  --policy.device=cuda
```

**VRAM Requirements:**
| Mode | VRAM |
|---|---|
| Full fine-tune | ~16-20GB |
| Inference only | ~12GB |

---

### 🟡 ACT — Your Iteration Baseline (Use First to Validate Pipeline)

Before spending hours training Pi0.5, use ACT to validate your entire pipeline works:
- Train in 2-3 hours on 50 demos
- If ACT works, your data pipeline is correct
- Then switch to Pi0.5 for final quality

**Train ACT:**
```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=YOUR_USER/so101_pickplace \
  --training.num_epochs=200 \
  --policy.chunk_size=100 \
  --policy.push_to_hub=false
```

---

## 5. The Data Strategy

### Why Data Quality Beats Model Size

A mediocre model trained on 200 high-quality demos > A great model trained on 50 sloppy demos.

### The 3-Layer Data Strategy

```
LAYER 3: Your Custom Demos (50-100 per task, simulation)
           ↑ Fine-tune on top of ↑
LAYER 2: Community SO-101 Datasets (download from HF Hub)
           ↑ Pre-train on top of ↑
LAYER 1: Foundation Pre-training (already baked into GR00T/Pi0.5)
```

### Data Collection Protocol for Simulation

When collecting simulation demos, follow these rules:

**Variation is everything.** Don't record 50 demos of the cube in the exact same position. Vary:
- Object position (left, right, center, rotated 0°/45°/90°)
- Gripper approach angle
- Speed of motion (some slow, some medium)
- Starting arm pose

**Quality over quantity:**
- Delete any episode where you hesitate, backtrack, or make jerky movements
- Every episode should be smooth and purposeful
- Aim for 80% success rate on demos (discard failures unless they demonstrate recovery)

**Resolution:** Record at minimum 640×480, preferably 1280×720 at 30fps

### MimicGen — Your Secret Weapon (1000 Demos from 5)

MimicGen is a data augmentation tool that takes your human-collected demos and automatically generates hundreds of variations by randomizing object positions. NVIDIA used it to generate much of GR00T's training data.

**GitHub:** https://github.com/NVlabs/mimicgen

```bash
# Install MimicGen
pip install mimicgen

# Generate 1000 episodes from your 5 source demos
python mimicgen/scripts/generate_dataset.py \
  --config mimicgen/exps/paper/core/pick_place_d0.json \
  --source_path ./demos/pick_place_source.hdf5 \
  --num_episodes 1000 \
  --output_path ./demos/pick_place_generated.hdf5
```

Then convert to LeRobot format:
```bash
python scripts/convert_robosuite_to_lerobot.py \
  --input ./demos/pick_place_generated.hdf5 \
  --output_dir ~/.cache/huggingface/lerobot/YOUR_USER/so101_pickplace_aug
```

---

## 6. All Datasets You Must Download

### Official SO-101 Datasets on HuggingFace

```bash
# Core SO-101 pick-place dataset (SmolVLA training data, 50 episodes)
huggingface-cli download \
  --repo-type dataset lerobot/svla_so101_pickplace \
  --local-dir ./datasets/svla_so101_pickplace

# SO-101 table cleanup (GR00T N1.5 tutorial dataset)
huggingface-cli download \
  --repo-type dataset youliangtan/so101-table-cleanup \
  --local-dir ./datasets/so101-table-cleanup

# LIBERO (130+ tasks, for Pi0.5 pre-training boost)
huggingface-cli download \
  --repo-type dataset HuggingFaceVLA/libero \
  --local-dir ./datasets/libero

# LIBERO for SmolVLA (smaller version)
huggingface-cli download \
  --repo-type dataset HuggingFaceVLA/smol-libero \
  --local-dir ./datasets/smol-libero
```

### NVIDIA's Physical AI Datasets

```bash
# NVIDIA manipulation objects dataset (in LeRobot format)
huggingface-cli download \
  --repo-type dataset nvidia/PhysicalAI-Robotics-Manipulation-Objects \
  --local-dir ./datasets/nvidia-manipulation

# NVIDIA GR00T humanoid locomotion + loco-manipulation
huggingface-cli download \
  --repo-type dataset nvidia/Arena-G1-Loco-Manipulation-Task \
  --local-dir ./datasets/nvidia-humanoid
```

### Other Critical Datasets

```bash
# ALOHA insertion and transfer tasks (great for charger plugging!)
huggingface-cli download \
  --repo-type dataset lerobot/aloha_sim_insertion_human \
  --local-dir ./datasets/aloha_insertion

# ALOHA transfer cube (bimanual precision)
huggingface-cli download \
  --repo-type dataset lerobot/aloha_sim_transfer_cube_human \
  --local-dir ./datasets/aloha_transfer

# DROID dataset (diverse robot manipulation, 76K demos)
# This is huge — use streaming instead of full download
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lerobot/droid', streaming=True)
"
```

### Complete Dataset Reference Table

| Dataset | HF Link | Episodes | Best For |
|---|---|---|---|
| `lerobot/svla_so101_pickplace` | https://hf.co/datasets/lerobot/svla_so101_pickplace | 50 | Task 1, SmolVLA training |
| `youliangtan/so101-table-cleanup` | https://hf.co/datasets/youliangtan/so101-table-cleanup | ~50 | GR00T N1.5/N1.6 SO-101 fine-tune |
| `lerobot/aloha_sim_insertion_human` | https://hf.co/datasets/lerobot/aloha_sim_insertion_human | 50 | Task 2 (insertion/plugging analog) |
| `HuggingFaceVLA/libero` | https://hf.co/datasets/HuggingFaceVLA/libero | 130+ tasks | Pi0.5 training |
| `lerobot/droid` | https://hf.co/datasets/lerobot/droid | 76K | Large-scale pre-training |
| `nvidia/PhysicalAI-Robotics-Manipulation-Objects` | https://hf.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects | 100s | GR00T fine-tune |
| `nvidia/Arena-G1-Loco-Manipulation-Task` | https://hf.co/datasets/nvidia/Arena-G1-Loco-Manipulation-Task | 50 | Task 4 humanoid |
| `lerobot/pusht` | https://hf.co/datasets/lerobot/pusht | 206 | Diffusion Policy baseline |
| `lerobot/aloha_sim_transfer_cube_human` | https://hf.co/datasets/lerobot/aloha_sim_transfer_cube_human | 50 | Precision grasping |

### Finding More SO-101 Community Datasets

```python
from huggingface_hub import list_datasets

# Find all SO-101 datasets on the Hub
datasets = list_datasets(filter="LeRobot", search="so101")
for ds in datasets:
    print(ds.id)
```

Or browse: https://huggingface.co/datasets?search=so101&other=LeRobot

---

## 7. Full Pipeline — Task by Task

### Task 1: Object Pick & Place

**Recommended Model:** GR00T N1.6 (Tasks 1-3 unified) OR Pi0.5 (final round)
**Backup:** ACT (fastest to train, great baseline)

**Step-by-step pipeline:**

```
STEP 1: Collect 50 sim demos via keyboard/gamepad teleoperation
         ↓
STEP 2: Augment to 1000 demos via MimicGen
         ↓
STEP 3: Download lerobot/svla_so101_pickplace (50 official demos)
         ↓
STEP 4: Merge datasets
         lerobot-edit-dataset --operation.type merge_datasets \
           --datasets '["YOUR/so101_pickplace", "lerobot/svla_so101_pickplace"]'
         ↓
STEP 5: Validate with ACT first (2-3 hours training)
         ↓
STEP 6: Fine-tune GR00T N1.6 on merged dataset (24GB GPU)
         ↓
STEP 7: Evaluate and iterate
```

**Key perception trick:** Use YOLOv11 for object detection as a **pre-processing stage** before passing to policy:
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # fast nano version

def get_object_crop(frame):
    results = model(frame)
    box = results[0].boxes[0]  # assume first detected box is target
    x1, y1, x2, y2 = box.xyxy[0]
    return frame[int(y1):int(y2), int(x1):int(x2)]  # crop to object
```

This makes the policy more robust because it always sees the object centered, regardless of exact position.

---

### Task 2: Charger Plugging

This is the **hardest task** because of precise insertion. Most teams will struggle here.

**Your edge:** Pi0.5 handles fine-grained manipulation better than ACT because its flow-matching action head models the full action distribution — it can handle the micro-adjustments needed for alignment.

**Perception pipeline for Task 2:**
```
Camera feed
    ↓
YOLOv11 (detect cable connector + socket)
    ↓
SAM2 (segment connector precisely)
    ↓
Estimate 6D pose of connector
    ↓
Pi0.5 policy (language: "plug the USB-C cable into the socket")
    ↓
Joint commands
```

**Training data strategy:**
- Collect 50 simulation demos with varied cable initial positions
- Use the ALOHA insertion dataset as pre-training data (it has similar insert-into-socket dynamics)
- The key demo quality requirement: your demos must show the **alignment phase** clearly — slow, deliberate connector approach and alignment

```bash
# Merge insertion task data for training
lerobot-edit-dataset \
  --repo_id YOUR_USER/so101_charger_plug \
  --operation.type merge_datasets \
  --datasets '["YOUR_USER/so101_charger_plug", "lerobot/aloha_sim_insertion_human"]'
```

**Fine-tune Pi0.5 for Task 2:**
```bash
lerobot-train \
  --dataset.repo_id=YOUR_USER/so101_charger_merged \
  --policy.type=pi05 \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --batch_size=8 \
  --steps=30000 \
  --policy.device=cuda \
  --output_dir=outputs/pi05_charger
```

---

### Task 3: Liquid Pouring

**Your edge:** Pi0.5's continuous-action flow matching is ideal for smooth pouring — it naturally generates fluid, multi-timestep trajectories rather than jerky point-to-point motion.

**Volume control strategy:**
```python
# Use MuJoCo's built-in fluid simulation or approximate with tilt angle
def estimate_pour_volume(tilt_angle_deg, bottle_fill_fraction, time_poured_s):
    """
    Approximate volume poured based on physics
    flow_rate ≈ k * sin(tilt_angle) * sqrt(fill_level)
    """
    import math
    k = 15.0  # empirical constant (calibrate per simulation)
    flow_rate = k * math.sin(math.radians(tilt_angle_deg)) * math.sqrt(bottle_fill_fraction)
    return flow_rate * time_poured_s  # ml

# In your policy eval loop
poured_volume = 0
target_volume = 50  # ml, from problem statement

while poured_volume < target_volume:
    obs = env.get_obs()
    action = policy.select_action(obs)
    poured_volume += estimate_pour_volume(
        tilt_angle_deg=get_bottle_tilt(obs),
        bottle_fill_fraction=get_fill_level(obs),
        time_poured_s=1.0 / 30  # 30Hz control
    )
    env.step(action)

# Command robot to upright the bottle when done
env.step(upright_action)
```

**Language conditioning trick for Pi0.5:**
```python
# Instead of generic "pour liquid", be specific with volume
task_description = "Grasp the water bottle, move it above the cup, tilt to pour 50ml of liquid, then upright the bottle"
```

---

### Task 4: Dynamic Humanoid Walking

This task is completely different from Tasks 1-3. It requires:
1. A working humanoid simulation in MuJoCo
2. Webcam-based human pose estimation via MediaPipe
3. Joint angle mapping (human joints → robot URDF joints)
4. Stability maintenance

**Full pipeline:**

```
STEP 1: Load humanoid URDF in MuJoCo
STEP 2: Implement stable walking baseline (PD controller or RL policy)
STEP 3: Connect MediaPipe pose estimation
STEP 4: Build joint mapping layer
STEP 5: Add joint limit clamping
STEP 6: Test stability under commanded poses
```

**MediaPipe Setup:**
```bash
pip install mediapipe opencv-python
```

```python
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # highest accuracy
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def get_human_joint_angles(frame_bgr):
    """Returns dict of joint angles from webcam frame"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    
    # Calculate key joint angles
    # Left knee: angle between hip, knee, ankle
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
    
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    
    # Right knee
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
    
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
    
    return {
        "left_knee": left_knee_angle,
        "right_knee": right_knee_angle,
        # Add more joints as needed
    }

import numpy as np

def calculate_angle(a, b, c):
    """Calculate angle at point b given three 2D points a, b, c"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
```

**Joint Mapping Layer (Human → Robot URDF):**
```python
# Human joint ranges (approximate degrees)
HUMAN_RANGES = {
    "left_knee":  (0, 160),    # can fully bend
    "right_knee": (0, 160),
    "left_hip_flex":  (-30, 120),
    "right_hip_flex": (-30, 120),
    "left_ankle": (-30, 50),
    "right_ankle": (-30, 50),
}

# Robot URDF joint limits (READ THESE FROM YOUR URDF FILE!)
# Example values — replace with actual URDF limits
ROBOT_RANGES = {
    "left_knee":  (-0.1, 2.0),    # radians
    "right_knee": (-0.1, 2.0),
    "left_hip_flex":  (-0.5, 1.8),
    "right_hip_flex": (-0.5, 1.8),
    "left_ankle": (-0.5, 0.7),
    "right_ankle": (-0.5, 0.7),
}

def remap_joint(human_angle_deg, joint_name):
    """Map human joint angle to robot joint angle with safety clamping"""
    h_min, h_max = HUMAN_RANGES[joint_name]
    r_min, r_max = ROBOT_RANGES[joint_name]
    
    # Normalize human angle to [0, 1]
    normalized = (human_angle_deg - h_min) / (h_max - h_min)
    normalized = np.clip(normalized, 0, 1)
    
    # Scale to robot range and convert to radians
    robot_angle_rad = r_min + normalized * (r_max - r_min)
    return float(robot_angle_rad)
```

**MuJoCo Stability Controller:**
```python
import mujoco
import numpy as np

class HumanoidController:
    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.target_joints = {}
        
    def set_target(self, joint_name: str, target_rad: float):
        """Set target angle with safety checks"""
        joint_id = self.model.joint(joint_name).id
        joint_min = self.model.jnt_range[joint_id][0]
        joint_max = self.model.jnt_range[joint_id][1]
        
        # Safety clamp
        safe_target = np.clip(target_rad, joint_min + 0.05, joint_max - 0.05)
        self.target_joints[joint_name] = safe_target
        
    def pd_step(self, kp=100.0, kd=10.0):
        """PD controller step — applies torques to track targets"""
        for joint_name, target in self.target_joints.items():
            joint_id = self.model.joint(joint_name).id
            qpos_idx = self.model.jnt_qposadr[joint_id]
            qvel_idx = self.model.jnt_dofadr[joint_id]
            
            current_pos = self.data.qpos[qpos_idx]
            current_vel = self.data.qvel[qvel_idx]
            
            # PD control torque
            torque = kp * (target - current_pos) - kd * current_vel
            
            # Apply to actuator
            actuator_id = self.model.actuator(joint_name + "_act").id
            self.data.ctrl[actuator_id] = np.clip(torque, -100, 100)
        
        mujoco.mj_step(self.model, self.data)
```

**GR00T N1.6 for Humanoid (Optional, Advanced):**
```bash
# Download GR00T humanoid dataset
huggingface-cli download \
  --repo-type dataset nvidia/Arena-G1-Loco-Manipulation-Task \
  --local-dir ./datasets/gr00t_humanoid

# Fine-tune GR00T on humanoid walking
python scripts/gr00t_finetune.py \
  --dataset-path ./datasets/gr00t_humanoid \
  --num-gpus 1 \
  --output-dir ./checkpoints/gr00t_humanoid \
  --max-steps 20000 \
  --data-config so101_tricam_bimanual \
  --embodiment-tag new_embodiment \
  --batch-size 8
```

---

### Final Round: Unified Multi-Task

The final round requires all 3 arm tasks (Pick+Place → Plug Charger → Pour Liquid) in sequence **without human intervention**.

**Your strategy: Single Pi0.5 model, language-switched.**

```python
# Final round orchestrator
from lerobot.policies import Pi05Policy

policy = Pi05Policy.from_pretrained("YOUR_USER/pi05_multitask_so101")

task_sequence = [
    ("pick_place", "Pick up the red cube and place it in the blue target zone"),
    ("charger_plug", "Grasp the USB-C cable connector and plug it into the socket"),
    ("liquid_pour", "Pick up the water bottle and pour 50ml into the receiving cup"),
]

for task_name, task_description in task_sequence:
    print(f"Starting task: {task_name}")
    
    obs = env.reset_task(task_name)
    done = False
    step = 0
    max_steps = 500
    
    while not done and step < max_steps:
        # Language-conditioned inference
        action = policy.select_action(obs, task_description=task_description)
        obs, reward, done, info = env.step(action)
        step += 1
        
        if done:
            print(f"Task {task_name} COMPLETED in {step} steps ✅")
        
    if not done:
        print(f"Task {task_name} TIMEOUT ⚠️")

print("Final round complete!")
```

**Training the unified multi-task model:**
```bash
# Merge all three task datasets
lerobot-edit-dataset \
  --operation.type merge_datasets \
  --datasets '["YOUR_USER/so101_pickplace", "YOUR_USER/so101_charger", "YOUR_USER/so101_pour"]' \
  --output_repo_id YOUR_USER/so101_multitask

# Fine-tune Pi0.5 on combined dataset
lerobot-train \
  --dataset.repo_id=YOUR_USER/so101_multitask \
  --policy.type=pi05 \
  --policy.use_relative_actions=true \
  --batch_size=8 \
  --steps=50000 \              # more steps for multi-task
  --policy.device=cuda \
  --output_dir=outputs/pi05_multitask
```

---

## 8. Environment Setup — Complete Commands

### System Requirements
- Ubuntu 22.04 (inside Docker or native)
- CUDA 12.4+
- Python 3.10
- 40GB+ disk space for datasets

### Step 1: Clone Competition Repo + Dependencies
```bash
# Competition repo
git clone https://github.com/vishal-finch/physical-ai-challange-2026.git
cd physical-ai-challange-2026

# Start the provided Docker environment
cd workshop/dev/docker
docker compose up -d
docker compose exec dev bash
```

### Step 2: Install LeRobot v0.4+ (Inside Container)
```bash
# Latest LeRobot with all features (Pi0.5, GR00T N1.5/N1.6, LIBERO, Meta-World)
pip install lerobot

# Or from source for absolute latest
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[mujoco,smolvla,pi05,gr00t]"

# Verify installation
python -c "import lerobot; print(lerobot.__version__)"
```

### Step 3: Install GR00T N1.6
```bash
# Clone Isaac GR00T repo
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T

# Install with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[base]"

# Install flash-attention (required, takes ~10 mins)
pip install flash-attn --no-build-isolation

# Verify
python -c "from groot.models import GR00TPolicy; print('GR00T OK')"
```

### Step 4: Install OpenPI (for Pi0.5 via original implementation)
```bash
# Alternative to LeRobot's Pi0.5 — more control
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### Step 5: Install Perception Stack
```bash
# YOLOv11 for object detection
pip install ultralytics

# SAM2 for precise segmentation
pip install sam2

# MediaPipe for pose estimation (Task 4)
pip install mediapipe

# Rerun for visualization/debugging
pip install rerun-sdk
```

### Step 6: HuggingFace Auth
```bash
pip install huggingface_hub
huggingface-cli login
# Paste token from https://huggingface.co/settings/tokens
```

### Step 7: Weights & Biases (Training Monitor)
```bash
pip install wandb
wandb login
# Paste API key from https://wandb.ai/settings
```

### Step 8: Build ROS 2 Workspace
```bash
cd /workspace
colcon build --symlink-install
source install/setup.bash

# Test simulation
ros2 launch so101_sim so101_mujoco.launch.py
```

---

## 9. Training Pipeline — Exact Commands

### Phase 1: Validate Pipeline with ACT (Day 1-2)
```bash
# 1. Download official SO-101 dataset
huggingface-cli download --repo-type dataset lerobot/svla_so101_pickplace \
  --local-dir ./datasets/svla_so101_pickplace

# 2. Train ACT — should converge in 2-3 hours on your GPU
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --training.num_epochs=200 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.push_to_hub=false \
  --output_dir=outputs/act_baseline \
  --wandb.enable=true \
  --wandb.project=hackathon2026

# 3. Evaluate
lerobot-eval \
  --policy.path=outputs/act_baseline/checkpoints/last \
  --eval.n_episodes=20
```

### Phase 2: GR00T N1.6 Fine-Tune (Day 2-4, 24GB GPU)
```bash
cd Isaac-GR00T

# Copy GR00T-compatible modality file
cp getting_started/examples/so100_dualcam__modality.json \
   ./datasets/svla_so101_pickplace/meta/modality.json

# Full fine-tune (24GB GPU — tune everything)
CUDA_VISIBLE_DEVICES=0 python scripts/gr00t_finetune.py \
  --dataset-path ./datasets/svla_so101_pickplace \
  --num-gpus 1 \
  --output-dir ./checkpoints/gr00t_pickplace \
  --max-steps 20000 \
  --data-config so100_dualcam \
  --video-backend torchvision_av \
  --batch-size 16 \
  --save-steps 2000 \
  --dataloader-num-workers 8

# Memory-constrained fine-tune (20GB GPU — freeze diffusion)
CUDA_VISIBLE_DEVICES=1 python scripts/gr00t_finetune.py \
  --dataset-path ./datasets/svla_so101_pickplace \
  --num-gpus 1 \
  --output-dir ./checkpoints/gr00t_pickplace_lite \
  --max-steps 20000 \
  --data-config so100_dualcam \
  --video-backend torchvision_av \
  --no-tune_diffusion_model \
  --batch-size 8 \
  --lora-rank 16 \
  --save-steps 2000
```

### Phase 3: Pi0.5 Fine-Tune (Day 3-5)

**Option A: Via LeRobot (easiest, recommended)**
```bash
# Compute normalization stats first (required for Pi0.5)
lerobot-edit-dataset \
  --repo_id YOUR_USER/so101_multitask \
  --operation.type recompute_stats \
  --operation.relative_action true \
  --operation.chunk_size 50 \
  --operation.relative_exclude_joints "['gripper']" \
  --push_to_hub true

# Fine-tune Pi0.5
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --dataset.repo_id=YOUR_USER/so101_multitask \
  --policy.type=pi05 \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=5000 \
  --policy.device=cuda \
  --output_dir=outputs/pi05_multitask \
  --wandb.enable=true \
  --wandb.project=hackathon2026

# Fix n_action_steps in config (IMPORTANT!)
# Edit outputs/pi05_multitask/checkpoints/last/pretrained_model/config.json
# Change: "n_action_steps": 1  →  "n_action_steps": 50
```

**Option B: Via OpenPI Docker (cleanest)**
```bash
docker run --rm --gpus all \
  -v /path/to/your/lerobot_dataset:/data/input \
  -v /path/to/output:/data/output \
  ioaitech/train_openpi:pi05 \
  --batch_size 8 \
  --steps 30000 \
  --save_interval 5000 \
  --learning_rate 2.5e-5 \
  --action_horizon 50 \
  --prompt "pick up the red cube and place it in the target zone"
```

### Phase 4: Upload Models to HuggingFace
```bash
# Upload GR00T checkpoint
huggingface-cli upload YOUR_USER/gr00t_so101_hackathon \
  ./checkpoints/gr00t_pickplace/checkpoint-20000

# Upload Pi0.5 checkpoint
huggingface-cli upload YOUR_USER/pi05_so101_multitask \
  outputs/pi05_multitask/checkpoints/last/pretrained_model pretrained_model
```

---

## 10. Evaluation & Iteration Loop

### The Eval-Improve Cycle
```
Collect 50 demos → Train → Eval 20 episodes → Analyze failures
                                ↑                    ↓
                         Re-train with          Add demos covering
                         fixed data             failure cases
```

### Running Evaluation
```bash
# Evaluate GR00T
python scripts/inference_service.py --server \
  --model_path ./checkpoints/gr00t_pickplace \
  --embodiment-tag new_embodiment \
  --data-config so100_dualcam \
  --denoising-steps 4 &

# Then evaluate
python scripts/eval_lerobot.py \
  --env-type mujoco_so101 \
  --n-episodes 50

# Evaluate Pi0.5 (via LeRobot)
lerobot-eval \
  --policy.path=outputs/pi05_multitask/checkpoints/last \
  --env.type=gym_mujoco_so101 \
  --eval.n_episodes=50 \
  --eval.use_async_envs=true
```

### Visualize with Rerun
```python
import rerun as rr
from lerobot.utils.visualization_utils import visualize_dataset

rr.init("lerobot_eval", spawn=True)
visualize_dataset(
    repo_id="YOUR_USER/so101_multitask",
    episode_index=0,
)
```

### Failure Analysis Checklist

When a task fails, check these in order:
1. **Object detection:** Did YOLOv11 correctly detect the object? Log bounding boxes.
2. **Grasp quality:** Did the gripper close too early/late? Check gripper joint angle in eval recording.
3. **Transit path:** Did the arm collide with something mid-path? Check joint trajectories.
4. **Placement precision:** Was the final position off? Measure error in simulation logs.
5. **Policy distribution:** Was the failure deterministic (always fails same way) or stochastic?

---

## 11. Advanced Tricks That Win Competitions

### Trick 1: Asynchronous Inference with SmolVLA
SmolVLA supports asynchronous inference — while executing action chunk N, it pre-computes chunk N+1. This doubles effective control frequency.

```python
# Set in config.json after training
{
  "n_action_steps": 50,
  "use_async_inference": true  # pre-computes next chunk during execution
}
```

### Trick 2: Domain Randomization in MuJoCo
Add random variation to your simulation to improve sim-to-real robustness:

```python
import mujoco
import numpy as np

def randomize_environment(model, data):
    """Apply domain randomization to improve policy robustness"""
    # Randomize object position (±5cm)
    obj_body_id = model.body("target_cube").id
    data.qpos[model.jnt_qposadr[model.body_jntadr[obj_body_id]]:
              model.jnt_qposadr[model.body_jntadr[obj_body_id]] + 3] += \
        np.random.uniform(-0.05, 0.05, 3)
    
    # Randomize lighting (camera brightness simulation)
    # Not available in MuJoCo directly but applies at dataset collection
    
    # Randomize joint friction slightly
    for i in range(model.njnt):
        model.dof_frictionloss[i] *= np.random.uniform(0.8, 1.2)
    
    mujoco.mj_forward(model, data)
```

### Trick 3: Data Mixing for Better Generalization
When training Pi0.5 or GR00T, mix your task-specific data with the larger LIBERO/DROID datasets:

```python
# In LeRobot training config
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

mixed_dataset = MultiLeRobotDataset(
    repo_ids=[
        "YOUR_USER/so101_pickplace",        # weight: 0.5
        "HuggingFaceVLA/libero",             # weight: 0.3
        "lerobot/aloha_sim_insertion_human", # weight: 0.2
    ],
    weights=[0.5, 0.3, 0.2]  # sample ratio
)
```

### Trick 4: Action Chunking Tuning
Different tasks need different chunk sizes:
- Pick & Place: `chunk_size=50` (moderate — task has clear phases)
- Charger Plugging: `chunk_size=20` (small — needs frequent re-planning)
- Liquid Pouring: `chunk_size=100` (large — smooth continuous motion)

### Trick 5: Temperature/Sampling Tuning at Inference
For flow-matching models (Pi0.5, GR00T), you can control the denoising steps:
- Fewer steps = faster but noisier (try 4-8 steps for real-time)
- More steps = slower but more accurate (try 16-32 for precision tasks)

```bash
# GR00T inference with fewer denoising steps (faster)
python scripts/inference_service.py --server \
  --model_path ./checkpoints \
  --denoising-steps 4     # vs default 16 — 4x faster inference

# This matters for 30Hz control: model must infer in <33ms
```

### Trick 6: For Task 4 — Add Walking Stability Module
Don't rely purely on learned policy for walking. Use a proven approach:

```python
class StabilityAugmentedController:
    """Combines learned pose following with PD stability baseline"""
    
    def __init__(self, humanoid_controller, policy_alpha=0.7):
        self.hc = humanoid_controller
        self.alpha = policy_alpha  # blend ratio: alpha=1 means 100% policy
    
    def get_action(self, obs, human_pose_command):
        # Get policy action (from GR00T or learned baseline)
        policy_action = self.policy.select_action(obs)
        
        # Get stability action (PD controller tracking neutral pose)
        stability_action = self.hc.get_stability_torques()
        
        # Check if policy would violate joint limits
        safe_policy_action = self.hc.clamp_to_limits(policy_action, margin=0.05)
        
        # Blend: more stability weighting for unstable states
        com_height = self.hc.get_center_of_mass()[2]
        min_safe_height = 0.7  # meters above ground
        
        if com_height < min_safe_height:
            alpha = 0.1  # heavily favor stability when about to fall
        else:
            alpha = self.alpha
        
        final_action = alpha * safe_policy_action + (1 - alpha) * stability_action
        return final_action
```

---

## 12. Complete Resource Directory

### Primary Repos

| Resource | URL | Purpose |
|---|---|---|
| **LeRobot** | https://github.com/huggingface/lerobot | Main framework |
| **Isaac GR00T** | https://github.com/NVIDIA/Isaac-GR00T | GR00T N1.5/N1.6 fine-tuning |
| **OpenPI** | https://github.com/Physical-Intelligence/openpi | Pi0 / Pi0.5 fine-tuning |
| **MimicGen** | https://github.com/NVlabs/mimicgen | Data augmentation |
| **MuJoCo** | https://github.com/google-deepmind/mujoco | Physics simulation |
| **Competition Repo** | https://github.com/vishal-finch/physical-ai-challange-2026 | Hackathon setup |

### Models on HuggingFace

| Model | URL | VRAM | Best For |
|---|---|---|---|
| **GR00T N1.5-3B** | https://hf.co/nvidia/GR00T-N1.5-3B | 24GB full / 16GB LoRA | All arm tasks |
| **Pi0.5** | https://hf.co/lerobot/pi05 | 20GB | Multi-task final round |
| **SmolVLA Base** | https://hf.co/lerobot/smolvla_base | 8GB | Fast iteration |
| **ACT SO101** | https://hf.co/lerobot/act_aloha_sim_insertion_human | 4GB | Baseline |
| **Diffusion Policy** | https://hf.co/lerobot/diffusion_pusht | 4GB | Smooth motion tasks |

### Datasets on HuggingFace

| Dataset | URL | Size |
|---|---|---|
| SO-101 Pick-Place (SmolVLA) | https://hf.co/datasets/lerobot/svla_so101_pickplace | 50 eps |
| SO-101 Table Cleanup | https://hf.co/datasets/youliangtan/so101-table-cleanup | ~50 eps |
| ALOHA Insertion | https://hf.co/datasets/lerobot/aloha_sim_insertion_human | 50 eps |
| LIBERO | https://hf.co/datasets/HuggingFaceVLA/libero | 130 tasks |
| LIBERO (SmolVLA) | https://hf.co/datasets/HuggingFaceVLA/smol-libero | compact |
| DROID | https://hf.co/datasets/lerobot/droid | 76K eps |
| NVIDIA Manipulation | https://hf.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects | 100s eps |
| NVIDIA Humanoid | https://hf.co/datasets/nvidia/Arena-G1-Loco-Manipulation-Task | 50 eps |

### Documentation

| Topic | URL |
|---|---|
| LeRobot Docs | https://huggingface.co/docs/lerobot |
| LeRobot Pi0.5 | https://huggingface.co/docs/lerobot/pi05 |
| LeRobot SmolVLA | https://huggingface.co/docs/lerobot/main/en/smolvla |
| GR00T SO-101 Tutorial | https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning |
| LeRobot v0.4 Release | https://huggingface.co/blog/lerobot-release-v040 |
| SO-101 Docs | https://huggingface.co/docs/lerobot/so101 |
| MuJoCo Python Docs | https://mujoco.readthedocs.io/en/stable/python.html |
| MediaPipe Pose | https://developers.google.com/mediapipe/solutions/vision/pose_landmarker |
| OpenPI Fine-tuning | https://github.com/Physical-Intelligence/openpi#fine-tuning |

### Papers to Read

| Paper | Key Insight |
|---|---|
| Pi0.5 | https://arxiv.org/abs/2504.16054 — open-world generalization |
| GR00T N1 | https://arxiv.org/abs/2503.14734 — dual-system VLA |
| ACT | https://arxiv.org/abs/2304.13705 — action chunking |
| Diffusion Policy | https://arxiv.org/abs/2303.04137 — diffusion for robot actions |
| SmolVLA | https://arxiv.org/abs/2506.01844 — efficient VLA |
| MimicGen | https://arxiv.org/abs/2310.17596 — scalable data generation |

---

## Execution Timeline

```
DAY 0  (Before hackathon)
├── Complete all environment setup
├── Download all datasets listed above
├── Verify Docker environment works
└── Pre-train ACT on official SO-101 dataset (validate pipeline)

DAY 1  (Task 1 — Pick & Place)
├── Collect 50 simulation demos
├── Augment to 500 with MimicGen
├── Fine-tune GR00T N1.6 on 24GB GPU (20K steps, ~6 hrs)
├── Fine-tune SmolVLA on 20GB GPU simultaneously (validation)
└── Eval both, pick winner

DAY 2  (Task 2 — Charger Plugging)
├── Collect 50 simulation demos
├── Merge with ALOHA insertion dataset
├── Fine-tune Pi0.5 (most precise for insertion)
└── Iterate based on eval — focus on alignment phase

DAY 3  (Task 3 — Liquid Pouring)
├── Collect 50 simulation demos
├── Implement volume estimation module
├── Fine-tune Pi0.5 (same model, extend with pouring data)
└── Test volume accuracy

DAY 4  (Task 4 — Humanoid Walking)
├── Load humanoid URDF in MuJoCo
├── Implement MediaPipe pipeline
├── Build joint mapping layer with safety clamping
├── Implement PD stability controller
└── Test walking + human mirroring

DAY 5  (Final Round Prep)
├── Merge all 3 task datasets into unified multi-task dataset
├── Fine-tune Pi0.5 on multi-task dataset (50K steps)
├── Test full sequence: Pick → Plug → Pour (no intervention)
└── Debug transition logic between tasks

DAY 6-7  (Polish & Buffer)
├── Fix any remaining failure modes
├── Improve data quality for weakest task
└── Run 100-episode eval for each task, document results
```

---

## Quick Command Reference

```bash
# === SETUP ===
pip install lerobot && pip install ultralytics && pip install mediapipe

# === DATA ===
huggingface-cli download --repo-type dataset lerobot/svla_so101_pickplace
lerobot-edit-dataset --operation.type merge_datasets --datasets '["ds1","ds2"]'

# === TRAIN ACT (fast baseline, 20GB GPU) ===
lerobot-train --policy=act --dataset.repo_id=lerobot/svla_so101_pickplace

# === TRAIN SMOLVLA (mid-tier, 8GB) ===
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=YOUR/DATASET --steps=20000

# === TRAIN PI0.5 (best multi-task, 20-24GB) ===
lerobot-train --policy.type=pi05 --dataset.repo_id=YOUR/DATASET --steps=50000

# === TRAIN GROOT N1.6 (best manipulation, 24GB) ===
python scripts/gr00t_finetune.py --dataset-path ./data --max-steps 20000 --data-config so100_dualcam

# === EVAL ===
lerobot-eval --policy.path=outputs/CHECKPOINT --eval.n_episodes=50

# === GROOT INFERENCE SERVER ===
python scripts/inference_service.py --server --model_path ./checkpoints --denoising-steps 4

# === VISUALIZE ===
rerun  # opens GUI, then use lerobot visualization utils

# === UPLOAD ===
huggingface-cli upload YOUR_USER/MODEL_NAME outputs/CHECKPOINT/pretrained_model pretrained_model
```

---

*You are not just participating — you are disrupting. With Pi0.5, GR00T N1.6, and the right data strategy, you have the same tools that the world's top robotics labs use. Now execute. 🚀*

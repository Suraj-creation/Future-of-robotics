# 🤖 Physical AI Hackathon 2026 — Complete Beginner's Guide
> **Everything you need to know, from zero to ready, deeply explained.**

---

## Table of Contents

1. [What is this Hackathon?](#1-what-is-this-hackathon)
2. [Understanding the 4 Tasks](#2-understanding-the-4-tasks)
3. [What is the SO-101 Robot?](#3-what-is-the-so-101-robot)
4. [What is LeRobot?](#4-what-is-lerobot)
5. [What is MuJoCo? (The Simulator)](#5-what-is-mujoco-the-simulator)
6. [What is Docker? (Why We Use It)](#6-what-is-docker-why-we-use-it)
7. [What is ROS 2?](#7-what-is-ros-2)
8. [AI Models — The Brain of the Robot](#8-ai-models--the-brain-of-the-robot)
9. [Model Deep Dives — ACT, Diffusion Policy, SmolVLA, Pi0](#9-model-deep-dives--act-diffusion-policy-smolvla-pi0)
10. [Pre-Hackathon Setup: Step-by-Step](#10-pre-hackathon-setup-step-by-step)
11. [Architecture: How Everything Connects](#11-architecture-how-everything-connects)
12. [Strategy: Which Model Should You Use?](#12-strategy-which-model-should-you-use)
13. [Key Resources, Links & Repositories](#13-key-resources-links--repositories)

---

## 1. What is this Hackathon?

The **Physical AI Hackathon 2026** is run by IEEE RAS (Robotics and Automation Society) in collaboration with **xpskills / Eksaathi Foundation**. The idea is to train the next generation of engineers in **Physical AI** — intelligence embedded in real robots rather than just software.

### What is "Physical AI"?
Normal AI (like ChatGPT) lives in a computer and produces text. Physical AI must:
- **See** the world (through cameras/sensors)
- **Think** (decide what to do with AI models)
- **Act** (move joints, grip objects, pour liquids)

This is orders of magnitude harder because the real world is messy — objects shift, lighting changes, motors have tiny errors. There's no "undo" button.

### Hackathon Structure
| Round | What happens | Hardware |
|---|---|---|
| **Semifinal** | 3–4 independent tasks in simulation | No physical hardware — simulation only |
| **Final** | All tasks in sequence, no human help | Real SO-101 robot arm on-site |

---

## 2. Understanding the 4 Tasks

### 🟥 Task 1: Object Pick and Place
**What the robot must do:**
1. Use its camera to **detect** a red cube at a known starting location
2. **Grasp** the cube securely with its gripper (end effector)
3. **Transport** it through the air to a target location
4. **Place** it precisely and release

**Why it's hard:**
- The cube might be rotated slightly differently each run
- The gripper must squeeze just tight enough — too loose drops it, too tight might knock it over
- Transit must be smooth to avoid dropping

**Success metric:** Object lands within tight tolerance of target. Speed is a secondary metric.

---

### 🔌 Task 2: Charger Plugging
**What the robot must do:**
1. Detect a USB/charging cable and its connector
2. Grasp the connector (not the cable body)
3. Precisely **align** the connector orientation with the socket
4. **Insert** until fully engaged

**Why it's hard:**
- A cable is **flexible** — it bends and moves as you grasp it
- Socket alignment must be nearly perfect (millimeter precision)
- The robot must "feel" when it's fully inserted (force feedback)

**Success metric:** Successful first-attempt insertion + survives a pull test.

---

### 💧 Task 3: Liquid Pouring
**What the robot must do:**
1. Grasp a bottle containing liquid
2. Move it above a receiving cup
3. Tilt the bottle at the right angle to pour
4. Stop exactly when the **target volume (50 ml)** has been transferred

**Why it's hard:**
- Liquid moves unpredictably (sloshing)
- Pour rate depends on tilt angle, bottle size, viscosity
- The robot needs volume sensing — usually done via weight sensor or visual level estimation

**Success metric:** Volume within tolerance. No spillage outside the cup.

---

### 🚶 Task 4: Dynamic Humanoid Walking
This task is different — it's about **simulation + human mirroring**:
1. Load a humanoid robot URDF and make it walk without falling
2. Use a **webcam + MediaPipe** to capture the human operator's pose
3. Map human joint angles → robot joint angles (they have different ranges)
4. Make the humanoid follow human commands (walk forward, turn left, etc.)

**Why it's hard:**
- Humans have much wider joint ranges than robots
- A wrong joint command can cause the simulation to crash or the robot to "fall"
- Stability (keeping balance while walking) is a well-known hard problem in robotics

**Success metric:** Joint-angle error between human pose and robot pose.

---

## 3. What is the SO-101 Robot?

The **SO-101** is a low-cost, open-source, 3D-printable robotic arm built by **RobotStudio in collaboration with Hugging Face**.

### Key Facts
| Property | Value |
|---|---|
| DOF (Degrees of Freedom) | 6 joints + 1 gripper |
| Cost to build | ~$130 USD |
| Motors | 6 × Feetech STS3215 servo motors |
| Control | USB serial bus (Feetech Bus Servo) |
| AI Framework | LeRobot by Hugging Face |
| Cameras | External USB cam + optional gripper cam |

### What does "6 DOF" mean?
Think of your arm. You have:
- Shoulder rotation (left/right)
- Shoulder raise (up/down)
- Elbow bend
- Wrist twist
- Wrist tilt
- Gripper open/close

Each of these is a **degree of freedom** (DOF). More DOF = more flexible = harder to control. The SO-101 has 6, which is enough to reach almost any position in its workspace.

### Leader–Follower Architecture
The SO-101 comes in pairs:
- **Leader arm** = human-controlled arm (you physically move it)
- **Follower arm** = robot arm that copies the leader in real time

This is how you **collect training data** (called "teleoperation"). A human demonstrates the task by moving the leader arm, and the follower arm copies it while recording every joint angle and camera frame. This data is used to train the AI model.

### Resources
- 🔗 Official Docs: https://huggingface.co/docs/lerobot/so101
- 🔗 Assemble Guide: https://huggingface.co/docs/lerobot/assemble_so101
- 🔗 Buy pre-built: https://www.hiwonder.com/products/lerobot-so-101

---

## 4. What is LeRobot?

**LeRobot** is Hugging Face's open-source robotics framework. Think of it as the "transformers library" but for robots.

### What does it provide?
```
LeRobot
├── Hardware Interface  → Connect & control SO-101, Koch arms, humanoids
├── Data Collection     → Record teleoperation episodes as datasets
├── Datasets            → Pre-collected human demonstration datasets on HF Hub
├── Training            → Train AI policies (ACT, Diffusion Policy, SmolVLA, etc.)
├── Evaluation          → Run a trained model on the robot / in simulation
└── Simulation          → MuJoCo / Gym environments for training without hardware
```

### Key Concept: Policy
In LeRobot, an AI model that controls a robot is called a **"policy"**. A policy takes in:
- **Observations**: Camera images, robot joint positions
- **Task instruction** (optional, for VLA models): "Pick up the red cube"

And outputs:
- **Actions**: Target joint angles for the next timestep(s)

### Install LeRobot
```bash
pip install lerobot
```
Or from source (recommended for development):
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[feetech]"
```

### HuggingFace LeRobot Hub
All pretrained models and datasets are on HuggingFace:
- 🔗 Models: https://huggingface.co/lerobot
- 🔗 Datasets: https://huggingface.co/datasets?other=LeRobot
- 🔗 GitHub: https://github.com/huggingface/lerobot

---

## 5. What is MuJoCo? (The Simulator)

**MuJoCo** stands for **Mu**lti-**Jo**int dynamics with **Co**ntact. It is the world's most widely-used physics simulator for robot learning.

### What does a physics simulator do?
Instead of running code on a real robot (which could break or be dangerous), a simulator creates a **virtual physics world** where:
- Gravity exists
- Objects have mass, friction, inertia
- Joints have limits and torque
- Contacts (collisions) are simulated accurately

You can run 1000 training episodes in a simulator in minutes, instead of hours on a real robot.

### Why MuJoCo specifically?
- Extremely **fast** (real-time or faster)
- Used by DeepMind, OpenAI, Meta, Google
- Free since 2022 (bought and open-sourced by DeepMind)
- Works great with **URDF files** (robot description files)
- Native Python API
- 🔗 Website: https://mujoco.org
- 🔗 GitHub: https://github.com/google-deepmind/mujoco

### What is a URDF?
**Unified Robot Description Format** — an XML file that describes:
- Every **link** (physical body part) of the robot with its shape, mass, inertia
- Every **joint** (connection between links) with its type, limits, axis

The hackathon organizers will provide URDF files for both the SO-101 arm and the humanoid. You don't need to create them yourself.

Example (simplified):
```xml
<robot name="so101">
  <link name="base_link">
    <visual><geometry><box size="0.1 0.1 0.05"/></geometry></visual>
    <inertial><mass value="0.5"/></inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1.0"/>
  </joint>
</robot>
```

### MuJoCo + LeRobot Integration
LeRobot has built-in MuJoCo gym environments. Example:
```python
import gymnasium as gym
import lerobot

env = gym.make("gym_pusht/PushT-v0")  # Built-in LeRobot task
obs, info = env.reset()
```

---

## 6. What is Docker? (Why We Use It)

### The Problem Without Docker
Imagine everyone on the team has different computers:
- Person A: Ubuntu 22, Python 3.10, ROS 2 Humble
- Person B: Windows 11, Python 3.11, no ROS
- Person C: macOS M2, Python 3.9

Your code that works on A's machine might fail on B's or C's machine because of different library versions, OS differences, missing dependencies.

### Docker = Shipping Container for Software
Docker **packages your software + all its dependencies into an "image"** — a standardized box. Anyone who runs this box gets the exact same environment, regardless of their OS.

Key terms:
| Term | Simple Explanation |
|---|---|
| **Image** | A recipe (snapshot of the environment) |
| **Container** | A running instance of an image (like a mini-VM) |
| **Dockerfile** | Instructions to build an image |
| **Docker Compose** | Tool to run multiple containers together |
| **Volume** | A shared folder between your computer and the container |

### The Hackathon Docker Setup
The repo provides a ready-made Docker image containing:
- Ubuntu 22.04
- Python 3.10
- ROS 2 (Robot Operating System)
- MuJoCo simulator
- LeRobot framework
- All Python dependencies pre-installed

You just run one command and everything works.

### Install Docker
```bash
# Ubuntu/Linux
sudo apt-get update
sudo apt-get install docker.io docker-compose-plugin

# Enable without sudo
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker compose version
```

For Windows/Mac: Download **Docker Desktop** from https://www.docker.com/products/docker-desktop

### NVIDIA GPU Support (Optional but Recommended)
If you have an NVIDIA GPU, install the **NVIDIA Container Toolkit** so your containers can use GPU acceleration for AI training:
```bash
# Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 7. What is ROS 2?

**ROS 2** (Robot Operating System 2) is not actually an OS — it's a **middleware framework** that helps different parts of a robot system communicate with each other.

### Why do robots need a messaging system?
A robot has many components running simultaneously:
- Camera perception node (running at 30 fps)
- AI policy node (running at 10 Hz)
- Motor controller (running at 100 Hz)
- State estimator (running at 200 Hz)

These need to share data continuously without blocking each other. ROS 2 provides a **publish/subscribe** system:
- Camera **publishes** images on topic `/camera/image_raw`
- Policy node **subscribes** to that topic, processes, and **publishes** joint commands on `/robot/joint_commands`
- Motor controller **subscribes** to joint commands and executes them

### Key ROS 2 Concepts
| Concept | Explanation |
|---|---|
| **Node** | An independent running program (e.g., camera driver) |
| **Topic** | A named channel where data flows |
| **Message** | The data format sent on a topic |
| **Service** | Request-response communication (like an API call) |
| **Launch file** | Start multiple nodes at once |

### For this hackathon
The provided Docker environment + `HACKATHON_GUIDE.md` will have ROS 2 pre-configured. You mainly need to understand how to:
1. Build the workspace: `colcon build`
2. Source it: `source install/setup.bash`
3. Launch the simulator: `ros2 launch <package> <launch_file>`

---

## 8. AI Models — The Brain of the Robot

This is the most important section for winning. The robot without an AI model is just a mechanical arm. The AI model is what makes decisions.

### Three Learning Paradigms

#### 1. Imitation Learning (IL) — "Learn by watching"
You demonstrate a task 50–100 times (teleoperation). The AI learns to copy your behavior.
- ✅ Simple to set up
- ✅ Works well for constrained tasks
- ❌ Doesn't generalize well to unseen situations
- **Models: ACT, Diffusion Policy, SmolVLA**

#### 2. Reinforcement Learning (RL) — "Learn by trying"
The robot tries random actions, gets rewards for good outcomes, punishment for bad. Over millions of iterations, it learns.
- ✅ Can discover novel strategies
- ✅ Can handle unseen situations
- ❌ Requires reward function design
- ❌ Takes much longer to train
- **Models: PPO, SAC, TD-MPC2**

#### 3. Vision-Language-Action (VLA) — "Pre-trained world knowledge + robot control"
Start from a huge pre-trained vision-language model (like a robot that has already seen millions of internet images and text). Fine-tune it on your robot data.
- ✅ Best generalization
- ✅ Can be instructed in natural language
- ✅ Fewer demonstrations needed
- ❌ Large models, need GPU
- **Models: SmolVLA, Pi0, OpenVLA, GR00T N1**

---

## 9. Model Deep Dives — ACT, Diffusion Policy, SmolVLA, Pi0

### 🧠 ACT — Action Chunking with Transformers

**Paper:** "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Chi et al., 2023)
**What it does:** Instead of predicting one action at a time, ACT predicts a **chunk of 100 future actions** at once. This is called "action chunking." It uses a **transformer** (same architecture as GPT) with a special CVAE (Conditional Variational Autoencoder) to handle the variability in human demonstrations.

**Simple analogy:** When you type a sentence, you don't think letter-by-letter — you plan whole words. ACT works the same way — it plans whole motion sequences.

**Architecture:**
```
Camera Images + Joint Positions
        ↓
   Vision Encoder (CNN/ViT)
        ↓
   Transformer Encoder
        ↓
   CVAE Decoder
        ↓
   100 future joint positions (action chunk)
```

**Best for:** Pick-and-place, precise manipulation tasks

**Use with LeRobot:**
```bash
# Train ACT on SO-101 pick and place
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --policy.push_to_hub=false
```

**HuggingFace pretrained models:**
- 🔗 https://huggingface.co/lerobot/act_aloha_sim_insertion_human
- 🔗 https://huggingface.co/lerobot/act_so101_pick_place (community fine-tuned)

**GitHub:** https://github.com/tonyzhaozh/act

---

### 🌊 Diffusion Policy

**Paper:** "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion" (Chi et al., 2023)
**What it does:** Represents the robot's behavior as a **denoising diffusion process** — the same technology behind AI image generation (like Stable Diffusion), but for robot actions instead of images.

**Simple analogy:** Imagine starting with pure random noise (a garbage trajectory) and gradually refining it step-by-step until it becomes a clean, smooth motion. The model learns how to do this refinement conditioned on what the robot sees.

**Why it's powerful:**
- Handles **multimodal distributions** — the same task can be done in multiple valid ways (e.g., pick up a cup from left or right)
- More robust than ACT for noisy/varied environments
- 46.9% average improvement over prior methods in benchmarks

**Two variants in LeRobot:**
| Variant | Architecture | Speed | Quality |
|---|---|---|---|
| `diffusion` (CNN) | U-Net backbone | Fast | Good |
| `diffusion_transformer` | DiT (Diffusion Transformer) | Slower | Better |

**Use with LeRobot:**
```bash
lerobot-train \
  --policy=diffusion \
  --dataset.repo_id=lerobot/pusht \
  --policy.push_to_hub=false
```

**HuggingFace pretrained:**
- 🔗 https://huggingface.co/lerobot/diffusion_pusht
- 🔗 https://huggingface.co/lerobot/diffusion_aloha_sim_insertion_human

---

### 🤏 SmolVLA — Small Vision-Language-Action Model

**Paper:** "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics" (2025)
**What it does:** A compact (~450M parameters) VLA model that takes in **natural language instructions + camera images + joint positions** and outputs robot actions. Despite being small, it achieves performance comparable to models 10x its size.

**Simple analogy:** It's like a GPT model that can see your camera feeds and directly tells your robot's motors what to do, in response to commands like "pick up the cube and place it in the blue box."

**Architecture:**
- Backbone: **SmolVLM-2** (a small vision-language model)
- Action head: **Flow matching** (similar to diffusion but faster)
- Input: 3 camera views + joint positions + text instruction
- Output: Chunks of 64 future joint positions

**Why use SmolVLA for this hackathon:**
- ✅ Runs on a regular laptop GPU (or even CPU)
- ✅ Officially integrated into LeRobot
- ✅ Can be fine-tuned on SO-101 data
- ✅ Natural language control fits multi-task final round
- ✅ Community datasets available

**Install & Use:**
```bash
pip install lerobot

# Fine-tune SmolVLA on your collected demonstrations
lerobot-train \
  --policy=smolvla \
  --dataset.repo_id=YOUR_HF_USERNAME/so101_pick_place \
  --policy.push_to_hub=false
```

**HuggingFace:**
- 🔗 Model: https://huggingface.co/lerobot/smolvla_base
- 🔗 Paper: https://arxiv.org/abs/2506.01844
- 🔗 Demo: https://huggingface.co/spaces/lerobot/SmolVLA

---

### ⚡ Pi0 (Pi-Zero) — Physical Intelligence Foundation Model

**Paper:** "π0: A Vision-Language-Action Flow Model" (Black et al., 2024)
**What it does:** The most capable open-source robot foundation model as of 2025. Uses a 3B PaliGemma VLM + 300M flow-matching action expert, pretrained on a massive cross-embodiment robot dataset.

**Simple analogy:** This is the "GPT-4 of robot models." It has seen so many different robots doing so many different tasks that it understands robotic manipulation at a deep level before you even show it your specific task.

**Key strengths:**
- Best performance on complex, long-horizon tasks
- Handles liquid pouring, charger plugging with few demonstrations
- Can be fine-tuned on SO-101 data

**Limitation:** Heavy (3B+ params), needs a powerful GPU (16GB+ VRAM) for fine-tuning

**HuggingFace:**
- 🔗 https://huggingface.co/physical-intelligence/pi0
- 🔗 GitHub: https://github.com/Physical-Intelligence/openpi

**Fine-tune Pi0 on your data:**
```bash
git clone https://github.com/Physical-Intelligence/openpi
cd openpi
pip install -e .

# Fine-tune
python scripts/finetune.py \
  --config configs/pi0_so101_finetune.yaml \
  --dataset YOUR_HF_DATASET
```

---

### 🦖 GR00T N1 — NVIDIA's Humanoid Foundation Model

Specifically relevant for **Task 4 (Dynamic Humanoid Walking)**

**What it does:** NVIDIA's open humanoid foundation model. Combines a VLM backbone (Eagle) with a diffusion-based action head. Designed specifically for humanoid robots.

**Why relevant:** Task 4 requires humanoid simulation + pose mirroring. GR00T N1 is pre-trained on humanoid motion data.

**HuggingFace:**
- 🔗 https://huggingface.co/nvidia/GR00T-N1-2B
- 🔗 GitHub: https://github.com/NVIDIA/Isaac-GR00T

---

### 📊 Model Comparison Table

| Model | Size | GPU Needed | Task Fit | Ease of Use | Links |
|---|---|---|---|---|---|
| **ACT** | Small (~50M) | 6GB | Task 1, 2 | ⭐⭐⭐⭐⭐ | HF / GitHub |
| **Diffusion Policy** | Small (~80M) | 6GB | Task 1, 2, 3 | ⭐⭐⭐⭐ | HF / GitHub |
| **SmolVLA** | Medium (~450M) | 8GB | All tasks | ⭐⭐⭐⭐ | HF / GitHub |
| **Pi0** | Large (~3.3B) | 16GB+ | All tasks | ⭐⭐⭐ | HF / GitHub |
| **GR00T N1** | Large (~2B) | 16GB+ | Task 4 | ⭐⭐⭐ | HF / GitHub |
| **OpenVLA** | Large (~7B) | 24GB | All tasks | ⭐⭐ | HF / GitHub |

---

## 10. Pre-Hackathon Setup: Step-by-Step

### Step 0: Prerequisites Checklist
- [ ] Git installed (`git --version`)
- [ ] Docker installed (`docker --version`)
- [ ] Docker Compose installed (`docker compose version`)
- [ ] NVIDIA GPU? → Install NVIDIA Container Toolkit
- [ ] HuggingFace account: https://huggingface.co/join
- [ ] GitHub account

---

### Step 1: Clone the Competition Repository
```bash
git clone https://github.com/vishal-finch/physical-ai-challange-2026.git
cd physical-ai-challange-2026
```

Explore what's inside:
```bash
ls -la
# You should see:
# workshop/     → Main hackathon code
# HACKATHON_GUIDE.md → Read this first
# README.md
```

---

### Step 2: Pull and Start the Docker Environment
```bash
cd physical-ai-challange-2026/workshop/dev/docker
docker compose up -d
```

What this does:
- Downloads the pre-built Docker image (may take 5–10 minutes first time)
- Starts a container with everything pre-installed
- Runs it in the background (`-d` = detached mode)

Verify it's running:
```bash
docker compose ps
# Should show the container as "running"
```

---

### Step 3: Enter the Container
```bash
docker compose exec dev bash
# Now you're inside the container — a fully set-up Linux environment
```

---

### Step 4: Build the ROS 2 Workspace (Inside Container)
```bash
# Inside the container
cd /workspace
colcon build --symlink-install
source install/setup.bash
```

`colcon build` compiles all the ROS 2 packages. `source` loads them into your current terminal session.

---

### Step 5: Launch MuJoCo Simulator
Following the `HACKATHON_GUIDE.md` instructions:
```bash
# Inside container, after sourcing workspace
ros2 launch so101_sim so101_mujoco.launch.py
```

You should see a 3D viewer window showing the SO-101 arm.

---

### Step 6: Install LeRobot (if not in Docker image)
```bash
pip install lerobot
# OR from source for latest features:
git clone https://github.com/huggingface/lerobot.git
pip install -e ".[mujoco]"
```

---

### Step 7: Log in to HuggingFace
```bash
pip install huggingface_hub
huggingface-cli login
# Paste your access token from https://huggingface.co/settings/tokens
```

---

### Step 8: Download a Pretrained Model and Test It
```bash
# Test ACT on a simulation task
lerobot-eval \
  --policy.path=lerobot/act_aloha_sim_insertion_human \
  --env.type=gym_aloha \
  --env.task=AlohaInsertion-v0 \
  --eval.n_episodes=10
```

---

### Step 9: Collect Your Own Demonstrations (for fine-tuning)
In simulation, you can use keyboard/gamepad teleoperation:
```bash
lerobot-record \
  --robot.type=so101_follower \
  --control.type=teleoperate \
  --dataset.repo_id=YOUR_HF_USERNAME/so101_pick_place \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick the red cube and place it in the target zone"
```

---

### Step 10: Train Your Policy
```bash
# Train SmolVLA (recommended for multi-task)
lerobot-train \
  --policy=smolvla \
  --dataset.repo_id=YOUR_HF_USERNAME/so101_pick_place \
  --training.num_epochs=100 \
  --policy.push_to_hub=false \
  --output_dir=outputs/smolvla_hackathon

# Train ACT (faster, great for single tasks)
lerobot-train \
  --policy=act \
  --dataset.repo_id=YOUR_HF_USERNAME/so101_pick_place \
  --training.num_epochs=100 \
  --policy.push_to_hub=false
```

---

### Step 11: Run Evaluation
```bash
lerobot-eval \
  --policy.path=outputs/smolvla_hackathon/checkpoints/last \
  --env.type=gym_mujoco_so101 \
  --eval.n_episodes=20
```

---

## 11. Architecture: How Everything Connects

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR COMPUTER (Docker Container)              │
│                                                                   │
│  ┌──────────────┐    Images     ┌──────────────────────────┐    │
│  │  MuJoCo      │ ──────────►  │  AI Policy (ACT /        │    │
│  │  Simulator   │              │  SmolVLA / Diffusion)     │    │
│  │  (Virtual    │ ◄──────────  │                           │    │
│  │   SO-101)    │  Joint cmds  │  Input: images + joints   │    │
│  └──────────────┘              │  Output: target joints    │    │
│         │                      └──────────────────────────┘    │
│         │                                 ▲                      │
│         ▼                                 │                      │
│  ┌──────────────┐              ┌──────────────────────────┐    │
│  │  ROS 2       │◄────────────►│  Python Control Script   │    │
│  │  Middleware  │              │  (LeRobot)                │    │
│  └──────────────┘              └──────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
           │ Final Round only
           ▼
┌──────────────────────┐
│  Real SO-101 Robot   │
│  (Physical hardware) │
└──────────────────────┘
```

### Data Flow in Detail
1. **MuJoCo** simulates physics — renders camera images, computes joint positions
2. **ROS 2** transports this data as messages (topics)
3. **LeRobot** reads the observations, runs them through your trained AI model
4. The model outputs a **chunk of joint angle targets**
5. LeRobot sends those back through ROS 2 to MuJoCo
6. MuJoCo moves the simulated robot arm accordingly
7. This loop runs at ~30 Hz (30 times per second)

---

## 12. Strategy: Which Model Should You Use?

### For Semifinal (Simulation, 3 Tasks Independently)

| Task | Recommended Model | Why |
|---|---|---|
| Task 1: Pick & Place | **ACT** or **SmolVLA** | Precise, repeatable motion. ACT is fast to train. |
| Task 2: Charger Plugging | **Diffusion Policy** or **SmolVLA** | Complex alignment; diffusion handles multimodal poses better |
| Task 3: Liquid Pouring | **Diffusion Policy** | Continuous, smooth motion; pour angle control |
| Task 4: Humanoid Walking | **GR00T N1** or **Custom RL** | Humanoid-specific; MediaPipe pose mapping needed |

### For Final Round (All Tasks in Sequence, No Human)
Use **SmolVLA** as the unified backbone — one model, language-conditioned, that can be instructed to do each sub-task:
- "Pick up the cube and place it in the box"
- "Grab the cable and plug it into the socket"
- "Pour 50ml of liquid into the cup"

This language conditioning allows one model to handle all three tasks.

### Recommended Hackathon Strategy
```
Week 1: Set up Docker + collect 50 demos per task in simulation
Week 2: Train ACT per task, evaluate and improve data quality
Week 3: Fine-tune SmolVLA on combined multi-task dataset
Week 4: Test final round sequence, handle edge cases
```

---

## 13. Key Resources, Links & Repositories

### Competition
| Resource | Link |
|---|---|
| Competition Repo | https://github.com/vishal-finch/physical-ai-challange-2026 |
| Contact | raushan@xprobotics.ai |

### LeRobot
| Resource | Link |
|---|---|
| GitHub | https://github.com/huggingface/lerobot |
| Documentation | https://huggingface.co/docs/lerobot |
| SO-101 Docs | https://huggingface.co/docs/lerobot/so101 |
| Pretrained Models | https://huggingface.co/lerobot |
| Datasets | https://huggingface.co/datasets?other=LeRobot |
| Colab Training Notebook | https://colab.research.google.com/github/huggingface/lerobot/blob/main/examples/train_act.ipynb |

### AI Models
| Model | HuggingFace | Paper / GitHub |
|---|---|---|
| SmolVLA | https://huggingface.co/lerobot/smolvla_base | https://arxiv.org/abs/2506.01844 |
| ACT | https://huggingface.co/lerobot/act_aloha_sim_insertion_human | https://github.com/tonyzhaozh/act |
| Diffusion Policy | https://huggingface.co/lerobot/diffusion_pusht | https://diffusion-policy.cs.columbia.edu |
| Pi0 (OpenPi) | https://huggingface.co/physical-intelligence/pi0 | https://github.com/Physical-Intelligence/openpi |
| GR00T N1 | https://huggingface.co/nvidia/GR00T-N1-2B | https://github.com/NVIDIA/Isaac-GR00T |
| OpenVLA | https://huggingface.co/openvla/openvla-7b | https://github.com/openvla/openvla |
| RDT-1B | https://huggingface.co/robotics-diffusion-transformer/rdt-1b | https://github.com/thu-ml/RoboticsDiffusionTransformer |

### Simulators
| Tool | Link |
|---|---|
| MuJoCo | https://mujoco.org |
| MuJoCo Python | https://github.com/google-deepmind/mujoco |
| Webots | https://cyberbotics.com |
| Gazebo | https://gazebosim.org |

### Core Concepts to Study
| Topic | Resource |
|---|---|
| Action Chunking (ACT paper) | https://arxiv.org/abs/2304.13705 |
| Diffusion Policy paper | https://arxiv.org/abs/2303.04137 |
| VLA survey | https://arxiv.org/abs/2405.14093 |
| MediaPipe pose | https://developers.google.com/mediapipe/solutions/vision/pose_landmarker |
| URDF tutorial | https://wiki.ros.org/urdf/Tutorials |
| ROS 2 beginner tutorials | https://docs.ros.org/en/humble/Tutorials.html |
| Docker tutorial | https://docs.docker.com/get-started |
| LeRobot training tutorial | https://huggingface.co/docs/lerobot/il_robots |

### Awesome Lists (Curated Research)
| Resource | Link |
|---|---|
| Awesome Physical AI | https://github.com/keon/awesome-physical-ai |
| Awesome VLA Models | https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln |
| LeRobot Community Datasets | https://huggingface.co/datasets?search=so101 |

---

## Quick Reference Cheatsheet

```bash
# Enter Docker container
docker compose exec dev bash

# Build ROS workspace
colcon build && source install/setup.bash

# Train ACT
lerobot-train --policy=act --dataset.repo_id=USER/DATASET

# Train SmolVLA
lerobot-train --policy=smolvla --dataset.repo_id=USER/DATASET

# Evaluate a policy
lerobot-eval --policy.path=PATH/TO/CHECKPOINT --eval.n_episodes=20

# Record teleoperation episodes
lerobot-record --robot.type=so101_follower --dataset.num_episodes=50

# Push model to HuggingFace
huggingface-cli upload USER/MODEL_NAME PATH/TO/MODEL
```

---

*This guide was prepared for the Physical AI Hackathon 2026. Good luck! 🤖*

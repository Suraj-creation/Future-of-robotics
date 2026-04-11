# 🔧 Physical AI Hackathon 2026 — Complete Ubuntu Setup Guide
> **Based on the ACTUAL official repo: github.com/vishal-finch/physical-ai-challenge-2026**
> Verified against: Dockerfile, docker-compose.yml, README.md, and the installation PDF

---

## ⚠️ CRITICAL: READ THIS FIRST — PDF vs REAL REPO DIFFERENCES

The installation PDF you received has **several errors** that will cause you to fail if you follow it blindly. Here is what the PDF says vs. what the actual repo contains:

| Thing | PDF Says (WRONG) | Actual Repo (CORRECT) |
|---|---|---|
| **Robot** | SO-101 / LeRobot arm | **UR5 (Universal Robots 5)** |
| **Simulator** | MuJoCo | **Gazebo Harmonic** |
| **ROS version** | ROS 2 Humble | **ROS 2 Jazzy** |
| **Container name** | `lerobot_hackathon_env` | **`ur5_hackathon`** |
| **Docker image** | (local build implied) | **`vishalrobotics/ur5-hackathon-env:latest`** (pre-built, just pull) |
| **Workspace path** | `~/workspace` | **`/ur5_ws`** (inside container) |
| **Launch command** | `python3 src/so101_mujoco/scripts/...` | **`ros2 launch ur5_moveit simulated_robot.launch.py`** |
| **Clone URL typo** | `visha;-finch/...` (semicolon!) | **`vishal-finch/physical-ai-challenge-2026`** |

**The PDF describes an older/different version of the hackathon. Everything in this guide is based on the live official repo.**

---

## Table of Contents

1. [What You're Setting Up (Overview)](#1-what-youre-setting-up-overview)
2. [System Requirements Check](#2-system-requirements-check)
3. [PHASE 1 — Host Machine Setup (Ubuntu)](#3-phase-1--host-machine-setup-ubuntu)
4. [PHASE 2 — Clone the Repository](#4-phase-2--clone-the-repository)
5. [PHASE 3 — Docker Setup](#5-phase-3--docker-setup)
6. [PHASE 4 — NVIDIA GPU Setup (If You Have a GPU)](#6-phase-4--nvidia-gpu-setup-if-you-have-a-gpu)
7. [PHASE 5 — Launch the Container](#7-phase-5--launch-the-container)
8. [PHASE 6 — Inside the Container (Full Setup)](#8-phase-6--inside-the-container-full-setup)
9. [PHASE 7 — Running the Simulation](#9-phase-7--running-the-simulation)
10. [PHASE 8 — Multi-Terminal Workflow](#10-phase-8--multi-terminal-workflow)
11. [Error Reference — Every Error You Can Hit](#11-error-reference--every-error-you-can-hit)
12. [Daily Workflow Cheatsheet](#12-daily-workflow-cheatsheet)
13. [Folder Structure Reference](#13-folder-structure-reference)

---

## 1. What You're Setting Up (Overview)

```
YOUR UBUNTU MACHINE (Host)
│
│   ┌─────────────────────────────────────────────────────────────┐
│   │           Docker Container: ur5_hackathon                   │
│   │                                                             │
│   │  Base Image: osrf/ros:jazzy-desktop                         │
│   │  Pre-built Image: vishalrobotics/ur5-hackathon-env:latest   │
│   │                                                             │
│   │  Inside:                                                    │
│   │  ├── ROS 2 Jazzy                                            │
│   │  ├── Gazebo Harmonic (simulator)                            │
│   │  ├── MoveIt 2 (motion planning)                             │
│   │  ├── /ur5_ws/ (ROS workspace, already built)               │
│   │  │   └── src/                                               │
│   │  │       └── ur5_moveit/  (the hackathon ROS package)       │
│   │  └── All dependencies pre-installed                         │
│   │                                                             │
│   └─────────────────────────────────────────────────────────────┘
│
│   GUI forwarding: Container → X11 → Your screen
│   (So Gazebo and RViz windows appear on your desktop)
```

**Key mental model:** You do NOT build anything yourself. The Docker image `vishalrobotics/ur5-hackathon-env:latest` is pre-built by the organizers with everything installed. You just pull it and run it.

---

## 2. System Requirements Check

Before starting, verify your system. Open a terminal and run each check:

```bash
# Check Ubuntu version (need 20.04 or 22.04 or 24.04)
lsb_release -a
# You should see: Ubuntu 20.04 / 22.04 / 24.04

# Check architecture (must be x86_64 / amd64)
uname -m
# Must output: x86_64

# Check available disk space (need at least 15GB free for Docker image)
df -h ~
# Look at "Avail" column — need 15GB+

# Check RAM
free -h
# Recommended: 8GB+ RAM

# Check if you have NVIDIA GPU
nvidia-smi
# If this works → you have an NVIDIA GPU → follow GPU section
# If "command not found" → no NVIDIA GPU (still works, just slower)

# Check internet connection (needed to pull Docker image)
curl -I https://hub.docker.com
# Should return HTTP/2 200
```

**Minimum requirements:**

| Resource | Minimum | Recommended |
|---|---|---|
| Ubuntu | 20.04 LTS | 22.04 LTS |
| RAM | 8 GB | 16 GB+ |
| Disk | 15 GB free | 30 GB free |
| Architecture | x86_64 | x86_64 |

---

## 3. PHASE 1 — Host Machine Setup (Ubuntu)

Open a terminal (`Ctrl+Alt+T`). Run every command in this section on your **host Ubuntu machine** (not inside Docker).

### Step 1.1 — Update System

```bash
sudo apt update && sudo apt upgrade -y
```

> ⚠️ **Possible Error:** `E: Could not get lock /var/lib/apt/lists/lock`
> **Fix:** Another apt process is running. Wait 1-2 minutes and try again. Or: `sudo rm /var/lib/apt/lists/lock && sudo apt update`

### Step 1.2 — Install Git

```bash
sudo apt install git -y

# Verify
git --version
# Expected output: git version 2.x.x
```

### Step 1.3 — Install curl and wget (needed for later steps)

```bash
sudo apt install curl wget -y
```

### Step 1.4 — Install Docker Engine

```bash
# Remove any old conflicting Docker versions first
sudo apt remove docker docker-engine docker.io containerd runc -y 2>/dev/null || true

# Install prerequisites
sudo apt install ca-certificates gnupg lsb-release -y

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin -y
```

> ⚠️ **Possible Error:** `Package 'docker-ce' has no installation candidate`
> **Fix:** Your Ubuntu version may not have Docker in its repo yet. Use: `sudo apt install docker.io -y` as a fallback.

### Step 1.5 — Install Docker Compose Plugin

```bash
# Install as plugin (modern way)
sudo apt install docker-compose-plugin -y

# Verify
docker compose version
# Expected: Docker Compose version v2.x.x
```

> ⚠️ **Note:** The PDF used `docker-compose` (old separate binary with hyphen). The correct modern command is `docker compose` (space, no hyphen). Both work but the plugin is recommended.

### Step 1.6 — Enable and Start Docker

```bash
sudo systemctl enable docker
sudo systemctl start docker

# Verify Docker is running
sudo systemctl status docker
# Look for: Active: active (running)
```

### Step 1.7 — Add Your User to the Docker Group (Avoid using sudo every time)

```bash
sudo usermod -aG docker $USER
```

> ⚠️ **IMPORTANT:** This change only takes effect after you **log out and log back in** (or restart). 
> To apply immediately without logging out, run:
> ```bash
> newgrp docker
> ```
> After this, test it: `docker ps` — if it runs without sudo and without permission errors, you're good.

### Step 1.8 — Verify Docker Works

```bash
docker run hello-world
# Expected: "Hello from Docker!" message
# If this works, Docker is correctly installed
```

> ⚠️ **Possible Error:** `permission denied while trying to connect to the Docker daemon socket`
> **Fix:** You haven't logged out/in after step 1.7. Run `newgrp docker` and try again.

---

## 4. PHASE 2 — Clone the Repository

Still on your **host machine** terminal:

```bash
# Navigate to your home directory first
cd ~

# Clone the repository (NOTE: correct URL — no semicolon like the PDF had)
git clone https://github.com/vishal-finch/physical-ai-challenge-2026.git

# Verify it cloned correctly
ls physical-ai-challenge-2026/
# Expected output: Dockerfile  docker-compose.yml  docker-compose.windows.yml  README.md  ur5_ws/
```

> ⚠️ **Possible Error:** `fatal: repository 'https://github.com/...' not found`
> **Fix:** Check your internet connection. Also verify the URL exactly — the PDF has `visha;-finch` with a semicolon, which is wrong. Correct URL is `vishal-finch` with a hyphen.

### Understanding the Repository Structure

```
physical-ai-challenge-2026/           ← Root of the repo (on your machine)
│
├── Dockerfile                        ← Instructions to build the Docker image
│                                       (you DON'T run this manually — it's already been built)
│
├── docker-compose.yml                ← The file that tells Docker how to run the container
│                                       This is what you USE (Linux/Windows 11)
│
├── docker-compose.windows.yml        ← Alternative for Windows 10 with VcXsrv
│
├── README.md                         ← Official instructions
│
└── ur5_ws/                           ← The ROS 2 workspace source code
    └── src/
        └── ur5_moveit/               ← The hackathon ROS 2 package
            ├── launch/               ← Launch files (how to start the simulation)
            ├── scripts/              ← Python nodes (add_scene_objects, etc.)
            └── ...
```

> **Key insight:** The `ur5_ws/src` folder on your machine gets **copied into the Docker image** during build. Since the image is pre-built by organizers, your local `ur5_ws/src` folder is there for reference — the actual workspace is already baked into the container at `/ur5_ws/`.

---

## 5. PHASE 3 — Docker Setup

### Step 3.1 — Allow Docker to Open GUI Windows on Your Screen

This step is critical. Without it, Gazebo and RViz won't be able to display their windows.

```bash
# Run on your HOST machine (not inside Docker)
xhost +local:root
```

> ⚠️ **What this does:** Grants Docker containers permission to draw windows on your X11 display (your screen). Without this, you'll get errors like `cannot connect to X server` when Gazebo tries to open.

> ⚠️ **Possible Error:** `xhost: unable to open display ""`
> **Fix:** You're running in a headless server without a display. This guide assumes you have a desktop (GUI) Ubuntu installation. If you're on a server, you need X11 forwarding setup, which is outside this guide's scope.

> ℹ️ **Note:** You need to run `xhost +local:root` every time you restart your computer. Add it to `~/.bashrc` to automate it:
> ```bash
> echo "xhost +local:root > /dev/null 2>&1" >> ~/.bashrc
> ```

---

## 6. PHASE 4 — NVIDIA GPU Setup (If You Have a GPU)

**Skip this entire section if you don't have an NVIDIA GPU.**

If you do have an NVIDIA GPU (verified by `nvidia-smi` working), you need the NVIDIA Container Toolkit so the Docker container can use your GPU.

```bash
# Step 1: Add NVIDIA GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

# Step 2: Add NVIDIA repository
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Step 3: Install the toolkit
sudo apt update
sudo apt install nvidia-container-toolkit -y

# Step 4: Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Step 5: Restart Docker to apply
sudo systemctl restart docker

# Step 6: Verify GPU is accessible inside Docker
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
# Expected: Should show your GPU model and driver version
```

> ⚠️ **Possible Error:** `distribution` variable is empty → your `/etc/os-release` may be missing. Fix: manually set `distribution=ubuntu22.04` (or whatever your version is).

> ⚠️ **Possible Error:** `unknown flag: --gpus` when running docker commands → Docker version is too old. The `--gpus` flag requires Docker 19.03+. Check with `docker --version`.

---

## 7. PHASE 5 — Launch the Container

Navigate to the repository root and start the container:

```bash
# Step 1: Navigate to the repo
cd ~/physical-ai-challenge-2026

# Step 2: Make sure xhost is allowed (if you haven't done it already)
xhost +local:root

# Step 3: Pull and start the container
# This downloads the pre-built image from Docker Hub the first time (~3-5 GB, may take 5-10 min)
docker compose up -d
```

> ⚠️ **What `-d` means:** "Detached mode" — the container runs in the background so your terminal stays free.

> ⚠️ **Possible Error:** `Error response from daemon: pull access denied`
> **Fix:** The image might require login. Try: `docker login` with your Docker Hub account.

> ⚠️ **Possible Error:** `Error response from daemon: Cannot connect to the Docker daemon`
> **Fix:** Docker isn't running. Run: `sudo systemctl start docker`

> ⚠️ **Possible Error (first time):** Takes very long to start → it's downloading the image (~3-5 GB). This is normal. Wait for it.

### Step 5.1 — Verify the Container Is Running

```bash
docker ps
```

Expected output:
```
CONTAINER ID   IMAGE                                      COMMAND   CREATED         STATUS        PORTS     NAMES
abc123def456   vishalrobotics/ur5-hackathon-env:latest   "bash"    2 minutes ago   Up 2 minutes            ur5_hackathon
```

> ⚠️ If the container is NOT in the list, check what happened:
> ```bash
> docker ps -a   # Shows all containers including stopped ones
> docker logs ur5_hackathon   # Shows why it crashed
> ```

---

## 8. PHASE 6 — Inside the Container (Full Setup)

### Step 6.1 — Enter the Container

Open a terminal and enter the running container:

```bash
docker exec -it ur5_hackathon bash
```

> ⚠️ **What this does:** Opens an interactive bash shell INSIDE the container. Your prompt will change to something like: `root@<container_id>:/ur5_ws#`
> 
> Everything from this point forward in this section happens INSIDE the container unless stated otherwise.

### Step 6.2 — Understand Where You Are

```bash
# You should be inside the container. Verify:
pwd
# Expected: /ur5_ws

# See the workspace structure
ls /ur5_ws/
# Expected: build/  install/  log/  src/

ls /ur5_ws/src/
# Expected: ur5_moveit/ (the main hackathon ROS package)

# Check ROS is available
ros2 --version
# Expected: ros2 cli version: 0.18.x (jazzy)
```

### Step 6.3 — Source the ROS Environment

The Docker image's `.bashrc` should auto-source ROS, but verify and run manually to be safe:

```bash
# Source ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# Source the built workspace
source /ur5_ws/install/setup.bash

# Verify both work
echo $ROS_DISTRO
# Expected: jazzy

# Verify the UR5 package is found
ros2 pkg list | grep ur5
# Expected: ur5_moveit (and possibly others)
```

> ⚠️ **Possible Error:** `bash: /ur5_ws/install/setup.bash: No such file or directory`
> **Fix:** The workspace hasn't been built yet (rare — it should be pre-built in the image). Run:
> ```bash
> cd /ur5_ws
> source /opt/ros/jazzy/setup.bash
> colcon build --symlink-install
> source install/setup.bash
> ```

> ⚠️ **Possible Error:** `colcon: command not found`
> **Fix:** `pip install colcon-common-extensions` or `sudo apt install python3-colcon-common-extensions -y`

### Step 6.4 — Verify the Simulation Packages Are Available

```bash
# Check that the key packages exist
ros2 pkg list | grep -E "ur5|moveit|gz|gazebo"
# Expected output (should include):
# ur5_moveit
# moveit_ros_planning_interface
# ros_gz_sim
# gz_ros2_control
```

### Step 6.5 — Make Environment Persistent (Optional but Recommended)

So you don't have to source every time you open a new shell inside the container:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source /ur5_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 9. PHASE 7 — Running the Simulation

> ⚠️ **IMPORTANT:** Everything below runs INSIDE the container. Make sure you're in the container terminal (prompt shows `root@<id>:/ur5_ws#`).

### Step 7.1 — Terminal 1: Launch the Full Simulation

This single command starts RViz (3D visualization), MoveIt (motion planning), and Gazebo (physics simulation) all at once:

```bash
# Make sure you're inside the container
# Make sure you've sourced the environment (Step 6.3)

ros2 launch ur5_moveit simulated_robot.launch.py
```

> ⚠️ **What to expect:** 
> - Several ROS nodes will start printing logs
> - After 10-30 seconds, **two windows** should appear on your screen:
>   1. **RViz** — shows the UR5 robot arm in 3D
>   2. **Gazebo** — shows the physics simulation
> - There will be a lot of log output — this is normal

> ⚠️ **Possible Error:** `[ERROR] Could not connect to display :0`
> **Fix:** You forgot to run `xhost +local:root` on the HOST machine. Open a NEW terminal on your host (not in Docker) and run it, then try again.

> ⚠️ **Possible Error:** `[gz] process has died`
> **Fix:** Gazebo crashed. Common causes:
> 1. Not enough VRAM/RAM — close other applications
> 2. Missing display — check xhost
> 3. OpenGL issues — try: `export LIBGL_ALWAYS_SOFTWARE=1` before launching

> ⚠️ **Possible Error:** `Package 'ur5_moveit' not found`
> **Fix:** Environment not sourced. Run: `source /opt/ros/jazzy/setup.bash && source /ur5_ws/install/setup.bash`

> ⚠️ **KEEP THIS TERMINAL OPEN.** Do not close it. The simulation runs in this terminal.

### Step 7.2 — Terminal 2: Open a Second Shell in the Container

Open a **NEW terminal** on your host machine (don't close Terminal 1). Then:

```bash
# On HOST machine — open a new shell inside the SAME running container
docker exec -it ur5_hackathon bash

# Inside the new shell — source the environment
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
```

> ⚠️ **Why a new terminal?** Terminal 1 is occupied by the running simulation. Any additional commands need their own terminal session inside the same container.

### Step 7.3 — Terminal 2: Add Scene Objects (Tables and Cubes)

```bash
# This spawns the tables and pick-place objects into the simulation
ros2 run ur5_moveit add_scene_objects
```

> ⚠️ **What to expect:** In RViz, you should see tables and colored cubes appear in the 3D view.

> ⚠️ **Possible Error:** `executable 'add_scene_objects' not found`
> **Fix:** The workspace isn't sourced. Make sure you ran both source commands above.

### Step 7.4 — Terminal 2: Test with Dynamic Obstacles (Optional)

```bash
# Add a cylindrical obstacle to test obstacle avoidance
ros2 run ur5_moveit insert_obstacle --x 0.3 --y -0.2 --z 0.5 --radius 0.04 --height 0.25

# Remove the obstacle
ros2 run ur5_moveit insert_obstacle --name obstacle --remove
```

---

## 10. PHASE 8 — Multi-Terminal Workflow

In real development, you'll have **multiple terminals** open, each connected to the container. Here's the standard workflow:

### How to Open Multiple Terminals Inside the Container

Every time you need a new terminal session inside the container, on your **HOST machine** run:

```bash
docker exec -it ur5_hackathon bash
```

Then inside each new shell, always run:
```bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
```

### Recommended Terminal Layout

| Terminal | Purpose | What's Running |
|---|---|---|
| **Terminal 1** | Simulation | `ros2 launch ur5_moveit simulated_robot.launch.py` |
| **Terminal 2** | Scene setup / commands | `ros2 run ur5_moveit add_scene_objects` |
| **Terminal 3** | Your AI/control code | Your Python scripts, ROS nodes |
| **Terminal 4** | Monitoring | `ros2 topic list`, `ros2 topic echo /...` |

### Using tmux for Multi-Window Inside Container (Highly Recommended)

Instead of opening multiple host terminals, use `tmux` inside the container to manage multiple panes:

```bash
# Install tmux (if not already in container)
apt-get install tmux -y

# Start tmux
tmux

# Create a horizontal split
Ctrl+B then %

# Create a vertical split
Ctrl+B then "

# Switch between panes
Ctrl+B then arrow keys

# Detach from tmux (container keeps running)
Ctrl+B then d

# Re-attach to existing tmux session
tmux attach
```

---

## 11. Error Reference — Every Error You Can Hit

### Docker Errors

| Error | Cause | Fix |
|---|---|---|
| `permission denied while trying to connect to the Docker daemon` | Not in docker group | `newgrp docker` or log out/in |
| `Cannot connect to the Docker daemon` | Docker not running | `sudo systemctl start docker` |
| `Error response from daemon: pull access denied` | Private image | `docker login` first |
| `port is already allocated` | Port conflict | Another container using same port. `docker ps` to see, then `docker stop <name>` |
| `no space left on device` | Disk full | `docker system prune -a` to clean up old images |
| `Error: No such container: ur5_hackathon` | Container not started | `cd ~/physical-ai-challenge-2026 && docker compose up -d` |

### ROS 2 Errors

| Error | Cause | Fix |
|---|---|---|
| `ros2: command not found` | ROS not sourced | `source /opt/ros/jazzy/setup.bash` |
| `Package 'ur5_moveit' not found` | Workspace not sourced | `source /ur5_ws/install/setup.bash` |
| `executable not found` | Same as above | Source both setup.bash files |
| `[ERROR] TF_REPEATED_DATA` | Timing warning | Usually harmless, can ignore |
| `Could not load controller` | ros2_control issue | Check that all launch dependencies started |

### GUI / Display Errors

| Error | Cause | Fix |
|---|---|---|
| `cannot connect to X server :0` | xhost not set | On HOST: `xhost +local:root` |
| `[gz] process has died` | Gazebo crash | Try `export LIBGL_ALWAYS_SOFTWARE=1` inside container |
| `OpenGL ES profile not supported` | GPU driver issue | Run `export MESA_GL_VERSION_OVERRIDE=3.3` inside container |
| No windows appear at all | DISPLAY not set | Check `echo $DISPLAY` in container. Should show `:0` or `:1` |

### Build Errors (If You Rebuild)

| Error | Cause | Fix |
|---|---|---|
| `CMake Error: could not find package 'moveit_ros_planning_interface'` | Missing ROS dep | `rosdep install --from-paths src --ignore-src -r -y` |
| `colcon: command not found` | Not installed | `sudo apt install python3-colcon-common-extensions -y` |
| `No module named 'catkin_pkg'` | Python package missing | `/opt/lerobot_venv/bin/python3 -m pip install catkin_pkg` |

---

## 12. Daily Workflow Cheatsheet

Once everything is set up, your daily routine is:

```bash
# ── ON HOST MACHINE ────────────────────────────────────────────

# 1. Allow GUI (run once per boot)
xhost +local:root

# 2. Start the container (from repo folder)
cd ~/physical-ai-challenge-2026
docker compose up -d

# 3. Verify it's running
docker ps
# Should show ur5_hackathon

# ── TERMINAL 1 (inside container) ──────────────────────────────
docker exec -it ur5_hackathon bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 launch ur5_moveit simulated_robot.launch.py
# → Keep this open. Gazebo + RViz open on your screen.

# ── TERMINAL 2 (inside container — new terminal) ───────────────
docker exec -it ur5_hackathon bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 run ur5_moveit add_scene_objects
# → Tables and cubes appear in simulation.

# ── TERMINAL 3 (inside container — your code) ──────────────────
docker exec -it ur5_hackathon bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
python3 your_pick_place_script.py

# ── STOPPING (when done for the day) ───────────────────────────
# Option 1: Stop just the container (keeps it for next time, fast restart)
docker stop ur5_hackathon

# Option 2: Stop and remove (clean slate, need docker compose up -d next time)
docker compose down
```

---

## 13. Folder Structure Reference

### On Your Host Machine

```
~/physical-ai-challenge-2026/
├── Dockerfile                  ← Image build recipe (don't modify)
├── docker-compose.yml          ← Container config (Linux/Win11)
├── docker-compose.windows.yml  ← Container config (Windows 10)
├── README.md                   ← Official quick start
└── ur5_ws/
    └── src/
        └── ur5_moveit/
            ├── CMakeLists.txt
            ├── package.xml
            ├── launch/
            │   └── simulated_robot.launch.py   ← Main launch file
            └── scripts/
                ├── add_scene_objects.py         ← Spawns tables/cubes
                └── insert_obstacle.py           ← Adds/removes obstacles
```

### Inside the Docker Container

```
/
├── opt/
│   └── ros/
│       └── jazzy/              ← ROS 2 Jazzy installation
│           └── setup.bash      ← Source this first
│
└── ur5_ws/                     ← Your ROS workspace (already built)
    ├── src/                    ← Source code (copy of the repo's ur5_ws/src)
    │   └── ur5_moveit/         ← Main package
    ├── build/                  ← Compiled files (auto-generated)
    ├── install/                ← Installed files
    │   └── setup.bash          ← Source this second
    └── log/                    ← Build and run logs
```

### Key ROS 2 Commands for Exploration

```bash
# Inside container, after sourcing:

# List all running ROS nodes
ros2 node list

# List all active topics
ros2 topic list

# See what's being published on a topic (e.g., joint states)
ros2 topic echo /joint_states

# List all available services
ros2 service list

# Get info about a package
ros2 pkg prefix ur5_moveit

# Run a node directly
ros2 run <package_name> <executable_name>

# Launch a launch file
ros2 launch <package_name> <launch_file.py>

# Check message type of a topic
ros2 topic info /joint_states
```

---

## Quick Sanity Check — Did Everything Work?

Run through this checklist after setup:

```
[ ] docker --version          → Shows version (not "command not found")
[ ] docker compose version    → Shows version
[ ] docker ps                 → Shows ur5_hackathon container as "Up"
[ ] (inside container) echo $ROS_DISTRO   → Shows "jazzy"
[ ] (inside container) ros2 pkg list | grep ur5_moveit  → Shows the package
[ ] Gazebo window opens when you launch simulated_robot.launch.py
[ ] RViz window opens with the UR5 arm visible
[ ] Tables and cubes appear after running add_scene_objects
```

If all boxes are checked, your environment is fully working. 🎉

---

## What to Code Next

The README says your challenge is to build upon the provided environment:

1. **Attach cameras/sensors** to the UR5 in the Gazebo simulation
2. **Develop perception logic** (object detection — use YOLOv11)
3. **Write your own IK/pick-and-place orchestrator** using MoveIt 2

```python
# Minimal Python example to start — move the UR5 to a position via MoveIt 2
# Save as /ur5_ws/src/ur5_moveit/scripts/my_pick_place.py
# Run: ros2 run ur5_moveit my_pick_place

import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy

class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')
        self.robot = MoveItPy(node_name="moveit_py")
        self.arm = self.robot.get_planning_component("ur_manipulator")
        self.get_logger().info("Pick and Place Node Ready!")

def main():
    rclpy.init()
    node = PickPlaceNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

*This guide is based on the verified contents of `github.com/vishal-finch/physical-ai-challenge-2026` as of April 2026. If the repo is updated, re-check the Dockerfile and docker-compose.yml for any changes.*

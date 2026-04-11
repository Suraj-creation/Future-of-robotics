# Physical AI Challenge 2026 — Canonical Ubuntu Setup Guide

**Source of truth:** `github.com/vishal-finch/physical-ai-challenge-2026`  
**Last verified:** April 2026  
**This is the single canonical setup document. All other guides are superseded by this one.**

---

## ⚠️ Critical Corrections vs. the PDF and Earlier Guides

The installation PDF distributed at the event and the earlier setup guides contain errors that will break your setup. Here is the complete correction table before you touch a single command:

| Item | PDF / Old Guide Says | Actual Repo (This Guide) |
|---|---|---|
| Robot | SO-101 / LeRobot arm | **UR5 (Universal Robots 5)** |
| Simulator | MuJoCo | **Gazebo Harmonic** |
| ROS version | ROS 2 Humble | **ROS 2 Jazzy** |
| Container name | `lerobot_hackathon_env` | **`ur5_hackathon`** |
| Docker image | local build implied | **`vishalrobotics/ur5-hackathon-env:latest`** (pre-built, just pull) |
| Workspace path (inside container) | `~/workspace` | **`/ur5_ws`** |
| Main launch command | `python3 src/so101_mujoco/scripts/...` | **`ros2 launch ur5_moveit simulated_robot.launch.py`** |
| Clone URL | `visha;-finch/...` (semicolon typo!) | **`vishal-finch/physical-ai-challenge-2026`** |
| Script location | `scripts/` folder | **`ur5_moveit/` Python module folder** |
| MoveIt planning group | (wrong name used) | **`ur_manipulator`** (from SRDF) |
| LeRobot venv fix | referenced in old guide | **Irrelevant — this is a UR5 repo, not LeRobot** |

---

## Table of Contents

1. [Mental Model — What You Are Building](#1-mental-model)
2. [System Requirements](#2-system-requirements)
3. [Terminal Map — How You Will Work](#3-terminal-map)
4. [Phase 1 — Host Machine Setup](#4-phase-1--host-machine-setup)
5. [Phase 2 — Clone the Repository](#5-phase-2--clone-the-repository)
6. [Phase 3 — Allow GUI Forwarding](#6-phase-3--allow-gui-forwarding)
7. [Phase 4 — NVIDIA GPU Setup (Optional)](#7-phase-4--nvidia-gpu-setup-optional)
8. [Phase 5 — Pull and Start the Container](#8-phase-5--pull-and-start-the-container)
9. [Phase 6 — Verify the Container Environment](#9-phase-6--verify-the-container-environment)
10. [Phase 7 — Run the Full Simulation Stack](#10-phase-7--run-the-full-simulation-stack)
11. [Phase 8 — Add Scene Objects](#11-phase-8--add-scene-objects)
12. [Phase 9 — Optional Obstacle Testing](#12-phase-9--optional-obstacle-testing)
13. [Workflow B — Build a Local Image](#13-workflow-b--build-a-local-image-optional)
14. [Workflow C — Bind-Mount Dev Container](#14-workflow-c--bind-mount-dev-container-for-code-editing)
15. [Workflow D — Full Workspace From Scratch Inside Docker](#15-workflow-d--full-workspace-from-scratch-inside-docker)
16. [Correct Repo and Package Layout](#16-correct-repo-and-package-layout)
17. [Writing Your Own Code — Correct Patterns](#17-writing-your-own-code--correct-patterns)
18. [Comprehensive Error Reference](#18-comprehensive-error-reference)
19. [Daily Workflow Cheatsheet](#19-daily-workflow-cheatsheet)
20. [Final Validation Checklist](#20-final-validation-checklist)

---

## 1. Mental Model

Before running any command, understand what you are building:

```
YOUR UBUNTU MACHINE (Host)
│
│   You run Docker commands here.
│   You do NOT run ROS commands here.
│
│   ┌───────────────────────────────────────────────────────────┐
│   │         Docker Container: ur5_hackathon                   │
│   │                                                           │
│   │   Base: vishalrobotics/ur5-hackathon-env:latest           │
│   │   (Pre-built image — you just pull it, build nothing)     │
│   │                                                           │
│   │   Inside the container:                                   │
│   │   ├── ROS 2 Jazzy (the ROS distribution)                  │
│   │   ├── Gazebo Harmonic (physics simulator)                 │
│   │   ├── MoveIt 2 (motion planning framework)                │
│   │   └── /ur5_ws/  ← pre-built ROS workspace                │
│   │       ├── src/                                            │
│   │       │   ├── ur5_description/   (robot URDF + worlds)   │
│   │       │   ├── ur5_controller/    (ros2_control config)   │
│   │       │   ├── ur5_moveit/        (main hackathon pkg)    │
│   │       │   ├── ur5_moveit_config/ (MoveIt SRDF/config)    │
│   │       │   └── hello_moveit/      (C++ example)           │
│   │       ├── build/   (compiled — do not touch)             │
│   │       ├── install/ (installed — source this)             │
│   │       └── log/                                           │
│   │                                                           │
│   └───────────────────────────────────────────────────────────┘
│
│   GUI forwarding: Container X11 → Your desktop screen
│   (Gazebo and RViz windows appear on your Ubuntu desktop)
```

**Key rule:** You do NOT build anything for the default workflow. The Docker image already has everything compiled. You pull it, run it, and start coding.

---

## 2. System Requirements

Open a terminal (`Ctrl+Alt+T`) and run these checks before you start:

```bash
# Ubuntu version — need 20.04, 22.04, or 24.04
lsb_release -a

# Architecture — must be x86_64
uname -m

# Free disk space — need at least 15 GB free (image is ~3-5 GB compressed)
df -h ~

# RAM check — 8 GB minimum, 16 GB recommended
free -h

# Check for NVIDIA GPU — if this works you have one; if not, skip the GPU phase
nvidia-smi

# Internet connectivity
curl -I https://hub.docker.com
```

**Minimum requirements table:**

| Resource | Minimum | Recommended |
|---|---|---|
| Ubuntu | 20.04 LTS | 22.04 or 24.04 LTS |
| RAM | 8 GB | 16 GB+ |
| Free disk | 15 GB | 30 GB |
| Architecture | x86_64 | x86_64 |
| Desktop | Required (X11 GUI) | Required |

> ⚠️ This guide requires a **graphical desktop Ubuntu session**. Pure SSH or headless servers will not work without additional X11 forwarding setup (not covered here).

---

## 3. Terminal Map — How You Will Work

You will use multiple terminals throughout this guide. Here is the naming convention used for every command that follows:

| Terminal | Label | Where it runs | Responsibility |
|---|---|---|---|
| Host terminal | **H1** | Your Ubuntu desktop | Docker lifecycle, git, filesystem |
| Container terminal 1 | **C1** | Inside `ur5_hackathon` container | Main simulation launch — keep open |
| Container terminal 2 | **C2** | Inside `ur5_hackathon` container | Scene setup and helper nodes |
| Container terminal 3 | **C3** | Inside `ur5_hackathon` container | Diagnostics, obstacles, your code |

**Rules:**
- Docker commands (`docker`, `docker compose`) always run in **H1**.
- ROS commands (`ros2`, `colcon`) always run in **C1/C2/C3** (inside container).
- Every new container shell you open must source the ROS environment before use (exact command shown each time).
- Keep C1 running the simulation — never close it while you need the sim.

---

## 4. Phase 1 — Host Machine Setup

All commands in this phase run in **H1** (your Ubuntu desktop terminal).

### Step 1.1 — Update the system

```bash
sudo apt update && sudo apt upgrade -y
```

> **If you see:** `E: Could not get lock /var/lib/apt/lists/lock`  
> **Fix:** Another apt process is running. Wait 1-2 minutes, then retry. Or force-remove the lock: `sudo rm /var/lib/apt/lists/lock && sudo apt update`

### Step 1.2 — Install all required host packages in one command

```bash
sudo apt install -y \
  git \
  curl \
  wget \
  ca-certificates \
  gnupg \
  lsb-release \
  x11-xserver-utils \
  mesa-utils
```

> **What each package does:**
> - `git` — clone the repo
> - `curl`, `wget` — download files and GPG keys
> - `ca-certificates`, `gnupg`, `lsb-release` — needed for adding Docker's apt repo securely
> - `x11-xserver-utils` — provides `xhost` for GUI forwarding
> - `mesa-utils` — provides `glxinfo` for OpenGL diagnostics

### Step 1.3 — Install Docker Engine

```bash
# Remove any old conflicting Docker versions
sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker's official apt repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
```

> **If you see:** `Package 'docker-ce' has no installation candidate`  
> **Fix:** Your Ubuntu version may not yet have a Docker CE entry. Use the fallback: `sudo apt install -y docker.io`

### Step 1.4 — Install Docker Compose plugin

```bash
sudo apt install -y docker-compose-plugin
```

> ⚠️ **Important naming note:** The modern command is `docker compose` (space, no hyphen). The old separate binary `docker-compose` (hyphen) still exists on some systems but is deprecated. This guide uses `docker compose` throughout.

### Step 1.5 — Enable and start Docker

```bash
sudo systemctl enable docker
sudo systemctl start docker

# Verify it is running
sudo systemctl status docker
# Look for: Active: active (running)
```

### Step 1.6 — Add your user to the Docker group

This lets you run `docker` commands without `sudo` every time.

```bash
sudo usermod -aG docker "$USER"

# Apply the group change immediately without logging out
newgrp docker
```

> ⚠️ The `usermod` change only fully takes effect after you log out and back in. The `newgrp docker` command applies it to your current terminal session only. If you open a new terminal and get permission errors, run `newgrp docker` in that new terminal too, or log out/in once.

### Step 1.7 — Verify Docker works end-to-end

```bash
docker run hello-world
# Expected output: "Hello from Docker!" and a description of what just happened
```

```bash
docker compose version
# Expected output: Docker Compose version v2.x.x
```

```bash
git --version
# Expected output: git version 2.x.x
```

If all three commands return expected output, your host is ready.

---

## 5. Phase 2 — Clone the Repository

Still in **H1**:

```bash
# Go to your home directory (or wherever you want the project)
cd ~

# Create a project folder to keep things organized
mkdir -p ~/Robothon
cd ~/Robothon

# Clone the repository
# IMPORTANT: The URL uses a hyphen in "vishal-finch", not a semicolon
git clone https://github.com/vishal-finch/physical-ai-challenge-2026.git

# Enter the repo
cd physical-ai-challenge-2026

# Verify the clone succeeded — you should see all of these
ls
# Expected: Dockerfile  docker-compose.yml  docker-compose.windows.yml  README.md  ur5_ws
```

> **If you see:** `fatal: repository not found`  
> **Fix 1:** Check your internet connection.  
> **Fix 2:** The PDF has `visha;-finch` with a semicolon — that is wrong. The correct URL is `vishal-finch` with a hyphen.

```bash
# Verify the workspace source packages are present
ls ur5_ws/src/
# Expected: hello_moveit  ur5_controller  ur5_description  ur5_moveit  ur5_moveit_config
```

> If `ur5_ws/src/` shows those 5 packages, the clone is complete and correct.

### Understanding the repo layout

```
~/Robothon/physical-ai-challenge-2026/
│
├── Dockerfile                  ← Instructions used to build the Docker image
│                                  You do NOT run this manually for the default workflow.
│                                  The image is already built and hosted on Docker Hub.
│
├── docker-compose.yml          ← Tells Docker how to run the container on Linux / Windows 11
│                                  This is the file you USE with `docker compose up`
│
├── docker-compose.windows.yml  ← Alternative for Windows 10 with VcXsrv display server
│
├── README.md                   ← Official quick start from organizers
│
└── ur5_ws/
    └── src/                    ← ROS 2 source packages
        ├── hello_moveit/           ← Minimal C++ MoveIt example
        ├── ur5_controller/         ← ros2_control launch files + controller YAML
        ├── ur5_description/        ← URDF/Xacro robot model, Gazebo worlds, meshes
        ├── ur5_moveit/             ← MAIN PACKAGE: MoveIt launch + Python helper nodes
        │   ├── launch/
        │   │   └── simulated_robot.launch.py   ← The unified all-in-one launcher
        │   ├── config/
        │   │   └── ur5_robot.srdf              ← Defines planning groups
        │   └── ur5_moveit/         ← Python module folder (not "scripts/")
        │       ├── add_scene_objects.py         ← Spawns tables and cubes
        │       └── insert_obstacle.py           ← Adds/removes cylindrical obstacles
        └── ur5_moveit_config/      ← MoveIt Setup Assistant generated config
```

---

## 6. Phase 3 — Allow GUI Forwarding

The container needs permission to draw windows (Gazebo, RViz) on your desktop screen. This is done through the X11 display server.

In **H1**:

```bash
# First verify your DISPLAY variable is set (it should be :0 or :1 on a desktop)
echo "$DISPLAY"
# Expected output: :0   (or :1, or similar — must NOT be empty)
```

> **If `$DISPLAY` is empty:**  
> Run `export DISPLAY=:0` and then continue.

```bash
# Grant Docker containers permission to access your X11 display
xhost +local:root
```

> **What this command does:** It tells your X11 display server to allow the root user (which Docker containers run as) to open windows on your screen. Without this, Gazebo and RViz will crash with `cannot connect to X server`.

> ⚠️ **This permission resets on every reboot.** You must run `xhost +local:root` again each time you restart your computer before launching the container. To automate this permanently, add it to your shell startup file:

```bash
# Add to ~/.bashrc so it runs automatically every time you open a terminal
echo 'xhost +local:root > /dev/null 2>&1' >> ~/.bashrc
```

> **If you see:** `xhost: unable to open display ""`  
> **Fix:** You are not in a graphical desktop session. This setup requires a desktop (GUI) Ubuntu installation.

---

## 7. Phase 4 — NVIDIA GPU Setup (Optional)

**Skip this entire section if `nvidia-smi` did not work in the system requirements check.**

If you have an NVIDIA GPU, install the NVIDIA Container Toolkit so the Docker container can use your GPU for faster Gazebo rendering:

```bash
# Add NVIDIA's GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg

# Add NVIDIA's apt repository
distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
curl -s -L https://nvidia.github.io/libnvidia-container/"$distribution"/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker to use the NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker to apply the new runtime
sudo systemctl restart docker

# Test — should show your GPU model and driver version inside Docker
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
```

> **If `$distribution` is empty:** Manually set it: `distribution=ubuntu22.04` (replace with your actual version).  
> **If you see:** `unknown flag: --gpus` — your Docker version is too old. Docker 19.03+ is required. Check: `docker --version`

---

## 8. Phase 5 — Pull and Start the Container

You are now ready to download the pre-built Docker image and start the container.

In **H1**, from the repository root:

```bash
# Make sure you are in the repo directory
cd ~/Robothon/physical-ai-challenge-2026

# Make sure GUI forwarding is allowed (if you haven't done it this session)
xhost +local:root

# Pull the pre-built image from Docker Hub (~3-5 GB, first time takes 5-15 min)
docker compose pull
# This downloads vishalrobotics/ur5-hackathon-env:latest

# Start the container in detached (background) mode
docker compose up -d
# -d means "detached" — the container runs in the background; your terminal is free
```

> ⚠️ The first `docker compose pull` downloads 3-5 GB. This is normal. Wait for it to complete before continuing.

```bash
# Verify the container is running
docker ps --filter name=ur5_hackathon
```

Expected output:
```
CONTAINER ID   IMAGE                                      COMMAND   CREATED        STATUS       NAMES
abc123def456   vishalrobotics/ur5-hackathon-env:latest   "bash"    1 minute ago   Up 1 minute  ur5_hackathon
```

> **If the container is not in the list:**
> ```bash
> # Show all containers including stopped ones
> docker ps -a
>
> # Read the crash logs
> docker compose logs ur5-sim --tail=300
> ```

Common errors at this stage:

| Error message | Cause | Fix |
|---|---|---|
| `permission denied while trying to connect to the Docker daemon` | Not in docker group | `newgrp docker` |
| `Cannot connect to the Docker daemon` | Docker not running | `sudo systemctl start docker` |
| `Error response from daemon: pull access denied` | Private image or rate limit | `docker login` then retry |
| `Error: No such container: ur5_hackathon` | Container never started | `docker compose up -d` from repo root |
| `no space left on device` | Disk full | `docker system prune -a` to free space |

---

## 9. Phase 6 — Verify the Container Environment

Open a shell inside the running container. This becomes **C1**:

```bash
# In H1 — open an interactive bash shell inside the container
docker exec -it ur5_hackathon bash
```

Your terminal prompt will change to something like: `root@<container_id>:/ur5_ws#`

> **Everything from this point in Phase 6 runs in C1 (inside the container).**

```bash
# Verify you are in the right location
pwd
# Expected: /ur5_ws

# Verify ROS 2 Jazzy is the active distribution
echo "$ROS_DISTRO"
# Expected: jazzy

# Verify ROS 2 CLI is available
ros2 --version
# Expected: ros2 cli version: 0.18.x (jazzy)

# Check the workspace contents
ls /ur5_ws/
# Expected: build  install  log  src

# Check the source packages
ls /ur5_ws/src/
# Expected: hello_moveit  ur5_controller  ur5_description  ur5_moveit  ur5_moveit_config

# Verify the hackathon package is discoverable by ROS
ros2 pkg list | grep ur5
# Expected output includes: ur5_moveit, ur5_moveit_config, ur5_description, ur5_controller
```

### Source the ROS environment

The Docker image's `.bashrc` should auto-source these, but run them manually to be safe and confirm they work:

```bash
# Source ROS 2 Jazzy system installation
source /opt/ros/jazzy/setup.bash

# Source the compiled workspace
source /ur5_ws/install/setup.bash

# Verify
echo "$ROS_DISTRO"
# Expected: jazzy

ros2 pkg list | grep ur5_moveit
# Expected: ur5_moveit
```

> **If you see:** `bash: /ur5_ws/install/setup.bash: No such file or directory`  
> The workspace was not pre-built in the image (rare). Fix — build it manually:
> ```bash
> cd /ur5_ws
> source /opt/ros/jazzy/setup.bash
> colcon build --symlink-install
> source install/setup.bash
> ```

> **If you see:** `colcon: command not found`  
> Install it: `sudo apt install -y python3-colcon-common-extensions`

### Make the environment auto-source for every new shell (recommended)

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source /ur5_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

This means every new shell you open inside the container will automatically have the ROS environment ready. You can skip the two `source` commands in future steps if you do this.

---

## 10. Phase 7 — Run the Full Simulation Stack

> ⚠️ **You are still in C1 (inside the container).** Do not close this terminal once the simulation starts.

### Option A — Unified launch (try this first)

This single command starts Gazebo, robot_state_publisher, ros2_control, controller spawners, MoveIt move_group, and RViz all at once:

```bash
# Make sure environment is sourced (skip if you did the .bashrc step above)
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash

# Launch everything
ros2 launch ur5_moveit simulated_robot.launch.py
```

**What to expect after running this:**
1. Lots of log output — this is normal.
2. After 10-30 seconds, two windows appear on your desktop:
   - **Gazebo** — 3D physics simulation window
   - **RViz** — 3D visualization with the UR5 arm model and MoveIt plugin
3. The terminal continues printing logs — this is the simulation running.

> ⚠️ **Keep C1 open and running.** If you close it or press Ctrl+C, the simulation stops.

Common errors during launch:

| Error | Cause | Fix |
|---|---|---|
| `[ERROR] Could not connect to display :0` | xhost not set | On HOST (H1): `xhost +local:root`, then retry |
| `Package 'ur5_moveit' not found` | Environment not sourced | `source /opt/ros/jazzy/setup.bash && source /ur5_ws/install/setup.bash` |
| `[gz] process has died` | Gazebo crash (GPU/RAM) | Try: `export LIBGL_ALWAYS_SOFTWARE=1` then relaunch |
| `OpenGL ES profile not supported` | GPU driver issue | Try: `export MESA_GL_VERSION_OVERRIDE=3.3` then relaunch |
| No windows appear | DISPLAY not set in container | Check: `echo $DISPLAY` in container — should be `:0` |

### Option B — Split launch (fallback if Option A fails)

Use this if the unified launch fails due to timing or ordering issues. It decouples Gazebo startup from controller and MoveIt startup, which resolves most timing-related failures.

In **C1** — start Gazebo with the robot description:
```bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 launch ur5_description gazebo.launch.py
```

Wait until Gazebo is fully open and the robot appears. Then open **C2** from H1:
```bash
# In H1 — open second container terminal
docker exec -it ur5_hackathon bash
```

In **C2** — start the controllers:
```bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 launch ur5_controller controller.launch.py
```

Wait until you see controller spawn confirmations. Then open **C3** from H1:
```bash
# In H1 — open third container terminal
docker exec -it ur5_hackathon bash
```

In **C3** — start MoveIt and RViz:
```bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 launch ur5_moveit moveit.launch.py
```

> Why this works: It gives Gazebo time to fully initialize before controllers try to connect to it, and gives controllers time to come up before MoveIt tries to connect to them.

---

## 11. Phase 8 — Add Scene Objects

The simulation is running in C1. Now add the planning scene — tables and cubes that the robot will interact with.

Open a new container terminal. In **H1**:
```bash
docker exec -it ur5_hackathon bash
```

This is now **C2**. In C2:

```bash
# Source the environment
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash

# Spawn the scene objects into the MoveIt planning scene
ros2 run ur5_moveit add_scene_objects
```

**What to expect:** In RViz, you should see these objects appear in the 3D view:
- `ground` — floor collision plane
- `pick_table` — table where objects start
- `place_table` — table where objects are placed
- `blue_cube_1`, `blue_cube_2`, `blue_cube_3` — pick-and-place targets

> **If objects do not appear:**
> 1. Confirm move_group is running: `ros2 node list | grep move_group`
> 2. Wait 5-10 seconds and re-run the command.
> 3. In RViz, confirm the "MotionPlanning" display is enabled and visible.

> ⚠️ **Note on frames:** `add_scene_objects.py` publishes objects in `base_link` frame. Keep this in mind when your code references object positions.

---

## 12. Phase 9 — Optional Obstacle Testing

To test your obstacle avoidance logic, you can dynamically add and remove a cylindrical obstacle.

Open another container terminal. In **H1**:
```bash
docker exec -it ur5_hackathon bash
```

This is **C3**. In C3:

```bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash

# Add a cylindrical obstacle at position x=0.3, y=-0.2, z=0.5
# with radius=0.04m and height=0.25m
ros2 run ur5_moveit insert_obstacle --x 0.3 --y -0.2 --z 0.5 --radius 0.04 --height 0.25

# Remove the obstacle
ros2 run ur5_moveit insert_obstacle --name obstacle --remove
```

> ⚠️ **Frame note:** `insert_obstacle.py` uses the `world` frame, while `add_scene_objects.py` uses `base_link`. If obstacles appear offset from where you expect, this frame difference is the cause. Account for the transform between `world` and `base_link` in your code.

---

## 13. Workflow B — Build a Local Image (Optional)

Use this workflow when you want a deterministic image built from your exact local checked-out repo state — for example, if you modify the Dockerfile or want full reproducibility.

In **H1**, from the repo root:

```bash
cd ~/Robothon/physical-ai-challenge-2026

docker build -t ur5-hackathon-local .
# This will take several minutes — it compiles the full workspace inside Docker
```

Run the locally built image:

```bash
xhost +local:root

docker run --rm -it \
  --name ur5_hackathon_local \
  --net=host \
  --privileged \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ur5-hackathon-local bash
```

Then inside the container, follow the launch steps in Phase 7.

Common failures:

| Failure | Fix |
|---|---|
| `rosdep update` fails during build | Transient network issue — retry `docker build` |
| Very long first build (20+ min) | Expected — apt + rosdep + colcon layers are large |

---

## 14. Workflow C — Bind-Mount Dev Container (For Code Editing)

Use this workflow when you want edits you make to source files on your host machine to be immediately visible inside the container — without rebuilding the image.

> ⚠️ **Prerequisite:** This workflow requires the `ur5-hackathon-local` image built in **Workflow B**. You must complete Workflow B before using this workflow. If you skip Workflow B, you will get: `Unable to find image 'ur5-hackathon-local' locally`.

### Step 1 — Start the dev container with your source mounted

In **H1**, from the repo root:

```bash
cd ~/Robothon/physical-ai-challenge-2026
xhost +local:root

docker run --rm -it \
  --name ur5_hackathon_dev \
  --net=host \
  --privileged \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)/ur5_ws/src:/ur5_ws/src" \
  ur5-hackathon-local bash
```

> The `-v "$(pwd)/ur5_ws/src:/ur5_ws/src"` flag bind-mounts your local source directory into the container. When you edit a `.py` file on your host, the container sees the change immediately.

### Step 2 — Build the workspace inside the container

Because you mounted new source, you need to build:

```bash
cd /ur5_ws
source /opt/ros/jazzy/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source /ur5_ws/install/setup.bash
```

### Step 3 — Launch and operate

Follow Phase 7 and Phase 8 as normal.

> ⚠️ **Every new terminal opened into the dev container must source the environment:**
> ```bash
> source /opt/ros/jazzy/setup.bash
> source /ur5_ws/install/setup.bash
> ```

Common failures:

| Failure | Cause | Fix |
|---|---|---|
| Package not found in new terminal | Missing source | Run both source commands in that terminal |
| Old code behavior persists | Using prebuilt compose container, not dev container | Confirm you are in the dev container: `hostname` should differ |
| Python entry point not found after editing | Need rebuild | `colcon build --symlink-install` then `source install/setup.bash` |

---

## 15. Workflow D — Full Workspace From Scratch Inside Docker

Use this when you want the entire workspace cloned and built inside a clean Docker container — no host repo involved.

> ⚠️ **Known issue fixed here:** Earlier guides created a directory and then tried to clone into it, which fails because git refuses to clone into a non-empty directory. This guide clones into a new subdirectory to avoid the conflict.

### Step 1 — Start a clean ROS base container

In **H1**:

```bash
xhost +local:root

docker run --rm -it \
  --name ur5_ws_builder \
  --net=host \
  --privileged \
  -e DISPLAY="$DISPLAY" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  osrf/ros:jazzy-desktop bash
```

### Step 2 — Install all required ROS packages inside the container

```bash
apt-get update
apt-get install -y \
  git \
  python3-pip \
  python3-rosdep \
  python3-colcon-common-extensions \
  ros-jazzy-moveit \
  ros-jazzy-ros-gz \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-controller-manager \
  ros-jazzy-xacro \
  ros-jazzy-joint-state-publisher \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-tf2-ros \
  ros-jazzy-rviz2
```

### Step 3 — Clone the repository directly inside the container

```bash
# Use /work as the base directory
mkdir -p /work
cd /work

# Clone into a new subdirectory — this avoids the non-empty directory conflict
git clone https://github.com/vishal-finch/physical-ai-challenge-2026.git

# The workspace source is now at this path:
ls /work/physical-ai-challenge-2026/ur5_ws/src/
# Expected: hello_moveit  ur5_controller  ur5_description  ur5_moveit  ur5_moveit_config
```

### Step 4 — Build the workspace

```bash
cd /work/physical-ai-challenge-2026/ur5_ws
source /opt/ros/jazzy/setup.bash
rosdep init || true    # "|| true" suppresses error if already initialized
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### Step 5 — Launch the simulation

```bash
ros2 launch ur5_moveit simulated_robot.launch.py
```

Use the split launch fallback from Phase 7 if the unified launch fails.

---

## 16. Correct Repo and Package Layout

This section documents the accurate layout — earlier guides described it incorrectly in several places.

### Actual source tree

```
ur5_ws/src/
│
├── hello_moveit/
│   └── src/
│       └── hello_moveit.cpp       ← Minimal C++ MoveIt 2 example
│
├── ur5_controller/
│   ├── config/                    ← Controller YAML definitions
│   └── launch/
│       └── controller.launch.py   ← Starts ros2_control + spawns controllers
│
├── ur5_description/
│   ├── config/                    ← Controller config for description
│   ├── launch/
│   │   └── gazebo.launch.py       ← Starts Gazebo + loads robot URDF
│   ├── urdf/                      ← Xacro/URDF robot model files
│   ├── worlds/                    ← Gazebo world SDF files
│   └── meshes/
│       ├── collision/
│       │   ├── rg2/               ← Gripper collision meshes (included, no manual download)
│       │   └── ur5/               ← UR5 arm collision meshes (included)
│       └── visual/
│           ├── rg2/               ← Gripper visual meshes (included)
│           └── ur5/               ← UR5 arm visual meshes (included)
│
├── ur5_moveit/                    ← MAIN HACKATHON PACKAGE
│   ├── config/
│   │   └── ur5_robot.srdf         ← Defines planning groups — see note below
│   ├── launch/
│   │   ├── simulated_robot.launch.py   ← Unified all-in-one launcher
│   │   └── moveit.launch.py            ← MoveIt-only launcher (for split mode)
│   ├── ur5_moveit/                ← Python module folder (NOT "scripts/")
│   │   ├── __init__.py
│   │   ├── add_scene_objects.py   ← Console entry point: adds tables + cubes
│   │   └── insert_obstacle.py     ← Console entry point: adds/removes obstacles
│   ├── setup.py                   ← Declares the two console_scripts entry points
│   └── package.xml
│
└── ur5_moveit_config/
    ├── config/                    ← MoveIt Setup Assistant generated files
    └── launch/                    ← MoveIt configuration launchers
```

### The SRDF and planning group name

The file `ur5_ws/src/ur5_moveit/config/ur5_robot.srdf` defines the planning groups. The correct planning group name for the arm is:

```
ur_manipulator
```

> ⚠️ **This is critical for your code.** If you use a different group name in your MoveIt Python or C++ code, planning will fail with "group not found" errors.

### How Python helper nodes are structured

The two helper nodes are **console_scripts entry points** declared in `setup.py`, NOT standalone scripts in a `scripts/` folder:

```python
# From ur5_ws/src/ur5_moveit/setup.py
entry_points={
    'console_scripts': [
        'add_scene_objects = ur5_moveit.add_scene_objects:main',
        'insert_obstacle = ur5_moveit.insert_obstacle:main',
    ],
},
```

This means:
- The source files live at `ur5_moveit/ur5_moveit/add_scene_objects.py` (inside the module, not `scripts/`)
- You run them with `ros2 run ur5_moveit add_scene_objects` (the command name comes from `setup.py`)

---

## 17. Writing Your Own Code — Correct Patterns

### Where to put your code

Add your Python nodes inside the `ur5_moveit` module folder:

```
ur5_ws/src/ur5_moveit/ur5_moveit/my_pick_place.py   ← your file goes here
```

And register it as a console_script entry point in `setup.py`:

```python
entry_points={
    'console_scripts': [
        'add_scene_objects = ur5_moveit.add_scene_objects:main',
        'insert_obstacle = ur5_moveit.insert_obstacle:main',
        'my_pick_place = ur5_moveit.my_pick_place:main',    # add your node here
    ],
},
```

After editing `setup.py`, rebuild the workspace inside the container:

```bash
cd /ur5_ws
colcon build --symlink-install
source install/setup.bash
```

Then run it:

```bash
ros2 run ur5_moveit my_pick_place
```

### Minimal correct MoveIt 2 Python node

The planning group name **must** be `ur_manipulator` (from the SRDF):

```python
# Save as: ur5_ws/src/ur5_moveit/ur5_moveit/my_pick_place.py
# Register in setup.py console_scripts, then:
# Run with: ros2 run ur5_moveit my_pick_place

import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy

class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')

        # Initialize MoveIt
        self.robot = MoveItPy(node_name='moveit_py')

        # CORRECT planning group name — must match ur5_robot.srdf
        self.arm = self.robot.get_planning_component('ur_manipulator')

        self.get_logger().info('Pick and Place Node Ready!')

def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Key ROS 2 diagnostic commands (run inside any container terminal)

```bash
# List all running ROS nodes
ros2 node list

# List all active topics
ros2 topic list

# Stream joint state data
ros2 topic echo /joint_states

# List all active services
ros2 service list

# Check which controllers are running and their state
ros2 control list_controllers

# Get info about a package
ros2 pkg prefix ur5_moveit

# Check what message type a topic uses
ros2 topic info /joint_states

# Run a node
ros2 run <package_name> <executable_name>

# Launch a launch file
ros2 launch <package_name> <launch_file.py>
```

---

## 18. Comprehensive Error Reference

### Docker errors

| Error | Cause | Fix |
|---|---|---|
| `permission denied while trying to connect to the Docker daemon socket` | Not in docker group | `newgrp docker` or log out/in |
| `Cannot connect to the Docker daemon` | Docker not running | `sudo systemctl start docker` |
| `Error response from daemon: pull access denied` | Private image or rate limit | `docker login` then retry |
| `port is already allocated` | Port conflict | `docker ps` then `docker stop <name>` |
| `no space left on device` | Disk full | `docker system prune -a` |
| `Error: No such container: ur5_hackathon` | Container not started | `cd ~/Robothon/physical-ai-challenge-2026 && docker compose up -d` |
| Container immediately exits | Crash on startup | `docker compose logs ur5-sim --tail=300` |
| Name conflict with existing container | Old container still exists | `docker rm -f ur5_hackathon` |

### ROS 2 errors

| Error | Cause | Fix |
|---|---|---|
| `ros2: command not found` | ROS not sourced | `source /opt/ros/jazzy/setup.bash` |
| `Package 'ur5_moveit' not found` | Workspace not sourced | `source /ur5_ws/install/setup.bash` |
| `executable 'add_scene_objects' not found` | Workspace not sourced | Source both setup.bash files |
| `[ERROR] TF_REPEATED_DATA` | Timing harmless warning | Ignore this — simulation still works |
| `Could not load controller` | ros2_control issue | Confirm all launch dependencies started |
| Group not found in MoveIt | Wrong planning group name | Use `ur_manipulator` (from SRDF) |

### GUI and display errors

| Error | Cause | Fix |
|---|---|---|
| `cannot connect to X server :0` | xhost not set | On HOST H1: `xhost +local:root` |
| `[gz] process has died` | Gazebo crash | `export LIBGL_ALWAYS_SOFTWARE=1` inside container, then relaunch |
| `OpenGL ES profile not supported` | GPU driver mismatch | `export MESA_GL_VERSION_OVERRIDE=3.3` inside container |
| No windows appear | DISPLAY not forwarded | `echo $DISPLAY` in container — should be `:0`; if empty, check compose file |

### Build errors (Workflow B/C/D)

| Error | Cause | Fix |
|---|---|---|
| `CMake Error: could not find package 'moveit_ros_planning_interface'` | Missing ROS dep | `rosdep install --from-paths src --ignore-src -r -y` |
| `colcon: command not found` | Not installed | `sudo apt install -y python3-colcon-common-extensions` |
| `rosdep update` fails during docker build | Transient network issue | Retry `docker build` |
| `fatal: destination path already exists` | Workflow D clone conflict | Clone into a new subdirectory (this guide does that correctly) |

---

## 19. Daily Workflow Cheatsheet

After initial setup, your daily routine is:

```bash
# ── H1: HOST MACHINE ─────────────────────────────────────────────────────

# 1. Allow GUI (run once per boot — skip if you added it to ~/.bashrc)
xhost +local:root

# 2. Start the container
cd ~/Robothon/physical-ai-challenge-2026
docker compose up -d

# 3. Verify it is running
docker ps
# Must show: ur5_hackathon  Up  N minutes

# ── C1: CONTAINER TERMINAL 1 (simulation) ────────────────────────────────
docker exec -it ur5_hackathon bash

source /opt/ros/jazzy/setup.bash     # skip if you added to .bashrc
source /ur5_ws/install/setup.bash    # skip if you added to .bashrc
ros2 launch ur5_moveit simulated_robot.launch.py
# → Keep this terminal open. Gazebo + RViz appear on your screen.

# ── C2: CONTAINER TERMINAL 2 (scene setup) — open new terminal ───────────
docker exec -it ur5_hackathon bash

source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 run ur5_moveit add_scene_objects
# → Tables and cubes appear in RViz.

# ── C3: CONTAINER TERMINAL 3 (your code) — open new terminal ─────────────
docker exec -it ur5_hackathon bash

source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 run ur5_moveit my_pick_place    # your node here

# ── H1: STOPPING FOR THE DAY ─────────────────────────────────────────────

# Stop simulation: in C1 press Ctrl+C
# Then in H1:
docker compose down
xhost -local:root
```

### tmux for multi-pane workflow inside the container

Instead of opening multiple host terminals, use `tmux` to manage multiple shells inside one container terminal:

```bash
# Install tmux (if not already in container)
apt-get install -y tmux

# Start a new tmux session
tmux

# Create a vertical split (two panes side by side)
# Press: Ctrl+B then %

# Create a horizontal split (pane above and below)
# Press: Ctrl+B then "

# Switch between panes
# Press: Ctrl+B then arrow keys

# Detach from tmux (container keeps running in background)
# Press: Ctrl+B then d

# Re-attach later
tmux attach
```

---

## 20. Final Validation Checklist

Run through this checklist after completing setup. All items must pass before you start challenge development.

```
HOST MACHINE
[ ] docker --version               → Returns a version number (not "command not found")
[ ] docker compose version         → Returns a version number
[ ] docker ps                      → Shows ur5_hackathon with status "Up"

INSIDE CONTAINER (docker exec -it ur5_hackathon bash)
[ ] echo $ROS_DISTRO               → Prints: jazzy
[ ] ros2 pkg list | grep ur5_moveit → Shows: ur5_moveit
[ ] ros2 pkg list | grep ur5_description → Shows: ur5_description

SIMULATION RUNNING
[ ] ros2 launch ur5_moveit simulated_robot.launch.py launches without crash
[ ] Gazebo window opens on your desktop with the UR5 arm visible
[ ] RViz window opens with the UR5 arm model and MoveIt plugin loaded

SCENE SETUP
[ ] ros2 run ur5_moveit add_scene_objects runs without error
[ ] Tables and blue cubes appear in the RViz 3D view

CONTROLLERS
[ ] ros2 control list_controllers shows active controllers
[ ] ros2 topic list includes /joint_states and /clock
[ ] ros2 node list includes /move_group
```

If all items are checked, your environment is fully working and ready for challenge development.

---

## What to Build Next

The challenge asks you to build on top of this environment:

1. **Attach sensors** — Add cameras to the UR5 in the Gazebo URDF/Xacro model
2. **Perception** — Develop object detection using YOLOv11 on camera feeds
3. **Pick and place** — Write your own orchestrator using MoveIt 2 with the `ur_manipulator` planning group

Your code goes in `ur5_ws/src/ur5_moveit/ur5_moveit/` and is registered as a `console_scripts` entry point in `setup.py`. See Section 17 for the correct code template and the correct planning group name.

---

*This guide supersedes `Complete_Setup_Guide_Hackathon2026.md` and the original `runbook.md`. It is based on verified repo contents from `github.com/vishal-finch/physical-ai-challenge-2026` as of April 2026.*

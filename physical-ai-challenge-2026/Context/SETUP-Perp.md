# SETUP.md — Physical AI Challenge 2026 (UR5 Hackathon)
# Complete Ubuntu + Docker Operator Guide
# Repository: https://github.com/vishal-finch/physical-ai-challenge-2026

---

## TABLE OF CONTENTS

1. [System Requirements](#1-system-requirements)
2. [Repository Structure — Full Tree](#2-repository-structure--full-tree)
3. [Host Machine Prerequisites (Ubuntu)](#3-host-machine-prerequisites-ubuntu)
4. [Docker Installation & Configuration](#4-docker-installation--configuration)
5. [GUI (X11) Forwarding Setup](#5-gui-x11-forwarding-setup)
6. [Clone the Repository](#6-clone-the-repository)
7. [Understanding the Three Run Paths](#7-understanding-the-three-run-paths)
8. [PATH A — Fastest: Prebuilt Image (Recommended)](#8-path-a--fastest-prebuilt-image-recommended)
9. [PATH B — Custom Local Build](#9-path-b--custom-local-build)
10. [PATH C — Editable Dev Mode with Bind-Mount](#10-path-c--editable-dev-mode-with-bind-mount)
11. [Terminal Management Strategy](#11-terminal-management-strategy)
12. [Launch Sequence Inside Container (All Paths)](#12-launch-sequence-inside-container-all-paths)
13. [Adding Scene Objects (MoveIt Planning Scene)](#13-adding-scene-objects-moveit-planning-scene)
14. [Dynamic Obstacle Insertion](#14-dynamic-obstacle-insertion)
15. [Environment Variables Reference](#15-environment-variables-reference)
16. [Verification Commands](#16-verification-commands)
17. [Manual Folder Tree (Reconstruction Only)](#17-manual-folder-tree-reconstruction-only)
18. [Native Ubuntu Without Docker (Optional)](#18-native-ubuntu-without-docker-optional)
19. [Comprehensive Error Encyclopedia](#19-comprehensive-error-encyclopedia)
20. [Graceful Shutdown](#20-graceful-shutdown)

---

## 1. SYSTEM REQUIREMENTS

| Component         | Minimum                          | Recommended                       |
|-------------------|----------------------------------|-----------------------------------|
| OS                | Ubuntu 22.04 LTS                 | Ubuntu 24.04 LTS (Noble)          |
| CPU               | 4-core x86_64                    | 8-core x86_64                     |
| RAM               | 8 GB                             | 16 GB                             |
| GPU               | Any (software rendering OK)      | NVIDIA + nvidia-docker2 for speed |
| Disk              | 20 GB free                       | 40 GB free                        |
| Docker            | 24.x+                            | Latest stable                     |
| Docker Compose    | v2 plugin (compose v2)           | Latest stable                     |
| Display           | Active X11 desktop session       | Same                              |
| Internet          | Required (first pull/build only) | Same                              |

> ⚠️ DO NOT run on WSL1 — GUI (Gazebo/RViz) will not work.
> WSL2 is possible but requires extra VcXsrv/WSLg configuration (not covered here).
> Pure Ubuntu 22.04/24.04 desktop session is strongly recommended.

---

## 2. REPOSITORY STRUCTURE — FULL TREE

After cloning, the on-disk layout is:

```
physical-ai-challenge-2026/
├── Dockerfile                          ← Builds the full ROS2 Jazzy image
├── docker-compose.yml                  ← Ubuntu/Linux compose file (uses prebuilt image)
├── docker-compose.windows.yml          ← Windows-specific compose (different display config)
├── README.md                           ← Official quickstart guide
└── ur5_ws/
    └── src/
        ├── hello_moveit/               ← Sample MoveIt2 C++ code entry point
        │   └── src/
        │       └── (C++ source files)
        ├── ur5_controller/             ← ros2_control configuration
        │   ├── config/
        │   │   └── (controller YAML files)
        │   └── launch/
        │       └── controller.launch.py
        ├── ur5_description/            ← URDF/Xacro robot model, meshes, Gazebo worlds
        │   ├── config/
        │   ├── launch/
        │   │   └── gazebo.launch.py
        │   ├── urdf/
        │   │   └── (ur5_rg2.urdf.xacro and related files)
        │   ├── worlds/
        │   │   ├── gazebo_world.sdf
        │   │   └── pick_place_world.sdf
        │   └── meshes/
        │       ├── collision/
        │       │   ├── rg2/            ← 17 STL collision meshes for RG2 gripper
        │       │   └── ur5/            ← UR5 collision STL meshes
        │       └── visual/
        │           ├── rg2/            ← 15 DAE visual meshes for RG2 gripper
        │           └── ur5/            ← UR5 visual DAE meshes
        ├── ur5_moveit/                 ← MoveIt2 scene + Python planning scripts
        │   ├── config/
        │   ├── launch/
        │   │   ├── moveit.launch.py
        │   │   └── simulated_robot.launch.py  ← Wrapper that calls all 3 launches
        │   ├── resource/
        │   ├── ur5_moveit/             ← Python package directory
        │   │   ├── add_scene_objects.py   ← Adds ground/table/cubes to MoveIt scene
        │   │   └── insert_obstacle.py     ← Dynamically inserts/removes obstacles
        │   └── worlds/
        └── ur5_moveit_config/          ← SRDF, kinematics, pipeline configs
            ├── config/
            │   ├── ur5_rg2.srdf
            │   ├── kinematics.yaml
            │   ├── joint_limits.yaml
            │   └── (other MoveIt2 YAML configs)
            └── launch/
```

> KEY INSIGHT: The `docker-compose.yml` references a **prebuilt image** from a registry.
> Your local `ur5_ws/src/` edits will NOT affect a running container unless you use
> Path B (rebuild image) or Path C (bind-mount source into container).

---

## 3. HOST MACHINE PREREQUISITES (UBUNTU)

Open one terminal on your Ubuntu desktop. This is your **H1 (Host Terminal 1)**.
You will use H1 for all host-side operations.

### Step 3.1 — Update System

```bash
# H1
sudo apt update && sudo apt upgrade -y
```

> ⚠️ Possible Error: `E: Could not get lock /var/lib/dpkg/lock`
> Fix: `sudo killall apt apt-get; sudo rm /var/lib/dpkg/lock*; sudo dpkg --configure -a`

### Step 3.2 — Install Core Host Dependencies

```bash
# H1
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

> ⚠️ Possible Error: `Package x11-xserver-utils not found`
> Fix on Ubuntu 22.04+: `sudo apt install -y x11-utils xauth`

---

## 4. DOCKER INSTALLATION & CONFIGURATION

> ⚠️ If Docker is already installed, verify version with `docker --version` and
> `docker compose version`. If both are ≥ 24.x and v2.x respectively, skip to Step 4.4.

### Step 4.1 — Remove Old Docker Versions (If Any)

```bash
# H1
sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
```

### Step 4.2 — Install Docker Engine (Official Method)

```bash
# H1 — Add Docker GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# H1 — Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# H1 — Install Docker Engine + Compose Plugin
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

> ⚠️ Possible Error: `Unable to locate package docker-ce`
> Fix: Your Ubuntu codename may not match. Run `lsb_release -cs` and confirm it returns
> `jammy` (22.04) or `noble` (24.04). If using a non-standard distro, replace
> `$(. /etc/os-release && echo "$VERSION_CODENAME")` with `jammy` or `noble` explicitly.

### Step 4.3 — Verify Docker Installation

```bash
# H1
docker --version
# Expected: Docker version 26.x.x or higher

docker compose version
# Expected: Docker Compose version v2.x.x
# ⚠️ Must be `docker compose` (v2 plugin), NOT `docker-compose` (v1 standalone)
```

> ⚠️ Possible Error: `docker: command not found`
> Fix: `sudo systemctl start docker && sudo systemctl enable docker`

### Step 4.4 — Add User to Docker Group (Avoid sudo Every Time)

```bash
# H1
sudo usermod -aG docker $USER

# Apply group change in CURRENT terminal session immediately:
newgrp docker

# Verify:
groups $USER
# Expected output should include: docker
```

> ⚠️ IMPORTANT: `newgrp docker` only applies to the current shell session.
> To make it permanent across all terminals → **Log out of your Ubuntu desktop and log back in**.
> After re-login, verify: `docker ps` (should work without sudo)

### Step 4.5 — Start and Enable Docker Service

```bash
# H1
sudo systemctl start docker
sudo systemctl enable docker
sudo systemctl status docker
# Expected: Active (running)
```

> ⚠️ Possible Error: `Failed to connect to bus: No such file or directory`
> This happens inside containers or minimal Ubuntu installs. Fix: Use
> `sudo service docker start` as an alternative.

---

## 5. GUI (X11) FORWARDING SETUP

Gazebo and RViz are graphical apps. They need X11 display access inside Docker.

### Step 5.1 — Grant X11 Access to Docker Container

```bash
# H1 — Run this EVERY TIME before starting the container
xhost +local:root
```

Expected output: `non-network local connections being added to access control list`

> ⚠️ Possible Error: `unable to open display ""`
> Fix: You are not in a graphical desktop session, or DISPLAY is unset.
> Run: `echo $DISPLAY` → it must return something like `:0` or `:1`
> If empty: `export DISPLAY=:0` then retry `xhost +local:root`

> ⚠️ Possible Error: `xhost: command not found`
> Fix: `sudo apt install -y x11-xserver-utils` then retry.

> ⚠️ Possible Error after reboot: GUI apps in container fail
> Fix: Simply re-run `xhost +local:root` after every reboot.

### Step 5.2 — Verify X11 Is Working

```bash
# H1
echo $DISPLAY
# Must return: :0 (or :1, :2 — any non-empty value)

xclock &   # Should show a small clock window (test X11)
# Kill it: kill %1
```

---

## 6. CLONE THE REPOSITORY

### Step 6.1 — Create Workspace Directory

```bash
# H1
mkdir -p ~/Robothon
cd ~/Robothon
```

> This is your host working directory. All project files live here.

### Step 6.2 — Clone the Repository

```bash
# H1
git clone https://github.com/vishal-finch/physical-ai-challenge-2026.git
```

Expected output:
```
Cloning into 'physical-ai-challenge-2026'...
remote: Enumerating objects: ...
...
Receiving objects: 100% ...
```

> ⚠️ Possible Error: `Repository not found` or 404
> Fix: Check your internet connection. Verify the URL is correct.
> If the repo is private: `git clone https://<your-token>@github.com/vishal-finch/physical-ai-challenge-2026.git`

> ⚠️ Possible Error: `destination path 'physical-ai-challenge-2026' already exists`
> Fix:
> ```bash
> cd ~/Robothon/physical-ai-challenge-2026
> git pull origin main
> ```

### Step 6.3 — Enter Project Directory

```bash
# H1
cd ~/Robothon/physical-ai-challenge-2026
ls -la
```

Expected: You should see `Dockerfile`, `docker-compose.yml`, `docker-compose.windows.yml`,
`README.md`, and `ur5_ws/` directory.

---

## 7. UNDERSTANDING THE THREE RUN PATHS

| Path | What It Does                             | Use When                             | Speed   |
|------|------------------------------------------|--------------------------------------|---------|
| A    | Pull prebuilt image, run immediately      | First time, just want it working     | Fastest |
| B    | Build image from local Dockerfile         | Dockerfile is modified               | Slow    |
| C    | Bind-mount local source into container   | Actively coding/modifying packages   | Medium  |

> **For your first run → Use Path A.**
> **For development where you edit Python/C++ code → Use Path C.**

---

## 8. PATH A — FASTEST: PREBUILT IMAGE (RECOMMENDED)

### Step 8.1 — Pull the Prebuilt Image

```bash
# H1 — Must be inside the project directory
cd ~/Robothon/physical-ai-challenge-2026

docker compose pull
```

Expected: Image layers downloading. This may take 5–20 minutes on first run (image is ~3–5 GB).

```
[+] Pulling 12/12
 ✔ ur5_hackathon Pulled
```

> ⚠️ Possible Error: `pull access denied` or `image not found`
> Fix: The prebuilt image registry may require authentication or the image tag changed.
> Check `docker-compose.yml` for the exact image name:
> ```bash
> cat docker-compose.yml | grep image:
> ```
> Then try: `docker pull <image-name>` manually.

> ⚠️ Possible Error: `connection timeout`
> Fix: Check internet. Try again. Possibly add DNS: `echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf`

### Step 8.2 — Allow X11 Access (Must Do Before Starting Container)

```bash
# H1
xhost +local:root
```

### Step 8.3 — Start the Container (Detached Mode)

```bash
# H1 — Still inside ~/Robothon/physical-ai-challenge-2026
docker compose up -d
```

Expected:
```
[+] Running 1/1
 ✔ Container ur5_hackathon  Started
```

> ⚠️ Possible Error: `Cannot connect to the Docker daemon`
> Fix: `sudo systemctl start docker` then retry.

> ⚠️ Possible Error: `port is already allocated`
> Fix: Another container is using the same port. Run:
> `docker ps -a` to see running containers. Stop conflicting ones:
> `docker stop <container_name>`

> ⚠️ Possible Error: `no configuration file provided: not found`
> Fix: You are not inside the project directory. Run:
> `cd ~/Robothon/physical-ai-challenge-2026` first.

### Step 8.4 — Verify Container Is Running

```bash
# H1
docker ps --filter name=ur5_hackathon
```

Expected:
```
CONTAINER ID   IMAGE           COMMAND        CREATED        STATUS        NAMES
abc123def456   <image:tag>     "/bin/bash"    10 seconds ago Up 9 seconds  ur5_hackathon
```

> ⚠️ If STATUS shows `Exited` or `Restarting`: Check logs:
> `docker logs ur5_hackathon`
> Common causes: DISPLAY not set, missing X11 socket, or entrypoint error.

---

## 9. PATH B — CUSTOM LOCAL BUILD

Use this ONLY if you have modified the `Dockerfile` or need a different base image.

### Step 9.1 — Build the Image Locally

```bash
# H1
cd ~/Robothon/physical-ai-challenge-2026
docker build -t ur5-hackathon-local .
```

> ⚠️ Expected build time: 15–60 minutes first time (downloads ROS2 packages)
> ⚠️ Possible Error: `rosdep install failed`
> Fix: Temporary network/apt mirror outage. Retry. Or check your DNS.

> ⚠️ Possible Error: `Dockerfile not found`
> Fix: Confirm you are in the root of the repo: `ls Dockerfile`

> ⚠️ Possible Error: `no space left on device`
> Fix: Free disk space. `docker system prune -a` to remove old images/containers.

### Step 9.2 — Run the Custom Image

```bash
# H1
xhost +local:root

docker run --rm -it \
  --name ur5_hackathon \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ur5-hackathon-local bash
```

> You are now INSIDE the container (your prompt changes to root@<container_id>).
> Proceed to Section 12 for the launch sequence.

---

## 10. PATH C — EDITABLE DEV MODE WITH BIND-MOUNT

Use this when you are **actively modifying** package source files and want changes reflected
without rebuilding the image.

### Step 10.1 — Build the Base Image First (if not done)

```bash
# H1
cd ~/Robothon/physical-ai-challenge-2026
docker build -t ur5-hackathon-local .
```

### Step 10.2 — Start Dev Container with Source Mounted

```bash
# H1
cd ~/Robothon/physical-ai-challenge-2026
xhost +local:root

docker run --rm -it \
  --name ur5_hackathon_dev \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/ur5_ws/src:/ur5_ws/src \
  ur5-hackathon-local bash
```

> The `-v $(pwd)/ur5_ws/src:/ur5_ws/src` flag mounts your local source tree
> into the container at `/ur5_ws/src`. Any file you edit on your HOST in
> `~/Robothon/physical-ai-challenge-2026/ur5_ws/src/` is IMMEDIATELY visible
> inside the container at `/ur5_ws/src/` — no container restart needed.

### Step 10.3 — Build Workspace Inside Container (After Mounting)

```bash
# INSIDE Container (C1 terminal)
cd /ur5_ws
source /opt/ros/jazzy/setup.bash

# Update rosdep
rosdep update

# Install all package dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build with symlink install (faster rebuild on Python changes)
colcon build --symlink-install

# Source the newly built workspace
source /ur5_ws/install/setup.bash
```

> ⚠️ Possible Error: `Package 'ur5_moveit' not found`
> Fix: Ensure `source /ur5_ws/install/setup.bash` was run. Check:
> `echo $AMENT_PREFIX_PATH` → should include `/ur5_ws/install`

> ⚠️ Possible Error: `colcon: command not found`
> Fix: `sudo apt install -y python3-colcon-common-extensions`

> ⚠️ Possible Error: `rosdep: command not found`
> Fix: `sudo apt install -y python3-rosdep`

> After every new file edit that adds new Python entry points (new scripts in setup.py),
> re-run `colcon build --symlink-install` and re-source.

---

## 11. TERMINAL MANAGEMENT STRATEGY

This project requires MULTIPLE simultaneous terminal windows. Here is the exact plan:

```
HOST (Ubuntu Desktop)
│
├── H1 — Host Terminal 1 (main host terminal)
│         Used for: docker commands, xhost, git, docker compose up/down
│         Keep this open the entire session
│
└── CONTAINER TERMINALS (opened as needed)
    ├── C1 — Container Terminal 1 (MAIN LAUNCH)
    │         Used for: First exec into container + main simulation launch
    │         Open: After docker compose up -d (Step 8.3)
    │         Command: docker exec -it ur5_hackathon bash
    │         Status: Keep open — do NOT close while simulation is running
    │
    ├── C2 — Container Terminal 2 (SCENE SETUP)
    │         Used for: Adding scene objects after simulation is up
    │         Open: After C1 launch shows "MoveIt is ready" (wait ~30-60s)
    │         Command: docker exec -it ur5_hackathon bash
    │
    └── C3 — Container Terminal 3 (OBSTACLE / DEBUG)
              Used for: Optional obstacle insertion, inspection commands
              Open: After C1 and C2 are stable
              Command: docker exec -it ur5_hackathon bash
```

> RULE: Always open C2/C3 AFTER C1 is fully launched and stable.
> NEVER run scene setup before move_group node is ready.

### Opening New Container Terminals

```bash
# In a NEW Ubuntu terminal window (not C1):
docker exec -it ur5_hackathon bash
# OR for dev container:
docker exec -it ur5_hackathon_dev bash
```

> ⚠️ Possible Error: `Error: No such container: ur5_hackathon`
> Fix: The container name is wrong or the container is not running.
> Check: `docker ps` → look for the exact container name.
> If using Path A: name is `ur5_hackathon`
> If using Path B/C docker run: name is whatever you passed to `--name`

---

## 12. LAUNCH SEQUENCE INSIDE CONTAINER (ALL PATHS)

### Step 12.1 — Enter Container in C1

```bash
# In a new Ubuntu terminal → becomes C1
docker exec -it ur5_hackathon bash
```

Your prompt changes to something like: `root@abc123:/ur5_ws#`

### Step 12.2 — Verify Environment in C1

```bash
# C1 — Inside container
echo $ROS_DISTRO
# Expected: jazzy

ls /ur5_ws/src
# Expected: hello_moveit  ur5_controller  ur5_description  ur5_moveit  ur5_moveit_config

# Verify ROS2 sourced
ros2 --help | head -3
# Should show ros2 help without errors
```

> ⚠️ Possible Error: `ros2: command not found`
> Fix: Source manually: `source /opt/ros/jazzy/setup.bash`
> For permanent fix inside container session:
> `echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && source ~/.bashrc`

> ⚠️ Possible Error: `ls: /ur5_ws/src: No such file or directory`
> Fix: Wrong container or image. Verify with `docker inspect ur5_hackathon | grep -A5 Mounts`

### Step 12.3 — Source Workspace (If Not Auto-Sourced)

```bash
# C1 — Check if workspace is sourced
echo $AMENT_PREFIX_PATH
# Expected: should include /ur5_ws/install

# If empty, manually source:
source /ur5_ws/install/setup.bash
```

### Step 12.4 — Launch Option 1: Unified Wrapper (Try First)

```bash
# C1 — This launches Gazebo + Controllers + MoveIt all at once
ros2 launch ur5_moveit simulated_robot.launch.py
```

Wait 30–60 seconds. Watch for these key lines:
- `[gazebo-1] [INFO] ... Gazebo started`
- `[controller_manager-X] ... Loaded joint_trajectory_controller`
- `[move_group-X] ... MoveIt is ready`
- `[rviz2-X] ... Initialized`

**If you see all the above → wrapper is working. Skip to Step 13.**

> ⚠️ Possible Error: `package 'ur5_moveit' not found`
> Fix: `source /ur5_ws/install/setup.bash` and retry.

> ⚠️ Possible Error: Wrapper crashes with `launch.InvalidLaunchFileError`
> Fix: Use Split Launch Mode (Step 12.5) below.

> ⚠️ Possible Error: Gazebo opens but shows blank/grey world
> Fix: Wait longer (up to 2 min for first load). If persists:
> Check `echo $DISPLAY` inside container. Should match host DISPLAY.

### Step 12.5 — Launch Option 2: Split Launch Mode (If Wrapper Fails)

**Use THREE separate container terminals: C1, C2 (new), C3 (new)**

**In C1 — Launch Gazebo:**
```bash
# C1
ros2 launch ur5_description gazebo.launch.py
```
Wait until you see Gazebo window open AND:
`[gzserver-1] [INFO] ... Loaded world: ...`

> ⚠️ DO NOT proceed to C2 until Gazebo is fully loaded (you see the 3D world).

**In C2 — Launch Controllers:**
```bash
# Open new Ubuntu terminal → C2
docker exec -it ur5_hackathon bash
source /ur5_ws/install/setup.bash   # If needed

ros2 launch ur5_controller controller.launch.py
```
Wait until you see:
`Configured and activated joint_trajectory_controller`
`Configured and activated gripper_controller`

> ⚠️ Possible Error: `Controller spawner: Waiting for controller_manager...` (hangs)
> Fix: Gazebo hasn't fully loaded. Wait for Gazebo window. Then retry C2.

> ⚠️ Possible Error: `Timeout waiting for controller_manager`
> Fix: Kill C2 process (Ctrl+C), wait 10 more seconds, retry.

**In C3 — Launch MoveIt:**
```bash
# Open new Ubuntu terminal → C3
docker exec -it ur5_hackathon bash
source /ur5_ws/install/setup.bash   # If needed

ros2 launch ur5_moveit moveit.launch.py
```
Wait until you see:
`[move_group-1] [INFO] ... MoveIt is ready`
`[rviz2-1] Initialized`

> ⚠️ Possible Error: RViz opens but shows no robot model
> Fix: `robot_state_publisher` is not running. Check:
> `ros2 node list | grep robot_state`
> If missing, it should have been started by Gazebo or controller launch.
> Restart controller launch in C2.

> ⚠️ Possible Error: `move_group` keeps restarting
> Fix: Joint state topic is missing. Verify: `ros2 topic echo /joint_states --once`
> If no output, the controller hasn't started correctly.

---

## 13. ADDING SCENE OBJECTS (MOVEIT PLANNING SCENE)

This step adds the physical world geometry (ground plane, tables, pick cubes) to the
MoveIt2 planning scene so the arm knows what obstacles to avoid.

> ⚠️ PRECONDITION: `move_group` node MUST be running before this step.
> Wait for "MoveIt is ready" message in C1 (or C3 in split mode).

### Step 13.1 — Open Scene Setup Terminal (C2 or New Terminal)

```bash
# Open new Ubuntu terminal → C2 (if not already open)
docker exec -it ur5_hackathon bash
source /ur5_ws/install/setup.bash   # If needed
```

### Step 13.2 — Run Scene Setup Script

```bash
# C2
ros2 run ur5_moveit add_scene_objects
```

Expected: Script runs and exits cleanly. In the MoveIt RViz window you should see:
- Ground plane mesh appear
- Table(s) appear
- Pickup cube objects appear

> ⚠️ Possible Error: `Package 'ur5_moveit' not found`
> Fix: `source /ur5_ws/install/setup.bash`

> ⚠️ Possible Error: Script exits immediately with no visible objects
> Fix: `move_group` is not fully ready. Wait 10 seconds and re-run.

> ⚠️ Possible Error: Objects appear in wrong position / floating / misaligned
> Cause: The `add_scene_objects.py` script uses `world` frame. If your planning
> group is referenced to `base_link`, there may be a transform offset.
> Fix: Check `ros2 run tf2_tools view_frames` and verify transform chain.
> This is a known partial inconsistency between scene scripts and world files.

### Step 13.3 — Verify Scene Objects in RViz

In the RViz window:
1. In left panel under "MotionPlanning" → "Scene Objects" tab
2. You should see listed: `ground_plane`, `table_1`, `cube_1`, etc.

---

## 14. DYNAMIC OBSTACLE INSERTION

This is OPTIONAL. Use to test collision avoidance with dynamic scene objects.

### Step 14.1 — Open Obstacle Terminal (C3)

```bash
# Open new Ubuntu terminal → C3
docker exec -it ur5_hackathon bash
source /ur5_ws/install/setup.bash   # If needed
```

### Step 14.2 — Insert an Obstacle

```bash
# C3 — Insert a cylinder obstacle at specified position
ros2 run ur5_moveit insert_obstacle --x 0.3 --y -0.2 --z 0.5 --radius 0.04 --height 0.25
```

Parameters:
- `--x`, `--y`, `--z`: Position in world frame (meters)
- `--radius`: Cylinder radius (meters)
- `--height`: Cylinder height (meters)
- `--name`: Optional name (default: `obstacle`)

### Step 14.3 — Remove an Obstacle

```bash
# C3
ros2 run ur5_moveit insert_obstacle --name obstacle --remove
```

> ⚠️ Possible Error: Obstacle appears at wrong position
> Cause: `insert_obstacle.py` uses `world` frame, while some scene objects
> from `add_scene_objects.py` may use `base_link` frame.
> Fix: Account for the transform offset between `world` and `base_link`.
> Usually `base_link` is offset from `world` by the robot mount position.

---

## 15. ENVIRONMENT VARIABLES REFERENCE

These variables are set inside the container automatically. Verify them after entering:

```bash
# Inside any container terminal (C1/C2/C3)
printenv | grep -E "ROS|DISPLAY|AMENT|COLCON|QT"
```

| Variable                | Value (Expected)              | Purpose                              |
|-------------------------|-------------------------------|--------------------------------------|
| `ROS_DISTRO`            | `jazzy`                       | ROS2 distribution                    |
| `ROS_VERSION`           | `2`                           | ROS major version                    |
| `AMENT_PREFIX_PATH`     | `/ur5_ws/install:/opt/ros/...`| ROS2 package discovery path          |
| `DISPLAY`               | `:0` (or `:1`, `:99`)         | X11 display for GUI apps             |
| `QT_X11_NO_MITSHM`      | `1`                           | Fix Qt shared memory errors in Docker|
| `COLCON_PREFIX_PATH`    | `/ur5_ws/install`             | Colcon build artifacts               |
| `LD_LIBRARY_PATH`       | Includes `/ur5_ws/install/...`| Shared library resolution            |
| `PYTHONPATH`            | Includes `/ur5_ws/install/...`| Python package resolution            |

### Setting Missing Variables Manually (If Needed)

```bash
# Inside container — if DISPLAY is wrong:
export DISPLAY=:0
export QT_X11_NO_MITSHM=1

# If workspace not sourced:
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
```

> ⚠️ For Path C dev mode, after `colcon build` inside container, ALWAYS run:
> `source /ur5_ws/install/setup.bash`
> in EVERY new container terminal that needs to use the built packages.
> The build output does NOT automatically update running terminals.

---

## 16. VERIFICATION COMMANDS

Run these from any container terminal to confirm everything is working:

### Check ROS2 Nodes

```bash
ros2 node list
```

Expected (when fully running):
```
/controller_manager
/gazebo
/joint_state_broadcaster
/move_group
/robot_state_publisher
/rviz2
/joint_trajectory_controller
/gripper_controller
```

### Check ROS2 Topics

```bash
ros2 topic list | grep -E "clock|joint_states|collision_object|planning_scene"
```

Expected:
```
/clock
/joint_states
/collision_object
/planning_scene
/joint_trajectory_controller/joint_trajectory
```

### Check Controllers

```bash
ros2 control list_controllers
```

Expected:
```
joint_trajectory_controller[joint_trajectory_controller/JointTrajectoryController] active
joint_state_broadcaster[joint_state_broadcaster/JointStateBroadcaster] active
gripper_controller[...] active
```

> ⚠️ If controllers show `inactive` or `unconfigured`:
> Fix: Restart the controller launch terminal (C2 in split mode).

### Check TF Tree

```bash
ros2 run tf2_tools view_frames
# Generates frames.pdf in current directory
```

### Check Joint States Are Publishing

```bash
ros2 topic echo /joint_states --once
```

Expected: Joint names and positions for all UR5 joints.

---

## 17. MANUAL FOLDER TREE (RECONSTRUCTION ONLY)

> ⚠️ USE THIS ONLY IF CLONE FAILED. Normally, `git clone` creates all folders.
> Many mesh files are BINARY (STL/DAE) — you cannot recreate them manually.
> Only the directory structure can be manually created; file content must come from git.

```bash
# H1 — Create full directory tree manually
mkdir -p ~/Robothon/physical-ai-challenge-2026
cd ~/Robothon/physical-ai-challenge-2026

# ROS2 workspace source packages
mkdir -p ur5_ws/src/hello_moveit/src

mkdir -p ur5_ws/src/ur5_controller/config
mkdir -p ur5_ws/src/ur5_controller/launch

mkdir -p ur5_ws/src/ur5_description/config
mkdir -p ur5_ws/src/ur5_description/launch
mkdir -p ur5_ws/src/ur5_description/urdf
mkdir -p ur5_ws/src/ur5_description/worlds

# Mesh directories (binary STL and DAE files go here — get from git)
mkdir -p ur5_ws/src/ur5_description/meshes/collision/rg2
mkdir -p ur5_ws/src/ur5_description/meshes/collision/ur5
mkdir -p ur5_ws/src/ur5_description/meshes/visual/rg2
mkdir -p ur5_ws/src/ur5_description/meshes/visual/ur5

mkdir -p ur5_ws/src/ur5_moveit/config
mkdir -p ur5_ws/src/ur5_moveit/launch
mkdir -p ur5_ws/src/ur5_moveit/resource
mkdir -p ur5_ws/src/ur5_moveit/ur5_moveit
mkdir -p ur5_ws/src/ur5_moveit/worlds

mkdir -p ur5_ws/src/ur5_moveit_config/config
mkdir -p ur5_ws/src/ur5_moveit_config/launch

# Verify
find . -type d | sort
```

**Mesh Assets Inventory (from repository):**
- `meshes/collision/rg2/` + `meshes/collision/ur5/` → **17 STL files total** (binary, collision geometry)
- `meshes/visual/rg2/` + `meshes/visual/ur5/` → **15 DAE files total** (binary, visual geometry)

> These binary mesh files CANNOT be typed by hand. They must come from `git clone`.

---

## 18. NATIVE UBUNTU WITHOUT DOCKER (OPTIONAL)

> ⚠️ This is significantly harder and error-prone. Only use if Docker is unavailable.
> Requires Ubuntu 24.04 (Noble) — the same base as the Docker image (ROS2 Jazzy target).

### Step 18.1 — Install ROS2 Jazzy Desktop

```bash
# H1 — Add ROS2 apt repository
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu \
  $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-jazzy-desktop
```

### Step 18.2 — Install ROS2 Build Tools + Dependencies

```bash
# H1
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  ros-jazzy-moveit \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-xacro \
  ros-jazzy-rviz2

sudo rosdep init
rosdep update
```

### Step 18.3 — Build Workspace

```bash
# H1
cd ~/Robothon/physical-ai-challenge-2026/ur5_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### Step 18.4 — Launch (Same as Container Steps)

Follow Section 12.4 or 12.5 exactly, but run all commands in H1 (no docker exec needed).

---

## 19. COMPREHENSIVE ERROR ENCYCLOPEDIA

### Docker Errors

| Error                                          | Cause                              | Fix                                                          |
|------------------------------------------------|------------------------------------|--------------------------------------------------------------|
| `Cannot connect to the Docker daemon`          | Docker service not running         | `sudo systemctl start docker`                               |
| `Permission denied while trying to connect`   | User not in docker group           | `sudo usermod -aG docker $USER && newgrp docker`            |
| `pull access denied for <image>`              | Image is private or wrong name     | Check `docker-compose.yml` image name, authenticate if needed|
| `no space left on device`                     | Disk full                          | `docker system prune -a` — removes unused images/containers  |
| `port is already allocated`                   | Port conflict                      | `docker ps` → stop conflicting container                    |
| `OCI runtime exec failed`                     | Container not running              | `docker start ur5_hackathon` then retry exec                |

### X11 / GUI Errors

| Error                                         | Cause                              | Fix                                                          |
|-----------------------------------------------|------------------------------------|--------------------------------------------------------------|
| `cannot open display: :0`                     | X11 access not granted             | `xhost +local:root` on host                                 |
| `qt.qpa.xcb: could not connect to display`   | DISPLAY env var wrong in container | Set `DISPLAY` correctly in docker run or compose file        |
| `libGL error: MESA-LOADER: failed to open`   | Mesa/OpenGL driver missing         | Add `--privileged` flag or install mesa in container         |
| Gazebo opens but no 3D view / black screen   | GPU/OpenGL not passing through     | Set `LIBGL_ALWAYS_SOFTWARE=1` for software rendering         |
| RViz crashes immediately                      | Shared memory Qt issue             | Set `QT_X11_NO_MITSHM=1`                                    |

### ROS2 Launch Errors

| Error                                              | Cause                                    | Fix                                                     |
|----------------------------------------------------|------------------------------------------|---------------------------------------------------------|
| `package 'ur5_moveit' not found`                  | Workspace not sourced                    | `source /ur5_ws/install/setup.bash`                    |
| `controller_manager timeout`                      | Gazebo not fully loaded when controller launched | Wait for Gazebo to open fully, then retry            |
| `MoveGroup: waiting for robot model`             | robot_state_publisher not running        | Confirm gazebo.launch.py started robot_state_publisher |
| `No transform from base_link to world`           | TF tree broken                           | Check all three launches are running                    |
| `invalid launch file`                             | Python syntax error in launch file       | Use split launch mode (Step 12.5)                       |
| `Failed to load plugin`                          | Missing ros-jazzy package in image       | Check Dockerfile for missing apt packages               |

### Build / Workspace Errors

| Error                                             | Cause                                   | Fix                                                     |
|---------------------------------------------------|-----------------------------------------|---------------------------------------------------------|
| `colcon build: stderr: CMakeError`               | Missing system dependency               | `rosdep install --from-paths src --ignore-src -r -y`   |
| `No module named 'rclpy'`                        | Python ROS2 packages not on PYTHONPATH  | `source /opt/ros/jazzy/setup.bash`                     |
| `Python entry point not found after build`       | setup.py entry_points not updated       | Edit setup.py, rebuild with `colcon build --symlink-install` |
| `Package not found after colcon build`           | install/setup.bash not sourced          | `source /ur5_ws/install/setup.bash`                    |

### Scene / Planning Errors

| Error                                             | Cause                                   | Fix                                                      |
|---------------------------------------------------|-----------------------------------------|----------------------------------------------------------|
| `add_scene_objects: move_group not available`    | MoveIt not fully initialized            | Wait for "MoveIt is ready", then rerun                  |
| `Objects not visible in RViz`                    | Scene objects added to wrong frame      | Verify frame in add_scene_objects.py matches config     |
| `Obstacle in wrong position`                     | world vs base_link frame mismatch       | Check TF offset, adjust --x --y --z accordingly        |
| `Planning failed: path planning timeout`         | Scene collision or IK failure           | Check joint_states updating, try different goal pose    |
| `MoveIt plan succeeded but arm doesn't move`    | Controllers not active                  | `ros2 control list_controllers` → activate if needed   |

---

## 20. GRACEFUL SHUTDOWN

### Stop Simulation (Inside Container Terminals)

```bash
# In EACH container terminal running a launch command (C1, C2, C3):
Ctrl+C
```
Wait 3–5 seconds for processes to terminate cleanly.

### Stop and Remove Container (In H1)

```bash
# H1 — For Path A (docker compose):
cd ~/Robothon/physical-ai-challenge-2026
docker compose down

# H1 — For Path B/C (docker run):
docker stop ur5_hackathon        # or ur5_hackathon_dev
docker rm ur5_hackathon          # Clean up container (if not using --rm flag)
```

### Remove X11 Access Grant (Security)

```bash
# H1 — Optional: revoke root X11 access after session
xhost -local:root
```

### Clean Up Docker Resources (If Needed)

```bash
# H1 — Remove stopped containers
docker container prune -f

# Remove unused images (careful — this removes ALL unused images)
docker image prune -a

# Full cleanup
docker system prune -a
```

---

## QUICK REFERENCE CHEATSHEET

```
SESSION START SEQUENCE:
H1: cd ~/Robothon/physical-ai-challenge-2026
H1: xhost +local:root
H1: docker compose up -d
H1: docker ps --filter name=ur5_hackathon

C1: docker exec -it ur5_hackathon bash
C1: ros2 launch ur5_moveit simulated_robot.launch.py
    [Wait for "MoveIt is ready"]

C2: docker exec -it ur5_hackathon bash
C2: ros2 run ur5_moveit add_scene_objects

C3 (optional): docker exec -it ur5_hackathon bash
C3: ros2 run ur5_moveit insert_obstacle --x 0.3 --y -0.2 --z 0.5 --radius 0.04 --height 0.25

SESSION END SEQUENCE:
C1/C2/C3: Ctrl+C  (each terminal)
H1: docker compose down
```

---

*Generated for: Physical AI Challenge 2026 | UR5 + RG2 | ROS2 Jazzy | Ubuntu 22.04/24.04 + Docker*
*Repository: https://github.com/vishal-finch/physical-ai-challenge-2026*

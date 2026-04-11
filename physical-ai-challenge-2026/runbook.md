# Physical AI Challenge 2026 - Complete Ubuntu and Docker Runbook

Last updated: 2026-04-11
Canonical setup guide: this file only

Official repository:
- https://github.com/vishal-finch/physical-ai-challenge-2026

This runbook is a merged, corrected, and detailed guide.
It keeps only repository-aligned instructions and removes broken or mismatched guidance.

------------------------------------------------------------------------------

## 1) Scope and Source of Truth

This guide is validated against the current repository files, including:
- README.md
- docker-compose.yml
- docker-compose.windows.yml
- Dockerfile
- ur5_ws/src/ur5_description/launch/gazebo.launch.py
- ur5_ws/src/ur5_controller/launch/controller.launch.py
- ur5_ws/src/ur5_moveit/launch/simulated_robot.launch.py
- ur5_ws/src/ur5_moveit/launch/moveit.launch.py
- ur5_ws/src/ur5_moveit/ur5_moveit/add_scene_objects.py
- ur5_ws/src/ur5_moveit/ur5_moveit/insert_obstacle.py
- ur5_ws/src/ur5_moveit/setup.py
- ur5_ws/src/ur5_moveit/config/ur5_robot.srdf

If any old PDF instructions conflict with this runbook, follow this runbook.

------------------------------------------------------------------------------

## 2) Quick Reality Check: Old PDF vs Current Repo

If you received older instructions, verify these core values first.

| Item | Current repo value |
|---|---|
| Robot | UR5 + gripper |
| Simulator | Gazebo (ros_gz stack) |
| ROS distro | Jazzy |
| Default container name | ur5_hackathon |
| Default compose image | vishalrobotics/ur5-hackathon-env:latest |
| Main launcher | ros2 launch ur5_moveit simulated_robot.launch.py |
| Workspace inside container | /ur5_ws |

------------------------------------------------------------------------------

## 3) System Requirements and Preflight Checks

Recommended host:
- Ubuntu 24.04 desktop

Minimum practical resources:
- x86_64 CPU
- 8 GB RAM (16 GB recommended)
- 15 GB free disk (30 GB recommended)

Run these checks on host terminal H1:

```bash
lsb_release -a
uname -m
df -h ~
free -h
nvidia-smi || true
curl -I https://hub.docker.com
```

Expected:
- Ubuntu desktop version appears
- architecture is x86_64
- internet reachable
- nvidia-smi may fail if no Nvidia GPU (that is fine)

------------------------------------------------------------------------------

## 4) Terminal Model (Strict)

Use this model throughout:

- H1: Host Ubuntu shell
  - Docker lifecycle, cloning, logs, host setup
- C1: Container shell for main launch
- C2: Container shell for scene setup
- C3: Container shell for obstacle/testing/diagnostics

Rules:

1. Docker commands run in H1 only.
2. ROS commands run in C1/C2/C3.
3. Keep C1 open while simulation is running.
4. Open C2 and C3 only after C1 is healthy.
5. In development mode, source ROS and workspace in each new container shell.

------------------------------------------------------------------------------

## 5) Host Setup on Ubuntu

### 5.1 Install required host packages

In H1:

```bash
sudo apt update
sudo apt install -y git curl docker.io docker-compose-plugin x11-xserver-utils mesa-utils
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker "$USER"
newgrp docker
```

Verify:

```bash
docker --version
docker compose version
git --version
```

Common failures:

1. permission denied on Docker socket
- run newgrp docker
- if needed, logout/login

2. docker compose command missing
- install docker-compose-plugin

3. Docker daemon not reachable

```bash
sudo systemctl status docker
sudo systemctl start docker
```

### 5.2 Allow GUI forwarding for container windows

In H1:

```bash
echo "$DISPLAY"
xhost +local:root
```

If DISPLAY is empty:

```bash
export DISPLAY=:0
xhost +local:root
```

------------------------------------------------------------------------------

## 6) Clone and Enter Repository

In H1:

```bash
mkdir -p ~/Robothon
cd ~/Robothon
git clone https://github.com/vishal-finch/physical-ai-challenge-2026.git
cd physical-ai-challenge-2026
```

Sanity checks:

```bash
pwd
ls
ls ur5_ws/src
```

Expected source packages under ur5_ws/src include:
- hello_moveit
- ur5_controller
- ur5_description
- ur5_moveit
- ur5_moveit_config

------------------------------------------------------------------------------

## 7) Choose One Runtime Workflow

1. Workflow A: fastest start, prebuilt image via compose
2. Workflow B: build your own local image
3. Workflow C: bind-mount source for active development
4. Workflow D: create/build full workspace inside Docker from scratch
5. Workflow E: native Ubuntu (optional, harder to keep consistent)

------------------------------------------------------------------------------

## 8) Workflow A - Prebuilt Image via Docker Compose (Recommended)

### 8.1 Start container

In H1 at repo root:

```bash
docker compose pull
docker compose up -d
docker ps --filter name=ur5_hackathon
```

Expected:
- ur5_hackathon is running

If container exits:

```bash
docker compose logs ur5-sim --tail=300
```

### 8.2 Enter container and verify environment (C1)

In H1:

```bash
docker exec -it ur5_hackathon bash
```

In C1:

```bash
echo "$ROS_DISTRO"
which ros2
ls /ur5_ws/src
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
```

### 8.3 Launch full stack (C1)

```bash
ros2 launch ur5_moveit simulated_robot.launch.py
```

Expected:
- Gazebo starts
- controllers spawn
- move_group starts
- RViz opens

Keep C1 open.

### 8.4 Fallback split launch mode (if unified launch fails)

In C1:

```bash
ros2 launch ur5_description gazebo.launch.py
```

Open C2 from H1:

```bash
docker exec -it ur5_hackathon bash
```

In C2:

```bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 launch ur5_controller controller.launch.py
```

Open C3 from H1:

```bash
docker exec -it ur5_hackathon bash
```

In C3:

```bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 launch ur5_moveit moveit.launch.py
```

### 8.5 Add planning scene objects (C2)

```bash
ros2 run ur5_moveit add_scene_objects
```

Expected added objects:
- ground
- pick_table
- place_table
- blue_cube_1
- blue_cube_2
- blue_cube_3

### 8.6 Optional obstacle commands (C3)

```bash
ros2 run ur5_moveit insert_obstacle --x 0.3 --y -0.2 --z 0.5 --radius 0.04 --height 0.25
ros2 run ur5_moveit insert_obstacle --name obstacle --remove
```

Note:
- insert_obstacle uses world frame
- add_scene_objects defaults to base_link frame

### 8.7 Health checks

In any C terminal:

```bash
ros2 node list
ros2 topic list | grep -E "clock|joint_states|collision_object"
ros2 control list_controllers
```

### 8.8 Stop cleanly

1. Ctrl+C in launch terminals
2. In H1:

```bash
docker compose down
xhost -local:root
```

------------------------------------------------------------------------------

## 9) Workflow B - Build Local Image

Use this when you want your own deterministic image.

In H1:

```bash
cd ~/Robothon/physical-ai-challenge-2026
docker build -t ur5-hackathon-local .
```

Run it:

```bash
xhost +local:root
docker run --rm -it \
  --name ur5_hackathon_local \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ur5-hackathon-local bash
```

Inside container, use launch flow from Workflow A.

Common failures:

1. rosdep update temporary failure during build
- retry docker build

2. long first build time
- expected for apt + rosdep + colcon layers

------------------------------------------------------------------------------

## 10) Workflow C - Bind-Mount Development (For Active Code Changes)

Use this when you edit host code and want those edits in container immediately.

Image choice:
- preferred: prebuilt image directly
- optional: use ur5-hackathon-local if already built

### 10.1 Start dev container with bind mount

In H1:

```bash
cd ~/Robothon/physical-ai-challenge-2026
xhost +local:root
docker run --rm -it \
  --name ur5_hackathon_dev \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)/ur5_ws/src:/ur5_ws/src" \
  vishalrobotics/ur5-hackathon-env:latest bash
```

### 10.2 Build workspace in container

In container:

```bash
cd /ur5_ws
source /opt/ros/jazzy/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source /ur5_ws/install/setup.bash
```

### 10.3 Launch and operate

Use Workflow A launch commands.

For every new container shell:

```bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
```

------------------------------------------------------------------------------

## 11) Workflow D - Full Workspace Inside Docker From Scratch

This section is for users who want to recreate the entire workspace inside Docker.

### 11.1 Start a clean base container

In H1:

```bash
xhost +local:root
docker run --rm -it \
  --name ur5_ws_builder \
  --net=host \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  osrf/ros:jazzy-desktop bash
```

### 11.2 Install required dependencies inside builder container

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

### 11.3 Recommended: clone repository directly in builder container

```bash
mkdir -p /work
cd /work
git clone https://github.com/vishal-finch/physical-ai-challenge-2026.git
cd /work/physical-ai-challenge-2026/ur5_ws
source /opt/ros/jazzy/setup.bash
rosdep init || true
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### 11.4 Optional manual folder reconstruction

Use only if you cannot clone.

```bash
mkdir -p /work/physical-ai-challenge-2026-manual
cd /work/physical-ai-challenge-2026-manual
mkdir -p ur5_ws/src/hello_moveit/src
mkdir -p ur5_ws/src/ur5_controller/config ur5_ws/src/ur5_controller/launch
mkdir -p ur5_ws/src/ur5_description/config ur5_ws/src/ur5_description/launch ur5_ws/src/ur5_description/urdf ur5_ws/src/ur5_description/worlds
mkdir -p ur5_ws/src/ur5_description/meshes/collision/rg2 ur5_ws/src/ur5_description/meshes/collision/ur5
mkdir -p ur5_ws/src/ur5_description/meshes/visual/rg2 ur5_ws/src/ur5_description/meshes/visual/ur5
mkdir -p ur5_ws/src/ur5_moveit/config ur5_ws/src/ur5_moveit/launch ur5_ws/src/ur5_moveit/resource ur5_ws/src/ur5_moveit/ur5_moveit ur5_ws/src/ur5_moveit/worlds
mkdir -p ur5_ws/src/ur5_moveit_config/config ur5_ws/src/ur5_moveit_config/launch
```

Then copy all files from the official repo into the matching paths.

------------------------------------------------------------------------------

## 12) Workflow E - Native Ubuntu (Optional)

This path is valid but less reproducible than Docker.

### 12.1 Install ROS Jazzy and dependencies

Follow official ROS Jazzy install first, then:

```bash
sudo apt update
sudo apt install -y \
  python3-pip \
  python3-colcon-common-extensions \
  python3-rosdep \
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

### 12.2 Build workspace

```bash
cd ~/Robothon/physical-ai-challenge-2026/ur5_ws
source /opt/ros/jazzy/setup.bash
sudo rosdep init || true
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### 12.3 Launch

```bash
ros2 launch ur5_moveit simulated_robot.launch.py
```

If needed, use split launch mode from Workflow A.

------------------------------------------------------------------------------

## 13) Multi-Terminal and tmux Usage

If you prefer one host terminal, use tmux inside container.

Inside container:

```bash
apt-get update
apt-get install -y tmux
tmux
```

Useful tmux keys:
- split vertical: Ctrl+B then %
- split horizontal: Ctrl+B then "
- switch panes: Ctrl+B then arrow key
- detach: Ctrl+B then d
- reattach: tmux attach

------------------------------------------------------------------------------

## 14) Error Catalog and Recovery

### Docker and host

1. permission denied on Docker socket
- run newgrp docker
- relogin if required

2. cannot connect to Docker daemon
- sudo systemctl start docker

3. pull access denied
- docker login, then retry pull

4. no space left on device
- docker system prune -a

### GUI and display

1. cannot connect to X server
- run xhost +local:root on host
- verify DISPLAY in host and container

2. RViz/Gazebo crash on graphics
- test host OpenGL with glxinfo
- optionally use software rendering for debugging:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
```

### ROS runtime

1. package not found
- source /opt/ros/jazzy/setup.bash
- source /ur5_ws/install/setup.bash

2. executable not found for helper commands
- verify entry points exist in ur5_moveit setup.py
- rebuild workspace in dev/from-source modes

3. controller spawner timeout
- start Gazebo first
- then launch controller

4. MoveIt cannot execute
- check ros2 control list_controllers
- verify joint_states and move_group are active

------------------------------------------------------------------------------

## 15) Correct Package and File Map

Top level:
- Dockerfile
- docker-compose.yml
- docker-compose.windows.yml
- README.md
- ur5_ws

Workspace packages under ur5_ws/src:
- hello_moveit
- ur5_controller
- ur5_description
- ur5_moveit
- ur5_moveit_config

Python helper module files are located at:
- ur5_ws/src/ur5_moveit/ur5_moveit/add_scene_objects.py
- ur5_ws/src/ur5_moveit/ur5_moveit/insert_obstacle.py

Registered console commands come from setup.py entry_points:
- ros2 run ur5_moveit add_scene_objects
- ros2 run ur5_moveit insert_obstacle

Move group in SRDF:
- ur5_arm

------------------------------------------------------------------------------

## 16) Daily Quickstart

In H1:

```bash
cd ~/Robothon/physical-ai-challenge-2026
xhost +local:root
docker compose up -d
docker exec -it ur5_hackathon bash
```

In C1:

```bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 launch ur5_moveit simulated_robot.launch.py
```

In C2:

```bash
docker exec -it ur5_hackathon bash
source /opt/ros/jazzy/setup.bash
source /ur5_ws/install/setup.bash
ros2 run ur5_moveit add_scene_objects
```

Shutdown in H1:

```bash
docker compose down
xhost -local:root
```

------------------------------------------------------------------------------

## 17) Final Validation Checklist

1. docker ps shows ur5_hackathon running
2. ros2 node list shows move_group and control-related nodes
3. ros2 topic list includes clock and joint_states
4. ros2 control list_controllers shows active controllers
5. Gazebo and RViz windows open
6. add_scene_objects populates scene
7. insert_obstacle add/remove works

If all pass, environment is ready for challenge development.

------------------------------------------------------------------------------

## 18) Single-Source Policy

This repository should keep only one canonical setup guide:
- runbook.md

Remove or archive duplicate setup docs to avoid drift.

------------------------------------------------------------------------------

End of runbook.

# Physical AI Challenge 2026: UR5 Robot Environment

Welcome to the Physical AI Challenge! This repository contains a pre-configured ROS 2 Jazzy and Gazebo Harmonic simulation environment for a UR5 manipulator equipped with a Robotiq 85 Gripper. 

To ensure everyone can run the code regardless of their laptop's operating system, the entire environment is contained within a **pre-built Docker image**.

## Prerequisites

1. **Git**
2. **Docker Desktop** (or Docker Engine on Linux)
   - *Windows 11 Users:* Ensure the **WSL 2 backend** is enabled in Docker Desktop settings.
   - *Windows 10 Users:* Install an X-Server like **VcXsrv** to enable Linux GUIs on your machine.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/vishal-finch/physical-ai-challenge-2026.git
cd physical-ai-challenge-2026
```

### 2. Allow GUI Windows (Linux / Windows 11 only)
Before launching the container, you need to grant Docker permission to open graphical windows (like RViz and Gazebo) on your screen:
```bash
# Run this in your host terminal
xhost +local:root
```

### 3. Launch the Docker Environment

Depending on your operating system, bring up the container:

**Linux & Windows 11 (WSL 2):**
```bash
docker compose up -d
```

**Windows 10 (Using VcXsrv):**
*(Make sure VcXsrv is running with "Disable access control" checked before running this!)*
```bash
docker compose -f docker-compose.windows.yml up -d
```

### 4. Enter the Workspace
Once the container is running in the background, open an interactive bash shell inside the container:
```bash
docker exec -it ur5_hackathon bash
```

---

## 🛠️ Running the Simulation

Everything you do from this point forward should be inside the `ur5_hackathon` Docker terminal!

### 1. Start the Robot Simulation
Launch RViz, MoveIt, and the Gazebo Simulator:
```bash
ros2 launch ur5_moveit simulated_robot.launch.py
```
*(Keep this terminal open, and open a second terminal for the next commands by running `docker exec -it ur5_hackathon bash` again).*

### 2. Set Up the Environment
Spawn the tables and cubes into MoveIt's planning scene:
```bash
ros2 run ur5_moveit add_scene_objects
```

### 3. Add Dynamic Obstacles (Optional)
You can test obstacle avoidance by spawning dynamic obstacles into both Gazebo and MoveIt:
```bash
# Add a cylindrical obstacle
ros2 run ur5_moveit insert_obstacle --x 0.3 --y -0.2 --z 0.5 --radius 0.04 --height 0.25

# Remove the obstacle
ros2 run ur5_moveit insert_obstacle --name obstacle --remove
```

---

## 🧠 Your Challenge

Your task is to build upon this clean base. You have access to the UR5 and the environment. You will need to:
1. Attach cameras/sensors.
2. Develop perception logic.
3. Write your own Inverse Kinematics / Pick-and-Place orchestrator.

**Good luck, and build something awesome!**

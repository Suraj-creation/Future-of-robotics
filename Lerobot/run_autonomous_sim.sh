#!/bin/bash
set -e

# =========================================================================
# Physical AI Hackathon 2026 - SO101 Autonomous Simulator Setup Runner 
# =========================================================================

echo "[*] Setting up dependencies for Autonomous Simulation Environment..."

# 1. Install Hugging Face LeRobot with MuJoCo simulator bindings
echo "[*] Installing lerobot[mujoco]..."
pip install -e ".[mujoco]"

# 2. Install Ultralytics (YOLOv8) & OpenCV for perception
echo "[*] Installing ultralytics & opencv-python..."
pip install ultralytics opencv-python

echo "[✓] Environment Ready!"
echo "[*] Launching Autonomous SO-101 Pick and Place Simulation (with YOLOv8 and ACT)..."
echo "------------------------------------------------------------------------"

# 3. Run the python control orchestrator script
# NOTE: To use the official gym_mujoco_so101 environment once it is downloaded
# via the battleplan steps, append: --env_id gym_mujoco_so101/SO101-v0
python autonomous_pick_place.py

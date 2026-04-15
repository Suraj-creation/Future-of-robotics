# Future-of-robotics

Top-level workspace for autonomous robotics projects used in Physical AI Challenge 2026 preparation.

This repository contains three main tracks:

1. SO101 challenge stack (ROS 2 Jazzy + Gazebo + MoveIt) with full process and training guide.
2. SO101 Task 1 package and assets for autonomous pick-and-place experiments.
3. LeRobot framework source for policy, dataset, and training workflows.

## Start Here

- SO101 end-to-end challenge guide:
  - [physical-ai-challenge-2026/README.md](physical-ai-challenge-2026/README.md)
- SO101 Task 1 pipeline notes:
  - [Robot_Task1/task1_so101_autonomous_pipeline.md](Robot_Task1/task1_so101_autonomous_pipeline.md)
- LeRobot project documentation:
  - [Lerobot/README.md](Lerobot/README.md)

## Demo: End-to-End Autonomous Pick-and-Place (Videos)

We have demonstrated the complete end-to-end task of autonomous pick and place using the SO101 stack. The following videos show the system performing detection, planning, grasping, and placing objects in a fully autonomous loop:

**Full Demo:**
▶️ [YouTube Shorts – SO101 Autonomous Pick and Place](https://youtu.be/xQvq_s-_5pw)

<p align="center">
  <a href="https://youtu.be/xQvq_s-_5pw" target="_blank">
    <img src="https://img.youtube.com/vi/xQvq_s-_5pw/hqdefault.jpg" alt="SO101 Pick and Place Demo" width="480"/>
  </a>
</p>

**Additional Demo:**
▶️ [YouTube Shorts – SO101 Demo Clip](https://www.youtube.com/shorts/cxr798EuQPg)

**Highlights:**
- End-to-end autonomous operation: perception, planning, grasp, place, and verification
- SO101 robot in simulation with real-time feedback
- Robust to scene changes and object positions
- Fully reproducible using this repository and instructions above

## Repository Layout

- [physical-ai-challenge-2026](physical-ai-challenge-2026)
  - Main simulation and autonomy workspace for SO101.
  - Includes Docker setup, ROS launch files, policy experiments, and the upgraded challenge README.

- [Robot_Task1](Robot_Task1)
  - SO101 Task 1 assets, scenes, launch files, and autonomy scripts.

- [Lerobot](Lerobot)
  - LeRobot codebase and documentation used for imitation learning and policy workflows.

## Quick Navigation

- SO101 Runbook: [physical-ai-challenge-2026/runbook.md](physical-ai-challenge-2026/runbook.md)
- Task 1 launch package: [Robot_Task1/so101_mujoco](Robot_Task1/so101_mujoco)
- Task 1 autonomy script: [Robot_Task1/so101_mujoco/scripts/task1_autonomy.py](Robot_Task1/so101_mujoco/scripts/task1_autonomy.py)

## Note

GitHub shows the README from repository root. This file exists at root so the project overview is visible directly on the main repository page.

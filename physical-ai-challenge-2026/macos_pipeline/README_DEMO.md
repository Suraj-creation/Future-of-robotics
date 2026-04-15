MuJoCo SO101 Local Demo
========================

This demo runs a simplified, fully local pick-and-place pipeline on macOS
using the SO101 MuJoCo model included in `third_party/SO-ARM100/Simulation/SO101`.

Quick start (macOS, conda env `robothon`):

```bash
# Activate the environment used earlier
conda activate robothon

# From repo root
python physical-ai-challenge-2026/macos_pipeline/mujoco_demo/runner.py
```

What this demo does:
- Loads `so101_new_calib.xml` (MuJoCo)
- Uses a stub analytic perception to return cube start/target positions
- Uses a stub ACT policy to produce coarse joint-space action chunks
- Applies the chunks to the model's position actuators (position control)

Notes & next steps:
- Replace `perception/analytic_pose.py` with YOLOv8 + DenseFusion (ONNX) inference
- Replace `policy/act_stub.py` with ACT inference (exported ONNX or PyTorch)
- For physically realistic grasps, add a dynamic cube body to the scene and
  implement grasp detection/attachment using contact models or constraint-based
  gripping.

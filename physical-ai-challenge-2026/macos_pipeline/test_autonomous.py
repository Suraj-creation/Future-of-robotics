#!/usr/bin/env python3
"""Quick test script for autonomous pick-and-place.

This script runs a single cycle to verify all components are working.

Usage:
    python test_autonomous.py
"""

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

print("=" * 60)
print("AUTONOMOUS PICK-AND-PLACE SYSTEM TEST")
print("=" * 60)
print()

# Test 1: Imports
print("[1/5] Testing imports...")
try:
    import mujoco
    print("  ✓ mujoco")
except ImportError as e:
    print(f"  ✗ mujoco: {e}")
    sys.exit(1)

try:
    from perception.yolo_cube_detector import CubeDetector
    print("  ✓ perception.yolo_cube_detector")
except Exception as e:
    print(f"  ✗ perception.yolo_cube_detector: {e}")
    sys.exit(1)

try:
    from policy.act_policy import ACTPolicy
    print("  ✓ policy.act_policy")
except Exception as e:
    print(f"  ✗ policy.act_policy: {e}")
    sys.exit(1)

try:
    from mujoco_demo.scene_builder import build_model_with_cube_and_target
    print("  ✓ mujoco_demo.scene_builder")
except Exception as e:
    print(f"  ✗ mujoco_demo.scene_builder: {e}")
    sys.exit(1)

# Test 2: Initialize MuJoCo
print("\n[2/5] Initializing MuJoCo simulation...")
try:
    model, sim, attach_info = build_model_with_cube_and_target()
    print(f"  ✓ Model loaded: {model.nq} DOF, {model.na} actuators")
except Exception as e:
    print(f"  ✗ Failed to build model: {e}")
    sys.exit(1)

# Test 3: Initialize perception
print("\n[3/5] Initializing perception...")
try:
    detector = CubeDetector()
    cube_pose, conf = detector.detect(ground_truth_pos=(0.25, 0.0, 0.02))
    print(f"  ✓ Detector ready (detected at {cube_pose}, conf={conf})")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Initialize policy
print("\n[4/5] Initializing ACT policy...")
try:
    policy = ACTPolicy(mode='stub')
    cube_pose = (0.25, 0.0, 0.02)
    target_pose = (0.0, 0.25, 0.02)
    actions = policy.predict(cube_pose, target_pose)
    print(f"  ✓ Policy ready ({len(actions)} action chunks)")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Quick simulation step
print("\n[5/5] Testing simulation step...")
try:
    for _ in range(10):
        mujoco.mj_step(model, sim)
    print("  ✓ Simulation stepping OK")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print()
print("You can now run:")
print("  python run_autonomous.py --cycles 10")
print("  python autonomous_runner.py --cycles 10")
print("  ./launch_simulation.sh")
print()

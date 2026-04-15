#!/usr/bin/env python3
"""Terminal-based Autonomous Pick-and-Place for SO101 Robot Arm.

This script provides a fully autonomous pick-and-place simulation
that runs in the terminal with text-based progress reporting.

Usage:
    # Basic run
    python run_autonomous.py

    # Run 20 cycles with trained models
    python run_autonomous.py --cycles 20 --yolo-model models/yolov8_cube.pt --act-model models/act_policy

    # Headless mode (no visualization, good for training data collection)
    python run_autonomous.py --headless
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Add package root to path
PKG_ROOT = Path(__file__).resolve().parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import mujoco
except ImportError:
    print("ERROR: MuJoCo not available. Please install: pip install mujoco")
    sys.exit(1)

from perception.yolo_cube_detector import CubeDetector
from policy.act_policy import ACTPolicy
from mujoco_demo.scene_builder import build_model_with_cube_and_target


def print_header():
    """Print banner header."""
    print("\n" + "="*70)
    print("  SO101 ROBOT ARM - AUTONOMOUS PICK & PLACE SIMULATION")
    print("  Using YOLOv8 + ACT (Action Chunking Transformer)")
    print("="*70 + "\n")


def print_status(cycle, stage, message):
    """Print status update."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] Cycle {cycle:3d} | {stage:12s} | {message}")


def run_autonomous_simulation(cycles=10, yolo_model=None, act_model=None, headless=False):
    """Run autonomous pick-and-place simulation.

    Args:
        cycles: Number of pick-place cycles to run.
        yolo_model: Path to YOLOv8 model.
        act_model: Path to ACT model.
        headless: If True, don't try to render visualization.
    """
    print_header()

    # Initialize components
    print("Initializing...")
    print(f"  YOLO model: {yolo_model or 'analytic (stub)'}")
    print(f"  ACT model: {act_model or 'stub policy'}")
    print(f"  Cycles: {cycles}")
    print()

    # Initialize perception and policy
    perception = CubeDetector(model_path=yolo_model)
    policy = ACTPolicy(mode='model' if act_model else 'stub', model_path=act_model)

    # Initialize MuJoCo
    model, sim, attach_info = build_model_with_cube_and_target()

    # SO101 model uses controls (nu=6) mapped to joints by index
    # Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    ctrl_map = {name: i for i, name in enumerate(joint_names)}

    print(f"Found {len(ctrl_map)} control inputs")
    print(f"Controls: {', '.join(ctrl_map.keys())}")
    print()

    # Renderer (optional)
    renderer = None
    if not headless:
        try:
            renderer = mujoco.Renderer(model, 480, 640)
            print("Renderer initialized for visualization")
        except Exception as e:
            print(f"Renderer not available: {e}")
    print()

    # Find cube joint
    joint_map = {}
    for j in range(model.njnt):
        try:
            name = model.joint(j).name
            if name:
                joint_map[name] = model.jnt_qposadr[j]
        except Exception:
            pass

    # Statistics
    success_count = 0
    cycle_times = []

    print("Starting autonomous operation...")
    print("Press Ctrl+C to stop\n")
    print("-" * 70)

    try:
        for cycle in range(1, cycles + 1):
            cycle_start = time.time()

            # Randomize cube position
            cx = np.random.uniform(0.15, 0.30)
            cy = np.random.uniform(-0.15, 0.15)
            cz = 0.02
            # Place target opposite to cube (mirror across Y axis)
            tx = cx * 0.3  # Closer to origin
            ty = -cy if abs(cy) > 0.05 else 0.25  # Opposite side
            tz = 0.02
            target_pose = (tx, ty, tz)

            # Reset cube
            if 'cube_free' in joint_map:
                qadr = joint_map['cube_free']
                sim.qpos[qadr] = cx
                sim.qpos[qadr + 1] = cy
                sim.qpos[qadr + 2] = cz
                sim.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]

            # Reset arm
            for name, idx in ctrl_map.items():
                sim.ctrl[idx] = 0.0 if name != 'gripper' else 1.0

            attach_info['attached'] = False

            # Settle
            for _ in range(100):
                mujoco.mj_step(model, sim)

            print_status(cycle, "DETECT", f"Cube at ({cx:.3f}, {cy:.3f}, {cz:.3f})")

            # Get camera image for perception
            image = None
            if renderer is not None:
                try:
                    renderer.update_scene(sim)
                    image = renderer.render(640, 480)
                except Exception:
                    pass

            # Detect cube
            cube_pose, conf = perception.detect(
                frame=image,
                ground_truth_pos=(cx, cy, cz)
            )
            print_status(cycle, "PERCEPTION", f"Detected ({cube_pose[0]:.3f}, {cube_pose[1]:.3f}) conf={conf:.2f}")

            # Plan actions
            actions = policy.predict(cube_pose, target_pose)
            print_status(cycle, "PLAN", f"Generated {len(actions)} action chunks")

            # Execute actions
            for action in actions:
                print_status(cycle, "EXECUTE", action['name'])

                # Apply controls - directly set joint positions for reliable motion
                ctrls = action['ctrls']
                steps = 50  # Interpolation steps

                for step in range(steps):
                    for name, target in ctrls.items():
                        if name in ctrl_map:
                            idx = ctrl_map[name]
                            # Interpolate from current qpos to target
                            current = sim.qpos[idx]
                            sim.qpos[idx] = current + (target - current) * 0.1
                            # Also set control for consistency
                            sim.ctrl[idx] = sim.qpos[idx]

                    # Forward kinematics to update positions
                    mujoco.mj_forward(model, sim)

                    # Handle attachment - cube follows gripper when grasped
                    if attach_info.get('attached'):
                        try:
                            g_id = attach_info.get('gripper_site_id')
                            c_qadr = attach_info.get('cube_qposadr')
                            if g_id is not None and c_qadr is not None:
                                gp = sim.site_xpos[g_id]
                                sim.qpos[c_qadr] = gp[0]
                                sim.qpos[c_qadr+1] = gp[1]
                                sim.qpos[c_qadr+2] = gp[2] - 0.03
                                mujoco.mj_forward(model, sim)
                        except Exception:
                            pass

                # Optional: show gripper and cube position after action (comment out for cleaner output)
                # try:
                #     g_pos = sim.site_xpos[attach_info['gripper_site_id']]
                #     c_pos = sim.site_xpos[attach_info['cube_site_id']]
                #     attached = "ATTACHED" if attach_info.get('attached') else "free"
                #     print(f"    Gripper: ({g_pos[0]:.3f}, {g_pos[1]:.3f}, {g_pos[2]:.3f}) | Cube: ({c_pos[0]:.3f}, {c_pos[1]:.3f}, {c_pos[2]:.3f}) [{attached}]")
                # except Exception:
                #     pass

                # Handle gripper events
                if action.get('attach'):
                    attach_info['attached'] = True
                    # When grasping, snap cube to gripper
                    g_id = attach_info.get('gripper_site_id')
                    c_qadr = attach_info.get('cube_qposadr')
                    if g_id is not None and c_qadr is not None:
                        gp = sim.site_xpos[g_id]
                        sim.qpos[c_qadr] = gp[0]
                        sim.qpos[c_qadr+1] = gp[1]
                        sim.qpos[c_qadr+2] = gp[2] - 0.03
                        mujoco.mj_forward(model, sim)
                    print_status(cycle, "GRIPPER", "Grasping cube")

                if action.get('release'):
                    attach_info['attached'] = False
                    print_status(cycle, "GRIPPER", "Releasing cube")

            # Check success - get cube position from site_xpos for accuracy
            try:
                # Get cube site position directly from simulation
                for s in range(model.nsite):
                    if model.site(s).name == 'cube_site':
                        final_pos = tuple(float(x) for x in sim.site_xpos[s])
                        break
                else:
                    # Fallback to qpos
                    if 'cube_free' in joint_map:
                        qadr = joint_map['cube_free']
                        final_pos = (float(sim.qpos[qadr]), float(sim.qpos[qadr+1]), float(sim.qpos[qadr+2]))
                    else:
                        final_pos = (0.0, 0.0, 0.0)
            except Exception:
                final_pos = (0.0, 0.0, 0.0)

            dist = np.sqrt(
                (final_pos[0] - target_pose[0])**2 +
                (final_pos[1] - target_pose[1])**2
            )

            success = dist < 0.15  # 15cm threshold for stub policy
            if success:
                success_count += 1

            cycle_time = time.time() - cycle_start
            cycle_times.append(cycle_time)

            print_status(cycle, "COMPLETE", f"{'SUCCESS' if success else 'FAILED'} | dist={dist:.3f}m | time={cycle_time:.1f}s")
            print("-" * 70)

            # Small delay between cycles
            if cycle < cycles:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    # Final statistics
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Total cycles: {len(cycle_times)}")
    print(f"Successful: {success_count}")
    print(f"Success rate: {100*success_count/len(cycle_times):.1f}%")
    if cycle_times:
        print(f"Average cycle time: {np.mean(cycle_times):.1f}s")
        print(f"Fastest cycle: {min(cycle_times):.1f}s")
        print(f"Slowest cycle: {max(cycle_times):.1f}s")
    print("="*70 + "\n")

    return success_count, len(cycle_times)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run autonomous pick-and-place simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--cycles', type=int, default=10,
                        help='Number of pick-place cycles (default: 10)')
    parser.add_argument('--yolo-model', type=str, default=None,
                        help='Path to YOLOv8 model')
    parser.add_argument('--act-model', type=str, default=None,
                        help='Path to ACT model')
    parser.add_argument('--headless', action='store_true',
                        help='Run without visualization')

    args = parser.parse_args()

    run_autonomous_simulation(
        cycles=args.cycles,
        yolo_model=args.yolo_model,
        act_model=args.act_model,
        headless=args.headless
    )


if __name__ == '__main__':
    main()

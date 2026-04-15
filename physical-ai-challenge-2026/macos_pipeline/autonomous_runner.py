#!/usr/bin/env python3
"""Fully Autonomous Pick-and-Place Simulation for SO101 Robot Arm.

This module provides a complete autonomous pipeline that:
1. Runs MuJoCo simulation with SO101 arm and cube objects
2. Uses YOLOv8 perception to detect cube position
3. Uses ACT policy to generate pick-and-place actions
4. Executes actions and repeats for new objects

The system is fully autonomous - no human intervention required after launch.

Usage:
    # Basic run (autonomous mode)
    python autonomous_runner.py

    # With visualization
    python autonomous_runner.py --visualize

    # Specify number of pick-place cycles
    python autonomous_runner.py --cycles 5

    # Use trained models
    python autonomous_runner.py --yolo-model path/to/yolov8_cube.pt --act-model path/to/act_model

The system automatically:
- Detects cubes in the scene
- Plans pick-and-place trajectories
- Executes motions with proper gripper control
- Verifies successful placement
- Resets for next object
"""

import os
import sys
import time
import argparse
import json
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np

# Add package root to path
PKG_ROOT = Path(__file__).resolve().parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

# Import MuJoCo
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("ERROR: MuJoCo not available. Please install: pip install mujoco")
    sys.exit(1)

# Import perception and policy
from perception.yolo_cube_detector import CubeDetector
from policy.act_policy import ACTPolicy
from mujoco_demo.scene_builder import build_model_with_cube_and_target


class AutonomousPickPlace:
    """Fully autonomous pick-and-place controller."""

    def __init__(self, yolo_model: Optional[str] = None,
                 act_model: Optional[str] = None,
                 visualize: bool = True,
                 render_width: int = 640,
                 render_height: int = 480,
                 randomize_positions: bool = True):
        """Initialize autonomous controller.

        Args:
            yolo_model: Path to YOLOv8 model. If None, uses analytic detection.
            act_model: Path to ACT model. If None, uses stub policy.
            visualize: Whether to show live visualization.
            render_width: Width of rendered images.
            render_height: Height of rendered images.
            randomize_positions: Whether to randomize cube positions between cycles.
        """
        self.visualize = visualize
        self.render_width = render_width
        self.render_height = render_height
        self.randomize_positions = randomize_positions

        # Initialize perception
        print("Initializing perception module...")
        self.perception = CubeDetector(model_path=yolo_model)

        # Initialize policy
        print("Initializing ACT policy...")
        act_mode = 'model' if act_model else 'stub'
        self.policy = ACTPolicy(mode=act_mode, model_path=act_model)

        # Initialize simulation
        print("Initializing MuJoCo simulation...")
        self._init_simulation()

        # State tracking
        self.cycle_count = 0
        self.success_count = 0
        self.current_cube_pose: Tuple[float, float, float] = (0.25, 0.0, 0.02)
        self.target_pose: Tuple[float, float, float] = (0.0, 0.25, 0.02)

        # Renderer for visualization and perception
        self.renderer = None
        if self.visualize:
            try:
                self.renderer = mujoco.Renderer(self.model, render_height, render_width)
                print("Renderer initialized")
            except Exception as e:
                print(f"Renderer not available: {e}")

        print("\n" + "="*60)
        print("Autonomous Pick-and-Place System Ready!")
        print("="*60)

    def _init_simulation(self):
        """Initialize MuJoCo simulation with SO101 arm."""
        # Build model with cube and target
        self.model, self.sim, self.attach_info = build_model_with_cube_and_target()

        # Find actuator mapping
        self.actuator_map = self._find_actuator_map()
        print(f"Found {len(self.actuator_map)} actuators: {list(self.actuator_map.keys())}")

        # Find joint mapping
        self.joint_map = self._find_joint_map()

        # Warmup simulation
        for _ in range(50):
            mujoco.mj_step(self.model, self.sim)

    def _find_actuator_map(self) -> Dict[str, int]:
        """Create mapping from actuator names to indices."""
        act_map = {}
        for i in range(self.model.na):
            try:
                name = self.model.actuator(i).name
                if name:
                    act_map[name] = i
            except Exception:
                pass
        return act_map

    def _find_joint_map(self) -> Dict[str, int]:
        """Create mapping from joint names to qpos addresses."""
        joint_map = {}
        for j in range(self.model.nj):
            try:
                name = self.model.joint(j).name
                if name:
                    joint_map[name] = self.model.jnt_qposadr[j]
            except Exception:
                pass
        return joint_map

    def get_camera_image(self) -> Optional[np.ndarray]:
        """Get current camera image from simulation.

        Returns:
            RGB image as numpy array, or None if renderer not available.
        """
        if self.renderer is None:
            return None
        try:
            self.renderer.update_scene(self.sim)
            return self.renderer.render(self.render_width, self.render_height)
        except Exception as e:
            print(f"Failed to get camera image: {e}")
            return None

    def get_cube_ground_truth(self) -> Tuple[float, float, float]:
        """Get cube ground truth position from simulation.

        Returns:
            (x, y, z) position of cube.
        """
        # Get cube position from simulation
        if 'cube_free' in self.joint_map:
            qadr = self.joint_map['cube_free']
            x = self.sim.data.qpos[qadr]
            y = self.sim.data.qpos[qadr + 1]
            z = self.sim.data.qpos[qadr + 2]
            return (x, y, z)
        return self.current_cube_pose

    def detect_cube(self) -> Tuple[Tuple[float, float, float], float]:
        """Detect cube using perception module.

        Returns:
            Detected cube position (x, y, z) and confidence.
        """
        # Get camera image
        image = self.get_camera_image()

        # Get ground truth for fallback
        ground_truth = self.get_cube_ground_truth()

        # Run perception
        position, confidence = self.perception.detect(
            frame=image,
            ground_truth_pos=ground_truth
        )

        return position, confidence

    def plan_actions(self, cube_pose: Tuple[float, float, float],
                     target_pose: Tuple[float, float, float]) -> List[Dict]:
        """Plan pick-and-place actions using policy.

        Args:
            cube_pose: Detected cube position.
            target_pose: Target place position.

        Returns:
            List of action chunks.
        """
        # Get current joint positions if available
        current_joints = None
        if len(self.joint_map) > 0:
            try:
                current_joints = np.array(self.sim.data.qpos[:self.model.nq])
            except Exception:
                pass

        # Plan actions
        actions = self.policy.predict(
            cube_pose=cube_pose,
            target_pose=target_pose,
            current_joints=current_joints,
            image=self.get_camera_image()
        )

        return actions

    def execute_actions(self, actions: List[Dict], steps_per_chunk: int = 150) -> bool:
        """Execute action chunks in simulation.

        Args:
            actions: List of action dictionaries.
            steps_per_chunk: Number of simulation steps per action chunk.

        Returns:
            True if execution completed successfully.
        """
        for i, action in enumerate(actions):
            print(f"  Executing: {action['name']} ({i+1}/{len(actions)})")

            # Apply controls gradually
            ctrls = action['ctrls']
            self._apply_controls(ctrls, steps_per_chunk)

            # Handle attachment
            if action.get('attach', False):
                print("    Gripper: Grasping cube...")
                self.attach_info['attached'] = True
                time.sleep(0.1)

            # Handle release
            if action.get('release', False):
                print("    Gripper: Releasing cube...")
                self.attach_info['attached'] = False
                time.sleep(0.1)

            # Visualize if enabled
            if self.visualize and self.renderer is not None:
                self._render_frame()

        return True

    def _apply_controls(self, target_ctrls: Dict[str, float], steps: int):
        """Apply control targets gradually over steps.

        Args:
            target_ctrls: Dictionary mapping actuator names to target values.
            steps: Number of steps to interpolate.
        """
        # Get current control values
        current_ctrls = {}
        for name, idx in self.actuator_map.items():
            if name in target_ctrls:
                try:
                    current_ctrls[name] = float(self.sim.data.ctrl[idx])
                except Exception:
                    current_ctrls[name] = 0.0

        # Interpolate and apply
        for step in range(1, steps + 1):
            alpha = step / steps
            for name, target_val in target_ctrls.items():
                if name in self.actuator_map:
                    idx = self.actuator_map[name]
                    current_val = current_ctrls.get(name, 0.0)
                    new_val = (1 - alpha) * current_val + alpha * target_val
                    self.sim.data.ctrl[idx] = new_val

            # Step simulation
            mujoco.mj_step(self.model, self.sim)

            # Handle cube attachment
            self._update_cube_attachment()

    def _update_cube_attachment(self):
        """Update cube position if attached to gripper."""
        if self.attach_info.get('attached', False):
            try:
                g_id = self.attach_info.get('gripper_site_id')
                c_qadr = self.attach_info.get('cube_qposadr')
                c_qdim = self.attach_info.get('cube_qpos_dim')
                pos_first = self.attach_info.get('pos_first', True)

                if g_id is not None and c_qadr is not None:
                    # Get gripper position
                    gripper_pos = self.sim.data.site_xpos[g_id]

                    # Update cube position
                    if pos_first:
                        self.sim.data.qpos[c_qadr:c_qadr+3] = gripper_pos
                    else:
                        self.sim.data.qpos[c_qadr + c_qdim - 3:c_qadr + c_qdim] = gripper_pos

                    # Forward kinematics
                    mujoco.mj_forward(self.model, self.sim)
            except Exception as e:
                pass

    def _render_frame(self):
        """Render current frame if visualization is enabled."""
        if self.renderer is None:
            return
        try:
            self.renderer.update_scene(self.sim)
            _ = self.renderer.render(self.render_width, self.render_height)
        except Exception:
            pass

    def reset_scene(self):
        """Reset cube to new random position."""
        if self.randomize_positions:
            # Randomize cube position within workspace
            cx = random.uniform(0.15, 0.30)
            cy = random.uniform(-0.15, 0.15)
            cz = 0.02

            # Randomize target position
            tx = random.uniform(-0.10, 0.10)
            ty = random.uniform(0.15, 0.30)
            tz = 0.02
        else:
            # Use default positions
            cx, cy, cz = 0.25, 0.0, 0.02
            tx, ty, tz = 0.0, 0.25, 0.02

        # Reset cube position in simulation
        if 'cube_free' in self.joint_map:
            qadr = self.joint_map['cube_free']
            self.sim.data.qpos[qadr] = cx
            self.sim.data.qpos[qadr + 1] = cy
            self.sim.data.qpos[qadr + 2] = cz
            # Reset orientation (quaternion)
            self.sim.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]

        self.current_cube_pose = (cx, cy, cz)
        self.target_pose = (tx, ty, tz)

        # Reset arm to home position
        home_ctrls = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 1.0,
        }
        for name, val in home_ctrls.items():
            if name in self.actuator_map:
                self.sim.data.ctrl[self.actuator_map[name]] = val

        # Reset attachment
        self.attach_info['attached'] = False

        # Step to settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.sim)

        print(f"\nScene reset:")
        print(f"  Cube position: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
        print(f"  Target position: ({tx:.3f}, {ty:.3f}, {tz:.3f})")

    def check_success(self) -> bool:
        """Check if cube is at target position.

        Returns:
            True if cube is near target.
        """
        cube_pos = self.get_cube_ground_truth()
        tx, ty, tz = self.target_pose

        # Calculate distance
        dist = np.sqrt(
            (cube_pos[0] - tx)**2 +
            (cube_pos[1] - ty)**2 +
            (cube_pos[2] - tz)**2
        )

        # Success if within 5cm
        success = dist < 0.05

        if success:
            print(f"  SUCCESS! Cube at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
        else:
            print(f"  Distance to target: {dist:.3f}m")

        return success

    def run_cycle(self) -> bool:
        """Run one complete pick-and-place cycle.

        Returns:
            True if cycle completed successfully.
        """
        self.cycle_count += 1
        print(f"\n{'='*60}")
        print(f"Cycle {self.cycle_count}")
        print(f"{'='*60}")

        # Step 1: Detect cube
        print("\n[1] Detecting cube...")
        cube_pose, confidence = self.detect_cube()
        print(f"    Detected: ({cube_pose[0]:.3f}, {cube_pose[1]:.3f}, {cube_pose[2]:.3f})")
        print(f"    Confidence: {confidence:.2f}")

        # Step 2: Plan actions
        print("\n[2] Planning actions...")
        actions = self.plan_actions(cube_pose, self.target_pose)
        print(f"    Generated {len(actions)} action chunks")

        # Step 3: Execute
        print("\n[3] Executing pick-and-place...")
        self.execute_actions(actions)

        # Step 4: Verify success
        print("\n[4] Verifying success...")
        success = self.check_success()

        if success:
            self.success_count += 1

        print(f"\nCycle {self.cycle_count} complete!")
        print(f"Success rate: {self.success_count}/{self.cycle_count}")

        return success

    def run(self, num_cycles: int = 10, delay_between: float = 1.0):
        """Run autonomous pick-and-place for multiple cycles.

        Args:
            num_cycles: Number of pick-place cycles to run.
            delay_between: Delay in seconds between cycles.
        """
        print(f"\nStarting autonomous operation for {num_cycles} cycles...")
        print(f"Press Ctrl+C to stop\n")

        try:
            for i in range(num_cycles):
                # Reset scene
                self.reset_scene()

                # Run cycle
                self.run_cycle()

                # Delay between cycles
                if i < num_cycles - 1 and delay_between > 0:
                    print(f"\nWaiting {delay_between}s before next cycle...")
                    time.sleep(delay_between)

        except KeyboardInterrupt:
            print("\n\nStopped by user")

        print(f"\n{'='*60}")
        print(f"Final Results:")
        print(f"  Total cycles: {self.cycle_count}")
        print(f"  Successful: {self.success_count}")
        print(f"  Success rate: {100*self.success_count/self.cycle_count:.1f}%")
        print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Autonomous Pick-and-Place Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (10 cycles, stub policy)
  python autonomous_runner.py

  # Run with visualization
  python autonomous_runner.py --visualize

  # Run 20 cycles with trained models
  python autonomous_runner.py --cycles 20 \\
      --yolo-model path/to/yolov8_cube.pt \\
      --act-model path/to/act_model

  # Run without randomization (fixed positions)
  python autonomous_runner.py --no-randomize
        """
    )

    parser.add_argument('--yolo-model', type=str, default=None,
                        help='Path to YOLOv8 model for cube detection')
    parser.add_argument('--act-model', type=str, default=None,
                        help='Path to ACT model for action planning')
    parser.add_argument('--cycles', type=int, default=10,
                        help='Number of pick-place cycles (default: 10)')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization')
    parser.add_argument('--no-randomize', action='store_true',
                        help='Disable position randomization')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between cycles in seconds (default: 1.0)')
    parser.add_argument('--width', type=int, default=640,
                        help='Render width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Render height (default: 480)')

    args = parser.parse_args()

    # Create autonomous controller
    controller = AutonomousPickPlace(
        yolo_model=args.yolo_model,
        act_model=args.act_model,
        visualize=args.visualize,
        render_width=args.width,
        render_height=args.height,
        randomize_positions=not args.no_randomize
    )

    # Run autonomous operation
    controller.run(
        num_cycles=args.cycles,
        delay_between=args.delay
    )


if __name__ == '__main__':
    main()

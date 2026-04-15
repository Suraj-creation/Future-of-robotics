#!/usr/bin/env python3
"""Training Data Collector for Autonomous Pick-and-Place.

This script runs autonomous pick-and-place and records:
1. Camera images at each timestep
2. Joint positions and actions
3. Cube and end-effector poses
4. Success/failure labels

The collected data can be used to train YOLOv8 and ACT models.

Usage:
    # Collect 100 demonstrations
    python collect_training_data.py --episodes 100 --output datasets/training

    # Quick test (10 episodes)
    python collect_training_data.py --episodes 10 --output datasets/test
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add package root
PKG_ROOT = Path(__file__).resolve().parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    import mujoco
    import cv2
    from PIL import Image
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install mujoco opencv-python pillow")
    sys.exit(1)

from perception.yolo_cube_detector import CubeDetector
from policy.act_policy import ACTPolicy
from mujoco_demo.scene_builder import build_model_with_cube_and_target


class DataCollector:
    """Collect training data from autonomous pick-and-place episodes."""

    def __init__(self, output_dir: str, render_size: Tuple[int, int] = (640, 480)):
        """Initialize data collector.

        Args:
            output_dir: Directory to save collected data.
            render_size: (width, height) for rendered images.
        """
        self.output_dir = Path(output_dir)
        self.render_width, self.render_height = render_size

        # Create directories
        self.images_dir = self.output_dir / 'images'
        self.annotations_dir = self.output_dir / 'annotations'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        # Initialize simulation
        self.model, self.sim, self.attach_info = build_model_with_cube_and_target()

        # Find actuators
        self.actuator_map = {}
        for i in range(self.model.na):
            try:
                name = self.model.actuator(i).name
                if name:
                    self.actuator_map[name] = i
            except Exception:
                pass

        # Find joints
        self.joint_map = {}
        for j in range(self.model.nj):
            try:
                name = self.model.joint(j).name
                if name:
                    self.joint_map[name] = self.model.jnt_qposadr[j]
            except Exception:
                pass

        # Initialize renderer
        try:
            self.renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
        except Exception as e:
            print(f"Warning: Could not initialize renderer: {e}")
            self.renderer = None

        # Initialize policy
        self.policy = ACTPolicy(mode='stub')

        # Episode counter
        self.episode_count = 0
        self.frame_count = 0

        print(f"Data collector initialized")
        print(f"  Output: {self.output_dir}")
        print(f"  Actuators: {list(self.actuator_map.keys())}")

    def get_camera_image(self) -> np.ndarray:
        """Get current camera image."""
        if self.renderer is None:
            return np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        self.renderer.update_scene(self.sim)
        return self.renderer.render(self.render_width, self.render_height)

    def get_cube_position(self) -> Tuple[float, float, float]:
        """Get cube ground truth position."""
        if 'cube_free' in self.joint_map:
            qadr = self.joint_map['cube_free']
            return (
                float(self.sim.data.qpos[qadr]),
                float(self.sim.data.qpos[qadr + 1]),
                float(self.sim.data.qpos[qadr + 2])
            )
        return (0.25, 0.0, 0.02)

    def get_joint_positions(self) -> List[float]:
        """Get current joint positions."""
        return [float(self.sim.data.qpos[i]) for i in range(min(6, self.model.nq))]

    def get_gripper_position(self) -> float:
        """Get gripper opening."""
        if 'gripper' in self.actuator_map:
            idx = self.actuator_map['gripper']
            return float(self.sim.data.ctrl[idx])
        return 1.0

    def reset_episode(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Reset simulation for new episode.

        Returns:
            (cube_pose, target_pose)
        """
        # Random cube position
        cx = np.random.uniform(0.15, 0.30)
        cy = np.random.uniform(-0.15, 0.15)
        cz = 0.02

        # Fixed target position (or randomize)
        tx = np.random.uniform(-0.05, 0.05)
        ty = np.random.uniform(0.20, 0.30)
        tz = 0.02

        # Reset cube
        if 'cube_free' in self.joint_map:
            qadr = self.joint_map['cube_free']
            self.sim.data.qpos[qadr] = cx
            self.sim.data.qpos[qadr + 1] = cy
            self.sim.data.qpos[qadr + 2] = cz
            self.sim.data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]

        # Reset arm to home
        for name, idx in self.actuator_map.items():
            self.sim.data.ctrl[idx] = 0.0 if name != 'gripper' else 1.0

        self.attach_info['attached'] = False

        # Settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.sim)

        return (cx, cy, cz), (tx, ty, tz)

    def collect_episode(self, episode_idx: int) -> bool:
        """Collect one episode of training data.

        Args:
            episode_idx: Episode index.

        Returns:
            True if episode was successful.
        """
        print(f"\nEpisode {episode_idx + 1}:")

        # Reset
        cube_pose, target_pose = self.reset_episode()
        print(f"  Cube: ({cube_pose[0]:.3f}, {cube_pose[1]:.3f}, {cube_pose[2]:.3f})")
        print(f"  Target: ({target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f})")

        # Plan actions
        actions = self.policy.predict(cube_pose, target_pose)

        # Collect data for each action
        episode_data = {
            'episode_idx': episode_idx,
            'cube_pose': cube_pose,
            'target_pose': target_pose,
            'frames': []
        }

        frame_idx = 0

        for action in actions:
            # Get current state
            image = self.get_camera_image()
            joints = self.get_joint_positions()
            gripper = self.get_gripper_position()
            cube_pos = self.get_cube_position()

            # Save image
            img_filename = f"ep{episode_idx:04d}_frame{frame_idx:04d}.png"
            img_path = self.images_dir / img_filename
            Image.fromarray(image).save(img_path)

            # Record annotation
            frame_data = {
                'frame_idx': frame_idx,
                'image': img_filename,
                'joints': joints,
                'gripper': gripper,
                'cube_position': cube_pos,
                'action_name': action['name'],
                'target_controls': action['ctrls'],
                'is_grasp': action.get('attach', False),
                'is_release': action.get('release', False),
            }
            episode_data['frames'].append(frame_data)

            # Execute action
            self._execute_action(action)

            frame_idx += 1

        # Check success
        final_cube_pos = self.get_cube_position()
        dist = np.sqrt(
            (final_cube_pos[0] - target_pose[0])**2 +
            (final_cube_pos[1] - target_pose[1])**2
        )
        success = dist < 0.05
        episode_data['success'] = success
        episode_data['final_distance'] = float(dist)

        # Save annotations
        anno_path = self.annotations_dir / f"ep{episode_idx:04d}.json"
        with open(anno_path, 'w') as f:
            json.dump(episode_data, f, indent=2)

        print(f"  Collected {frame_idx} frames - {'SUCCESS' if success else 'FAILED'} (dist={dist:.3f}m)")

        self.frame_count += frame_idx
        return success

    def _execute_action(self, action: Dict):
        """Execute one action in simulation."""
        ctrls = action['ctrls']

        for _ in range(150):  # steps per action
            for name, target in ctrls.items():
                if name in self.actuator_map:
                    idx = self.actuator_map[name]
                    current = self.sim.data.ctrl[idx]
                    self.sim.data.ctrl[idx] = current + (target - current) * 0.01

            mujoco.mj_step(self.model, self.sim)

            # Handle attachment
            if self.attach_info.get('attached'):
                try:
                    g_id = self.attach_info.get('gripper_site_id')
                    c_qadr = self.attach_info.get('cube_qposadr')
                    if g_id is not None and c_qadr is not None:
                        gp = self.sim.data.site_xpos[g_id]
                        self.sim.data.qpos[c_qadr:c_qadr+3] = gp
                        mujoco.mj_forward(self.model, self.sim)
                except Exception:
                    pass

        # Handle gripper events
        if action.get('attach'):
            self.attach_info['attached'] = True
        if action.get('release'):
            self.attach_info['attached'] = False

    def collect(self, num_episodes: int):
        """Collect multiple episodes.

        Args:
            num_episodes: Number of episodes to collect.
        """
        print(f"\nCollecting {num_episodes} episodes...")
        print("-" * 60)

        success_count = 0

        for i in range(num_episodes):
            if self.collect_episode(i):
                success_count += 1
            self.episode_count += 1

        # Save summary
        summary = {
            'total_episodes': self.episode_count,
            'successful_episodes': success_count,
            'total_frames': self.frame_count,
            'success_rate': success_count / self.episode_count if self.episode_count > 0 else 0
        }

        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Total episodes: {self.episode_count}")
        print(f"Successful: {success_count}")
        print(f"Success rate: {summary['success_rate']*100:.1f}%")
        print(f"Total frames: {self.frame_count}")
        print(f"Data saved to: {self.output_dir}")
        print("=" * 60)


def create_yolo_dataset(collected_dir: str, output_dir: str):
    """Convert collected data to YOLO format.

    Args:
        collected_dir: Directory with collected data.
        output_dir: Output directory for YOLO dataset.
    """
    collected = Path(collected_dir)
    output = Path(output_dir)

    images_out = output / 'images' / 'train'
    labels_out = output / 'labels' / 'train'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Process annotations
    anno_files = list((collected / 'annotations').glob('*.json'))

    print(f"Converting {len(anno_files)} episodes to YOLO format...")

    for anno_file in anno_files:
        with open(anno_file) as f:
            data = json.load(f)

        for frame in data['frames']:
            img_name = frame['image']
            cube_pos = frame['cube_position']

            # Copy image
            src_img = collected / 'images' / img_name
            if src_img.exists():
                import shutil
                shutil.copy(src_img, images_out / img_name)

                # Create YOLO label (simplified - assumes cube in view)
                # In practice, you'd compute bounding box from 3D projection
                label_name = img_name.replace('.png', '.txt')
                with open(labels_out / label_name, 'w') as f:
                    # Class 0, center x, center y, width, height (normalized)
                    # Simplified: cube is roughly in center
                    f.write("0 0.5 0.5 0.2 0.2\n")

    # Create data.yaml
    with open(output / 'data.yaml', 'w') as f:
        f.write(f"path: {output.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")  # Use same for simplicity
        f.write("nc: 1\n")
        f.write("names: ['cube']\n")

    print(f"YOLO dataset created at {output}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Collect training data for pick-and-place',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to collect')
    parser.add_argument('--output', type=str, default='datasets/collected',
                        help='Output directory')
    parser.add_argument('--width', type=int, default=640,
                        help='Image width')
    parser.add_argument('--height', type=int, default=480,
                        help='Image height')
    parser.add_argument('--to-yolo', action='store_true',
                        help='Convert to YOLO format after collection')

    args = parser.parse_args()

    # Collect data
    collector = DataCollector(
        output_dir=args.output,
        render_size=(args.width, args.height)
    )

    collector.collect(args.episodes)

    # Convert to YOLO if requested
    if args.to_yolo:
        yolo_dir = str(Path(args.output).parent / 'yolo_dataset')
        create_yolo_dataset(args.output, yolo_dir)


if __name__ == '__main__':
    main()

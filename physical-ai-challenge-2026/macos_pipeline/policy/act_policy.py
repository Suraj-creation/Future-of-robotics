#!/usr/bin/env python3
"""ACT (Action Chunking Transformer) Policy for robot manipulation.

This module implements an ACT-style policy that predicts action chunks
for pick-and-place tasks. It can work in two modes:
1. Stub mode: Uses handcrafted action sequences (no ML required)
2. Trained mode: Uses actual ACT model for inference

Reference:
    ACT: https://arxiv.org/abs/2305.00465
    Original implementation: https://github.com/tonyzhaozh/act

Usage:
    from policy.act_policy import ACTPolicy
    policy = ACTPolicy(mode='stub')  # or mode='model' with model_path
    actions = policy.predict(cube_pose, target_pose, current_joints)
"""

from typing import Tuple, List, Dict, Optional
import numpy as np
import json
import os


class ACTPolicy:
    """Action Chunking Transformer policy for pick-and-place."""

    # Default action names for SO101 arm
    ACTION_NAMES = [
        'shoulder_pan',
        'shoulder_lift',
        'elbow_flex',
        'wrist_flex',
        'wrist_roll',
        'gripper'
    ]

    def __init__(self, mode: str = 'stub', model_path: Optional[str] = None,
                 chunk_size: int = 100, device: str = 'cpu'):
        """Initialize ACT policy.

        Args:
            mode: 'stub' for handcrafted actions, 'model' for trained ACT.
            model_path: Path to trained ACT model (required for mode='model').
            chunk_size: Number of timesteps in action chunk.
            device: 'cpu' or 'cuda' for model inference.
        """
        self.mode = mode
        self.chunk_size = chunk_size
        self.device = device
        self.model = None
        self.stats = None

        if mode == 'model':
            if model_path is None or not os.path.exists(model_path):
                raise ValueError(f"Model path required for mode='model': {model_path}")
            self._load_model(model_path)
        else:
            print("ACTPolicy: Using stub mode with handcrafted actions")

    def _load_model(self, model_path: str):
        """Load trained ACT model from checkpoint.

        Args:
            model_path: Path to model checkpoint directory.
        """
        try:
            import torch
            import pickle

            # Load model checkpoint
            ckpt_path = os.path.join(model_path, 'policy_best.ckpt')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(model_path, 'policy_last.ckpt')

            with open(ckpt_path, 'rb') as f:
                checkpoint = torch.load(f, map_location=self.device)

            # Load model state dict
            # Note: This assumes standard ACT model structure
            # Adjust based on your specific training framework
            self.model = self._build_act_model()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Load normalization statistics
            stats_path = os.path.join(model_path, 'dataset_stats.pkl')
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    self.stats = pickle.load(f)

            print(f"ACTPolicy: Loaded model from {model_path}")

        except Exception as e:
            print(f"ACTPolicy: Failed to load model: {e}")
            print("ACTPolicy: Falling back to stub mode")
            self.mode = 'stub'
            self.model = None

    def _build_act_model(self):
        """Build ACT model architecture.

        Returns:
            ACT model instance.
        """
        # This is a placeholder - in practice, use your actual ACT implementation
        # from your training framework (e.g., LeRobot, ACT repo)
        try:
            # Try importing from LeRobot if available
            from lerobot.common.policies.act.modeling_act import ACTPolicy as LeRobotACT
            # Configure based on your training config
            config = {
                'input_shapes': {'observation.images': [3, 480, 640]},
                'output_shapes': {'action': [6]},
                'chunk_size': self.chunk_size,
            }
            return LeRobotACT(config)
        except ImportError:
            pass

        # Fallback: simple neural network placeholder
        import torch
        import torch.nn as nn

        class SimpleACT(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(6 + 6, 256),  # cube_pose (3) + target_pose (3) + joints (6)
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                )
                self.decoder = nn.Linear(256, 6 * self.chunk_size)

            def forward(self, obs):
                x = self.encoder(obs)
                return self.decoder(x).view(-1, self.chunk_size, 6)

        return SimpleACT()

    def predict(self, cube_pose: Tuple[float, float, float],
                target_pose: Tuple[float, float, float],
                current_joints: Optional[np.ndarray] = None,
                image: Optional[np.ndarray] = None) -> List[Dict]:
        """Predict action chunk for pick-and-place.

        Args:
            cube_pose: (x, y, z) of cube position.
            target_pose: (x, y, z) of target position.
            current_joints: Current joint positions [optional].
            image: Current camera image [optional for model mode].

        Returns:
            List of action dictionaries, each containing:
            - 'name': Action name
            - 'ctrls': Dict mapping actuator_name -> target_value
            - 'attach': True if this is a grasp action
            - 'release': True if this is a release action
        """
        if self.mode == 'model' and self.model is not None:
            return self._predict_model(cube_pose, target_pose, current_joints, image)
        else:
            return self._predict_stub(cube_pose, target_pose)

    def _predict_stub(self, cube_pose: Tuple[float, float, float],
                      target_pose: Tuple[float, float, float]) -> List[Dict]:
        """Generate handcrafted action sequence for pick-and-place.

        This creates a sequence of waypoints that form a complete pick-and-place
        motion, adapted to the actual cube and target positions.

        Args:
            cube_pose: (x, y, z) of cube.
            target_pose: (x, y, z) of target.

        Returns:
            List of action chunks.
        """
        cx, cy, cz = cube_pose
        tx, ty, tz = target_pose

        # Calculate approach positions based on cube location
        # These are joint configurations for SO101 arm

        # Home position
        home = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 1.0,  # open
        }

        # Calculate pick approach based on cube position
        # SO101 arm orientation: positive angles move arm toward negative X
        pick_pan = -np.arctan2(cy, cx)  # Angle to cube
        pick_dist = np.sqrt(cx**2 + cy**2)

        # Map distance to joint angles (simplified IK)
        pick_lift = max(0.3, 1.0 - pick_dist * 1.5)
        pick_elbow = min(-0.5, -1.5 + pick_dist * 1.5)

        approach = {
            'shoulder_pan': np.clip(pick_pan, -1.0, 1.0),
            'shoulder_lift': pick_lift,
            'elbow_flex': pick_elbow,
            'wrist_flex': 0.2,
            'wrist_roll': 0.0,
            'gripper': 1.0,
        }

        # Lower to grasp height
        lower = {k: v for k, v in approach.items()}
        lower['shoulder_lift'] = pick_lift + 0.4
        lower['wrist_flex'] = 0.0

        # Grasp
        grasp = {k: v for k, v in lower.items()}
        grasp['gripper'] = 0.0  # close

        # Lift after grasp
        lift = {k: v for k, v in grasp.items()}
        lift['shoulder_lift'] = pick_lift - 0.2

        # Calculate place position based on target
        place_pan = -np.arctan2(ty, tx)
        place_dist = np.sqrt(tx**2 + ty**2)
        place_lift = max(0.3, 0.9 - place_dist * 1.2)
        place_elbow = min(-0.5, -1.3 + place_dist * 1.2)

        deliver = {
            'shoulder_pan': np.clip(place_pan, -1.0, 1.0),
            'shoulder_lift': place_lift,
            'elbow_flex': place_elbow,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0,  # keep closed
        }

        # Lower to place
        place = {k: v for k, v in deliver.items()}
        place['shoulder_lift'] = place_lift + 0.3

        # Release
        release = {k: v for k, v in place.items()}
        release['gripper'] = 1.0  # open

        # Return to home
        return_home = {k: v for k, v in home.items()}

        # Compose action chunks
        chunks = [
            {'name': 'move_home', 'ctrls': home},
            {'name': 'approach', 'ctrls': approach},
            {'name': 'lower', 'ctrls': lower},
            {'name': 'grasp', 'ctrls': grasp, 'attach': True},
            {'name': 'lift', 'ctrls': lift},
            {'name': 'move_deliver', 'ctrls': deliver},
            {'name': 'place', 'ctrls': place},
            {'name': 'release', 'ctrls': release, 'release': True},
            {'name': 'return_home', 'ctrls': return_home},
        ]

        return chunks

    def _predict_model(self, cube_pose: Tuple[float, float, float],
                       target_pose: Tuple[float, float, float],
                       current_joints: Optional[np.ndarray],
                       image: Optional[np.ndarray]) -> List[Dict]:
        """Predict actions using trained ACT model.

        Args:
            cube_pose: (x, y, z) of cube.
            target_pose: (x, y, z) of target.
            current_joints: Current joint positions.
            image: Current camera image.

        Returns:
            List of action chunks.
        """
        try:
            import torch

            # Prepare observation
            obs = np.array([
                *cube_pose,      # 3 values
                *target_pose,    # 3 values
                *(current_joints if current_joints is not None else [0] * 6)  # 6 values
            ], dtype=np.float32)

            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                action_chunk = self.model(obs_tensor)

            # Convert to action list
            action_chunk = action_chunk.cpu().numpy()[0]  # (chunk_size, 6)

            # Split into discrete actions (every N steps)
            step_size = max(1, len(action_chunk) // 8)
            chunks = []

            for i in range(0, len(action_chunk), step_size):
                action = action_chunk[i]
                ctrls = {name: float(val) for name, val in zip(self.ACTION_NAMES, action)}

                chunk = {'name': f'step_{i}', 'ctrls': ctrls}
                # Mark attach/release based on gripper state change
                if i > 0:
                    prev_gripper = action_chunk[i-1][-1]
                    curr_gripper = action[-1]
                    if prev_gripper > 0.5 and curr_gripper <= 0.5:
                        chunk['attach'] = True
                    elif prev_gripper <= 0.5 and curr_gripper > 0.5:
                        chunk['release'] = True

                chunks.append(chunk)

            return chunks

        except Exception as e:
            print(f"ACT model inference failed: {e}, falling back to stub")
            return self._predict_stub(cube_pose, target_pose)

    def train(self, dataset_path: str, num_epochs: int = 1000, batch_size: int = 64):
        """Train ACT model on collected demonstrations.

        Args:
            dataset_path: Path to training dataset.
            num_epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        print(f"Training ACT on {dataset_path} for {num_epochs} epochs...")
        print("Note: Use external training script for full ACT training")
        print("Recommended: Use LeRobot training pipeline")

        # Placeholder for training logic
        # In practice, use your training framework (e.g., LeRobot)


def load_trained_policy(model_dir: str) -> ACTPolicy:
    """Convenience function to load a trained policy.

    Args:
        model_dir: Directory containing trained model.

    Returns:
        Loaded ACTPolicy.
    """
    return ACTPolicy(mode='model', model_path=model_dir)


def create_stub_policy() -> ACTPolicy:
    """Create a stub policy for testing without training.

    Returns:
        ACTPolicy in stub mode.
    """
    return ACTPolicy(mode='stub')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ACT Policy')
    parser.add_argument('--test', action='store_true', help='Test stub policy')
    parser.add_argument('--model-path', type=str, help='Path to trained model')

    args = parser.parse_args()

    if args.test:
        print("Testing ACT Policy...")
        policy = ACTPolicy(mode='stub')
        cube_pose = (0.25, 0.0, 0.02)
        target_pose = (0.0, 0.25, 0.02)
        actions = policy.predict(cube_pose, target_pose)
        print(f"Generated {len(actions)} action chunks")
        for i, action in enumerate(actions):
            print(f"  {i+1}. {action['name']}: {list(action['ctrls'].keys())}")
    elif args.model_path:
        policy = load_trained_policy(args.model_path)
        print(f"Loaded policy from {args.model_path}")
    else:
        parser.print_help()

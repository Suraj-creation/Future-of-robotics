#!/usr/bin/env python3
"""YOLOv8-based cube detector for autonomous pick-and-place.

This module provides cube detection using YOLOv8 trained on cube objects.
It can operate in two modes:
1. Simulation mode: Uses ground truth from MuJoCo when YOLO model is not available
2. Real mode: Uses actual YOLOv8 inference on camera images

Usage:
    from perception.yolo_cube_detector import CubeDetector
    detector = CubeDetector(model_path='path/to/yolov8_cube.pt')
    cube_pos, confidence = detector.detect(frame)
"""

from typing import Tuple, Optional
import numpy as np
import os


class CubeDetector:
    """Cube detector using YOLOv8 with fallback to analytic detection."""

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.7):
        """Initialize the cube detector.

        Args:
            model_path: Path to YOLOv8 model. If None, uses analytic detection.
            confidence_threshold: Minimum confidence for valid detection.
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.use_yolo = False

        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.use_yolo = True
                print(f"CubeDetector: Loaded YOLOv8 model from {model_path}")
            except Exception as e:
                print(f"CubeDetector: Failed to load YOLO model: {e}")
                print("CubeDetector: Falling back to analytic detection")
        else:
            print("CubeDetector: No YOLO model provided, using analytic detection")

    def detect(self, frame: Optional[np.ndarray] = None,
               ground_truth_pos: Optional[Tuple[float, float, float]] = None) -> Tuple[Tuple[float, float, float], float]:
        """Detect cube in the scene.

        Args:
            frame: Camera image (BGR format). Required for YOLO mode.
            ground_truth_pos: Ground truth position for fallback mode.

        Returns:
            Tuple of ((x, y, z), confidence)
        """
        if self.use_yolo and frame is not None:
            return self._detect_yolo(frame)
        elif ground_truth_pos is not None:
            return self._detect_analytic(ground_truth_pos)
        else:
            # Default position if nothing available
            return (0.25, 0.0, 0.02), 0.5

    def _detect_yolo(self, frame: np.ndarray) -> Tuple[Tuple[float, float, float], float]:
        """Run YOLOv8 inference on the frame.

        Args:
            frame: Camera image in BGR format.

        Returns:
            Detected cube position and confidence.
        """
        if self.model is None:
            return (0.25, 0.0, 0.02), 0.5

        results = self.model(frame, verbose=False)

        best_conf = 0.0
        best_box = None

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf and conf >= self.confidence_threshold:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()

        if best_box is not None:
            # Convert bounding box center to normalized coordinates
            x1, y1, x2, y2 = best_box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Estimate 3D position from 2D detection
            # This is a simplified projection - in real setup, use camera calibration
            width = frame.shape[1]
            height = frame.shape[0]

            # Normalize to [-1, 1] range
            nx = (cx - width / 2) / (width / 2)
            ny = (cy - height / 2) / (height / 2)

            # Map to workspace coordinates (approximate)
            # Workspace: x in [0.1, 0.4], y in [-0.2, 0.2]
            x = 0.25 + nx * 0.15
            y = ny * 0.2
            z = 0.02  # Assume cube is on table

            return (x, y, z), best_conf

        # No detection
        return (0.25, 0.0, 0.02), 0.0

    def _detect_analytic(self, ground_truth_pos: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], float]:
        """Return ground truth position with simulated noise.

        Args:
            ground_truth_pos: True position from simulator.

        Returns:
            Position with small noise and high confidence.
        """
        # Add small noise to simulate sensor uncertainty
        noise = np.random.normal(0, 0.005, 3)
        pos = tuple(np.array(ground_truth_pos) + noise)
        return pos, 0.95

    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640):
        """Train YOLOv8 on custom dataset.

        Args:
            data_yaml: Path to data.yaml file for YOLO training.
            epochs: Number of training epochs.
            imgsz: Input image size.
        """
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')  # Start with pretrained model

        print(f"Training YOLOv8 on {data_yaml} for {epochs} epochs...")
        self.model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
        self.use_yolo = True
        print("Training complete!")


def generate_synthetic_training_data(output_dir: str, num_samples: int = 1000):
    """Generate synthetic training data for YOLO from MuJoCo.

    This function creates synthetic images of cubes in various positions
    with automatic annotations for YOLO training.

    Args:
        output_dir: Directory to save training images and labels.
        num_samples: Number of synthetic samples to generate.
    """
    import cv2
    from pathlib import Path
    import mujoco
    import sys

    # Import scene builder
    pkg_root = Path(__file__).resolve().parent.parent
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from mujoco_demo.scene_builder import build_model_with_cube_and_target

    output_path = Path(output_dir)
    images_dir = output_path / 'images' / 'train'
    labels_dir = output_path / 'labels' / 'train'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Build model with cube
    model, sim, _ = build_model_with_cube_and_target()

    # Create renderer
    renderer = mujoco.Renderer(model, 640, 480)

    for i in range(num_samples):
        # Randomize cube position
        cx = np.random.uniform(0.15, 0.35)
        cy = np.random.uniform(-0.15, 0.15)
        cz = 0.02

        # Set cube position in simulation
        sim.data.qpos[:7] = [cx, cy, cz, 1, 0, 0, 0]  # position + quaternion

        # Forward kinematics
        mujoco.mj_forward(model, sim)

        # Render image
        renderer.update_scene(sim)
        img = renderer.render(640, 480)

        # Save image
        img_path = images_dir / f'cube_{i:05d}.png'
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Compute bounding box from 3D projection (simplified)
        # In practice, use camera matrix for accurate projection
        # For now, create approximate bounding box
        bx = (cx - 0.15) / 0.2 * 0.6 + 0.2  # Map to image space
        by = (cy + 0.15) / 0.3 * 0.6 + 0.2
        bw = 0.15  # Approximate box width
        bh = 0.15  # Approximate box height

        # Save YOLO format label: class x_center y_center width height
        label_path = labels_dir / f'cube_{i:05d}.txt'
        with open(label_path, 'w') as f:
            f.write(f"0 {bx} {by} {bw} {bh}\n")

        if i % 100 == 0:
            print(f"Generated {i}/{num_samples} samples...")

    # Create data.yaml
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_path.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")  # Use same for simplicity
        f.write("nc: 1\n")
        f.write("names: ['cube']\n")

    print(f"Synthetic dataset created at {output_dir}")
    print(f"data.yaml: {yaml_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='YOLO Cube Detector')
    parser.add_argument('--generate-data', type=str, help='Generate synthetic training data to directory')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of synthetic samples')
    parser.add_argument('--train', type=str, help='Train YOLO on data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')

    args = parser.parse_args()

    if args.generate_data:
        generate_synthetic_training_data(args.generate_data, args.num_samples)
    elif args.train:
        detector = CubeDetector()
        detector.train(args.train, epochs=args.epochs)
    else:
        # Test mode
        print("Testing CubeDetector...")
        detector = CubeDetector()
        pos, conf = detector.detect(ground_truth_pos=(0.25, 0.0, 0.02))
        print(f"Detected cube at {pos} with confidence {conf}")

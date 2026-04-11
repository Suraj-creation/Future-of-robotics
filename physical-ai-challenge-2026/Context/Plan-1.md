# Plan-1: Autonomous Robotic Manipulation System for UR5
## Physical AI Challenge 2026 — Research + Implementation Plan

**Robot:** UR5 (Universal Robots 5)  
**Stack:** ROS2 Jazzy + Gazebo Harmonic + MoveIt 2  
**Strategy:** Simulation-first → Sim2Real  
**Document type:** Canonical research + execution plan  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Full System Architecture](#2-full-system-architecture)
3. [Section 1 — Perception System](#3-section-1--perception-system)
4. [Section 2 — Policy and Decision Layer](#4-section-2--policy-and-decision-layer)
5. [Section 3 — Control System](#5-section-3--control-system)
6. [Section 4 — End-to-End Pipeline](#6-section-4--end-to-end-pipeline)
7. [Section 5 — Simulation Strategy](#7-section-5--simulation-strategy)
8. [Section 6 — Implementation Roadmap](#8-section-6--implementation-roadmap)
9. [Section 7 — Trade-off Analysis](#9-section-7--trade-off-analysis)
10. [Open-Source Tools and Repos](#10-open-source-tools-and-repos)
11. [What NOT To Do](#11-what-not-to-do)
12. [Risks and Failure Points](#12-risks-and-failure-points)
13. [Decision Summary Sheet](#13-decision-summary-sheet)

---

## 1. Executive Summary

### The three-sentence version

Build a UR5 pick-and-place system starting with YOLO-based 2D detection + depth unprojection + MoveIt grasp execution. Once that baseline is solid, layer in imitation learning via behavior cloning using the LeRobot framework to replace hard-coded grasps with learned policies. For research-grade autonomy, graduate to Diffusion Policy or a lightweight Vision-Language-Action model.

### Recommended stack at each level

| Level | Perception | Policy | Control |
|---|---|---|---|
| **Phase 1 — Baseline** | YOLOv8n + RealSense D435i (sim) | Rule-based + MoveIt grasp pipeline | MoveIt 2 `ur_manipulator` group |
| **Phase 2 — Robust** | YOLOv8m + SAM2 + depth fusion | MoveIt with learned grasp scoring | MoveIt 2 + collision-aware replanning |
| **Phase 3 — Learning** | Same + temporal tracking | Behavior Cloning (LeRobot ACT) | Hybrid: policy sets goal pose, MoveIt executes |
| **Phase 4 — Advanced** | Vision encoder from pretrained VLM | Diffusion Policy or OpenVLA | Direct joint-space policy with safety wrapper |

### Hackathon time budget (rough guide)

| Phase | Time | Go/No-Go Gate |
|---|---|---|
| Phase 1 | Day 1-2 | Robot arm picks a cube reliably 7/10 times |
| Phase 2 | Day 3-4 | Works with moved/rotated cubes, collision avoidance holds |
| Phase 3 | Day 5-6 | Imitation policy matches or beats Phase 2 baseline |
| Phase 4 | Day 7+ | Research stretch goal — only if Phase 3 is solid |

---

## 2. Full System Architecture

### Top-level data flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SENSOR LAYER                                      │
│                                                                          │
│   RGB-D Camera (Gazebo sim: depth camera plugin)                         │
│   ├── /camera/color/image_raw        (RGB stream)                        │
│   ├── /camera/depth/image_raw        (depth image, 16-bit mm)            │
│   └── /camera/depth/camera_info      (intrinsics K matrix)               │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       PERCEPTION NODE                                    │
│                    (ROS2 node: /perception_node)                         │
│                                                                          │
│   1. YOLOv8  → 2D bounding boxes + class labels                          │
│   2. SAM2    → pixel-accurate segmentation mask (Phase 2+)               │
│   3. Depth unprojection → 3D centroid in camera frame                    │
│   4. TF transform → 3D centroid in world frame / base_link frame         │
│                                                                          │
│   Publishes: /detected_objects  (custom msg: [label, pose, confidence])  │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       POLICY / DECISION NODE                             │
│                   (ROS2 node: /policy_node)                              │
│                                                                          │
│   Phase 1: Rule-based — pick highest-confidence cube, assign grasp pose  │
│   Phase 3: BC policy — query learned policy for target pose              │
│   Phase 4: Diffusion/VLA — end-to-end action sequence                   │
│                                                                          │
│   Publishes: /grasp_target  (geometry_msgs/PoseStamped)                  │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    MOTION PLANNING NODE                                  │
│               (MoveIt 2 — planning group: ur_manipulator)                │
│                                                                          │
│   1. Receives grasp target pose                                          │
│   2. Runs OMPL/STOMP planner → collision-free trajectory                 │
│   3. Executes on ros2_control joint trajectory controller                │
│   4. Monitors execution status → republishes result                      │
│                                                                          │
│   Topics used:                                                           │
│   ├── /joint_states           (feedback)                                 │
│   ├── /move_group/goal        (action client)                            │
│   └── /planning_scene         (collision objects from add_scene_objects) │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                                     │
│              (ros2_control + UR5 joint controllers)                      │
│                                                                          │
│   Controllers active (from ur5_moveit_config):                           │
│   ├── joint_trajectory_controller  (arm: ur_manipulator group)           │
│   ├── gripper_controller           (RG2 gripper)                         │
│   └── joint_state_broadcaster      (publishes /joint_states)             │
└──────────────────────────────────────────────────────────────────────────┘
```

### ROS2 node graph

```
/camera_node  ──────────────────────────────────────────────────────────┐
  publishes:                                                             │
  /camera/color/image_raw                                                │
  /camera/depth/image_raw                                                │
  /camera/depth/camera_info                                              │
                                                                         ▼
/perception_node  ───────────────────────────────────────────────────────┐
  subscribes: /camera/* topics                                           │
  publishes:  /detected_objects  [DetectedObjectArray]                   │
                                                                         ▼
/policy_node  ───────────────────────────────────────────────────────────┐
  subscribes: /detected_objects                                          │
              /joint_states  (for state-aware policies in Phase 3+)      │
  publishes:  /grasp_target  [PoseStamped]                               │
              /place_target  [PoseStamped]                               │
                                                                         ▼
/moveit_interface_node  ─────────────────────────────────────────────────┐
  subscribes: /grasp_target, /place_target                              │
  calls:      MoveIt 2 MoveGroupInterface (action)                      │
  calls:      gripper_controller (action)                               │
  publishes:  /task_status  [String]                                    │
```

---

## 3. Section 1 — Perception System

### 3.1 Sensor setup recommendation

**For simulation (Gazebo Harmonic):** Use the `ros_gz_bridge` to expose a simulated `camera` plugin from the UR5 URDF. Add a depth camera plugin that publishes both RGB and depth topics. Mount it wrist-mounted on the UR5 end-effector or fixed overhead — both are viable, with trade-offs:

| Mount | Pros | Cons |
|---|---|---|
| Fixed overhead | Simple calibration, no occlusion from arm | Fixed FOV, can't inspect from multiple angles |
| Wrist-mounted | Always aimed at target, enables active perception | Moves with robot — must transform into world frame on every frame |

**Recommendation for hackathon:** Start with a fixed overhead camera. It is simpler to calibrate, simpler to debug, and gives a stable view of the entire pick-table. Switch to wrist-mounted when you have a working baseline.

**For Sim2Real:** Use an Intel RealSense D435i. It outputs aligned RGB + depth at 30 Hz, is natively supported in ROS2 via `realsense2_camera`, and its simulated equivalent in Gazebo is straightforward to replicate.

### 3.2 Object detection — model comparison

| Model | mAP (COCO) | Latency (RTX 3060) | ROS2 support | Best use |
|---|---|---|---|---|
| YOLOv8n (nano) | 37.3 | ~4 ms | Excellent | Real-time, resource-limited |
| YOLOv8m (medium) | 50.2 | ~11 ms | Excellent | Balanced accuracy/speed |
| YOLOv8x (extra) | 53.9 | ~35 ms | Excellent | High accuracy, slower |
| YOLOv9c | 53.0 | ~14 ms | Good | Slight accuracy gain over v8m |
| RT-DETR-L | 53.0 | ~32 ms | Moderate | Transformer-based, no NMS |
| Grounding DINO | Open-vocab | ~120 ms | Limited | Zero-shot, text-prompted |

**Recommendation:** Start with **YOLOv8n** in Phase 1 for speed. Upgrade to **YOLOv8m** in Phase 2 for better precision on partially occluded cubes. YOLOv8 has the best-maintained ROS2 wrapper ecosystem (`yolov8_ros` package by mgonzs13).

### 3.3 Segmentation — when and why

You only need segmentation if you require:
- Precise pixel-level object boundary for grasp point estimation
- Handling irregular objects (not needed for cubes in Phase 1-2)
- Generating training masks for your own dataset

**SAM2 (Segment Anything Model v2)** by Meta is the state-of-the-art. It supports video tracking, which means once you segment the cube in frame 1, it tracks it across frames without re-running the full model. Relevant for dynamic pick-and-place where the arm occludes the view.

**Recommendation:** Skip SAM2 in Phase 1. Add it in Phase 2 once the baseline is working. Use it to refine the grasp point from the center-of-bounding-box assumption to the center-of-mask, which is more accurate for partially visible objects.

### 3.4 Depth estimation

**RGB-D (preferred):** Use the depth channel from your depth camera directly. For cubes at 0.3-0.8m range, a RealSense D435i gives ±2mm accuracy. In Gazebo, the depth camera plugin is similarly accurate.

**Monocular depth (avoid for grasping):** Models like DepthAnything v2 or DPT produce relative, not metric depth. They are fine for scene understanding but cannot give you an absolute Z coordinate in meters needed for a grasp pose. Avoid for pick-and-place.

**Recommendation: Always use RGB-D for manipulation.** Monocular depth is a research tool, not a production grasping tool.

### 3.5 2D detection → 3D pose pipeline

This is the most critical and most commonly broken step. Here is the exact math and the code pattern:

**Step 1 — Deproject 2D centroid to 3D camera frame**

Given a bounding box center `(u, v)` in pixels and depth value `d` (in meters from aligned depth image):

```python
import numpy as np

def deproject_pixel_to_point(u, v, depth_m, camera_info):
    """
    camera_info: sensor_msgs/CameraInfo
    Returns: (X, Y, Z) in camera optical frame, meters
    """
    fx = camera_info.k[0]   # focal length x
    fy = camera_info.k[4]   # focal length y
    cx = camera_info.k[2]   # principal point x
    cy = camera_info.k[5]   # principal point y

    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return np.array([X, Y, Z])
```

**Step 2 — Transform from camera frame to robot base_link frame**

```python
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

def camera_point_to_base_link(point_camera, stamp, tf_buffer):
    """
    Transforms a 3D point from camera optical frame to base_link.
    point_camera: np.array [X, Y, Z] in camera frame
    """
    ps = PointStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = 'camera_color_optical_frame'
    ps.point.x = float(point_camera[0])
    ps.point.y = float(point_camera[1])
    ps.point.z = float(point_camera[2])

    transform = tf_buffer.lookup_transform(
        'base_link',
        'camera_color_optical_frame',
        stamp,
        timeout=rclpy.duration.Duration(seconds=1.0)
    )
    return do_transform_point(ps, transform)
```

**Step 3 — Construct grasp pose**

For a top-down grasp on a flat cube:

```python
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

def build_top_down_grasp_pose(object_pos_base_link):
    """
    Generates a top-down grasp pose directly above the detected object.
    Gripper Z-axis points down toward the object.
    """
    pose = PoseStamped()
    pose.header.frame_id = 'base_link'

    pose.pose.position.x = object_pos_base_link.point.x
    pose.pose.position.y = object_pos_base_link.point.y
    # Approach from above: add clearance offset above object Z
    pose.pose.position.z = object_pos_base_link.point.z + 0.15  # 15cm above

    # Rotation: gripper pointing straight down (X forward, Z down)
    # ROS convention: gripper Z = approach direction
    r = Rotation.from_euler('xyz', [np.pi, 0, 0])  # flip Z to point down
    quat = r.as_quat()  # [x, y, z, w]
    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]

    return pose
```

**Step 4 — Depth noise filtering**

Raw depth from a single pixel is noisy. Use the median over the bounding box region:

```python
def get_robust_depth(depth_image, bbox, margin=5):
    """
    bbox: (x1, y1, x2, y2) from YOLO detection
    Returns median depth in meters from the center region of the bbox
    """
    x1, y1, x2, y2 = bbox
    # Use the inner 50% of bbox to avoid edge noise
    cx, cy = (x1+x2)//2, (y1+y2)//2
    w, h = (x2-x1)//4, (y2-y1)//4
    roi = depth_image[cy-h:cy+h, cx-w:cx+w]
    valid = roi[roi > 0]  # remove zero (invalid) depth pixels
    if len(valid) == 0:
        return None
    return float(np.median(valid)) / 1000.0  # mm to meters (RealSense format)
```

### 3.6 Perception node structure (ROS2)

```python
# ur5_ws/src/ur5_moveit/ur5_moveit/perception_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
# ... (full imports)

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # Phase 1: nano; Phase 2: yolov8m.pt

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self.info_callback, 10)

        # Publisher
        self.detection_pub = self.create_publisher(
            PoseStamped, '/detected_object_pose', 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.latest_depth = None
        self.camera_info = None

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '16UC1')

    def info_callback(self, msg):
        self.camera_info = msg

    def rgb_callback(self, msg):
        if self.latest_depth is None or self.camera_info is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(rgb, conf=0.5, classes=[0])  # class 0 = cube

        for result in results[0].boxes:
            bbox = result.xyxy[0].cpu().numpy().astype(int)
            depth_m = get_robust_depth(self.latest_depth, bbox)
            if depth_m is None:
                continue

            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            point_cam = deproject_pixel_to_point(cx, cy, depth_m, self.camera_info)
            pose_base = camera_point_to_base_link(point_cam, msg.header.stamp, self.tf_buffer)
            grasp_pose = build_top_down_grasp_pose(pose_base)
            self.detection_pub.publish(grasp_pose)
```

---

## 4. Section 2 — Policy and Decision Layer

### 4.1 Approach overview and honest assessment

```
APPROACHES (least to most complex):

A. Rule-based + MoveIt grasp pipeline
   Complexity: ★☆☆☆☆   Robustness: ★★★☆☆   Time to implement: 4-8 hours
   Best for: Phase 1 baseline. Get a pick working FIRST.

B. Behavior Cloning (BC) with LeRobot / ACT
   Complexity: ★★★☆☆   Robustness: ★★★★☆   Time to implement: 2-3 days
   Best for: Phase 3. Replace brittle rule-based grasps with learned ones.

C. Diffusion Policy
   Complexity: ★★★★☆   Robustness: ★★★★★   Time to implement: 3-5 days
   Best for: Phase 4. High-quality multi-modal action distributions.

D. Vision-Language-Action (OpenVLA, RT-2)
   Complexity: ★★★★★   Robustness: ★★★★★   Time to implement: 1+ week
   Best for: Research. Requires significant GPU and fine-tuning infrastructure.
```

### 4.2 Phase 1 — Rule-based grasp planning

The policy is a simple state machine:

```
State 1: IDLE
  → Wait for /detected_object_pose message
  → Transition to State 2 when pose received

State 2: PLANNING_PICK
  → Compute pre-grasp pose (10cm above grasp pose)
  → Call MoveIt plan to pre-grasp pose
  → Transition to State 3 if plan succeeds

State 3: APPROACHING
  → Execute plan to pre-grasp pose
  → Wait for execution completion
  → Transition to State 4

State 4: GRASPING
  → Move straight down from pre-grasp to grasp pose (Cartesian path)
  → Close gripper
  → Transition to State 5

State 5: LIFTING
  → Move straight up (reverse Cartesian path)
  → Transition to State 6

State 6: PLACING
  → Compute place pose above place_table (from add_scene_objects known position)
  → Plan and execute to place pose
  → Open gripper
  → Return to State 1
```

```python
# ur5_ws/src/ur5_moveit/ur5_moveit/pick_place_node.py
# Registers in setup.py as: 'pick_place = ur5_moveit.pick_place_node:main'

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
import numpy as np


class PickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_node')

        # MoveIt 2 — CORRECT planning group from ur5_robot.srdf
        self.robot = MoveItPy(node_name='moveit_py')
        self.arm = self.robot.get_planning_component('ur_manipulator')
        self.robot_model = self.robot.get_robot_model()

        self.sub = self.create_subscription(
            PoseStamped, '/detected_object_pose',
            self.detection_callback, 1)

        self.state = 'IDLE'
        self.current_target = None

        self.get_logger().info('PickPlaceNode ready. Waiting for detections...')

    def detection_callback(self, msg):
        if self.state != 'IDLE':
            return  # busy
        self.current_target = msg
        self.execute_pick_place()

    def execute_pick_place(self):
        self.state = 'PICKING'
        target = self.current_target

        # Pre-grasp pose: 15cm above detection
        pre_grasp = PoseStamped()
        pre_grasp.header.frame_id = 'base_link'
        pre_grasp.pose = target.pose
        pre_grasp.pose.position.z += 0.15

        # Plan and execute to pre-grasp
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(pose_stamped_msg=pre_grasp,
                                pose_link='tool0')
        plan_result = self.arm.plan()
        if not plan_result:
            self.get_logger().error('Planning to pre-grasp failed')
            self.state = 'IDLE'
            return

        self.robot.execute(plan_result.trajectory, controllers=[])

        # TODO: Cartesian descent to grasp, gripper close, lift, place
        # See full implementation in Section 8 Phase 1 code

        self.state = 'IDLE'


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

### 4.3 Phase 3 — Behavior Cloning with LeRobot ACT

**ACT (Action Chunking with Transformers)** is the best-performing imitation learning policy for robotic manipulation as of 2024-2026. It predicts a chunk (sequence) of future actions rather than a single action, which dramatically improves temporal consistency.

**Why ACT over basic BC:**
- Basic BC suffers from distribution shift — small errors compound
- ACT predicts 100-step action chunks, smoothing over instantaneous errors
- ACT uses a CVAE encoder to handle multi-modality in demonstrations
- Demonstrated on real robot arms achieving 80-90%+ success rates on tabletop pick-place

**Data collection workflow:**

```
1. Run simulation in Gazebo
2. Use teleoperation (spacemouse, keyboard, or scripted expert) to collect N demonstrations
3. Record with LeRobot dataset format:
   - Observations: RGB frames + joint states + gripper state
   - Actions: joint velocities or end-effector delta poses
4. Store as HuggingFace LeRobot dataset
5. Train ACT policy on dataset
6. Deploy policy as a ROS2 node that publishes /grasp_target
```

**LeRobot integration with this repo:**

```bash
# Install LeRobot (inside container or separate venv)
pip install lerobot

# Record a dataset from Gazebo
python -m lerobot.record \
  --robot-path lerobot/configs/robot/ur5_sim.yaml \
  --fps 30 \
  --repo-id my_ur5_pick_place \
  --num-episodes 100

# Train ACT policy
python -m lerobot.train \
  --policy-path lerobot/configs/policy/act.yaml \
  --dataset-repo-id my_ur5_pick_place \
  --output-dir outputs/train/ur5_act

# Evaluate
python -m lerobot.eval \
  --policy-path outputs/train/ur5_act/checkpoints/last/pretrained_model
```

**Minimum demo count for ACT to converge:**

| Task complexity | Min demos | Recommended demos |
|---|---|---|
| Single cube, fixed position | 20 | 50 |
| Single cube, randomized position | 50 | 200 |
| Multiple cubes, sort by color | 100 | 500 |

### 4.4 Phase 4 — Diffusion Policy

Diffusion Policy treats action generation as a denoising diffusion process. Starting from Gaussian noise, it iteratively refines the action trajectory conditioned on the current observation. This gives it a key advantage: it can represent multi-modal action distributions (multiple valid ways to grasp an object) that BC collapses into a single blurry average.

**Implementation approach:**

```
Architecture options for Diffusion Policy:
1. CNN-based (original paper): Lower compute, simpler. ~3 FPS inference.
2. Transformer-based (improved): Higher quality. ~1-2 FPS without optimization.
3. DDIM sampling (accelerated): 10x faster inference. Use this.

Inputs:
- RGB observation: (T_obs=2 frames, H, W, 3) — recent history
- Robot state: (T_obs=2, joint_dim=7)

Output:
- Action chunk: (T_pred=16, action_dim=7) — predicted joint deltas

Training:
- ~500-1000 demos for tabletop tasks
- ~2-4 hours on RTX 3090 to convergence
```

**ROS2 integration pattern:**

```python
# Diffusion policy runs at ~5-10 Hz (limited by inference)
# Bridge to ROS2 as a service or action server
# MoveIt executes the resulting waypoints at its native rate (~50 Hz)
class DiffusionPolicyNode(Node):
    def __init__(self):
        super().__init__('diffusion_policy_node')
        self.policy = DiffusionPolicy.from_pretrained('path/to/checkpoint')

        # Buffer for observation history
        self.obs_buffer = deque(maxlen=2)  # T_obs=2 frames

        self.create_subscription(Image, '/camera/color/image_raw',
                                 self.obs_callback, 10)
        self.action_pub = self.create_publisher(
            JointTrajectory, '/diffusion_action', 10)

    def obs_callback(self, rgb_msg):
        self.obs_buffer.append(self.process_obs(rgb_msg))
        if len(self.obs_buffer) == 2:
            actions = self.policy.predict_action(list(self.obs_buffer))
            self.publish_action_chunk(actions)
```

### 4.5 VLA models — realistic assessment

| Model | Parameters | GPU needed | Fine-tune needed | Recommendation |
|---|---|---|---|---|
| RT-1 (Google) | 35M | 1x A100 | Yes, from scratch | Skip — not open enough |
| RT-2 (Google) | 55B | 4x A100 | Prohibitive | Skip for hackathon |
| OpenVLA | 7B | 1x A100 40GB | LoRA ~24h | Phase 4 stretch goal |
| Octo | 93M | 1x RTX 3090 | Yes, ~8h | Most practical VLA |
| pi0 (Physical Intelligence) | ~3B | 2x A100 | Yes | Research-only |

**Honest recommendation:** For a hackathon, do not target RT-2 or OpenVLA unless you have A100-class hardware and a week. **Octo** is the most practical VLA option — it was designed for cross-embodiment transfer, has fine-tuning documentation, and runs on a single 24GB GPU.

---

## 5. Section 3 — Control System

### 5.1 MoveIt 2 architecture in this repo

```
ur_manipulator planning group (from ur5_robot.srdf)
│
├── Planning plugins:
│   ├── OMPL (default — good for complex scenes)
│   ├── STOMP (better for smooth trajectories, slower)
│   └── PILZ Industrial Motion (best for Cartesian straight-line moves)
│
├── Controllers (from ur5_controller/config/):
│   ├── joint_trajectory_controller  → arm joints
│   └── gripper_controller           → RG2 gripper
│
└── Collision checking:
    ├── FCL (Flexible Collision Library) — default
    └── Planning scene updated by add_scene_objects (tables, cubes as collision objects)
```

### 5.2 When to use MoveIt planning vs. Cartesian paths vs. policy control

| Situation | Use | Reason |
|---|---|---|
| Moving arm from home to above table | MoveIt OMPL plan | Obstacle-rich, complex joint space |
| Final approach to grasp (last 15cm) | Cartesian path (computeCartesianPath) | Must be straight-line to avoid knocking object |
| Post-grasp lift | Cartesian path | Same — keep object stable |
| Transport with object held | MoveIt OMPL plan | Need collision-free path, object attached |
| Policy-controlled phases (Phase 3+) | Hybrid: policy sets goal, MoveIt executes | Policy output is imprecise, MoveIt ensures collision safety |

### 5.3 Recommended control architecture per phase

**Phase 1-2 (MoveIt-centric):**

```
/perception_node  →  /grasp_target (PoseStamped)
                          │
                          ▼
/pick_place_node  →  MoveItPy.plan()  →  robot.execute()
                          │
                          ▼
                  ros2_control joint_trajectory_controller
```

**Phase 3 (Hybrid: BC policy + MoveIt execution):**

```
/perception_node  →  /obs_topic
                          │
                          ▼
/policy_node (ACT)  →  /target_ee_pose (PoseStamped) at 10 Hz
                          │
                          ▼
/moveit_interface_node  →  Servo mode (MoveIt Servo) for smooth real-time following
                           OR
                           Full plan/execute for each pose update
```

**Phase 4 (Direct policy control with safety wrapper):**

```
/policy_node (Diffusion)  →  /joint_deltas at 5-10 Hz
                                    │
                                    ▼
                         Safety wrapper node:
                         - Clamps joint velocity limits
                         - Checks workspace bounds
                         - Monitors collision proximity
                                    │
                                    ▼
                         ros2_control joint_trajectory_controller
```

### 5.4 MoveIt Servo — real-time Cartesian control

MoveIt Servo enables real-time teleoperation and policy-driven control at 100-500 Hz by accepting incremental Cartesian/joint commands:

```bash
# Start servo in a container terminal
ros2 run moveit_servo servo_node --ros-args \
  -p robot_description_semantic:=$(cat /ur5_ws/src/ur5_moveit/config/ur5_robot.srdf)
```

```python
# Publish incremental end-effector commands for policy control
from geometry_msgs.msg import TwistStamped

servo_pub = node.create_publisher(TwistStamped,
    '/servo_node/delta_twist_cmds', 10)

twist = TwistStamped()
twist.header.frame_id = 'base_link'
twist.twist.linear.x = 0.01   # 1 cm/s forward
servo_pub.publish(twist)
```

### 5.5 Latency budget

| Component | Typical latency | Notes |
|---|---|---|
| Camera capture | 33 ms | 30 Hz RGB-D |
| YOLO inference (v8n, GPU) | 4-8 ms | Negligible |
| Depth unproject + TF | 1-2 ms | Negligible |
| MoveIt OMPL plan | 50-500 ms | Highly scene-dependent |
| MoveIt Cartesian plan | 5-20 ms | Much faster |
| ros2_control execution | ~1 ms dispatch | Hardware runs in own loop |
| **Total Phase 1 cycle** | **~150-600 ms** | Acceptable for pick-place |
| ACT policy inference | 15-30 ms | GPU inference |
| Diffusion policy inference | 100-300 ms | With DDIM sampling |

### 5.6 Failure handling strategies

```
Failure type: MoveIt plan returns no solution
  Detection: plan_result is None / empty trajectory
  Recovery:
    1. Retry with different random seed (OMPL stochastic)
    2. Relax orientation constraints (±15° from top-down)
    3. Move arm to a neutral home pose, retry from there
    4. Log failure, skip this object, try next detection

Failure type: Execution timeout / controller not following
  Detection: Action server returns ABORTED
  Recovery:
    1. Stop execution
    2. Check /joint_states — is arm actually moving?
    3. Republish to controller with fresh timestamp
    4. In simulation: restart the controller spawner

Failure type: Grasp fails (object drops during lift)
  Detection: Check depth image after lift — is object still visible at expected pose?
  Recovery:
    1. Open gripper, re-approach
    2. Adjust grasp pose slightly (±5mm lateral offset)
    3. Log failure for dataset analysis

Failure type: Perception false positive (grasps empty space)
  Detection: Gripper closes but force/torque reading doesn't rise (Phase 2+ sensors)
  Recovery:
    1. Open gripper, return to home
    2. Run perception again with higher confidence threshold
```

---

## 6. Section 4 — End-to-End Pipeline

### 6.1 Complete data flow diagram

```
GAZEBO SIMULATION
│
│   Simulated UR5 arm with RG2 gripper
│   Simulated RGB-D camera (depth_camera Gazebo plugin)
│   Physics: pick_table, place_table, blue_cube_1/2/3
│
└──► /camera/color/image_raw         (sensor_msgs/Image, 30 Hz)
     /camera/depth/image_raw         (sensor_msgs/Image, 30 Hz, 16UC1 mm)
     /camera/depth/camera_info       (sensor_msgs/CameraInfo)
     /joint_states                   (sensor_msgs/JointState, 50 Hz)
     /tf                             (transforms including camera→base_link)
          │
          ▼
     /perception_node
          │  Runs YOLO on RGB
          │  Unprojects bounding box center using depth
          │  Transforms camera frame → base_link via TF
          │
          └──► /detected_objects     (custom: DetectedObjectArray)
                    │
                    ▼
               /policy_node
                    │  Phase 1: picks highest-confidence detection
                    │  Phase 3: feeds obs to ACT policy
                    │  Computes grasp pose + place pose
                    │
                    └──► /grasp_target   (geometry_msgs/PoseStamped)
                         /place_target   (geometry_msgs/PoseStamped)
                              │
                              ▼
                         /moveit_interface_node
                              │  Plans collision-free trajectory with MoveIt
                              │  Executes via ros2_control
                              │  Reports status
                              │
                              └──► /task_status   (std_msgs/String)
                                        │
                                        ▼
                                   FEEDBACK LOOP
                                   (re-trigger perception
                                    if task fails)
```

### 6.2 Custom ROS2 message definitions

Create these in `ur5_ws/src/ur5_moveit/msg/`:

```
# DetectedObject.msg
std_msgs/Header header
string label
float32 confidence
geometry_msgs/PoseStamped pose_in_base_link
geometry_msgs/Vector3 dimensions   # estimated from bbox + depth

---

# DetectedObjectArray.msg
std_msgs/Header header
DetectedObject[] objects
```

Register in `CMakeLists.txt` and `package.xml` of `ur5_moveit`.

### 6.3 Launch file for the full stack

```python
# ur5_ws/src/ur5_moveit/launch/full_autonomy.launch.py
# Starts simulation + perception + policy + moveit interface

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ur5_moveit'),
                         'launch', 'simulated_robot.launch.py')
        )
    )

    perception_node = Node(
        package='ur5_moveit',
        executable='perception_node',
        name='perception_node',
        output='screen'
    )

    policy_node = Node(
        package='ur5_moveit',
        executable='pick_place_node',
        name='pick_place_node',
        output='screen'
    )

    return LaunchDescription([
        sim_launch,
        perception_node,
        policy_node,
    ])
```

---

## 7. Section 5 — Simulation Strategy

### 7.1 Adding a depth camera to the UR5 in Gazebo

Add this to the UR5 URDF/Xacro in `ur5_ws/src/ur5_description/urdf/`:

```xml
<!-- Add to ur5.urdf.xacro — fixed overhead camera -->
<link name="overhead_camera_link">
  <visual>
    <geometry><box size="0.05 0.05 0.02"/></geometry>
  </visual>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
  </inertial>
</link>

<joint name="overhead_camera_joint" type="fixed">
  <parent link="world"/>
  <child link="overhead_camera_link"/>
  <!-- Positioned 1.2m above table center, looking straight down -->
  <origin xyz="0.4 0.0 1.2" rpy="0 1.5707 0"/>
</joint>

<gazebo reference="overhead_camera_link">
  <sensor name="depth_camera" type="rgbd_camera">
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip><near>0.1</near><far>2.0</far></clip>
      <depth_camera><output>true</output></depth_camera>
    </camera>
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <topic>/camera</topic>
  </sensor>
</gazebo>
```

Then add the ros_gz bridge for camera topics in the launch file:

```python
# In gazebo.launch.py or simulated_robot.launch.py
from launch_ros.actions import Node

camera_bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/camera/image@sensor_msgs/msg/Image@gz.msgs.Image',
        '/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
        '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
    ],
    output='screen'
)
```

### 7.2 Synthetic data generation for YOLO training

For custom classes (if the default YOLO weights don't detect your simulation cubes well):

**Option 1 — BlenderProc (recommended):**

```bash
pip install blenderproc

# Generate 10,000 annotated images of cubes on tables
blenderproc run generate_cube_dataset.py \
  --output-dir /data/cube_dataset \
  --num-scenes 10000
```

BlenderProc can randomize: lighting, camera angle, cube color, cube position, table texture — generating rich domain-randomized training data without a physical robot.

**Option 2 — Gazebo screenshot pipeline:**

```python
# Use Gazebo's camera subscriber to collect frames
# Randomize cube positions using a ROS2 service:
#   /world/hackathon/set_model_pose (gz.msgs.Pose)
# Then capture frames with labels from ground-truth model poses
```

### 7.3 Domain randomization strategy

The gap between simulation and reality (sim2real gap) is the primary challenge. Randomize these parameters during training:

| Parameter | Simulation range | Real-world typical range |
|---|---|---|
| Cube color/texture | Random RGB + texture | Solid colors |
| Table surface texture | Random wood/metal patterns | Fixed |
| Lighting direction | Random azimuth 0-360° | Fixed lab lighting |
| Lighting intensity | 0.3x – 2.0x baseline | ~1.0x |
| Camera noise | σ=0-5 pixel Gaussian noise | Real sensor noise |
| Camera position | ±2cm from nominal | Fixed mount |
| Cube position | Full table surface | Random placement |
| Cube orientation | 0-360° yaw rotation | Any orientation |
| Depth noise | ±5mm random noise on depth image | RealSense noise model |

### 7.4 Sim2Real transfer checklist

Before deploying to a real UR5:

```
[ ] Perception tested on real images (not just simulation)
[ ] Camera calibration done (intrinsics + extrinsics to robot base)
    Tool: ros2 run camera_calibration cameracalibrator.py
[ ] TF chain verified: camera_optical_frame → base_link is correct
[ ] Workspace limits checked: robot cannot collide with real table
[ ] Joint speed limits reduced 50% for first real deployment
[ ] Emergency stop (e-stop) within arm's reach during testing
[ ] Collision objects in MoveIt match physical table dimensions exactly
[ ] Gripper force calibration: correct open/close width for real cube size
```

---

## 8. Section 6 — Implementation Roadmap

### Phase 1 — Minimal Working System

**Goal:** Robot arm picks a cube and places it on the place_table reliably 7/10 times.

**Duration:** 1-2 days

**Tools:**
- YOLOv8n (pretrained on COCO, fine-tune if needed)
- MoveIt 2 `ur_manipulator` group
- Gazebo Harmonic simulation (already set up from Setup.md)
- `add_scene_objects` for collision objects

**Step-by-step tasks:**

```
Day 1 Morning:
[ ] Add depth camera to UR5 URDF in Gazebo (Section 5.1 above)
[ ] Verify camera topics publish: ros2 topic echo /camera/image --once
[ ] Write and test perception_node.py (YOLO + depth unproject)
[ ] Verify /detected_object_pose publishes correct 3D positions
    Test: echo /detected_object_pose, move cube in Gazebo, check pose updates

Day 1 Afternoon:
[ ] Write pick_place_node.py (rule-based state machine)
[ ] Test pre-grasp planning only (arm moves above cube without grasping)
[ ] Add Cartesian descent and gripper close
[ ] Test full pick: arm grasps cube and lifts it

Day 2:
[ ] Add place logic: arm moves to place_table and opens gripper
[ ] Tune grasp pose offset and approach height
[ ] Run 20 trials, record success rate
[ ] Fix top failures (usually: depth noise, planning failure, grasp pose offset)
```

**Evaluation metrics:**
- Pick success rate: 7/10 = go to Phase 2
- Planning success rate: >90% (plans found for valid poses)
- Perception accuracy: 3D position error <1cm vs ground truth

**Expected failure modes:**
- YOLO fails to detect simulation-rendered cubes → fine-tune on 100 Gazebo screenshots
- Depth unprojection gives wrong Z → check depth image format (16UC1 mm vs 32FC1 m)
- MoveIt can't plan to pose → check if pose is in workspace, relax orientation

### Phase 2 — Robustness

**Goal:** Works with cubes in any position on the table, handles partial occlusion, collision avoidance is reliable.

**Duration:** 1-2 days

**Tools added:**
- YOLOv8m (upgrade from nano)
- SAM2 for better segmentation masks
- insert_obstacle for dynamic obstacle testing
- Replanning on failure

**Step-by-step tasks:**

```
[ ] Upgrade YOLO model to YOLOv8m for better detection accuracy
[ ] Add SAM2 to refine segmentation mask → more accurate centroid estimate
[ ] Implement grasp pose rotation estimation:
    - Use PCA on the segmentation mask to find object principal axis
    - Align gripper to long axis of cube
[ ] Add collision monitoring: check if arm hits obstacle during execution
[ ] Implement auto-replanning: if execution fails, try 3 times before aborting
[ ] Test with insert_obstacle to verify collision avoidance works
[ ] Test with cubes at edges and corners of table (harder poses for MoveIt)
[ ] Test with multiple cubes: pick them all one by one
```

**Evaluation metrics:**
- Pick success rate: >85% across all cube positions
- Works with dynamic obstacle inserted mid-task
- No collisions with table or other scene objects

### Phase 3 — Imitation Learning

**Goal:** ACT policy matches or beats Phase 2 baseline, enabling more flexible grasping behavior.

**Duration:** 2-3 days

**Tools added:**
- LeRobot framework
- ACT policy
- Dataset collection pipeline
- Training infrastructure (CUDA GPU required)

**Step-by-step tasks:**

```
[ ] Install LeRobot: pip install lerobot
[ ] Write a UR5 robot config for LeRobot
    (maps ROS2 joint topics to LeRobot observation/action format)
[ ] Write scripted expert for data collection
    (runs the Phase 2 rule-based system but records all obs+actions)
[ ] Collect 100-200 demonstration episodes
    (each episode: one cube pick and place)
[ ] Verify dataset: visualize 10 episodes, check obs/action alignment
[ ] Train ACT policy on dataset (4-8 hours on RTX 3090)
[ ] Write policy_node.py that loads ACT checkpoint and queries policy at 10 Hz
[ ] Evaluate policy in simulation: compare success rate to Phase 2 baseline
[ ] Collect more data for failure cases and retrain
```

**Evaluation metrics:**
- ACT policy success rate ≥ Phase 2 baseline (>85%)
- Policy generalizes to unseen cube positions not in training set
- Rollout is smooth (no jerky movements from compounding errors)

**Minimum dataset requirements:**

```
Phase 3 minimum viable dataset:
- 100 successful episodes
- Cubes at 10+ different positions
- 5+ different orientations per position
- Both blue_cube_1, blue_cube_2, blue_cube_3 included

Phase 3 recommended dataset:
- 300 successful episodes
- Full table coverage
- Some partially occluded cases
- Some near-failure recoveries if available
```

### Phase 4 — Advanced Autonomy

**Goal:** Diffusion Policy or Octo VLA model achieves research-grade generalization.

**Duration:** 3-5 days (stretch goal)

**Tools added:**
- Diffusion Policy (chi-robotics/diffusion_policy on GitHub)
- Optionally: Octo (octo-models/octo on GitHub) for VLA
- DDIM sampler for fast inference

**Step-by-step tasks:**

```
[ ] Set up Diffusion Policy repo alongside this workspace
[ ] Convert LeRobot dataset to Diffusion Policy format
    (zarr format with obs/actions as numpy arrays)
[ ] Train CNN-based Diffusion Policy (faster, lower GPU requirement)
    Training time: ~8-12h on RTX 3090 for 200 demos
[ ] Integrate as ROS2 policy_node: publish action chunks at 5-10 Hz
[ ] Evaluate: should outperform ACT on multi-modal grasps
[ ] If Octo: set up Octo repo, fine-tune from pretrained checkpoint
    on your dataset for 1-2 hours (transfer learning)
```

**Evaluation metrics:**
- Diffusion Policy success rate: >90% in simulation
- Handles novel cube colors/textures not seen in training
- Inference latency: <200ms per action chunk (DDIM sampling)

---

## 9. Section 7 — Trade-off Analysis

### 9.1 Full comparison matrix

| Approach | Robustness | Compute | Implementation complexity | Sim2Real transfer | Hackathon feasibility |
|---|---|---|---|---|---|
| Rule-based + MoveIt | ★★★☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ | ★★★★★ |
| MoveIt grasp pipeline | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★★☆ |
| Behavior Cloning (basic) | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ |
| ACT (LeRobot) | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★☆☆ |
| Diffusion Policy | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |
| OpenVLA | ★★★★★ | ★★★★★ | ★★★★★ | ★★★☆☆ | ★☆☆☆☆ |
| Octo | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |

### 9.2 The hybrid architecture argument

The most practical and robust approach at research level is a **hybrid architecture**: AI policy decides what to do and roughly where to go, but MoveIt executes with collision safety guarantees. This is how industrial robot AI actually works in 2024-2026:

```
Policy (AI)          →    Goal pose / waypoints    →    MoveIt / ros2_control
(decides intent)           (rough target)               (safe execution)

Advantages:
- Policy doesn't need to learn joint limits, collision geometry
- MoveIt provides hard safety guarantees that policy cannot
- Can swap policies without rebuilding the whole system
- Much easier to debug (policy failure vs. execution failure are separate)

Disadvantage:
- Sub-optimal: policy might want a motion MoveIt's planner finds difficult
- Two-layer latency: policy inference + planning time
- "Semantic gap": policy thinks in task space, MoveIt plans in joint space
```

### 9.3 Sensor choice trade-offs

| Sensor | Accuracy | ROS2 support | Cost | Sim equivalent | Recommendation |
|---|---|---|---|---|---|
| Intel RealSense D435i | ±2mm at 0.5m | Excellent | ~$200 | Good Gazebo plugin | **Recommended** |
| ZED 2i (stereo) | ±5mm at 1m | Good | ~$450 | Fair | Phase 2+ |
| Azure Kinect | ±1mm at 0.5m | Moderate | ~$400 | Poor | Skip |
| Monocular RGB only | N/A (needs estimation) | N/A | $30 | Perfect | Phase 4 VLA only |

---

## 10. Open-Source Tools and Repos

### Perception

| Tool | Repo | Use |
|---|---|---|
| YOLOv8 | `ultralytics/ultralytics` | Primary object detector |
| YOLOv8 ROS2 wrapper | `mgonzs13/yolov8_ros` | Ready-made ROS2 node |
| SAM2 | `facebookresearch/sam2` | Segmentation refinement |
| DepthAnything v2 | `DepthAnything/Depth-Anything-V2` | Monocular depth (Phase 4) |
| Open3D | `isl-org/Open3D` | Point cloud processing |

### Policy / Learning

| Tool | Repo | Use |
|---|---|---|
| LeRobot | `huggingface/lerobot` | ACT policy, data collection |
| Diffusion Policy | `real-stanford/diffusion_policy` | Phase 4 policy |
| Octo | `octo-models/octo` | Pretrained VLA for fine-tuning |
| BlenderProc | `DLR-RM/BlenderProc` | Synthetic data generation |
| Robomimic | `ARISE-Initiative/robomimic` | BC baselines, dataset tools |

### Control / Planning

| Tool | Repo / Package | Use |
|---|---|---|
| MoveIt 2 | `moveit/moveit2` | Motion planning (already in container) |
| MoveIt Servo | Part of MoveIt 2 | Real-time Cartesian control |
| ros2_control | `ros-controls/ros2_control` | Controller infrastructure |
| STOMP planner | `moveit/moveit_stomp` | Smooth trajectory planning |
| ros_gz | `gazebosim/ros_gz` | Gazebo-ROS2 bridge |

### Evaluation and Debugging

| Tool | Use |
|---|---|
| RViz 2 | 3D visualization of robot, scene, camera |
| PlotJuggler | Real-time ROS2 topic plotting |
| ros2 bag | Record and replay ROS2 sessions for debugging |
| Foxglove Studio | Web-based ROS2 visualization |

---

## 11. What NOT To Do

These are common mistakes that waste days of hackathon time:

**Perception mistakes:**

- **Do NOT use monocular depth for grasping.** Relative depth from DepthAnything has no absolute scale. You will get Z coordinates that are proportionally right but metrically wrong, causing the arm to always be at the wrong height.

- **Do NOT skip the depth filtering step.** Raw single-pixel depth is noisy. Using the center pixel of the bounding box directly without median filtering over the center region will give you ±3-5cm noise on Z, which is the difference between a successful grasp and hitting the table.

- **Do NOT assume YOLO's pretrained weights work perfectly on simulation-rendered images.** COCO-trained YOLOv8 does not know what a "blue cube on a table in Gazebo" looks like. If detection fails, fine-tune on 100-200 Gazebo screenshots. It takes 30 minutes.

**Planning mistakes:**

- **Do NOT use the wrong planning group name.** The SRDF defines `ur_manipulator` for the arm. Using any other string (like `arm`, `manipulator`, `ur5_arm`) gives silent failure — MoveIt returns "group not found" and nothing moves.

- **Do NOT command MoveIt to plan to a pose outside the robot's reachable workspace.** Check the UR5 workspace sphere (0.2m-0.85m from base) before publishing a target pose. Unreachable poses cause the planner to spin for the full timeout (5-10 seconds) and return nothing.

- **Do NOT execute Cartesian paths at full speed.** Cartesian paths don't respect joint velocity limits the same way OMPL paths do. Scale velocity to 20-30% for approach/descent moves.

**Learning mistakes:**

- **Do NOT try to train ACT or Diffusion Policy without a GPU.** CPU training for these models is not a viable option. ACT on CPU takes 10-20x longer — what should be 4 hours becomes 3 days.

- **Do NOT collect demonstrations using your Phase 1 baseline as the only expert.** If the expert policy has systematic biases (always approaches from one direction), the learned policy inherits those biases rigidly. Add some variability in how demonstrations are collected.

- **Do NOT skip dataset visualization before training.** Always visually inspect 10-20 episodes of your training data before starting a multi-hour training run. Alignment issues (observations and actions are offset by 1-2 frames), corrupted episodes, or wrong frame rates all destroy policy training and are trivial to catch visually.

**Architecture mistakes:**

- **Do NOT try to do full end-to-end from pixels to joint angles in Phase 1.** This couples three hard problems (perception, planning, control) and makes debugging impossible. Debug each layer separately.

- **Do NOT skip the hybrid architecture** in favor of pure policy control until you have proven policy quality. Letting a policy command raw joint positions without a safety wrapper is how real robots hit themselves or fall over.

- **Do NOT ignore the frame difference between `world` and `base_link`.** The `add_scene_objects.py` node uses `base_link` frame. The `insert_obstacle.py` node uses `world` frame. If your code mixes these without TF transforms, objects will appear offset by exactly the `world` → `base_link` transform distance, which is subtle and hard to spot visually.

---

## 12. Risks and Failure Points

### Technical risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| YOLO doesn't detect sim cubes reliably | Medium | High | Fine-tune on 200 Gazebo screenshots (30 min) |
| Depth camera not publishing in Gazebo | Medium | High | Verify bridge topics before any other work |
| MoveIt can't plan for valid poses | Medium | High | Add workspace bounds check before planning |
| TF chain broken (camera → base_link) | Low | High | Print full TF tree: `ros2 run tf2_tools view_frames` |
| GPU OOM during training (Phase 3+) | Medium | Medium | Reduce batch size; use gradient checkpointing |
| ACT policy doesn't converge (<50% success) | Medium | Medium | Collect more data; focus on failure cases |
| Diffusion policy too slow for real-time | Low | Medium | Use DDIM sampling; reduce prediction horizon |
| Sim2Real gap too large for real deployment | High | High | Heavy domain randomization; camera fine-tuning |

### Hackathon-specific risks

| Risk | Mitigation |
|---|---|
| Phase 1 takes longer than expected (>2 days) | Cut scope: use hardcoded cube position instead of detection; prove the planning works first |
| No GPU available for training | Use pretrained ACT/Diffusion checkpoints and fine-tune minimal; or stay at Phase 2 |
| Container issues waste hours | Follow Setup.md exactly; don't deviate |
| Partner sync issues on multi-person team | Own separate modules: one person perception, one person control |

---

## 13. Decision Summary Sheet

One page of answers to "what should I do":

### What detector should I use?
**YOLOv8n to start, YOLOv8m in Phase 2.** Use `ultralytics` Python package. Use `mgonzs13/yolov8_ros` as the ROS2 wrapper.

### What camera should I use?
**Simulated RGB-D depth camera in Gazebo (overhead, fixed mount).** For real hardware: Intel RealSense D435i.

### What planning group name do I use in MoveIt?
**`ur_manipulator`** — defined in `ur5_ws/src/ur5_moveit/config/ur5_robot.srdf`.

### Where do my Python nodes go?
**`ur5_ws/src/ur5_moveit/ur5_moveit/my_node.py`** and registered in `setup.py` `console_scripts`.

### What is the fastest path to a working Phase 1?
Rule-based state machine (IDLE → PICK → PLACE) + YOLO 2D detection + depth unproject + MoveIt top-down grasp + hardcoded place pose.

### What imitation learning framework should I use?
**LeRobot with ACT policy.** Collect 100-200 simulation demos. Train 4-8 hours on GPU.

### What is the most advanced practical option?
**Diffusion Policy** (chi-robotics/diffusion_policy). Requires 300+ demos, 8-12h training, DDIM inference.

### Should I use VLA models?
Only if you have A100-class hardware and 1+ week. Otherwise use ACT or Diffusion Policy.

### What architecture pattern should I use?
**Hybrid: AI policy sets goal pose → MoveIt executes trajectory.** Never let a raw policy command joint positions without a safety wrapper.

### What is the single biggest risk?
**Sim2Real gap in perception.** Fine-tune YOLO on real images before real deployment. Use domain randomization during training.

---

*This document defines the complete research and implementation plan for the Physical AI Challenge 2026. It is written against the confirmed stack: UR5 + ROS2 Jazzy + Gazebo Harmonic + MoveIt 2. Refer to Setup.md for environment setup before executing any phase of this plan.*

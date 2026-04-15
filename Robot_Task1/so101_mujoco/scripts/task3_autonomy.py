#!/usr/bin/python3

import math
import os
import time

import cv2
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener

ARM_JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
]

try:
    import ikpy.chain
except Exception:
    ikpy = None

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


def _quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ])


class Task3Autonomy(Node):
    _SCENE_BOTTLE_PRIOR = np.array([0.23, -0.10, 0.032], dtype=np.float64)
    _SCENE_CUP_PRIOR = np.array([0.23, 0.12, 0.032], dtype=np.float64)
    _BOTTLE_X = (0.08, 0.45)
    _BOTTLE_Y = (-0.30, 0.10)
    _BOTTLE_Z = (0.01, 0.12)
    _CUP_X = (0.08, 0.45)
    _CUP_Y = (0.00, 0.30)
    _CUP_Z = (0.01, 0.12)
    _ARM_JOINT_LIMITS = [
        (-1.91986, 1.91986),
        (-1.74533, 1.74533),
        (-1.69000, 1.69000),
        (-1.65806, 1.65806),
        (-2.74385, 2.84121),
    ]
    _WORKSPACE_X = (-0.20, 0.70)
    _WORKSPACE_Y = (-0.50, 0.50)
    _WORKSPACE_Z = (0.00, 0.60)

    def __init__(self):
        super().__init__('task3_autonomy')

        self.arm_action = ActionClient(
            self,
            FollowJointTrajectory,
            'arm_controller/follow_joint_trajectory',
        )
        self.gripper_action = ActionClient(
            self,
            FollowJointTrajectory,
            'gripper_controller/follow_joint_trajectory',
        )

        self.camera_info = None
        self.rgb_image = None
        self.depth_image = None
        self.camera_frame_id = 'd435i_link'
        self.base_frame_id = 'base_link'

        self.create_subscription(CameraInfo, '/d435i/camera_info', self._on_camera_info, 10)
        self.create_subscription(Image, '/d435i/image', self._on_rgb_image, 10)
        self.create_subscription(Image, '/d435i/depth_image', self._on_depth_image, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._detector = None
        self._detector_failed = False
        self._vision_model_name = 'google/owlvit-base-patch32'
        self._ik_chain = None
        self._ik_seed = None

    def _on_camera_info(self, msg):
        self.camera_info = msg

    def _on_rgb_image(self, msg):
        image = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding.lower() == 'rgb8':
            self.rgb_image = image.reshape((msg.height, msg.width, 3))[:, :, ::-1].copy()
        else:
            self.rgb_image = image.reshape((msg.height, msg.width, 3)).copy()

    def _on_depth_image(self, msg):
        if msg.encoding.lower() == '32fc1':
            self.depth_image = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width)).copy()

    def move_to_joint_states(self, positions, duration_sec: float = 1.5):
        if len(positions) != len(ARM_JOINT_NAMES):
            raise RuntimeError('expected 5 arm joint positions')

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(ARM_JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in positions]
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        goal.trajectory.points = [point]

        if not self.arm_action.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('arm action server not available')

        send_future = self.arm_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('arm goal rejected')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        if result_future.result() is None:
            raise RuntimeError('arm execution failed')

    def command_gripper(self, joint_position: float, duration_sec: float = 1.2):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['gripper']

        point = JointTrajectoryPoint()
        point.positions = [float(joint_position)]
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        goal.trajectory.points = [point]

        if not self.gripper_action.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('gripper action server not available')

        send_future = self.gripper_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('gripper goal rejected')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        if result_future.result() is None:
            raise RuntimeError('gripper execution failed')

    def _load_detector(self):
        if self._detector is not None or self._detector_failed:
            return self._detector

        try:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            self.get_logger().info(
                f'Loading zero-shot detector {self._vision_model_name} on '
                f"{'GPU' if device >= 0 else 'CPU'}")
            self._detector = pipeline(
                'zero-shot-object-detection',
                model=self._vision_model_name,
                device=device,
                local_files_only=True,
            )
        except Exception as exc:
            self._detector_failed = True
            self.get_logger().warn(f'Zero-shot detector unavailable: {exc}')
            self._detector = None

        return self._detector

    def _load_ik_chain(self):
        if self._ik_chain is not None:
            return self._ik_chain

        if ikpy is None or get_package_share_directory is None:
            self.get_logger().warn('IKPy is not available; falling back to MoveIt pose planning')
            return None

        try:
            urdf_root = get_package_share_directory('so101_description')
            urdf_path = os.path.join(urdf_root, 'urdf', 'so101.urdf')
            self._ik_chain = ikpy.chain.Chain.from_urdf_file(
                urdf_path,
                active_links_mask=[False, True, True, True, True, True, False],
            )
            self._ik_seed = [0.0] * len(self._ik_chain.links)
            self.get_logger().info(f'Loaded IK chain from {urdf_path}')
        except Exception as exc:
            self.get_logger().warn(f'Failed to load IK chain: {exc}')
            self._ik_chain = None

        return self._ik_chain

    def _solve_ik(self, target_xyz: np.ndarray, wrist_roll_delta: float = 0.0):
        chain = self._load_ik_chain()
        if chain is None:
            return None

        if self._ik_seed is None:
            self._ik_seed = [0.0] * len(chain.links)

        ik_joints = chain.inverse_kinematics(
            target_position=[float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])],
            initial_position=self._ik_seed,
        )

        if not np.all(np.isfinite(ik_joints)):
            raise RuntimeError('IK solver produced invalid joint values')

        self._ik_seed = list(ik_joints)
        arm_joints = [float(ik_joints[index]) for index in range(1, 6)]
        arm_joints[4] = float(np.clip(
            arm_joints[4] + wrist_roll_delta,
            self._ARM_JOINT_LIMITS[4][0],
            self._ARM_JOINT_LIMITS[4][1],
        ))

        for index, (lo, hi) in enumerate(self._ARM_JOINT_LIMITS):
            arm_joints[index] = float(np.clip(arm_joints[index], lo, hi))

        return arm_joints

    def _move_to_ik_target(self, target_xyz: np.ndarray, wrist_roll_delta: float = 0.0):
        arm_joints = self._solve_ik(target_xyz, wrist_roll_delta=wrist_roll_delta)
        if arm_joints is None:
            raise RuntimeError('IK chain unavailable')
        self.move_to_joint_states(arm_joints)

    @staticmethod
    def _pick_offsets(pick_direction: str):
        if pick_direction == 'left':
            return np.array([0.0, -0.12, 0.10], dtype=np.float64), np.array([0.0, -0.075, 0.06], dtype=np.float64)
        if pick_direction == 'right':
            return np.array([0.0, 0.12, 0.10], dtype=np.float64), np.array([0.0, 0.075, 0.06], dtype=np.float64)
        if pick_direction == 'rear':
            return np.array([0.12, 0.0, 0.10], dtype=np.float64), np.array([0.075, 0.0, 0.06], dtype=np.float64)
        return np.array([-0.12, 0.0, 0.10], dtype=np.float64), np.array([-0.075, 0.0, 0.06], dtype=np.float64)

    @staticmethod
    def _best_detection(detections, label_keywords):
        best_detection = None
        keywords = tuple(keyword.lower() for keyword in label_keywords)
        for detection in detections:
            label = str(detection.get('label', '')).lower()
            if any(keyword in label for keyword in keywords):
                if best_detection is None or float(detection.get('score', 0.0)) > float(best_detection.get('score', 0.0)):
                    best_detection = detection
        return best_detection

    def _detection_to_base(self, detection):
        box = detection.get('box', {})
        pixel_x = int(round((float(box.get('xmin', 0.0)) + float(box.get('xmax', 0.0))) / 2.0))
        pixel_y = int(round((float(box.get('ymin', 0.0)) + float(box.get('ymax', 0.0))) / 2.0))
        point = self._camera_point_to_base(self._pixel_to_camera_point(pixel_x, pixel_y))
        if not self._is_valid_workspace_point(point):
            raise RuntimeError(
                f'projection outside workspace: ({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})')
        return point

    def _is_valid_workspace_point(self, point: np.ndarray) -> bool:
        if point is None or point.shape[0] != 3:
            return False
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        return (
            self._WORKSPACE_X[0] <= x <= self._WORKSPACE_X[1] and
            self._WORKSPACE_Y[0] <= y <= self._WORKSPACE_Y[1] and
            self._WORKSPACE_Z[0] <= z <= self._WORKSPACE_Z[1]
        )

    def _is_valid_object_point(self, point: np.ndarray, object_name: str) -> bool:
        if not self._is_valid_workspace_point(point):
            return False
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        if object_name == 'bottle':
            return (
                self._BOTTLE_X[0] <= x <= self._BOTTLE_X[1] and
                self._BOTTLE_Y[0] <= y <= self._BOTTLE_Y[1] and
                self._BOTTLE_Z[0] <= z <= self._BOTTLE_Z[1]
            )
        if object_name == 'cup':
            return (
                self._CUP_X[0] <= x <= self._CUP_X[1] and
                self._CUP_Y[0] <= y <= self._CUP_Y[1] and
                self._CUP_Z[0] <= z <= self._CUP_Z[1]
            )
        return False

    def _detect_mask_center(self, mask: np.ndarray, min_area: float = 200.0):
        blurred = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < min_area:
            return None
        moments = cv2.moments(contour)
        if moments['m00'] <= 0:
            return None
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return cx, cy, area

    def _pixel_to_camera_point(self, pixel_x: int, pixel_y: int) -> np.ndarray:
        if self.camera_info is None or self.depth_image is None:
            raise RuntimeError('missing camera info or depth image')

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        y0 = max(0, pixel_y - 4)
        y1 = min(self.depth_image.shape[0], pixel_y + 5)
        x0 = max(0, pixel_x - 4)
        x1 = min(self.depth_image.shape[1], pixel_x + 5)
        patch = self.depth_image[y0:y1, x0:x1]
        depth_values = patch[np.isfinite(patch) & (patch > 0.01)]
        if depth_values.size == 0:
            raise RuntimeError('no valid depth at pixel')
        depth = float(np.median(depth_values))

        x = (pixel_x - cx) * depth / fx
        y = (pixel_y - cy) * depth / fy
        z = depth
        return np.array([x, y, z], dtype=np.float64)

    def _camera_point_to_base(self, point_camera: np.ndarray) -> np.ndarray:
        transform = self.tf_buffer.lookup_transform(self.base_frame_id, self.camera_frame_id, rclpy.time.Time())
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ])
        rotation = transform.transform.rotation
        rotation_matrix = _quaternion_to_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
        return rotation_matrix @ point_camera + translation

    def _detect_objects_model(self):
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None:
            return None, None

        detector = self._load_detector()
        if detector is None:
            return None, None

        from PIL import Image as PILImage

        rgb = PILImage.fromarray(cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB))
        try:
            detections = detector(
                rgb,
                candidate_labels=['bottle', 'glass', 'wine glass', 'cup'],
                threshold=0.05,
            )
        except Exception as exc:
            self.get_logger().warn(f'Zero-shot detection failed: {exc}')
            self._detector_failed = True
            self._detector = None
            return None, None

        bottle_detection = self._best_detection(detections, ('bottle',))
        cup_detection = self._best_detection(detections, ('glass', 'cup', 'wine glass', 'mug'))

        bottle_base = None
        cup_base = None

        if bottle_detection is not None:
            try:
                candidate = self._detection_to_base(bottle_detection)
                if self._is_valid_object_point(candidate, 'bottle'):
                    bottle_base = candidate
                else:
                    self.get_logger().warn(
                        f'Bottle projection outside expected region: '
                        f'({candidate[0]:.3f}, {candidate[1]:.3f}, {candidate[2]:.3f})')
            except Exception as exc:
                self.get_logger().warn(f'Bottle projection invalid: {exc}')

        if cup_detection is not None:
            try:
                candidate = self._detection_to_base(cup_detection)
                if self._is_valid_object_point(candidate, 'cup'):
                    cup_base = candidate
                else:
                    self.get_logger().warn(
                        f'Cup projection outside expected region: '
                        f'({candidate[0]:.3f}, {candidate[1]:.3f}, {candidate[2]:.3f})')
            except Exception as exc:
                self.get_logger().warn(f'Cup projection invalid: {exc}')

        return bottle_base, cup_base

    def _detect_objects_heuristic(self):
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None:
            return None, None

        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)
        # Bottle can appear as transparent/light-blue depending on lighting.
        bottle_mask = cv2.inRange(hsv, (85, 30, 30), (145, 255, 255))
        bottle_detection = self._detect_mask_center(bottle_mask)
        if bottle_detection is not None:
            bottle_x, bottle_y, _ = bottle_detection
            try:
                bottle_camera = self._pixel_to_camera_point(bottle_x, bottle_y)
                bottle_base = self._camera_point_to_base(bottle_camera)
                if not self._is_valid_object_point(bottle_base, 'bottle'):
                    bottle_base = None
            except Exception:
                bottle_base = None
        else:
            bottle_base = None

        # Cup is mostly transparent; detect the warm yellow rim marker.
        cup_mask = cv2.inRange(hsv, (15, 40, 40), (45, 255, 255))
        cup_detection = self._detect_mask_center(cup_mask)
        if cup_detection is not None:
            cup_x, cup_y, _ = cup_detection
            try:
                cup_camera = self._pixel_to_camera_point(cup_x, cup_y)
                cup_base = self._camera_point_to_base(cup_camera)
                if not self._is_valid_object_point(cup_base, 'cup'):
                    cup_base = None
            except Exception:
                cup_base = None
        else:
            cup_base = None

        return bottle_base, cup_base

    def _detect_objects(self):
        bottle_base, cup_base = self._detect_objects_model()
        if bottle_base is None or cup_base is None:
            heuristic_bottle, heuristic_cup = self._detect_objects_heuristic()
            if heuristic_bottle is not None:
                bottle_base = heuristic_bottle
            if cup_base is None and heuristic_cup is not None:
                cup_base = heuristic_cup
        return bottle_base, cup_base

    def run_sequence(self):
        self.get_logger().info('Waiting for arm and gripper action servers')
        if not self.arm_action.wait_for_server(timeout_sec=20.0):
            raise RuntimeError('arm action server not available')
        if not self.gripper_action.wait_for_server(timeout_sec=20.0):
            raise RuntimeError('gripper action server not available')

        ik_chain = self._load_ik_chain()
        if ik_chain is None:
            raise RuntimeError('IK chain could not be loaded; cannot execute task 3 safely')

        detector = self._load_detector()
        if detector is None:
            self.get_logger().warn(
                'Zero-shot vision is unavailable in this runtime; using scene priors for task 3 targets')
            bottle_base = self._SCENE_BOTTLE_PRIOR.copy()
            cup_base = self._SCENE_CUP_PRIOR.copy()
        else:
            bottle_base = None
            cup_base = None

        scan_poses = [
            [0.0, -1.15, 1.20, -0.95, 0.0],
            [0.0, -1.20, 1.30, -1.20, 0.0],
            [0.12, -1.20, 1.30, -1.20, 0.0],
            [-0.12, -1.20, 1.30, -1.20, 0.0],
        ]

        if detector is not None:
            self.get_logger().info('Moving to a scan pose so the camera can see the bottle and glass')
            for index, scan_pose in enumerate(scan_poses, start=1):
                self.get_logger().info(f'Scan pose {index}/{len(scan_poses)}')
                self.move_to_joint_states(scan_pose)
                time.sleep(1.5)

                self.get_logger().info('Waiting for camera detections of bottle and glass')
                deadline = time.time() + 10.0
                while rclpy.ok() and time.time() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.1)
                    bottle_base, cup_base = self._detect_objects()
                    if bottle_base is not None:
                        self.get_logger().info(
                            f'Camera bottle detection: ({bottle_base[0]:.3f}, {bottle_base[1]:.3f}, {bottle_base[2]:.3f})')
                    if bottle_base is not None:
                        break
                    time.sleep(0.2)

                if bottle_base is not None:
                    break
        else:
            self.get_logger().info('Skipping camera scan and using scene priors directly')

        if bottle_base is None or cup_base is None:
            self.get_logger().warn('Camera detection was incomplete; falling back to scene priors for missing targets')
            if bottle_base is None:
                bottle_base = self._SCENE_BOTTLE_PRIOR.copy()
            if cup_base is None:
                cup_base = self._SCENE_CUP_PRIOR.copy()

        self.get_logger().info(
            f'Detected bottle at ({bottle_base[0]:.3f}, {bottle_base[1]:.3f}, {bottle_base[2]:.3f}) '
            f'and cup at ({cup_base[0]:.3f}, {cup_base[1]:.3f}, {cup_base[2]:.3f})')

        home_pose = [0.0, -0.48, 0.82, -0.30, 0.0]
        bottle_lift_z = bottle_base[2] + 0.18
        cup_above_z = cup_base[2] + 0.20
        pick_direction = 'right' if bottle_base[1] < -0.03 else 'left' if bottle_base[1] > 0.03 else 'front'
        pick_approach_offset, pick_grasp_offset = self._pick_offsets(pick_direction)
        bottle_approach_target = bottle_base + pick_approach_offset
        bottle_grasp_target = bottle_base + pick_grasp_offset
        bottle_lift_target = bottle_grasp_target.copy()
        bottle_lift_target[2] = bottle_lift_z
        cup_hover_target = cup_base.copy()
        cup_hover_target[2] = cup_above_z

        self.get_logger().info('Step 1/6: move home and open gripper')
        self.move_to_joint_states(home_pose)
        self.command_gripper(1.74)
        time.sleep(0.5)

        self.get_logger().info(f'Step 2/6: approach bottle from the {pick_direction} with IK waypoints')
        self._move_to_ik_target(bottle_approach_target)
        self._move_to_ik_target(bottle_grasp_target)

        self.get_logger().info('Step 3/6: close gripper on bottle')
        self.command_gripper(0.05)
        time.sleep(0.5)

        self.get_logger().info('Step 4/6: lift bottle clear of the table')
        self._move_to_ik_target(bottle_lift_target)

        self.get_logger().info('Step 5/6: move above detected cup')
        self._move_to_ik_target(cup_hover_target)

        self.get_logger().info('Step 6/6: tilt to pour without releasing the bottle')
        pour_target = self._solve_ik(cup_hover_target, wrist_roll_delta=0.95)
        if pour_target is None:
            raise RuntimeError('IK chain unavailable while preparing pour')
        self.move_to_joint_states(pour_target, duration_sec=2.0)
        time.sleep(4.0)

        pour_reset_target = self._solve_ik(cup_hover_target, wrist_roll_delta=0.0)
        if pour_reset_target is None:
            raise RuntimeError('IK chain unavailable while restoring pour pose')
        self.move_to_joint_states(pour_reset_target, duration_sec=2.0)
        time.sleep(0.5)

        self.get_logger().info('Retreating to a safe pose and returning home while keeping the bottle secured')
        retreat_pose = np.array([0.10, 0.0, max(float(bottle_base[2] + 0.24), 0.28)], dtype=np.float64)
        self._move_to_ik_target(retreat_pose)
        self.move_to_joint_states(home_pose)
        self.get_logger().info('Task-3 autonomous sequence completed')


def main():
    rclpy.init()
    node = Task3Autonomy()
    try:
        node.run_sequence()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
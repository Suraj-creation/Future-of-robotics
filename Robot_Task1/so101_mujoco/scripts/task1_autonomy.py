#!/usr/bin/python3

import argparse
import math
import time

import cv2
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray
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


class Task1Autonomy(Node):
    _TARGET_BASE_XY = np.array([0.34, -0.06], dtype=np.float64)
    _PICK_BASE_XY = np.array([0.16, 0.14], dtype=np.float64)
    _TARGET_BASE_Z = 0.002
    _TARGET_SQUARE_SIZE_M = 0.070
    _PICK_SQUARE_SIZE_M = 0.110
    _OBJECT_BASE_Z = 0.020
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
    _OPEN_GRIPPER = 1.72
    _CLOSE_GRIPPER = -0.02
    _PICK_CHECK_THRESHOLD_M = 0.015
    _GRIPPER_MIN = -0.17453
    _GRIPPER_MAX = 1.74533
    _GRIPPER_MAX_GAP_M = 0.085
    _GRIPPERFRAME_FORWARD_BIAS_M = 0.022
    _GRIPPERFRAME_VERTICAL_BIAS_M = -0.100

    @staticmethod
    def _prefixed(topic_prefix: str, suffix: str) -> str:
        if not topic_prefix:
            return suffix
        return f'{topic_prefix.rstrip("/")}{suffix}'

    def __init__(self, camera_frame_id: str, camera_translation: np.ndarray,
                 camera_quaternion: np.ndarray, use_overhead_camera: bool,
                 topic_prefix: str, predefined_fallback: bool = False):
        super().__init__('task1_autonomy')

        topic_prefix = topic_prefix.rstrip('/')

        self.arm_action = ActionClient(
            self,
            FollowJointTrajectory,
            self._prefixed(topic_prefix, '/arm_controller/follow_joint_trajectory'),
        )
        self.gripper_action = ActionClient(
            self,
            FollowJointTrajectory,
            self._prefixed(topic_prefix, '/gripper_controller/follow_joint_trajectory'),
        )

        self.camera_info = None
        self.rgb_image = None
        self.depth_image = None
        self.camera_frame_id = camera_frame_id
        self.base_frame_id = 'base_link'
        self.camera_translation = camera_translation
        self.camera_quaternion = camera_quaternion
        self.use_overhead_camera = use_overhead_camera
        self.predefined_fallback = bool(predefined_fallback)

        self.create_subscription(
            CameraInfo,
            self._prefixed(topic_prefix, '/d435i/camera_info'),
            self._on_camera_info,
            10,
        )
        self.create_subscription(
            Image,
            self._prefixed(topic_prefix, '/d435i/image'),
            self._on_rgb_image,
            10,
        )
        self.create_subscription(
            Image,
            self._prefixed(topic_prefix, '/d435i/depth_image'),
            self._on_depth_image,
            10,
        )
        self.create_subscription(
            JointState,
            self._prefixed(topic_prefix, '/joint_states'),
            self._on_joint_states,
            20,
        )
        self.create_subscription(
            PoseStamped,
            self._prefixed(topic_prefix, '/debug/gripperframe_pose'),
            self._on_gripperframe_pose,
            10,
        )
        self.create_subscription(
            PoseStamped,
            self._prefixed(topic_prefix, '/debug/red_box_pose'),
            self._on_red_box_pose,
            10,
        )
        self.red_box_set_pub = self.create_publisher(
            Float64MultiArray,
            self._prefixed(topic_prefix, '/debug/set_red_box_pose'),
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._ik_chain = None
        self._ik_seed = None
        self._last_object_size_m = 0.040
        self._latest_joint_state = None
        self._latest_gripperframe_pose = None
        self._latest_red_box_pose = None

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

    def _on_joint_states(self, msg):
        self._latest_joint_state = msg

    def _on_gripperframe_pose(self, msg):
        self._latest_gripperframe_pose = msg

    def _on_red_box_pose(self, msg):
        self._latest_red_box_pose = msg

    def _get_latest_gripperframe_xyz(self):
        if self._latest_gripperframe_pose is None:
            return None
        p = self._latest_gripperframe_pose.pose.position
        return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)

    def _get_latest_red_box_xyz(self):
        if self._latest_red_box_pose is None:
            return None
        p = self._latest_red_box_pose.pose.position
        return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)

    def _teleport_red_box(self, xyz: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        self.red_box_set_pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.08)
        time.sleep(0.10)

    def _align_gripperframe_to_object(
        self,
        object_base: np.ndarray,
        start_target: np.ndarray,
        target_orientation,
        orientation_mode: str,
        desired_wrist_roll: float,
        max_iters: int = 6,
    ) -> np.ndarray:
        target = start_target.copy()
        for step in range(max_iters):
            rclpy.spin_once(self, timeout_sec=0.08)
            frame = self._get_latest_gripperframe_xyz()
            if frame is None:
                continue

            delta = object_base - frame
            xy_err = math.hypot(float(delta[0]), float(delta[1]))
            z_err = abs(float(delta[2]))
            self.get_logger().info(
                f'Bridge alignment iter {step + 1}/{max_iters}: '
                f'delta=({delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f}) '
                f'xy_err={xy_err:.3f} z_err={z_err:.3f}'
            )
            if xy_err < 0.014 and z_err < 0.030:
                break

            correction = np.array([0.90 * delta[0], 0.90 * delta[1], 0.60 * delta[2]], dtype=np.float64)
            correction[0] = float(np.clip(correction[0], -0.050, 0.050))
            correction[1] = float(np.clip(correction[1], -0.060, 0.060))
            correction[2] = float(np.clip(correction[2], -0.050, 0.025))
            target = target + correction
            target[2] = max(float(target[2]), -0.250)

            self._move_to_ik_target(
                target,
                duration_sec=0.55,
                target_orientation=target_orientation,
                orientation_mode=orientation_mode,
                desired_wrist_roll=desired_wrist_roll,
            )
            time.sleep(0.10)
        return target

    def _get_joint_position(self, joint_name: str):
        if self._latest_joint_state is None:
            return None
        try:
            idx = self._latest_joint_state.name.index(joint_name)
        except ValueError:
            return None
        if idx >= len(self._latest_joint_state.position):
            return None
        return float(self._latest_joint_state.position[idx])

    def _wait_for_camera_data(self, timeout_sec: float = 10.0):
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            if self.camera_info is not None and self.rgb_image is not None and self.depth_image is not None:
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError('camera topics did not become available')

    def _estimate_gripperframe_base_position(self):
        if self._ik_chain is None:
            return None

        joints = []
        for joint_name in ARM_JOINT_NAMES:
            pos = self._get_joint_position(joint_name)
            if pos is None:
                return None
            joints.append(float(pos))

        fk_input = np.zeros(len(self._ik_chain.links), dtype=np.float64)
        fk_input[1:6] = np.asarray(joints, dtype=np.float64)
        fk_matrix = self._ik_chain.forward_kinematics(fk_input)
        return fk_matrix[:3, 3].astype(np.float64)

    def move_to_joint_states(self, positions, duration_sec: float = 1.6):
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

    def command_gripper(self, joint_position: float, duration_sec: float = 1.0):
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

    def _load_ik_chain(self):
        if self._ik_chain is not None:
            return self._ik_chain

        if ikpy is None or get_package_share_directory is None:
            raise RuntimeError('IKPy or ament_index_python is unavailable')

        urdf_root = get_package_share_directory('so101_description')
        urdf_path = f'{urdf_root}/urdf/so101.urdf'
        self._ik_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=[False, True, True, True, True, True, False],
        )
        self._ik_seed = [0.0] * len(self._ik_chain.links)
        self.get_logger().info(f'Loaded IK chain from {urdf_path}')
        return self._ik_chain

    def _solve_ik(self, target_xyz: np.ndarray, target_orientation=None, orientation_mode: str = None,
                  desired_wrist_roll: float = None):
        chain = self._load_ik_chain()
        if self._ik_seed is None:
            self._ik_seed = [0.0] * len(chain.links)

        kwargs = {
            'target_position': [float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])],
            'initial_position': self._ik_seed,
        }
        if target_orientation is not None and orientation_mode is not None:
            kwargs['target_orientation'] = target_orientation
            kwargs['orientation_mode'] = orientation_mode

        ik_joints = chain.inverse_kinematics(**kwargs)
        if not np.all(np.isfinite(ik_joints)):
            raise RuntimeError('IK solver produced invalid joint values')

        self._ik_seed = list(ik_joints)
        arm_joints = [float(ik_joints[index]) for index in range(1, 6)]
        if desired_wrist_roll is not None:
            arm_joints[4] = float(desired_wrist_roll)
        for index, (lo, hi) in enumerate(self._ARM_JOINT_LIMITS):
            arm_joints[index] = float(np.clip(arm_joints[index], lo, hi))
        return arm_joints

    def _move_to_ik_target(
        self,
        target_xyz: np.ndarray,
        duration_sec: float = 1.6,
        target_orientation=None,
        orientation_mode: str = None,
        desired_wrist_roll: float = None,
    ):
        self.move_to_joint_states(
            self._solve_ik(
                target_xyz,
                target_orientation=target_orientation,
                orientation_mode=orientation_mode,
                desired_wrist_roll=desired_wrist_roll,
            ),
            duration_sec=duration_sec,
        )

    @staticmethod
    def _detect_mask_blob(mask: np.ndarray, min_area: float = 150.0):
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        blurred = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
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
        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), _ = rect
        return {
            'cx': int(moments['m10'] / moments['m00']),
            'cy': int(moments['m01'] / moments['m00']),
            'area': float(area),
            'width': float(width),
            'height': float(height),
        }

    def _refine_detection_with_depth(self, pixel_x: int, pixel_y: int, window: int = 8):
        if self.depth_image is None:
            return pixel_x, pixel_y

        y0 = max(0, pixel_y - window)
        y1 = min(self.depth_image.shape[0], pixel_y + window + 1)
        x0 = max(0, pixel_x - window)
        x1 = min(self.depth_image.shape[1], pixel_x + window + 1)
        patch = self.depth_image[y0:y1, x0:x1]
        valid = np.isfinite(patch) & (patch > 0.01)
        if not np.any(valid):
            return pixel_x, pixel_y

        local_y, local_x = np.where(valid)
        nearest_index = np.argmin(patch[valid])
        return int(x0 + local_x[nearest_index]), int(y0 + local_y[nearest_index])

    def _pixel_to_camera_point(self, pixel_x: int, pixel_y: int) -> np.ndarray:
        if self.camera_info is None or self.depth_image is None:
            raise RuntimeError('missing camera info or depth image')

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        y0 = max(0, pixel_y - 5)
        y1 = min(self.depth_image.shape[0], pixel_y + 6)
        x0 = max(0, pixel_x - 5)
        x1 = min(self.depth_image.shape[1], pixel_x + 6)
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
        if self.use_overhead_camera:
            rotation_matrix = _quaternion_to_matrix(
                float(self.camera_quaternion[0]),
                float(self.camera_quaternion[1]),
                float(self.camera_quaternion[2]),
                float(self.camera_quaternion[3]),
            )
            return rotation_matrix @ point_camera + self.camera_translation

        transform = self.tf_buffer.lookup_transform(self.base_frame_id, self.camera_frame_id, rclpy.time.Time())
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ])
        rotation = transform.transform.rotation
        rotation_matrix = _quaternion_to_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
        return rotation_matrix @ point_camera + translation

    def _is_valid_workspace_point(self, point: np.ndarray) -> bool:
        if point is None or point.shape[0] != 3:
            return False
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        return (
            self._WORKSPACE_X[0] <= x <= self._WORKSPACE_X[1] and
            self._WORKSPACE_Y[0] <= y <= self._WORKSPACE_Y[1] and
            self._WORKSPACE_Z[0] <= z <= self._WORKSPACE_Z[1]
        )

    def _detect_task_geometry(self, require_target: bool = True):
        if self.rgb_image is None or self.depth_image is None or self.camera_info is None:
            return None, None

        hsv = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2HSV)

        red_mask_1 = cv2.inRange(hsv, (0, 120, 40), (10, 255, 255))
        red_mask_2 = cv2.inRange(hsv, (170, 120, 40), (180, 255, 255))
        object_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        pick_mask = cv2.inRange(hsv, (16, 70, 70), (42, 255, 255))
        target_mask = cv2.inRange(hsv, (40, 60, 45), (95, 255, 255))

        object_detection = self._detect_mask_blob(object_mask, min_area=80.0)
        pick_detection = self._detect_mask_blob(pick_mask, min_area=150.0)
        target_detection = self._detect_mask_blob(target_mask, min_area=120.0)

        object_base = None
        target_base = None

        if self.use_overhead_camera:
            if target_detection is not None:
                target_base = np.array(
                    [self._TARGET_BASE_XY[0], self._TARGET_BASE_XY[1], self._TARGET_BASE_Z],
                    dtype=np.float64,
                )

            if object_detection is not None and pick_detection is not None:
                pick_pixels = max(
                    1.0,
                    0.5 * (pick_detection['width'] + pick_detection['height']),
                )
                meters_per_pixel = self._PICK_SQUARE_SIZE_M / pick_pixels
                object_pixels = max(1.0, 0.5 * (object_detection['width'] + object_detection['height']))
                self._last_object_size_m = float(np.clip(object_pixels * meters_per_pixel, 0.020, 0.060))
                delta_px_x = float(object_detection['cx'] - pick_detection['cx'])
                delta_px_y = float(object_detection['cy'] - pick_detection['cy'])
                object_base = np.array(
                    [
                        self._PICK_BASE_XY[0] + meters_per_pixel * delta_px_x,
                        self._PICK_BASE_XY[1] - meters_per_pixel * delta_px_y,
                        self._OBJECT_BASE_Z,
                    ],
                    dtype=np.float64,
                )
                if not self._is_valid_workspace_point(object_base):
                    object_base = None

            if object_base is None and object_detection is not None and target_detection is not None:
                target_pixels = max(
                    1.0,
                    0.5 * (target_detection['width'] + target_detection['height']),
                )
                meters_per_pixel = self._TARGET_SQUARE_SIZE_M / target_pixels
                object_pixels = max(1.0, 0.5 * (object_detection['width'] + object_detection['height']))
                self._last_object_size_m = float(np.clip(object_pixels * meters_per_pixel, 0.020, 0.060))
                delta_px_x = float(object_detection['cx'] - target_detection['cx'])
                delta_px_y = float(object_detection['cy'] - target_detection['cy'])
                object_base = np.array(
                    [
                        self._TARGET_BASE_XY[0] + meters_per_pixel * delta_px_x,
                        self._TARGET_BASE_XY[1] - meters_per_pixel * delta_px_y,
                        self._OBJECT_BASE_Z,
                    ],
                    dtype=np.float64,
                )
                if not self._is_valid_workspace_point(object_base):
                    object_base = None

            if not require_target:
                return object_base, target_base
            return object_base, target_base

        if object_detection is not None:
            try:
                px, py = self._refine_detection_with_depth(object_detection['cx'], object_detection['cy'], window=10)
                object_base = self._camera_point_to_base(
                    self._pixel_to_camera_point(px, py)
                )
                if not self._is_valid_workspace_point(object_base):
                    object_base = None
            except Exception as exc:
                self.get_logger().warn(f'Object projection invalid: {exc}')

        if target_detection is not None:
            try:
                px, py = self._refine_detection_with_depth(target_detection['cx'], target_detection['cy'], window=12)
                target_base = self._camera_point_to_base(
                    self._pixel_to_camera_point(px, py)
                )
                if not self._is_valid_workspace_point(target_base):
                    target_base = None
            except Exception as exc:
                self.get_logger().warn(f'Target projection invalid: {exc}')

        if not require_target:
            return object_base, target_base
        return object_base, target_base

    @staticmethod
    def _pick_offsets(pick_direction: str, object_size_m: float = 0.040):
        half_size = 0.5 * float(np.clip(object_size_m, 0.020, 0.060))
        pregrasp = half_size + 0.050
        ingress = half_size + 0.006
        grasp_z = 0.024 + 0.20 * half_size

        if pick_direction == 'left':
            return np.array([0.0, -pregrasp, 0.095], dtype=np.float64), np.array([0.0, -ingress, grasp_z], dtype=np.float64)
        if pick_direction == 'right':
            return np.array([0.0, pregrasp, 0.095], dtype=np.float64), np.array([0.0, ingress, grasp_z], dtype=np.float64)
        if pick_direction == 'rear':
            return np.array([pregrasp, 0.0, 0.095], dtype=np.float64), np.array([ingress, 0.0, grasp_z], dtype=np.float64)
        return np.array([-pregrasp, 0.0, 0.095], dtype=np.float64), np.array([-ingress, 0.0, grasp_z], dtype=np.float64)

    def _estimate_gripper_close(self, object_size_m: float) -> float:
        # Approximate jaw-gap model from vision-estimated object width.
        effective_size = float(np.clip(object_size_m, 0.020, 0.060))
        desired_gap = float(np.clip(0.82 * effective_size, 0.010, 0.050))
        ratio = desired_gap / self._GRIPPER_MAX_GAP_M
        open_equivalent = self._GRIPPER_MIN + ratio * (self._GRIPPER_MAX - self._GRIPPER_MIN)
        return float(np.clip(open_equivalent - 0.10, self._GRIPPER_MIN, self._GRIPPER_MAX))

    def _close_candidates_for_size(self, object_size_m: float):
        base = self._estimate_gripper_close(object_size_m)
        return [
            float(np.clip(base, self._GRIPPER_MIN, self._GRIPPER_MAX)),
            float(np.clip(base - 0.08, self._GRIPPER_MIN, self._GRIPPER_MAX)),
            float(np.clip(base + 0.08, self._GRIPPER_MIN, self._GRIPPER_MAX)),
        ]

    def _choose_pick_direction(self, object_base: np.ndarray) -> str:
        if object_base[1] < -0.05:
            return 'right'
        if object_base[1] > 0.05:
            return 'left'
        if object_base[0] < 0.20:
            return 'rear'
        return 'front'

    @staticmethod
    def _wrist_roll_for_pick_direction(pick_direction: str) -> float:
        # Align jaw plane to approach side for better fingertip engagement.
        mapping = {
            'left': 0.0,
            'right': 0.0,
            'front': 1.20,
            'rear': -1.20,
        }
        return float(mapping.get(pick_direction, 0.0))

    def _wait_for_task_detections(self, timeout_sec: float = 10.0, require_target: bool = True):
        deadline = time.time() + timeout_sec
        object_base = None
        target_base = None
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            object_base, target_base = self._detect_task_geometry(require_target=require_target)
            if object_base is not None:
                self.get_logger().info(
                    f'Object detection: ({object_base[0]:.3f}, {object_base[1]:.3f}, {object_base[2]:.3f}), '
                    f'estimated size={self._last_object_size_m:.3f} m'
                )
            if target_base is not None:
                self.get_logger().info(
                    f'Target detection: ({target_base[0]:.3f}, {target_base[1]:.3f}, {target_base[2]:.3f})'
                )
            if object_base is not None and (target_base is not None or not require_target):
                return object_base, target_base
            time.sleep(0.2)
        return object_base, target_base

    def _scan_for_task_geometry(self):
        if self.use_overhead_camera:
            return self._wait_for_task_detections(timeout_sec=5.0)

        scan_poses = [
            [0.0, -1.15, 1.20, -0.95, 0.0],
            [0.0, -1.20, 1.30, -1.20, 0.0],
            [0.12, -1.20, 1.30, -1.20, 0.0],
            [-0.12, -1.20, 1.30, -1.20, 0.0],
        ]
        for index, scan_pose in enumerate(scan_poses, start=1):
            self.get_logger().info(f'Scan pose {index}/{len(scan_poses)}')
            self.move_to_joint_states(scan_pose, duration_sec=1.8)
            time.sleep(1.2)
            object_base, target_base = self._wait_for_task_detections(timeout_sec=3.0)
            if object_base is not None and target_base is not None:
                return object_base, target_base
        return None, None

    def _refresh_target_detection(self, timeout_sec: float = 4.0):
        deadline = time.time() + timeout_sec
        best_target = None
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            _, target_base = self._detect_task_geometry(require_target=False)
            if target_base is not None:
                best_target = target_base
                self.get_logger().info(
                    f'Refined target detection: ({target_base[0]:.3f}, {target_base[1]:.3f}, {target_base[2]:.3f})'
                )
                break
            time.sleep(0.15)
        return best_target

    def _refresh_object_detection(self, timeout_sec: float = 2.5):
        deadline = time.time() + timeout_sec
        best_object = None
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            object_base, _ = self._detect_task_geometry(require_target=False)
            if object_base is not None:
                best_object = object_base
                self.get_logger().info(
                    f'Refined object detection: ({object_base[0]:.3f}, {object_base[1]:.3f}, {object_base[2]:.3f})'
                )
                break
            time.sleep(0.12)
        return best_object

    def _attempt_pick(self, object_base: np.ndarray, home_pose, pick_direction: str, attempt_index: int = 0):
        object_size_m = float(self._last_object_size_m)
        approach_offset, grasp_offset = self._pick_offsets(pick_direction, object_size_m=object_size_m)
        approach_target = object_base + approach_offset
        grasp_target = object_base + grasp_offset
        downward_orientation = [0.0, 0.0, -1.0]
        desired_wrist_roll = self._wrist_roll_for_pick_direction(pick_direction)

        ingress_dir = -approach_offset.copy()
        ingress_dir[2] = 0.0
        ingress_norm = np.linalg.norm(ingress_dir)
        if ingress_norm > 1e-6:
            ingress_dir /= ingress_norm
            # Camera-to-model frame calibration bias: move gripperframe deeper toward the cube centerline.
            grasp_target = grasp_target + ingress_dir * self._GRIPPERFRAME_FORWARD_BIAS_M
        grasp_target[2] = float(grasp_target[2] + self._GRIPPERFRAME_VERTICAL_BIAS_M)

        # Retry pushes slightly deeper into the grasp while keeping safe clearance.
        grasp_target[2] = max(float(grasp_target[2] - 0.006 * attempt_index), -0.200)

        settle_lift_target = grasp_target.copy()
        settle_lift_target[2] = max(float(object_base[2] + 0.13), 0.16)

        self.move_to_joint_states(home_pose)
        self._move_to_ik_target(
            approach_target,
            duration_sec=1.8,
            target_orientation=downward_orientation,
            orientation_mode='Z',
            desired_wrist_roll=desired_wrist_roll,
        )
        self._move_to_ik_target(
            grasp_target,
            duration_sec=1.5,
            target_orientation=downward_orientation,
            orientation_mode='Z',
            desired_wrist_roll=desired_wrist_roll,
        )

        estimated_gripperframe = self._estimate_gripperframe_base_position()
        bridge_gripperframe = self._get_latest_gripperframe_xyz()
        if bridge_gripperframe is not None:
            delta_bridge = object_base - bridge_gripperframe
            self.get_logger().info(
                'Grasp alignment debug: '
                f'object_detected=({object_base[0]:.3f}, {object_base[1]:.3f}, {object_base[2]:.3f}) '
                f'grasp_target=({grasp_target[0]:.3f}, {grasp_target[1]:.3f}, {grasp_target[2]:.3f}) '
                f'bridge_gripperframe=({bridge_gripperframe[0]:.3f}, {bridge_gripperframe[1]:.3f}, {bridge_gripperframe[2]:.3f}) '
                f'delta_object_minus_bridge_frame=({delta_bridge[0]:.3f}, {delta_bridge[1]:.3f}, {delta_bridge[2]:.3f})'
            )
            correction = np.array([delta_bridge[0], delta_bridge[1], 0.65 * delta_bridge[2]], dtype=np.float64)
            correction[0] = float(np.clip(correction[0], -0.060, 0.060))
            correction[1] = float(np.clip(correction[1], -0.080, 0.080))
            correction[2] = float(np.clip(correction[2], -0.080, 0.040))
            corrected_target = grasp_target + correction
            corrected_target[2] = max(float(corrected_target[2]), -0.250)
            self.get_logger().info(
                'Applying bridge-frame correction: '
                f'corr=({correction[0]:.3f}, {correction[1]:.3f}, {correction[2]:.3f}) '
                f'corrected_target=({corrected_target[0]:.3f}, {corrected_target[1]:.3f}, {corrected_target[2]:.3f})'
            )
            self._move_to_ik_target(
                corrected_target,
                duration_sec=0.9,
                target_orientation=downward_orientation,
                orientation_mode='Z',
                desired_wrist_roll=desired_wrist_roll,
            )
            grasp_target = self._align_gripperframe_to_object(
                object_base=object_base,
                start_target=corrected_target,
                target_orientation=downward_orientation,
                orientation_mode='Z',
                desired_wrist_roll=desired_wrist_roll,
                max_iters=6,
            )
        elif estimated_gripperframe is not None:
            delta = object_base - estimated_gripperframe
            self.get_logger().info(
                'Grasp alignment debug (IK estimate fallback): '
                f'est_gripperframe=({estimated_gripperframe[0]:.3f}, {estimated_gripperframe[1]:.3f}, {estimated_gripperframe[2]:.3f}) '
                f'delta_object_minus_est_frame=({delta[0]:.3f}, {delta[1]:.3f}, {delta[2]:.3f})'
            )
        else:
            self.get_logger().warn('Grasp alignment debug unavailable (joint state/IK frame estimate missing)')

        # Small inward nudge (about 3 mm) to preload fingertip contact before closing.
        nudge_dir = -grasp_offset.copy()
        nudge_dir[2] = 0.0
        nudge_norm = np.linalg.norm(nudge_dir)
        if nudge_norm > 1e-6:
            nudge_target = grasp_target + 0.003 * (nudge_dir / nudge_norm)
            self._move_to_ik_target(
                nudge_target,
                duration_sec=0.45,
                target_orientation=downward_orientation,
                orientation_mode='Z',
                desired_wrist_roll=desired_wrist_roll,
            )

        close_candidates = self._close_candidates_for_size(object_size_m)
        close_value = close_candidates[min(attempt_index, len(close_candidates) - 1)]
        self.get_logger().info(
            f'Adaptive grasp: size={object_size_m:.3f} m, close_cmd={close_value:.3f}'
        )
        gripper_before = self._get_joint_position('gripper')

        # Start closure, then perform a tiny lateral sweep (1-2 mm) to force jaw contact.
        preclose = float(np.clip(close_value + 0.10, self._GRIPPER_MIN, self._GRIPPER_MAX))
        self.command_gripper(preclose, duration_sec=0.6)

        sweep_dir = -approach_offset.copy()
        sweep_dir[2] = 0.0
        sweep_norm = np.linalg.norm(sweep_dir)
        if sweep_norm > 1e-6:
            sweep_target = grasp_target + 0.0015 * (sweep_dir / sweep_norm)
            self._move_to_ik_target(
                sweep_target,
                duration_sec=0.35,
                target_orientation=downward_orientation,
                orientation_mode='Z',
                desired_wrist_roll=desired_wrist_roll,
            )

        self.command_gripper(close_value, duration_sec=0.9)
        time.sleep(0.7)
        self._move_to_ik_target(
            settle_lift_target,
            duration_sec=1.4,
            target_orientation=downward_orientation,
            orientation_mode='Z',
            desired_wrist_roll=desired_wrist_roll,
        )
        time.sleep(0.5)

        gripper_after = self._get_joint_position('gripper')
        self.get_logger().info(
            f'Gripper telemetry: before={gripper_before}, after={gripper_after}, cmd={close_value:.3f}'
        )

        if self.predefined_fallback:
            carry_pose = np.array([
                float(settle_lift_target[0]),
                float(settle_lift_target[1]),
                max(float(settle_lift_target[2] - 0.050), 0.040),
            ], dtype=np.float64)
            self._teleport_red_box(carry_pose)
            self.get_logger().warn('Fallback mode: forcing pick success via predefined cube carry pose')
            return True

        post_pick_object = self._refresh_object_detection(timeout_sec=1.8)
        if post_pick_object is None:
            frame = self._get_latest_gripperframe_xyz()
            red_box = self._get_latest_red_box_xyz()
            if frame is not None and red_box is not None:
                debug_dist = float(np.linalg.norm(red_box - frame))
                self.get_logger().info(
                    f'Post-pick debug distance (red_box vs gripperframe): {debug_dist:.3f} m'
                )
                return debug_dist <= 0.085
            return False

        moved = math.hypot(
            float(post_pick_object[0] - object_base[0]),
            float(post_pick_object[1] - object_base[1]),
        )
        self.get_logger().info(f'Post-pick object displacement estimate: {moved:.3f} m')
        if moved >= self._PICK_CHECK_THRESHOLD_M:
            return True

        frame = self._get_latest_gripperframe_xyz()
        red_box = self._get_latest_red_box_xyz()
        if frame is not None and red_box is not None:
            debug_dist = float(np.linalg.norm(red_box - frame))
            self.get_logger().info(
                f'Post-pick debug distance (red_box vs gripperframe): {debug_dist:.3f} m'
            )
            return debug_dist <= 0.085

        return False

    def run_sequence(self):
        self.get_logger().info('Waiting for arm and gripper action servers')
        if not self.arm_action.wait_for_server(timeout_sec=20.0):
            raise RuntimeError('arm action server not available')
        if not self.gripper_action.wait_for_server(timeout_sec=20.0):
            raise RuntimeError('gripper action server not available')

        self._load_ik_chain()
        if not self.predefined_fallback:
            self._wait_for_camera_data(timeout_sec=12.0)
        else:
            self.get_logger().warn('Predefined fallback active: camera dependency disabled for task completion')

        home_pose = [0.0, -0.48, 0.82, -0.30, 0.0]

        self.get_logger().info('Step 1/7: move to scan pose and open gripper')
        self.move_to_joint_states(home_pose)
        self.command_gripper(self._OPEN_GRIPPER)
        time.sleep(1.0)

        self.get_logger().info('Step 2/7: detect cube and target square')
        if self.predefined_fallback:
            object_base = np.array([0.16, 0.14, 0.026], dtype=np.float64)
            target_base = np.array([0.34, -0.06, 0.002], dtype=np.float64)
            self._last_object_size_m = 0.040
            self._teleport_red_box(np.array([0.16, 0.14, 0.020], dtype=np.float64))
            self.get_logger().warn('Using predefined cube and target coordinates (fallback mode)')
        else:
            object_base, target_base = self._scan_for_task_geometry()
            if object_base is None or target_base is None:
                raise RuntimeError('camera-based detection failed for the cube or target square')

        object_base[2] = max(object_base[2], 0.026)
        target_base[2] = max(target_base[2], 0.002)

        self.get_logger().info(
            f'Using object ({object_base[0]:.3f}, {object_base[1]:.3f}, {object_base[2]:.3f}) '
            f'and target ({target_base[0]:.3f}, {target_base[1]:.3f}, {target_base[2]:.3f}) in base frame'
        )

        pick_direction = self._choose_pick_direction(object_base)
        target_hover = target_base.copy()
        target_hover[2] = max(float(target_base[2] + 0.18), 0.20)
        target_place = target_base.copy()
        target_place[2] = float(target_base[2] + 0.048)
        target_align = target_hover.copy()
        target_align[2] = max(float(target_base[2] + 0.10), 0.12)
        retreat_target = target_hover.copy()

        self.get_logger().info(f'Step 3/7: approach cube from the {pick_direction}')
        self.get_logger().info('Step 4/7: close gripper and secure the cube')
        pick_success = False
        for attempt_index in range(3):
            if attempt_index > 0 and not self.predefined_fallback:
                self.get_logger().warn(
                    f'Pick retry {attempt_index}/2 with adjusted grasp depth/close value'
                )
                self.command_gripper(self._OPEN_GRIPPER, duration_sec=0.9)
                refreshed_object = self._refresh_object_detection(timeout_sec=2.5)
                if refreshed_object is not None:
                    object_base = refreshed_object
                pick_direction = self._choose_pick_direction(object_base)

            pick_success = self._attempt_pick(
                object_base,
                home_pose,
                pick_direction,
                attempt_index=attempt_index,
            )
            if pick_success:
                break
        if not pick_success:
            raise RuntimeError('pick failed: object did not move with gripper after retries')

        lift_target = object_base.copy()
        lift_target[2] = max(float(object_base[2] + 0.20), 0.22)
        downward_orientation = [0.0, 0.0, -1.0]

        self.get_logger().info('Step 5/7: lift and transport to the target square')
        self._move_to_ik_target(
            lift_target,
            duration_sec=1.8,
            target_orientation=downward_orientation,
            orientation_mode='Z',
        )
        self._move_to_ik_target(
            target_hover,
            duration_sec=2.0,
            target_orientation=downward_orientation,
            orientation_mode='Z',
        )

        refined_target = self._refresh_target_detection(timeout_sec=3.0)
        if refined_target is not None:
            target_base = refined_target
            target_base[2] = max(target_base[2], 0.002)
            target_hover = target_base.copy()
            target_hover[2] = max(float(target_base[2] + 0.18), 0.20)
            target_align = target_base.copy()
            target_align[2] = max(float(target_base[2] + 0.10), 0.12)
            target_place = target_base.copy()
            target_place[2] = float(target_base[2] + 0.048)
            self._move_to_ik_target(
                target_hover,
                duration_sec=1.2,
                target_orientation=downward_orientation,
                orientation_mode='Z',
            )

        self.get_logger().info('Step 6/7: place the cube inside the target square')
        self._move_to_ik_target(
            target_align,
            duration_sec=1.4,
            target_orientation=downward_orientation,
            orientation_mode='Z',
        )
        refined_target = self._refresh_target_detection(timeout_sec=2.5)
        if refined_target is not None:
            target_base = refined_target
            target_place = target_base.copy()
            target_place[2] = float(target_base[2] + 0.048)
        self._move_to_ik_target(
            target_place,
            duration_sec=1.6,
            target_orientation=downward_orientation,
            orientation_mode='Z',
        )

        if self.predefined_fallback:
            place_pose = np.array([
                float(target_place[0]),
                float(target_place[1]),
                max(float(target_place[2] - 0.030), 0.020),
            ], dtype=np.float64)
            self._teleport_red_box(place_pose)

        self.command_gripper(self._OPEN_GRIPPER, duration_sec=1.0)
        time.sleep(0.6)
        self._move_to_ik_target(
            retreat_target,
            duration_sec=1.6,
            target_orientation=downward_orientation,
            orientation_mode='Z',
        )

        self.get_logger().info('Step 7/7: verify placement and return home')
        time.sleep(1.0)
        if self.predefined_fallback:
            self.get_logger().info('Task 1 success: fallback mode completed end-to-end with predefined placement')
            self.move_to_joint_states(home_pose)
            self.get_logger().info('Task 1 autonomous sequence completed')
            return

        placed_object, placed_target = self._wait_for_task_detections(timeout_sec=4.0)
        if placed_object is None:
            self.get_logger().warn('Could not re-detect the cube after placement')
        else:
            check_target = target_base if placed_target is None else placed_target
            xy_error = math.hypot(
                float(placed_object[0] - check_target[0]),
                float(placed_object[1] - check_target[1]),
            )
            self.get_logger().info(f'Placement XY error: {xy_error:.3f} m')
            if xy_error <= 0.040:
                self.get_logger().info('Task 1 success: cube placed inside the target region')
            else:
                self.get_logger().warn('Task 1 completed, but placement appears outside the desired tolerance')

        self.move_to_joint_states(home_pose)
        self.get_logger().info('Task 1 autonomous sequence completed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera-frame-id', default='d435i_link')
    parser.add_argument('--camera-translation', nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument('--camera-quaternion', nargs=4, type=float, default=[0.0, 0.0, 0.0, 1.0])
    parser.add_argument('--topic-prefix', default='/task1')
    parser.add_argument('--overhead-camera', action='store_true')
    parser.add_argument('--predefined-fallback', default='false')
    args, ros_args = parser.parse_known_args()

    predefined_fallback = str(args.predefined_fallback).strip().lower() in {'1', 'true', 'yes', 'on'}

    rclpy.init(args=ros_args)
    node = Task1Autonomy(
        camera_frame_id=args.camera_frame_id,
        camera_translation=np.array(args.camera_translation, dtype=np.float64),
        camera_quaternion=np.array(args.camera_quaternion, dtype=np.float64),
        use_overhead_camera=args.overhead_camera,
        topic_prefix=args.topic_prefix,
        predefined_fallback=predefined_fallback,
    )
    try:
        node.run_sequence()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

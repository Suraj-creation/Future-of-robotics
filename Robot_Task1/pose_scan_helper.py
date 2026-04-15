import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectoryPoint


POSES = [
    [0.0, -1.2, 1.3, -1.2, 0.0],
]


class PoseTester(Node):
    def __init__(self):
        super().__init__('pose_tester')
        self.arm = ActionClient(self, FollowJointTrajectory, 'arm_controller/follow_joint_trajectory')
        self.image = None
        self.create_subscription(Image, '/d435i/image', self.on_img, 10)

    def on_img(self, msg):
        self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3)).copy()

    def move(self, positions):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [
            'shoulder_pan',
            'shoulder_lift',
            'elbow_flex',
            'wrist_flex',
            'wrist_roll',
        ]
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start.sec = 2
        goal.trajectory.points = [pt]

        if not self.arm.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('arm action server unavailable')

        fut = self.arm.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        goal_handle = fut.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError('goal rejected')
        fut2 = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, fut2)


def main():
    rclpy.init()
    node = PoseTester()
    outdir = Path('/tmp/task1_scan_poses')
    outdir.mkdir(exist_ok=True)

    for index, pose in enumerate(POSES):
        node.move(pose)
        end_time = time.time() + 2.0
        while time.time() < end_time:
            rclpy.spin_once(node, timeout_sec=0.2)

        if node.image is None:
            print(index, 'NO_IMAGE')
            continue

        output = cv2.cvtColor(node.image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(outdir / f'pose_{index}.png'), output)
        print(index, pose, node.image.mean(axis=(0, 1)).tolist())

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

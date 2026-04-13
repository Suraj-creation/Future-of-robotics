import select
import sys
import termios
import tty
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

KEY_BINDINGS: Dict[str, Tuple[str, float]] = {
    "q": ("shoulder_pan_joint", +1.0),
    "a": ("shoulder_pan_joint", -1.0),
    "w": ("shoulder_lift_joint", +1.0),
    "s": ("shoulder_lift_joint", -1.0),
    "e": ("elbow_joint", +1.0),
    "d": ("elbow_joint", -1.0),
    "r": ("wrist_1_joint", +1.0),
    "f": ("wrist_1_joint", -1.0),
    "t": ("wrist_2_joint", +1.0),
    "g": ("wrist_2_joint", -1.0),
    "y": ("wrist_3_joint", +1.0),
    "h": ("wrist_3_joint", -1.0),
}

HOME_POSE = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]

HELP = """
UR5 Keyboard Teleop
-------------------
q/a : shoulder_pan +/-
w/s : shoulder_lift +/-
e/d : elbow +/-
r/f : wrist_1 +/-
t/g : wrist_2 +/-
y/h : wrist_3 +/-
z   : go to home pose
?   : print help
x   : quit

Tip: focus this terminal and press keys repeatedly to jog joints.
"""


class KeyboardTeleop(Node):
    def __init__(self) -> None:
        super().__init__("ur5_keyboard_teleop")

        self.step_rad = float(self.declare_parameter("step_rad", 0.08).value)
        self.move_time_sec = float(self.declare_parameter("move_time_sec", 0.35).value)

        self.current_positions: Dict[str, float] = {joint: 0.0 for joint in ARM_JOINTS}
        self.have_joint_state = False

        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_cb,
            20,
        )
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            "/ur5_arm_controller/joint_trajectory",
            10,
        )

        self.get_logger().info(
            "Keyboard teleop ready. Waiting for /joint_states before sending commands."
        )

    def joint_state_cb(self, msg: JointState) -> None:
        index_by_name = {name: idx for idx, name in enumerate(msg.name)}
        updated = False

        for joint in ARM_JOINTS:
            idx = index_by_name.get(joint)
            if idx is None or idx >= len(msg.position):
                continue
            self.current_positions[joint] = float(msg.position[idx])
            updated = True

        if updated:
            self.have_joint_state = True

    def publish_joint_target(self, target_positions: List[float]) -> None:
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS

        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.time_from_start = Duration(seconds=self.move_time_sec).to_msg()

        traj.points = [point]
        self.traj_pub.publish(traj)

    def jog_joint(self, joint_name: str, direction: float) -> None:
        if not self.have_joint_state:
            self.get_logger().warn("No /joint_states yet. Wait for simulation to finish starting.")
            return

        target = [self.current_positions[j] for j in ARM_JOINTS]
        idx = ARM_JOINTS.index(joint_name)
        target[idx] += direction * self.step_rad
        self.publish_joint_target(target)

    def go_home(self) -> None:
        if not self.have_joint_state:
            self.get_logger().warn("No /joint_states yet. Wait for simulation to finish starting.")
            return
        self.publish_joint_target(HOME_POSE)


def read_key(timeout_sec: float = 0.1) -> Optional[str]:
    readable, _, _ = select.select([sys.stdin], [], [], timeout_sec)
    if not readable:
        return None
    return sys.stdin.read(1)


def main(args=None) -> None:
    if not sys.stdin.isatty():
        print("keyboard_teleop requires an interactive TTY terminal.")
        return

    rclpy.init(args=args)
    node = KeyboardTeleop()

    old_settings = termios.tcgetattr(sys.stdin)
    print(HELP)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = read_key(0.1)

            if key is None:
                continue

            if key == "x" or key == "\x03":
                break

            if key == "z":
                node.go_home()
                continue

            if key == "?":
                print(HELP)
                continue

            binding = KEY_BINDINGS.get(key.lower())
            if binding is None:
                continue

            joint_name, direction = binding
            node.jog_joint(joint_name, direction)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mujoco_share = get_package_share_directory('so101_mujoco')
    mujoco_prefix = get_package_prefix('so101_mujoco')
    description_share = get_package_share_directory('so101_description')

    task1_scene = os.path.join(mujoco_share, 'mujoco', 'task1_pick_place_scene.xml')
    mujoco_bridge_script = os.path.join(mujoco_prefix, 'lib', 'so101_mujoco', 'so101_mujoco_bridge.py')
    task1_autonomy_script = os.path.join(mujoco_prefix, 'lib', 'so101_mujoco', 'task1_autonomy.py')
    urdf_path = os.path.join(description_share, 'urdf', 'so101.urdf')

    with open(urdf_path, 'r', encoding='utf-8') as urdf_file:
        robot_description = urdf_file.read()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='task1_robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': False}],
        remappings=[('joint_states', [LaunchConfiguration('task_namespace'), '/joint_states'])],
    )

    mujoco_bridge_with_viewer = ExecuteProcess(
        cmd=[
            '/usr/bin/python3', mujoco_bridge_script,
            '--model', LaunchConfiguration('mujoco_scene'),
            '--startup-pose', LaunchConfiguration('startup_pose'),
            '--camera-name', LaunchConfiguration('camera_name'),
            '--camera-frame-id', LaunchConfiguration('camera_frame_id'),
            '--camera-fovy-deg', LaunchConfiguration('camera_fovy_deg'),
            '--topic-prefix', LaunchConfiguration('task_namespace'),
            '--disable-camera', LaunchConfiguration('disable_camera'),
        ],
        condition=IfCondition(LaunchConfiguration('show_viewer')),
        output='screen',
    )

    mujoco_bridge_headless = ExecuteProcess(
        cmd=[
            '/usr/bin/python3', mujoco_bridge_script,
            '--model', LaunchConfiguration('mujoco_scene'),
            '--startup-pose', LaunchConfiguration('startup_pose'),
            '--camera-name', LaunchConfiguration('camera_name'),
            '--camera-frame-id', LaunchConfiguration('camera_frame_id'),
            '--camera-fovy-deg', LaunchConfiguration('camera_fovy_deg'),
            '--topic-prefix', LaunchConfiguration('task_namespace'),
            '--disable-camera', LaunchConfiguration('disable_camera'),
            '--no-viewer',
        ],
        condition=UnlessCondition(LaunchConfiguration('show_viewer')),
        output='screen',
    )

    autonomous_task = ExecuteProcess(
        cmd=[
            '/usr/bin/python3', task1_autonomy_script,
            '--camera-frame-id', LaunchConfiguration('camera_frame_id'),
            '--camera-translation', '0.21', '0.10', '0.65',
            '--camera-quaternion', '1.0', '0.0', '0.0', '0.0',
            '--topic-prefix', LaunchConfiguration('task_namespace'),
            '--overhead-camera',
            '--predefined-fallback', LaunchConfiguration('predefined_fallback'),
        ],
        condition=IfCondition(LaunchConfiguration('autonomous')),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('mujoco_scene', default_value=task1_scene),
        DeclareLaunchArgument('startup_pose', default_value='home', description='Initial arm pose: home or upright'),
        DeclareLaunchArgument('camera_name', default_value='task1_overhead', description='MuJoCo camera used for Task 1 perception'),
        DeclareLaunchArgument('task_namespace', default_value='/task1', description='Namespace/prefix for Task 1 topics and actions'),
        DeclareLaunchArgument('camera_frame_id', default_value='task1_overhead_camera', description='Frame id stamped on published camera topics'),
        DeclareLaunchArgument('camera_fovy_deg', default_value='45.0', description='Vertical field of view for the selected Task 1 camera'),
        DeclareLaunchArgument('disable_camera', default_value='false', description='Disable bridge camera rendering/publishing for headless fallback runs'),
        DeclareLaunchArgument('predefined_fallback', default_value='false', description='Use predefined fallback to complete task even if grasp perception/contacts fail'),
        DeclareLaunchArgument('show_viewer', default_value='false', description='Launch MuJoCo GUI viewer (true|false)'),
        DeclareLaunchArgument('autonomous', default_value='true', description='Run the autonomous Task 1 pick-and-place sequence (true|false)'),
        mujoco_bridge_with_viewer,
        mujoco_bridge_headless,
        robot_state_publisher,
        TimerAction(period=8.0, actions=[autonomous_task]),
    ])

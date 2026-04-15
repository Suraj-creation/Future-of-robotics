import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    mujoco_share = get_package_share_directory('so101_mujoco')
    mujoco_prefix = get_package_prefix('so101_mujoco')
    unified_share = get_package_share_directory('so101_unified_bringup')

    task3_scene = os.path.join(mujoco_share, 'mujoco', 'task3_pouring_scene.xml')
    mujoco_bridge_script = os.path.join(mujoco_prefix, 'lib', 'so101_mujoco', 'so101_mujoco_bridge.py')
    task3_autonomy_script = os.path.join(mujoco_prefix, 'lib', 'so101_mujoco', 'task3_autonomy.py')
    moveit_rviz_config = os.path.join(unified_share, 'config', 'moveit.rviz')

    moveit_config = MoveItConfigsBuilder(
        'so101_new_calib', package_name='so101_moveit_config'
    ).to_moveit_configs()

    task3_ros_remaps = [
        '--ros-args',
        '-r', '/d435i/image:=/task3/d435i/image',
        '-r', '/d435i/depth_image:=/task3/d435i/depth_image',
        '-r', '/d435i/camera_info:=/task3/d435i/camera_info',
        '-r', '/d435i/points:=/task3/d435i/points',
        '-r', '/arm_controller/follow_joint_trajectory:=/task3/arm_controller/follow_joint_trajectory',
        '-r', '/gripper_controller/follow_joint_trajectory:=/task3/gripper_controller/follow_joint_trajectory',
    ]

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[moveit_config.robot_description, {'use_sim_time': False}],
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict(), {'use_sim_time': False}],
        arguments=['--ros-args', '--log-level', 'warn'],
        condition=IfCondition(LaunchConfiguration('use_moveit')),
    )

    moveit_server = Node(
        package='so101_unified_bringup',
        executable='moveit_server',
        name='moveit_server',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            {
                'use_sim_time': False,
                'move_group_name': LaunchConfiguration('move_group_name'),
                'collision_object_frame': LaunchConfiguration('collision_object_frame'),
                'base_frame_id': LaunchConfiguration('base_frame_id'),
                'end_effector_frame_id': LaunchConfiguration('end_effector_frame_id'),
                'wrist_roll_joint_name': LaunchConfiguration('wrist_roll_joint_name'),
            },
        ],
        condition=IfCondition(LaunchConfiguration('use_moveit')),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', moveit_rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {'use_sim_time': False},
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    mujoco_bridge_with_viewer = ExecuteProcess(
        cmd=[
            '/usr/bin/python3', mujoco_bridge_script,
            '--model', LaunchConfiguration('mujoco_scene'),
            '--startup-pose', LaunchConfiguration('startup_pose'),
            *task3_ros_remaps,
        ],
        condition=IfCondition(LaunchConfiguration('show_viewer')),
        output='screen',
    )

    mujoco_bridge_headless = ExecuteProcess(
        cmd=[
            '/usr/bin/python3', mujoco_bridge_script,
            '--model', LaunchConfiguration('mujoco_scene'),
            '--startup-pose', LaunchConfiguration('startup_pose'),
            '--no-viewer',
            *task3_ros_remaps,
        ],
        condition=UnlessCondition(LaunchConfiguration('show_viewer')),
        output='screen',
    )

    autonomous_task = ExecuteProcess(
        cmd=['/usr/bin/python3', task3_autonomy_script, *task3_ros_remaps],
        condition=IfCondition(LaunchConfiguration('autonomous')),
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('mujoco_scene', default_value=task3_scene),
        DeclareLaunchArgument('startup_pose', default_value='home', description='Initial arm pose: home or upright'),
        DeclareLaunchArgument('show_viewer', default_value='false', description='Launch MuJoCo GUI viewer (true|false)'),
        DeclareLaunchArgument('autonomous', default_value='true', description='Run the autonomous task-3 sequence (true|false)'),
        DeclareLaunchArgument('rviz', default_value='false', description='Launch RViz with MoveIt config (true|false)'),
        DeclareLaunchArgument('use_moveit', default_value='false', description='Launch MoveIt planning stack for debugging (true|false)'),
        DeclareLaunchArgument('move_group_name', default_value='arm'),
        DeclareLaunchArgument('collision_object_frame', default_value='world'),
        DeclareLaunchArgument('base_frame_id', default_value='base_link'),
        DeclareLaunchArgument('end_effector_frame_id', default_value='gripper_link'),
        DeclareLaunchArgument('wrist_roll_joint_name', default_value='wrist_roll'),
        mujoco_bridge_with_viewer,
        mujoco_bridge_headless,
        robot_state_publisher,
        TimerAction(period=2.0, actions=[move_group]),
        TimerAction(period=5.0, actions=[moveit_server]),
        TimerAction(period=20.0, actions=[autonomous_task]),
        TimerAction(period=7.0, actions=[rviz]),
    ])
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

def generate_launch_description():
    ur5_description = get_package_share_directory("ur5_description")

    world_arg = DeclareLaunchArgument(
        name="world",
        default_value=os.path.join(ur5_description, "worlds", "gazebo_world.sdf"),
        description="Absolute path to Gazebo world file"
    )

    headless_arg = DeclareLaunchArgument(
        name="headless",
        default_value="false",
        description="Run Gazebo server only (-s), disables GUI for better performance"
    )

    verbosity_arg = DeclareLaunchArgument(
        name="verbosity",
        default_value="2",
        description="Gazebo log verbosity"
    )
    
    model_arg = DeclareLaunchArgument(
        name="model", 
        default_value=os.path.join(ur5_description, "urdf", "ur5_robot.urdf.xacro"),
        description="Absolute path to robot urdf file"
    )

    headless_flag = PythonExpression([
        "' -s' if '",
        LaunchConfiguration("headless"),
        "'.lower() in ['true', '1', 'yes'] else ''"
    ])
    
    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[str(Path(ur5_description).parent.resolve())]
    )
    
    ros_distro = os.environ["ROS_DISTRO"]
    is_ignition = "True" if ros_distro == "humble" else "False"
    
    robot_description = ParameterValue(
        Command(["xacro ", LaunchConfiguration("model"), " is_ignition:=", is_ignition]),
        value_type=str
    )
    
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True}]
    )
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch'), 
            '/gz_sim.launch.py'
        ]),
        launch_arguments=[
            ('gz_args', [
                LaunchConfiguration("world"),
                ' -v ',
                LaunchConfiguration("verbosity"),
                ' -r',
                headless_flag,
                ' --physics-engine gz-physics-bullet-featherstone-plugin'
            ])
        ]
    )
    
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description", "-name", "bumperbot"],
    )
    
    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            'config_file': os.path.join(ur5_description, 'config', 'ros2_gz_bridge.yaml'),
        }],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        headless_arg,
        verbosity_arg,
        model_arg,
        gazebo_resource_path,
        robot_state_publisher_node,
        gazebo,
        gz_spawn_entity,
        gz_ros2_bridge,
    ])

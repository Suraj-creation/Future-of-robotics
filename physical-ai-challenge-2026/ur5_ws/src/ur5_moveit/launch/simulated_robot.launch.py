import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo in server-only mode for low-lag simulation"
    )

    verbosity_arg = DeclareLaunchArgument(
        "verbosity",
        default_value="2",
        description="Gazebo verbosity level"
    )

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz (set false for lower CPU/GPU usage)"
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ur5_description"),
                "launch",
                "gazebo.launch.py"
            )
        ),
        launch_arguments={
            "headless": LaunchConfiguration("headless"),
            "verbosity": LaunchConfiguration("verbosity"),
        }.items(),
    )

    controller = IncludeLaunchDescription(
        os.path.join(
            get_package_share_directory("ur5_controller"),
            "launch",
            "controller.launch.py"
        )
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ur5_moveit"),
                "launch",
                "moveit.launch.py"
            )
        ),
        launch_arguments={
            "use_rviz": LaunchConfiguration("use_rviz"),
        }.items(),
    )

    return LaunchDescription([
        headless_arg,
        verbosity_arg,
        use_rviz_arg,
        gazebo,
        controller,
        moveit_launch
    ])
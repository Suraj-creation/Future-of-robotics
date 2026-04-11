FROM osrf/ros:jazzy-desktop

# Install ROS 2 dependencies for UR5 simulation
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-jazzy-moveit \
    ros-jazzy-ros-gz \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-controller-manager \
    ros-jazzy-xacro \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
ENV ROS_WS=/ur5_ws
WORKDIR $ROS_WS

# Copy source code
COPY ur5_ws/src ./src/

# Install any remaining rosdep dependencies
RUN apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*

# Build the workspace
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install"

# Source ROS and workspace in every new shell
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /ur5_ws/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]

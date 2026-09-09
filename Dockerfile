# 예시 (JetPack 버전에 맞게 l4t-r32.7.1 부분을 수정해야 할 수 있음)
FROM dustynv/ros:foxy-ros-base-l4t-r32.7.1
RUN apt-key del F42ED6FBAB17C654 || true && \
    apt-get update || true && \
    apt-get install -y curl && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -
    
# 1. Install basic utilities and build tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-tk \
    git \
    vim \
    tmux \
    wget \
    libeigen3-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# 2. Install SLAM and Navigation2 packages
RUN apt-get update && apt-get install -y \
    ros-foxy-slam-toolbox \
    ros-foxy-navigation2 \
    ros-foxy-nav2-bringup \
    ros-foxy-robot-localization \
    && rm -rf /var/lib/apt/lists/*

# 3. Install F1TENTH and simulation dependencies
RUN apt-get update && apt-get install -y \
    ros-foxy-ackermann-msgs \
    ros-foxy-joy \
    ros-foxy-teleop-twist-joy \
    ros-foxy-teleop-twist-keyboard \
    ros-foxy-urg-node \
    ros-foxy-cartographer-ros \
    ros-foxy-serial-driver \
    ros-foxy-rosbridge-server \
    ros-foxy-joint-state-publisher \
    ros-foxy-gazebo-ros \
    ros-foxy-test-msgs \
    ros-foxy-control-msgs \
    ros-foxy-tf2-ros \
    ros-foxy-tf2-geometry-msgs \
    ros-foxy-pcl-ros \
    ros-foxy-xacro \
    ros-foxy-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python packages with pinned versions
RUN python3 -m pip install --no-cache-dir \
    "typing-extensions==4.13.2" \
    "numpy==1.23.5" \
    "llvmlite==0.39.1" \
    "numba==0.56.4" \
    "scipy==1.10.1" \
    "pandas==2.0.3" \
    pyclothoids \
    rosbags \
    opencv-contrib-python \
    matplotlib \
    trajectory_planning_helpers

# 5. Install PyTorch for CUDA 11.8 <- Check your CUDA version
RUN python3 -m pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118

# 7. Configure shell environment
RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc \
    && echo "alias cb='cd /root/f1tenth_ws && colcon build --symlink-install && source /root/f1tenth_ws/install/setup.bash'" >> /root/.bashrc \
    && echo "alias cs='source /root/f1tenth_ws/install/setup.bash'" >> /root/.bashrc \
    && echo "alias extract='python3 /root/f1tenth_ws/src/f1tenth_lattice_ros2/tools/extract_bag_csv.py'" >> /root/.bashrc \
    && echo "export ROS_DOMAIN_ID=1" >> /root/.bashrc

WORKDIR /root/f1tenth_ws

CMD ["bash"]

FROM osrf/ros:foxy-desktop

# 1. 기본 유틸리티 및 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    git \
    vim \
    tmux \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. SLAM 및 Navigation2 관련 패키지 설치
RUN apt-get update && apt-get install -y \
    ros-foxy-slam-toolbox \
    ros-foxy-navigation2 \
    ros-foxy-nav2-bringup \
    ros-foxy-robot-localization \
    && rm -rf /var/lib/apt/lists/*

# 3. F1TENTH 전용 의존성 (Ackermann, 센서 등)
# 3. F1TENTH 및 시뮬레이션 핵심 의존성 추가
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

RUN apt-get update && apt-get install -y \
    python3-numpy \
    python3-numba \
    python3-scipy \
    python3-pandas \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir pyclothoids

RUN sudo rosdep update --include-eol-distros || true

# 5. 작업 환경 설정 (편의성)
RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc \
    && echo "alias cb='cd /root/f1tenth_ws && colcon build --symlink-install && source /root/f1tenth_ws/install/setup.bash'" >> /root/.bashrc \
    && echo "alias cs='source /root/f1tenth_ws/install/setup.bash'" >> /root/.bashrc \
    && echo "export ROS_DOMAIN_ID=1" >> /root/.bashrc

WORKDIR /root/f1tenth_ws

CMD ["bash"]

#!/usr/bin/env python3
"""
gazebo_replay.launch.py

data/0529 같이 f1tenth_lattice_gazebo.launch.py로 녹화한 bag을
Gazebo에서 시각적으로 재생하는 런치 파일.

Gazebo 물리를 끈 상태에서 bag의 /car1/odom, /car2/odom을
bag2gazebo_node가 받아 set_entity_state로 차량을 텔레포트함.

사용법:
    ros2 launch f1tenth_lattice_ros2 gazebo_replay.launch.py \\
        bag:=<path_to_db3_file>

예시 (staging bag):
    ros2 launch f1tenth_lattice_ros2 gazebo_replay.launch.py \\
        bag:=/root/f1tenth_ws/data/0529/staging/ep0000_20260528_133011/ep0000_20260528_133011_0.db3

재생 속도 조절:
    ros2 launch f1tenth_lattice_ros2 gazebo_replay.launch.py \\
        bag:=<path> rate:=0.5

맵 지정 (bag 경로에 Simple_vNN 같은 맵 이름이 있으면 자동 추론, 없으면 simple):
    ros2 launch f1tenth_lattice_ros2 gazebo_replay.launch.py \\
        bag:=<path> gazebo:=true map:=Simple_v02

RViz만 실행 (Gazebo 생략):
    ros2 launch f1tenth_lattice_ros2 gazebo_replay.launch.py \\
        bag:=<path> gazebo:=false
"""

import math
import os
import re
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    lattice_pkg = get_package_share_directory('f1tenth_lattice_ros2')
    description_pkg = get_package_share_directory('racecar_description')

    _config_path = os.path.join(lattice_pkg, 'config', 'lattice_config.yaml')
    with open(_config_path) as _f:
        _cfg = yaml.safe_load(_f)
    _s1 = _cfg.get('car1', {}).get('spawn', {})
    _s2 = _cfg.get('car2', {}).get('spawn', {})

    def deg2rad(d): return str(float(d) * math.pi / 180.0)

    # Launch arguments
    bag_arg = DeclareLaunchArgument(
        'bag',
        default_value=(
            '/root/f1tenth_ws/data/0529/staging/'
            'ep0000_20260528_133011/'
            'ep0000_20260528_133011_0.db3'
        ),
        description='.db3 bag 파일 경로'
    )
    rate_arg = DeclareLaunchArgument(
        'rate',
        default_value='1.0',
        description='재생 속도 배율 (0.5 = 절반 속도, 2.0 = 2배속)'
    )
    gazebo_arg = DeclareLaunchArgument(
        'gazebo',
        default_value='false',
        description='Gazebo 시뮬레이터 실행 여부 (false면 RViz로만 재생)'
    )
    gazebo = LaunchConfiguration('gazebo')
    seed_joint_states_arg = DeclareLaunchArgument(
        'seed_joint_states',
        default_value='false',
        description='bag에 /carN/joint_states가 없을 때만 true. '
                    'true면 joint_state_publisher가 0 위치를 1 Hz로 내보내 바퀴 TF를 만든다.'
    )
    seed_joint_states = LaunchConfiguration('seed_joint_states')

    # 이전 Gazebo 프로세스 정리 (gazebo:=false면 스킵)
    kill_gazebo = ExecuteProcess(
        condition=IfCondition(gazebo),
        cmd=['bash', '-c',
             'pkill -9 -f gzserver; pkill -9 -f gzclient; '
             'sleep 1.5; echo "[replay] Gazebo cleaned up"'],
        output='screen'
    )

    map_arg = DeclareLaunchArgument(
        'map',
        default_value='',
        description='Gazebo world 맵 이름 (예: Simple_v02). 비우면 bag 경로에서 '
                    'maps/ 아래 맵 이름을 찾아 쓰고, 못 찾으면 simple'
    )

    def resolve_map(context):
        map_name = LaunchConfiguration('map').perform(context)
        if map_name:
            return map_name
        # bag은 녹화한 맵 좌표계 그대로이므로 world가 다르면 차가 트랙 밖에서 달리는 것처럼 보인다.
        # data/simple_variants_h2h/Simple_v02/clean/... 처럼 경로에 맵 이름이 들어있으면 그걸 쓴다.
        known = set(os.listdir(os.path.join(lattice_pkg, 'maps')))
        bag_path = os.path.abspath(LaunchConfiguration('bag').perform(context))
        for part in reversed(bag_path.split(os.sep)):
            if part in known:
                return part
            m = re.match(r'(Simple_v\d+)', part)
            if m and m.group(1) in known:
                return m.group(1)
        return 'simple'

    def spawn_cars(context):
        map_name = resolve_map(context)
        print(f'[replay] Gazebo world map: {map_name}')

        # Gazebo + car1 스폰
        spawn_car1 = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(description_pkg, 'launch', 'spawn_car.launch.py')
            ),
            launch_arguments={
                'namespace': 'car1',
                'map': map_name,
                'x': str(_s1.get('x', 6.4)),
                'y': str(_s1.get('y', 16.0)),
                'yaw': deg2rad(_s1.get('yaw_deg', -90.0)),
                'color': 'blue',
            }.items()
        )

        # car2 스폰 (Gazebo는 이미 실행 중)
        spawn_car2 = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(description_pkg, 'launch', 'spawn_car.launch.py')
            ),
            launch_arguments={
                'namespace': 'car2',
                'map': map_name,
                'x': str(_s2.get('x', 6.4)),
                'y': str(_s2.get('y', 12.0)),
                'yaw': deg2rad(_s2.get('yaw_deg', -90.0)),
                'launch_gazebo': 'false',
                'color': 'orange',
            }.items()
        )
        return [spawn_car1, spawn_car2]

    # gazebo:=false면 스킵
    spawn_cars_action = OpaqueFunction(function=spawn_cars, condition=IfCondition(gazebo))

    # gazebo:=false일 때는 spawn_car.launch.py(=robot_state_publisher+Gazebo+spawn_entity) 대신
    # state publisher들만 실행해서 RViz가 RobotModel/TF를 그릴 수 있게 함
    xacro_file = os.path.join(description_pkg, 'urdf', 'racecar.xacro')

    def state_publishers(namespace, color, visualize_lidar):
        robot_description_content = ParameterValue(
            Command(['xacro ', xacro_file,
                     ' namespace:=', namespace,
                     ' color:=', color,
                     ' visualize_lidar:=', visualize_lidar]),
            value_type=str
        )
        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            namespace=namespace,
            output='screen',
            condition=UnlessCondition(gazebo),
            parameters=[{
                'robot_description': robot_description_content,
                'use_sim_time': True,
                'frame_prefix': [namespace, '/'],
            }]
        )

        # A RobotModel asks TF for every movable URDF link immediately, but the
        # bag does not start until several seconds later.  Bags that contain
        # /carN/joint_states need no seed — a 1 Hz zero-position publisher would
        # just fight the recorded values — so this is opt-in via
        # seed_joint_states:=true for bags recorded without joint_states.
        # use_sim_time must match robot_state_publisher; with wall-clock stamps
        # the wheel TFs are unusable once the bag's /clock takes over.
        joint_state_publisher = Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            namespace=namespace,
            output='screen',
            condition=IfCondition(PythonExpression(
                ['"', gazebo, '".lower() not in ("1", "true") and "',
                 seed_joint_states, '".lower() in ("1", "true")'])),
            parameters=[{
                'robot_description': robot_description_content,
                'rate': 1,
                'use_sim_time': True,
            }]
        )
        return robot_state_publisher, joint_state_publisher

    rsp_car1, jsp_car1 = state_publishers('car1', 'blue', 'true')
    rsp_car2, jsp_car2 = state_publishers('car2', 'orange', 'false')

    # bag의 odom을 받아 Gazebo 모델 위치를 업데이트하는 브리지 노드 (gazebo:=false면 스킵)
    bag2gazebo = Node(
        package='f1tenth_lattice_ros2',
        executable='bag2gazebo_node',
        name='bag2gazebo',
        output='screen',
        condition=IfCondition(gazebo),
        parameters=[{
            'cars': ['car1', 'car2'],
            'pause_on_start': True,   # 물리 엔진 정지 (시각 재생 전용)
        }]
    )

    # map ↔ odom static TF
    map_to_odom1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_car1',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'car1/odom']
    )
    map_to_odom2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_car2',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'car2/odom']
    )

    # RViz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(description_pkg, 'rviz', 'f1tenth_default.rviz')],
        output='screen'
    )

    # --- 순서 ---
    # 0s  : kill_gazebo
    # 1.5s: Gazebo + car1/car2 스폰, RViz, static TF
    # 5.0s: bag2gazebo_node (set_entity_state 서비스 대기)
    # 8.0s: bag play 시작

    delayed_spawn = TimerAction(
        period=1.5,
        actions=[
            spawn_cars_action,
            rsp_car1, rsp_car2, jsp_car1, jsp_car2,
            map_to_odom1, map_to_odom2, rviz_node,
        ]
    )

    delayed_bridge = TimerAction(
        period=5.0,
        actions=[bag2gazebo]
    )

    # bag play: odom/joint_states/scan/tf/clock + 시각화 토픽만 선택
    # (gazebo_msgs 같은 Gazebo 내부 토픽은 제외)
    delayed_bag_play = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play',
                    '-s', 'sqlite3',
                    LaunchConfiguration('bag'),
                    '--rate', LaunchConfiguration('rate'),
                    '--topics',
                    '/clock',
                    '/tf', '/tf_static',
                    '/car1/odom', '/car2/odom',
                    '/car1/joint_states', '/car2/joint_states',
                    '/car1/scan', '/car2/scan',
                    '/car1/best_traj_marker', '/car2/best_traj_marker',
                    '/car1/best_traj_marker_array', '/car2/best_traj_marker_array',
                    '/car1/raceline_marker', '/car2/raceline_marker',
                    '/direction_marker', '/direction_marker_array',
                    '/episode/control',
                ],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        bag_arg,
        rate_arg,
        gazebo_arg,
        seed_joint_states_arg,
        map_arg,
        kill_gazebo,
        delayed_spawn,
        delayed_bridge,
        delayed_bag_play,
    ])

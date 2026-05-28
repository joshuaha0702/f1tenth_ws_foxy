import os
import re
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, TimerAction, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    lattice_pkg = get_package_share_directory('f1tenth_lattice_ros2')
    description_pkg = get_package_share_directory('racecar_description')

    # Read spawn defaults from config at launch-description-generation time
    _config_path = os.path.join(lattice_pkg, 'config', 'lattice_config.yaml')
    with open(_config_path) as _f:
        _cfg = yaml.safe_load(_f)
    _s1 = _cfg.get('car1', {}).get('spawn', {})
    _s2 = _cfg.get('car2', {}).get('spawn', {})

    # 이전 가제보 좀비 프로세스를 자동으로 제거 (위치 변경이 안 되는 문제 방지)
    kill_gazebo = ExecuteProcess(
        cmd=['bash', '-c', 'pkill -9 -f gzserver; pkill -9 -f gzclient; sleep 1.5; echo "[launch] Cleaned up previous Gazebo processes"'],
        output='screen'
    )

    # Namespace argument (mirrors existing FGM launch)
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='car1',
        description='Robot namespace'
    )
    namespace = LaunchConfiguration('namespace')

    # Head-to-head mode argument
    head2head_arg = DeclareLaunchArgument(
        'head2head',
        default_value='false',
        description='car2도 함께 생성하여 head-to-head 주행 (true/false)'
    )
    head2head = LaunchConfiguration('head2head')

    # Headless mode argument
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Gazebo gzclient와 RViz2를 실행하지 않음 (RTF 향상)'
    )
    headless = LaunchConfiguration('headless')
    gui_value = PythonExpression(['"false" if "', headless, '".lower() == "true" else "true"'])

    # Spawn coordinates arguments (car1) — defaults from lattice_config.yaml
    x_arg = DeclareLaunchArgument('x', default_value=str(_s1.get('x', 6.4)), description='Car1 Spawn X position')
    y_arg = DeclareLaunchArgument('y', default_value=str(_s1.get('y', 16.0)), description='Car1 Spawn Y position')
    yaw_arg = DeclareLaunchArgument('yaw_deg', default_value=str(_s1.get('yaw_deg', -90.0)), description='Car1 Spawn Yaw angle (degrees)')
    spawn_x = LaunchConfiguration('x')
    spawn_y = LaunchConfiguration('y')
    spawn_yaw = PythonExpression(['str(float("', LaunchConfiguration('yaw_deg'), '") * 3.14159265358979 / 180.0)'])

    # Spawn coordinates arguments (car2) — defaults from lattice_config.yaml
    x2_arg = DeclareLaunchArgument('x2', default_value=str(_s2.get('x', 6.4)), description='Car2 Spawn X position')
    y2_arg = DeclareLaunchArgument('y2', default_value=str(_s2.get('y', 20.0)), description='Car2 Spawn Y position')
    yaw_deg2_arg = DeclareLaunchArgument('yaw_deg2', default_value=str(_s2.get('yaw_deg', -90.0)), description='Car2 Spawn Yaw angle (degrees)')
    spawn_x2 = LaunchConfiguration('x2')
    spawn_y2 = LaunchConfiguration('y2')
    spawn_yaw2 = PythonExpression(['str(float("', LaunchConfiguration('yaw_deg2'), '") * 3.14159265358979 / 180.0)'])

    # Spawn Gazebo + car1
    spawn_car_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(description_pkg, 'launch', 'spawn_car.launch.py')
        ),
        launch_arguments={
            'namespace': 'car1',
            'x': spawn_x, 'y': spawn_y, 'yaw': spawn_yaw,
            'gui': gui_value,
        }.items()
    )

    # Spawn car2 only (Gazebo already running)
    spawn_car2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(description_pkg, 'launch', 'spawn_car.launch.py')
        ),
        launch_arguments={
            'namespace': 'car2',
            'x': spawn_x2, 'y': spawn_y2, 'yaw': spawn_yaw2,
            'launch_gazebo': 'false',
            'color': 'orange',
            'visualize_lidar': 'false',
        }.items()
    )

    # Paths to lattice planner resources
    # raceline1.csv = center lane (3 lanes: inner=0, center=1, outer=2)
    config_path = os.path.join(lattice_pkg, 'config', 'lattice_config.yaml')
    map_path = os.path.join(lattice_pkg, 'maps', 'Simple_map')   # no extension
    raceline_path = os.path.join(lattice_pkg, 'maps', 'raceline1.csv')

    # Lattice planner node — car1
    lattice_node = Node(
        package='f1tenth_lattice_ros2',
        executable='lattice_planner_node',
        name='lattice_planner',
        namespace='car1',
        output='screen',
        parameters=[{
            'config_path': config_path,
            'raceline_path': raceline_path,
            'map_path': map_path,
            'max_speed': 3.0,
            'max_steering_angle': 0.4189,
            'opponent_namespace': 'car2',
        }]
    )

    # Ackermann to Twist Bridge — car1
    bridge_node = Node(
        package='f1tenth_fgm_ros2',
        executable='ackermann_to_twist.py',
        name='ackermann_to_twist',
        namespace='car1',
        output='screen'
    )

    # Lattice planner node — car2 (head2head only)
    lattice_node_car2 = Node(
        package='f1tenth_lattice_ros2',
        executable='lattice_planner_node',
        name='lattice_planner',
        namespace='car2',
        output='screen',
        parameters=[{
            'config_path': config_path,
            'raceline_path': raceline_path,
            'map_path': map_path,
            'max_speed': 3.0,
            'max_steering_angle': 0.4189,
            'opponent_namespace': 'car1',
        }]
    )

    # Ackermann to Twist Bridge — car2
    bridge_node_car2 = Node(
        package='f1tenth_fgm_ros2',
        executable='ackermann_to_twist.py',
        name='ackermann_to_twist',
        namespace='car2',
        output='screen'
    )

    # RViz2 with existing config (headless 시 비활성화)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(description_pkg, 'rviz', 'f1tenth_default.rviz')],
        output='screen',
        condition=UnlessCondition(headless)
    )

    # Static TFs — car1
    map_to_odom_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'car1/odom']
    )
    laser_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_tf_fix',
        arguments=['0', '0', '0', '0', '0', '0', 'car1/laser', 'laser']
    )

    # Static TFs — car2 (head2head only)
    map_to_odom_node_car2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_car2',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'car2/odom']
    )
    laser_tf_node_car2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_tf_fix_car2',
        arguments=['0', '0', '0', '0', '0', '0', 'car2/laser', 'laser']
    )

    # ROS Bag 녹화 인자
    record_arg = DeclareLaunchArgument(
        'record',
        default_value='false',
        description='ROS Bag 녹화 여부 (true/false)'
    )
    bag_output_arg = DeclareLaunchArgument(
        'bag_output',
        default_value='./bags/output',
        description='ROS Bag 저장 폴더 경로 (단일 에피소드 모드용)'
    )
    episodes_arg = DeclareLaunchArgument(
        'episodes',
        default_value='',
        description='다중 에피소드 yaml 경로. 지정하면 episode_manager가 녹화를 인계받음.'
    )

    # 단일 에피소드(기존) 녹화: record=true 이고 episodes가 비어있을 때만 동작
    # PythonExpression으로 "record true && episodes empty" 조건을 만듦
    single_bag_condition = PythonExpression([
        '"', LaunchConfiguration('record'), '".lower() == "true" and "',
        LaunchConfiguration('episodes'), '" == ""'
    ])

    record_bag = ExecuteProcess(
        condition=IfCondition(single_bag_condition),
        cmd=[
            'bash', '-c',
            'python3 -c "'
            'import rclpy\n'
            'from rclpy.node import Node\n'
            'from ackermann_msgs.msg import AckermannDriveStamped\n'
            'rclpy.init()\n'
            'node = Node(\\\"_bag_trigger\\\")\n'
            'done = [False]\n'
            'def cb(msg): done[0] = True\n'
            'node.create_subscription(AckermannDriveStamped, \\\"/$1/drive\\\", cb, 10)\n'
            'print(\\\"[bag] Waiting for first drive command...\\\", flush=True)\n'
            'while not done[0]: rclpy.spin_once(node, timeout_sec=0.1)\n'
            'node.destroy_node(); rclpy.shutdown()\n'
            '" && '
            'echo "[bag] Drive started — starting bag recording" && '
            'TIME_STAMP=$(date -d "+9 hours" +%Y%m%d_%H%M%S) && '
            'BAG_PATH="${2}_${TIME_STAMP}" && '
            'mkdir -p "$(dirname "$BAG_PATH")" && '
            'ros2 bag record -o "$BAG_PATH" -a',
            '--',
            namespace,
            LaunchConfiguration('bag_output'),
        ],
        output='screen'
    )

    car2_group = GroupAction(
        condition=IfCondition(head2head),
        actions=[
            spawn_car2_launch,
            lattice_node_car2,
            bridge_node_car2,
            map_to_odom_node_car2,
            laser_tf_node_car2,
        ]
    )

    # 다중 에피소드 매니저 (episodes 인자가 비어있지 않을 때만 실행)
    # 단일 bag 모드와 동일하게 record가 true일 때만 의미가 있음.
    episode_manager_condition = PythonExpression([
        '"', LaunchConfiguration('record'), '".lower() == "true" and "',
        LaunchConfiguration('episodes'), '" != ""'
    ])
    episode_manager_node = Node(
        package='f1tenth_lattice_ros2',
        executable='episode_manager_node',
        name='episode_manager',
        output='screen',
        condition=IfCondition(episode_manager_condition),
        parameters=[{
            'episodes_yaml': LaunchConfiguration('episodes'),
            'head2head': LaunchConfiguration('head2head'),
            'use_sim_time': True,
        }]
    )

    # 가제보 정리 후 1.5초 대기 후 나머지 노드 시작
    delayed_launch = TimerAction(
        period=1.5,
        actions=[
            spawn_car_launch,
            lattice_node,
            bridge_node,
            rviz_node,
            map_to_odom_node,
            laser_tf_node,
            car2_group,
            record_bag,
        ]
    )

    # 에피소드 매니저는 모든 노드가 자리잡은 뒤 시동 (Gazebo 서비스 가용 + planner 구독자 준비)
    delayed_manager = TimerAction(
        period=6.0,
        actions=[episode_manager_node],
    )

    return LaunchDescription([
        namespace_arg,
        head2head_arg,
        headless_arg,
        x_arg,
        y_arg,
        yaw_arg,
        x2_arg,
        y2_arg,
        yaw_deg2_arg,
        record_arg,
        bag_output_arg,
        episodes_arg,
        kill_gazebo,
        delayed_launch,
        delayed_manager,
    ])

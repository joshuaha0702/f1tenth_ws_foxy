import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'f1tenth_end2race_ros2'
    end2race_pkg = get_package_share_directory(pkg_name)
    description_pkg = get_package_share_directory('racecar_description')
    lattice_pkg = get_package_share_directory('f1tenth_lattice_ros2')

    kill_gazebo = ExecuteProcess(
        cmd=['bash', '-c', 'pkill -9 -f gzserver; pkill -9 -f gzclient; sleep 1.5'],
        output='screen'
    )

    namespace_arg = DeclareLaunchArgument('namespace', default_value='car1')
    namespace = LaunchConfiguration('namespace')

    # 초기 스폰 위치
    spawn_x = LaunchConfiguration('x', default='6.4')
    spawn_y = LaunchConfiguration('y', default='16.0')
    spawn_yaw = PythonExpression(['str(float("-90.0") * 3.14159265358 / 180.0)'])

    spawn_car_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description_pkg, 'launch', 'spawn_car.launch.py')),
        launch_arguments={'x': spawn_x, 'y': spawn_y, 'yaw': spawn_yaw}.items()
    )

    # car2 (lattice 플래너 기반 정속 주행) 활성화 여부
    car2_arg = DeclareLaunchArgument(
        'car2',
        default_value='true',
        description='Lattice 플래너로 정속 주행하는 car2를 함께 생성 (true/false)'
    )
    car2_enabled = LaunchConfiguration('car2')

    # car2 초기 스폰 위치
    spawn_x2 = LaunchConfiguration('x2', default='6.4')
    spawn_y2 = LaunchConfiguration('y2', default='12.0')
    spawn_yaw2 = PythonExpression(['str(float("-90.0") * 3.14159265358 / 180.0)'])

    spawn_car2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description_pkg, 'launch', 'spawn_car.launch.py')),
        launch_arguments={
            'namespace': 'car2',
            'x': spawn_x2, 'y': spawn_y2, 'yaw': spawn_yaw2,
            'launch_gazebo': 'false',
            'color': 'orange',
            'visualize_lidar': 'false',
        }.items(),
        condition=IfCondition(car2_enabled)
    )

    # Lattice 플래너 노드 — car2 (정속 주행, opponent tracking 없음)
    lattice_config_path = os.path.join(lattice_pkg, 'config', 'lattice_config.yaml')
    lattice_map_path = os.path.join(lattice_pkg, 'maps', 'Simple_map')
    lattice_raceline_path = os.path.join(lattice_pkg, 'maps', 'raceline1.csv')

    lattice_node_car2 = Node(
        package='f1tenth_lattice_ros2',
        executable='lattice_planner_node',
        name='lattice_planner',
        namespace='car2',
        output='screen',
        parameters=[{
            'config_path': lattice_config_path,
            'raceline_path': lattice_raceline_path,
            'map_path': lattice_map_path,
            'max_speed': 2.0,
            'max_steering_angle': 0.4189,
            'opponent_namespace': '',
        }],
        condition=IfCondition(car2_enabled)
    )

    # Ackermann to Twist Bridge — car2
    bridge_node_car2 = Node(
        package='f1tenth_fgm_ros2',
        executable='ackermann_to_twist.py',
        name='ackermann_to_twist',
        namespace='car2',
        output='screen',
        condition=IfCondition(car2_enabled)
    )

    # Static TFs — car2
    map_to_odom_node_car2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom_car2',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'car2/odom'],
        condition=IfCondition(car2_enabled)
    )
    laser_tf_node_car2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_tf_fix_car2',
        arguments=['0', '0', '0', '0', '0', '0', 'car2/laser', 'laser'],
        condition=IfCondition(car2_enabled)
    )

    config_path = os.path.join(end2race_pkg, 'config', 'end2race_config.yaml')

    # Agent 노드 설정
    agent_node = Node(
        package=pkg_name,
        executable='agent_node',
        name='end2race_agent',
        namespace=namespace,
        output='log',
        parameters=[config_path, {'use_sim_time': True}] # YAML 파일을 직접 리스트에 추가
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(description_pkg, 'rviz', 'f1tenth_default.rviz')],
    )

    map_to_odom_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', [namespace, '/odom']]
    )

    laser_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', [namespace, '/laser'], 'laser']
    )

    delayed_launch = TimerAction(
        period=2.0,
        actions=[
            spawn_car_launch, agent_node, rviz_node, map_to_odom_node, laser_tf_node,
            spawn_car2_launch, lattice_node_car2, bridge_node_car2,
            map_to_odom_node_car2, laser_tf_node_car2,
        ]
    )

    return LaunchDescription([namespace_arg, car2_arg, kill_gazebo, delayed_launch])

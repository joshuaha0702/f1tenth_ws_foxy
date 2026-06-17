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

    # ============================================================
    # Launch 인자 정의
    # ============================================================
    namespace_arg = DeclareLaunchArgument(
        'namespace', default_value='car1',
        description='car1 네임스페이스'
    )
    head2head_arg = DeclareLaunchArgument(
        'head2head', default_value='true',
        description='Lattice 플래너로 정속 주행하는 car2를 함께 생성 (true/false)'
    )
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(end2race_pkg, 'models', 'end2race.pth'),
        description='end2race 모델 가중치(.pth) 경로'
    )
    x_arg = DeclareLaunchArgument('x', default_value='6.4', description='car1 스폰 X')
    y_arg = DeclareLaunchArgument('y', default_value='16.0', description='car1 스폰 Y')
    x2_arg = DeclareLaunchArgument('x2', default_value='6.4', description='car2 스폰 X')
    y2_arg = DeclareLaunchArgument('y2', default_value='12.0', description='car2 스폰 Y')

    # LaunchConfiguration 핸들
    namespace = LaunchConfiguration('namespace')
    car2_enabled = LaunchConfiguration('head2head')
    model_path = LaunchConfiguration('model_path')
    spawn_x = LaunchConfiguration('x')
    spawn_y = LaunchConfiguration('y')
    spawn_x2 = LaunchConfiguration('x2')
    spawn_y2 = LaunchConfiguration('y2')
    spawn_yaw = PythonExpression(['str(float("-90.0") * 3.14159265358 / 180.0)'])
    spawn_yaw2 = PythonExpression(['str(float("-90.0") * 3.14159265358 / 180.0)'])

    all_launch_args = [namespace_arg, head2head_arg, model_path_arg, x_arg, y_arg, x2_arg, y2_arg]

    # ============================================================
    # 노드 / 액션 정의
    # ============================================================
    kill_gazebo = ExecuteProcess(
        cmd=['bash', '-c', 'pkill -9 -f gzserver; pkill -9 -f gzclient; sleep 1.5'],
        output='screen'
    )

    spawn_car_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description_pkg, 'launch', 'spawn_car.launch.py')),
        launch_arguments={'x': spawn_x, 'y': spawn_y, 'yaw': spawn_yaw}.items()
    )

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

    # Agent 노드 (car1) — config 뒤 dict가 우선이라 model_path가 config 값을 덮어씀
    config_path = os.path.join(end2race_pkg, 'config', 'end2race_config.yaml')
    agent_node = Node(
        package=pkg_name,
        executable='agent_node',
        name='end2race_agent',
        namespace=namespace,
        output='log',
        parameters=[config_path, {'use_sim_time': False, 'model_path': model_path}]
    )

    # Ackermann to Twist Bridge — car1
    # agent_node가 publish하는 /car1/drive(AckermannDriveStamped)를
    # Gazebo가 구독하는 /car1/drive_twist(Twist)로 변환
    bridge_node = Node(
        package='f1tenth_fgm_ros2',
        executable='ackermann_to_twist.py',
        name='ackermann_to_twist',
        namespace=namespace,
        output='screen'
    )

    # rviz의 tf2 frame 경고(console_bridge)가 stderr로 직접 쏟아져 콘솔을 도배하므로,
    # 셸에서 stderr를 버려서 막음. (output='log'로는 C++ 레벨 stderr가 안 막힘)
    _rviz_cfg = os.path.join(description_pkg, 'rviz', 'f1tenth_default.rviz')
    rviz_node = ExecuteProcess(
        cmd=['bash', '-c', f'exec rviz2 -d "{_rviz_cfg}" 2>/dev/null'],
        output='log',
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
            spawn_car_launch, agent_node, bridge_node, rviz_node, map_to_odom_node, laser_tf_node,
            spawn_car2_launch, lattice_node_car2, bridge_node_car2,
            map_to_odom_node_car2, laser_tf_node_car2,
        ]
    )

    return LaunchDescription(all_launch_args + [kill_gazebo, delayed_launch])

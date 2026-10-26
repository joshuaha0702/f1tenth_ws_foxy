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
    map_arg = DeclareLaunchArgument(
        'map', default_value='Simple',
        description='맵 이름 (예: Simple, monza_track, interlagos_track, silverstone_track 등)'
    )
    raceline_arg = DeclareLaunchArgument(
        'raceline', default_value='',
        description='플래너가 추종할 raceline CSV 경로 (미지정 시 maps/<map>/raceline1.csv)'
    )
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
    use_norm_arg = DeclareLaunchArgument(
        'use_norm', default_value='auto',
        description='정규화 모델 사용 여부 (auto / true / false, 기본값: auto)'
    )
    model_rate_hz_arg = DeclareLaunchArgument(
        'model_rate_hz', default_value='100.0',
        description='모델 추론 주기 (Hz, 100Hz 모델은 100.0, 10Hz 모델은 10.0, 기본값: 100.0)'
    )
    interp_mode_arg = DeclareLaunchArgument(
        'interp_mode', default_value='linear',
        description='10Hz 제어 보간 모드 (linear / zoh, 기본값: linear)'
    )
    interp_steps_arg = DeclareLaunchArgument(
        'interp_steps', default_value='4',
        description='선형 보간 램프 스텝 수 (기본값: 4)'
    )
    x_arg = DeclareLaunchArgument('x', default_value='6.4', description='car1 스폰 X')
    y_arg = DeclareLaunchArgument('y', default_value='16.0', description='car1 스폰 Y')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='-90.0', description='car1 스폰 Yaw 각도(deg)')
    x2_arg = DeclareLaunchArgument('x2', default_value='6.4', description='car2 스폰 X')
    y2_arg = DeclareLaunchArgument('y2', default_value='12.0', description='car2 스폰 Y')
    yaw2_arg = DeclareLaunchArgument('yaw2', default_value='-90.0', description='car2 스폰 Yaw 각도(deg)')

    # LaunchConfiguration 핸들
    map_name = LaunchConfiguration('map')
    raceline = LaunchConfiguration('raceline')
    namespace = LaunchConfiguration('namespace')
    car2_enabled = LaunchConfiguration('head2head')
    model_path = LaunchConfiguration('model_path')
    use_norm = LaunchConfiguration('use_norm')
    model_rate_hz = LaunchConfiguration('model_rate_hz')
    interp_mode = LaunchConfiguration('interp_mode')
    interp_steps = LaunchConfiguration('interp_steps')
    spawn_x = LaunchConfiguration('x')
    spawn_y = LaunchConfiguration('y')
    spawn_yaw = PythonExpression(['str(float("', LaunchConfiguration('yaw'), '") * 3.14159265358 / 180.0)'])
    spawn_x2 = LaunchConfiguration('x2')
    spawn_y2 = LaunchConfiguration('y2')
    spawn_yaw2 = PythonExpression(['str(float("', LaunchConfiguration('yaw2'), '") * 3.14159265358 / 180.0)'])

    all_launch_args = [
        map_arg, raceline_arg,
        namespace_arg, head2head_arg, model_path_arg, use_norm_arg,
        model_rate_hz_arg, interp_mode_arg, interp_steps_arg,
        x_arg, y_arg, yaw_arg, x2_arg, y2_arg, yaw2_arg
    ]

    # ============================================================
    # 노드 / 액션 정의
    # ============================================================
    kill_gazebo = ExecuteProcess(
        cmd=['bash', '-c', 'pkill -9 -f gzserver; pkill -9 -f gzclient; sleep 1.5'],
        output='screen'
    )

    spawn_car_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description_pkg, 'launch', 'spawn_car.launch.py')),
        launch_arguments={'map': map_name, 'x': spawn_x, 'y': spawn_y, 'yaw': spawn_yaw}.items()
    )

    spawn_car2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description_pkg, 'launch', 'spawn_car.launch.py')),
        launch_arguments={
            'map': map_name,
            'namespace': 'car2',
            'x': spawn_x2, 'y': spawn_y2, 'yaw': spawn_yaw2,
            'launch_gazebo': 'false',
            'color': 'orange',
            'visualize_lidar': 'false',
        }.items(),
        condition=IfCondition(car2_enabled)
    )

    # Lattice 플래너 노드 — car2 (정속 주행, opponent tracking 없음)
    # 맵은 maps/<map>/ 폴더 구조, config는 평가 launch(f1tenth_lattice_gazebo)와 동일한 sim 설정 사용
    lattice_config_path = os.path.join(lattice_pkg, 'config', 'sim_lattice_config.yaml')
    lattice_map_path = PythonExpression(["'", lattice_pkg, "/maps/' + '", map_name, "' + '/' + '", map_name, "_map'"])
    lattice_raceline_path = PythonExpression(["'", lattice_pkg, "/maps/' + '", map_name, "' + '/raceline1.csv' if '", raceline, "' == '' else '", raceline, "'"])

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
            'use_sim_time': True,
        }],
        condition=IfCondition(car2_enabled)
    )

    # Pure Pursuit 제어기 — car2 (플래너와 분리된 프로세스가 /car2/drive를 publish)
    controller_node_car2 = Node(
        package='f1tenth_lattice_ros2',
        executable='pure_pursuit_controller_node',
        name='pure_pursuit_controller',
        namespace='car2',
        output='screen',
        parameters=[{
            'config_path': lattice_config_path,
            'raceline_path': lattice_raceline_path,
            'max_speed': 2.0,
            'max_steering_angle': 0.4189,
            'use_sim_time': True,
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
        output='screen',
        parameters=[config_path, {
            'use_sim_time': False,
            'model_path': model_path,
            'use_norm': use_norm,
            'model_rate_hz': PythonExpression(["float('", model_rate_hz, "')"]),
            'interp_mode': interp_mode,
            'interp_steps': PythonExpression(["int('", interp_steps, "')"]),
        }]
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
            spawn_car2_launch, lattice_node_car2, controller_node_car2, bridge_node_car2,
            map_to_odom_node_car2, laser_tf_node_car2,
        ]
    )

    return LaunchDescription(all_launch_args + [kill_gazebo, delayed_launch])

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

    # 최신 기본 모델 경로 탐색 (30Hz 정규화 모델 우선)
    default_model = os.path.join(end2race_pkg, 'models', 'norm_ds30_reflector_v1_ep140.pth')
    if not os.path.exists(default_model):
        default_model = os.path.join(end2race_pkg, 'models', 'end2race.pth')

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
        'head2head', default_value='false',
        description='Lattice 플래너로 정속 주행하는 car2를 함께 생성 (true/false, 기본값: false)'
    )
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model,
        description='end2race 모델 가중치(.pth) 경로'
    )
    use_norm_arg = DeclareLaunchArgument(
        'use_norm', default_value='auto',
        description='정규화 모델 사용 여부 (auto / true / false, 기본값: auto)'
    )
    model_rate_hz_arg = DeclareLaunchArgument(
        'model_rate_hz', default_value='30.0',
        description='모델 추론 주기 (Hz, 기본값: 30.0)'
    )
    interp_mode_arg = DeclareLaunchArgument(
        'interp_mode', default_value='linear',
        description='제어 보간 모드 (linear / zoh, 기본값: linear)'
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

    # 기존 Gazebo 프로세스 정리
    kill_gazebo = ExecuteProcess(
        cmd=['bash', '-c', 'pkill -9 -f gzserver; pkill -9 -f gzclient; sleep 1.5'],
        output='screen'
    )

    # 1. 현실성 보완 가제보 모델로 car1 스폰
    spawn_car_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description_pkg, 'launch', 'spawn_car_realistic.launch.py')),
        launch_arguments={
            'map': map_name,
            'x': spawn_x,
            'y': spawn_y,
            'yaw': spawn_yaw,
            'namespace': namespace,
            'color': 'blue'
        }.items()
    )

    # 2. car1 Ackermann -> Twist 변환 브릿지 (f1tenth_fgm_ros2)
    bridge_node = Node(
        package='f1tenth_fgm_ros2',
        executable='ackermann_to_twist.py',
        name='ackermann_to_twist',
        namespace=namespace,
        output='screen'
    )

    # 3. car2 스폰 및 Lattice 플래너 (head2head:=true 시)
    spawn_car2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(description_pkg, 'launch', 'spawn_car_realistic.launch.py')),
        launch_arguments={
            'map': map_name,
            'x': spawn_x2,
            'y': spawn_y2,
            'yaw': spawn_yaw2,
            'namespace': 'car2',
            'color': 'red',
            'launch_gazebo': 'false'
        }.items(),
        condition=IfCondition(car2_enabled)
    )

    bridge_node_car2 = Node(
        package='f1tenth_fgm_ros2',
        executable='ackermann_to_twist.py',
        name='ackermann_to_twist_car2',
        namespace='car2',
        output='screen',
        condition=IfCondition(car2_enabled)
    )

    car2_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(lattice_pkg, 'launch', 'f1tenth_lattice_gazebo.launch.py')),
        launch_arguments={
            'map': map_name,
            'raceline': raceline,
            'spawn_car1': 'false',
            'spawn_car2': 'false',
            'launch_gazebo': 'false',
            'launch_rviz': 'false',
            'rviz': 'false',
            'kill_prev_gazebo': 'false',
            'overtake_mode': 'off',
            'max_speed': '2.0'
        }.items(),
        condition=IfCondition(car2_enabled)
    )

    # 4. End2Race Agent 추론 노드 (실제 패키지 실행체: agent_node)
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

    # 5. RViz 시각화 및 TF 브릿지
    rviz_config_path = os.path.join(description_pkg, 'rviz', 'f1tenth_default.rviz')
    rviz_node = ExecuteProcess(
        cmd=['bash', '-c', f'exec rviz2 -d "{rviz_config_path}" 2>/dev/null'],
        output='log'
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

    return LaunchDescription([
        map_arg, raceline_arg,
        namespace_arg, head2head_arg, model_path_arg, use_norm_arg,
        model_rate_hz_arg, interp_mode_arg, interp_steps_arg,
        x_arg, y_arg, yaw_arg, x2_arg, y2_arg, yaw2_arg,
        kill_gazebo,
        spawn_car_launch,
        bridge_node,
        map_to_odom_node,
        laser_tf_node,
        TimerAction(period=2.5, actions=[spawn_car2_launch, bridge_node_car2, car2_planner_launch]),
        TimerAction(period=3.5, actions=[agent_node, rviz_node])
    ])

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    lattice_pkg = get_package_share_directory('f1tenth_lattice_ros2')

    # 1. namespace: 로봇의 네임스페이스 (기본값: 'car1')
    # 실차 주행 시 사용하는 차량 이름에 맞춰 네임스페이스를 지정합니다. (예: car1, racecar 등)
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='car1',
        description='Robot namespace'
    )
    namespace = LaunchConfiguration('namespace')

    # 2. max_speed: 최대 선속도 한계치 (기본값: 3.0 m/s)
    # 실차 첫 테스트 시 안전을 위해 1.0 ~ 1.5 m/s 정도로 낮춰서 테스트하는 것을 추천합니다.
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='3.0',
        description='Maximum speed for the lattice planner'
    )
    max_speed = LaunchConfiguration('max_speed')

    # 3. max_steering_angle: 최대 조향각 한계치 (기본값: 0.4189 rad = 약 24도)
    # 실제 차량의 물리적 조향각 한계 범위 내로 설정해야 모터 서보에 무리가 가지 않습니다.
    max_steering_angle_arg = DeclareLaunchArgument(
        'max_steering_angle',
        default_value='0.4189',
        description='Maximum steering angle (radians)'
    )
    max_steering_angle = LaunchConfiguration('max_steering_angle')

    # 4. plan_frequency: 래티스 플래너 재계획 주기 (기본값: 10.0 Hz)
    # 실차 온보드 PC의 CPU 부하에 따라 조절합니다. 일반적으로 10.0Hz ~ 20.0Hz 범위로 사용합니다.
    plan_freq_arg = DeclareLaunchArgument(
        'plan_frequency',
        default_value='10.0',
        description='Planning frequency (Hz)'
    )
    plan_freq = LaunchConfiguration('plan_frequency')

    # 5. map_path: 사용할 2D Grid Map 파일 경로 (확장자인 .yaml / .png 제외)
    # AMCL 위치추정 및 장애물 코스트맵에 사용됩니다. 실차 맵핑으로 생성한 지도 경로를 입력합니다.
    map_path_arg = DeclareLaunchArgument(
        'map_path',
        default_value=os.path.join(lattice_pkg, 'maps', 'Simple_map'),
        description='Path to map file (without extension: .yaml and .png will be appended)'
    )
    map_path = LaunchConfiguration('map_path')

    # 6. use_map_server: 맵 서버 노드를 이 런치 파일에서 직접 켤지 여부 (기본값: true)
    # 외부에서 이미 map_server 또는 slam 노드가 켜져 있는 상태라면 false로 변경하여 충돌을 방지합니다.
    use_map_server_arg = DeclareLaunchArgument(
        'use_map_server',
        default_value='true',
        description='Launch map_server to publish the /map topic'
    )
    use_map_server = LaunchConfiguration('use_map_server')

    # 7. localization_mode: 위치 추정 방식 선택 (기본값: fusion)
    # - fusion: AMCL 포즈와 Odometry를 융합하여 센서 드리프트를 방지하고 정밀도를 높임
    # - amcl: 오직 AMCL 포즈만을 차량 위치로 사용하여 주행
    # - odom: AMCL 없이 순수 Odometry 정보로만 주행 (시간 경과 시 누적 오차 발생)
    localization_mode_arg = DeclareLaunchArgument(
        'localization_mode',
        default_value='fusion',
        choices=['fusion', 'amcl', 'odom'],
        description='Localization mode: fusion (AMCL+Odometry), amcl (AMCL only), odom (Odometry only)'
    )
    localization_mode = LaunchConfiguration('localization_mode')

    # 8~10. initial_x, initial_y, initial_yaw: AMCL/Odom 시작 위치 초기값 (기본값: 0.0)
    # 실차를 맵의 어디에 올려놓고 시작할지 명확히 지정해 주어야 AMCL이 첫 포즈를 정확하게 잡습니다. (라디안 단위)
    initial_x_arg = DeclareLaunchArgument(
        'initial_x',
        default_value='0.0',
        description='Initial X position for AMCL or odom start'
    )
    initial_x = LaunchConfiguration('initial_x')

    initial_y_arg = DeclareLaunchArgument(
        'initial_y',
        default_value='0.0',
        description='Initial Y position for AMCL or odom start'
    )
    initial_y = LaunchConfiguration('initial_y')

    initial_yaw_arg = DeclareLaunchArgument(
        'initial_yaw',
        default_value='0.0',
        description='Initial yaw for AMCL or odom start'
    )
    initial_yaw = LaunchConfiguration('initial_yaw')

    # 11. odom_cov_scale: 융합 시 오도메트리 공분산 스케일 팩터 (기본값: 100.0)
    # 이 값이 클수록 AMCL에 더 높은 신뢰(가중치)를 두고, 작을수록 오도메트리를 더 신뢰합니다.
    odom_cov_scale_arg = DeclareLaunchArgument(
        'odom_cov_scale',
        default_value='100.0',
        description='Odometry covariance scale when fusing with AMCL'
    )
    odom_cov_scale = LaunchConfiguration('odom_cov_scale')

    config_path = os.path.join(lattice_pkg, 'config', 'lattice_config.yaml')
    amcl_config_path = os.path.join(lattice_pkg, 'config', 'slam_amcl_config.yaml')
    raceline_path = os.path.join(lattice_pkg, 'maps', 'raceline1.csv')

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_path,
        }],
        condition=IfCondition(
            PythonExpression(["'", use_map_server, "' == 'true' and '", localization_mode, "' != 'odom'"])
        )
    )

    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        namespace=namespace,
        output='screen',
        parameters=[amcl_config_path],
        remappings=[('scan', 'scan')],
        condition=IfCondition(
            PythonExpression(["'", localization_mode, "' != 'odom'"])
        )
    )

    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_amcl',
        namespace=namespace,
        output='screen',
        parameters=[{'autostart': True, 'node_names': ['amcl']}],
        condition=IfCondition(
            PythonExpression(["'", localization_mode, "' != 'odom'"])
        )
    )

    slam_planner_node = Node(
        package='f1tenth_lattice_ros2',
        executable='slam_lattice_planner_node',
        name='slam_lattice_planner',
        namespace=namespace,
        output='screen',
        parameters=[{
            'config_path': config_path,
            'raceline_path': raceline_path,
            'map_path': map_path,
            'max_speed': max_speed,
            'max_steering_angle': max_steering_angle,
            'plan_frequency': plan_freq,
            'localization_mode': localization_mode,
            'initial_x': initial_x,
            'initial_y': initial_y,
            'initial_yaw': initial_yaw,
            'odom_cov_scale': odom_cov_scale,
        }],
        remappings=[
            ('odom', 'odom'),
            ('drive', 'drive'),
        ]
    )

    return LaunchDescription([
        namespace_arg,
        max_speed_arg,
        max_steering_angle_arg,
        plan_freq_arg,
        map_path_arg,
        use_map_server_arg,
        localization_mode_arg,
        initial_x_arg,
        initial_y_arg,
        initial_yaw_arg,
        odom_cov_scale_arg,
        map_server_node,
        amcl_node,
        lifecycle_mgr,
        slam_planner_node,
    ])

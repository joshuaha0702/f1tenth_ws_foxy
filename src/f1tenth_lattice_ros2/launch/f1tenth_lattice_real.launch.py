import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    lattice_pkg = get_package_share_directory('f1tenth_lattice_ros2')

    default_lattice_config = os.path.join(lattice_pkg, 'config', 'lattice_config.yaml')
    default_amcl_config = os.path.join(lattice_pkg, 'config', 'amcl_config.yaml')

    # --- Launch Arguments ---
    map_name_arg = DeclareLaunchArgument(
        'map_name',
        default_value='monza_track',
        description='Name of the map folder (e.g. Simple, monza_track)'
    )
    config_arg = DeclareLaunchArgument(
        'config',
        default_value=default_lattice_config,
        description='Path to lattice_config.yaml'
    )
    amcl_config_arg = DeclareLaunchArgument(
        'amcl_config',
        default_value=default_amcl_config,
        description='Path to amcl_config.yaml'
    )
    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='1.0',
        description='Max speed limit for safety'
    )
    drive_topic_arg = DeclareLaunchArgument(
        'drive_topic',
        default_value='/drive',
        description='Output drive topic'
    )
    plan_freq_arg = DeclareLaunchArgument(
        'plan_freq',
        default_value='20.0',
        description='Planning and drive publish frequency (Hz)'
    )
    loc_mode_arg = DeclareLaunchArgument(
        'loc_mode',
        default_value='scan',
        description='Localization mode (scan, amcl, odom, fusion)'
    )

    map_name = LaunchConfiguration('map_name')
    config_path = LaunchConfiguration('config')
    amcl_config_path = LaunchConfiguration('amcl_config')
    max_speed = LaunchConfiguration('max_speed')
    drive_topic = LaunchConfiguration('drive_topic')
    plan_freq = LaunchConfiguration('plan_freq')
    loc_mode = LaunchConfiguration('loc_mode')

    # --- Dynamic Path Resolution ---
    map_dir = PythonExpression(["'", lattice_pkg, "/maps/' + '", map_name, "'"])
    map_yaml_file = PythonExpression(["'", map_dir, "/' + '", map_name, "_map.yaml'"])
    default_dynamic_raceline = PythonExpression(["'", map_dir, "/raceline1.csv'"])

    raceline_arg = DeclareLaunchArgument(
        'raceline',
        default_value=default_dynamic_raceline,
        description='Path to raceline csv file'
    )
    raceline = LaunchConfiguration('raceline')

    # 1. Nav2 Map Server Node (지도 파일 발행)
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'yaml_filename': map_yaml_file
        }]
    )

    # 2. Nav2 AMCL Node (파티클 필터 위치 추정)
    amcl_node = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[amcl_config_path]
    )

    # 3. Lifecycle Manager (Map Server와 AMCL 자동 활성화)
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server', 'amcl']
        }]
    )

    use_global_loc_arg = DeclareLaunchArgument(
        'use_global_loc',
        default_value='false',
        description='Whether to call /reinitialize_global_localization (scatters particles randomly across map)'
    )
    use_global_loc = LaunchConfiguration('use_global_loc')

    # 4. Global Localization Automatic Trigger (선택적 자동 위치 무작위 리셋)
    trigger_global_loc = ExecuteProcess(
        cmd=['ros2', 'service', 'call', '/reinitialize_global_localization', 'std_srvs/srv/Empty'],
        output='screen',
        condition=IfCondition(use_global_loc)
    )

    delay_global_loc = TimerAction(
        period=2.0,
        actions=[trigger_global_loc]
    )

    # 5. SLAM Lattice Planner Node (주행 플래너)
    # [수정] max_steering_angle을 0.4189에서 0.31로 낮추어 VESC 서보 클리핑(0.15) 발생 차단
    lattice_planner_node = Node(
        package='f1tenth_lattice_ros2',
        executable='slam_planner_node',
        name='slam_lattice_planner',
        output='screen',
        parameters=[{
            'config_path': config_path,
            'raceline_path': raceline,
            'map_path': map_dir,
            'max_speed': max_speed,
            'max_steering_angle': 0.26,  # <--- VESC 서보 0.85 클리핑 완전 방지 한계값 (-0.26 rad -> servo 0.8459)
            'plan_frequency': plan_freq,
            'localization_mode': loc_mode,
            'use_sim_time': False
        }],
        remappings=[
            ('drive', drive_topic),
            ('/drive', drive_topic)
        ]
    )

    # 6. Joy Bag Recorder Node (조이스틱 버튼으로 ros2 bag 시작/중지)
    record_bag_arg = DeclareLaunchArgument(
        'record_bag',
        default_value='true',
        description='Enable joy_bag_recorder_node for joystick triggered ros2 bag recording'
    )
    record_bag = LaunchConfiguration('record_bag')

    joy_bag_recorder_node = Node(
        package='f1tenth_lattice_ros2',
        executable='joy_bag_recorder_node',
        name='joy_bag_recorder',
        output='screen',
        parameters=[{
            'start_button': 7,  # 7번 버튼: 시작
            'stop_button': 6,   # 6번 버튼: 중지
            'bag_prefix': 'f1tenth_run',
            'output_dir': 'bags'
        }],
        condition=IfCondition(record_bag)
    )

    return LaunchDescription([
        map_name_arg,
        raceline_arg,
        config_arg,
        amcl_config_arg,
        max_speed_arg,
        drive_topic_arg,
        plan_freq_arg,
        loc_mode_arg,
        use_global_loc_arg,
        record_bag_arg,
        map_server_node,
        amcl_node,
        lifecycle_manager_node,
        delay_global_loc,
        lattice_planner_node,
        joy_bag_recorder_node
    ])
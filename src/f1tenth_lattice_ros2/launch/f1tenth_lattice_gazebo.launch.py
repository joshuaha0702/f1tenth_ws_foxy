import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    lattice_pkg = get_package_share_directory('f1tenth_lattice_ros2')
    description_pkg = get_package_share_directory('racecar_description')

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

    # Spawn coordinates arguments
    x_arg = DeclareLaunchArgument('x', default_value='6.4', description='Spawn X position')
    y_arg = DeclareLaunchArgument('y', default_value='16.0', description='Spawn Y position')
    yaw_arg = DeclareLaunchArgument('yaw_deg', default_value='-90.0', description='Spawn Yaw angle (degrees)')
    spawn_x = LaunchConfiguration('x')
    spawn_y = LaunchConfiguration('y')
    # degree -> radian 변환 (PythonExpression으로 런타임에 계산)
    spawn_yaw = PythonExpression(['str(float("', LaunchConfiguration('yaw_deg'), '") * 3.14159265358979 / 180.0)'])

    # Spawn Gazebo + robot (same as FGM launch)
    spawn_car_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(description_pkg, 'launch', 'spawn_car.launch.py')
        ),
        launch_arguments={'x': spawn_x, 'y': spawn_y, 'yaw': spawn_yaw}.items()
    )

    # Paths to lattice planner resources
    # raceline1.csv = center lane (3 lanes: inner=0, center=1, outer=2)
    config_path = os.path.join(lattice_pkg, 'config', 'lattice_config.yaml')
    map_path = os.path.join(lattice_pkg, 'maps', 'Simple_map')   # no extension
    raceline_path = os.path.join(lattice_pkg, 'maps', 'raceline1.csv')

    # Lattice planner node
    lattice_node = Node(
        package='f1tenth_lattice_ros2',
        executable='lattice_planner_node',
        name='lattice_planner',
        namespace=namespace,
        output='screen',
        parameters=[{
            'config_path': config_path,
            'raceline_path': raceline_path,
            'map_path': map_path,
            'max_speed': 3.0,
            'max_steering_angle': 0.4189,
            'plan_frequency': 10.0,
        }]
    )

    # RViz2 with existing config
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(description_pkg, 'rviz', 'f1tenth_default.rviz')],
        output='screen'
    )

    # Static transform map -> car1/odom (odom is already absolute in Gazebo world)
    map_to_odom_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', [namespace, '/odom']]
    )

    # Static transform car1/laser -> laser (fixes Gazebo LiDAR plugin ignoring frame_name)
    laser_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_tf_fix',
        arguments=['0', '0', '0', '0', '0', '0', [namespace, '/laser'], 'laser']
    )

    # 가제보 정리 후 1.5초 대기 후 나머지 노드 시작
    delayed_launch = TimerAction(
        period=1.5,
        actions=[
            spawn_car_launch,
            lattice_node,
            rviz_node,
            map_to_odom_node,
            laser_tf_node,
        ]
    )

    return LaunchDescription([
        namespace_arg,
        x_arg,
        y_arg,
        yaw_arg,
        kill_gazebo,
        delayed_launch,
    ])

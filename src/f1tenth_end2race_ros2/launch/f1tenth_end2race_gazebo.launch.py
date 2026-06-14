import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'f1tenth_end2race_ros2'
    end2race_pkg = get_package_share_directory(pkg_name)
    description_pkg = get_package_share_directory('racecar_description')

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

    config_path = os.path.join(end2race_pkg, 'config', 'end2race_config.yaml')

    # Agent 노드 설정
    agent_node = Node(
        package=pkg_name,
        executable='agent_node',
        name='end2race_agent',
        namespace=namespace,
        output='screen',
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
        actions=[spawn_car_launch, agent_node, rviz_node, map_to_odom_node, laser_tf_node]
    )

    return LaunchDescription([namespace_arg, kill_gazebo, delayed_launch])

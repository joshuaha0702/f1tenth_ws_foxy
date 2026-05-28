import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    lattice_pkg = get_package_share_directory('f1tenth_lattice_ros2')

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='car1',
        description='Robot namespace'
    )
    namespace = LaunchConfiguration('namespace')

    max_speed_arg = DeclareLaunchArgument(
        'max_speed',
        default_value='3.0',
        description='Maximum speed for the lattice planner'
    )
    max_speed = LaunchConfiguration('max_speed')

    max_steering_angle_arg = DeclareLaunchArgument(
        'max_steering_angle',
        default_value='0.4189',
        description='Maximum steering angle (radians)'
    )
    max_steering_angle = LaunchConfiguration('max_steering_angle')

    plan_freq_arg = DeclareLaunchArgument(
        'plan_frequency',
        default_value='10.0',
        description='Planning frequency (Hz)'
    )
    plan_freq = LaunchConfiguration('plan_frequency')

    map_path_arg = DeclareLaunchArgument(
        'map_path',
        default_value=os.path.join(lattice_pkg, 'maps', 'Simple_map'),
        description='Path to map file (without extension: .yaml and .png will be appended)'
    )
    map_path = LaunchConfiguration('map_path')

    use_map_server_arg = DeclareLaunchArgument(
        'use_map_server',
        default_value='true',
        description='Launch map_server to publish the /map topic'
    )
    use_map_server = LaunchConfiguration('use_map_server')

    localization_mode_arg = DeclareLaunchArgument(
        'localization_mode',
        default_value='fusion',
        choices=['fusion', 'amcl', 'odom'],
        description='Localization mode: fusion (AMCL+Odometry), amcl (AMCL only), odom (Odometry only)'
    )
    localization_mode = LaunchConfiguration('localization_mode')

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
            PythonExpression([use_map_server, " == 'true' and ", localization_mode, " != 'odom'"])
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
            PythonExpression([localization_mode, " != 'odom'"])
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
            PythonExpression([localization_mode, " != 'odom'"])
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

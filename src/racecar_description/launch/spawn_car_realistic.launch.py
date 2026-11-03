import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def launch_setup(context, *args, **kwargs):
    # 1. 경로 및 설정값 세팅
    pkg_path = get_package_share_directory('racecar_description')
    
    # 현실성 보완 xacro 모델 기본 사용
    xacro_file = os.path.join(pkg_path, 'urdf', 'racecar_realistic.xacro')
    
    map_name = LaunchConfiguration('map').perform(context)
    namespace_val = LaunchConfiguration('namespace').perform(context)
    color = LaunchConfiguration('color')
    launch_gazebo = LaunchConfiguration('launch_gazebo')
    visualize_lidar = LaunchConfiguration('visualize_lidar')
    gui = LaunchConfiguration('gui')
    gazebo_record = LaunchConfiguration('gazebo_record')
    
    spawn_x = LaunchConfiguration('x').perform(context)
    spawn_y = LaunchConfiguration('y').perform(context)
    spawn_yaw = LaunchConfiguration('yaw').perform(context)
    
    # 맵에 따른 world 파일 선택
    if map_name in ['monza', 'silverstone', 'interlagos']:
        world_file_path = os.path.join(pkg_path, 'worlds', f'{map_name}_track.world')
    elif map_name in ['monza_track', 'silverstone_track', 'interlagos_track', 'racecar_walker', 'simple', 'Simple_augmented', 'f110_competition','f110_racetrack']:
        world_file_path = os.path.join(pkg_path, 'worlds', f'{map_name.lower()}.world')
    elif os.path.exists(os.path.join(pkg_path, 'worlds', f'{map_name.lower()}.world')):
        world_file_path = os.path.join(pkg_path, 'worlds', f'{map_name.lower()}.world')
    else:
        world_file_path = os.path.join(pkg_path, 'worlds', 'simple.world')

    # 2. Xacro 실행 및 URDF 생성 (현실성 보완 모델 적용)
    robot_description_content = ParameterValue(
        Command(['xacro ', xacro_file,
                 ' namespace:=', namespace_val,
                 ' color:=', color,
                 ' visualize_lidar:=', visualize_lidar]),
        value_type=str
    )

    # 3. Robot State Publisher
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=namespace_val,
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': True,
            'frame_prefix': namespace_val + '/'
        }]
    )

    # Joint state fallback
    node_joint_state_fallback = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        namespace=namespace_val,
        name='joint_state_fallback',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'source_list': ['joint_states_gz'],
            'rate': 30,
            'use_sim_time': True,
        }]
    )

    # 4. Gazebo 실행
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            'world': world_file_path,
            'gui': gui
        }.items(),
        condition=IfCondition(launch_gazebo)
    )

    # 5. Entity Spawn
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', f'/{namespace_val}/robot_description' if namespace_val else '/robot_description',
            '-entity', namespace_val if namespace_val else 'racecar',
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', '0.05',
            '-Y', spawn_yaw
        ],
        output='screen'
    )

    # 6. Ackermann to Twist Bridge (f1tenth_fgm_ros2)
    ackermann_to_twist = Node(
        package='f1tenth_fgm_ros2',
        executable='ackermann_to_twist.py',
        namespace=namespace_val,
        name='ackermann_to_twist',
        output='screen'
    )

    return [
        gazebo,
        node_robot_state_publisher,
        node_joint_state_fallback,
        spawn_entity,
        ackermann_to_twist
    ]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value='Simple', description='Map name'),
        DeclareLaunchArgument('x', default_value='6.4', description='Spawn X position'),
        DeclareLaunchArgument('y', default_value='16.0', description='Spawn Y position'),
        DeclareLaunchArgument('yaw', default_value='-1.57079632679', description='Spawn Yaw angle in radians'),
        DeclareLaunchArgument('namespace', default_value='car1', description='Robot namespace'),
        DeclareLaunchArgument('color', default_value='blue', description='Car color'),
        DeclareLaunchArgument('launch_gazebo', default_value='true', description='Whether to launch Gazebo'),
        DeclareLaunchArgument('visualize_lidar', default_value='false', description='Visualize lidar rays'),
        DeclareLaunchArgument('gui', default_value='true', description='Launch Gazebo GUI'),
        DeclareLaunchArgument('gazebo_record', default_value='false', description='Enable Gazebo recording'),
        OpaqueFunction(function=launch_setup)
    ])

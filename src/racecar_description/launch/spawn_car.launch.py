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
    xacro_file = os.path.join(pkg_path, 'urdf', 'racecar.xacro')
    
    map_name = LaunchConfiguration('map').perform(context)
    namespace_val = LaunchConfiguration('namespace').perform(context)
    color = LaunchConfiguration('color')
    launch_gazebo = LaunchConfiguration('launch_gazebo')
    visualize_lidar = LaunchConfiguration('visualize_lidar')
    gui = LaunchConfiguration('gui')
    
    # 전달받은 좌표 (기본적으로 f1tenth_lattice_gazebo.launch.py 에서 전달됨)
    spawn_x = LaunchConfiguration('x').perform(context)
    spawn_y = LaunchConfiguration('y').perform(context)
    spawn_yaw = LaunchConfiguration('yaw').perform(context)
    
    # 맵별 world 파일 및 초기 위치 설정
    # monza -10 5 90
    # silverstone -10 2 90
    # interlagos 2.5 -10 90
    # car2를 위해 namespace에 따라 위치를 약간 다르게 줄 수도 있지만, 
    # 기본적으로 car1 기준으로 맵별 고정 좌표를 덮어씁니다.
    
    # 맵에 따른 world 파일 선택
    if map_name in ['monza', 'silverstone', 'interlagos']:
        world_file_path = os.path.join(pkg_path, 'worlds', f'{map_name}_track.world')
    elif map_name in ['monza_track', 'silverstone_track', 'interlagos_track', 'racecar_walker', 'simple', 'f110_competition','f110_racetrack']:
        world_file_path = os.path.join(pkg_path, 'worlds', f'{map_name}.world')
    else:
        world_file_path = os.path.join(pkg_path, 'worlds', 'simple.world')



    # 맵 이름이 특정 트랙일 경우의 좌표 덮어쓰기 하드코딩을 제거했습니다.
    # 이제 터미널이나 부모 런치파일에서 넘겨준 x, y, yaw 인자값이 그대로 사용됩니다.

    # 2. Xacro 실행 및 URDF 생성 (namespace, color 인자를 xacro에 전달)
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
            'frame_prefix': namespace_val + '/' # TF 프레임 이름 앞에 네임스페이스 추가
        }]
    )

    # 4. Gazebo 실행
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            'world': world_file_path,
            'verbose': 'true',
            'gui': gui,
        }.items(),
        condition=IfCondition(launch_gazebo)
    )

    # 5. Gazebo에 로봇 소환 (Spawn)
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-topic', [namespace_val, '/robot_description'],
            '-entity', namespace_val,
            '-x', spawn_x,
            '-y', spawn_y,
            '-Y', spawn_yaw,
            '-z', '0.05'
        ]
    )

    return [node_robot_state_publisher, gazebo, spawn_entity]

def generate_launch_description():
    declare_map_cmd = DeclareLaunchArgument('map', default_value='simple', description='Map name for setting world and poses')
    declare_namespace_cmd = DeclareLaunchArgument('namespace', default_value='car1')
    declare_x_cmd = DeclareLaunchArgument('x', default_value='-10.0') 
    declare_y_cmd = DeclareLaunchArgument('y', default_value='2.0')
    declare_yaw_cmd = DeclareLaunchArgument('yaw', default_value='1.570796') # gazebo는 라디안을 사용하므로 90도를 라디안으로 기본 세팅
    declare_launch_gazebo_cmd = DeclareLaunchArgument('launch_gazebo', default_value='true')
    declare_gui_cmd = DeclareLaunchArgument('gui', default_value='true',
                                            description='Set to "false" to run Gazebo headless (gzserver only)')
    declare_color_cmd = DeclareLaunchArgument('color', default_value='blue')
    declare_visualize_lidar_cmd = DeclareLaunchArgument('visualize_lidar', default_value='true',
                                                        description='LiDAR 레이 시각화 여부')

    return LaunchDescription([
        declare_map_cmd,
        declare_namespace_cmd,
        declare_x_cmd,
        declare_y_cmd,
        declare_yaw_cmd,
        declare_launch_gazebo_cmd,
        declare_gui_cmd,
        declare_color_cmd,
        declare_visualize_lidar_cmd,
        OpaqueFunction(function=launch_setup)
    ])

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # 1. 경로 및 설정값 세팅
    pkg_path = get_package_share_directory('racecar_description')
    xacro_file = os.path.join(pkg_path, 'urdf', 'racecar.xacro')
    world_file_path = os.path.join(pkg_path, 'worlds', 'simple.world')

    # 런치 인자 선언 (외부에서 namespace, x, y 좌표를 받을 수 있게 함)
    namespace = LaunchConfiguration('namespace')
    spawn_x = LaunchConfiguration('x')
    spawn_y = LaunchConfiguration('y')
    launch_gazebo = LaunchConfiguration('launch_gazebo') # 가제보 실행 여부 플래그

    declare_namespace_cmd = DeclareLaunchArgument('namespace', default_value='car1')
    declare_x_cmd = DeclareLaunchArgument('x', default_value='4.0')
    declare_y_cmd = DeclareLaunchArgument('y', default_value='2.0')
    declare_launch_gazebo_cmd = DeclareLaunchArgument('launch_gazebo', default_value='true')

    # 2. Xacro 실행 및 URDF 생성 (namespace 인자를 xacro에 전달)
    # Command를 사용하여 런타임에 namespace 값이 반영된 URDF를 생성합니다.
    robot_description_content = ParameterValue(
        Command(['xacro ', xacro_file, ' namespace:=', namespace]),
        value_type=str
    )

    # 3. Robot State Publisher
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace=namespace,
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': True,
            'frame_prefix': [namespace, '/'] # TF 프레임 이름 앞에 네임스페이스 추가
        }]
    )

    # 4. Gazebo 실행
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={'world': world_file_path, 'verbose': 'true'}.items(),
        condition=IfCondition(launch_gazebo)
    )

    # 5. Gazebo에 로봇 소환 (Spawn)
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-topic', [namespace, '/robot_description'], # 네임스페이스가 붙은 정확한 토픽 지정
            '-entity', namespace, # 충돌 방지를 위해 로봇 이름을 namespace(car1, car2)로 지정
            '-x', spawn_x,
            '-y', spawn_y,
            '-z', '0.05'
        ]
    )

    return LaunchDescription([
        declare_namespace_cmd,
        declare_x_cmd,
        declare_y_cmd,
        declare_launch_gazebo_cmd,
        node_robot_state_publisher,
        gazebo,
        spawn_entity
    ])
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # 1. 경로 설정
    pkg_path = get_package_share_directory('racecar_description')
    xacro_file = os.path.join(pkg_path, 'urdf', 'racecar.xacro')
    
    # [수정] 월드 파일의 전체 경로를 변수로 만듭니다. (따옴표 필수!)
    world_file_path = os.path.join(pkg_path, 'worlds', 'simple.world')
    
    # 2. Xacro를 URDF로 변환
    robot_description_config = xacro.process_file(xacro_file)
    params = {'robot_description': robot_description_config.toxml()}

    # 3. Robot State Publisher
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    # 4. Gazebo 실행
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            # [수정] 위에서 정의한 변수를 넣어줍니다.
            'world': world_file_path,
            'verbose': 'true'
        }.items()
    )

    # 5. Gazebo에 로봇 소환 (Spawn)
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', 
        '-entity', 'racecar', 
        '-x', '4.0', '-y', '2.0'],
        output='screen'
    )

    return LaunchDescription([
        node_robot_state_publisher,
        gazebo,
        spawn_entity
    ])
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 패키지 경로 설정
    fgm_pkg_share = get_package_share_directory('f1tenth_fgm_ros2')
    description_pkg_share = get_package_share_directory('racecar_description')

    # 2. spawn_car.launch.py 포함 (Gazebo + Robot Spawn)
    spawn_car_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(description_pkg_share, 'launch', 'spawn_car.launch.py')
        ])
    )

    # 3. FGM 노드 설정 (YAML 파라미터 포함)
    fgm_config = os.path.join(fgm_pkg_share, 'config', 'fgm_config.yaml')
    fgm_node = Node(
        package='f1tenth_fgm_ros2',
        executable='fgm_node', # CMakeLists의 add_executable 이름 확인!
        name='fgm_disparities',
        output='screen',
        parameters=[fgm_config]
    )

    # 4. RViz2 실행
    # (팁: 나중에 저장된 .rviz 설정파일이 생기면 인자로 추가할 수 있습니다)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(description_pkg_share, 'rviz', 'f1tenth_default.rviz')], # RViz 설정파일 경로
        output='screen'
    )

    # 마스터 실행 리스트 반환
    return LaunchDescription([
        spawn_car_launch,
        fgm_node,
        rviz_node
    ])
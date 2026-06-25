import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 패키지 경로 설정
    fgm_pkg_share = get_package_share_directory('f1tenth_fgm_ros2')
    description_pkg_share = get_package_share_directory('racecar_description')

    # 2. 네임스페이스 및 FGM 노드 설정
    # 외부에서 namespace='car1' 형태로 값을 넘겨받음.
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='car1',
        description='Namespace for the FGM node'
    )
    namespace = LaunchConfiguration('namespace')

    # 추가: 맵과 초기 좌표 인자 선언
    map_arg = DeclareLaunchArgument('map', default_value='Simple', description='Map name')
    x_arg = DeclareLaunchArgument('x', default_value='6.4')
    y_arg = DeclareLaunchArgument('y', default_value='16.0')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='-1.570796') # -90도를 라디안으로 기본값 설정
    
    map_name = LaunchConfiguration('map')
    spawn_x = LaunchConfiguration('x')
    spawn_y = LaunchConfiguration('y')
    spawn_yaw = LaunchConfiguration('yaw')

    # 3. spawn_car.launch.py 포함 (Gazebo + Robot Spawn)
    spawn_car_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(description_pkg_share, 'launch', 'spawn_car.launch.py')
        ]),
        launch_arguments={
            'namespace': namespace,
            'map': map_name,
            'x': spawn_x,
            'y': spawn_y,
            'yaw': spawn_yaw
        }.items()
    )
    fgm_config = os.path.join(fgm_pkg_share, 'config', 'fgm_config.yaml')
    fgm_node = Node(
        package='f1tenth_fgm_ros2',
        executable='fgm_node',
        name='fgm_disparities',
        namespace=namespace, # 네임스페이스 적용
        output='screen',
        parameters=[
            fgm_config, 
            {'robot_name': namespace} # 코드 내부의 frame_id 설정 등을 위해 같이 넘겨줌
        ]
    )

    # 4. Ackermann to Twist Bridge (Gazebo 시뮬레이션용)
    bridge_node = Node(
        package='f1tenth_fgm_ros2',
        executable='ackermann_to_twist.py',
        name='ackermann_to_twist',
        namespace=namespace,
        output='screen'
    )

    # 5. RViz2 실행
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(description_pkg_share, 'rviz', 'f1tenth_default.rviz')], # RViz 설정파일 경로
        output='screen'
    )

    # 마스터 실행 리스트 반환
    return LaunchDescription([
        namespace_arg,
        map_arg,
        x_arg,
        y_arg,
        yaw_arg,
        spawn_car_launch,
        fgm_node,
        bridge_node,
        rviz_node
    ])
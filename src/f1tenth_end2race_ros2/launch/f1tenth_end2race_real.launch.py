import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_name = 'f1tenth_end2race_ros2'
    end2race_pkg = get_package_share_directory(pkg_name)
    description_pkg = get_package_share_directory('racecar_description')

    # ============================================================
    # Launch 인자 정의
    # ============================================================
    namespace_arg = DeclareLaunchArgument(
        'namespace', default_value='car1',
        description='로봇 네임스페이스 (agent_node의 robot_name으로 주입되어 토픽 경로 결정)'
    )
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(end2race_pkg, 'models', 'end2race.pth'),
        description='end2race 모델 가중치(.pth) 경로'
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description='원격 모니터링용 RViz2 실행 여부 (실차에선 보통 false, 개발 PC에서 볼 때 true)'
    )

    namespace = LaunchConfiguration('namespace')
    model_path = LaunchConfiguration('model_path')
    use_rviz = LaunchConfiguration('use_rviz')

    # ============================================================
    # 노드 정의
    # ============================================================
    # Agent 노드 (실차) — scan 구독 → RNN 추론 → drive 발행 (반응형, odom/map 불필요)
    # config 뒤 dict가 우선이라 model_path/robot_name이 config 값을 덮어씀.
    #
    # 실차 하드웨어 스택(LiDAR 드라이버, VESC)은 네임스페이스 없이 /scan, /drive를
    # 사용하므로, agent_node가 robot_name으로 만드는 /{namespace}/scan,
    # /{namespace}/drive 절대경로를 /scan, /drive로 remap한다.
    config_path = os.path.join(end2race_pkg, 'config', 'end2race_config.yaml')
    agent_node = Node(
        package=pkg_name,
        executable='agent_node',
        name='end2race_agent',
        namespace=namespace,
        output='screen',
        parameters=[config_path, {
            'use_sim_time': False,
            'model_path': model_path,
            'robot_name': namespace,
        }],
        remappings=[
            (['/', namespace, '/scan'], '/scan'),
            (['/', namespace, '/drive'], '/drive'),
        ]
    )

    # RViz2 — 원격 개발 PC 모니터링용 (실차 기본 비활성).
    # TF/scan은 하드웨어 드라이버(VESC, LiDAR) 스택이 제공한다고 가정.
    # console_bridge의 tf2 경고가 stderr를 도배하므로 셸에서 stderr를 버림.
    _rviz_cfg = os.path.join(description_pkg, 'rviz', 'f1tenth_default.rviz')
    rviz_node = ExecuteProcess(
        cmd=['bash', '-c', f'exec rviz2 -d "{_rviz_cfg}" 2>/dev/null'],
        output='log',
        condition=IfCondition(use_rviz)
    )

    return LaunchDescription([
        namespace_arg,
        model_path_arg,
        use_rviz_arg,
        agent_node,
        rviz_node,
    ])

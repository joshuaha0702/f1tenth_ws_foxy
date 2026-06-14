#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bag의 /odom을 Gazebo 엔티티 위치로 매핑하는 브리지.

Gazebo를 물리 시뮬레이터가 아니라 3D 뷰어로 쓰기 위한 노드.
시작 시 physics를 pause하고, 이후 bag play가 흘려보내는 /{car}/odom 마다
/gazebo/set_entity_state를 호출해 차량을 그 위치로 텔레포트함.

런치 예:
    ros2 launch f1tenth_lattice_ros2 gazebo_replay.launch.py head2head:=true
    # 다른 터미널에서:
    ros2 bag play <bag_dir> --clock
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty as EmptySrv


class Bag2GazeboNode(Node):
    def __init__(self):
        super().__init__('bag2gazebo')

        self.declare_parameter('cars', ['car1', 'car2'])
        self.declare_parameter('pause_on_start', True)
        self.declare_parameter('gui_update_rate', 30.0)

        cars = list(self.get_parameter('cars').value)
        pause_on_start = bool(self.get_parameter('pause_on_start').value)
        gui_update_rate = float(self.get_parameter('gui_update_rate').value)

        self.set_state_cli = self.create_client(
            SetEntityState, '/gazebo/set_entity_state'
        )
        self.pause_cli = self.create_client(EmptySrv, '/pause_physics')

        self.get_logger().info('Waiting for /gazebo/set_entity_state...')
        if not self.set_state_cli.wait_for_service(timeout_sec=30.0):
            raise RuntimeError('Gazebo set_entity_state service not available')

        if pause_on_start:
            self.get_logger().info('Pausing Gazebo physics (visual-only replay mode)...')
            if self.pause_cli.wait_for_service(timeout_sec=5.0):
                future = self.pause_cli.call_async(EmptySrv.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
                self.get_logger().info('Physics paused.')
            else:
                self.get_logger().warn(
                    '/pause_physics not available — 차량이 자유낙하 or drift 할 수 있음'
                )

        # bag play의 /odom QoS는 일반적으로 reliable + depth 10
        qos = QoSProfile(depth=20)
        qos.reliability = ReliabilityPolicy.RELIABLE

        # bag의 odom rate(보통 100Hz)대로 매번 set_entity_state를 호출하면
        # gzserver의 (단일 스레드) ROS 서비스 핸들러가 밀려 GUI가 버벅임.
        # 콜백에서는 최신 pose만 캐싱하고, 별도 타이머에서 낮은 주기로만 텔레포트함.
        self.latest_pose = {car: None for car in cars}

        for car in cars:
            topic = f'/{car}/odom'
            self.create_subscription(
                Odometry, topic,
                lambda msg, name=car: self._on_odom(name, msg),
                qos,
            )
            self.get_logger().info(f'Subscribed: {topic} -> set_entity_state({car})')

        self.create_timer(1.0 / gui_update_rate, self._publish_states)

    def _on_odom(self, name: str, msg: Odometry):
        self.latest_pose[name] = msg.pose.pose

    def _publish_states(self):
        for name, pose in self.latest_pose.items():
            if pose is None:
                continue
            req = SetEntityState.Request()
            req.state.name = name
            req.state.pose = pose
            # twist는 0으로 둠 (paused 상태라 무관하지만, 만약 unpause 되더라도 안전)
            req.state.twist.linear.x = 0.0
            req.state.twist.linear.y = 0.0
            req.state.twist.linear.z = 0.0
            req.state.twist.angular.x = 0.0
            req.state.twist.angular.y = 0.0
            req.state.twist.angular.z = 0.0
            req.state.reference_frame = 'world'
            # fire-and-forget — 응답 안 기다림 (실시간성 우선)
            self.set_state_cli.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Bag2GazeboNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None:
            node.get_logger().error(f'bag2gazebo crashed: {e}')
        else:
            print(f'bag2gazebo failed to start: {e}')
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

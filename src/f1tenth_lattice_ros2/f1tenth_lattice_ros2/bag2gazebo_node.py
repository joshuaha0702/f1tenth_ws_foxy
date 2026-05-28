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

        cars = list(self.get_parameter('cars').value)
        pause_on_start = bool(self.get_parameter('pause_on_start').value)

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

        for car in cars:
            topic = f'/{car}/odom'
            self.create_subscription(
                Odometry, topic,
                lambda msg, name=car: self._on_odom(name, msg),
                qos,
            )
            self.get_logger().info(f'Subscribed: {topic} -> set_entity_state({car})')

    def _on_odom(self, name: str, msg: Odometry):
        req = SetEntityState.Request()
        req.state.name = name
        req.state.pose = msg.pose.pose
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

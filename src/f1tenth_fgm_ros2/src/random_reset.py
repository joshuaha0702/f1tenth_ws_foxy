#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import Empty
import random
import math

class RandomReset(Node):
    def __init__(self):
        super().__init__('random_reset')

        # 리셋 신호를 보낼 퍼블리셔 추가
        self.reset_pub = self.create_publisher(Empty, '/map_reset', 10)
        
        # 가제보 상태 설정 서비스 클라이언트 생성
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('가제보 서비스 대기 중...')

    def reset_car(self):
        # --- 랜덤 범위 설정 (내 맵에 맞춰 수정 필요!) ---
        # 예: x는 -5~5m, y는 -3~3m 사이
        random_x = random.uniform(-2.0, 2.0)
        random_y = random.uniform(-1.0, 1.0)
        # 방향(Yaw)도 랜덤하게 (0 ~ 360도)
        random_yaw = random.uniform(0, 2 * math.pi)

        # Quaternion 변환 (간단한 Yaw -> Quat)
        qz = math.sin(random_yaw / 2.0)
        qw = math.cos(random_yaw / 2.0)

        # 요청 메시지 작성
        request = SetEntityState.Request()
        request.state.name = 'racecar' # 내 차의 가제보 이름 (보통 ego_racecar)
        request.state.pose.position.x = random_x
        request.state.pose.position.y = random_y
        request.state.pose.position.z = 0.05
        request.state.pose.orientation.z = qz
        request.state.pose.orientation.w = qw
        
        # 서비스 호출
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info('소환 성공! 신호를 보냅니다.')
            self.reset_pub.publish(Empty()) # 로거에게 알림!
        else:
            self.get_logger().error('소환 실패!')

def main():
    rclpy.init()
    node = RandomReset()
    node.reset_car() # 실행 시 한 번 리셋
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
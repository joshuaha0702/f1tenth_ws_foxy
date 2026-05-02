#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Point # Empty 대신 Point 임포트
import random
import math

class RandomReset(Node):
    def __init__(self):
        super().__init__('random_reset')

        # [추가됨] 로봇 이름 파라미터 선언 (기본값: car1)
        self.declare_parameter('robot_name', 'car1')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value

        # [수정됨] 토픽 이름 동적 생성 (DataLogger와 연동)
        reset_topic = f'/{self.robot_name}/map_reset'

        # 리셋 신호를 보낼 퍼블리셔 추가
        self.reset_pub = self.create_publisher(Point, reset_topic, 10)
        
        # 가제보 상태 설정 서비스 클라이언트 생성
        self.client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('가제보 서비스 대기 중...')

        # --- 안전 구역(Safe Zones) 리스트 정의 ---
        # simple 맵(STL)에서 차량이 스폰되어도 벽과 충돌하지 않는 사각형 구역들을 정의합니다.
        # 실제 맵 좌표에 맞게 x_min, x_max, y_min, y_max 값을 수정 및 추가하세요.
        self.safe_zones = [
            {'x_min': 2.0, 'x_max': 6.0, 'y_min': 1.8,  'y_max': 3.0},   # 안전 구역 1 (하단)
            {'x_min': 1.0, 'x_max': 3.0, 'y_min': 9.4,  'y_max': 15.4},  # 안전 구역 2 (좌측 중간)
            {'x_min': 6.0, 'x_max': 8.0, 'y_min': 9.4,  'y_max': 15.4},  # 안전 구역 3 (우측 중간)
            {'x_min': 2.0, 'x_max': 6.0, 'y_min': 21.8, 'y_max': 23.0}   # 안전 구역 4 (상단)
        ]

    def reset_car(self):
        # 정의된 안전 구역 중 하나를 무작위로 선택
        selected_zone = random.choice(self.safe_zones)

        # 선택된 구역 내에서 무작위 x, y 좌표 생성
        random_x = random.uniform(selected_zone['x_min'], selected_zone['x_max'])
        random_y = random.uniform(selected_zone['y_min'], selected_zone['y_max'])
        
        # 방향(Yaw)도 랜덤하게 (-45 ~ 45도)
        random_yaw = random.uniform(-math.pi/4, math.pi/4)

        # Quaternion 변환 (간단한 Yaw -> Quat)
        qz = math.sin(random_yaw / 2.0)
        qw = math.cos(random_yaw / 2.0)

        # 요청 메시지 작성
        request = SetEntityState.Request()
        
        # [수정됨] 가제보 내 객체 이름도 파라미터로 받은 로봇 이름과 일치시킵니다.
        # 기존의 고정된 'racecar' 대신 self.robot_name 사용
        request.state.name = self.robot_name

        request.state.pose.position.x = random_x
        request.state.pose.position.y = random_y
        request.state.pose.position.z = 0.05
        request.state.pose.orientation.z = qz
        request.state.pose.orientation.w = qw
        
        # 서비스 호출
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f'소환 성공! [위치: X={random_x:.2f}, Y={random_y:.2f}] 신호를 보냅니다.')
            msg = Point()
            msg.x = random_x
            msg.y = random_y
            msg.z = 0.0
            self.reset_pub.publish(msg)
        else:
            self.get_logger().error('소환 실패!')

def main(args=None):
    rclpy.init(args=args)
    node = RandomReset()
    node.reset_car() # 실행 시 한 번 리셋
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
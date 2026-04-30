#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np
import csv
import os
import math
from datetime import datetime, timezone, timedelta

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point # Empty 대신 Point 임포트
from geometry_msgs.msg import Twist
import nav_msgs.msg

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')

        # --- 파라미터 선언 ---
        # 기본값은 20Hz로 설정 (원하는 대로 10, 20 등으로 변경 가능)
        self.declare_parameter('target_hz', 20.0)
        self.declare_parameter('scan_downsample_factor', 360)

        self.target_hz = self.get_parameter('target_hz').get_parameter_value().double_value
        self.scan_downsample_factor = self.get_parameter('scan_downsample_factor').get_parameter_value().integer_value
        
        self.max_count = 40.0 / self.target_hz  # 저장 간격 (초)
        self.last_count = 1.0

        self.scan_factor = 1080 //self.scan_downsample_factor

        # 저장할 폴더 생성
        self.save_path = os.path.expanduser('/root/f1tenth_data')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        

        self.kst = timezone(timedelta(hours=9))
        now_kst = datetime.now(self.kst)
        
        # 파일명 결정 (현재 시간 기준)
        filename = now_kst.strftime('driving_data_%Y%m%d_%H%M%S.csv')
        self.csv_file = open(os.path.join(self.save_path, filename), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # 최신 조종 값을 저장할 변수
        self.current_speed = 0.0
        self.current_steering = 0.0

        # 랩 카운트와 결승선 통과 여부
        self.lap_count = 0
        self.passed_finish_line = False
        self.odom_sub = self.create_subscription(nav_msgs.msg.Odometry, '/odom', self.odom_callback, 10)
        
        # 구독자 설정
        self.reset_sub = self.create_subscription(Point, '/map_reset', self.reset_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_sub = self.create_subscription(Twist, '/drive', self.drive_callback, 10)
        
        # 리셋 신호 구독
        self.is_resetting = False # 리셋 직후 데이터를 거르기 위한 플래그

        # --- 추가된 부분: 스폰 위치와 출발 상태 플래그 ---
        self.start_x = 0.0
        self.start_y = 0.0
        self.left_start_zone = False # 차가 출발지점을 벗어났는지 확인
        
        # CSV 헤더 작성 (Timestamp, Speed, Steering, Lidar_0 ... Lidar_1080)
        header = ['lab', 'timestamp', 'speed', 'steering'] + [f'scan_{i}' for i in range(self.scan_downsample_factor+1)]
        self.csv_writer.writerow(header)
        
        self.get_logger().info(f'로깅 시작: {filename}')
        self.get_logger().info(f'설정: {self.target_hz}Hz, (빔 개수: {self.scan_downsample_factor})')

    def reset_callback(self, msg):
        self.lap_count += 1
        self.is_resetting = True # 플래그 On
        # 전달받은 스폰 위치를 새로운 출발/결승선으로 등록
        self.start_x = msg.x
        self.start_y = msg.y
        self.left_start_zone = False # 초기화 직후에는 아직 출발 전
        
        self.get_logger().info(f'리셋 수신! 새 결승선 [X={self.start_x:.2f}, Y={self.start_y:.2f}] 설정 완료 (Lap: {self.lap_count})')
        
        self.reset_timer = self.create_timer(0.5, self.end_reset_period)

    def end_reset_period(self):
        self.is_resetting = False
        self.reset_timer.cancel()

    def drive_callback(self, msg):
        # 조이스틱에서 들어오는 최신 조종 값을 저장
        self.current_speed = msg.linear.x
        self.current_steering = msg.angular.z

    def scan_callback(self, msg):
        # 리셋 중이면 데이터를 저장하지 않고 건너뜀
        if self.is_resetting:
            return

        # 현재 메시지의 타임스탬프 (초 단위)
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # --- 다운스케일링 로직 ---
        # 마지막 저장 시간으로부터 log_interval(예: 0.1초) 이상 지났는지 확인
        if self.last_count < self.max_count:
            self.last_count += 1.0
            return  # 설정한 주기가 안 되었으면 저장하지 않고 무시
        
        self.last_count = 1.0  # 저장 후 카운트 초기화
        
        # 라이다 데이터가 들어올 때마다 현재 조종 값과 매칭하여 저장
        # -inf나 inf 값 처리 (최대 거리 10m로 제한)
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 0.0, ranges)

        # 다운스케일링
       
        reduced_ranges = ranges[::self.scan_factor]
        
        # 데이터 한 줄 만들기
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        row = [self.lap_count, timestamp, self.current_speed, self.current_steering] + reduced_ranges.tolist()
        
        # CSV에 쓰기
        self.csv_writer.writerow(row)
    
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 현재 위치와 새 결승선(스폰 위치) 사이의 거리 계산
        distance = math.sqrt((x - self.start_x)**2 + (y - self.start_y)**2)

        # 1. 차가 결승선에서 2.0m 이상 멀어지면 '제대로 출발했다'고 판단
        if not self.left_start_zone and distance > 2.0:
            self.left_start_zone = True
            self.get_logger().info('출발지점을 벗어났습니다. 랩 측정을 시작합니다.')

        # 2. 이미 출발한 상태에서 다시 결승선 반경 1.5m 이내로 들어오면 한 바퀴 완료!
        if self.left_start_zone and distance < 1.5:
            self.lap_count += 1
            self.left_start_zone = False # 다음 랩을 위해 다시 False로 초기화
            self.get_logger().info(f'한 바퀴 완료! 현재 Lap: {self.lap_count}')

    def __del__(self):
        self.csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('데이터 로깅 종료 중...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
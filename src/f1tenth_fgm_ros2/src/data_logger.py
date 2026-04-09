#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import numpy as np
import csv
import os
from datetime import datetime, timezone, timedelta

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import nav_msgs.msg

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')
        
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
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.drive_sub = self.create_subscription(Twist, '/drive', self.drive_callback, 10)
        
        # CSV 헤더 작성 (Timestamp, Speed, Steering, Lidar_0 ... Lidar_1080)
        header = ['lab', 'timestamp', 'speed', 'steering'] + [f'scan_{i}' for i in range(1081)]
        self.csv_writer.writerow(header)
        
        self.get_logger().info(f'데이터 로깅 시작: {filename}')

    def drive_callback(self, msg):
        # 조이스틱에서 들어오는 최신 조종 값을 저장
        self.current_speed = msg.linear.x
        self.current_steering = msg.angular.z

    def scan_callback(self, msg):
        # 라이다 데이터가 들어올 때마다 현재 조종 값과 매칭하여 저장
        # -inf나 inf 값 처리 (최대 거리 10m로 제한)
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 0.0, ranges)
        
        # 데이터 한 줄 만들기
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        row = [self.lap_count, timestamp, self.current_speed, self.current_steering] + ranges.tolist()
        
        # CSV에 쓰기
        self.csv_writer.writerow(row)
    
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # 예: x가 0 근처이고 y가 특정 범위일 때 결승선 통과로 간주
        if -0.5 < x < 0.5 and -1.0 < y < 1.0:
            if not self.passed_finish_line:
                self.lap_count += 1
                self.passed_finish_line = True
                self.get_logger().info(f'자동 Lap 감지! 현재 Lap: {self.lap_count}')
        else:
            self.passed_finish_line = False

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
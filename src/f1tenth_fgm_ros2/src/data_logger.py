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
from geometry_msgs.msg import Point
from geometry_msgs.msg import TwistStamped
import nav_msgs.msg

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')

        # --- 파라미터 선언 ---
        self.declare_parameter('robot_name', 'car1')
        self.declare_parameter('target_hz', 20.0)
        self.declare_parameter('scan_downsample_factor', 360)
        self.declare_parameter('save_dir', '/root/f1tenth_data')

        robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.target_hz = self.get_parameter('target_hz').get_parameter_value().double_value
        self.scan_downsample_factor = self.get_parameter('scan_downsample_factor').get_parameter_value().integer_value
        save_dir = self.get_parameter('save_dir').get_parameter_value().string_value

        self.max_count = 40.0 / self.target_hz 
        self.last_count = 1.0
        self.scan_factor = 1080 // self.scan_downsample_factor

        # 저장 폴더 및 파일 설정
        self.save_path = os.path.expanduser(save_dir)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.kst = timezone(timedelta(hours=9))
        now_kst = datetime.now(self.kst)
        filename = now_kst.strftime(f'{robot_name}_driving_data_%Y%m%d_%H%M%S.csv')
        self.csv_file = open(os.path.join(self.save_path, filename), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # 랩 카운트 초기화
        self.lap_count = 0
        self.scan_buffer = {}

        # 토픽 네임스페이스 설정
        odom_topic = f'/{robot_name}/odom'
        reset_topic = f'/{robot_name}/map_reset'
        scan_topic = f'/{robot_name}/scan'
        drive_topic = f'/{robot_name}/drive_stamped'

        # 구독자 설정
        self.odom_sub = self.create_subscription(nav_msgs.msg.Odometry, odom_topic, self.odom_callback, 10)
        self.reset_sub = self.create_subscription(Point, reset_topic, self.reset_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.drive_sub = self.create_subscription(TwistStamped, drive_topic, self.drive_callback, 10)

        self.is_resetting = False 
        self.start_x = 0.0
        self.start_y = 0.0
        self.left_start_zone = False 

        # [복구됨] CSV 헤더에 'lap' 추가
        header = ['lap', 'time', 'steer', 'desired_speed'] + [f'lidar_{i}' for i in range(self.scan_downsample_factor)]
        self.csv_writer.writerow(header)

        self.get_logger().info(f'로깅 시작: {filename} (로봇: {robot_name})')

    def reset_callback(self, msg):
        # [핵심] 리셋 신호가 오면 랩 카운트를 올리고 새로운 기준점을 잡습니다.
        self.lap_count += 1
        self.is_resetting = True 
        self.start_x = msg.x
        self.start_y = msg.y
        self.left_start_zone = False 

        self.get_logger().info(f'리셋 감지! 신규 랩 시작 (Lap: {self.lap_count})')
        self.reset_timer = self.create_timer(0.5, self.end_reset_period)

    def end_reset_period(self):
        self.is_resetting = False
        self.reset_timer.cancel()

    def drive_callback(self, msg):
        drive_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if drive_stamp not in self.scan_buffer:
            return  

        timestamp, reduced_ranges = self.scan_buffer.pop(drive_stamp)
        
        # [복구됨] CSV 행 데이터 맨 앞에 현재 lap_count를 넣습니다.
        row = [self.lap_count, timestamp, msg.twist.angular.z, msg.twist.linear.x] + reduced_ranges
        self.csv_writer.writerow(row)

        # 버퍼 정리
        stale_keys = [k for k in self.scan_buffer if k < drive_stamp]
        for k in stale_keys:
            self.scan_buffer.pop(k)

    def scan_callback(self, msg):
        if self.is_resetting:
            return

        if self.last_count < self.max_count:
            self.last_count += 1.0
            return  

        self.last_count = 1.0  

        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 0.0, ranges)

        reduced_ranges = []
        reduced_ranges.append(min(ranges[0:self.scan_factor//2+1]))
        for i in range(1, self.scan_downsample_factor-1):
            reduced_ranges.append(min(ranges[i*self.scan_factor - self.scan_factor//2 : i*self.scan_factor + self.scan_factor//2 + 1]))
        reduced_ranges.append(min(ranges[-self.scan_factor//2-1:]))

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.scan_buffer[timestamp] = (timestamp, reduced_ranges)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        distance = math.sqrt((x - self.start_x)**2 + (y - self.start_y)**2)

        if not self.left_start_zone and distance > 2.0:
            self.left_start_zone = True
            self.get_logger().info('출발지점 이탈 확인. 랩 측정을 진행합니다.')

        if self.left_start_zone and distance < 1.5:
            self.lap_count += 1
            self.left_start_zone = False 
            self.get_logger().info(f'결승선 통과! 현재 Lap: {self.lap_count}')

    def __del__(self):
        self.csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('로깅을 중단합니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import glob
import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

def extract_bag_to_csv(bag_dir_path, save_dir='/root/f1tenth_ws/f1tenth_data', robot_name='car1'):
    scan_downsample_factor = 360
    scan_factor = 1080 // scan_downsample_factor

    # 저장 폴더 생성
    save_path = os.path.expanduser(save_dir)
    os.makedirs(save_path, exist_ok=True)

    # 출력 CSV 파일 설정
    bag_name = os.path.basename(os.path.normpath(bag_dir_path))
    filename = f'{robot_name}_extracted_{bag_name}.csv'
    csv_file_path = os.path.join(save_path, filename)
    
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # 헤더 작성
    header = ['time', 'steer', 'desired_speed'] + [f'lidar_{i}' for i in range(scan_downsample_factor)]
    csv_writer.writerow(header)

    # 비동기 오차 해결의 핵심: 항상 가장 최근에 들어온 스캔 배열을 들고 대기함
    latest_scan_ranges = None
    
    scan_topic = f'/{robot_name}/scan'
    drive_topic = f'/{robot_name}/drive_stamped'

    # ROS 2 Foxy 버전에 맞는 typestore
    typestore = get_typestore(Stores.ROS2_FOXY)

    print(f"데이터 추출 시작... (대상: {bag_name})")
    count_written = 0

    try:
        with Reader(bag_dir_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic not in [scan_topic, drive_topic]:
                    continue
                
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                # 1. Scan 처리 (Hz 건너뛰기 없이 들어오는 족족 최신 상태로 업데이트)
                if connection.topic == scan_topic:
                    ranges = np.array(msg.ranges)
                    ranges = np.where(np.isinf(ranges), 10.0, ranges)
                    ranges = np.where(np.isnan(ranges), 0.0, ranges)

                    reduced_ranges = []
                    reduced_ranges.append(float(np.min(ranges[0:scan_factor//2+1])))
                    for i in range(1, scan_downsample_factor-1):
                        reduced_ranges.append(float(np.min(ranges[i*scan_factor - scan_factor//2 : i*scan_factor + scan_factor//2 + 1])))
                    reduced_ranges.append(float(np.min(ranges[-scan_factor//2-1:])))

                    # 스캔 상태 최신화
                    latest_scan_ranges = reduced_ranges

                # 2. Drive 처리 및 매칭 (Drive 신호가 오면 무조건 현재 들고 있는 최신 Scan과 묶어서 CSV에 작성)
                elif connection.topic == drive_topic:
                    if latest_scan_ranges is not None:
                        # 헤더 시간이 꼬여있을 가능성에 대비하여 확실한 '로스백 기록 시간'을 사용함
                        drive_time_sec = timestamp * 1e-9
                        
                        row = [drive_time_sec, msg.twist.angular.z, msg.twist.linear.x] + latest_scan_ranges
                        csv_writer.writerow(row)
                        count_written += 1
                        
                        # 다음 Drive 명령 전까지 같은 Scan이 중복 사용되는 것을 막기 위해 비워둠
                        latest_scan_ranges = None

    except Exception as e:
        print(f"Bag 파일을 읽는 중 오류 발생: {e}")
        return

    csv_file.close()
    print(f"추출 완료! 총 {count_written}개의 행이 기록됨.")
    print(f"저장 위치: {csv_file_path}")

if __name__ == '__main__':
    # bags 폴더가 위치한 기본 경로 지정
    base_bags_dir = '/root/f1tenth_ws/bags'
    
    # 'output_'으로 시작하는 모든 폴더 경로 탐색 및 시간순 정렬
    search_pattern = os.path.join(base_bags_dir, 'output_*')
    target_folders = sorted(glob.glob(search_pattern))
    
    if not target_folders:
        print(f"지정된 경로에 처리할 대상 폴더가 없음: {search_pattern}")
    else:
        print(f"총 {len(target_folders)}개의 bag 폴더를 발견함. 일괄 추출 처리를 시작함.")
        
        for folder in target_folders:
            # 해당 경로가 실제 디렉토리인 경우에만 처리
            if os.path.isdir(folder):
                print("=" * 60)
                extract_bag_to_csv(folder)
                
        print("=" * 60)
        print("모든 bag 파일의 데이터 추출이 완료되었음.")
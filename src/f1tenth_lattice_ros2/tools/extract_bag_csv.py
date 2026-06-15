#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import glob
import bisect
import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


# rosbags 기본 typestore에는 ackermann_msgs가 없어서 직접 등록해야 함
ACKERMANN_DRIVE_MSG = """
float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""

ACKERMANN_DRIVE_STAMPED_MSG = """
std_msgs/Header header
ackermann_msgs/AckermannDrive drive
"""


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def extract_bag_to_csv(bag_dir_path, save_dir='/root/f1tenth_ws/f1tenth_data', robot_name='car1'):
    scan_downsample_factor = 360
    scan_factor = 1080 // scan_downsample_factor

    save_path = os.path.expanduser(save_dir)
    os.makedirs(save_path, exist_ok=True)

    bag_name = os.path.basename(os.path.normpath(bag_dir_path))
    filename = f'{robot_name}_extracted_{bag_name}.csv'
    csv_file_path = os.path.join(save_path, filename)

    scan_topic = f'/{robot_name}/scan'
    drive_topic = f'/{robot_name}/drive'

    typestore = get_typestore(Stores.ROS2_FOXY)
    typestore.register(get_types_from_msg(ACKERMANN_DRIVE_MSG, 'ackermann_msgs/msg/AckermannDrive'))
    typestore.register(get_types_from_msg(ACKERMANN_DRIVE_STAMPED_MSG, 'ackermann_msgs/msg/AckermannDriveStamped'))

    # Pass 1: 두 토픽을 헤더 stamp와 함께 전부 수집함
    # (rosbag 기록 시간 대신 publisher가 채운 header.stamp를 정합 기준으로 사용)
    scans = []   # (stamp_ns, reduced_ranges)
    drives = []  # (stamp_ns, steer, desired_speed)

    print(f"데이터 추출 시작... (대상: {bag_name})")

    try:
        with Reader(bag_dir_path) as reader:
            for connection, _bag_ts, rawdata in reader.messages():
                if connection.topic not in (scan_topic, drive_topic):
                    continue

                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                stamp_ns = stamp_to_ns(msg.header.stamp)

                if connection.topic == scan_topic:
                    ranges = np.array(msg.ranges)
                    ranges = np.where(np.isinf(ranges), 30.0, ranges)
                    ranges = np.where(np.isnan(ranges), 0.0, ranges)

                    reduced_ranges = []
                    reduced_ranges.append(float(np.min(ranges[ : scan_factor//2])))
                    for i in range(1, scan_downsample_factor - 1):
                        reduced_ranges.append(float(np.min(ranges[(i-1)*scan_factor : (i+1)*scan_factor])))
                    reduced_ranges.append(float(np.min(ranges[-(scan_factor//2) : ])))
                    scans.append((stamp_ns, reduced_ranges))

                elif connection.topic == drive_topic:
                    drives.append((
                        stamp_ns,
                        float(msg.drive.steering_angle),
                        float(msg.drive.speed),
                    ))

    except Exception as e:
        print(f"Bag 파일을 읽는 중 오류 발생: {e}")
        return

    # rosbag 기록 순서가 헤더 stamp 순서와 어긋날 수 있어서 명시적으로 정렬함
    scans.sort(key=lambda r: r[0])
    drives.sort(key=lambda r: r[0])

    scan_stamps = [s[0] for s in scans]

    # Pass 2: 각 drive에 대해 stamp <= drive_stamp인 가장 최근 scan을 인과적으로 매칭함
    # (drive보다 미래의 센서값은 절대로 사용하지 않음. 동일 stamp면 그걸 채택)
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    header = (
        ['time', 'steer', 'desired_speed', 'lidar_delay']
        + [f'lidar_{i}' for i in range(scan_downsample_factor)]
    )
    csv_writer.writerow(header)

    count_written = 0
    skipped_no_match = 0

    for drive_stamp, steer, desired_speed in drives:
        # bisect_right - 1 : drive_stamp 이하 중 가장 마지막 인덱스 (동일 stamp 존재시 그걸 선택)
        s_idx = bisect.bisect_right(scan_stamps, drive_stamp) - 1

        if s_idx < 0:
            # drive 시점 이전에 아직 도착한 scan이 없는 경우 스킵 (인과성 보장)
            skipped_no_match += 1
            continue

        scan_stamp, ranges_row = scans[s_idx]
        lidar_delay = (drive_stamp - scan_stamp) * 1e-9

        row = [drive_stamp * 1e-9, steer, desired_speed, lidar_delay] + ranges_row
        csv_writer.writerow(row)
        count_written += 1

    csv_file.close()
    print(f"추출 완료! 총 {count_written}개의 행이 기록됨. (인과적 매칭 실패로 스킵: {skipped_no_match})")
    print(f"저장 위치: {csv_file_path}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("사용법: extract_bag_csv.py <data 하위 폴더 이름> (예: 0614)")
        sys.exit(1)

    folder_name = sys.argv[1]

    # data/<folder_name>/clean 안의 모든 rosbag을 data/<folder_name>/csv 에 추출
    base_data_dir = '/root/f1tenth_ws/data'
    target_dir = os.path.join(base_data_dir, folder_name)
    clean_dir = os.path.join(target_dir, 'clean')
    save_dir = os.path.join(target_dir, 'csv')

    if not os.path.isdir(clean_dir):
        print(f"clean 폴더를 찾을 수 없음: {clean_dir}")
        sys.exit(1)

    # clean 폴더 내 모든 bag 폴더 탐색 및 이름순 정렬
    target_folders = sorted(glob.glob(os.path.join(clean_dir, '*')))
    target_folders = [f for f in target_folders if os.path.isdir(f)]

    if not target_folders:
        print(f"지정된 경로에 처리할 대상 폴더가 없음: {clean_dir}")
    else:
        print(f"총 {len(target_folders)}개의 bag 폴더를 발견함. 일괄 추출 처리를 시작함.")

        for folder in target_folders:
            print("=" * 60)
            extract_bag_to_csv(folder, save_dir=save_dir)

        print("=" * 60)
        print(f"모든 bag 파일의 데이터 추출이 완료되었음. 저장 위치: {save_dir}")

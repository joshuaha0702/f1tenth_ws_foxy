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
    clock_topic = '/clock'

    typestore = get_typestore(Stores.ROS2_FOXY)
    typestore.register(get_types_from_msg(ACKERMANN_DRIVE_MSG, 'ackermann_msgs/msg/AckermannDrive'))
    typestore.register(get_types_from_msg(ACKERMANN_DRIVE_STAMPED_MSG, 'ackermann_msgs/msg/AckermannDriveStamped'))

    # Pass 1: scan/drive는 bag 기록 시각(real wall-clock, 나노초)으로 수집하고,
    # /clock은 (bag 기록 시각 -> 그 순간의 sim-time) 앵커로 따로 모음.
    #   - bag_ts: 나노초 해상도라 같은 /clock 틱 안에서도 scan-drive 순서/매칭을 정확히 구분 가능
    #   - /clock 앵커: wall->sim 매핑을 만들어, bag_ts를 고해상도 sim-time으로 환산함
    #     (header.stamp는 /clock 10Hz로 양자화되어 있고, 중복 퍼블리셔 오염도 있어 사용하지 않음)
    scans = []          # (bag_ns, reduced_ranges)
    drives = []         # (bag_ns, steer, desired_speed)
    clock_wall = []     # bag 기록 시각(ns)
    clock_sim = []      # 그 순간의 sim-time(ns)

    print(f"데이터 추출 시작... (대상: {bag_name})")

    try:
        with Reader(bag_dir_path) as reader:
            for connection, bag_ts, rawdata in reader.messages():
                if connection.topic not in (scan_topic, drive_topic, clock_topic):
                    continue

                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                if connection.topic == clock_topic:
                    clock_wall.append(bag_ts)
                    clock_sim.append(stamp_to_ns(msg.clock))

                elif connection.topic == scan_topic:
                    ranges = np.array(msg.ranges)
                    ranges = np.where(np.isinf(ranges), 30.0, ranges)
                    ranges = np.where(np.isnan(ranges), 0.0, ranges)

                    reduced_ranges = []
                    reduced_ranges.append(float(np.min(ranges[ : scan_factor//2])))
                    for i in range(1, scan_downsample_factor - 1):
                        reduced_ranges.append(float(np.min(ranges[(i-1)*scan_factor : (i+1)*scan_factor])))
                    reduced_ranges.append(float(np.min(ranges[-(scan_factor//2) : ])))
                    scans.append((bag_ts, reduced_ranges))

                elif connection.topic == drive_topic:
                    drives.append((
                        bag_ts,
                        float(msg.drive.steering_angle),
                        float(msg.drive.speed),
                    ))

    except Exception as e:
        print(f"Bag 파일을 읽는 중 오류 발생: {e}")
        return

    if not clock_wall:
        print("경고: /clock 토픽이 없어 wall->sim 환산을 할 수 없음. bag을 확인하세요.")
        return

    # wall(ns) -> sim(ns) 선형보간 매핑. /clock 앵커 사이의 기울기 = 그 구간의 국소 RTF.
    # 따라서 RTF가 시간에 따라 출렁여도 자동으로 보정됨. (구간 밖은 np.interp가 양 끝값으로 클램프)
    clock_wall = np.array(clock_wall, dtype=np.float64)
    clock_sim = np.array(clock_sim, dtype=np.float64)
    order = np.argsort(clock_wall)
    clock_wall = clock_wall[order]
    clock_sim = clock_sim[order]

    def wall_to_sim_ns(wall_ns):
        return np.interp(wall_ns, clock_wall, clock_sim)

    # 인과 매칭은 bag 기록 시각(나노초) 기준으로 정렬/탐색함
    scans.sort(key=lambda r: r[0])
    drives.sort(key=lambda r: r[0])

    scan_bag_stamps = [s[0] for s in scans]

    # Pass 2: 각 drive에 대해 bag_ns <= drive_bag_ns인 가장 최근 scan을 인과적으로 매칭함
    # (drive보다 미래의 센서값은 절대로 사용하지 않음)
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    header = (
        ['time', 'steer', 'desired_speed', 'lidar_delay']
        + [f'lidar_{i}' for i in range(scan_downsample_factor)]
    )
    csv_writer.writerow(header)

    count_written = 0
    skipped_no_match = 0

    for drive_bag_ns, steer, desired_speed in drives:
        # bisect_right - 1 : drive_bag_ns 이하 중 가장 마지막 인덱스
        s_idx = bisect.bisect_right(scan_bag_stamps, drive_bag_ns) - 1

        if s_idx < 0:
            # drive 시점 이전에 아직 도착한 scan이 없는 경우 스킵 (인과성 보장)
            skipped_no_match += 1
            continue

        scan_bag_ns, ranges_row = scans[s_idx]

        # bag_ts(wall)를 /clock 보간으로 고해상도 sim-time으로 환산
        drive_sim_ns = wall_to_sim_ns(drive_bag_ns)
        scan_sim_ns = wall_to_sim_ns(scan_bag_ns)

        # lidar_delay: RTF 보정된 sim-time 기준 scan-drive 지연 (실제 로봇이 겪을 지연과 일치)
        lidar_delay = (drive_sim_ns - scan_sim_ns) * 1e-9

        # time 컬럼: drive의 고해상도 sim-time
        row = [drive_sim_ns * 1e-9, steer, desired_speed, lidar_delay] + ranges_row
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

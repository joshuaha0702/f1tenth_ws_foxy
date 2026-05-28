#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import glob
import bisect
import math
import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def quat_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def extract_bag_to_csv(bag_dir_path, save_dir='/root/f1tenth_ws/f1tenth_data', robot_name='car1'):
    scan_downsample_factor = 360
    scan_factor = 1080 // scan_downsample_factor

    save_path = os.path.expanduser(save_dir)
    os.makedirs(save_path, exist_ok=True)

    bag_name = os.path.basename(os.path.normpath(bag_dir_path))
    filename = f'{robot_name}_extracted_{bag_name}.csv'
    csv_file_path = os.path.join(save_path, filename)

    scan_topic = f'/{robot_name}/scan'
    drive_topic = f'/{robot_name}/drive_stamped'
    odom_topic = f'/{robot_name}/odom'

    typestore = get_typestore(Stores.ROS2_FOXY)

    # Pass 1: 세 토픽을 헤더 stamp와 함께 전부 수집함
    # (rosbag 기록 시간 대신 publisher가 채운 header.stamp를 정합 기준으로 사용)
    scans = []   # (stamp_ns, reduced_ranges)
    odoms = []   # (stamp_ns, x, y, yaw, vx, wz)
    drives = []  # (stamp_ns, steer, desired_speed)

    print(f"데이터 추출 시작... (대상: {bag_name})")

    try:
        with Reader(bag_dir_path) as reader:
            for connection, _bag_ts, rawdata in reader.messages():
                if connection.topic not in (scan_topic, drive_topic, odom_topic):
                    continue

                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                stamp_ns = stamp_to_ns(msg.header.stamp)

                if connection.topic == scan_topic:
                    ranges = np.array(msg.ranges)
                    ranges = np.where(np.isinf(ranges), 30.0, ranges)
                    ranges = np.where(np.isnan(ranges), 0.0, ranges)

                    reduced_ranges = []
                    for i in range(1, scan_downsample_factor - 1):
                        reduced_ranges.append(float(np.min(ranges[(i-1)*scan_factor : (i+1)*scan_factor])))

                    scans.append((stamp_ns, reduced_ranges))

                elif connection.topic == odom_topic:
                    q = msg.pose.pose.orientation
                    odoms.append((
                        stamp_ns,
                        float(msg.pose.pose.position.x),
                        float(msg.pose.pose.position.y),
                        quat_to_yaw(q.x, q.y, q.z, q.w),
                        float(msg.twist.twist.linear.x),
                        float(msg.twist.twist.angular.z),
                    ))

                elif connection.topic == drive_topic:
                    drives.append((
                        stamp_ns,
                        float(msg.twist.angular.z),
                        float(msg.twist.linear.x),
                    ))

    except Exception as e:
        print(f"Bag 파일을 읽는 중 오류 발생: {e}")
        return

    # rosbag 기록 순서가 헤더 stamp 순서와 어긋날 수 있어서 명시적으로 정렬함
    scans.sort(key=lambda r: r[0])
    odoms.sort(key=lambda r: r[0])
    drives.sort(key=lambda r: r[0])

    scan_stamps = [s[0] for s in scans]
    odom_stamps = [o[0] for o in odoms]

    # Pass 2: 각 drive에 대해 stamp <= drive_stamp인 가장 최근 scan/odom을 인과적으로 매칭함
    # (drive보다 미래의 센서값은 절대로 사용하지 않음. 동일 stamp면 그것을 채택)
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    header = (
        ['time', 'steer', 'desired_speed',
         'odom_x', 'odom_y', 'odom_yaw', 'odom_vx', 'odom_wz']
        + [f'lidar_{i}' for i in range(scan_downsample_factor)]
    )
    csv_writer.writerow(header)

    count_written = 0
    skipped_no_match = 0

    for drive_stamp, steer, desired_speed in drives:
        # bisect_right - 1 : drive_stamp 이하 중 가장 마지막 인덱스 (동일 stamp 존재시 그걸 선택)
        s_idx = bisect.bisect_right(scan_stamps, drive_stamp) - 1
        o_idx = bisect.bisect_right(odom_stamps, drive_stamp) - 1

        if s_idx < 0 or o_idx < 0:
            # drive 시점 이전에 아직 도착한 scan/odom이 없는 경우 스킵 (인과성 보장)
            skipped_no_match += 1
            continue

        _, ox, oy, oyaw, ovx, owz = odoms[o_idx]
        _, ranges_row = scans[s_idx]

        row = [drive_stamp * 1e-9, steer, desired_speed,
               ox, oy, oyaw, ovx, owz] + ranges_row
        csv_writer.writerow(row)
        count_written += 1

    csv_file.close()
    print(f"추출 완료! 총 {count_written}개의 행이 기록됨. (인과적 매칭 실패로 스킵: {skipped_no_match})")
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

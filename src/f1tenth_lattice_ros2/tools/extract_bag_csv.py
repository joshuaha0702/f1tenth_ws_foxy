#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import glob
import bisect
import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# 추론(agent_node.py)과 같은 LiDAR 전처리를 쓰도록 End2Race 패키지 모듈을 직접 import.
# (2026-05-29~10-06 추출본은 겹치는 6빔 min-pooling이라 추론 입력과 달랐음)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'f1tenth_end2race_ros2'))
from f1tenth_end2race_ros2.scan_preprocess import (  # noqa: E402
    NUM_FEATURES, build_scan_mapping, pool_scan)


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


def header_stamps_valid(stamps_ns, label='drive'):
    """header.stamp를 시각으로 믿을 수 있는지: 엄격히 증가하고 간격이 일정한지.

    planner/controller 프로세스 분리 이후 drive stamp는 triggering odom stamp라 정확하다.
    그 이전 데이터는 get_clock().now()로 찍혀 중복(dt==0)·/clock 양자화가 있어 이 검사에 걸린다.
    제어 주기(50/100 Hz 등)에 의존하지 않도록 절대 주파수는 넓게만 보고, 대신
    간격의 산포(IQR/median)로 양자화 여부를 판별한다.
    """
    if len(stamps_ns) < 10:
        return False
    dt = np.diff(np.asarray(stamps_ns, dtype=np.int64))
    if not bool((dt > 0).all()):
        return False
    median = float(np.median(dt))
    # 1 ms ~ 200 ms (5 Hz ~ 1 kHz): 어떤 제어 주기로 수집했더라도 통과
    if not (1e6 <= median <= 200e6):
        print(f"  경고: {label} header stamp 간격 중앙값 {median*1e-6:.1f} ms 가 비정상 범위")
        return False
    # /clock(10 Hz)으로 양자화된 stamp는 median 대비 산포가 매우 크다
    spread = float(np.percentile(dt, 75) - np.percentile(dt, 25))
    if spread > 0.5 * median:
        print(f"  경고: {label} header stamp 간격 산포가 큼 "
              f"(IQR {spread*1e-6:.1f} ms / median {median*1e-6:.1f} ms)")
        return False
    return True


def extract_bag_to_csv(bag_dir_path, save_dir='/root/f1tenth_ws/f1tenth_data', robot_name='car1',
                       min_row_gap_s=0.006, time_source='auto'):
    """time_source: 'header' = drive/scan header.stamp(sim time) 기준, 모든 drive를 한 행으로.
    'receipt' = bag 수신 시각을 /clock으로 sim time에 보간하고 버스트를 건너뜀(예전 데이터용).
    'auto' = drive header stamp가 유효하면 header, 아니면 receipt."""
    scan_downsample_factor = NUM_FEATURES
    scan_mappings = {}  # (빔 수, angle_min, angle_increment) -> build_scan_mapping 결과
    min_row_gap_ns = min_row_gap_s * 1e9

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
    #     (예전 데이터의 header.stamp는 /clock 10Hz로 양자화되고 중복 퍼블리셔 오염도 있음.
    #      header stamp가 유효한 최신 데이터는 time_source='auto'에서 header 경로를 쓴다)
    scans = []          # (bag_ns, reduced_ranges)
    drives = []         # (bag_ns, steer, desired_speed)
    scan_headers = []   # scans와 같은 순서의 header.stamp(ns)
    drive_headers = []  # drives와 같은 순서의 header.stamp(ns)
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
                    key = (len(msg.ranges), msg.angle_min, msg.angle_increment)
                    if key not in scan_mappings:
                        scan_mappings[key] = build_scan_mapping(*key)
                    reduced_ranges = pool_scan(msg.ranges, scan_mappings[key]).tolist()
                    scans.append((bag_ts, reduced_ranges))
                    scan_headers.append(stamp_to_ns(msg.header.stamp))

                elif connection.topic == drive_topic:
                    drives.append((
                        bag_ts,
                        float(msg.drive.steering_angle),
                        float(msg.drive.speed),
                    ))
                    drive_headers.append(stamp_to_ns(msg.header.stamp))

    except Exception as e:
        print(f"Bag 파일을 읽는 중 오류 발생: {e}")
        return 0

    if not clock_wall:
        print("경고: /clock 토픽이 없어 wall->sim 환산을 할 수 없음. bag을 확인하세요.")
        return 0
    if not scans or not drives:
        print(f"{bag_name}: {robot_name} scan/drive 데이터가 없어 건너뜁니다 "
              f"(scans={len(scans)}, drives={len(drives)})")
        return 0

    if time_source == 'auto':
        use_header = (header_stamps_valid(drive_headers, 'drive')
                      and header_stamps_valid(scan_headers, 'scan'))
        time_source = 'header' if use_header else 'receipt'
        print(f"  time_source=auto -> {time_source}")
    if time_source == 'header':
        return _write_header_rows(csv_file_path, scans, scan_headers, drives, drive_headers,
                                  scan_downsample_factor)

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
    skipped_burst = 0
    last_written_sim_ns = None

    for drive_bag_ns, steer, desired_speed in drives:
        # bag_ts(wall)를 /clock 보간으로 고해상도 sim-time으로 환산
        drive_sim_ns = wall_to_sim_ns(drive_bag_ns)

        # 버스트 제거: 직전에 "기록된" 행과의 sim-time 간격이 너무 좁으면 생략함.
        # (직전 기록 행 기준이라, 출력 행들은 항상 min_row_gap_s 이상 간격을 유지)
        if last_written_sim_ns is not None and (drive_sim_ns - last_written_sim_ns) <= min_row_gap_ns:
            skipped_burst += 1
            continue

        # bisect_right - 1 : drive_bag_ns 이하 중 가장 마지막 인덱스
        s_idx = bisect.bisect_right(scan_bag_stamps, drive_bag_ns) - 1

        if s_idx < 0:
            # drive 시점 이전에 아직 도착한 scan이 없는 경우 스킵 (인과성 보장)
            skipped_no_match += 1
            continue

        scan_bag_ns, ranges_row = scans[s_idx]
        scan_sim_ns = wall_to_sim_ns(scan_bag_ns)

        # lidar_delay: RTF 보정된 sim-time 기준 scan-drive 지연 (실제 로봇이 겪을 지연과 일치)
        lidar_delay = (drive_sim_ns - scan_sim_ns) * 1e-9

        # time 컬럼: drive의 고해상도 sim-time
        row = [drive_sim_ns * 1e-9, steer, desired_speed, lidar_delay] + ranges_row
        csv_writer.writerow(row)
        count_written += 1
        last_written_sim_ns = drive_sim_ns

    csv_file.close()
    print(f"추출 완료! 총 {count_written}개의 행이 기록됨. "
          f"(인과 매칭 실패 스킵: {skipped_no_match}, 버스트 스킵(<={min_row_gap_s*1000:.0f}ms): {skipped_burst})")
    print(f"저장 위치: {csv_file_path}")
    return count_written


def _write_header_rows(csv_file_path, scans, scan_headers, drives, drive_headers, num_beams):
    """drive마다 한 행: time = drive header.stamp, LiDAR = 그 시각 이전(<=)의 가장 최근 scan."""
    scan_order = np.argsort(np.asarray(scan_headers, dtype=np.int64), kind='stable')
    scan_stamps = [scan_headers[i] for i in scan_order]
    drive_order = list(np.argsort(np.asarray(drive_headers, dtype=np.int64), kind='stable'))
    # EpisodeManager STOP은 에피소드 끝에 정확히 (steer 0, speed 0) 명령을 한 번 보낸다.
    # 주행 중인 장면에 '즉시 정지' 라벨을 붙이는 수집 부산물이므로 학습 행에서 뺀다.
    dropped_stop = 0
    if drive_order and drives[drive_order[-1]][1] == 0.0 and drives[drive_order[-1]][2] == 0.0:
        drive_order.pop()
        dropped_stop = 1
    count_written = 0
    skipped_no_match = 0
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['time', 'steer', 'desired_speed', 'lidar_delay']
                            + [f'lidar_{i}' for i in range(num_beams)])
        for d_idx in drive_order:
            drive_ns = drive_headers[d_idx]
            _, steer, desired_speed = drives[d_idx]
            s_pos = bisect.bisect_right(scan_stamps, drive_ns) - 1
            if s_pos < 0:
                skipped_no_match += 1  # 첫 scan 이전의 drive (인과성 보장)
                continue
            scan_ns = scan_stamps[s_pos]
            ranges_row = scans[scan_order[s_pos]][1]
            csv_writer.writerow([drive_ns * 1e-9, steer, desired_speed, (drive_ns - scan_ns) * 1e-9]
                                + ranges_row)
            count_written += 1
    print(f"추출 완료(header stamp 기준)! 총 {count_written}개의 행이 기록됨. "
          f"(인과 매칭 실패 스킵: {skipped_no_match}, 끝의 STOP 정지 명령 제외: {dropped_stop})")
    print(f"저장 위치: {csv_file_path}")
    return count_written


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract ego and leader training CSVs from clean episode bags')
    parser.add_argument('folder_name', help='data/ 아래 수집 폴더 이름')
    parser.add_argument('--robots', nargs='+', choices=('car1', 'car2'), default=('car1', 'car2'))
    parser.add_argument('--time-source', choices=('auto', 'header', 'receipt'), default='auto',
                        help='auto: drive header stamp가 유효하면 header, 아니면 bag 수신 시각')
    args = parser.parse_args()

    # data/<folder_name>/clean 안의 모든 rosbag을 data/<folder_name>/csv 에 추출
    base_data_dir = '/root/f1tenth_ws/data'
    target_dir = os.path.join(base_data_dir, args.folder_name)
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

        totals = {robot: 0 for robot in args.robots}
        for folder in target_folders:
            print("=" * 60)
            for robot in args.robots:
                totals[robot] += extract_bag_to_csv(folder, save_dir=save_dir, robot_name=robot,
                                                    time_source=args.time_source)

        print("=" * 60)
        print(f"모든 bag 파일의 데이터 추출이 완료되었음. 저장 위치: {save_dir}")
        print(f"차량별 출력 행 수: {totals}")

#!/usr/bin/env python3
"""Gazebo 에서 car1 LiDAR 가 car2(앞차)를 몇 빔/몇 bin 으로 보는지 측정.

직선 구간에서 car2 를 car1 바로 앞에 스폰한 상태로 /car1/scan 을 받아, 전방 4 m 이내·횡위치 |y| ≤ 0.6 m 로
돌아온 빔(= 앞차; 직선 구간 벽은 |y| ≥ 1.4 m) 수와 scan_preprocess 의 360-bin
min-pooling 후 같은 조건의 bin 수를 센다. (odom 은 쓰지 않음 — 스폰 기준 좌표라 혼동 유발)

    python3 measure_leader_footprint.py --spawn1 6.4 16.0 -90 --spawn2 6.4 13.5 -90 [--seconds 5] [--tol 0.5]

odom 은 각 차량의 스폰 자세 기준이므로 스폰 자세(x y yaw_deg)를 받아 월드 좌표로 바꾼다.
"""
import argparse
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

sys.path.insert(0, '/root/f1tenth_ws/src/f1tenth_end2race_ros2/f1tenth_end2race_ros2')
from scan_preprocess import build_scan_mapping, pool_scan, FOV_MIN, FOV_MAX, NUM_FEATURES  # noqa: E402


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Meter(Node):
    def __init__(self, tol, spawn1, spawn2):
        super().__init__('leader_footprint_meter')
        self.tol = tol
        self.spawn = (spawn1, spawn2)
        self.o1 = self.o2 = None
        self.rows = []
        self.rear_min = []
        self.mapping = None
        self.create_subscription(Odometry, '/car1/odom', lambda m: setattr(self, 'o1', m), 20)
        self.create_subscription(Odometry, '/car2/odom', lambda m: setattr(self, 'o2', m), 20)
        self.create_subscription(LaserScan, '/car1/scan', self.scan_cb, qos_profile_sensor_data)

    @staticmethod
    def world(odom, spawn):
        sx, sy, syaw = spawn[0], spawn[1], math.radians(spawn[2])
        p = odom.pose.pose
        x = sx + math.cos(syaw) * p.position.x - math.sin(syaw) * p.position.y
        y = sy + math.sin(syaw) * p.position.x + math.cos(syaw) * p.position.y
        return x, y, syaw + yaw_of(p.orientation)

    def scan_cb(self, msg):
        r = np.asarray(msg.ranges, dtype=np.float64)
        r = np.where(np.isinf(r), 30.0, r)
        n = len(r)
        ang = msg.angle_min + np.arange(n) * msg.angle_increment
        # 앞차 = 전방 4 m 이내, 횡위치 |y| <= 0.6 m (직선 구간 벽은 |y| >= 1.4 m)
        x, y = r * np.cos(ang), r * np.sin(ang)
        hit = (x > 0.1) & (x < 4.0) & (np.abs(y) <= 0.6)
        beams = int(hit.sum())
        if beams == 0:
            self.rows.append((np.nan, np.nan, 0, 0, 0.0, 0.0)); return
        gap = float(r[hit].min()); bearing = float(np.degrees(ang[hit].mean()))
        if self.mapping is None:
            self.mapping = build_scan_mapping(n, msg.angle_min, msg.angle_increment)
        pooled = pool_scan(msg.ranges, self.mapping)
        edges = np.linspace(FOV_MIN, FOV_MAX, NUM_FEATURES + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        px, py = pooled * np.cos(centers), pooled * np.sin(centers)
        bins = int(((px > 0.1) & (px < 4.0) & (np.abs(py) <= 0.6)).sum())
        width_deg = beams * math.degrees(msg.angle_increment)
        # 자가-히트 검사: 후방 |ang| > 100° 섹터의 최소 거리 (자기 차체 셸이 보이면 0.1~0.3 m 가 찍힘)
        rear = np.abs(ang) > math.radians(100)
        self.rear_min.append(float(r[rear].min()) if rear.any() else np.nan)
        self.rows.append((gap, bearing, beams, bins, width_deg, 2 * gap * math.tan(math.radians(width_deg) / 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=5.0)
    ap.add_argument('--tol', type=float, default=0.5)
    ap.add_argument('--spawn1', type=float, nargs=3, default=[6.4, 16.0, -90.0], metavar=('X', 'Y', 'YAW_DEG'))
    ap.add_argument('--spawn2', type=float, nargs=3, default=[6.4, 13.5, -90.0], metavar=('X', 'Y', 'YAW_DEG'))
    a = ap.parse_args()
    rclpy.init()
    node = Meter(a.tol, a.spawn1, a.spawn2)
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    rows = [x for x in node.rows if x[2] > 0]
    node.destroy_node(); rclpy.shutdown()
    print(f'scans without any return < 4 m in the forward sector: {sum(1 for x in node.rows if x[2] == 0)}')
    if not rows:
        print('no scans with both odoms'); return
    R = np.array(rows)
    print(f'{len(R)} scans | gap {R[:,0].min():.2f}-{R[:,0].max():.2f} m, bearing {R[:,1].mean():+.1f} deg')
    print(f'raw beams on leader : median {np.median(R[:,2]):.0f} (min {R[:,2].min():.0f}, max {R[:,2].max():.0f})')
    print(f'360-bin on leader   : median {np.median(R[:,3]):.0f} (min {R[:,3].min():.0f}, max {R[:,3].max():.0f})')
    print(f'apparent width      : median {np.median(R[:,5]):.3f} m  (angular {np.median(R[:,4]):.2f} deg)')
    rm = np.array(node.rear_min); rm = rm[np.isfinite(rm)]
    if len(rm):
        print(f'self-hit check: min range in rear sector (|ang|>100 deg) = {rm.min():.2f} m  (<0.35 m 이면 자기 셸이 보이는 것)')
    g = np.median(R[:, 0])
    print(f'expected for 0.28 m body at {g:.2f} m: {math.degrees(2*math.atan(0.14/g)):.2f} deg = '
          f'{math.degrees(2*math.atan(0.14/g))/0.25:.0f} beams(0.25deg) / {math.degrees(2*math.atan(0.14/g))/0.75:.1f} bins')


if __name__ == '__main__':
    main()

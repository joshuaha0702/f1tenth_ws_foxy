"""h2h 평가 bag → 에피소드별 시계열 (컨테이너에서 실행: rosbags 필요).

    python3 extract_episodes.py <out.pkl> <eval root> [--raceline-name raceline1_ccw.csv]

evaluate_overtakes.py 와 같은 raceline 투영으로 진행 간격 g = s(car2) - s(car1) [m]
(앞차가 앞이면 +, g < -0.6 이면 추월)을 구하고, 같은 raceline 기준 부호 있는 횡 오프셋 d [m]
(진행 방향 왼쪽 +), 두 차의 속도, ego 의 조향·속도 명령을 odom 시각(100 Hz)에 맞춰 저장한다.
에피소드 판정(추월·충돌 상대)은 그 root 의 scores.csv 를 그대로 쓴다.
"""
import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

sys.path.insert(0, '/root/f1tenth_ws/src/f1tenth_lattice_ros2/tools')
from evaluate_overtakes import load_raceline, unwrapped_progress  # noqa: E402

MAPS = '/root/f1tenth_ws/src/f1tenth_lattice_ros2/maps'
ts = get_typestore(Stores.ROS2_FOXY)
ts.register(get_types_from_msg("float32 steering_angle\nfloat32 steering_angle_velocity\nfloat32 speed\n"
                               "float32 acceleration\nfloat32 jerk\n", 'ackermann_msgs/msg/AckermannDrive'))
ts.register(get_types_from_msg("std_msgs/Header header\nackermann_msgs/AckermannDrive drive\n",
                               'ackermann_msgs/msg/AckermannDriveStamped'))


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def lateral(xy, pts):
    """raceline 기준 부호 있는 횡 오프셋 (진행 방향 왼쪽 +)."""
    d = np.linalg.norm(pts[:, None, :] - xy[None, :, :], axis=2)
    i = np.argmin(d, axis=1)
    tan = np.roll(xy, -1, axis=0)[i] - np.roll(xy, 1, axis=0)[i]
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    rel = pts - xy[i]
    return tan[:, 0] * rel[:, 1] - tan[:, 1] * rel[:, 0]


def read(bag):
    odom = {'car1': [], 'car2': []}
    drive = []
    with Reader(bag) as r:
        cons = [c for c in r.connections if c.topic in ('/car1/odom', '/car2/odom', '/car1/drive')]
        for c, _, raw in r.messages(connections=cons):
            m = ts.deserialize_cdr(raw, c.msgtype)
            if c.topic == '/car1/drive':
                drive.append((stamp(m.header), m.drive.steering_angle, m.drive.speed))
            else:
                odom[c.topic[1:5]].append((stamp(m.header), m.pose.pose.position.x, m.pose.pose.position.y,
                                           m.twist.twist.linear.x))
    return {k: np.array(v) for k, v in odom.items()}, np.array(drive)


def episode(bag, xy, s, lap):
    odom, drive = read(bag)
    a, b = odom['car1'], odom['car2']
    if len(a) < 10 or len(b) < 10 or len(drive) == 0:
        return None
    t0, t1 = max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0])
    a = a[(a[:, 0] >= t0) & (a[:, 0] <= t1)]
    b = b[(b[:, 0] >= t0) & (b[:, 0] <= t1)]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    s1 = unwrapped_progress(xy, s, lap, a[:, 1:3])
    s2 = unwrapped_progress(xy, s, lap, b[:, 1:3])
    gap0 = (s2[0] - s1[0]) % lap
    gap = gap0 + (s2 - s2[0]) - (s1 - s1[0])
    order = np.argsort(drive[:, 0])
    drive = drive[order]
    # ego 명령: 각 odom 시각 직전 drive 메시지 (zero-order hold)
    k = np.clip(np.searchsorted(drive[:, 0], a[:, 0], side='right') - 1, 0, len(drive) - 1)
    return {
        't': (a[:, 0] - a[0, 0]).astype(np.float32),
        'gap': gap.astype(np.float32),
        'd1': lateral(xy, a[:, 1:3]).astype(np.float32),
        'd2': lateral(xy, b[:, 1:3]).astype(np.float32),
        'v1': np.abs(a[:, 3]).astype(np.float32),
        'v2': np.abs(b[:, 3]).astype(np.float32),
        'steer': drive[k, 1].astype(np.float32),
        'cmd_v': drive[k, 2].astype(np.float32),
        'xy1': a[:, 1:3].astype(np.float32),
        'xy2': b[:, 1:3].astype(np.float32),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out')
    p.add_argument('root')
    p.add_argument('--raceline-name', default='raceline1.csv')
    args = p.parse_args()
    scores = {}
    with open(os.path.join(args.root, 'scores.csv')) as f:
        for row in csv.DictReader(f):
            scores[(row['map'], row['episode'])] = row
    eps = []
    for map_dir in sorted(glob.glob(os.path.join(args.root, '*'))):
        name = os.path.basename(map_dir)
        rl = os.path.join(MAPS, name, args.raceline_name)
        if not os.path.isdir(map_dir) or not os.path.exists(rl):
            continue
        xy, s, lap = load_raceline(rl)
        for bag in sorted(glob.glob(os.path.join(map_dir, '*', 'ep*'))):
            label = os.path.basename(bag)
            sc = scores.get((name, label))
            if sc is None:          # evaluate_overtakes 가 건너뛴 에피소드 (dead agent 등)
                continue
            ep = episode(bag, xy, s, lap)
            if ep is None:
                continue
            ep.update(map=name, episode=label, overtake=sc['overtake'] == 'True',
                      collision=sc['collision'] == 'True', collision_with=sc['collision_with'],
                      raceline=xy.astype(np.float32))
            eps.append(ep)
    with open(args.out, 'wb') as f:
        pickle.dump(eps, f)
    print(f"{args.root}: {len(eps)} episodes (scores.csv rows {len(scores)}) -> {args.out}")


if __name__ == '__main__':
    main()

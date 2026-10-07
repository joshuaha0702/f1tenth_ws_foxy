#!/usr/bin/env python3
"""Score head-to-head evaluation episodes: overtakes, collisions and progress.

Expects the layout written by collect_variants.sh / evaluate_ego.sh:
<root>/<map>/{clean,collision}/ep*  and  <root>/logs/<map>.log
Both cars' odom is projected on maps/<map>/raceline1.csv. The gap
g = s(car2) - s(car1) along the lap starts positive (car2 ahead); an overtake is
counted when car1 gets a car length ahead (g < -OVERTAKE_MARGIN_M).
Collisions are split by the contact partner named in the episode manager log
(car = car1 touched car2, wall_car1 / wall_car2 = that car touched a wall).
Leader columns (speed, stuck) are for judging an End2Race leader (LEADER=end2race).

    evaluate_overtakes.py <root> [--maps-dir src/f1tenth_lattice_ros2/maps] [--csv out.csv]
"""

import argparse
import csv
import glob
import os
import re

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

OVERTAKE_MARGIN_M = 0.6      # about one car length (0.58 m)
STUCK_SPEED_MPS = 0.2
STUCK_WINDOW_S = 2.0


def load_raceline(path):
    rl = np.loadtxt(path, delimiter=';', comments='#')
    xy = rl[:-1, 1:3] if np.allclose(rl[0, 1:3], rl[-1, 1:3]) else rl[:, 1:3]
    seg = np.linalg.norm(np.diff(np.vstack((xy, xy[:1])), axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)[:-1]))
    return xy, s, float(seg.sum())


def unwrapped_progress(xy_track, s_track, lap, points):
    d = np.linalg.norm(points[:, None, :] - xy_track[None, :, :], axis=2)
    s = s_track[np.argmin(d, axis=1)]
    return np.concatenate(([s[0]], s[0] + np.cumsum((np.diff(s) + lap / 2) % lap - lap / 2)))


def read_odom(bag, typestore):
    out = {'car1': [], 'car2': []}
    with Reader(bag) as reader:
        for conn, _, raw in reader.messages():
            for car in out:
                if conn.topic == f'/{car}/odom':
                    msg = typestore.deserialize_cdr(raw, conn.msgtype)
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    out[car].append((t, msg.pose.pose.position.x, msg.pose.pose.position.y,
                                     msg.twist.twist.linear.x))
    return {car: np.array(v) for car, v in out.items()}


def parse_log(path):
    """episode label -> 'car' | 'wall_car1' | 'wall_car2' | None from '[collision] ...' lines."""
    kinds, current = {}, None
    if not os.path.exists(path):
        return kinds
    for line in open(path, errors='replace'):
        m = re.search(r'\[(ep\d+)\] starting', line)
        if m:
            current = m.group(1)
            kinds[current] = None
            continue
        m = re.search(r'\[collision\] (car\d): (.*)', line)
        if m and current and kinds.get(current) is None:
            names = m.group(2)
            models = set(re.findall(r'(\w+)::', names))
            if {'car1', 'car2'} <= models:
                kinds[current] = 'car'
            else:
                kinds[current] = 'wall_car2' if 'car2' in models else 'wall_car1'
    return kinds


def score_episode(bag, xy, s, lap, typestore):
    odom = read_odom(bag, typestore)
    a, b = odom['car1'], odom['car2']
    if len(a) < 10 or len(b) < 10:
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
    passed = np.nonzero(gap < -OVERTAKE_MARGIN_M)[0]
    tail = a[:, 0] >= a[-1, 0] - STUCK_WINDOW_S
    return {
        'duration_s': float(a[-1, 0] - a[0, 0]),
        'start_gap_m': float(gap0),
        'min_gap_m': float(gap.min()),
        'end_gap_m': float(gap[-1]),
        'overtake': bool(len(passed)),
        'overtake_time_s': float(a[passed[0], 0] - a[0, 0]) if len(passed) else float('nan'),
        'ego_progress_m': float(s1[-1] - s1[0]),
        'leader_progress_m': float(s2[-1] - s2[0]),
        'ego_mean_speed': float(np.abs(a[:, 3]).mean()),
        'stuck': bool(np.abs(a[tail, 3]).mean() < STUCK_SPEED_MPS),
        'leader_mean_speed': float(np.abs(b[:, 3]).mean()),
        'leader_stuck': bool(np.abs(b[b[:, 0] >= b[-1, 0] - STUCK_WINDOW_S, 3]).mean() < STUCK_SPEED_MPS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('root')
    parser.add_argument('--maps-dir', default=os.path.join(os.path.dirname(__file__), '..', 'maps'))
    parser.add_argument('--csv', help='write one row per episode')
    args = parser.parse_args()

    typestore = get_typestore(Stores.ROS2_FOXY)
    rows = []
    for map_dir in sorted(glob.glob(os.path.join(args.root, '*', ''))):
        name = os.path.basename(os.path.dirname(map_dir))
        raceline = os.path.join(args.maps_dir, name, 'raceline1.csv')
        if name == 'logs' or not os.path.exists(raceline):
            continue
        xy, s, lap = load_raceline(raceline)
        kinds = parse_log(os.path.join(args.root, 'logs', f'{name}.log'))
        for bag in sorted(glob.glob(os.path.join(map_dir, '*', 'ep*'))):
            res = score_episode(bag, xy, s, lap, typestore)
            if res is None:
                continue
            label = os.path.basename(bag).split('_')[0]
            collided = os.path.basename(os.path.dirname(bag)) == 'collision'
            res.update(map=name, episode=os.path.basename(bag), collision=collided,
                       collision_with=(kinds.get(label) or 'unknown') if collided else '')
            rows.append(res)

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def report(label, sel):
        if not sel:
            return
        n = len(sel)
        ot = sum(r['overtake'] for r in sel)
        clean_ot = sum(r['overtake'] and not r['collision'] for r in sel)
        car = sum(r['collision_with'] == 'car' for r in sel)
        wall1 = sum(r['collision_with'] == 'wall_car1' for r in sel)
        wall2 = sum(r['collision_with'] == 'wall_car2' for r in sel)
        stuck = sum(r['stuck'] for r in sel)
        lstuck = sum(r['leader_stuck'] for r in sel)
        print(f'{label:18s} n={n:3d} | overtake {ot / n:6.1%} (collision-free {clean_ot / n:6.1%}) | '
              f'collision car {car / n:6.1%}, wall ego {wall1 / n:5.1%} leader {wall2 / n:5.1%} | '
              f'ego stuck {stuck / n:5.1%}, speed {np.mean([r["ego_mean_speed"] for r in sel]):4.2f} m/s | '
              f'leader stuck {lstuck / n:5.1%}, speed {np.mean([r["leader_mean_speed"] for r in sel]):4.2f} m/s')

    for name in sorted({r['map'] for r in rows}):
        report(name, [r for r in rows if r['map'] == name])
    report('ALL', rows)


if __name__ == '__main__':
    main()

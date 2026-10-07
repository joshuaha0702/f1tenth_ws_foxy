#!/usr/bin/env python3
"""Validate the control rate of recorded head-to-head episode bags.

Applies the pass criteria of CONTROL_RATE_100HZ_ISSUE.md to every bag, per car:
drive header.stamp spacing (mean ~10 ms, no duplicate/backward stamps, no gap
longer than the nominal period) and drive stamps matching an odom stamp.
It also reports the scan rate and the real-time factor (RTF) of each bag.

Each bag/car gets two verdicts:
  strict  - every criterion above holds (the document's definition).
  usable  - fit for training: mean dt 10 +/- 0.2 ms, no duplicate/backward stamps,
            no drive gap above 20 ms (at most one missing step), the controller
            answered every recorded odom inside its drive span, and every drive
            inside the recorded odom span matches an odom stamp.
Drive gaps are split into controller_miss (odom recorded, drive not) and
sim_odom_gaps (odom missing from the bag too). Despite the names, both were
mostly bag recorder drops in the 2026-09 data: the controller's own log showed
100 Hz over the same windows. Drives before the bag caught the first odom
message are not counted as mismatches (topic discovery latency).

Bags written by the episode manager since 2026-10 carry episode_info.yaml with
the episode's t_start/t_end (sim). Then the bag must cover the whole episode:
the first drive at most COVERAGE_TOLERANCE after that car's first drive seen
live (first_drive, falling back to t_start) and the last drive at
most that before t_end, for both verdicts. Older bags without the file skip
this check (head_missing_s / tail_missing_s left empty).

    validate_episodes.py <bag glob> [--cars car1 car2] [--summary out.csv]
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import yaml
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

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

PERIOD_NS = 10_000_000          # 100 Hz
GAP_TOLERANCE_NS = 1_000_000    # a dt above 11 ms counts as a missed control step
COVERAGE_TOLERANCE_S = 0.02     # head/tail of the episode the bag may miss (2 steps)
FIELDS = ['bag', 'car', 'strict', 'usable', 'reason', 'drive_n', 'odom_n', 'rate_hz',
          'dt_mean_ms', 'dt_std_ms', 'dt_min_ms', 'dt_max_ms', 'dup', 'backward', 'gaps',
          'controller_miss', 'sim_odom_gaps', 'odom_max_dt_ms', 'odom_match',
          'unrecorded_odom_drives', 'span_s', 'head_missing_s', 'tail_missing_s', 'scan_hz', 'rtf']


def make_typestore():
    typestore = get_typestore(Stores.ROS2_FOXY)
    typestore.register(get_types_from_msg(ACKERMANN_DRIVE_MSG, 'ackermann_msgs/msg/AckermannDrive'))
    typestore.register(get_types_from_msg(ACKERMANN_DRIVE_STAMPED_MSG,
                                          'ackermann_msgs/msg/AckermannDriveStamped'))
    return typestore


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def read_bag(path, cars, typestore):
    topics = {}
    for car in cars:
        for kind in ('drive', 'odom', 'scan'):
            topics[f'/{car}/{kind}'] = (car, kind)
    data = {car: {'drive': [], 'odom': [], 'scan': []} for car in cars}
    clock_wall, clock_sim = [], []
    with Reader(path) as reader:
        for conn, bag_ns, raw in reader.messages():
            if conn.topic == '/clock':
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                clock_wall.append(bag_ns)
                clock_sim.append(stamp_ns(msg.clock))
            elif conn.topic in topics:
                car, kind = topics[conn.topic]
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                data[car][kind].append(stamp_ns(msg.header.stamp))
    return data, np.array(clock_wall, dtype=np.int64), np.array(clock_sim, dtype=np.int64)


def check_car(streams, episode=None, car=None):
    drive = np.array(streams['drive'], dtype=np.int64)
    odom = np.array(sorted(set(streams['odom'])), dtype=np.int64)
    res = {'drive_n': len(drive), 'odom_n': len(odom)}
    if len(drive) < 2 or len(odom) < 2:
        res.update(strict=False, usable=False, reason='fewer than 2 drive/odom messages')
        return res
    dt = np.diff(drive)
    span = (drive[-1] - drive[0]) * 1e-9
    odom_set = set(odom.tolist())
    drive_set = set(drive.tolist())
    in_odom_span = drive[(drive >= odom[0]) & (drive <= odom[-1])]
    matched = sum(1 for t in in_odom_span.tolist() if t in odom_set)
    lo, hi = max(drive[0], odom[0]), min(drive[-1], odom[-1])
    odom_in_span = odom[(odom >= lo) & (odom <= hi)]
    odom_dt = np.diff(odom)
    res.update(
        rate_hz=(len(drive) - 1) / span if span > 0 else 0.0,
        dt_mean_ms=dt.mean() * 1e-6, dt_std_ms=dt.std() * 1e-6,
        dt_min_ms=dt.min() * 1e-6, dt_max_ms=dt.max() * 1e-6,
        dup=int((dt == 0).sum()), backward=int((dt < 0).sum()),
        gaps=int((dt > PERIOD_NS + GAP_TOLERANCE_NS).sum()),
        controller_miss=int(sum(1 for t in odom_in_span.tolist() if t not in drive_set)),
        sim_odom_gaps=int((odom_dt > PERIOD_NS + GAP_TOLERANCE_NS).sum()),
        odom_max_dt_ms=odom_dt.max() * 1e-6,
        odom_match=matched / len(in_odom_span) if len(in_odom_span) else 0.0,
        unrecorded_odom_drives=int(len(drive) - len(in_odom_span)),
        span_s=span,
    )
    scan = np.array(streams['scan'], dtype=np.int64)
    res['scan_hz'] = (len(scan) - 1) / ((scan[-1] - scan[0]) * 1e-9) if len(scan) > 1 else 0.0

    problems = []
    if abs(res['dt_mean_ms'] - 10.0) > 0.2:
        problems.append(f"mean dt {res['dt_mean_ms']:.3f} ms")
    if res['dup'] or res['backward']:
        problems.append(f"dup={res['dup']} backward={res['backward']}")
    if res['gaps']:
        problems.append(f"{res['gaps']} gaps (max {res['dt_max_ms']:.0f} ms; controller misses "
                        f"{res['controller_miss']}, sim odom gaps {res['sim_odom_gaps']})")
    if res['odom_match'] < 1.0:
        problems.append(f"odom match {res['odom_match']:.4f}")
    covered = True
    if episode is not None:
        # first_drive: 차량별 첫 drive 시각 (car2는 car1보다 늦게 출발할 수 있음)
        start = (episode.get('first_drive') or {}).get(car)
        start = float(episode['t_start']) if start is None else float(start)
        res['head_missing_s'] = drive[0] * 1e-9 - start
        res['tail_missing_s'] = float(episode['t_end']) - drive[-1] * 1e-9
        covered = max(res['head_missing_s'], res['tail_missing_s']) <= COVERAGE_TOLERANCE_S
        if not covered:
            problems.append(f"episode not covered (head missing {res['head_missing_s']:.3f} s, "
                            f"tail missing {res['tail_missing_s']:.3f} s)")
    res['strict'] = not problems
    res['usable'] = (abs(res['dt_mean_ms'] - 10.0) <= 0.2 and not res['dup'] and not res['backward']
                     and res['dt_max_ms'] <= 20.5 and res['controller_miss'] == 0
                     and res['odom_match'] == 1.0 and covered)
    res['reason'] = '; '.join(problems)
    return res


def validate(bags, cars, typestore):
    rows = []
    for bag in bags:
        try:
            data, clock_wall, clock_sim = read_bag(bag, cars, typestore)
        except Exception as e:  # an unreadable bag is not usable
            for car in cars:
                rows.append({'bag': bag, 'car': car, 'strict': False, 'usable': False,
                             'reason': f'read error: {e}'})
            continue
        rtf = float('nan')
        if len(clock_wall) > 1 and clock_wall[-1] > clock_wall[0]:
            rtf = (clock_sim[-1] - clock_sim[0]) / (clock_wall[-1] - clock_wall[0])
        episode = None
        info_path = os.path.join(bag, 'episode_info.yaml')
        if os.path.isfile(info_path):
            with open(info_path) as f:
                episode = yaml.safe_load(f)
        for car in cars:
            res = check_car(data[car], episode, car)
            res.update(bag=bag, car=car, rtf=rtf)
            rows.append(res)
    return rows


def print_report(rows, cars):
    for car in cars:
        sel = [r for r in rows if r['car'] == car]
        measured = [r for r in sel if 'rate_hz' in r]
        strict = sum(1 for r in sel if r['strict'])
        usable = sum(1 for r in sel if r['usable'])
        print(f'{car}: strict pass {strict}/{len(sel)}, usable {usable}/{len(sel)}')
        if measured:
            def col(key):
                return np.array([r[key] for r in measured], dtype=float)
            dt_std = col('dt_std_ms')
            print(f'  rate {col("rate_hz").mean():.4f} Hz (min {col("rate_hz").min():.4f}, '
                  f'max {col("rate_hz").max():.4f}) | dt mean {col("dt_mean_ms").mean():.4f} ms | '
                  f'per-bag dt std median {np.median(dt_std):.4f} ms, mean {dt_std.mean():.4f}, '
                  f'max {dt_std.max():.4f} | dt range [{col("dt_min_ms").min():.2f}, '
                  f'{col("dt_max_ms").max():.2f}] ms')
            print(f'  gaps {int(col("gaps").sum())} (controller misses {int(col("controller_miss").sum())}, '
                  f'sim odom gaps {int(col("sim_odom_gaps").sum())}) | dup {int(col("dup").sum())}, '
                  f'backward {int(col("backward").sum())} | odom match min {col("odom_match").min():.4f} | '
                  f'scan {col("scan_hz").mean():.1f} Hz | RTF mean {np.nanmean(col("rtf")):.2f} '
                  f'(min {np.nanmin(col("rtf")):.2f})')
        bad = [r for r in sel if not r['usable']]
        for r in bad[:10]:
            print(f'  NOT USABLE {os.path.basename(r["bag"])}: {r["reason"]}')
        if len(bad) > 10:
            print(f'  ... {len(bad) - 10} more')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('bags', help='glob of bag directories, e.g. "data/run/*/clean/ep*"')
    parser.add_argument('--cars', nargs='+', default=['car1', 'car2'])
    parser.add_argument('--summary', help='write one row per bag and car to this CSV')
    args = parser.parse_args()

    bags = sorted(p for p in glob.glob(args.bags) if os.path.isdir(p))
    if not bags:
        sys.exit(f'no bag directories match {args.bags}')
    rows = validate(bags, args.cars, make_typestore())
    if args.summary:
        with open(args.summary, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    print(f'{len(bags)} bags')
    print_report(rows, args.cars)
    sys.exit(0 if all(r['usable'] for r in rows) else 1)


if __name__ == '__main__':
    main()

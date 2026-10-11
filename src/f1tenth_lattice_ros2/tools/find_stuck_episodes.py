#!/usr/bin/env python3
"""Flag End2Race training episodes in which the car never really drives ("stuck").

Scans one or more training CSV directories (columns: time, steer, desired_speed,
lidar_delay, lidar_0..359[, odom_speed]; one CSV per episode named
<map>_ep<NNNN>_<stamp>.csv) and prints a per-map table of total / stuck episodes.
The stuck list (one CSV basename per line, with the reasons) is written to
<csv_dir>/stuck_episodes.txt unless --no-write is given.

Criteria with an odom_speed column (any one flags the episode):
  * slow_frac : odom_speed < --v-stuck AND desired_speed > --cmd-min in more than
                --slow-frac of all rows
  * start     : odom_speed < --v-stuck for the whole first --start-s seconds
                although desired_speed > --cmd-min at some point in that window
  * distance  : integral(odom_speed dt) < --min-dist m in an episode >= --min-dur s
Without odom_speed (older sets), a LiDAR-static criterion is used:
  * lidar_static : mean over rows of mean(|lidar_t - lidar_{t-1}|) < --lidar-eps m
                   (default 0.005; stuck episodes measure <= 0.002, driving ones >= 0.009)

Usage:
  python3 find_stuck_episodes.py data/train_mix_odom_spline_20261009/car1 \
      data/train_mix_odom_spline_20261009/car2
"""
import argparse
import csv
import os
import re
import sys
from collections import OrderedDict

import numpy as np

MAP_RE = re.compile(r'^(.*?)_ep\d+')


def map_of(name):
    m = MAP_RE.match(name)
    return m.group(1) if m else name.split('_')[0]


def load_csv(path):
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return list(df.columns), df.to_numpy(dtype=np.float64)
    except ImportError:
        pass
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    if not rows:
        return header, np.zeros((0, len(header)), dtype=np.float32)
    data = np.asarray(rows, dtype=np.float64)
    return header, data


def check_episode(path, a):
    header, d = load_csv(path)
    col = {h: i for i, h in enumerate(header)}
    info = {'rows': len(d), 'reasons': []}
    if len(d) < 2:
        info['reasons'].append('empty')
        return info
    t = d[:, col['time']]
    dur = float(t[-1] - t[0])
    info['dur'] = dur
    cmd = d[:, col['desired_speed']]
    lidar_cols = [col['lidar_%d' % i] for i in range(360)]
    lid = d[:, lidar_cols]
    lid = np.where(np.isfinite(lid), lid, 0.0)
    dl = float(np.mean(np.abs(np.diff(lid, axis=0)))) if len(lid) > 1 else 0.0
    info['lidar_change'] = dl

    if 'odom_speed' in col:
        v = np.abs(d[:, col['odom_speed']])
        dt = np.diff(t, prepend=t[0])
        dist = float(np.sum(v * dt))
        info['dist'] = dist
        cmd_on = cmd > a.cmd_min
        slow = v < a.v_stuck
        # fraction of ALL rows in which the car is commanded (> cmd_min) but slow;
        # a few command spikes in an otherwise slow-commanded episode do not count.
        frac = float(np.mean(slow & cmd_on))
        info['slow_frac'] = frac
        if frac > a.slow_frac:
            info['reasons'].append('slow_frac=%.2f' % frac)
        w = t < t[0] + a.start_s
        if dur >= a.start_s and cmd_on[w].any() and slow[w].all():
            info['reasons'].append('start_%gs' % a.start_s)
        if dur >= a.min_dur and dist < a.min_dist:
            info['reasons'].append('dist=%.2fm' % dist)
    else:
        if dl < a.lidar_eps:
            info['reasons'].append('lidar_static=%.4f' % dl)
    return info


def scan_dir(csv_dir, a):
    files = sorted(f for f in os.listdir(csv_dir) if f.endswith('.csv'))
    per_map = OrderedDict()
    stuck = []
    for f in files:
        info = check_episode(os.path.join(csv_dir, f), a)
        m = map_of(f)
        tot, st = per_map.get(m, (0, 0))
        is_stuck = bool(info['reasons'])
        per_map[m] = (tot + 1, st + int(is_stuck))
        if is_stuck:
            stuck.append((f, info))
    return per_map, stuck


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('csv_dirs', nargs='+')
    p.add_argument('--v-stuck', type=float, default=0.1)
    p.add_argument('--cmd-min', type=float, default=1.0)
    p.add_argument('--slow-frac', type=float, default=0.5)
    p.add_argument('--start-s', type=float, default=3.0)
    p.add_argument('--min-dist', type=float, default=3.0)
    p.add_argument('--min-dur', type=float, default=8.0)
    p.add_argument('--lidar-eps', type=float, default=0.005,
                   help='LiDAR-static threshold (m). Calibrated on the odom set: stuck <= 0.002, '
                        'driving >= 0.009 (median 0.025), so 0.02 would flag ~half the moving episodes')
    p.add_argument('--no-write', action='store_true',
                   help='do not write <csv_dir>/stuck_episodes.txt')
    p.add_argument('--all-maps', action='store_true',
                   help='print maps without stuck episodes too')
    a = p.parse_args()

    for csv_dir in a.csv_dirs:
        if not os.path.isdir(csv_dir):
            print('skip (not a dir): %s' % csv_dir, file=sys.stderr)
            continue
        per_map, stuck = scan_dir(csv_dir, a)
        tot = sum(v[0] for v in per_map.values())
        nst = sum(v[1] for v in per_map.values())
        print('\n== %s : %d episodes, %d stuck' % (csv_dir, tot, nst))
        print('%-16s %6s %6s' % ('map', 'total', 'stuck'))
        for m, (n, s) in per_map.items():
            if s or a.all_maps:
                print('%-16s %6d %6d' % (m, n, s))
        if not a.no_write:
            out = os.path.join(csv_dir, 'stuck_episodes.txt')
            with open(out, 'w') as f:
                for name, info in stuck:
                    f.write('%s\t%s\n' % (name, ','.join(info['reasons'])))
            print('wrote %s (%d entries)' % (out, len(stuck)))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Write per-row driving context next to End2Race training CSVs: <csv>.meta.npz.

For every row of <dir>/<map>_<episode>.csv (time = drive header stamp, as written by
extract_bag_csv.py) the episode bag gives both cars' odom; the sidecar stores
  gap    leader arc-length gap along maps/<map>/raceline1.csv (m, > 0 = leader ahead)
  kappa  raceline curvature at the ego position (1/m)
train.py --balance_leader_near uses them to stop following-behind-the-leader rows from
dominating (on Freeform tracks 65 % of the corner time is spent crawling behind the leader).

    annotate_training_csv.py <csv dir> --bag-roots data/simple_variants_h2h_12s data/freeform_h2h_12s
"""

import argparse
import glob
import os

import numpy as np

from analyze_collisions import load_raceline, read_bag
from validate_episodes import make_typestore


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('csv_dir')
    parser.add_argument('--bag-roots', nargs='+', required=True)
    parser.add_argument('--car', default='car1')
    parser.add_argument('--maps-dir', default=os.path.join(os.path.dirname(__file__), '..', 'maps'))
    args = parser.parse_args()
    typestore = make_typestore()
    tracks = {}
    done = missing = 0
    for path in sorted(glob.glob(os.path.join(args.csv_dir, '*.csv'))):
        name = os.path.basename(path)[:-4]
        map_name, episode = name.split('_ep')[0], 'ep' + name.split('_ep')[1]
        bags = [b for root in args.bag_roots for b in glob.glob(os.path.join(root, map_name, '*', episode))]
        if not bags:
            missing += 1
            continue
        if map_name not in tracks:
            rl = os.path.join(args.maps_dir, map_name, 'raceline1.csv')
            track = load_raceline(rl)
            tracks[map_name] = (track, np.loadtxt(rl, delimiter=';', comments='#')[:len(track[0]), 4])
        (xy, s, lap, _), kappa_track = tracks[map_name]
        t = np.loadtxt(path, delimiter=',', skiprows=1, usecols=0)
        d = read_bag(bags[0], typestore)
        ego, other = (d['odom1'], d['odom2']) if args.car == 'car1' else (d['odom2'], d['odom1'])
        ts = t * 1e9
        p1 = np.column_stack([np.interp(ts, ego[:, 0], ego[:, k]) for k in (1, 2)])
        p2 = np.column_stack([np.interp(ts, other[:, 0], other[:, k]) for k in (1, 2)])
        i1 = np.argmin(np.linalg.norm(p1[:, None] - xy[None], axis=2), axis=1)
        i2 = np.argmin(np.linalg.norm(p2[:, None] - xy[None], axis=2), axis=1)
        gap = ((s[i2] - s[i1]) + lap / 2) % lap - lap / 2
        np.savez(path + '.meta.npz', gap=gap.astype(np.float32), kappa=kappa_track[i1].astype(np.float32))
        done += 1
    print(f'{args.csv_dir}: annotated {done}, no bag found {missing}')


if __name__ == '__main__':
    main()

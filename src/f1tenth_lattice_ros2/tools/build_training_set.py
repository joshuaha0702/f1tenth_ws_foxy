#!/usr/bin/env python3
"""Build per-car training CSV folders from validated episode bags.

Reads the summary written by validate_episodes.py, keeps clean-episode bags
whose car row is `usable`, and extracts them with extract_bag_csv.py into
<out>/<car>/<map>_<episode>.csv (the folder train.py --data_path expects).

    build_training_set.py <summary.csv> <out dir> [--cars car1 car2] [--exclude-maps Simple_v07 ...]
"""

import argparse
import csv
import os
import shutil
import tempfile

from extract_bag_csv import extract_bag_to_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('summary', help='CSV written by validate_episodes.py --summary')
    parser.add_argument('out_dir')
    parser.add_argument('--cars', nargs='+', default=['car1', 'car2'])
    parser.add_argument('--exclude-maps', nargs='*', default=[],
                        help='maps held out from training (matched against the bag path)')
    args = parser.parse_args()

    with open(args.summary) as f:
        rows = list(csv.DictReader(f))
    counts = {car: {'rows': 0, 'episodes': 0, 'skipped_unusable': 0, 'skipped_map': 0}
              for car in args.cars}
    for row in rows:
        car = row['car']
        if car not in args.cars or '/clean/' not in row['bag']:
            continue
        parts = row['bag'].rstrip('/').split('/')
        map_name, episode = parts[-3], parts[-1]
        if map_name in args.exclude_maps:
            counts[car]['skipped_map'] += 1
            continue
        if row['usable'] != 'True':
            counts[car]['skipped_unusable'] += 1
            continue
        car_dir = os.path.join(args.out_dir, car)
        os.makedirs(car_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            written = extract_bag_to_csv(row['bag'], save_dir=tmp, robot_name=car)
            if not written:
                continue
            src = os.path.join(tmp, f'{car}_extracted_{episode}.csv')
            shutil.move(src, os.path.join(car_dir, f'{map_name}_{episode}.csv'))
        counts[car]['rows'] += written
        counts[car]['episodes'] += 1
    for car, c in counts.items():
        print(f'{car}: {c["episodes"]} episodes, {c["rows"]} rows '
              f'(skipped: {c["skipped_unusable"]} unusable, {c["skipped_map"]} held-out map)')


if __name__ == '__main__':
    main()

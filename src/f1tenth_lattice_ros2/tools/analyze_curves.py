#!/usr/bin/env python3
"""Curve-tracking analysis of head-to-head evaluation runs (ego = car1).

For every ego odom sample the raceline curvature at the ego's projected position puts it in a
curvature class. Only solo driving is used (leader more than --solo-gap m away along the
raceline, after the first second, before any contact), so curve handling is not mixed with
overtaking. Per run and class it reports:
  steer jitter        |steer - 0.2 s moving average| (fast steering oscillation)
  yaw-rate wobble     |yaw rate - 0.5 s moving average| (body oscillation the eye sees)
  lateral offset      mean (+ = inside of the curve), std, and its 1-3 Hz component
                      (|offset - 1 s moving average|, i.e. weaving around the line)
  steer ratio         ego steer / geometric raceline steer atan(L*kappa) (<1 understeer)
  steer at limit      share of commands at |steer| >= 0.5 rad
  speed

    analyze_curves.py <out dir> <run dir> [<run dir> ...] [--maps-dir ...]
"""

import argparse
import glob
import os
import re

import numpy as np
from rosbags.rosbag2 import Reader

from analyze_collisions import load_raceline, parse_log, wall_to_sim
from validate_episodes import make_typestore, stamp_ns

WHEELBASE = 0.3302
CLASSES = ('straight', 'entry', 'curve', 'exit')   # |kappa| < 0.05 / ramping up / >= 0.25 / ramping down


def moving_average(x, n):
    if len(x) < n:
        return np.full_like(x, np.mean(x))
    k = np.ones(n) / n
    pad = n // 2
    return np.convolve(np.pad(x, (pad, n - 1 - pad), mode='edge'), k, mode='valid')


def read(bag, typestore):
    o1, o2, dr, clock = [], [], [], []
    with Reader(bag) as reader:
        for conn, bag_ns, raw in reader.messages():
            if conn.topic not in ('/car1/odom', '/car2/odom', '/car1/drive', '/clock'):
                continue
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            if conn.topic == '/clock':
                clock.append((bag_ns, stamp_ns(msg.clock)))
            elif conn.topic == '/car1/drive':
                dr.append((stamp_ns(msg.header.stamp), msg.drive.steering_angle, msg.drive.speed))
            else:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                (o1 if conn.topic == '/car1/odom' else o2).append(
                    (stamp_ns(msg.header.stamp), p.x, p.y, yaw, msg.twist.twist.linear.x))
    return [np.array(a, dtype=float) for a in (o1, o2, dr, clock)]


def episode_samples(bag, info, track, kappa_track, typestore, solo_gap):
    o1, o2, dr, clock = read(bag, typestore)
    if len(o1) < 100 or len(dr) < 100:
        return None
    xy, s_track, lap, tangent = track
    t0 = info.get('t_start', o1[0, 0] * 1e-9)
    t = o1[:, 0] * 1e-9 - t0
    idx = np.argmin(np.linalg.norm(o1[:, None, 1:3] - xy[None], axis=2), axis=1)
    rel = o1[:, 1:3] - xy[idx]
    lat = tangent[idx, 0] * rel[:, 1] - tangent[idx, 1] * rel[:, 0]
    kappa = kappa_track[idx]
    n = len(kappa_track)
    ramp = np.abs(kappa_track[(idx + 5) % n]) - np.abs(kappa_track[(idx - 5) % n])   # 1 m ahead vs behind
    k = np.abs(kappa)
    phase = np.where(k < 0.05, 0, np.where(k >= 0.25, 2, np.where(ramp > 0, 1, 3)))
    # leader gap along the raceline
    x2 = np.interp(t, o2[:, 0] * 1e-9 - t0, o2[:, 1]); y2 = np.interp(t, o2[:, 0] * 1e-9 - t0, o2[:, 2])
    idx2 = np.argmin(np.linalg.norm(np.column_stack((x2, y2))[:, None] - xy[None], axis=2), axis=1)
    gap = ((s_track[idx2] - s_track[idx]) + lap / 2) % lap - lap / 2
    yaw = np.unwrap(o1[:, 3])
    yaw_rate = (np.interp(t + 0.025, t, yaw) - np.interp(t - 0.025, t, yaw)) / 0.05
    td = dr[:, 0] * 1e-9 - t0
    steer = np.interp(t, td, dr[:, 1])
    steer_jitter = np.abs(dr[:, 1] - moving_average(dr[:, 1], 20))
    jitter = np.interp(t, td, steer_jitter)
    wobble = np.abs(yaw_rate - moving_average(yaw_rate, 50))
    weave = np.abs(lat - moving_average(lat, 100))
    t_end = t[-1]
    if 'wall' in info:
        t_end = wall_to_sim(clock, info['wall']) - t0 - 0.5
    ok = (t > 1.0) & (t < t_end) & (np.abs(gap) > solo_gap)
    inside = lat * np.sign(kappa)          # + = towards the inside of the curve
    geom = np.arctan(WHEELBASE * kappa)
    return dict(phase=phase[ok], s=s_track[idx][ok], lat=lat[ok], kappa=kappa[ok], jitter=jitter[ok], wobble=wobble[ok], weave=weave[ok],
                inside=inside[ok], steer=steer[ok], geom=geom[ok], speed=o1[ok, 4])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('out_dir')
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--maps-dir', default=os.path.join(os.path.dirname(__file__), '..', 'maps'))
    parser.add_argument('--solo-gap', type=float, default=4.0)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    typestore = make_typestore()
    lines = []

    def p(s=''):
        lines.append(s)
        print(s)

    results = {}
    for run in args.runs:
        acc = {}
        for map_dir in sorted(glob.glob(os.path.join(run, 'Simple*', ''))):
            name = os.path.basename(os.path.dirname(map_dir))
            rl_path = os.path.join(args.maps_dir, name, 'raceline1.csv')
            track = load_raceline(rl_path)
            raw = np.loadtxt(rl_path, delimiter=';', comments='#')
            kappa_track = raw[:len(track[0]), 4]
            log = parse_log(os.path.join(run, 'logs', f'{name}.log'))
            for bag in sorted(glob.glob(os.path.join(map_dir, '*', 'ep*'))):
                info = log.get(os.path.basename(bag).split('_')[0], {})
                ep = episode_samples(bag, info, track, kappa_track, typestore, args.solo_gap)
                if ep is None:
                    continue
                ep['map'] = np.full(len(ep['s']), name)
                for k, v in ep.items():
                    acc.setdefault(k, []).append(v)
        results[os.path.basename(run.rstrip('/'))] = {k: np.concatenate(v) for k, v in acc.items()}

    p(f'solo driving only (leader > {args.solo_gap} m away), 100 Hz samples; + inside = towards curve centre')
    p(f'{"run":34s} {"class":9s} {"time s":>7s} {"steer jit mrad":>14s} {"yaw wobble":>10s} '
      f'{"weave cm":>8s} {"offset cm":>9s} {"offset std":>10s} {"steer ratio":>11s} {"at limit":>8s} {"speed":>6s}')
    for run, r in results.items():
        for ci, cname in enumerate(CLASSES):
            sel = r['phase'] == ci
            if sel.sum() < 100:
                continue
            ratio = np.median(r['steer'][sel] / r['geom'][sel]) if cname == 'curve' else np.nan
            p(f'{run:34s} {cname:9s} {sel.sum() / 100:7.0f} {r["jitter"][sel].mean() * 1000:14.2f} '
              f'{r["wobble"][sel].mean():10.3f} {r["weave"][sel].mean() * 100:8.2f} '
              f'{r["inside"][sel].mean() * 100:+9.1f} {r["inside"][sel].std() * 100:10.1f} '
              f'{ratio:11.2f} {np.mean(np.abs(r["steer"][sel]) >= 0.5):8.1%} {r["speed"][sel].mean():6.2f}')
        p()
    with open(os.path.join(args.out_dir, 'curves_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    edges = np.array([0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 3.0])
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for run, r in results.items():
        k = np.abs(r['kappa'])
        for ax, key, scale, title in ((axes[0], 'jitter', 1000, 'steer jitter (mrad)'),
                                      (axes[1], 'wobble', 1, 'yaw-rate wobble (rad/s)'),
                                      (axes[2], 'weave', 100, 'lateral weave (cm)'),
                                      (axes[3], 'inside', 100, 'lateral offset, + inside (cm)')):
            vals = [r[key][(k >= a) & (k < b)].mean() * scale if ((k >= a) & (k < b)).sum() > 50 else np.nan
                    for a, b in zip(edges[:-1], edges[1:])]
            ax.plot(centers, vals, marker='o', label=run)
            ax.set_title(title); ax.set_xlabel('|raceline curvature| (1/m)'); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'curves_by_curvature.png'), dpi=120)

    # lateral offset along the lap, per map: where on the track each run leaves the raceline
    maps = sorted({m for r in results.values() for m in np.unique(r['map'])})
    fig, axes = plt.subplots(len(maps), 1, figsize=(16, 3.2 * len(maps)), squeeze=False)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, name in zip(axes[:, 0], maps):
        for ci, (run, r) in enumerate(results.items()):
            sel = r['map'] == name
            s_, lat_, kap = r['s'][sel], r['lat'][sel], r['kappa'][sel]
            bins = np.arange(0, s_.max() + 0.2, 0.2)
            which = np.digitize(s_, bins)
            mean = np.array([lat_[which == b].mean() if (which == b).sum() > 5 else np.nan for b in range(1, len(bins))])
            p10 = np.array([np.percentile(lat_[which == b], 10) if (which == b).sum() > 5 else np.nan for b in range(1, len(bins))])
            p90 = np.array([np.percentile(lat_[which == b], 90) if (which == b).sum() > 5 else np.nan for b in range(1, len(bins))])
            c = colors[ci % len(colors)]
            ax.plot(bins[:-1], mean * 100, color=c, label=f'{run} mean')
            ax.fill_between(bins[:-1], p10 * 100, p90 * 100, color=c, alpha=0.15)
        ax2 = ax.twinx()
        r0 = next(iter(results.values()))
        sel = r0['map'] == name
        order = np.argsort(r0['s'][sel])
        ax2.plot(r0['s'][sel][order], r0['kappa'][sel][order], color='k', lw=0.6, alpha=0.5)
        ax2.set_ylabel('curvature (1/m)', fontsize=8)
        ax.set_title(f'{name}: lateral offset from raceline (+ left), mean and p10-p90, solo driving', fontsize=9)
        ax.set_ylabel('cm'); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')
    axes[-1, 0].set_xlabel('raceline arc length s (m)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'lateral_offset_along_track.png'), dpi=110)


if __name__ == '__main__':
    main()

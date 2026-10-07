#!/usr/bin/env python3
"""Paired collision analysis of two head-to-head evaluation runs (e.g. End2Race ego vs lattice ego).

Both runs must come from run_episodes.sh with the same maps, seed base and episode count, so
episode k of a map starts from the same state in both. For every episode this tool:
  1. pairs the outcomes (both / model-only / baseline-only / neither collided),
  2. finds the contact time (episode manager '[collision]' log line -> sim time via /clock in
     the bag) and the geometry at contact: leader position in the ego frame, speeds, contacting
     links, lateral offsets from the raceline -> a contact type,
  3. describes the 3 s before contact: gap along the raceline, closing speed, minimum
     time-to-collision, ego braking, and where the ego path left the baseline ego path,
  4. measures overtaking behaviour in every episode: gap at which the ego starts to move
     sideways out of the leader's line, and ego speed while closing in.
Writes episodes.csv, summary.txt and one figure per model-only collision to <out>.

    analyze_collisions.py <model run> <baseline run> <out dir> [--maps-dir ...]
"""

import argparse
import csv
import glob
import os
import re
from collections import Counter

import numpy as np
from rosbags.rosbag2 import Reader

from validate_episodes import make_typestore, stamp_ns

CAR_LENGTH = 0.58
CAR_WIDTH = 0.30
PRE_WINDOW_S = 3.0
SIDE_STEP_M = 0.25        # |d_ego - d_leader| that counts as being out of the leader's line


# ---------------------------------------------------------------- inputs

def load_raceline(path):
    rl = np.loadtxt(path, delimiter=';', comments='#')
    xy = rl[:-1, 1:3] if np.allclose(rl[0, 1:3], rl[-1, 1:3]) else rl[:, 1:3]
    seg = np.linalg.norm(np.diff(np.vstack((xy, xy[:1])), axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)[:-1]))
    tangent = np.roll(xy, -1, axis=0) - xy
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return xy, s, float(seg.sum()), tangent


def project(track, points):
    """Unwrapped arc length and signed lateral offset (+ = left of the raceline)."""
    xy, s_track, lap, tangent = track
    idx = np.argmin(np.linalg.norm(points[:, None, :] - xy[None, :, :], axis=2), axis=1)
    # project onto the segment before or after the nearest raceline point (points are 0.2 m
    # apart, so snapping to the nearest point makes the gap jump in 0.2 m steps)
    seg_len = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    best_s = np.empty(len(points)); best_lat = np.empty(len(points)); best_d = np.full(len(points), np.inf)
    for k in (idx - 1) % len(xy), idx:
        rel = points - xy[k]
        along = np.clip(np.einsum('ij,ij->i', rel, tangent[k]), 0.0, seg_len[k])
        lat = tangent[k, 0] * rel[:, 1] - tangent[k, 1] * rel[:, 0]
        dist = np.hypot(np.einsum('ij,ij->i', rel, tangent[k]) - along, lat)
        better = dist < best_d
        best_d[better] = dist[better]; best_s[better] = (s_track[k] + along)[better]; best_lat[better] = lat[better]
    s, lat = best_s, best_lat
    s = np.concatenate(([s[0]], s[0] + np.cumsum((np.diff(s) + lap / 2) % lap - lap / 2)))
    return s, lat


def read_bag(bag, typestore):
    data = {'odom1': [], 'odom2': [], 'drive1': [], 'clock': []}
    with Reader(bag) as reader:
        for conn, bag_ns, raw in reader.messages():
            topic = conn.topic
            if topic not in ('/car1/odom', '/car2/odom', '/car1/drive', '/clock'):
                continue
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            if topic == '/clock':
                data['clock'].append((bag_ns, stamp_ns(msg.clock)))
            elif topic == '/car1/drive':
                data['drive1'].append((stamp_ns(msg.header.stamp), msg.drive.steering_angle,
                                       msg.drive.speed))
            else:
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                key = 'odom1' if topic == '/car1/odom' else 'odom2'
                data[key].append((stamp_ns(msg.header.stamp), p.x, p.y, yaw,
                                  msg.twist.twist.linear.x))
    return {k: np.array(v, dtype=float) for k, v in data.items()}


def parse_log(path):
    """{episode label: {'t_start': sim s, 'wall': contact wall s, 'links': str, 'source': car}}
    for the attempt that was kept (retried attempts are dropped)."""
    out, cur, attempt = {}, None, {}
    line_re = re.compile(r'\[episode_manager\]: (.*)$')
    stamp_re = re.compile(r'\[(\d+\.\d+)\] \[episode_manager\]')
    with open(path, errors='ignore') as f:
        for line in f:
            m = line_re.search(line)
            if not m:
                continue
            text = m.group(1)
            wall = float(stamp_re.search(line).group(1))
            m = re.match(r'\[(ep\d+)\] recording started — t_start=([\d.]+)', text)
            if m:
                cur, attempt = m.group(1), {'t_start': float(m.group(2))}
                continue
            m = re.match(r'\[collision\] (car\d): (.*)', text)
            if m and cur and 'wall' not in attempt:
                attempt.update(wall=wall, source=m.group(1), links=m.group(2))
                continue
            m = re.match(r'\[(ep\d+)\] (CLEAN|COLLISION)', text)
            if m and cur == m.group(1):
                out[cur] = attempt
                cur = None
    return out


def wall_to_sim(clock, wall_s):
    wall_ns = wall_s * 1e9
    return float(np.interp(wall_ns, clock[:, 0], clock[:, 1])) * 1e-9


# ---------------------------------------------------------------- per-episode analysis

def contact_type(dx, dy, links):
    """dx, dy: leader centre in the ego frame (m)."""
    if dx > CAR_LENGTH * 0.7 and abs(dy) < CAR_WIDTH * 1.2:
        return 'rear_end'           # ego drove into the leader's tail
    if dx < -CAR_LENGTH * 0.7 and abs(dy) < CAR_WIDTH * 1.2:
        return 'cut_in'             # ego (already ahead) moved into the leader's path
    return 'side'                   # side by side while passing


def analyse_episode(bag, info, track, typestore):
    d = read_bag(bag, typestore)
    o1, o2, dr = d['odom1'], d['odom2'], d['drive1']
    if len(o1) < 10 or len(o2) < 10:
        return None
    t0 = info['t_start']
    t = (o1[:, 0] * 1e-9) - t0
    # leader state at the ego odom times
    t2 = (o2[:, 0] * 1e-9) - t0
    x2 = np.interp(t, t2, o2[:, 1]); y2 = np.interp(t, t2, o2[:, 2]); v2 = np.interp(t, t2, o2[:, 4])
    yaw2 = np.interp(t, t2, np.unwrap(o2[:, 3]))
    s1, lat1 = project(track, o1[:, 1:3])
    s2, lat2 = project(track, np.column_stack((x2, y2)))
    lap = track[2]
    gap = ((s2 - s1) + lap / 2) % lap - lap / 2          # >0: leader ahead along the raceline
    v1 = o1[:, 4]
    res = {'t': t, 'x1': o1[:, 1], 'y1': o1[:, 2], 'x2': x2, 'y2': y2, 'lat1': lat1, 'lat2': lat2,
           'gap': gap, 'v1': v1, 'v2': v2}
    if len(dr):
        res['td'] = dr[:, 0] * 1e-9 - t0
        res['steer'] = dr[:, 1]
        res['cmd_speed'] = dr[:, 2]

    row = {}
    # overtaking behaviour: first time the ego is out of the leader's line while closing in
    behind = (gap > 0) & (gap < 4.0)
    out_of_line = np.abs(lat1 - lat2) > SIDE_STEP_M
    start = np.where(behind & out_of_line)[0]
    row['sidestep_gap_m'] = float(gap[start[0]]) if len(start) else np.nan
    close = behind & (gap < 1.5) & ~out_of_line
    row['speed_in_line_close_mps'] = float(v1[close].mean()) if close.any() else np.nan
    # lateral clearance while alongside: leader centre within one car length along the ego heading
    yaw1 = o1[:, 3]
    rx, ry = x2 - o1[:, 1], y2 - o1[:, 2]
    dx_all = np.cos(yaw1) * rx + np.sin(yaw1) * ry
    dy_all = -np.sin(yaw1) * rx + np.cos(yaw1) * ry
    alongside = np.abs(dx_all) < CAR_LENGTH
    if 'wall' in info:
        alongside &= t < wall_to_sim(d['clock'], info['wall']) - t0 - 0.05   # before contact
    row['alongside_s'] = float(alongside.sum() * np.median(np.diff(t))) if alongside.any() else 0.0
    row['alongside_min_dy_m'] = float(np.abs(dy_all[alongside]).min()) if alongside.any() else np.nan
    row['min_gap_in_line_m'] = float(gap[behind & ~out_of_line].min()) if (behind & ~out_of_line).any() else np.nan

    if 'wall' in info:
        tc = wall_to_sim(d['clock'], info['wall']) - t0
        i = int(np.clip(np.searchsorted(t, tc), 0, len(t) - 1))
        yaw1 = o1[i, 3]
        rx, ry = x2[i] - o1[i, 1], y2[i] - o1[i, 2]
        dx = np.cos(yaw1) * rx + np.sin(yaw1) * ry
        dy = -np.sin(yaw1) * rx + np.cos(yaw1) * ry
        pre = (t >= tc - PRE_WINDOW_S) & (t <= tc)
        # >0: gap shrinking; 0.2 s central difference to keep odom jitter out
        closing = -(np.interp(t + 0.1, t, gap) - np.interp(t - 0.1, t, gap)) / 0.2
        ttc = np.where((gap > 0) & (closing > 0.05), (gap - CAR_LENGTH) / np.maximum(closing, 1e-6), np.inf)
        row.update(
            t_contact=tc, links=info['links'], source=info['source'],
            ego_part=('wheel' if 'car1::' in info['links'] and 'wheel' in info['links'].split('<->')[0] else 'body'),
            leader_dx=dx, leader_dy=dy, rel_yaw_deg=float(np.degrees((yaw2[i] - yaw1 + np.pi) % (2 * np.pi) - np.pi)),
            type=contact_type(dx, dy, info['links']),
            v_ego=float(v1[i]), v_leader=float(v2[i]), gap_at_contact=float(gap[i]),
            lat_ego=float(lat1[i]), lat_leader=float(lat2[i]),
            closing_max_mps=float(closing[pre].max()) if pre.any() else np.nan,
            min_ttc_s=float(ttc[pre].min()) if pre.any() else np.nan,
            ego_speed_drop_mps=float(v1[pre].max() - v1[i]) if pre.any() else np.nan,
        )
        if 'td' in res:
            j = int(np.clip(np.searchsorted(res['td'], tc), 0, len(res['td']) - 1))
            row['steer_at_contact'] = float(res['steer'][j])
            row['cmd_speed_at_contact'] = float(res['cmd_speed'][j])
    return row, res


def divergence(model_res, base_res, t_end):
    """First time (s) the two ego paths are > 0.3 m apart, and the mean lateral offsets after."""
    t = model_res['t']
    sel = t <= t_end
    bx = np.interp(t[sel], base_res['t'], base_res['x1'])
    by = np.interp(t[sel], base_res['t'], base_res['y1'])
    dist = np.hypot(model_res['x1'][sel] - bx, model_res['y1'][sel] - by)
    k = np.where(dist > 0.3)[0]
    return (float(t[sel][k[0]]) if len(k) else np.nan), float(dist.max()) if len(dist) else np.nan


# ---------------------------------------------------------------- plots

def plot_pair(path, title, track, model, base, row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tc = row.get('t_contact', model['t'][-1])
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(*np.vstack((track[0], track[0][:1])).T, color='0.8', lw=8, zorder=0)
    m = model['t'] <= tc + 0.2
    bm = base['t'] <= tc + 0.2
    ax.plot(model['x2'][m], model['y2'][m], color='0.4', lw=1.5, label='leader (model run)')
    ax.plot(base['x1'][bm], base['y1'][bm], color='tab:blue', lw=1.5, label='ego lattice (same start)')
    ax.plot(model['x1'][m], model['y1'][m], color='tab:red', lw=1.5, label='ego End2Race')
    ax.plot(model['x1'][0], model['y1'][0], 'o', color='tab:red')
    ax.plot(model['x2'][0], model['y2'][0], 'o', color='0.4')
    if 't_contact' in row:
        i = np.searchsorted(model['t'], tc)
        ax.plot(model['x1'][i], model['y1'][i], 'kx', ms=12, mew=3, label='contact')
        x, y = model['x1'][i], model['y1'][i]
        ax.set_xlim(x - 4, x + 4); ax.set_ylim(y - 4, y + 4)
    ax.set_aspect('equal'); ax.legend(fontsize=8, loc='best'); ax.set_title(title, fontsize=9)

    lo = max(0.0, tc - 5.0)
    for k, (key, label) in enumerate((('gap', 'gap along raceline (m), leader ahead > 0'),
                                      ('lat1', 'ego lateral offset (m), + = left'),
                                      ('v1', 'ego speed (m/s)'))):
        a = fig.add_subplot(3, 2, 2 * k + 2)
        mm = (model['t'] >= lo) & (model['t'] <= tc + 0.5)
        bb = (base['t'] >= lo) & (base['t'] <= tc + 0.5)
        a.plot(model['t'][mm], model[key][mm], color='tab:red', label='End2Race run')
        a.plot(base['t'][bb], base[key][bb], color='tab:blue', label='lattice run')
        if key == 'lat1':
            a.plot(model['t'][mm], model['lat2'][mm], color='0.4', ls='--', label='leader (End2Race run)')
        if key == 'v1':
            a.plot(model['t'][mm], model['v2'][mm], color='0.4', ls='--', label='leader (End2Race run)')
        if key == 'gap':
            a.axhline(CAR_LENGTH, color='k', lw=0.5, ls=':')
        a.axvline(tc, color='k', lw=0.8)
        a.set_ylabel(label, fontsize=8); a.grid(alpha=0.3)
        if k == 0:
            a.legend(fontsize=7)
    a.set_xlabel('time since episode start (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('model_run')
    parser.add_argument('baseline_run')
    parser.add_argument('out_dir')
    parser.add_argument('--maps-dir', default=os.path.join(os.path.dirname(__file__), '..', 'maps'))
    args = parser.parse_args()
    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)
    typestore = make_typestore()

    rows = []
    for map_dir in sorted(glob.glob(os.path.join(args.model_run, 'Simple*', ''))):
        name = os.path.basename(os.path.dirname(map_dir))
        track = load_raceline(os.path.join(args.maps_dir, name, 'raceline1.csv'))
        logs = {run: parse_log(os.path.join(run, 'logs', f'{name}.log'))
                for run in (args.model_run, args.baseline_run)}
        bags = {run: {os.path.basename(b).split('_')[0]: b
                      for b in glob.glob(os.path.join(run, name, '*', 'ep*'))}
                for run in (args.model_run, args.baseline_run)}
        for label in sorted(set(bags[args.model_run]) & set(bags[args.baseline_run])):
            res = {}
            row = {'map': name, 'episode': label}
            for tag, run in (('model', args.model_run), ('base', args.baseline_run)):
                bag = bags[run][label]
                out = analyse_episode(bag, logs[run].get(label, {'t_start': 0.0}), track, typestore)
                if out is None:
                    break
                r, res[tag] = out
                row[f'{tag}_collision'] = os.path.basename(os.path.dirname(bag)) == 'collision'
                row.update({f'{tag}_{k}': v for k, v in r.items()})
            else:
                mc, bc = row['model_collision'], row['base_collision']
                row['pair'] = ('both' if mc and bc else 'model_only' if mc else
                               'base_only' if bc else 'neither')
                t_end = row.get('model_t_contact', res['model']['t'][-1])
                row['diverge_t'], row['diverge_max_m'] = divergence(res['model'], res['base'], t_end)
                rows.append(row)
                if mc and not bc:
                    plot_pair(os.path.join(args.out_dir, 'figures', f'{name}_{label}.png'),
                              f'{name} {label}: End2Race collided ({row.get("model_type")}), lattice clean',
                              track, res['model'], res['base'], {k[6:]: v for k, v in row.items()
                                                                 if k.startswith('model_')})

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('map', 'episode', 'pair'), k))
    with open(os.path.join(args.out_dir, 'episodes.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = []
    def p(s=''):
        lines.append(s)
        print(s)
    nanmean = lambda v: float(np.nanmean(v)) if len(v) and not np.all(np.isnan(v)) else np.nan
    p(f'episodes paired: {len(rows)}')
    for scope in sorted({r['map'] for r in rows}) + ['ALL']:
        sel = rows if scope == 'ALL' else [r for r in rows if r['map'] == scope]
        c = Counter(r['pair'] for r in sel)
        p(f'{scope:17s} both {c["both"]:3d} | model-only {c["model_only"]:3d} | base-only {c["base_only"]:3d} | neither {c["neither"]:3d}')
    for tag, who in (('model', 'End2Race'), ('base', 'lattice')):
        col = [r for r in rows if r[f'{tag}_collision'] and f'{tag}_type' in r]
        p(f'\n{who} collisions by contact type: {dict(Counter(r[f"{tag}_type"] for r in col))}')
        for typ in ('rear_end', 'side', 'cut_in'):
            s = [r for r in col if r[f'{tag}_type'] == typ]
            if not s:
                continue
            g = lambda k: np.array([r[f'{tag}_{k}'] for r in s], dtype=float)
            p(f'  {typ:9s} n={len(s):2d} | ego {nanmean(g("v_ego")):.2f} m/s, leader {nanmean(g("v_leader")):.2f} m/s'
              f' | max closing {nanmean(g("closing_max_mps")):.2f} m/s | min TTC {np.nanmedian(g("min_ttc_s")):.2f} s (median)'
              f' | ego speed drop in last {PRE_WINDOW_S:.0f} s {nanmean(g("ego_speed_drop_mps")):.2f} m/s'
              f' | leader dx {nanmean(g("leader_dx")):+.2f} dy {nanmean(g("leader_dy")):+.2f} m')
    p('\nlateral clearance while alongside (episodes with >= 0.1 s alongside before any contact):')
    for tag, who in (('model', 'End2Race'), ('base', 'lattice')):
        v = np.array([r[f'{tag}_alongside_min_dy_m'] for r in rows if r.get(f'{tag}_alongside_s', 0) >= 0.1], dtype=float)
        if len(v):
            p(f'  {who:9s} n={len(v):3d} | min |dy| median {np.median(v):.2f} m, p10 {np.percentile(v, 10):.2f} m |'
              f' share < 0.40 m {np.mean(v < 0.40):.0%}, < 0.35 m {np.mean(v < 0.35):.0%}')
    p('\novertaking behaviour (all episodes, mean):')
    for tag, who in (('model', 'End2Race'), ('base', 'lattice')):
        g = lambda k: np.array([r.get(f'{tag}_{k}', np.nan) for r in rows], dtype=float)
        p(f'  {who:9s} side-step starts at gap {nanmean(g("sidestep_gap_m")):.2f} m'
          f' (never side-stepped while behind: {int(np.isnan(g("sidestep_gap_m")).sum())}) |'
          f' speed in leader line at gap<1.5 m {nanmean(g("speed_in_line_close_mps")):.2f} m/s |'
          f' min gap in line {nanmean(g("min_gap_in_line_m")):.2f} m')
    mo = [r for r in rows if r['pair'] == 'model_only']
    if mo:
        p(f'\nmodel-only collisions: ego path left the lattice path {nanmean([r["diverge_t"] for r in mo]):.2f} s'
          f' after start on average; contact at {nanmean([r["model_t_contact"] for r in mo]):.2f} s')
        for r in mo:
            p(f'  {r["map"]:17s} {r["episode"]} {r.get("model_type", "?"):9s} t={r.get("model_t_contact", np.nan):5.2f}s'
              f' diverge={r["diverge_t"]:5.2f}s dx={r.get("model_leader_dx", np.nan):+.2f} dy={r.get("model_leader_dy", np.nan):+.2f}'
              f' v_ego={r.get("model_v_ego", np.nan):.2f} v_lead={r.get("model_v_leader", np.nan):.2f}'
              f' TTCmin={r.get("model_min_ttc_s", np.nan):.2f}s lat_ego={r.get("model_lat_ego", np.nan):+.2f} lat_lead={r.get("model_lat_leader", np.nan):+.2f}')
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

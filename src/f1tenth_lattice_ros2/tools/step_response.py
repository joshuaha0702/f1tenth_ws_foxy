#!/usr/bin/env python3
"""Longitudinal step-response measurement (Gazebo and real car).

Publishes AckermannDriveStamped speed steps (steering 0) at 100 Hz, records
(t, cmd, odom vx) at the odom rate, then writes a CSV + PNG and prints per-step
metrics:

  tau63   time to cover 63.2 % of the step (measured start speed v0 -> commanded level)
  t90     time to cover 90 % of the step
          (n/a if never reached within the hold, or if v0 is already within 25 % of the
          commanded change of the new level, i.e. the previous step never settled)
  peak_a  peak accel (+) / decel (-) [m/s^2]: speed smoothed by a 0.1 s centered moving
          average, then differentiated across 0.1 s (central difference)
  final   mean measured speed over the last 0.2 s of the hold (steady-state check)
  ovs     overshoot beyond the commanded level, % of the step size
  v@+1s   residual speed 1 s after a 0 command (only for steps to 0)

Steps: ``--steps 0:2,2:0,0:2,2:1,1:2,2:0`` -- each ``a:b`` holds level ``b`` for
``--hold`` seconds; if the current level is not ``a`` a level-``a`` hold is
inserted first (not analysed). The car is held at 0 for ``--pre`` seconds
before the first step, and 0 is commanded on exit.

Gazebo:   python3 step_response.py --use-sim-time --out data/step_response/gazebo_p1.csv
Real car: python3 step_response.py --drive-topic /drive --odom-topic /odom --out real.csv
          (use_sim_time stays false; see data/step_response/README.md)

Time base: odom header stamps (sim time in Gazebo) drive both the step schedule
and the samples, so the result does not depend on /clock granularity or on the
clock offset between machines. Stamps of 0 (or --receive-time) fall back to the
receive time on the node clock. If odom stops for --odom-timeout s the tool
commands 0 and aborts.
"""
import argparse
import csv
import math
import os
import sys
import time

try:  # ROS imports are optional so that --analyze also works off-robot
    import rclpy
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from ackermann_msgs.msg import AckermannDriveStamped
    from nav_msgs.msg import Odometry
except ImportError:  # pragma: no cover
    rclpy = None
    Node = object


def parse_steps(text):
    steps = []
    for tok in text.split(','):
        tok = tok.strip()
        if not tok:
            continue
        a, b = tok.split(':')
        steps.append((float(a), float(b)))
    if not steps:
        raise ValueError('empty --steps')
    return steps


def build_schedule(steps, hold, pre):
    """Return list of segments (t_start, t_end, level, step_or_None) relative to start."""
    segs = []
    t = 0.0
    level = 0.0
    if pre > 0:
        segs.append((t, t + pre, 0.0, None))
        t += pre
    for a, b in steps:
        if abs(a - level) > 1e-9:
            segs.append((t, t + hold, a, None))  # move to the 'from' level first
            t += hold
            level = a
        segs.append((t, t + hold, b, (a, b)))
        t += hold
        level = b
    return segs


class StepResponse(Node):
    """Schedule and samples both run on the odom time base (header stamps).

    Gazebo publishes /clock at only ~10 Hz and runs at RTF ~1.25 here, so the node
    clock is too coarse to time 100 Hz steps; the 100 Hz odom stamps are not. The
    drive command is published on every odom message (odom rate) and additionally
    from a steady wall-clock timer at --rate, so the command keeps flowing at
    100 Hz real time on the car and never depends on the coarse sim clock.
    """

    def __init__(self, args):
        overrides = [Parameter('use_sim_time', Parameter.Type.BOOL, True)] if args.use_sim_time else []
        super().__init__('step_response', parameter_overrides=overrides)
        self.args = args
        self.segs = build_schedule(parse_steps(args.steps), args.hold, args.pre)
        self.total = self.segs[-1][1]
        self.pub = self.create_publisher(AckermannDriveStamped, args.drive_topic, 10)
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        # BEST_EFFORT subscription matches both reliable and best-effort publishers.
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, qos)
        self.t0 = None          # schedule start on the odom time base [s]
        self.cmd = 0.0          # level currently commanded
        self.rows = []          # (t_rel, cmd, vx)
        self.n_odom = 0
        self.warned_stamp = False
        self.done = False
        self.aborted = False
        self.last_odom_wall = None
        self.create_timer(1.0 / args.rate, self.on_timer,
                          clock=Clock(clock_type=ClockType.STEADY_TIME))

    def odom_time(self, msg):
        st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if st > 0.0 and not self.args.receive_time:
            return st
        if not self.warned_stamp and not self.args.receive_time:
            self.get_logger().warn('odom header stamp is zero: using receive time on the node clock')
            self.warned_stamp = True
        return self.get_clock().now().nanoseconds * 1e-9

    def publish(self, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = 0.0
        self.pub.publish(msg)

    def level_at(self, t_rel):
        for s, e, lvl, _ in self.segs:
            if s <= t_rel < e:
                return lvl
        return 0.0

    def on_timer(self):
        if self.t0 is not None and not self.done:
            # odom watchdog (safety on the real car): no odom -> stop and abort
            if time.monotonic() - self.last_odom_wall > self.args.odom_timeout:
                self.get_logger().error(
                    f'no odom for {self.args.odom_timeout} s: commanding 0 and aborting')
                self.cmd = 0.0
                self.done = True
                self.aborted = True
        self.publish(self.cmd if not self.done else 0.0)

    def on_odom(self, msg):
        self.n_odom += 1
        self.last_odom_wall = time.monotonic()
        if self.done:
            return
        t = self.odom_time(msg)
        if self.t0 is None:
            if self.n_odom < 5:
                return
            self.t0 = t
            self.get_logger().info(
                f'start: {len(self.segs)} segments, {self.total:.1f} s, '
                f'drive={self.args.drive_topic} odom={self.args.odom_topic} '
                f'sim_time={self.args.use_sim_time}')
        t_rel = t - self.t0
        if t_rel < 0.0:
            return
        if t_rel >= self.total:
            self.cmd = 0.0
            self.done = True
            self.publish(0.0)
            return
        self.cmd = self.level_at(t_rel)
        self.publish(self.cmd)
        self.rows.append((t_rel, self.cmd, msg.twist.twist.linear.x))


# ----------------------------------------------------------------------------- analysis

def half_window(t, win):
    """Half window in samples for a time window `win` (from the median sample period)."""
    dts = sorted(b - a for a, b in zip(t, t[1:]) if b > a)
    dt = dts[len(dts) // 2] if dts else 0.01
    return max(1, int(round(win / dt / 2)))


def smooth(t, v, win):
    """Centered moving average over `win` seconds (fixed sample count, so tiny stamp
    jitter cannot change the window membership)."""
    k = half_window(t, win)
    n = len(v)
    out = []
    for i in range(n):
        a, b = max(0, i - k), min(n, i + k + 1)
        out.append(sum(v[a:b]) / (b - a))
    return out


def derivative(t, v, win):
    """Central difference across `win` seconds."""
    k = half_window(t, win)
    n = len(v)
    d = [0.0] * n
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        dt = t[b] - t[a]
        d[i] = (v[b] - v[a]) / dt if dt > 0 else 0.0
    return d


def analyse(rows, segs, smooth_win=0.1):
    t = [r[0] for r in rows]
    v = [r[2] for r in rows]
    vs = smooth(t, v, smooth_win)
    acc = derivative(t, vs, smooth_win)
    results = []
    for s, e, lvl, step in segs:
        if step is None:
            continue
        a, b = step
        idx = [i for i, ti in enumerate(t) if s <= ti < e]
        if len(idx) < 3:
            results.append(dict(step=f'{a:g}->{b:g}', t_start=s))
            continue
        i0 = idx[0]
        # start speed: mean of the 0.05 s before the step (fallback: first sample)
        pre = [v[i] for i, ti in enumerate(t) if s - 0.05 <= ti < s]
        v0 = sum(pre) / len(pre) if pre else v[i0]
        dv = b - v0
        res = dict(step=f'{a:g}->{b:g}', t_start=s, v0=v0)

        # Speed already (almost) at the new level, e.g. the previous step never
        # settled: no meaningful time constant / overshoot for this step.
        no_step = abs(dv) < 0.25 * abs(b - a)

        def t_frac(frac):
            if no_step or abs(dv) < 1e-6:
                return float('nan')
            for i in idx:
                if (v[i] - v0) / dv >= frac:
                    return t[i] - s
            return float('nan')

        res['tau63'] = t_frac(0.632)
        res['t90'] = t_frac(0.9)
        seg_acc = [acc[i] for i in idx]
        res['peak_a'] = max(seg_acc) if dv >= 0 else min(seg_acc)
        tail = [v[i] for i in idx if t[i] >= e - 0.2]
        res['final'] = sum(tail) / len(tail) if tail else float('nan')
        if not no_step and abs(dv) > 1e-6:
            ext = max(v[i] for i in idx) if dv > 0 else min(v[i] for i in idx)
            res['overshoot_pct'] = max(0.0, (ext - b) / dv * 100.0)
        else:
            res['overshoot_pct'] = float('nan')
        res['v_1s'] = float('nan')
        if b == 0.0:
            later = [i for i in idx if t[i] >= s + 1.0]
            if later:
                res['v_1s'] = v[later[0]]
        results.append(res)
    return results, vs, acc


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'n/a'
    return f'{x:.{nd}f}'


def write_outputs(args, rows, segs, results, vs, acc):
    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'cmd', 'vx', 'vx_smooth', 'accel_smooth'])
        for (t, c, v), s_, a_ in zip(rows, vs, acc):
            w.writerow([f'{t:.4f}', f'{c:.3f}', f'{v:.4f}', f'{s_:.4f}', f'{a_:.4f}'])
    stem = os.path.splitext(out)[0]
    keys = ['step', 't_start', 'v0', 'tau63', 't90', 'peak_a', 'final', 'overshoot_pct', 'v_1s']
    with open(stem + '_metrics.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in results:
            w.writerow([r.get(k, '') if k == 'step' else fmt(r.get(k, float('nan')), 4) for k in keys])

    print(f'\nstep response: {len(rows)} samples -> {out}')
    print(f'{"step":>8} {"v0":>6} {"tau63[s]":>9} {"t90[s]":>7} {"peak_a":>8} '
          f'{"final":>6} {"ovs%":>6} {"v@+1s":>6}')
    for r in results:
        print(f'{r["step"]:>8} {fmt(r.get("v0"), 2):>6} {fmt(r.get("tau63")):>9} '
              f'{fmt(r.get("t90")):>7} {fmt(r.get("peak_a"), 2):>8} {fmt(r.get("final"), 2):>6} '
              f'{fmt(r.get("overshoot_pct"), 1):>6} {fmt(r.get("v_1s"), 3):>6}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available: PNG skipped')
        return
    t = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
    ax1.step(t, [r[1] for r in rows], where='post', color='#888888', lw=1.5, label='cmd')
    ax1.plot(t, [r[2] for r in rows], color='#1f77b4', lw=1.2, label='odom vx')
    for r in results:
        if 'tau63' in r and not math.isnan(r['tau63']):
            ax1.axvline(r['t_start'] + r['tau63'], color='#d62728', lw=0.6, ls=':')
    ax1.set_ylabel('speed [m/s]')
    ax1.set_title(args.title or os.path.basename(stem))
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')
    ax2.plot(t, acc, color='#2ca02c', lw=1.0)
    ax2.set_ylabel('accel [m/s²]\n(0.1 s smoothed)')
    ax2.set_xlabel('t [s]')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(stem + '.png', dpi=110)
    plt.close(fig)
    print(f'plot -> {stem}.png')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--drive-topic', default='/car1/drive')
    p.add_argument('--odom-topic', default='/car1/odom')
    p.add_argument('--steps', default='0:2,2:0,2:1,1:2,2:0')
    p.add_argument('--hold', type=float, default=3.0, help='hold time per level [s]')
    p.add_argument('--pre', type=float, default=1.0, help='zero-speed settle time before the first step [s]')
    p.add_argument('--rate', type=float, default=100.0, help='drive publish rate [Hz]')
    p.add_argument('--use-sim-time', action='store_true', help='use /clock (Gazebo)')
    p.add_argument('--out', default='step_response.csv')
    p.add_argument('--title', default='')
    p.add_argument('--receive-time', action='store_true',
                   help='time-stamp odom by receive time instead of its header stamp')
    p.add_argument('--odom-timeout', type=float, default=0.3,
                   help='wall seconds without odom during the run before commanding 0 and aborting')
    p.add_argument('--timeout', type=float, default=30.0,
                   help='abort if no odom/clock within this many wall seconds')
    p.add_argument('--analyze', metavar='CSV', default='',
                   help='no ROS: recompute metrics/plot from an existing CSV (same --steps/--hold/--pre '
                        'as the run); outputs go next to --out (default: next to the CSV)')
    args, ros_args = p.parse_known_args(argv)

    if args.analyze:
        with open(args.analyze) as f:
            rows = [(float(r['t']), float(r['cmd']), float(r['vx'])) for r in csv.DictReader(f)]
        if args.out == 'step_response.csv':
            args.out = args.analyze
        segs = build_schedule(parse_steps(args.steps), args.hold, args.pre)
        results, vs, acc = analyse(rows, segs)
        write_outputs(args, rows, segs, results, vs, acc)
        return 0

    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = StepResponse(args)
    t_wall = time.time()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.t0 is None and time.time() - t_wall > args.timeout:
                node.get_logger().error(f'no odom on {args.odom_topic} / no clock after {args.timeout} s')
                return 1
    except KeyboardInterrupt:
        print('interrupted: commanding 0 and saving partial data')
    finally:
        try:
            for _ in range(20):
                node.publish(0.0)
                time.sleep(0.01)
        except Exception as exc:  # context already shut down by SIGINT
            print(f'WARNING: could not publish final 0 command: {exc}')
    if node.rows:
        results, vs, acc = analyse(node.rows, node.segs)
        write_outputs(args, node.rows, node.segs, results, vs, acc)
    aborted = node.aborted
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 2 if aborted else 0


if __name__ == '__main__':
    sys.exit(main())

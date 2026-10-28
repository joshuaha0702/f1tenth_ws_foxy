#!/usr/bin/env python3
"""실차 bag 점검: End2Race 입력/출력이 학습 분포와 맞는지, 직선 스윙이 지연성 발진인지 노이즈인지.

    python3 check_real_bag.py <bag_dir> [--scan /scan] [--drive /drive] [--odom /odom]

보고 항목
  /scan : range_min/max, 빔 수, 각도 범위(학습 FOV ±135°·360 bin 매핑과 비교), 주기,
          inf / NaN / 0 / (>= range_max 유한값) 비율  ← NaN·0 은 pool_scan 에서 0 m(접촉)로 들어가므로 위험
          전방 ±30° 에서 앞차로 보이는 군집(4 m 이내, |y| <= 0.6 m)의 빔 수 분포
  /drive: 발행 주기(100 Hz 여부), 조향 명령 통계, 조향 스펙트럼 피크(지연 유발 발진은 1~3 Hz 뾰족한 피크,
          센서 노이즈는 광대역), 조향 변화율 분포
  /odom : 속도 분포 (학습 라벨 상한과 비교)
rosbags 가 필요 (f1tenth_foxy 컨테이너 안에서 실행).
"""
import argparse
import math
import sys

import numpy as np
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


def stamp_ns(st):
    return int(st.sec) * 1_000_000_000 + int(st.nanosec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--scan', default='/scan')
    ap.add_argument('--drive', default='/drive')
    ap.add_argument('--odom', default='/odom')
    a = ap.parse_args()
    ts = get_typestore(Stores.ROS2_FOXY)
    ts.register(get_types_from_msg(ACKERMANN_DRIVE_MSG, 'ackermann_msgs/msg/AckermannDrive'))
    ts.register(get_types_from_msg(ACKERMANN_DRIVE_STAMPED_MSG, 'ackermann_msgs/msg/AckermannDriveStamped'))

    scans, drives, odoms = [], [], []
    meta = None
    with Reader(a.bag) as r:
        topics = {c.topic: c.msgtype for c in r.connections}
        print('topics:', ', '.join(f'{t} ({m.split("/")[-1]})' for t, m in sorted(topics.items())))
        for c, t, raw in r.messages():
            if c.topic == a.scan:
                m = ts.deserialize_cdr(raw, c.msgtype)
                if meta is None:
                    meta = (m.angle_min, m.angle_max, m.angle_increment, m.range_min, m.range_max, len(m.ranges))
                scans.append((t, np.asarray(m.ranges, dtype=np.float64)))
            elif c.topic == a.drive:
                m = ts.deserialize_cdr(raw, c.msgtype)
                drives.append((t, stamp_ns(m.header.stamp), float(m.drive.steering_angle), float(m.drive.speed)))
            elif c.topic == a.odom:
                m = ts.deserialize_cdr(raw, c.msgtype)
                odoms.append((t, float(m.twist.twist.linear.x)))

    if scans:
        amin, amax, inc, rmin, rmax, n = meta
        R = np.stack([s[1] for s in scans])
        tt = np.array([s[0] for s in scans]) * 1e-9
        fin = np.isfinite(R)
        print(f'\n[{a.scan}] {len(scans)} msgs, {np.median(np.diff(tt))*1000:.1f} ms period ({1/np.median(np.diff(tt)):.1f} Hz)')
        print(f'  beams {n}, angle {math.degrees(amin):+.1f}..{math.degrees(amax):+.1f} deg, step {math.degrees(inc):.4f} deg, '
              f'range_min {rmin} range_max {rmax}')
        print(f'  학습 FOV ±135°: {"포함" if amin <= -2.35 and amax >= 2.35 else "⚠ 좁음 — 빈 bin 은 max_range(30 m) 로 채워짐"}')
        print(f'  inf {np.isinf(R).mean():.3%} | NaN {np.isnan(R).mean():.3%} | ==0 {(R == 0).mean():.3%} | '
              f'유한값 >= range_max {(fin & (R >= rmax)).mean():.3%} | 유한값 > 30 m {(fin & (R > 30)).mean():.3%}')
        print(f'  ⚠ NaN/0 은 pool_scan 에서 0 m(접촉) 로 들어감 — 비율이 0.1 % 를 넘으면 max_range 로 치환해야 함' if
              (np.isnan(R).mean() + (R == 0).mean()) > 1e-3 else '  NaN/0 비율 양호')
        ang = amin + np.arange(n) * inc
        x, y = R * np.cos(ang), R * np.sin(ang)
        lead = (x > 0.1) & (x < 4.0) & (np.abs(y) <= 0.6) & fin
        cnt = lead.sum(axis=1)
        has = cnt > 0
        if has.any():
            dist = np.where(lead, x, np.inf).min(axis=1)[has]
            print(f'  앞차 후보(전방 4 m, |y|<=0.6): {has.mean():.0%} 의 스캔에서 보임, 빔 수 median {np.median(cnt[has]):.0f} '
                  f'(p10 {np.percentile(cnt[has],10):.0f}, p90 {np.percentile(cnt[has],90):.0f}), 거리 median {np.median(dist):.2f} m')
            w = np.median(cnt[has]) * math.degrees(inc)
            print(f'  → 각폭 {w:.2f}° ≈ 360-bin {w/0.75:.1f} bin, 겉보기 폭 {2*np.median(dist)*math.tan(math.radians(w)/2):.2f} m')

    if drives:
        D = np.array([(d[0], d[1], d[2], d[3]) for d in drives])
        hs = D[:, 1] * 1e-9; rec = D[:, 0] * 1e-9
        dt = np.diff(hs) if np.all(np.diff(hs) > 0) else np.diff(rec)
        print(f'\n[{a.drive}] {len(D)} msgs, header dt median {np.median(dt)*1000:.2f} ms (p99 {np.percentile(dt,99)*1000:.1f}), '
              f'{1/np.median(dt):.1f} Hz')
        st = D[:, 2]; sp = D[:, 3]
        print(f'  steer: mean {st.mean():+.3f} rad, std {st.std():.3f}, |max| {np.abs(st).max():.3f}; speed mean {sp.mean():.2f} max {sp.max():.2f} m/s')
        dsteer = np.diff(st) / np.maximum(dt, 1e-3)
        print(f'  steer rate: p50 {np.percentile(np.abs(dsteer),50):.2f} p95 {np.percentile(np.abs(dsteer),95):.2f} rad/s')
        # 스펙트럼 (균일 100 Hz 가정, 평균 제거)
        fs = 1.0 / np.median(dt)
        x = st - st.mean()
        if len(x) > 256:
            f = np.fft.rfftfreq(len(x), 1 / fs); P = np.abs(np.fft.rfft(x)) ** 2
            band = (f > 0.3) & (f < 10)
            top = np.argsort(P[band])[-3:][::-1]
            tot = P[band].sum()
            print('  steer spectrum 0.3-10 Hz 상위 피크: ' + ', '.join(f'{f[band][i]:.2f} Hz ({P[band][i]/tot:.0%})' for i in top))
            print('  → 한 피크가 30 % 이상이면 지연성 발진(저역통과/변화율 제한/주기 조정), 고르게 퍼지면 센서 노이즈 쪽')

    if odoms:
        v = np.array([o[1] for o in odoms])
        print(f'\n[{a.odom}] {len(v)} msgs, speed mean {v.mean():.2f}, p95 {np.percentile(v,95):.2f}, max {v.max():.2f} m/s (학습 라벨 상한 2.0 m/s)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""End2Race agent_node 타이밍 진단 bag 분석 (diag.timing:=true 로 녹화한 bag).

    python3 analyze_timing.py <bag_dir> [<bag_dir> ...]

rosbags 가 필요 (f1tenth_foxy 컨테이너 안에서 실행).

scan 한 장이 추론에 쓰이기까지를 구간별로 나눠 본다.
  t_header   LiDAR 드라이버가 찍은 scan 시각
  t_cb_start agent_node 의 scan 콜백 시작  → cb_wait = t_cb_start - t_header (콜백 대기)
  dur_preproc / dur_pub                    → 1440빔→360 bin 전처리 / RViz 용 전처리 scan 발행
  scan_age   추론 시작 시각 - 그 추론이 쓴 scan 의 t_header (모델이 본 scan 이 얼마나 오래됐나)
  dur_infer  inference_step 소요 (GPU 동기화 포함)
비교 기준: bag 녹화기(다른 프로세스)가 /scan 을 받은 시각 - t_header (= scan 이 제때 퍼졌는지).
"""
import sys

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

ts = get_typestore(Stores.ROS2_FOXY)


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def load(bag):
    rows = {'scan': [], 'tick': [], 'sys': []}
    fields = {}
    rec_delay, rec_hdr = [], []
    with Reader(bag) as r:
        cons = [c for c in r.connections if c.topic.startswith('/end2race/diag/') or c.topic == '/scan']
        for c, t, raw in r.messages(connections=cons):
            m = ts.deserialize_cdr(raw, c.msgtype)
            if c.topic == '/scan':
                rec_delay.append(t * 1e-9 - stamp(m.header))
                rec_hdr.append(stamp(m.header))
                continue
            key = c.topic.rsplit('/', 1)[1]
            f = m.layout.dim[1].label.split(',')
            fields[key] = f
            a = np.asarray(m.data, float)
            if len(a):
                rows[key].append(a.reshape(-1, len(f)))
    out = {}
    for k, v in rows.items():
        if v:
            a = np.concatenate(v)
            out[k] = {name: a[:, i] for i, name in enumerate(fields[k])}
    return out, np.asarray(rec_delay), np.asarray(rec_hdr)


def pct(x, qs=(50, 95, 99)):
    x = x[np.isfinite(x)]
    if not len(x):
        return 'n/a'
    return ' / '.join(f'{np.percentile(x, q) * 1000:6.1f}' for q in qs) + f' / max {x.max() * 1000:6.1f} ms'


def report(bag):
    d, rec, rec_hdr = load(bag)
    print(f'\n==== {bag}')
    if 'tick' not in d:
        print('  /end2race/diag/* 없음 — diag.timing:=true 로 녹화했는지 확인')
        return
    S, T, Y = d.get('scan'), d['tick'], d.get('sys')
    if Y is not None:
        print(f'  설정: scan 큐 {int(Y["scan_qos_depth"][0])}, 콜백 그룹 '
              f'{"수정 전(공유 Reentrant)" if Y["legacy_groups"][0] > 0.5 else "분리"}, '
              f'preprocess_scan 발행 {"켜짐" if Y["publish_pre"][0] > 0.5 else "꺼짐"}, '
              f'실행기 스레드 {int(Y["executor_threads"][0])}')

    print('\n  [1] scan 도착 (다른 프로세스 기준)')
    if len(rec):
        print(f'      bag 녹화기 수신 - 헤더     p50/p95/p99: {pct(rec)}')

    if S is not None:
        print('  [2] agent_node scan 콜백')
        print(f'      콜백 대기 (시작-헤더)     p50/p95/p99: {pct(S["cb_wait"])}')
        print(f'      전처리 시간                p50/p95/p99: {pct(S["dur_preproc"])}')
        print(f'      전처리 scan 발행 시간      p50/p95/p99: {pct(S["dur_pub"])}')
        gaps = np.diff(S['t_cb_start']); hdr = np.diff(S['t_header'])
        burst = np.mean((gaps < 0.005) & (hdr > 0.02)) * 100 if len(gaps) else float('nan')
        print(f'      몰아서 처리된 scan (콜백 간격 <5 ms 인데 헤더 간격 >20 ms): {burst:.1f}%  ← 높으면 큐 적체')
        if len(rec_hdr) and len(S['t_header']):
            # 노드 기록과 bag /scan 이 둘 다 있는 구간만 비교
            t0 = max(S['t_header'].min(), rec_hdr.min()); t1 = min(S['t_header'].max(), rec_hdr.max())
            n_bag = int(np.sum((rec_hdr >= t0) & (rec_hdr <= t1)))
            h = S['t_header']
            n_cb = len(np.unique(np.round(h[(h >= t0) & (h <= t1)], 6)))
            print(f'      노드가 처리한 scan {n_cb} / 같은 구간 bag 의 /scan {n_bag} → 처리 못 하고 버려진 scan '
                  f'{max(n_bag - n_cb, 0)} ({max(n_bag - n_cb, 0) / max(n_bag, 1) * 100:.1f}%)')
        busy = S['drive_active'] > 0
        if busy.any() and (~busy).any():
            print(f'      콜백 대기: 추론 콜백 실행 중 시작 {pct(S["cb_wait"][busy], (50, 95))}'
                  f' | 아닐 때 {pct(S["cb_wait"][~busy], (50, 95))}')

    # 시작 직후 3 초(모델·CUDA 초기화, bag 녹화 시작 등)는 따로 표시하고 이후 구간 통계에서 제외
    t_first = T['t_start'][0]
    warm = T['t_start'] < t_first + 3.0
    if warm.any():
        dtw = T['dt_prev'][warm]; dtw = dtw[np.isfinite(dtw)]
        print(f'  (시작 3 초 구간: 최대 틱 간격 {dtw.max() * 1000 if len(dtw) else float("nan"):.0f} ms — 이하 통계에서 제외)')
    T = {k: v[~warm] for k, v in T.items()}
    if S is not None:
        S = {k: v[S['t_cb_start'] >= t_first + 3.0] for k, v in S.items()}
    inf = T['inferred'] > 0.5
    print('  [3] 추론에 쓴 scan 의 나이 (추론 시작 - scan 헤더)')
    age = T['scan_age'][inf]
    print(f'      p50/p95/p99: {pct(age)}')
    for th in (0.05, 0.1):
        print(f'      > {th * 1000:.0f} ms: {np.mean(age > th) * 100:5.1f}%')
    if S is not None and inf.any():
        used = np.unique(T['scan_id'][inf]); total = np.unique(S['scan_id'])
        print(f'      추론에 한 번도 안 쓰인 scan: {len(np.setdiff1d(total, used))}/{len(total)}')

    print('  [4] drive 틱 / 추론')
    dt = T['dt_prev'][np.isfinite(T['dt_prev'])]
    print(f'      틱 간격 p50/p95/p99: {pct(dt)}   | 15 ms 넘게 벌어진 비율 {np.mean(dt > 0.015) * 100:.1f}%')
    print(f'      추론 시간 p50/p95/p99: {pct(T["dur_infer"][inf])}')
    print(f'      틱 전체 시간 p50/p95/p99: {pct(T["dur_total"])}')
    print(f'      다른 drive 콜백과 겹친 틱: {np.mean(T["drive_active"] > 0) * 100:.2f}%'
          f'   | scan 콜백 실행 중 시작한 틱: {np.mean(T["scan_active"] > 0) * 100:.1f}%')
    # 노드 전체 정지: drive 틱과 scan 콜백이 동시에 100 ms 넘게 멈춘 구간 (한쪽만이면 노드 내부 경합)
    stalls = []
    if S is not None and len(S['t_cb_start']) > 1:
        sc = np.sort(S['t_cb_start'])
        for k in np.where(T['dt_prev'] > 0.1)[0]:
            a, b = T['t_start'][k] - T['dt_prev'][k], T['t_start'][k]
            inside = np.sum((sc > a + 0.02) & (sc < b - 0.02))
            stalls.append((a - T['t_start'][0], b - a, inside == 0))
    n_all = sum(1 for x in stalls if x[2])
    print(f'      100 ms 넘는 틱 정지 {len(stalls)}회 — 그중 scan 콜백도 같이 멈춘 "노드 전체 정지" {n_all}회'
          + (': ' + ', '.join(f'{x[0]:.1f}s 부터 {x[1] * 1000:.0f} ms' for x in stalls[:5]) if stalls else ''))

    if Y is not None:
        print('  [5] 프로세스 / 시스템 (1 초 단위)')
        print(f'      CPU 부하(loadavg1) 평균 {np.nanmean(Y["loadavg1"]):.2f} | 노드 CPU 사용 평균 {np.nanmean(Y["proc_cpu_pct"]):.0f}%'
              f' 최대 {np.nanmax(Y["proc_cpu_pct"]):.0f}% | 스레드 {int(np.nanmax(Y["n_threads"]))}')
        print(f'      파이썬 GC 정지: 초당 평균 {np.mean(Y["gc_n"]):.1f}회, 합 {np.mean(Y["gc_total_ms"]):.2f} ms/s, 최대 1회 {np.max(Y["gc_max_ms"]):.1f} ms')
        if S is not None and len(Y['t']) > 3:
            sec = np.floor(S['t_cb_start'] - Y['t'][0]).astype(int)
            w = np.array([np.mean(S['cb_wait'][sec == i]) if np.any(sec == i) else np.nan for i in range(len(Y['t']))])
            ok = np.isfinite(w)
            for name in ('proc_cpu_pct', 'gc_total_ms', 'loadavg1'):
                v = Y[name][ok]
                if np.std(v) > 0 and np.std(w[ok]) > 0:
                    print(f'      초별 콜백 대기와 {name} 상관 {np.corrcoef(w[ok], v)[0, 1]:+.2f}')

    print('  [판정 힌트]')
    if len(age):
        a50 = np.percentile(age, 50)
        print(f'      추론 scan 나이 중앙값 {a50 * 1000:.0f} ms (30 Hz scan 이면 정상 ~17 ms + 처리 시간)')
    if S is not None:
        w95 = np.percentile(S['cb_wait'], 95)
        if len(rec) and w95 > np.percentile(rec, 95) + 0.02:
            print('      → 녹화기는 제때 받는데 노드 scan 콜백이 늦게 시작: 노드 내부 실행 순서/부하 문제')
        if inf.any() and np.percentile(T['dur_infer'][inf], 95) > 0.008:
            print('      → 추론 1 회가 8 ms 이상 (100 Hz 틱 예산 10 ms 에 근접): 연산 부하 의심')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for b in sys.argv[1:]:
        report(b)

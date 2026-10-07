#!/usr/bin/env python3
"""
GRU hidden state 의존도 분석
============================
정상 입력으로 T 스텝을 돌려 hidden 을 정상 상태로 만든 뒤, t=T 한 스텝에서만
아래 조건으로 교란하고 이후는 정상 입력으로 계속 돌림.

  hidden_shuffle : h_T 를 배치 내 다른 시퀀스의 h_T 로 교체 (기억만 다른 상황)
  hidden_zero    : h_T 를 0 으로 리셋 (기억 삭제)
  lidar@t        : t 스텝의 LiDAR 만 다른 시퀀스 것으로 교체
  speed@t        : t 스텝의 speed 입력만 다른 시퀀스 것으로 교체
  input@t        : t 스텝의 LiDAR + speed 모두 교체

측정:
  1) t 스텝의 조향/속력 MSE 증가율 (즉시 효과) — 전체/직선/커브/가속 구간별
  2) 교란 이후 H 스텝 동안의 오차 회복 곡선
  3) 기억 길이: t 시점 출력을 직전 K 스텝만 보고(0 hidden 에서 시작) 계산했을 때의 오차

사용법 (워크스페이스 루트에서):
  python3 feature_importance_results/hidden_dependence.py \
      --data "data/simple_variants_h2h/Simple_v07/csv/car1_*.csv" \
             "data/simple_variants_h2h/Simple_v15/csv/car1_*.csv" \
      --models src/f1tenth_end2race_ros2/models/origin_20260722.pth \
      --out feature_importance_results/hidden_car1
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import End2Race  # noqa: E402

CONDITIONS = ['hidden_shuffle', 'hidden_zero', 'lidar@t', 'speed@t', 'input@t']
CONTEXT_K = [1, 2, 5, 10, 20, 50, 100, 200]
STRAIGHT_STEER = 0.05   # |target steer| < 이 값이면 직선
CURVE_STEER = 0.15      # |target steer| >= 이 값이면 커브
ACCEL_SPEED = 1.9       # 직전 속도 명령이 이 값 미만이면 가속/감속 구간 (car1 순항 속도 2.0)


def load_windows(patterns, warmup, horizon, stride):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise FileNotFoundError(f"No CSV matched: {patterns}")
    lidar_cols = [f"lidar_{i}" for i in range(360)]
    win = warmup + horizon
    out = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) < win + 1:
            continue
        lidar = df[lidar_cols].values.astype(np.float32)
        actions = df[["steer", "desired_speed"]].values.astype(np.float32)
        lidar_v, action_v, speed_prev = lidar[1:], actions[1:], actions[:-1, 1:2]
        for s in range(0, len(lidar_v) - win + 1, stride):
            out.append((lidar_v[s:s + win], speed_prev[s:s + win], action_v[s:s + win]))
    return files, out


def batches(windows, batch_size, device, seed):
    order = np.random.default_rng(seed).permutation(len(windows))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        if len(idx) < 2:
            continue
        yield tuple(torch.from_numpy(np.stack([windows[j][k] for j in idx])).to(device) for k in range(3))


def derangement(n, gen):
    """자기 자신으로 매핑되는 원소가 없는 순열 (교체가 항상 다른 시퀀스에서 오도록)."""
    while True:
        p = torch.randperm(n, generator=gen)
        if not bool((p == torch.arange(n)).any()):
            return p


@torch.no_grad()
def analyze(model, windows, T, H, batch_size, repeats, device, seed, curve_steer=CURVE_STEER):
    gen = torch.Generator().manual_seed(seed)
    curve = {c: np.zeros((H, 2)) for c in ['baseline'] + CONDITIONS}
    n_curve = 0
    imm = {c: [] for c in ['baseline'] + CONDITIONS}   # t 스텝 샘플별 제곱오차 [N,2]
    ctx = {k: [] for k in CONTEXT_K}
    labels = []                                        # (target steer, speed_prev) at t

    for L, S, A in batches(windows, batch_size, device, seed):
        B = L.shape[0]
        _, hT = model(L[:, :T], S[:, :T])
        Lf, Sf, Af = L[:, T:], S[:, T:], A[:, T:]
        base, _ = model(Lf, Sf, hT)
        e = ((base - Af) ** 2).cpu().numpy()
        curve['baseline'] += e.sum(0)
        imm['baseline'].append(e[:, 0])
        labels.append(np.stack([Af[:, 0, 0].cpu().numpy(), Sf[:, 0, 0].cpu().numpy()], 1))

        for c in CONDITIONS:
            acc_curve = np.zeros((H, 2))
            acc_imm = np.zeros((B, 2))
            for _ in range(repeats):
                p = derangement(B, gen).to(device)
                h, Lc, Sc = hT, Lf, Sf
                if c == 'hidden_shuffle':
                    h = hT[:, p]
                elif c == 'hidden_zero':
                    h = torch.zeros_like(hT)
                if c in ('lidar@t', 'input@t'):
                    Lc = Lf.clone()
                    Lc[:, 0] = Lf[p, 0]
                if c in ('speed@t', 'input@t'):
                    Sc = Sf.clone()
                    Sc[:, 0] = Sf[p, 0]
                out, _ = model(Lc, Sc, h)
                ec = ((out - Af) ** 2).cpu().numpy()
                acc_curve += ec.sum(0)
                acc_imm += ec[:, 0]
            curve[c] += acc_curve / repeats
            imm[c].append(acc_imm / repeats)
        n_curve += B

        for k in CONTEXT_K:
            out, _ = model(L[:, T - k + 1:T + 1], S[:, T - k + 1:T + 1])
            ctx[k].append(((out[:, -1] - A[:, T]) ** 2).cpu().numpy())

    curve = {c: v / n_curve for c, v in curve.items()}
    imm = {c: np.concatenate(v) for c, v in imm.items()}
    ctx = {k: np.concatenate(v).mean(0) for k, v in ctx.items()}
    labels = np.concatenate(labels)

    steer_abs, spd = np.abs(labels[:, 0]), labels[:, 1]
    segments = {
        'all': np.ones(len(labels), bool),
        'straight': (steer_abs < STRAIGHT_STEER) & (spd >= ACCEL_SPEED),
        'curve': (steer_abs >= curve_steer) & (spd >= ACCEL_SPEED),
        'accel': spd < ACCEL_SPEED,
    }
    seg_res = {}
    for sname, m in segments.items():
        if m.sum() == 0:
            continue
        b = imm['baseline'][m].mean(0)
        seg_res[sname] = {'n': int(m.sum()), 'baseline_rmse': np.sqrt(b).tolist(), 'conditions': {}}
        for c in CONDITIONS:
            v = imm[c][m].mean(0)
            seg_res[sname]['conditions'][c] = {
                'steer_pct': float((v[0] - b[0]) / b[0] * 100), 'speed_pct': float((v[1] - b[1]) / b[1] * 100),
                'steer_rmse': float(np.sqrt(v[0])), 'speed_rmse': float(np.sqrt(v[1]))}
    return {'segments': seg_res,
            'recovery_rmse': {c: np.sqrt(v).tolist() for c, v in curve.items()},
            'context_rmse': {str(k): np.sqrt(v).tolist() for k, v in ctx.items()},
            'context_full_rmse': np.sqrt(imm['baseline'].mean(0)).tolist()}


def plot(name, r, T, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    segs = list(r['segments'].keys())
    colors = {'all': '#555555', 'straight': '#4a90d9', 'curve': '#d9534f', 'accel': '#e8a33d'}
    x = np.arange(len(CONDITIONS))
    w = 0.8 / len(segs)
    for row, key, label in ((0, 'steer', 'Steer'), (1, 'speed', 'Speed')):
        ax = axes[row, 0]
        for i, s in enumerate(segs):
            vals = [r['segments'][s]['conditions'][c][f'{key}_pct'] for c in CONDITIONS]
            ax.bar(x + (i - (len(segs) - 1) / 2) * w, vals, w, label=f"{s} (n={r['segments'][s]['n']})",
                   color=colors.get(s), alpha=0.85)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_yscale('symlog', linthresh=10)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, fontsize=9)
        ax.set_ylabel(f'{label} MSE increase at t (%)')
        ax.set_title(f'{label}: immediate effect of a one-step perturbation')
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        idx = 0 if key == 'steer' else 1
        for c in ['baseline'] + CONDITIONS:
            y = [v[idx] for v in r['recovery_rmse'][c]]
            ax.plot(range(len(y)), y, label=c, lw=2 if c == 'baseline' else 1.2,
                    color='black' if c == 'baseline' else None)
        ax.set_xlabel('steps after perturbation (10 ms each)')
        ax.set_ylabel(f'{label} RMSE')
        ax.set_title(f'{label}: recovery after perturbation at t')
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        ks = [int(k) for k in r['context_rmse']]
        y = [r['context_rmse'][str(k)][idx] for k in ks]
        ax.plot(ks, y, 'o-', color='#2a7ab0', label='last K steps only')
        ax.axhline(r['context_full_rmse'][idx], color='black', ls='--', label=f'full context ({T}+ steps)')
        ax.set_xscale('log')
        ax.set_xlabel('context length K (steps)')
        ax.set_ylabel(f'{label} RMSE at t')
        ax.set_title(f'{label}: how much history is needed')
        ax.legend(fontsize=8)
    fig.suptitle(f'{name} — hidden state dependence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name}_hidden_dependence.png'), dpi=130, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--warmup', type=int, default=200, help='교란 전 정상 입력 스텝 수 T')
    ap.add_argument('--horizon', type=int, default=100, help='교란 후 관찰 스텝 수 H')
    ap.add_argument('--stride', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--curve_steer', type=float, default=CURVE_STEER, help='|target steer| >= 이면 커브 구간')
    args = ap.parse_args()
    assert max(CONTEXT_K) <= args.warmup + 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    files, windows = load_windows(args.data, args.warmup, args.horizon, args.stride)
    print(f"{len(files)} files -> {len(windows)} windows (warmup {args.warmup}, horizon {args.horizon})")

    results = {}
    for mp in sorted({f for p in args.models for f in glob.glob(p)}):
        name = os.path.basename(mp).replace('.pth', '')
        model = End2Race(mask_prob=0.0, hidden_scale=4).to(device)
        model.load_state_dict(torch.load(mp, map_location=device, weights_only=False))
        model.eval()
        r = analyze(model, windows, args.warmup, args.horizon, args.batch_size, args.repeats, device, args.seed, args.curve_steer)
        results[name] = r
        plot(name, r, args.warmup, args.out)

        print(f"\n=== {name} ===")
        for s, sr in r['segments'].items():
            b = sr['baseline_rmse']
            print(f"[{s}] n={sr['n']}  baseline RMSE steer {b[0]:.3f} rad, speed {b[1]:.3f} m/s")
            for c, v in sr['conditions'].items():
                print(f"   {c:15s} steer {v['steer_pct']:+9.1f}% ({v['steer_rmse']:.3f})   "
                      f"speed {v['speed_pct']:+10.1f}% ({v['speed_rmse']:.3f})")
        rec = r['recovery_rmse']
        for c in CONDITIONS:
            st = np.array(rec[c])[:, 0]
            bs = np.array(rec['baseline'])[:, 0]
            excess = st - bs
            back = next((i for i in range(len(excess)) if excess[i] <= 0.1 * max(excess[0], 1e-9)), None)
            print(f"   recovery {c:15s} steer excess t0 {excess[0]:.3f} -> t+10 {excess[min(10, len(excess)-1)]:.3f} "
                  f"-> t+50 {excess[min(50, len(excess)-1)]:.3f}  (90% recovered at +{back} steps)")
        print("   context K: " + "  ".join(
            f"K={k}: {v[0]:.3f}/{v[1]:.3f}" for k, v in r['context_rmse'].items())
            + f"  | full {r['context_full_rmse'][0]:.3f}/{r['context_full_rmse'][1]:.3f}  (steer/speed RMSE)")

    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump({'data_files': files, 'num_windows': len(windows), 'warmup': args.warmup,
                   'horizon': args.horizon, 'models': results}, f, indent=2)
    print(f"\nsaved -> {args.out}/")


if __name__ == '__main__':
    main()

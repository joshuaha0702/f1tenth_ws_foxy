#!/usr/bin/env python3
"""
Permutation Importance — 조향/속력 오차 분리 버전
==================================================
feature_importance.py 의 Permutation Importance 를 다음과 같이 바꾼 버전:
  * 합산 loss(steer + 0.05*speed) 대신 조향 MSE / 속력 MSE 증가율을 따로 기록
  * 배치를 만들기 전에 시퀀스 순서를 섞어 한 배치 안에 여러 에피소드가 섞이게 함
    (원본은 파일 순서대로 배치를 묶어 같은 에피소드끼리만 섞이는 경우가 많았음)
  * 데이터/모델/출력 경로를 인자로 받음

피쳐 그룹(LiDAR 60빔 영역 6개 + Speed Input)을 배치 내 다른 시퀀스의 값으로
100스텝 전체를 통째로 바꿔 넣는 방식은 원본과 같음. 따라서 GRU hidden state 가
교란된 효과까지 포함된 수치임.

사용법 (워크스페이스 루트에서):
  python3 feature_importance_results/feature_importance_split.py \
      --data "data/simple_variants_h2h/Simple_v07/csv/car1_*.csv" \
             "data/simple_variants_h2h/Simple_v15/csv/car1_*.csv" \
      --out feature_importance_results/split_car1
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

REGIONS = {
    "Right Rear\n(0-59)":     list(range(0, 60)),
    "Right\n(60-119)":        list(range(60, 120)),
    "Front-Right\n(120-179)": list(range(120, 180)),
    "Front-Left\n(180-239)":  list(range(180, 240)),
    "Left\n(240-299)":        list(range(240, 300)),
    "Left Rear\n(300-359)":   list(range(300, 360)),
}
GROUPS = list(REGIONS.keys()) + ["Speed\nInput"]


def load_sequences(patterns, seq_len):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise FileNotFoundError(f"No CSV matched: {patterns}")
    lidar_cols = [f"lidar_{i}" for i in range(360)]
    seqs = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) < seq_len + 1:
            continue
        lidar = df[lidar_cols].values.astype(np.float32)
        actions = df[["steer", "desired_speed"]].values.astype(np.float32)
        # 학습(train.py 기본)과 동일: 입력 speed = 직전 스텝의 desired_speed
        lidar_v, action_v, speed_prev = lidar[1:], actions[1:], actions[:-1, 1:2]
        for end in range(seq_len - 1, len(lidar_v), seq_len):
            s = end - seq_len + 1
            seqs.append((lidar_v[s:end + 1], speed_prev[s:end + 1], action_v[s:end + 1]))
    return files, seqs


def make_batches(seqs, batch_size, device, seed):
    order = np.random.default_rng(seed).permutation(len(seqs))
    batches = []
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        if len(idx) < 2:
            continue
        batches.append(tuple(torch.from_numpy(np.stack([seqs[j][k] for j in idx])).to(device)
                             for k in range(3)))
    return batches


@torch.no_grad()
def eval_mse(model, batches, group=None, gen=None):
    """조향/속력 MSE (샘플 평균). group이 주어지면 해당 피쳐를 배치 내에서 섞음."""
    se_steer = se_speed = 0.0
    n = 0
    for lidar, speed, action in batches:
        if group is not None:
            perm = torch.randperm(lidar.shape[0], generator=gen).to(lidar.device)
            if group == "Speed\nInput":
                speed = speed[perm]
            else:
                idx = REGIONS[group]
                lidar = lidar.clone()
                lidar[:, :, idx] = lidar[perm][:, :, idx]
        out, _ = model(lidar, speed)
        err = (out - action) ** 2
        se_steer += err[..., 0].sum().item()
        se_speed += err[..., 1].sum().item()
        n += err[..., 0].numel()
    return se_steer / n, se_speed / n


def analyze(model, batches, repeats, seed):
    model.eval()
    b_steer, b_speed = eval_mse(model, batches)
    gen = torch.Generator().manual_seed(seed)
    res = {}
    for g in GROUPS:
        ds, dv = [], []
        for _ in range(repeats):
            s, v = eval_mse(model, batches, g, gen)
            ds.append((s - b_steer) / b_steer * 100)
            dv.append((v - b_speed) / b_speed * 100)
        res[g] = {'steer_pct': float(np.mean(ds)), 'steer_pct_std': float(np.std(ds)),
                  'speed_pct': float(np.mean(dv)), 'speed_pct_std': float(np.std(dv))}
    return {'baseline_steer_mse': b_steer, 'baseline_speed_mse': b_speed, 'groups': res}


def plot_model(name, r, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(GROUPS))
    w = 0.38
    for off, key, color, label in ((-w / 2, 'steer', '#d9534f', 'Steer MSE'),
                                   (w / 2, 'speed', '#2a7ab0', 'Speed MSE')):
        vals = [r['groups'][g][f'{key}_pct'] for g in GROUPS]
        errs = [r['groups'][g][f'{key}_pct_std'] for g in GROUPS]
        bars = ax.bar(x + off, vals, w, yerr=errs, capsize=3, color=color, alpha=0.85, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, max(v, 0), f'{v:+.0f}%', ha='center', va='bottom', fontsize=7)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, fontsize=9)
    ax.set_ylabel('MSE increase vs. baseline (%)')
    ax.set_title(f"{name}\nPermutation importance, steer / speed separated  "
                 f"(baseline steer MSE {r['baseline_steer_mse']:.2e}, speed MSE {r['baseline_speed_mse']:.2e})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{name}_perm_split.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_heatmap(all_res, out_dir, title):
    names = list(all_res.keys())
    fig, axes = plt.subplots(1, 2, figsize=(20, max(5, 0.8 * len(names) + 2)))
    for ax, key, label in ((axes[0], 'steer_pct', 'Steer MSE increase (%)'),
                           (axes[1], 'speed_pct', 'Speed MSE increase (%)')):
        data = np.array([[all_res[n]['groups'][g][key] for g in GROUPS] for n in names])
        vmax = max(1.0, float(np.abs(data).max()))
        # Speed Input 열이 수천 %라 선형 스케일이면 나머지가 안 보여서 symlog 사용
        norm = matplotlib.colors.SymLogNorm(linthresh=10, vmin=-vmax, vmax=vmax, base=10)
        im = ax.imshow(data, cmap='RdBu_r', norm=norm, aspect='auto')
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUPS, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(label)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(names)):
            for j in range(len(GROUPS)):
                ax.text(j, i, f'{data[i, j]:+.0f}', ha='center', va='center', fontsize=7)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cross_model_perm_split_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True, help='CSV glob 패턴 (여러 개 가능)')
    ap.add_argument('--models', nargs='+', default=['src/f1tenth_end2race_ros2/models/*.pth'],
                    help='.pth 경로 또는 glob 패턴')
    ap.add_argument('--out', required=True, help='결과 저장 폴더')
    ap.add_argument('--seq_len', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    model_files = sorted({f for p in args.models for f in glob.glob(p)})
    files, seqs = load_sequences(args.data, args.seq_len)
    batches = make_batches(seqs, args.batch_size, device, args.seed)
    print(f"{len(files)} CSV files -> {len(seqs)} sequences, {len(batches)} batches, device {device}")

    all_res = {}
    for mp in model_files:
        name = os.path.basename(mp).replace('.pth', '')
        model = End2Race(mask_prob=0.0, hidden_scale=4).to(device)
        try:
            model.load_state_dict(torch.load(mp, map_location=device, weights_only=False))
        except RuntimeError as e:
            print(f"[skip] {name}: {e}")
            continue
        r = analyze(model, batches, args.repeats, args.seed)
        all_res[name] = r
        plot_model(name, r, args.out)
        print(f"\n{name}  baseline steer {r['baseline_steer_mse']:.3e}  speed {r['baseline_speed_mse']:.3e}")
        for g in GROUPS:
            v = r['groups'][g]
            print(f"  {g.replace(chr(10), ' '):24s} steer {v['steer_pct']:+8.1f}%   speed {v['speed_pct']:+8.1f}%")

    plot_heatmap(all_res, args.out, f"Permutation importance (steer / speed separated) — {len(files)} files")
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump({'data_files': files, 'num_sequences': len(seqs), 'models': {
            n: {**{k: v for k, v in r.items() if k != 'groups'},
                'groups': {g.replace('\n', ' '): v for g, v in r['groups'].items()}}
            for n, r in all_res.items()}}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {args.out}/")


if __name__ == '__main__':
    main()

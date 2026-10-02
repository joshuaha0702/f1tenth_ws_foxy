#!/usr/bin/env python3
"""
Feature Importance Analysis for End2Race GRU Models
=====================================================
3가지 방법으로 피쳐 중요도를 분석합니다:
  1. 학습된 k 파라미터 크기 (LiDAR 시그모이드 변환 가중치)
  2. Gradient-based Saliency (평균 |∂output/∂input|)
  3. Permutation Importance (피쳐 그룹별 셔플 후 성능 저하 측정)

사용법:
  python3 feature_importance.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# Import model classes
from model import End2Race, End2RaceWithDelay


# ============================================================
# Configuration
# ============================================================
MODELS_DIR = "src/f1tenth_end2race_ros2/models"
DATA_DIR = "data"
OUTPUT_DIR = "feature_importance_results"
NUM_SAMPLE_FILES = 30
SEQUENCE_LENGTH = 100
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# LiDAR beam regions (270° FOV, 360 beams)
# beam 0 = -135°, beam 180 = 0° (정면), beam 359 = +134.25°
REGIONS = {
    "Right Rear\n(0-59)":    list(range(0, 60)),
    "Right\n(60-119)":       list(range(60, 120)),
    "Front-Right\n(120-179)":list(range(120, 180)),
    "Front-Left\n(180-239)": list(range(180, 240)),
    "Left\n(240-299)":       list(range(240, 300)),
    "Left Rear\n(300-359)":  list(range(300, 360)),
}


# ============================================================
# Data Loading
# ============================================================
def load_sample_data(data_dir, num_files=30, seq_len=100):
    """분석용 데이터 샘플 로드."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # 균등 샘플링
    if len(csv_files) > num_files:
        indices = np.linspace(0, len(csv_files) - 1, num_files, dtype=int)
        csv_files = [csv_files[i] for i in indices]

    lidar_cols = [f"lidar_{i}" for i in range(360)]
    sequences = []

    for f in tqdm(csv_files, desc="Loading data"):
        df = pd.read_csv(f)
        if not all(c in df.columns for c in lidar_cols + ["steer", "desired_speed"]):
            continue
        if len(df) < seq_len + 1:
            continue

        lidar = df[lidar_cols].values.astype(np.float32)
        actions = df[["steer", "desired_speed"]].values.astype(np.float32)

        lidar_v = lidar[1:]
        action_v = actions[1:]
        speed_prev = actions[:-1, 1:2]

        for end in range(seq_len - 1, len(lidar_v), seq_len):
            start = end - seq_len + 1
            sequences.append({
                'lidar':  torch.tensor(lidar_v[start:end + 1],  dtype=torch.float32),
                'speed':  torch.tensor(speed_prev[start:end + 1], dtype=torch.float32),
                'action': torch.tensor(action_v[start:end + 1], dtype=torch.float32),
            })

    print(f"Loaded {len(sequences)} sequences from {len(csv_files)} files")
    return sequences


def get_batches(sequences, batch_size, device):
    """시퀀스를 배치로 묶기."""
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        if len(batch_seqs) < 2:
            continue
        batches.append((
            torch.stack([s['lidar']  for s in batch_seqs]).to(device),
            torch.stack([s['speed']  for s in batch_seqs]).to(device),
            torch.stack([s['action'] for s in batch_seqs]).to(device),
        ))
    return batches


# ============================================================
# Method 1: k Parameter Analysis
# ============================================================
def analyze_k_parameters(model):
    """학습된 k 파라미터(LiDAR 시그모이드 변환 계수) 분석."""
    k_values = model.k.detach().cpu().numpy()
    k_magnitude = np.abs(k_values)

    region_importance = {}
    for rname, indices in REGIONS.items():
        region_importance[rname] = float(np.mean(k_magnitude[indices]))

    return {'k_values': k_values, 'k_magnitude': k_magnitude, 'region_importance': region_importance}


# ============================================================
# Method 2: Gradient-based Saliency
# ============================================================
def compute_gradient_saliency(model, batches):
    """입력에 대한 출력의 기울기로 피쳐 중요도 계산."""
    # cuDNN GRU는 eval 모드에서 backward 불가 → train 모드 사용
    # mask_prob=0.0이므로 train/eval 출력 동일
    model.train()

    lidar_grads_steer = np.zeros(360)
    lidar_grads_speed = np.zeros(360)
    speed_grad_steer = 0.0
    speed_grad_speed = 0.0
    total_samples = 0

    for lidar, speed, action in tqdm(batches, desc="  Gradient saliency"):
        lidar = lidar.detach().requires_grad_(True)
        speed = speed.detach().requires_grad_(True)

        outputs, _ = model(lidar, speed)
        bs = lidar.shape[0]

        # ---- steer output ----
        outputs[:, :, 0].sum().backward(retain_graph=True)
        lidar_grads_steer += lidar.grad.abs().mean(dim=[0, 1]).detach().cpu().numpy() * bs
        speed_grad_steer  += speed.grad.abs().mean().item() * bs

        model.zero_grad()
        lidar.grad = None
        speed.grad = None

        # ---- speed output ----
        outputs[:, :, 1].sum().backward()
        lidar_grads_speed += lidar.grad.abs().mean(dim=[0, 1]).detach().cpu().numpy() * bs
        speed_grad_speed  += speed.grad.abs().mean().item() * bs

        total_samples += bs
        model.zero_grad()

    lidar_grads_steer /= total_samples
    lidar_grads_speed /= total_samples
    speed_grad_steer  /= total_samples
    speed_grad_speed  /= total_samples

    lidar_grads_combined = lidar_grads_steer + lidar_grads_speed * 0.05

    region_steer = {}
    region_speed = {}
    region_combined = {}
    for rname, indices in REGIONS.items():
        region_steer[rname]    = float(np.mean(lidar_grads_steer[indices]))
        region_speed[rname]    = float(np.mean(lidar_grads_speed[indices]))
        region_combined[rname] = float(np.mean(lidar_grads_combined[indices]))

    model.eval()  # 복원

    return {
        'lidar_steer': lidar_grads_steer,
        'lidar_speed': lidar_grads_speed,
        'lidar_combined': lidar_grads_combined,
        'speed_input_steer': speed_grad_steer,
        'speed_input_speed': speed_grad_speed,
        'region_steer': region_steer,
        'region_speed': region_speed,
        'region_combined': region_combined,
    }


# ============================================================
# Method 3: Permutation Importance
# ============================================================
def _eval_loss(model, batches):
    """MSE 손실 계산 (steer + 0.05*speed, 학습과 동일 가중치)."""
    criterion = nn.MSELoss(reduction='sum')
    total_steer = 0.0
    total_speed = 0.0
    n = 0
    with torch.no_grad():
        for lidar, speed, action in batches:
            out, _ = model(lidar, speed)
            pred = out.view(-1, 2)
            tgt  = action.view(-1, 2)
            total_steer += criterion(pred[:, 0], tgt[:, 0]).item()
            total_speed += criterion(pred[:, 1], tgt[:, 1]).item()
            n += pred.shape[0]
    return (total_steer / n) + (total_speed / n) * 0.05, total_steer / n, total_speed / n


def compute_permutation_importance(model, batches, n_repeats=5):
    """피쳐 그룹별 순열 중요도 계산."""
    model.eval()
    base_loss, base_steer, base_speed = _eval_loss(model, batches)
    print(f"  Baseline loss: {base_loss:.6f}  (steer {base_steer:.6f}  speed {base_speed:.6f})")

    feature_groups = dict(REGIONS)
    feature_groups["Speed\nInput"] = "speed"

    results = {}
    for gname, indices in tqdm(feature_groups.items(), desc="  Permutation importance"):
        increases = []
        for _ in range(n_repeats):
            total_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for lidar, speed, action in batches:
                    lidar_p = lidar.clone()
                    speed_p = speed.clone()

                    if indices == "speed":
                        speed_p = speed_p[torch.randperm(speed_p.shape[0])]
                    else:
                        perm = torch.randperm(lidar_p.shape[0])
                        lidar_p[:, :, indices] = lidar_p[perm][:, :, indices]

                    out, _ = model(lidar_p, speed_p)
                    pred = out.view(-1, 2)
                    tgt  = action.view(-1, 2)
                    ns = pred.shape[0]
                    criterion = nn.MSELoss(reduction='sum')
                    sl = criterion(pred[:, 0], tgt[:, 0]).item() / ns
                    spl = criterion(pred[:, 1], tgt[:, 1]).item() / ns
                    total_loss += sl + spl * 0.05
                    n_batches += 1

            increases.append(total_loss / n_batches - base_loss)

        results[gname] = {
            'mean_increase': float(np.mean(increases)),
            'std_increase':  float(np.std(increases)),
            'relative_increase': float(np.mean(increases) / base_loss * 100) if base_loss > 0 else 0,
        }

    return results, base_loss


# ============================================================
# Visualization Helpers
# ============================================================
def _polar_angles():
    """LiDAR 빔 인덱스 → 라디안 각도 (-135° ~ +135°)."""
    return np.linspace(-3 * np.pi / 4, 3 * np.pi / 4, 360)


def plot_k_polar(k_mag, model_name, out_dir):
    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(121)
    ax1.bar(range(360), k_mag, width=1.0, alpha=0.7, color='steelblue')
    ax1.set_xlabel('LiDAR Beam Index')
    ax1.set_ylabel('|k| Magnitude')
    ax1.set_title(f'{model_name}\nLearned k Parameter Magnitude')
    for rname, indices in REGIONS.items():
        ax1.axvline(x=indices[0], color='gray', ls='--', alpha=0.3)

    ax2 = fig.add_subplot(122, projection='polar')
    angles = _polar_angles()
    ax2.plot(angles, k_mag, 'b-', lw=0.5, alpha=0.7)
    ax2.fill_between(angles, 0, k_mag, alpha=0.3)
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_title(f'k Magnitude (Polar)', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_name}_k_params.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_gradient_saliency(gd, model_name, out_dir):
    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(221)
    ax1.bar(range(360), gd['lidar_steer'], width=1.0, alpha=0.7, color='coral')
    ax1.set_xlabel('LiDAR Beam Index');  ax1.set_ylabel('Mean |Gradient|')
    ax1.set_title('Gradient Saliency → Steer')

    ax2 = fig.add_subplot(222)
    ax2.bar(range(360), gd['lidar_speed'], width=1.0, alpha=0.7, color='seagreen')
    ax2.set_xlabel('LiDAR Beam Index');  ax2.set_ylabel('Mean |Gradient|')
    ax2.set_title('Gradient Saliency → Speed')

    ax3 = fig.add_subplot(223, projection='polar')
    angles = _polar_angles()
    ax3.plot(angles, gd['lidar_steer'], 'r-', lw=0.5, alpha=0.7, label='Steer')
    ax3.plot(angles, gd['lidar_speed'], 'g-', lw=0.5, alpha=0.7, label='Speed')
    ax3.set_theta_zero_location('N');  ax3.set_theta_direction(-1)
    ax3.set_title('Polar View', pad=20);  ax3.legend(loc='upper right')

    ax4 = fig.add_subplot(224)
    regions = list(gd['region_steer'].keys())
    sv = [gd['region_steer'][r] for r in regions]
    spv = [gd['region_speed'][r] for r in regions]
    x = np.arange(len(regions));  w = 0.35
    ax4.bar(x - w / 2, sv, w, label='Steer', color='coral', alpha=0.8)
    ax4.bar(x + w / 2, spv, w, label='Speed', color='seagreen', alpha=0.8)
    ax4.set_xticks(x);  ax4.set_xticklabels(regions, rotation=30, ha='right', fontsize=8)
    ax4.set_ylabel('Mean |Gradient|');  ax4.set_title('Region-level');  ax4.legend()
    txt = (f"Speed Input Grad:\n"
           f"  → Steer: {gd['speed_input_steer']:.6f}\n"
           f"  → Speed: {gd['speed_input_speed']:.6f}")
    ax4.text(0.02, 0.98, txt, transform=ax4.transAxes, fontsize=8,
             va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'{model_name} – Gradient-based Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_name}_gradient_saliency.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_permutation_importance(perm, base_loss, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    groups = list(perm.keys())
    means = [perm[g]['mean_increase'] for g in groups]
    stds  = [perm[g]['std_increase']  for g in groups]
    rels  = [perm[g]['relative_increase'] for g in groups]
    colors = ['steelblue'] * (len(groups) - 1) + ['coral']

    bars = ax.bar(range(len(groups)), means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Loss Increase (↑ = more important)')
    ax.set_title(f'{model_name}\nPermutation Importance  (baseline loss={base_loss:.6f})')
    for bar, r in zip(bars, rels):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., max(h, 0),
                f'+{r:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.axhline(y=0, color='black', lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_name}_permutation_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_model_comparison(all_results, out_dir):
    """전체 모델 비교 그래프."""
    names = list(all_results.keys())
    regions = list(REGIONS.keys())

    # ---- k parameter comparison (stacked subplots) ----
    fig, axes = plt.subplots(len(names), 1, figsize=(16, 3 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, nm in zip(axes, names):
        ax.bar(range(360), all_results[nm]['k_analysis']['k_magnitude'], width=1.0, alpha=0.7, color='steelblue')
        ax.set_ylabel('|k|');  ax.set_title(nm, fontsize=10)
    axes[-1].set_xlabel('LiDAR Beam Index')
    fig.suptitle('Cross-Model: Learned k Parameters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cross_model_k_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ---- Heatmap: gradient + permutation ----
    short_names = [n.replace('.pth', '').replace('_', '\n', 1) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(6, len(names) * 0.9)))

    # gradient steer heatmap
    data_g = np.array([[all_results[n]['gradient_saliency']['region_steer'][r]
                         for r in regions] for n in names])
    im1 = ax1.imshow(data_g, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(regions)));  ax1.set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(len(names)));    ax1.set_yticklabels(short_names, fontsize=8)
    ax1.set_title('Gradient Importance → Steer');  plt.colorbar(im1, ax=ax1, shrink=0.8)
    for i in range(len(names)):
        for j in range(len(regions)):
            ax1.text(j, i, f'{data_g[i, j]:.4f}', ha='center', va='center', fontsize=6)

    # permutation heatmap
    all_groups = list(REGIONS.keys()) + ["Speed\nInput"]
    data_p = np.array([[all_results[n]['permutation_importance'].get(g, {}).get('relative_increase', 0)
                         for g in all_groups] for n in names])
    im2 = ax2.imshow(data_p, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(len(all_groups)));  ax2.set_xticklabels(all_groups, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(len(names)));       ax2.set_yticklabels(short_names, fontsize=8)
    ax2.set_title('Permutation Importance (% Loss ↑)');  plt.colorbar(im2, ax=ax2, shrink=0.8)
    for i in range(len(names)):
        for j in range(len(all_groups)):
            ax2.text(j, i, f'{data_p[i, j]:.1f}%', ha='center', va='center', fontsize=6)

    fig.suptitle('Cross-Model Feature Importance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cross_model_importance_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pth")))
    if not model_files:
        print(f"No model files found in {MODELS_DIR}");  return

    print(f"Models ({len(model_files)}): {[os.path.basename(f) for f in model_files]}")
    print(f"Device: {DEVICE}\n")

    # ---- Load data once ----
    print("=" * 60)
    print("Loading sample data...")
    print("=" * 60)
    sequences = load_sample_data(DATA_DIR, num_files=NUM_SAMPLE_FILES, seq_len=SEQUENCE_LENGTH)
    batches = get_batches(sequences, BATCH_SIZE, DEVICE)
    print(f"Batches: {len(batches)}\n")

    all_results: Dict[str, dict] = {}

    for model_path in model_files:
        mname = os.path.basename(model_path).replace('.pth', '')
        print(f"\n{'=' * 60}")
        print(f"  Analyzing: {mname}")
        print(f"{'=' * 60}")

        model = End2Race(mask_prob=0.0, hidden_scale=4).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
        model.eval()

        res: dict = {}

        # 1) k parameters
        print("\n[1/3] k parameter analysis...")
        res['k_analysis'] = analyze_k_parameters(model)
        plot_k_polar(res['k_analysis']['k_magnitude'], mname, OUTPUT_DIR)
        top5 = np.argsort(res['k_analysis']['k_magnitude'])[-5:][::-1]
        for idx in top5:
            print(f"  beam {idx}: |k| = {res['k_analysis']['k_magnitude'][idx]:.4f}")

        # 2) Gradient saliency
        print("\n[2/3] Gradient saliency...")
        res['gradient_saliency'] = compute_gradient_saliency(model, batches)
        plot_gradient_saliency(res['gradient_saliency'], mname, OUTPUT_DIR)
        for r, v in res['gradient_saliency']['region_steer'].items():
            print(f"  {r.replace(chr(10), ' ')}: {v:.6f}")
        print(f"  Speed→Steer: {res['gradient_saliency']['speed_input_steer']:.6f}")
        print(f"  Speed→Speed: {res['gradient_saliency']['speed_input_speed']:.6f}")

        # 3) Permutation importance
        print("\n[3/3] Permutation importance...")
        perm, bl = compute_permutation_importance(model, batches, n_repeats=5)
        res['permutation_importance'] = perm
        res['baseline_loss'] = bl
        plot_permutation_importance(perm, bl, mname, OUTPUT_DIR)
        for g, v in sorted(perm.items(), key=lambda x: x[1]['relative_increase'], reverse=True):
            print(f"  {g.replace(chr(10), ' ')}: +{v['relative_increase']:.2f}%")

        all_results[mname] = res

    # ---- Cross-model comparison ----
    print(f"\n{'=' * 60}")
    print("Cross-model comparison plots...")
    print(f"{'=' * 60}")
    plot_cross_model_comparison(all_results, OUTPUT_DIR)

    # ---- Save JSON ----
    json_out = {}
    for mname, res in all_results.items():
        k_top10 = np.argsort(res['k_analysis']['k_magnitude'])[-10:][::-1]
        json_out[mname] = {
            'k_analysis': {
                'region_importance': {k.replace('\n', ' '): v
                                      for k, v in res['k_analysis']['region_importance'].items()},
                'top10_beams': {str(i): float(res['k_analysis']['k_magnitude'][i]) for i in k_top10},
            },
            'gradient_saliency': {
                'region_steer': {k.replace('\n', ' '): v
                                 for k, v in res['gradient_saliency']['region_steer'].items()},
                'region_speed': {k.replace('\n', ' '): v
                                 for k, v in res['gradient_saliency']['region_speed'].items()},
                'speed_input_steer': res['gradient_saliency']['speed_input_steer'],
                'speed_input_speed': res['gradient_saliency']['speed_input_speed'],
            },
            'permutation_importance': {k.replace('\n', ' '): v
                                       for k, v in res['permutation_importance'].items()},
            'baseline_loss': res['baseline_loss'],
        }

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 결과 저장 완료 → {OUTPUT_DIR}/")
    print(f"  모델별: *_k_params.png, *_gradient_saliency.png, *_permutation_importance.png")
    print(f"  비교:   cross_model_k_comparison.png, cross_model_importance_heatmap.png")
    print(f"  수치:   results.json")


if __name__ == "__main__":
    main()

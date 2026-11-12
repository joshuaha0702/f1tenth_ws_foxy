#!/usr/bin/env python3
"""
Deep Comparative Analysis: 100Hz vs 30Hz Uniform Random k Models
1. Permutation Feature Importance & Gradient Saliency
2. Learnt k-parameter profile comparison
3. Actual OOD Driving Smoothness & Steering Jitter Analysis from Rosbag2
"""

import os
import sys
import glob
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ROS 2 CDR Deserialization
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, '/root/f1tenth_ws')
import model
from train_k_random import SequenceDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AckermannMsg = get_message('ackermann_msgs/msg/AckermannDriveStamped')

REGIONS = {
    "Right Rear (0-59)":     list(range(0, 60)),
    "Right (60-119)":        list(range(60, 120)),
    "Front-Right (120-179)": list(range(120, 180)),
    "Front-Left (180-239)":  list(range(180, 240)),
    "Left (240-299)":        list(range(240, 300)),
    "Left Rear (300-359)":   list(range(300, 360)),
}
GROUPS = list(REGIONS.keys()) + ["Speed Input"]

MODEL_100HZ_PATH = '/root/f1tenth_ws/src/f1tenth_end2race_ros2/models/best_selffeed_real_car166_8000_100hz_k_random.pth'
MODEL_30HZ_PATH = '/root/f1tenth_ws/src/f1tenth_end2race_ros2/models/ego_norm_ds30_real_car166_8000_k_random_ep100_ep070.pth'

BAGS_100HZ_DIR = '/root/f1tenth_ws/data/ood_eval_car166_100hz_k_random/episodes'
BAGS_30HZ_DIR = '/root/f1tenth_ws/data/ood_eval_car166_k_random_ep070/episodes'
TRAIN_CAR1_DIR = '/root/f1tenth_ws/data/train_mix_real_car166_8000/car1'
OUT_DIR = '/root/f1tenth_ws/data/ood_eval'


def load_model(path: str, hidden_scale: int = 1):
    state = torch.load(path, map_location=DEVICE)
    m = model.End2RaceNorm(hidden_scale=hidden_scale).to(DEVICE)
    m.load_state_dict(state)
    m.eval()
    return m


def evaluate_feature_importance(m, val_loader, num_batches=30):
    """Compute baseline loss and permutation importance."""
    base_steer_losses, base_speed_losses = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            l_in, s_in, target = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            pred, _ = m(l_in, s_in)
            steer_loss = nn.functional.mse_loss(pred[:, :, 0], target[:, :, 0]).item()
            speed_loss = nn.functional.mse_loss(pred[:, :, 1], target[:, :, 1]).item()
            base_steer_losses.append(steer_loss)
            base_speed_losses.append(speed_loss)

    b_steer = np.mean(base_steer_losses)
    b_speed = np.mean(base_speed_losses)
    b_total = 0.5 * b_steer + 0.5 * b_speed

    perm_results = {}
    for grp in GROUPS:
        p_steer_losses, p_speed_losses = [], []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break
                l_in, s_in, target = batch[0], batch[1], batch[2]
                l_perm = l_in.clone().to(DEVICE)
                s_perm = s_in.clone().to(DEVICE)
                target = target.to(DEVICE)

                if grp == "Speed Input":
                    perm_idx = torch.randperm(s_perm.size(0))
                    s_perm = s_perm[perm_idx]
                else:
                    beams = REGIONS[grp]
                    perm_idx = torch.randperm(l_perm.size(0))
                    l_perm[:, :, beams] = l_perm[perm_idx][:, :, beams]

                pred, _ = m(l_perm, s_perm)
                steer_loss = nn.functional.mse_loss(pred[:, :, 0], target[:, :, 0]).item()
                speed_loss = nn.functional.mse_loss(pred[:, :, 1], target[:, :, 1]).item()
                p_steer_losses.append(steer_loss)
                p_speed_losses.append(speed_loss)

        ps_m = np.mean(p_steer_losses)
        psp_m = np.mean(p_speed_losses)
        pt_m = 0.5 * ps_m + 0.5 * psp_m

        perm_results[grp] = {
            "steer_inc_pct": (ps_m - b_steer) / max(b_steer, 1e-7) * 100.0,
            "speed_inc_pct": (psp_m - b_speed) / max(b_speed, 1e-7) * 100.0,
            "total_inc_pct": (pt_m - b_total) / max(b_total, 1e-7) * 100.0,
        }
    return perm_results, b_total, b_steer, b_speed


def compute_gradient_saliency(m, val_loader, num_batches=15):
    """Compute per-beam saliency across all 360 beams."""
    orig_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    saliency_steer = np.zeros(360, dtype=np.float64)
    saliency_speed = np.zeros(360, dtype=np.float64)
    count = 0

    try:
        m.train()
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break
            l_in = batch[0].clone().detach().to(DEVICE).requires_grad_(True)
            s_in = batch[1].clone().detach().to(DEVICE).requires_grad_(True)

            out = m(l_in, s_in)[0]
            out_steer = out[:, :, 0].sum()
            out_speed = out[:, :, 1].sum()

            grad_steer_l = torch.autograd.grad(out_steer, l_in, retain_graph=True)[0]
            grad_speed_l = torch.autograd.grad(out_speed, l_in, retain_graph=False)[0]

            saliency_steer += grad_steer_l.abs().mean(dim=(0, 1)).detach().cpu().numpy()
            saliency_speed += grad_speed_l.abs().mean(dim=(0, 1)).detach().cpu().numpy()
            count += 1
    finally:
        m.eval()
        torch.backends.cudnn.enabled = orig_cudnn

    if count > 0:
        saliency_steer /= count
        saliency_speed /= count
    return saliency_steer, saliency_speed


def extract_drive_from_bag(bag_path: str) -> pd.DataFrame:
    """Extract /car1/drive time series from SQLite db3 file and resample cleanly."""
    conn = sqlite3.connect(bag_path)
    c = conn.cursor()
    # Find topic id for /car1/drive
    c.execute("SELECT id FROM topics WHERE name = '/car1/drive'")
    row = c.fetchone()
    if not row:
        conn.close()
        return pd.DataFrame()
    topic_id = row[0]

    c.execute("SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC", (topic_id,))
    rows = c.fetchall()
    conn.close()

    records = []
    last_ts = None
    for ts, data in rows:
        t_sec = ts * 1e-9
        # Ignore duplicate/too close timestamps (< 2ms)
        if last_ts is not None and (t_sec - last_ts) < 0.003:
            continue
        last_ts = t_sec
        msg = deserialize_message(data, AckermannMsg)
        records.append({
            "timestamp": t_sec,
            "steer": float(msg.drive.steering_angle),
            "speed": float(msg.drive.speed)
        })
    df = pd.DataFrame(records)
    if len(df) > 2:
        df["dt"] = df["timestamp"].diff().fillna(0.0)
        # Numerical gradient with actual dt (clamped to min 0.005s)
        dt_safe = np.clip(df["dt"].values, 0.005, 0.1)
        steer_diff = df["steer"].diff().fillna(0.0).values
        df["steer_vel"] = steer_diff / dt_safe
        vel_diff = df["steer_vel"].diff().fillna(0.0).values
        df["steer_jerk"] = vel_diff / dt_safe
    return df


def analyze_bags_smoothness(bag_dir: str, max_episodes=40) -> Dict:
    db3_files = sorted(glob.glob(os.path.join(bag_dir, "*/*/*/*.db3")))
    if not db3_files:
        db3_files = sorted(glob.glob(os.path.join(bag_dir, "*/*/*.db3")))

    all_steer_vels = []
    all_steer_jerks = []
    all_sample_rates = []
    all_steer_std = []

    for bag_p in db3_files[:max_episodes]:
        df = extract_drive_from_bag(bag_p)
        if len(df) < 50:
            continue
        dt_nonzero = df["dt"][df["dt"] > 0]
        if len(dt_nonzero) > 0:
            rate = 1.0 / dt_nonzero.median()
            all_sample_rates.append(rate)

        all_steer_vels.extend(df["steer_vel"].abs().values)
        all_steer_jerks.extend(df["steer_jerk"].abs().values)
        all_steer_std.append(df["steer"].std())

    return {
        "mean_sample_rate": np.mean(all_sample_rates) if all_sample_rates else 0.0,
        "mean_steer_std": np.mean(all_steer_std) if all_steer_std else 0.0,
        "mean_steer_vel": np.mean(all_steer_vels) if all_steer_vels else 0.0,
        "p95_steer_vel": np.percentile(all_steer_vels, 95) if all_steer_vels else 0.0,
        "mean_steer_jerk": np.mean(all_steer_jerks) if all_steer_jerks else 0.0,
        "p95_steer_jerk": np.percentile(all_steer_jerks, 95) if all_steer_jerks else 0.0,
        "count_episodes": len(all_sample_rates)
    }


def main():
    print("[1/4] Loading models and validation dataset...")
    m100 = load_model(MODEL_100HZ_PATH, hidden_scale=1)
    m30 = load_model(MODEL_30HZ_PATH, hidden_scale=1)

    val_files = sorted(glob.glob(f"{TRAIN_CAR1_DIR}/*.csv"))[:100]  # sample 100 episodes
    print(f"Loaded {len(val_files)} evaluation episodes.")

    # Datasets for 100Hz and 30Hz
    ds_100 = SequenceDataset(data_path="", csv_files=val_files, sequence_length=200, stride=100, downsample_factor=1,
                             downsample_mode="fixed", lidar_stride=1, speed_input="command", add_lidar_noise=False)
    ds_30 = SequenceDataset(data_path="", csv_files=val_files, sequence_length=60, stride=30, downsample_factor=1,
                            downsample_mode="lidar_sync", lidar_stride=3, speed_input="command", add_lidar_noise=False)

    print(f"Dataset samples - 100Hz: {len(ds_100)}, 30Hz: {len(ds_30)}")

    loader_100 = torch.utils.data.DataLoader(ds_100, batch_size=32, shuffle=False)
    loader_30 = torch.utils.data.DataLoader(ds_30, batch_size=32, shuffle=False)

    print("[2/4] Computing Permutation Feature Importance & Gradient Saliency...")
    p100_res, b100_tot, b100_st, b100_sp = evaluate_feature_importance(m100, loader_100, num_batches=20)
    p30_res, b30_tot, b30_st, b30_sp = evaluate_feature_importance(m30, loader_30, num_batches=20)

    sal100_st, sal100_sp = compute_gradient_saliency(m100, loader_100, num_batches=15)
    sal30_st, sal30_sp = compute_gradient_saliency(m30, loader_30, num_batches=15)

    k_vec_100 = m100.k.detach().cpu().numpy()
    k_vec_30 = m30.k.detach().cpu().numpy()

    print("[3/4] Analyzing Steering Smoothness & Jitter from OOD Rosbags...")
    smooth_100 = analyze_bags_smoothness(BAGS_100HZ_DIR, max_episodes=50)
    smooth_30 = analyze_bags_smoothness(BAGS_30HZ_DIR, max_episodes=50)

    print("\n--- RESULTS SUMMARY ---")
    print(f"100Hz Smoothness: Rate={smooth_100['mean_sample_rate']:.1f}Hz, Mean |dSteer/dt|={smooth_100['mean_steer_vel']:.3f} rad/s, p95={smooth_100['p95_steer_vel']:.3f} rad/s")
    print(f"30Hz Smoothness:  Rate={smooth_30['mean_sample_rate']:.1f}Hz, Mean |dSteer/dt|={smooth_30['mean_steer_vel']:.3f} rad/s, p95={smooth_30['p95_steer_vel']:.3f} rad/s")
    print(f"100Hz Jerk: Mean={smooth_100['mean_steer_jerk']:.2f} rad/s^2, p95={smooth_100['p95_steer_jerk']:.2f} rad/s^2")
    print(f"30Hz Jerk:  Mean={smooth_30['mean_steer_jerk']:.2f} rad/s^2, p95={smooth_30['p95_steer_jerk']:.2f} rad/s^2")

    # Sample trajectory comparison for plotting
    bag100_sample = sorted(glob.glob(os.path.join(BAGS_100HZ_DIR, "Simple/clean/*/*.db3")))[0]
    bag30_sample = sorted(glob.glob(os.path.join(BAGS_30HZ_DIR, "Simple/clean/*/*.db3")))[0]
    df_sample_100 = extract_drive_from_bag(bag100_sample)
    df_sample_30 = extract_drive_from_bag(bag30_sample)

    print("[4/4] Plotting comprehensive 4-panel analysis figure...")
    fig = plt.figure(figsize=(18, 12))

    # Panel 1: Permutation Importance Comparison (Steering % Degradation)
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(GROUPS))
    width = 0.35
    imp100_st = [p100_res[g]["steer_inc_pct"] for g in GROUPS]
    imp30_st = [p30_res[g]["steer_inc_pct"] for g in GROUPS]
    ax1.bar(x - width/2, imp100_st, width, label="100Hz Model", color="#2ca02c", alpha=0.85)
    ax1.bar(x + width/2, imp30_st, width, label="30Hz Model", color="#1f77b4", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(GROUPS, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("Steering MSE Increase (%)", fontsize=11)
    ax1.set_title("(A) Permutation Feature Importance (Steering Sensitivity)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Panel 2: Saliency & k-parameter across 360 LiDAR Beams
    ax2 = fig.add_subplot(2, 2, 2)
    angles = np.linspace(-180, 180, 360)
    ax2.plot(angles, sal100_st, label="100Hz Steer Saliency", color="#2ca02c", linewidth=2.0)
    ax2.plot(angles, sal30_st, label="30Hz Steer Saliency", color="#1f77b4", linewidth=1.8, linestyle="--")
    ax2.set_xlabel("LiDAR Angle (deg, 0° = Front)", fontsize=11)
    ax2.set_ylabel("Gradient Saliency |∂Steer/∂Beam|", fontsize=11)
    ax2.set_title("(B) Gradient Saliency Profile across 360° LiDAR Beams", fontsize=12, fontweight="bold")
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax2.axvline(-90, color="gray", linestyle=":", alpha=0.5)
    ax2.axvline(90, color="gray", linestyle=":", alpha=0.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    # Secondary axis for k vector
    ax2_k = ax2.twinx()
    ax2_k.plot(angles, k_vec_100, label="100Hz Learnt k", color="#ff7f0e", linewidth=1.5, alpha=0.8)
    ax2_k.set_ylabel("Learnt k Value", color="#ff7f0e", fontsize=11)
    ax2_k.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2_k.legend(loc="upper right", fontsize=9)

    # Panel 3: Steering Angle Profile Over Time (Actual OOD Simple Track)
    ax3 = fig.add_subplot(2, 2, 3)
    t100 = df_sample_100["timestamp"] - df_sample_100["timestamp"].iloc[0]
    t30 = df_sample_30["timestamp"] - df_sample_30["timestamp"].iloc[0]
    # Slice first 15 seconds
    mask100 = t100 <= 15.0
    mask30 = t30 <= 15.0
    ax3.plot(t100[mask100], np.degrees(df_sample_100["steer"][mask100]), label="100Hz Actual Steer (100Hz cmd)", color="#2ca02c", linewidth=1.5)
    ax3.plot(t30[mask30], np.degrees(df_sample_30["steer"][mask30]), label="30Hz Actual Steer (30Hz cmd)", color="#1f77b4", linewidth=1.5, linestyle="--")
    ax3.set_xlabel("Time (s)", fontsize=11)
    ax3.set_ylabel("Steering Angle (deg)", fontsize=11)
    ax3.set_title("(C) Real-Time Steering Control Trajectory (Simple Track OOD)", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=10)

    # Panel 4: Steering Velocity Distribution & Jitter Metrics
    ax4 = fig.add_subplot(2, 2, 4)
    vel_metrics = [
        ("Mean |dSteer/dt| (rad/s)", smooth_100['mean_steer_vel'], smooth_30['mean_steer_vel']),
        ("P95 |dSteer/dt| (rad/s)", smooth_100['p95_steer_vel'], smooth_30['p95_steer_vel']),
        ("Mean Jerk (rad/s²)", smooth_100['mean_steer_jerk'] / 10.0, smooth_30['mean_steer_jerk'] / 10.0), # scaled for plot
    ]
    labels = [m[0] for m in vel_metrics]
    val100 = [m[1] for m in vel_metrics]
    val30 = [m[2] for m in vel_metrics]

    x_m = np.arange(len(labels))
    ax4.bar(x_m - width/2, val100, width, label="100Hz OOD Episodes", color="#2ca02c", alpha=0.85)
    ax4.bar(x_m + width/2, val30, width, label="30Hz OOD Episodes", color="#1f77b4", alpha=0.85)
    ax4.set_xticks(x_m)
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.set_ylabel("Metric Value", fontsize=11)
    ax4.set_title("(D) Steering Smoothness & Jitter Metrics across 120 OOD Episodes", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "car166_100hz_vs_30hz_feature_and_jitter_analysis.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot to {plot_path}")

    # Save metrics JSON / CSV
    summary_dict = {
        "100hz_permutation_importance": p100_res,
        "30hz_permutation_importance": p30_res,
        "100hz_smoothness": smooth_100,
        "30hz_smoothness": smooth_30,
    }
    import json
    with open(os.path.join(OUT_DIR, "car166_feature_jitter_metrics.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)
    print("Saved metrics to car166_feature_jitter_metrics.json")

if __name__ == "__main__":
    main()

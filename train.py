import os
import math
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Tuple
from model import End2Race, End2RaceWithDelay, End2RaceNorm, speed_dims_from_state_dict


MAX_STEER = 0.52
MAX_SPEED = 2.0
SPEED_LOSS_WEIGHT = 0.05
CMD_DROP_MAX_SPEED = 2.0  # --speed_input command, --cmd_drop: replacement speed ~ U[0, this]


def get_loss_output_paths(model_path: str) -> Tuple[str, str]:
    """Return CSV and plot paths derived from the model output path."""
    output_stem = os.path.splitext(model_path)[0]
    return f"{output_stem}_loss.csv", f"{output_stem}_loss.png"


def save_loss_history(loss_history, csv_path: str, plot_path: str):
    """Persist completed epochs and render their loss curves."""
    if not loss_history:
        return

    output_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(loss_history).to_csv(csv_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = pd.DataFrame(loss_history)
    epochs = history["epoch"].to_numpy()
    panels = (("loss", "total loss (steer + %.2f x speed MSE)" % SPEED_LOSS_WEIGHT),
              ("steer_rmse_deg", "steering RMSE (deg)"),
              ("speed_rmse_mps", "speed RMSE (m/s)"))
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, (key, title) in zip(axes, panels):
        if key in history:
            axis.plot(epochs, history[key].to_numpy(), label="train (windows, noise/mask on)", alpha=0.8)
        if "val_" + key in history:
            axis.plot(epochs, history["val_" + key].to_numpy(), label="validation (full episodes)", linewidth=2)
        if "val_selffeed_" + key in history:
            sf = history[["epoch", "val_selffeed_" + key]].dropna()
            axis.plot(sf["epoch"].to_numpy(), sf["val_selffeed_" + key].to_numpy(), "o-",
                      label="validation, self-fed command", linewidth=2)
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(plot_path, dpi=120)
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train End2Race speed-conditioned model (Standard / Normalized / Downsampled)')

    parser.add_argument("--data_path", type=str, default="data/simple_variants_h2h/train")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model save path. Auto-generated from mode, downsample factor, and date if not specified.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Optional checkpoint path to initialize model weights before training.")
    parser.add_argument("--mode", type=str, choices=["origin", "lidar_delay"], default="origin",
                        help="'origin': train without lidar_delay / 'lidar_delay': train with lidar_delay input")

    # 정규화 및 다운샘플링 파라미터
    parser.add_argument("--use_norm", action="store_true", default=False,
                        help="Train End2RaceNorm model (Tanh output [-1, 1], LayerNorm, normalized speed input)")
    parser.add_argument("--downsample_factor", type=int, default=1,
                        help="Downsampling factor for time series data (e.g. 10 produces 10Hz data from raw 100Hz). Default: 1")
    parser.add_argument("--downsample_mode", choices=["lidar_sync", "fixed"], default="lidar_sync",
                        help="'lidar_sync': Downsample based on actual LiDAR scan arrival times (fresh scans). 'fixed': Blindly slice rows.")
    parser.add_argument("--lidar_stride", type=int, default=3,
                        help="For lidar_sync mode: sample every Nth unique LiDAR scan (default: 3, for ~30Hz -> ~10Hz)")
    parser.add_argument("--no_multi_phase", action="store_true",
                        help="For lidar_sync mode: only train on phase offset 0 instead of all offsets")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save interval checkpoint every N epochs (default: 10)")

    parser.add_argument("--sequence_length", type=int, default=400,
                        help="Number of timesteps per training sequence (default: 400 for 100Hz, recommend 40 for 10Hz)")
    parser.add_argument("--stride", type=int, default=50,
                        help="Step size for sliding window over each episode (default: 50 for 100Hz, recommend 5 for 10Hz)")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None,
                        help="Train on windows of several lengths instead of --sequence_length/--stride.")
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--disable_lidar_noise", action="store_true",
                        help="Disable distance-dependent Gaussian noise augmentation")
    parser.add_argument("--start_from_rest", action="store_true",
                        help="Also train on each episode's first row with previous speed 0")
    parser.add_argument("--speed_input", choices=["command", "odom", "hybrid", "none"], default="command",
                        help="none = 속도 입력 없음, LiDAR 전용 (End2Race speed_dims=0; --use_norm 과 함께 쓸 수 없음)")
    parser.add_argument("--mirror_prob", type=float, default=0.0,
                        help="미러 증강: 윈도우마다 이 확률로 360 LiDAR bin 좌우 반전 + 조향 라벨 부호 반전 "
                             "(대칭 차량에서 역방향 주행과 동치, 0.5 = CW/CCW 균형). 속도 입력/라벨은 그대로.")
    parser.add_argument("--cmd_drop", type=float, default=None)
    parser.add_argument("--cmd_noise", type=float, default=None)
    parser.add_argument("--cmd_lag_prob", type=float, default=None)
    parser.add_argument("--cmd_lag_s", type=float, default=0.5)
    parser.add_argument("--selffeed_every", type=int, default=10)
    parser.add_argument("--selffeed_episodes", type=int, default=40)
    parser.add_argument("--selffeed_max_speed", type=float, default=3.0)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader worker processes (default: 4)")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--lr_schedule", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--warmup_epochs", type=float, default=5.0)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--resume_warmup_epochs", type=float, default=1.0)
    parser.add_argument("--balance_leader_near", type=float, default=None)
    parser.add_argument("--near_gap", type=float, default=3.0)
    parser.add_argument("--skip_first_s", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--val_seed", type=int, default=42)

    args = parser.parse_args()
    if args.downsample_factor == 1:
        args.downsample_mode = "fixed"
        args.lidar_stride = 1
    return args


def extract_lidar_sync_df(df, lidar_cols, stride: int = 3, offset: int = 0):
    """Samples df at rows where LiDAR scan was newly updated (fresh arrival)."""
    if len(df) < 5 or stride < 1:
        return df, np.arange(len(df))
    vals = df[lidar_cols].values
    diff = np.max(np.abs(np.diff(vals, axis=0)), axis=1)
    new_scans = np.where(diff > 1e-4)[0] + 1
    new_scans = np.concatenate(([0], new_scans))

    if len(new_scans) >= stride:
        sub_idx = new_scans[offset::stride]
        return df.iloc[sub_idx].reset_index(drop=True), sub_idx
    else:
        step = stride * 3
        sub_idx = np.arange(offset, len(df), step)
        return df.iloc[sub_idx].reset_index(drop=True), sub_idx


class SequenceDataset(Dataset):
    def __init__(self, data_path: str, sequence_length: int = 50,
                 use_lidar_delay: bool = False, stride: int = 1, device: str = "cuda",
                 add_lidar_noise: bool = True, start_from_rest: bool = False,
                 seq_lengths: Optional[list] = None, csv_files: Optional[list] = None,
                 balance_leader_near: Optional[float] = None, near_gap: float = 3.0,
                 speed_input: str = "command", skip_first_s: float = 0.0,
                 cmd_drop: Optional[float] = None, cmd_noise: Optional[float] = None,
                 cmd_lag_prob: Optional[float] = None, cmd_lag_s: float = 0.5,
                 downsample_factor: int = 1, downsample_mode: str = "lidar_sync",
                 lidar_stride: int = 3, multi_phase: bool = True, mirror_prob: float = 0.0):
        self.sequence_length = sequence_length
        self.mirror_prob = mirror_prob
        self.use_lidar_delay = use_lidar_delay
        self.stride = stride
        self.device = device
        self.add_lidar_noise = add_lidar_noise
        self.start_from_rest = start_from_rest
        self.seq_lengths = sorted(seq_lengths) if seq_lengths else None
        self.balance_leader_near = balance_leader_near
        self.near_gap = near_gap
        self.speed_input = speed_input
        self.skip_first_s = skip_first_s
        self.downsample_factor = downsample_factor
        self.downsample_mode = downsample_mode
        self.lidar_stride = lidar_stride
        self.multi_phase = multi_phase

        if speed_input == "hybrid":
            self.cmd_drop = 0.3 if cmd_drop is None else cmd_drop
            self.cmd_noise = 0.15 if cmd_noise is None else cmd_noise
            self.cmd_lag_prob = 0.2 if cmd_lag_prob is None else cmd_lag_prob
            self.augment_command = (self.cmd_drop > 0 or self.cmd_noise > 0 or self.cmd_lag_prob > 0)
        elif speed_input == "command":
            self.cmd_drop = 0.0 if cmd_drop is None else cmd_drop
            self.cmd_noise = 0.0 if cmd_noise is None else cmd_noise
            self.cmd_lag_prob = 0.0 if cmd_lag_prob is None else cmd_lag_prob
            self.augment_command = (self.cmd_drop > 0 or self.cmd_noise > 0 or self.cmd_lag_prob > 0)
        else:
            self.cmd_drop = self.cmd_noise = self.cmd_lag_prob = 0.0
            self.augment_command = False
        self.cmd_lag_s = cmd_lag_s

        self.episodes = []
        self.indices = []
        self._load_data(data_path, csv_files)
        self.class_weights = None
        if self.balance_leader_near is not None:
            self._set_class_weights()

        if self.seq_lengths:
            self.current_windows = []
            self.resample_windows()

    def _load_data(self, data_path: str, csv_files: Optional[list]):
        if csv_files is None:
            csv_files = sorted(glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True))

        lidar_cols = [f"lidar_{i}" for i in range(360)]

        for csv_file in csv_files:
            raw_df = pd.read_csv(csv_file)
            check_speed_column(raw_df, csv_file, self.speed_input)

            offsets = [0]
            if self.downsample_mode == "lidar_sync" and self.lidar_stride > 1 and self.multi_phase:
                offsets = list(range(self.lidar_stride))
            elif self.downsample_factor > 1 and self.multi_phase and self.downsample_mode == "fixed":
                offsets = list(range(self.downsample_factor))

            for offset in offsets:
                if self.downsample_mode == "lidar_sync" and self.lidar_stride > 1:
                    df, _ = extract_lidar_sync_df(raw_df, lidar_cols, stride=self.lidar_stride, offset=offset)
                elif self.downsample_factor > 1:
                    df = raw_df.iloc[offset::self.downsample_factor].reset_index(drop=True)
                else:
                    df = raw_df

                min_len = (max(self.seq_lengths) if self.seq_lengths else self.sequence_length)
                if len(df) < min_len:
                    continue

                lidar_data = df[lidar_cols].values.astype(np.float32)
                action_data = df[["steer", "desired_speed"]].values.astype(np.float32)

                prev_command = np.roll(df["desired_speed"].values, 1).astype(np.float32)
                if not self.start_from_rest:
                    prev_command[0] = df["desired_speed"].values[0]
                else:
                    prev_command[0] = 0.0

                if self.speed_input == "hybrid":
                    odom = df["odom_speed"].values.astype(np.float32)
                    speed_data = np.stack((odom, prev_command), axis=1)
                elif self.speed_input == "odom":
                    speed_data = df[["odom_speed"]].values.astype(np.float32)
                else:
                    speed_data = prev_command.reshape(-1, 1)

                delay_data = (df["lidar_delay"].values.astype(np.float32).reshape(-1, 1)
                              if self.use_lidar_delay else None)

                loss_mask = skip_first_mask(df, self.skip_first_s)
                classes = classify_rows(df, self.near_gap) if self.balance_leader_near is not None else None

                ep_idx = len(self.episodes)
                self.episodes.append((lidar_data, speed_data, action_data, delay_data, classes, loss_mask))

                if not self.seq_lengths:
                    for start in range(0, len(df) - self.sequence_length + 1, self.stride):
                        self.indices.append((ep_idx, start, self.sequence_length))

    def _set_class_weights(self):
        counts = np.bincount(np.concatenate([ep[4] for ep in self.episodes]))
        counts = np.pad(counts, (0, max(0, 4 - len(counts))))
        total = float(len(np.concatenate([ep[4] for ep in self.episodes])))
        t = self.balance_leader_near
        p_c = (counts[2] + counts[3]) / total
        w = np.ones(4, dtype=np.float32)
        if 0 < counts[1] and counts[0] > 0:
            w[1] = (t / (1.0 - t)) * (counts[0] / counts[1])
        if 0 < counts[3] and counts[2] > 0:
            w[3] = (t / (1.0 - t)) * (counts[2] / counts[3])
        w /= w.max()
        self.class_weights = w

    def resample_windows(self):
        rng = np.random.default_rng()
        windows = []
        for ep_idx, ep in enumerate(self.episodes):
            n_rows = ep[0].shape[0]
            for L in self.seq_lengths:
                if n_rows < L:
                    continue
                start = 0 if rng.random() < 0.5 else int(rng.integers(0, min(self.stride, n_rows - L + 1)))
                for s in range(start, n_rows - L + 1, self.stride):
                    windows.append((ep_idx, s, L))
        self.current_windows = windows

    def __len__(self):
        return len(self.current_windows) if self.seq_lengths else len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start, length = self.current_windows[idx] if self.seq_lengths else self.indices[idx]
        ep = self.episodes[ep_idx]
        lidar = torch.from_numpy(ep[0][start:start + length].copy())
        speed = torch.from_numpy(ep[1][start:start + length].copy())
        action = torch.from_numpy(ep[2][start:start + length].copy())

        if self.add_lidar_noise:
            std = (0.01 + 0.005 * lidar).clamp(max=0.1)
            lidar = (lidar + torch.randn_like(lidar) * std).clamp(min=0.0)

        if self.augment_command:
            speed = self._augment_command(speed)

        if self.mirror_prob > 0 and torch.rand(()) < self.mirror_prob:
            # 거울 증강: bin i <-> 359-i (FOV 가 대칭이라 각도 반전과 같음), 조향 부호 반전
            lidar = lidar.flip(-1)
            action = torch.stack((-action[:, 0], action[:, 1]), dim=1)

        if self.class_weights is not None:
            w = self.class_weights[ep[4][start:start + length]]
            if ep[5] is not None:
                w = w * ep[5][start:start + length]
            weight = torch.from_numpy(w.copy())
        elif ep[5] is not None:
            weight = torch.from_numpy(ep[5][start:start + length].copy())
        else:
            weight = torch.ones(length, dtype=torch.float32)

        return lidar, speed, action, weight

    def _augment_command(self, speed: torch.Tensor) -> torch.Tensor:
        if self.speed_input == "hybrid":
            cmd = speed[:, 1].clone()
        else:
            cmd = speed[:, 0].clone()

        T = cmd.shape[0]
        if self.cmd_lag_prob > 0:
            lag_steps = max(1, int(round(self.cmd_lag_s / 0.01))) if self.downsample_factor == 1 else max(1, int(round(self.cmd_lag_s / (0.01 * self.downsample_factor))))
            lagged = torch.roll(cmd, lag_steps)
            lagged[:lag_steps] = 0.0
            mask_lag = torch.rand(T) < self.cmd_lag_prob
            cmd = torch.where(mask_lag, lagged, cmd)

        if self.cmd_drop > 0:
            mask_drop = torch.rand(T) < self.cmd_drop
            if self.speed_input == "hybrid":
                replacement = speed[:, 0]
            else:
                replacement = torch.empty(T).uniform_(0.0, CMD_DROP_MAX_SPEED)
            cmd = torch.where(mask_drop, replacement, cmd)

        if self.cmd_noise > 0:
            noise = torch.randn(T) * self.cmd_noise
            cmd = (cmd + noise).clamp(min=0.0)

        if self.speed_input == "hybrid":
            return torch.stack((speed[:, 0], cmd), dim=1)
        return cmd[:, None]


class LengthGroupedBatchSampler(Sampler):
    def __init__(self, dataset: SequenceDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        by_len = {}
        for idx, (_, _, L) in enumerate(self.dataset.current_windows):
            by_len.setdefault(L, []).append(idx)
        batches = []
        for L, idxs in by_len.items():
            if self.shuffle:
                np.random.shuffle(idxs)
            for i in range(0, len(idxs) - self.batch_size + 1, self.batch_size):
                batches.append(idxs[i:i + self.batch_size])
        if self.shuffle:
            np.random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(len(idxs) // self.batch_size for idxs in self._group_counts().values())

    def _group_counts(self):
        by_len = {}
        for _, _, L in self.dataset.current_windows:
            by_len.setdefault(L, []).append(1)
        return by_len


def classify_rows(df: pd.DataFrame, near_gap: float) -> np.ndarray:
    has_leader = ("leader_dist_signed" in df.columns) and ("curvature" in df.columns)
    if not has_leader:
        return np.zeros(len(df), dtype=np.int64)
    dist = df["leader_dist_signed"].to_numpy()
    curv = df["curvature"].abs().to_numpy()
    near = (dist > 0.0) & (dist <= near_gap)
    corner = curv >= 0.25
    return (corner.astype(np.int64) * 2) + near.astype(np.int64)


def skip_first_mask(df: pd.DataFrame, skip_first_s: float) -> Optional[np.ndarray]:
    if skip_first_s <= 0 or "time" not in df.columns:
        return None
    t = df["time"].to_numpy()
    return (t >= t[0] + skip_first_s).astype(np.float32)


def check_speed_column(df, csv_file: str, speed_input: str):
    if speed_input in ("odom", "hybrid") and "odom_speed" not in df.columns:
        raise ValueError(f"Speed input '{speed_input}' requires column 'odom_speed' in {csv_file}")


def split_episode_files(data_path: str, val_fraction: float, seed: int = 42):
    csv_files = sorted(glob.glob(os.path.join(data_path, "**", "*.csv"), recursive=True))
    if val_fraction <= 0 or not csv_files:
        return csv_files, []
    rng = np.random.default_rng(seed)
    by_map = {}
    for f in csv_files:
        map_name = os.path.basename(os.path.dirname(f))
        by_map.setdefault(map_name, []).append(f)
    train_files, val_files = [], []
    for files in by_map.values():
        shuffled = rng.permutation(files).tolist()
        k = max(1, int(round(len(files) * val_fraction)))
        val_files.extend(shuffled[:k])
        train_files.extend(shuffled[k:])
    return train_files, val_files


def load_validation_episodes(val_files, device, use_lidar_delay: bool, speed_input: str = "command",
                             skip_first_s: float = 0.0, downsample_mode: str = "lidar_sync",
                             lidar_stride: int = 3, downsample_factor: int = 1):
    episodes = []
    lidar_cols = [f"lidar_{i}" for i in range(360)]
    for f in val_files:
        df = pd.read_csv(f)
        if downsample_mode == "lidar_sync" and lidar_stride > 1:
            df, _ = extract_lidar_sync_df(df, lidar_cols, stride=lidar_stride, offset=0)
        elif downsample_factor > 1:
            df = df.iloc[::downsample_factor].reset_index(drop=True)
        check_speed_column(df, f, speed_input)
        action = df[["steer", "desired_speed"]].values.astype(np.float32)
        if speed_input == "odom":
            speed_in = df[["odom_speed"]].values.astype(np.float32)
        else:
            prev = np.roll(df["desired_speed"].values, 1).astype(np.float32)
            prev[0] = 0.0
            prev = prev[:, None]
            if speed_input == "hybrid":
                prev = np.concatenate((df[["odom_speed"]].values.astype(np.float32), prev), axis=1)
            speed_in = prev
        ep = {
            "lidar": torch.tensor(df[lidar_cols].values.astype(np.float32), device=device)[None],
            "speed": torch.tensor(speed_in, device=device)[None],
            "action": torch.tensor(action, device=device),
        }
        if use_lidar_delay:
            ep["delay"] = torch.tensor(df["lidar_delay"].values.astype(np.float32).reshape(-1, 1), device=device)[None]
        mask = skip_first_mask(df, skip_first_s)
        if mask is not None:
            ep["mask"] = torch.tensor(mask > 0, device=device)
        episodes.append(ep)
    return episodes


def evaluate(model, episodes, use_lidar_delay: bool, is_norm: bool = False):
    model.eval()
    steer_se = speed_se = 0.0
    steps = 0
    with torch.no_grad():
        for ep in episodes:
            if use_lidar_delay:
                pred, _ = model(ep["lidar"], ep["speed"], ep["delay"])
            else:
                pred, _ = model(ep["lidar"], ep["speed"])

            if is_norm:
                pred_real = torch.stack([pred[0, :, 0] * MAX_STEER, pred[0, :, 1] * MAX_SPEED], dim=-1)
                err = pred_real - ep["action"]
            else:
                err = pred[0] - ep["action"]

            if "mask" in ep:
                err = err[ep["mask"]]
            steer_se += float((err[:, 0] ** 2).sum())
            speed_se += float((err[:, 1] ** 2).sum())
            steps += err.shape[0]
    model.train()
    steps = max(steps, 1)
    steer_mse, speed_mse = steer_se / steps, speed_se / steps
    return {"loss": steer_mse + SPEED_LOSS_WEIGHT * speed_mse, "steer_loss": steer_mse, "speed_loss": speed_mse}


def build_selffeed_batch(episodes, max_episodes: int):
    if not episodes or max_episodes <= 0:
        return None
    idx = np.unique(np.linspace(0, len(episodes) - 1, min(max_episodes, len(episodes))).round().astype(int))
    eps = [episodes[i] for i in idx]
    lengths = [ep["action"].shape[0] for ep in eps]
    B, T = len(eps), max(lengths)
    device = eps[0]["action"].device
    lidar = torch.zeros(B, T, eps[0]["lidar"].shape[-1], device=device)
    odom = torch.zeros(B, T, device=device)
    action = torch.zeros(B, T, 2, device=device)
    valid = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b, (ep, n) in enumerate(zip(eps, lengths)):
        lidar[b, :n] = ep["lidar"][0]
        odom[b, :n] = ep["speed"][0, :, 0]
        action[b, :n] = ep["action"]
        valid[b, :n] = ep["mask"] if "mask" in ep else True
    return {"lidar": lidar, "odom": odom, "action": action, "valid": valid}


def evaluate_selffeed(model, batch, max_speed: float, is_norm: bool = False):
    model.eval()
    B, T = batch["odom"].shape
    last = torch.zeros(B, device=batch["odom"].device)
    hidden = None
    preds = torch.zeros(B, T, 2, device=last.device)
    with torch.no_grad():
        for t in range(T):
            s = (torch.stack((batch["odom"][:, t], last), dim=1) if model.speed_dims == 2
                 else last[:, None])[:, None, :]
            a, hidden = model(batch["lidar"][:, t:t + 1], s, hidden)
            if is_norm:
                pred_steer = a[:, 0, 0] * MAX_STEER
                pred_speed = a[:, 0, 1] * MAX_SPEED
                preds[:, t, 0] = pred_steer
                preds[:, t, 1] = pred_speed
                last = pred_speed.clamp(0.0, max_speed)
            else:
                preds[:, t] = a[:, 0]
                last = a[:, 0, 1].clamp(0.0, max_speed)
    model.train()
    err = (preds - batch["action"])[batch["valid"]]
    steps = max(err.shape[0], 1)
    steer_mse = float((err[:, 0] ** 2).sum()) / steps
    speed_mse = float((err[:, 1] ** 2).sum()) / steps
    return {"loss": steer_mse + SPEED_LOSS_WEIGHT * speed_mse, "steer_loss": steer_mse, "speed_loss": speed_mse}


def error_columns(prefix: str, metrics: dict) -> dict:
    return {
        prefix + "loss": metrics["loss"],
        prefix + "steer_loss": metrics["steer_loss"],
        prefix + "speed_loss": metrics["speed_loss"],
        prefix + "steer_rmse_deg": math.degrees(math.sqrt(metrics["steer_loss"])),
        prefix + "speed_rmse_mps": math.sqrt(metrics["speed_loss"]),
    }


def train(model_path, model, train_loader, optimizer, scheduler,
          num_epochs=100, use_lidar_delay=False, start_epoch=0, best_loss=float("inf"),
          val_episodes=None, selffeed_batch=None, selffeed_every=10, selffeed_max_speed=3.0,
          save_interval=10, is_norm=False, add_lidar_noise=True):
    loss_history = []
    loss_csv_path, loss_plot_path = get_loss_output_paths(model_path)
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    if start_epoch > 0 and os.path.exists(loss_csv_path):
        previous = pd.read_csv(loss_csv_path)
        loss_history = previous[previous["epoch"] <= start_epoch].to_dict("records")
        print(f"Kept {len(loss_history)} earlier epochs from {loss_csv_path}")

    interval_best_loss = float("inf")
    interval_best_epoch = -1
    interval_best_state = None
    device = next(model.parameters()).device

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = total_steer_loss = total_speed_loss = 0.0

            if hasattr(train_loader.dataset, "resample_windows"):
                train_loader.dataset.resample_windows()

            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    batch = [b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch]
                    if add_lidar_noise:
                        lidar = batch[0]
                        std = (0.01 + 0.005 * lidar).clamp(max=0.1)
                        batch[0] = (lidar + torch.randn_like(lidar) * std).clamp(min=0.0)
                    if callable(scheduler):
                        scheduler(epoch + (batch_idx / len(train_loader)))
                    optimizer.zero_grad()
                    if use_lidar_delay:
                        predicted_actions, _ = model(batch[0], batch[1], batch[2])
                        target_actions = batch[3]
                        weights = batch[4]
                    else:
                        predicted_actions, _ = model(batch[0], batch[1])
                        target_actions = batch[2]
                        weights = batch[3]

                    pred_flat = predicted_actions.view(-1, predicted_actions.shape[-1])
                    tgt_flat = target_actions.view(-1, target_actions.shape[-1])
                    w_flat = weights.view(-1)
                    w_sum = w_flat.sum().clamp_min(1e-8)

                    if is_norm:
                        tgt_norm_steer = tgt_flat[:, 0] / MAX_STEER
                        tgt_norm_speed = tgt_flat[:, 1] / MAX_SPEED
                        steer_loss = (w_flat * (pred_flat[:, 0] - tgt_norm_steer) ** 2).sum() / w_sum
                        speed_loss = (w_flat * (pred_flat[:, 1] - tgt_norm_speed) ** 2).sum() / w_sum
                        loss = steer_loss + speed_loss * SPEED_LOSS_WEIGHT
                        steer_loss_real = steer_loss * (MAX_STEER ** 2)
                        speed_loss_real = speed_loss * (MAX_SPEED ** 2)
                    else:
                        steer_loss = (w_flat * (pred_flat[:, 0] - tgt_flat[:, 0]) ** 2).sum() / w_sum
                        speed_loss = (w_flat * (pred_flat[:, 1] - tgt_flat[:, 1]) ** 2).sum() / w_sum
                        loss = steer_loss + speed_loss * SPEED_LOSS_WEIGHT
                        steer_loss_real = steer_loss
                        speed_loss_real = speed_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    total_steer_loss += steer_loss_real.item()
                    total_speed_loss += speed_loss_real.item()
                    pbar.set_postfix(loss=loss.item())

            num_batches = len(train_loader)
            avg_loss = total_loss / num_batches
            avg_steer_loss = total_steer_loss / num_batches
            avg_speed_loss = total_speed_loss / num_batches
            current_lr = optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch + 1}
            row.update(error_columns("", {"loss": avg_loss, "steer_loss": avg_steer_loss,
                                          "speed_loss": avg_speed_loss}))
            message = (f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f} "
                       f"(steer {row['steer_rmse_deg']:.2f} deg, speed {row['speed_rmse_mps']:.3f} m/s RMSE)")
            select_loss = avg_loss
            if val_episodes:
                val = error_columns("val_", evaluate(model, val_episodes, use_lidar_delay, is_norm=is_norm))
                row.update(val)
                select_loss = val["val_loss"]
                message += (f" | Val: {val['val_loss']:.5f} (steer {val['val_steer_rmse_deg']:.2f} deg, "
                            f"speed {val['val_speed_rmse_mps']:.3f} m/s RMSE)")
            if selffeed_batch is not None and selffeed_every > 0 and \
                    ((epoch + 1) % selffeed_every == 0 or epoch + 1 == num_epochs):
                sf = error_columns("val_selffeed_", evaluate_selffeed(model, selffeed_batch, selffeed_max_speed, is_norm=is_norm))
                row.update(sf)
                message += (f" | Val self-fed ({selffeed_batch['odom'].shape[0]} ep): "
                            f"steer {sf['val_selffeed_steer_rmse_deg']:.2f} deg, "
                            f"speed {sf['val_selffeed_speed_rmse_mps']:.3f} m/s RMSE")
            row["learning_rate"] = current_lr
            loss_history.append(row)
            print(message)

            if not callable(scheduler):
                scheduler.step(select_loss)

            if select_loss < best_loss:
                best_loss = select_loss
                os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"  New best {'val ' if val_episodes else ''}loss: {best_loss:.5f}. Model saved to {model_path}.")

            if select_loss < interval_best_loss:
                interval_best_loss = select_loss
                interval_best_epoch = epoch + 1
                interval_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if save_interval > 0 and ((epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs):
                path_stem, path_ext = os.path.splitext(model_path)
                interval_path = f"{path_stem}_ep{epoch + 1:03d}{path_ext}"
                os.makedirs(os.path.dirname(os.path.abspath(interval_path)), exist_ok=True)
                if interval_best_state is None:
                    interval_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    interval_best_epoch = epoch + 1
                    interval_best_loss = select_loss
                torch.save(interval_best_state, interval_path)
                print(f"  [Interval checkpoint] Saved best model for epochs "
                      f"{max(1, epoch + 2 - save_interval)}~{epoch + 1} "
                      f"(from ep {interval_best_epoch}, {'val' if val_episodes else 'train'} loss: {interval_best_loss:.5f}) "
                      f"to {interval_path}")
                interval_best_loss = float("inf")
                interval_best_state = None
                interval_best_epoch = -1

            if (epoch + 1) == num_epochs:
                path_stem, path_ext = os.path.splitext(model_path)
                last_path = f"{path_stem}_last{path_ext}"
                torch.save(model.state_dict(), last_path)
                print(f"  [Final checkpoint] Saved last epoch weights to {last_path}")
    finally:
        save_loss_history(loss_history, loss_csv_path, loss_plot_path)
        if loss_history:
            print(f"Loss history saved: {loss_csv_path}")
            print(f"Loss plot saved: {loss_plot_path}")


if __name__ == "__main__":
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 다운샘플링이 활성화된 경우 기본적으로 use_norm 권장
    is_norm = args.use_norm or (args.downsample_factor > 1)

    date_str = datetime.now().strftime("%Y%m%d")
    if args.model_path is None:
        norm_suffix = "_norm" if is_norm else ""
        ds_suffix = f"_ds{args.downsample_factor}" if args.downsample_factor > 1 else ""
        args.model_path = f"{args.mode}{norm_suffix}{ds_suffix}_{date_str}.pth"
    print(f"Mode: {args.mode} | Model path: {args.model_path} | Normalized: {is_norm}")
    if args.downsample_factor > 1:
        print(f"Downsample factor: {args.downsample_factor} (Raw 100Hz -> ~{100.0/args.downsample_factor:.1f}Hz, dt={0.01*args.downsample_factor:.2f}s)")

    use_lidar_delay = (args.mode == "lidar_delay")
    if use_lidar_delay and args.speed_input == "hybrid":
        raise ValueError("--speed_input hybrid is only implemented for --mode origin")
    if use_lidar_delay and is_norm:
        raise NotImplementedError("End2RaceWithDelay is not supported with normalized model. Please use standard model.")

    train_files, val_files = split_episode_files(args.data_path, args.val_fraction, args.val_seed)
    print(f"Episodes: {len(train_files)} train, {len(val_files)} validation "
          f"(val_fraction {args.val_fraction}, seed {args.val_seed})")

    dataset = SequenceDataset(
        csv_files=train_files,
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        use_lidar_delay=use_lidar_delay,
        stride=args.stride,
        device=device,
        add_lidar_noise=False,  # Optimized: moved to GPU batch augmentation in train()
        start_from_rest=args.start_from_rest,
        seq_lengths=args.seq_lengths,
        balance_leader_near=args.balance_leader_near,
        near_gap=args.near_gap,
        speed_input=args.speed_input,
        skip_first_s=args.skip_first_s,
        cmd_drop=args.cmd_drop,
        cmd_noise=args.cmd_noise,
        cmd_lag_prob=args.cmd_lag_prob,
        cmd_lag_s=args.cmd_lag_s,
        downsample_factor=args.downsample_factor,
        downsample_mode=args.downsample_mode,
        lidar_stride=args.lidar_stride,
        multi_phase=not args.no_multi_phase,
        mirror_prob=args.mirror_prob,
    )
    print(f"Speed input: {args.speed_input}")
    if args.speed_input == "none":
        print("LiDAR-only model: no speed input (dummy embedding every step); mask_prob/cmd_* ignored")
    print(f"Mirror augmentation: {'off' if args.mirror_prob <= 0 else f'p={args.mirror_prob}'}")
    if args.speed_input == "hybrid":
        print(f"Command channel augmentation: drop->odom {dataset.cmd_drop}, noise sigma {dataset.cmd_noise} m/s, "
              f"lag {args.cmd_lag_s} s with prob {dataset.cmd_lag_prob}")
    elif args.speed_input == "command":
        if dataset.augment_command:
            print(f"Previous-command augmentation: drop->uniform[0, {CMD_DROP_MAX_SPEED}] {dataset.cmd_drop}, "
                  f"noise sigma {dataset.cmd_noise} m/s (clamped >= 0), lag {args.cmd_lag_s} s with prob {dataset.cmd_lag_prob}")
        else:
            print("Previous-command augmentation: off")

    if args.skip_first_s > 0:
        print(f"Loss/validation skip the first {args.skip_first_s} s of every episode (rows still fed to the GRU)")
    print(f"LiDAR noise augmentation: {'disabled' if args.disable_lidar_noise else 'enabled (GPU batch-accelerated)'}")
    if args.downsample_factor > 1:
        print(f"Downsample mode: {args.downsample_mode} (lidar_stride={args.lidar_stride}, multi_phase={not args.no_multi_phase})")

    val_episodes = (load_validation_episodes(val_files, device, use_lidar_delay, args.speed_input,
                                             args.skip_first_s, downsample_mode=args.downsample_mode,
                                             lidar_stride=args.lidar_stride, downsample_factor=args.downsample_factor)
                    if val_files else None)
    selffeed_batch = (build_selffeed_batch(val_episodes, args.selffeed_episodes)
                      if val_episodes and not use_lidar_delay and args.speed_input in ("hybrid", "command", "none")
                      else None)

    dataloader_kwargs = {'pin_memory': True, 'num_workers': args.num_workers} if device.type != "cpu" else {}
    print(f"DataLoader: {args.num_workers} workers, pin_memory={device.type != 'cpu'}")
    if args.seq_lengths:
        train_loader = DataLoader(
            dataset,
            batch_sampler=LengthGroupedBatchSampler(dataset, args.batch_size),
            **dataloader_kwargs
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **dataloader_kwargs
        )

    speed_dims = {"hybrid": 2, "none": 0}.get(args.speed_input, 1)
    if is_norm and speed_dims == 0:
        raise ValueError("--speed_input none 은 End2Race(비정규화) 전용입니다 (--use_norm / --downsample_factor>1 과 함께 쓸 수 없음)")
    if is_norm:
        model = End2RaceNorm(
            mask_prob=args.mask_prob,
            hidden_scale=args.hidden_scale,
            speed_dims=speed_dims,
            max_steer=MAX_STEER,
            max_speed=MAX_SPEED
        ).to(device)
    elif use_lidar_delay:
        model = End2RaceWithDelay(mask_prob=args.mask_prob, hidden_scale=args.hidden_scale).to(device)
    else:
        model = End2Race(mask_prob=args.mask_prob, hidden_scale=args.hidden_scale, speed_dims=speed_dims).to(device)

    print(f"Model architecture: {type(model).__name__}")
    print(f"Train batches: {len(train_loader)}")

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None and os.path.exists(args.model_path):
        checkpoint_path = args.model_path

    best_loss = float("inf")
    if checkpoint_path:
        print(f"Loading checkpoint weights from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        ck_dims = speed_dims_from_state_dict(state)
        if ck_dims != speed_dims:
            raise ValueError(f"Checkpoint speed_dims ({ck_dims}) does not match current model speed_dims ({speed_dims})")
        model.load_state_dict(state)
        if checkpoint_path == args.model_path and val_episodes:
            val_init = evaluate(model, val_episodes, use_lidar_delay, is_norm=is_norm)
            best_loss = val_init["loss"]
            print(f"Initial best val loss from existing checkpoint: {best_loss:.5f}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.lr_schedule == "cosine":
        total_steps = args.num_epochs
        warmup_steps = args.warmup_epochs
        start_step = args.start_epoch
        resume_steps = args.resume_warmup_epochs if start_step > 0 else 0.0

        def lr_at(epoch_float):
            if start_step > 0 and epoch_float < start_step + resume_steps:
                target_lr = (args.min_lr + 0.5 * (args.learning_rate - args.min_lr) *
                             (1.0 + math.cos(math.pi * (start_step + resume_steps) / total_steps)))
                progress = (epoch_float - start_step) / max(resume_steps, 1e-6)
                return progress * target_lr
            if epoch_float < warmup_steps:
                return (epoch_float / max(warmup_steps, 1e-6)) * args.learning_rate
            progress = (epoch_float - warmup_steps) / max(total_steps - warmup_steps, 1e-6)
            progress = min(max(progress, 0.0), 1.0)
            return args.min_lr + 0.5 * (args.learning_rate - args.min_lr) * (1.0 + math.cos(math.pi * progress))

        def step_lr(epoch_float):
            lr = lr_at(epoch_float)
            for group in optimizer.param_groups:
                group["lr"] = lr

        scheduler = step_lr
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6, verbose=True
        )

    train(
        model_path=args.model_path,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        use_lidar_delay=use_lidar_delay,
        start_epoch=args.start_epoch,
        best_loss=best_loss,
        val_episodes=val_episodes,
        selffeed_batch=selffeed_batch,
        selffeed_every=args.selffeed_every,
        selffeed_max_speed=args.selffeed_max_speed,
        save_interval=args.save_interval,
        is_norm=is_norm,
        add_lidar_noise=not args.disable_lidar_noise,
    )

    print("\nTraining completed successfully!")

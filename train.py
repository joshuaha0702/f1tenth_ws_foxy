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
from model import End2Race, End2RaceWithDelay, speed_dims_from_state_dict


SPEED_LOSS_WEIGHT = 0.05
CMD_DROP_MAX_SPEED = 2.5  # --speed_input command, --cmd_drop: replacement speed ~ U[0, this]


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

    # Use a non-interactive backend so plotting also works in Docker/headless runs.
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
        axis.grid(True, alpha=0.3, which="both")
        axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train End2Race speed-conditioned model')

    parser.add_argument("--data_path", type=str, default="data/simple_variants_h2h/train")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model save path. Auto-generated from mode and date if not specified.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Optional checkpoint path to initialize model weights before training.")
    parser.add_argument("--mode", type=str, choices=["origin", "lidar_delay"], default="origin",
                        help="'origin': train without lidar_delay / 'lidar_delay': train with lidar_delay input")

    parser.add_argument("--sequence_length", type=int, default=400,
                        help="Number of timesteps per training sequence")
    parser.add_argument("--stride", type=int, default=50,
                        help="Step size for sliding window over each episode")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=None,
                        help="Train on windows of several lengths instead of --sequence_length/--stride. "
                             "Every epoch each episode is tiled with windows of each length at a random "
                             "offset (offset 0 half of the time, i.e. from the episode start), so the GRU "
                             "sees horizons up to the longest length. Batches hold a single length.")
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--decimate", type=int, default=1,
                        help="Keep every N-th row of each episode (100 Hz -> 100/N Hz) before windowing, like the "
                             "End2Race paper's 10 Hz downsampling (N=10). Scale --seq_lengths/--sequence_length by 1/N. "
                             "The agent must then infer at the same rate (agent_node inference_decimate / timer_rate_hz).")
    parser.add_argument("--mirror_prob", type=float, default=0.0,
                        help="Mirror augmentation: with this probability per training window flip the "
                             "360 LiDAR bins left<->right and negate the steering label (speed inputs "
                             "unchanged). Equivalent to driving the mirrored track, i.e. the opposite "
                             "direction on a symmetric vehicle; 0.5 balances CW/CCW.")
    parser.add_argument("--disable_lidar_noise", action="store_true",
                        help="Disable distance-dependent Gaussian noise augmentation")
    parser.add_argument("--start_from_rest", action="store_true",
                        help="Also train on each episode's first row with previous speed 0, "
                             "matching inference that starts every episode from rest")
    parser.add_argument("--speed_input", choices=["command", "odom", "hybrid", "none"], default="command",
                        help="'command': speed input = previous speed command (desired_speed of the "
                             "previous row; the agent feeds back its own last output). 'odom': speed "
                             "input = measured speed at the row's time (CSV column odom_speed, same row, "
                             "no shift; the agent feeds /<robot>/odom twist.linear.x). Copying the "
                             "previous command reproduces the label to 0.035 m/s RMSE, so 'command' "
                             "lets the model skip learning to brake from the LiDAR. 'hybrid': 2-dim speed "
                             "input [odom_speed (same row), previous command (as in 'command')]; during "
                             "training the command channel is made unreliable (--cmd_drop/--cmd_noise/"
                             "--cmd_lag_prob, redrawn every epoch); the agent feeds [odom, own last output].")
    parser.add_argument("--cmd_drop", type=float, default=None,
                        help="hybrid/command: per-timestep probability of replacing the previous-command "
                             "input; hybrid: by the measured odom speed of the row, command: by a random "
                             f"speed uniform in [0, {CMD_DROP_MAX_SPEED}] (default: hybrid 0.3, command 0)")
    parser.add_argument("--cmd_noise", type=float, default=None,
                        help="hybrid/command: sigma (m/s) of Gaussian noise added to the previous-command "
                             "input (command: then clamped at >= 0) (default: hybrid 0.15, command 0)")
    parser.add_argument("--cmd_lag_prob", type=float, default=None,
                        help="hybrid/command: per-timestep probability of using the command from "
                             "--cmd_lag_s earlier (default: hybrid 0.2, command 0)")
    parser.add_argument("--cmd_lag_s", type=float, default=0.5,
                        help="hybrid/command: lag of --cmd_lag_prob in seconds (rows from the CSV time step; "
                             "0 before the episode start)")
    parser.add_argument("--selffeed_every", type=int, default=10,
                        help="hybrid/command: every N epochs (and on the last epoch) also run the validation "
                             "episodes step by step feeding the model's own previous speed output into the "
                             "command input (val_selffeed_* columns; the checkpoint still follows val_loss)")
    parser.add_argument("--selffeed_episodes", type=int, default=40,
                        help="hybrid/command: at most this many validation episodes (evenly spaced) for val_selffeed")
    parser.add_argument("--selffeed_max_speed", type=float, default=3.0,
                        help="hybrid/command: self-fed command is clipped to [0, this] like agent_node control.max_speed")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--lr_schedule", choices=["plateau", "cosine"], default="plateau",
                        help="'plateau': halve lr after 10 epochs without improvement (old default). "
                             "'cosine': linear warmup, then cosine decay to --min_lr at the last epoch, "
                             "updated every batch. Use cosine with --seq_lengths: windows are redrawn "
                             "every epoch, so the epoch loss is noisy and plateau keeps halving the lr "
                             "(it hit ~1e-8 by epoch 500 on 2026-10-06).")
    parser.add_argument("--warmup_epochs", type=float, default=5.0,
                        help="Cosine schedule: epochs of linear warmup from 0 to --learning_rate")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Cosine schedule: learning rate reached at the last epoch")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Resume: epochs already completed by the weights being loaded. Training runs "
                             "epochs start_epoch+1..num_epochs and the cosine schedule continues from "
                             "there instead of restarting. Rows of an existing loss CSV up to start_epoch "
                             "are kept.")
    parser.add_argument("--resume_warmup_epochs", type=float, default=1.0,
                        help="Resume: optimizer state is not saved, so ramp the lr from 0 back onto the "
                             "schedule over this many epochs to let Adam's moments settle")
    parser.add_argument("--balance_leader_near", type=float, default=None,
                        help="Target share of 'leader within --near_gap m ahead' rows, applied separately "
                             "in corners (|raceline curvature| >= 0.25) and elsewhere, by down-weighting "
                             "those rows in the loss. Needs <csv>.meta.npz from "
                             "tools/annotate_training_csv.py. On Freeform data 65 %% of the corner time is "
                             "spent crawling behind the leader and the model learned 'corner -> slow'.")
    parser.add_argument("--near_gap", type=float, default=3.0)
    parser.add_argument("--skip_first_s", type=float, default=0.0,
                        help="Exclude the rows of the first SEC seconds of every episode (time < first "
                             "time + SEC) from the training loss and the validation metrics by giving them "
                             "weight 0. They are still fed to the GRU, so the hidden state is warmed up. "
                             "0 (default) keeps every row.")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Share of episodes per map held out for validation (0 disables). Every epoch "
                             "the model runs each validation episode from its start with a fresh hidden "
                             "state, as when driving, and the checkpoint keeps the lowest validation loss.")
    parser.add_argument("--val_seed", type=int, default=0,
                        help="Seed of the per-map validation split (keep it fixed across runs)")
    parser.add_argument("--best_loss", type=float, default=float("inf"),
                        help="Resume: epoch loss of the loaded weights; the checkpoint is only "
                             "overwritten by an epoch below it")

    return parser.parse_args()


class SequenceDataset(Dataset):

    def __init__(self, data_path: str, sequence_length: int = 50,
                 use_lidar_delay: bool = False, stride: int = 1, device: str = "cuda",
                 add_lidar_noise: bool = True, start_from_rest: bool = False,
                 seq_lengths: Optional[list] = None, csv_files: Optional[list] = None,
                 balance_leader_near: Optional[float] = None, near_gap: float = 3.0,
                 speed_input: str = "command", skip_first_s: float = 0.0,
                 cmd_drop: Optional[float] = None, cmd_noise: Optional[float] = None,
                 cmd_lag_prob: Optional[float] = None, cmd_lag_s: float = 0.5,
                 mirror_prob: float = 0.0, decimate: int = 1):
        self.sequence_length = sequence_length
        self.mirror_prob = mirror_prob
        self.decimate = max(int(decimate), 1)
        self.seq_lengths = sorted(seq_lengths) if seq_lengths else None
        self.episodes = []
        self.csv_files = csv_files
        self.balance_leader_near = balance_leader_near
        self.near_gap = near_gap
        # row classes: 0 elsewhere/leader far, 1 elsewhere/near, 2 corner/far, 3 corner/near, 4 unknown
        self.class_weight = np.ones(5, dtype=np.float32)
        self.stride = stride
        self.device = device
        self.use_lidar_delay = use_lidar_delay
        self.add_lidar_noise = add_lidar_noise
        self.start_from_rest = start_from_rest
        self.speed_input = speed_input
        self.skip_first_s = skip_first_s
        self.cmd_drop, self.cmd_noise, self.cmd_lag_prob = resolve_cmd_augmentation(
            speed_input, cmd_drop, cmd_noise, cmd_lag_prob)
        self.cmd_lag_s = cmd_lag_s
        # command mode with any augmentation: speed stored as [previous command, lagged command]
        self.augment_command = (speed_input == "command" and
                                (self.cmd_drop > 0 or self.cmd_noise > 0 or self.cmd_lag_prob > 0))

        self.lidar_columns = [f"lidar_{i}" for i in range(360)]
        self.action_columns = ["steer", "desired_speed"]

        self.sequences = []
        self._load_episodes(data_path)
        if self.balance_leader_near is not None:
            self._set_class_weights()

        if self.seq_lengths:
            self.resample()
            counts = {length: sum(len(s['action']) == length for s in self.sequences)
                      for length in self.seq_lengths}
            print(f"Sequence lengths: {self.seq_lengths} | {len(self.episodes)} episodes | "
                  f"windows per epoch (first draw): {counts}")
        else:
            print(f"Sequence_length: {sequence_length} | Loaded {len(self.sequences)} sequences")

    def _load_episodes(self, data_path: str):
        csv_files = self.csv_files if self.csv_files is not None else sorted(glob.glob(os.path.join(data_path, "*.csv")))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if self.decimate > 1:
                # 100 Hz -> 100/N Hz: N행마다 한 행. speed_prev/odom/lidar_delay 는 남은 행 기준으로 아래에서 계산됨
                df = df.iloc[::self.decimate].reset_index(drop=True)

            required_cols = self.lidar_columns + self.action_columns
            if self.use_lidar_delay:
                required_cols = required_cols + ["lidar_delay"]

            if not all(col in df.columns for col in required_cols):
                continue
            check_speed_column(df, csv_file, self.speed_input)
            min_rows = (min(self.seq_lengths) if self.seq_lengths else self.sequence_length) + 1
            if len(df) < min_rows:  # +1 for speed_prev offset
                continue

            lidar_data = df[self.lidar_columns].values.astype(np.float32)
            action_data = df[self.action_columns].values.astype(np.float32)
            delay_data = df["lidar_delay"].values.astype(np.float32).reshape(-1, 1) if self.use_lidar_delay else None
            odom_data = (df["odom_speed"].values.astype(np.float32).reshape(-1, 1)
                         if self.speed_input in ("odom", "hybrid") else None)
            if self.speed_input == "hybrid" or self.augment_command:
                # stored speed columns: hybrid [odom, previous command, previous command cmd_lag_s
                # earlier], __getitem__ augments column 1 (using 0 and 2) and returns the first two;
                # augmented command mode [previous command, lagged], __getitem__ returns 1 column
                lag_rows = 0
                if self.cmd_lag_prob > 0 and len(df) > 1:
                    lag_rows = max(1, int(round(self.cmd_lag_s / float(np.median(np.diff(df["time"].values))))))
                cmd_prev = np.concatenate(([0.0], action_data[:-1, 1])).astype(np.float32)
                cmd_lag = np.concatenate((np.zeros(min(lag_rows, len(df)), dtype=np.float32),
                                          cmd_prev[:max(len(df) - lag_rows, 0)]))
                odom_data = (np.stack((odom_data[:, 0], cmd_prev, cmd_lag), axis=1)
                             if self.speed_input == "hybrid" else np.stack((cmd_prev, cmd_lag), axis=1))

            classes = np.full(len(df), 4, dtype=np.int64)
            meta_path = csv_file + ".meta.npz"
            if self.balance_leader_near is not None and os.path.exists(meta_path):
                meta = np.load(meta_path)
                meta = {k: meta[k][::self.decimate] for k in meta.files}
                if len(meta["gap"]) == len(df):
                    near = (meta["gap"] > 0) & (meta["gap"] < self.near_gap)
                    corner = np.abs(meta["kappa"]) >= 0.25
                    classes = (2 * corner + near).astype(np.int64)
            loss_mask = skip_first_mask(df, self.skip_first_s)
            self._create_sequences(lidar_data, action_data, delay_data, classes, odom_data, loss_mask)

    def _set_class_weights(self):
        counts = np.bincount(np.concatenate([ep[4] for ep in self.episodes] if self.seq_lengths
                                            else [s["cls"] for s in self.sequences]), minlength=5)
        target = self.balance_leader_near
        for far, near, name in ((0, 1, "elsewhere"), (2, 3, "corner")):
            share = counts[near] / max(counts[near] + counts[far], 1)
            if share > target:
                self.class_weight[near] = target * counts[far] / ((1 - target) * counts[near])
            print(f"Leader-near share {name}: {share:.1%} -> weight {self.class_weight[near]:.3f} "
                  f"(target {target:.0%}, rows near {counts[near]}, far {counts[far]})")
        if counts[4]:
            print(f"Rows without .meta.npz context (weight 1): {counts[4]}")

    def _create_sequences(self, lidar_data: np.ndarray, action_data: np.ndarray,
                          delay_data: Optional[np.ndarray] = None, classes: Optional[np.ndarray] = None,
                          odom_data: Optional[np.ndarray] = None, loss_mask: Optional[np.ndarray] = None):
        if classes is None:
            classes = np.full(len(lidar_data), 4, dtype=np.int64)
        # loss_mask (--skip_first_s): per-row 0/1 loss weight, sliced like the classes below
        first_row = 0 if self.start_from_rest else 1
        mask_valid = loss_mask[first_row:] if loss_mask is not None else None
        if odom_data is not None:
            # (hybrid: odom_data holds [odom, previous command (0 at row 0), lagged command], and
            # augmented command mode [previous command, lagged command], so the same slicing gives
            # the 'command' mode alignment for the command columns.)
            # Measured speed when the command is issued: same row, no shift. Row 0 needs no
            # special case (its measured speed is ~0 at an episode start); without
            # --start_from_rest it is dropped anyway, as in command mode.
            first = 0 if self.start_from_rest else 1
            lidar_valid = lidar_data[first:]
            action_valid = action_data[first:]
            speed_prev = odom_data[first:]
            delay_valid = delay_data[first:] if delay_data is not None else None
            classes_valid = classes[first:]
        elif self.start_from_rest:
            # The End2Race agent resets its previous speed to 0 at every episode start,
            # so keep row 0 and feed it a previous speed of 0 instead of dropping it.
            lidar_valid = lidar_data
            action_valid = action_data
            speed_prev = np.concatenate((np.zeros((1, 1), dtype=action_data.dtype),
                                         action_data[:-1, 1:2]))
            delay_valid = delay_data
            classes_valid = classes
        else:
            lidar_valid = lidar_data[1:]
            action_valid = action_data[1:]
            speed_prev = action_data[:-1, 1:2]
            delay_valid = delay_data[1:] if delay_data is not None else None
            classes_valid = classes[1:]
        num_samples = len(lidar_valid)

        if self.seq_lengths:
            # windows are drawn per epoch in resample()
            self.episodes.append((lidar_valid, speed_prev, action_valid, delay_valid, classes_valid,
                                  mask_valid))
            return

        for end_idx in range(self.sequence_length - 1, num_samples, self.stride):
            start_idx = end_idx - self.sequence_length + 1
            if mask_valid is not None and not mask_valid[start_idx:end_idx + 1].any():
                continue  # window lies entirely in the skipped start: no loss rows
            seq = {
                'lidar': lidar_valid[start_idx:end_idx + 1],
                'speed': speed_prev[start_idx:end_idx + 1],
                'action': action_valid[start_idx:end_idx + 1],
                'cls': classes_valid[start_idx:end_idx + 1],
            }
            if delay_valid is not None:
                seq['delay'] = delay_valid[start_idx:end_idx + 1]
            if mask_valid is not None:
                seq['mask'] = mask_valid[start_idx:end_idx + 1]
            self.sequences.append(seq)

    def resample(self):
        """Redraw the multi-length windows (seq_lengths mode)."""
        self.sequences = []
        for lidar, speed, action, delay, classes, mask in self.episodes:
            n = len(lidar)
            for length in self.seq_lengths:
                if n < length:
                    continue
                offset = 0 if np.random.rand() < 0.5 else np.random.randint(0, min(length, n - length + 1))
                for start in range(offset, n - length + 1, length):
                    if mask is not None and not mask[start:start + length].any():
                        continue  # window lies entirely in the skipped start: no loss rows
                    seq = {
                        'lidar': lidar[start:start + length],
                        'speed': speed[start:start + length],
                        'action': action[start:start + length],
                        'cls': classes[start:start + length],
                    }
                    if delay is not None:
                        seq['delay'] = delay[start:start + length]
                    if mask is not None:
                        seq['mask'] = mask[start:start + length]
                    self.sequences.append(seq)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        lidar = torch.tensor(seq['lidar'], dtype=torch.float32, device=self.device)
        if self.add_lidar_noise:
            # Linearly interpolate sigma from 0.9 mm at 1 m to 1.8 mm at
            # 12 m. Values outside that range use the nearest endpoint.
            distance_for_sigma = lidar.clamp(min=1.0, max=12.0)
            sigma = 0.0009 + (distance_for_sigma - 1.0) * (0.0018 - 0.0009) / 11.0
            lidar = (lidar + torch.randn_like(lidar) * sigma).clamp_min(0.0)
        speed = torch.tensor(seq['speed'], dtype=torch.float32, device=self.device)
        if self.speed_input == "hybrid":
            speed = self._augment_command(speed)
        elif self.augment_command:
            speed = self._augment_command_only(speed)
        action = torch.tensor(seq['action'], dtype=torch.float32, device=self.device)
        if self.mirror_prob > 0 and torch.rand(()) < self.mirror_prob:
            # 거울 증강: bin i <-> 359-i (FOV 가 대칭이라 각도 반전과 같음), 조향 부호 반전.
            # 속도 입력/라벨, 행 가중치, lidar_delay 는 그대로.
            lidar = lidar.flip(-1)
            action = torch.stack((-action[:, 0], action[:, 1]), dim=1)
        row_weight = self.class_weight[seq['cls']]
        if 'mask' in seq:
            row_weight = row_weight * seq['mask']
        weight = torch.tensor(row_weight, dtype=torch.float32, device=self.device)

        if self.use_lidar_delay:
            delay = torch.tensor(seq['delay'], dtype=torch.float32, device=self.device)
            return lidar, speed, delay, action, weight

        return lidar, speed, action, weight


    def _augment_command(self, speed: torch.Tensor) -> torch.Tensor:
        """hybrid: [odom, cmd, cmd_lagged] (T, 3) -> (T, 2) [odom, unreliable cmd]. Per timestep:
        lagged command with prob cmd_lag_prob, then odom instead of the command with prob cmd_drop,
        then Gaussian noise cmd_noise on the command channel. Redrawn on every access."""
        odom, cmd = speed[:, 0], speed[:, 1]
        if self.cmd_lag_prob > 0:
            cmd = torch.where(torch.rand_like(cmd) < self.cmd_lag_prob, speed[:, 2], cmd)
        if self.cmd_drop > 0:
            cmd = torch.where(torch.rand_like(cmd) < self.cmd_drop, odom, cmd)
        if self.cmd_noise > 0:
            cmd = cmd + torch.randn_like(cmd) * self.cmd_noise
        return torch.stack((odom, cmd), dim=1)

    def _augment_command_only(self, speed: torch.Tensor) -> torch.Tensor:
        """command: [cmd, cmd_lagged] (T, 2) -> (T, 1) unreliable previous command. Per timestep:
        lagged command with prob cmd_lag_prob, then a random speed uniform in [0, CMD_DROP_MAX_SPEED]
        with prob cmd_drop, then Gaussian noise cmd_noise, clamped at >= 0. Redrawn on every access."""
        cmd = speed[:, 0]
        if self.cmd_lag_prob > 0:
            cmd = torch.where(torch.rand_like(cmd) < self.cmd_lag_prob, speed[:, 1], cmd)
        if self.cmd_drop > 0:
            cmd = torch.where(torch.rand_like(cmd) < self.cmd_drop,
                              torch.rand_like(cmd) * CMD_DROP_MAX_SPEED, cmd)
        if self.cmd_noise > 0:
            cmd = cmd + torch.randn_like(cmd) * self.cmd_noise
        return cmd.clamp_min(0.0)[:, None]


def resolve_cmd_augmentation(speed_input: str, cmd_drop, cmd_noise, cmd_lag_prob):
    """None -> mode default: hybrid (0.3, 0.15, 0.2), otherwise 0 (command stays clean unless asked)."""
    defaults = (0.3, 0.15, 0.2) if speed_input == "hybrid" else (0.0, 0.0, 0.0)
    return tuple(d if v is None else float(v) for v, d in zip((cmd_drop, cmd_noise, cmd_lag_prob), defaults))


class LengthGroupedBatchSampler(Sampler):
    """Redraws the dataset's multi-length windows every epoch and yields shuffled
    batches whose windows share one length (so they stack without padding)."""

    def __init__(self, dataset: SequenceDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self._batches = self._make_batches()

    def _make_batches(self):
        by_length = {}
        for idx, seq in enumerate(self.dataset.sequences):
            by_length.setdefault(len(seq['action']), []).append(idx)
        batches = []
        for indices in by_length.values():
            np.random.shuffle(indices)
            batches += [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        np.random.shuffle(batches)
        return batches

    def __iter__(self):
        batches, self._batches = self._batches, None
        if batches is None:
            self.dataset.resample()
            batches = self._make_batches()
        self._last_len = len(batches)
        return iter(batches)

    def __len__(self):
        return len(self._batches) if self._batches is not None else self._last_len


def split_episode_files(data_path: str, val_fraction: float, seed: int):
    """Hold out val_fraction of the episodes of every map (file prefix before '_ep')."""
    files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    if val_fraction <= 0:
        return files, []
    by_map = {}
    for f in files:
        by_map.setdefault(os.path.basename(f).split("_ep")[0], []).append(f)
    rng = np.random.default_rng(seed)
    train_files, val_files = [], []
    for name in sorted(by_map):
        group = by_map[name]
        n_val = max(1, int(round(len(group) * val_fraction))) if len(group) > 1 else 0
        picked = set(rng.choice(len(group), size=n_val, replace=False).tolist())
        for i, f in enumerate(group):
            (val_files if i in picked else train_files).append(f)
    return train_files, val_files


def skip_first_mask(df, skip_first_s: float) -> Optional[np.ndarray]:
    """--skip_first_s: float 0/1 per-row loss mask (0 for time < first time + skip), or None if off."""
    if not skip_first_s or skip_first_s <= 0:
        return None
    t = df["time"].values.astype(np.float64)
    return (t >= t[0] + skip_first_s).astype(np.float32)


def check_speed_column(df, csv_file: str, speed_input: str):
    if speed_input in ("odom", "hybrid"):
        if "odom_speed" not in df.columns:
            raise ValueError(f"--speed_input odom: {csv_file} has no 'odom_speed' column. Re-extract it "
                             f"with the current src/f1tenth_lattice_ros2/tools/extract_bag_csv.py.")
        if df["odom_speed"].isna().any():
            raise ValueError(f"--speed_input odom: {csv_file} has NaN odom_speed (no odom in the bag?)")


def load_validation_episodes(val_files, device, use_lidar_delay: bool, speed_input: str = "command",
                             skip_first_s: float = 0.0, decimate: int = 1):
    """Whole episodes as tensors, run from the first row like the agent: previous speed
    command 0 at row 0 ('command'), or the measured speed of each row ('odom'), or both
    ('hybrid': [odom, expert previous command], clean)."""
    episodes = []
    lidar_cols = [f"lidar_{i}" for i in range(360)]
    for f in val_files:
        df = pd.read_csv(f)
        if decimate > 1:
            df = df.iloc[::decimate].reset_index(drop=True)
        check_speed_column(df, f, speed_input)
        action = df[["steer", "desired_speed"]].values.astype(np.float32)
        if speed_input == "odom":
            prev = df[["odom_speed"]].values.astype(np.float32)
        else:
            prev = np.concatenate(([[0.0]], action[:-1, 1:2])).astype(np.float32)
            if speed_input == "hybrid":
                prev = np.concatenate((df[["odom_speed"]].values.astype(np.float32), prev), axis=1)
        ep = {
            "lidar": torch.tensor(df[lidar_cols].values.astype(np.float32), device=device)[None],
            "speed": torch.tensor(prev, device=device)[None],
            "action": torch.tensor(action, device=device),
        }
        if use_lidar_delay:
            ep["delay"] = torch.tensor(df["lidar_delay"].values.astype(np.float32).reshape(-1, 1), device=device)[None]
        mask = skip_first_mask(df, skip_first_s)
        if mask is not None:
            ep["mask"] = torch.tensor(mask > 0, device=device)
        episodes.append(ep)
    return episodes


def evaluate(model, episodes, use_lidar_delay: bool):
    """Step-weighted steer/speed MSE over whole validation episodes (eval mode: no mask)."""
    model.eval()
    steer_se = speed_se = 0.0
    steps = 0
    with torch.no_grad():
        for ep in episodes:
            if use_lidar_delay:
                pred, _ = model(ep["lidar"], ep["speed"], ep["delay"])
            else:
                pred, _ = model(ep["lidar"], ep["speed"])
            err = pred[0] - ep["action"]
            if "mask" in ep:  # --skip_first_s: the whole episode is run, metrics skip the start
                err = err[ep["mask"]]
            steer_se += float((err[:, 0] ** 2).sum())
            speed_se += float((err[:, 1] ** 2).sum())
            steps += err.shape[0]
    model.train()
    steps = max(steps, 1)
    steer_mse, speed_mse = steer_se / steps, speed_se / steps
    return {"loss": steer_mse + SPEED_LOSS_WEIGHT * speed_mse, "steer_loss": steer_mse, "speed_loss": speed_mse}


def build_selffeed_batch(episodes, max_episodes: int):
    """hybrid/command: up to max_episodes evenly spaced validation episodes, zero-padded into one batch
    for evaluate_selffeed (built once)."""
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


def evaluate_selffeed(model, batch, max_speed: float):
    """hybrid/command: run all batched validation episodes step by step from rest, command input =
    the model's own previous speed output clipped to [0, max_speed] (as agent_node drives).
    hybrid feeds [odom, own command]; command (model.speed_dims == 1) feeds [own command]."""
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
            preds[:, t] = a[:, 0]
            last = a[:, 0, 1].clamp(0.0, max_speed)
    model.train()
    err = (preds - batch["action"])[batch["valid"]]
    steps = max(err.shape[0], 1)
    steer_mse = float((err[:, 0] ** 2).sum()) / steps
    speed_mse = float((err[:, 1] ** 2).sum()) / steps
    return {"loss": steer_mse + SPEED_LOSS_WEIGHT * speed_mse, "steer_loss": steer_mse, "speed_loss": speed_mse}


def error_columns(prefix: str, metrics: dict) -> dict:
    """loss/steer_loss/speed_loss plus RMSE in driving units."""
    return {
        prefix + "loss": metrics["loss"],
        prefix + "steer_loss": metrics["steer_loss"],
        prefix + "speed_loss": metrics["speed_loss"],
        prefix + "steer_rmse_deg": math.degrees(math.sqrt(metrics["steer_loss"])),
        prefix + "speed_rmse_mps": math.sqrt(metrics["speed_loss"]),
    }


def cosine_lr(progress_epochs: float, num_epochs: int, peak_lr: float, min_lr: float,
              warmup_epochs: float) -> float:
    """Learning rate at a fractional epoch: linear warmup, then cosine decay to min_lr."""
    if progress_epochs < warmup_epochs:
        return peak_lr * progress_epochs / warmup_epochs
    t = (progress_epochs - warmup_epochs) / max(num_epochs - warmup_epochs, 1e-9)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * min(t, 1.0)))


def train(model_path, model, train_loader, criterion, optimizer, scheduler,
          num_epochs=100, use_lidar_delay=False, start_epoch=0, best_loss=float("inf"),
          val_episodes=None, selffeed_batch=None, selffeed_every=10, selffeed_max_speed=3.0):
    """scheduler: a ReduceLROnPlateau stepped with the epoch loss, or a function
    lr(fractional_epoch) applied before every batch. start_epoch: epochs already done.
    val_episodes: if given, the checkpoint (and a plateau scheduler) follow the validation loss."""
    loss_history = []
    loss_csv_path, loss_plot_path = get_loss_output_paths(model_path)
    if start_epoch > 0 and os.path.exists(loss_csv_path):
        previous = pd.read_csv(loss_csv_path)
        loss_history = previous[previous["epoch"] <= start_epoch].to_dict("records")
        print(f"Kept {len(loss_history)} earlier epochs from {loss_csv_path}")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_loss = 0.0
            total_steer_loss = 0.0
            total_speed_loss = 0.0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                num_batches_est = max(len(train_loader), 1)
                for batch_idx, batch in enumerate(pbar):
                    if callable(scheduler):
                        lr = scheduler(epoch + batch_idx / num_batches_est)
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                    optimizer.zero_grad()

                    if use_lidar_delay:
                        lidar_seq, speed_seq, delay_seq, target_actions, weights = batch
                        predicted_actions, _ = model(lidar_seq, speed_seq, delay_seq)
                    else:
                        lidar_seq, speed_seq, target_actions, weights = batch
                        predicted_actions, _ = model(lidar_seq, speed_seq)

                    pred_flat = predicted_actions.view(-1, predicted_actions.shape[-1])
                    tgt_flat = target_actions.view(-1, target_actions.shape[-1])

                    # row-weighted MSE (all weights 1 unless --balance_leader_near)
                    w_flat = weights.reshape(-1)
                    # (--skip_first_s rows have weight 0; the clamp only guards an all-skipped batch)
                    w_sum = w_flat.sum().clamp_min(1e-8)
                    steer_loss = (w_flat * (pred_flat[:, 0] - tgt_flat[:, 0]) ** 2).sum() / w_sum
                    speed_loss = (w_flat * (pred_flat[:, 1] - tgt_flat[:, 1]) ** 2).sum() / w_sum
                    loss = steer_loss + speed_loss * SPEED_LOSS_WEIGHT

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    total_steer_loss += steer_loss.item()
                    total_speed_loss += speed_loss.item()
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
                val = error_columns("val_", evaluate(model, val_episodes, use_lidar_delay))
                row.update(val)
                select_loss = val["val_loss"]
                message += (f" | Val: {val['val_loss']:.5f} (steer {val['val_steer_rmse_deg']:.2f} deg, "
                            f"speed {val['val_speed_rmse_mps']:.3f} m/s RMSE)")
            if selffeed_batch is not None and selffeed_every > 0 and \
                    ((epoch + 1) % selffeed_every == 0 or epoch + 1 == num_epochs):
                sf = error_columns("val_selffeed_", evaluate_selffeed(model, selffeed_batch, selffeed_max_speed))
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
                torch.save(model.state_dict(), model_path)
                print(f"  New best {'val ' if val_episodes else ''}loss: {best_loss:.5f}. Model saved.")
    finally:
        save_loss_history(loss_history, loss_csv_path, loss_plot_path)
        if loss_history:
            print(f"Loss history saved: {loss_csv_path}")
            print(f"Loss plot saved: {loss_plot_path}")


if __name__ == "__main__":
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    date_str = datetime.now().strftime("%Y%m%d")
    if args.model_path is None:
        args.model_path = f"{args.mode}_{date_str}.pth"
    print(f"Mode: {args.mode} | Model path: {args.model_path}")

    use_lidar_delay = (args.mode == "lidar_delay")
    if use_lidar_delay and args.speed_input == "hybrid":
        raise ValueError("--speed_input hybrid is only implemented for --mode origin")

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
        add_lidar_noise=not args.disable_lidar_noise,
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
        mirror_prob=args.mirror_prob,
        decimate=args.decimate,
    )
    if args.decimate > 1:
        print(f"Decimation: every {args.decimate}-th row ({100 // args.decimate} Hz); the agent must run at the same rate")
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
                  f"noise sigma {dataset.cmd_noise} m/s (clamped >= 0), lag {args.cmd_lag_s} s with prob "
                  f"{dataset.cmd_lag_prob}")
        else:
            print("Previous-command augmentation: off")
    if args.skip_first_s > 0:
        print(f"Loss/validation skip the first {args.skip_first_s} s of every episode (rows still fed to the GRU)")
    print(f"LiDAR noise augmentation: {'disabled' if args.disable_lidar_noise else 'enabled'}")
    val_episodes = (load_validation_episodes(val_files, device, use_lidar_delay, args.speed_input,
                                             args.skip_first_s, args.decimate)
                    if val_files else None)
    selffeed_batch = (build_selffeed_batch(val_episodes, args.selffeed_episodes)
                      if val_episodes and not use_lidar_delay and args.speed_input in ("hybrid", "command", "none")
                      else None)

    dataloader_kwargs = {'pin_memory': False, 'num_workers': 0} if device.type != "cpu" else {}
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

    if use_lidar_delay:
        model = End2RaceWithDelay(mask_prob=args.mask_prob, hidden_scale=args.hidden_scale).to(device)
    else:
        model = End2Race(mask_prob=args.mask_prob, hidden_scale=args.hidden_scale,
                         speed_dims={"hybrid": 2, "none": 0}.get(args.speed_input, 1)).to(device)

    print(f"Train batches: {len(train_loader)}")

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None and os.path.exists(args.model_path):
        checkpoint_path = args.model_path

    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading pretrained weights from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not use_lidar_delay and speed_dims_from_state_dict(state) != model.speed_dims:
            raise ValueError(f"{checkpoint_path} has speed input width {speed_dims_from_state_dict(state)}, "
                             f"--speed_input {args.speed_input} needs {model.speed_dims}")
        ckpt_hidden = int(state["gru.weight_hh_l0"].shape[1])
        if ckpt_hidden != model.gru.hidden_size:
            raise ValueError(f"{checkpoint_path} has GRU hidden size {ckpt_hidden} "
                             f"(hidden_scale {ckpt_hidden / (model.gru.hidden_size / args.hidden_scale):g}), "
                             f"--hidden_scale {args.hidden_scale} needs {model.gru.hidden_size}")
        model.load_state_dict(state)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.lr_schedule == "cosine":
        def scheduler(progress_epochs):
            lr = cosine_lr(progress_epochs, args.num_epochs, args.learning_rate,
                           args.min_lr, args.warmup_epochs)
            since_resume = progress_epochs - args.start_epoch
            if args.start_epoch > 0 and since_resume < args.resume_warmup_epochs:
                lr *= since_resume / args.resume_warmup_epochs
            return lr
        print(f"LR schedule: warmup {args.warmup_epochs} epochs to {args.learning_rate}, "
              f"cosine to {args.min_lr} at epoch {args.num_epochs}")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    if args.start_epoch > 0:
        print(f"Resuming after epoch {args.start_epoch} (lr {scheduler(args.start_epoch + args.resume_warmup_epochs) if callable(scheduler) else args.learning_rate:.2e} "
              f"after {args.resume_warmup_epochs} warmup epochs), best loss {args.best_loss}")

    train(
        model_path=args.model_path,
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        start_epoch=args.start_epoch,
        best_loss=args.best_loss,
        val_episodes=val_episodes,
        use_lidar_delay=use_lidar_delay,
        selffeed_batch=selffeed_batch,
        selffeed_every=args.selffeed_every,
        selffeed_max_speed=args.selffeed_max_speed,
    )

    print("\nTraining completed successfully!")

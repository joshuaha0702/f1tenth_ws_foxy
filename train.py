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
from model import End2Race, End2RaceWithDelay


SPEED_LOSS_WEIGHT = 0.05


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
    parser.add_argument("--disable_lidar_noise", action="store_true",
                        help="Disable distance-dependent Gaussian noise augmentation")
    parser.add_argument("--start_from_rest", action="store_true",
                        help="Also train on each episode's first row with previous speed 0, "
                             "matching inference that starts every episode from rest")

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
                 seq_lengths: Optional[list] = None, csv_files: Optional[list] = None):
        self.sequence_length = sequence_length
        self.seq_lengths = sorted(seq_lengths) if seq_lengths else None
        self.episodes = []
        self.csv_files = csv_files
        self.stride = stride
        self.device = device
        self.use_lidar_delay = use_lidar_delay
        self.add_lidar_noise = add_lidar_noise
        self.start_from_rest = start_from_rest

        self.lidar_columns = [f"lidar_{i}" for i in range(360)]
        self.action_columns = ["steer", "desired_speed"]

        self.sequences = []
        self._load_episodes(data_path)

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

            required_cols = self.lidar_columns + self.action_columns
            if self.use_lidar_delay:
                required_cols = required_cols + ["lidar_delay"]

            if not all(col in df.columns for col in required_cols):
                continue
            min_rows = (min(self.seq_lengths) if self.seq_lengths else self.sequence_length) + 1
            if len(df) < min_rows:  # +1 for speed_prev offset
                continue

            lidar_data = df[self.lidar_columns].values.astype(np.float32)
            action_data = df[self.action_columns].values.astype(np.float32)
            delay_data = df["lidar_delay"].values.astype(np.float32).reshape(-1, 1) if self.use_lidar_delay else None

            self._create_sequences(lidar_data, action_data, delay_data)

    def _create_sequences(self, lidar_data: np.ndarray, action_data: np.ndarray,
                          delay_data: Optional[np.ndarray] = None):
        if self.start_from_rest:
            # The End2Race agent resets its previous speed to 0 at every episode start,
            # so keep row 0 and feed it a previous speed of 0 instead of dropping it.
            lidar_valid = lidar_data
            action_valid = action_data
            speed_prev = np.concatenate((np.zeros((1, 1), dtype=action_data.dtype),
                                         action_data[:-1, 1:2]))
            delay_valid = delay_data
        else:
            lidar_valid = lidar_data[1:]
            action_valid = action_data[1:]
            speed_prev = action_data[:-1, 1:2]
            delay_valid = delay_data[1:] if delay_data is not None else None
        num_samples = len(lidar_valid)

        if self.seq_lengths:
            # windows are drawn per epoch in resample()
            self.episodes.append((lidar_valid, speed_prev, action_valid, delay_valid))
            return

        for end_idx in range(self.sequence_length - 1, num_samples, self.stride):
            start_idx = end_idx - self.sequence_length + 1
            seq = {
                'lidar': lidar_valid[start_idx:end_idx + 1],
                'speed': speed_prev[start_idx:end_idx + 1],
                'action': action_valid[start_idx:end_idx + 1],
            }
            if delay_valid is not None:
                seq['delay'] = delay_valid[start_idx:end_idx + 1]
            self.sequences.append(seq)

    def resample(self):
        """Redraw the multi-length windows (seq_lengths mode)."""
        self.sequences = []
        for lidar, speed, action, delay in self.episodes:
            n = len(lidar)
            for length in self.seq_lengths:
                if n < length:
                    continue
                offset = 0 if np.random.rand() < 0.5 else np.random.randint(0, min(length, n - length + 1))
                for start in range(offset, n - length + 1, length):
                    seq = {
                        'lidar': lidar[start:start + length],
                        'speed': speed[start:start + length],
                        'action': action[start:start + length],
                    }
                    if delay is not None:
                        seq['delay'] = delay[start:start + length]
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
        action = torch.tensor(seq['action'], dtype=torch.float32, device=self.device)

        if self.use_lidar_delay:
            delay = torch.tensor(seq['delay'], dtype=torch.float32, device=self.device)
            return lidar, speed, delay, action

        return lidar, speed, action


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


def load_validation_episodes(val_files, device, use_lidar_delay: bool):
    """Whole episodes as tensors; previous speed is 0 at the first row, like the agent."""
    episodes = []
    lidar_cols = [f"lidar_{i}" for i in range(360)]
    for f in val_files:
        df = pd.read_csv(f)
        action = df[["steer", "desired_speed"]].values.astype(np.float32)
        prev = np.concatenate(([[0.0]], action[:-1, 1:2])).astype(np.float32)
        ep = {
            "lidar": torch.tensor(df[lidar_cols].values.astype(np.float32), device=device)[None],
            "speed": torch.tensor(prev, device=device)[None],
            "action": torch.tensor(action, device=device),
        }
        if use_lidar_delay:
            ep["delay"] = torch.tensor(df["lidar_delay"].values.astype(np.float32).reshape(-1, 1), device=device)[None]
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
            steer_se += float((err[:, 0] ** 2).sum())
            speed_se += float((err[:, 1] ** 2).sum())
            steps += err.shape[0]
    model.train()
    steer_mse, speed_mse = steer_se / steps, speed_se / steps
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
          val_episodes=None):
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
                        lidar_seq, speed_seq, delay_seq, target_actions = batch
                        predicted_actions, _ = model(lidar_seq, speed_seq, delay_seq)
                    else:
                        lidar_seq, speed_seq, target_actions = batch
                        predicted_actions, _ = model(lidar_seq, speed_seq)

                    pred_flat = predicted_actions.view(-1, predicted_actions.shape[-1])
                    tgt_flat = target_actions.view(-1, target_actions.shape[-1])

                    steer_loss = criterion(pred_flat[:, 0], tgt_flat[:, 0])
                    speed_loss = criterion(pred_flat[:, 1], tgt_flat[:, 1])
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
        seq_lengths=args.seq_lengths
    )
    print(f"LiDAR noise augmentation: {'disabled' if args.disable_lidar_noise else 'enabled'}")
    val_episodes = load_validation_episodes(val_files, device, use_lidar_delay) if val_files else None

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
        model = End2Race(mask_prob=args.mask_prob, hidden_scale=args.hidden_scale).to(device)

    print(f"Train batches: {len(train_loader)}")

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None and os.path.exists(args.model_path):
        checkpoint_path = args.model_path

    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading pretrained weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))

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
        use_lidar_delay=use_lidar_delay
    )

    print("\nTraining completed successfully!")

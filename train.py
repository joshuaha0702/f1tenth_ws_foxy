import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from tqdm import tqdm
from typing import Optional, Tuple
from model import End2Race, End2RaceWithDelay

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train End2Race speed-conditioned model')

    parser.add_argument("--data_path", type=str, default="Dataset_Austin/success")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model save path. Auto-generated from mode and date if not specified.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Optional checkpoint path to initialize model weights before training.")
    parser.add_argument("--mode", type=str, choices=["origin", "lidar_delay"], default="origin",
                        help="'origin': train without lidar_delay / 'lidar_delay': train with lidar_delay input")

    parser.add_argument("--sequence_length", type=int, default=100,
                        help="Number of timesteps per training sequence")
    parser.add_argument("--stride", type=int, default=50,
                        help="Step size for sliding window over each episode")
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--mask_prob", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=500)

    return parser.parse_args()


class SequenceDataset(Dataset):

    def __init__(self, data_path: str, sequence_length: int = 50,
                 use_lidar_delay: bool = False, stride: int = 1, device: str = "cuda"):
        self.sequence_length = sequence_length
        self.stride = stride
        self.device = device
        self.use_lidar_delay = use_lidar_delay

        self.lidar_columns = [f"lidar_{i}" for i in range(360)]
        self.action_columns = ["steer", "desired_speed"]

        self.sequences = []
        self._load_episodes(data_path)

        print(f"Sequence_length: {sequence_length} | Loaded {len(self.sequences)} sequences")

    def _load_episodes(self, data_path: str):
        csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            required_cols = self.lidar_columns + self.action_columns
            if self.use_lidar_delay:
                required_cols = required_cols + ["lidar_delay"]

            if not all(col in df.columns for col in required_cols):
                continue
            if len(df) < self.sequence_length + 1:  # +1 for speed_prev offset
                continue

            lidar_data = df[self.lidar_columns].values.astype(np.float32)
            action_data = df[self.action_columns].values.astype(np.float32)
            delay_data = df["lidar_delay"].values.astype(np.float32).reshape(-1, 1) if self.use_lidar_delay else None

            self._create_sequences(lidar_data, action_data, delay_data)

    def _create_sequences(self, lidar_data: np.ndarray, action_data: np.ndarray,
                          delay_data: Optional[np.ndarray] = None):
        lidar_valid = lidar_data[1:]
        action_valid = action_data[1:]
        speed_prev = action_data[:-1, 1:2]
        delay_valid = delay_data[1:] if delay_data is not None else None
        num_samples = len(lidar_valid)

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

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        lidar = torch.tensor(seq['lidar'], dtype=torch.float32, device=self.device)
        speed = torch.tensor(seq['speed'], dtype=torch.float32, device=self.device)
        action = torch.tensor(seq['action'], dtype=torch.float32, device=self.device)

        if self.use_lidar_delay:
            delay = torch.tensor(seq['delay'], dtype=torch.float32, device=self.device)
            return lidar, speed, delay, action

        return lidar, speed, action


def train(model_path, model, train_loader, criterion, optimizer, scheduler,
          num_epochs=100, use_lidar_delay=False):
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch in pbar:
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
                loss = steer_loss + speed_loss * 0.05

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}")

        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"  New best loss: {best_loss:.5f}. Model saved.")


if __name__ == "__main__":
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    date_str = datetime.now().strftime("%Y%m%d")
    if args.model_path is None:
        args.model_path = f"{args.mode}_{date_str}.pth"
    print(f"Mode: {args.mode} | Model path: {args.model_path}")

    use_lidar_delay = (args.mode == "lidar_delay")

    dataset = SequenceDataset(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        use_lidar_delay=use_lidar_delay,
        stride=args.stride,
        device=device
    )

    dataloader_kwargs = {'pin_memory': False, 'num_workers': 0} if device.type != "cpu" else {}
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    train(
        model_path=args.model_path,
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        use_lidar_delay=use_lidar_delay
    )

    print("\nTraining completed successfully!")

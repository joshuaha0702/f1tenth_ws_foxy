#!/bin/bash
set -e

SOURCE_DIR=/root/f1tenth_ws
DATA_DIR=$SOURCE_DIR/data
LOG_CHAIN=$SOURCE_DIR/master_100hz_k_random_chain.log
TARGET_MODEL=$SOURCE_DIR/src/f1tenth_end2race_ros2/models/ego_norm_100hz_real_car166_8000_k_random_ep100.pth
TRAIN_LOG=$SOURCE_DIR/training_car166_100hz_k_random_100ep.log

exec > >(tee -a "$LOG_CHAIN") 2>&1

echo "======================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 100Hz Random k Master Autonomous Chain Started"
echo "Step 1: 100-Epoch 100Hz Uniform Random k Training (8,000 ep)"
echo "Step 2: Best Checkpoint Selection"
echo "Step 3: 3-Worker Parallel 100Hz OOD Benchmark Evaluation (12 Tracks x 10 ep)"
echo "Step 4: Generate Comparative Analysis Summary"
echo "======================================================================"

# Step 1: Run Training
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Step 1: Training..."
bash $SOURCE_DIR/src/run_car166_100hz_k_random_100ep.sh
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 1 Complete: Training Finished."

# Step 2: Select Best Checkpoint
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Step 2: Selecting Best Checkpoint..."
python3 -c "
import re, os, glob

log_file = '$TRAIN_LOG'
best_loss = float('inf')
best_ep = 100

if os.path.exists(log_file):
    with open(log_file) as f:
        for line in f:
            # Matches: Epoch 70/100, Loss: ... | Val: 0.00215 (steer 2.15 deg, speed 0.081 m/s RMSE)
            m = re.search(r'Epoch\s+(\d+)/\d+.*Val:\s*([\d\.]+)', line)
            if m:
                ep = int(m.group(1))
                val_l = float(m.group(2))
                if val_l < best_loss and ep % 10 == 0:
                    best_loss = val_l
                    best_ep = ep

print(f'Best evaluated checkpoint from training: Epoch {best_ep} (Val loss {best_loss:.5f})')
best_model_path = f'/root/f1tenth_ws/src/f1tenth_end2race_ros2/models/ego_norm_100hz_real_car166_8000_k_random_ep100_ep{best_ep:03d}.pth'
if not os.path.exists(best_model_path):
    best_model_path = '$TARGET_MODEL'

symlink_path = '/root/f1tenth_ws/src/f1tenth_end2race_ros2/models/best_selffeed_real_car166_8000_100hz_k_random.pth'
if os.path.exists(symlink_path) or os.path.islink(symlink_path):
    os.remove(symlink_path)
os.symlink(best_model_path, symlink_path)
print(f'Symlinked {symlink_path} -> {best_model_path}')
"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2 Complete: Best Checkpoint Symlinked."

# Step 3: Run OOD Benchmark
BEST_EVAL_MODEL="$SOURCE_DIR/src/f1tenth_end2race_ros2/models/best_selffeed_real_car166_8000_100hz_k_random.pth"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Step 3: 100Hz OOD Evaluation with $BEST_EVAL_MODEL..."
bash $SOURCE_DIR/src/run_car166_100hz_k_random_ood_eval_3workers.sh "$BEST_EVAL_MODEL"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 3 Complete: OOD Benchmark Finished."

echo "======================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 100Hz Random k Pipeline Steps Completed Successfully!"
echo "======================================================================"

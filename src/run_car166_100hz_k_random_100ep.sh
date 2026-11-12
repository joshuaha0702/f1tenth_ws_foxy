#!/bin/bash
set -e

SOURCE_DIR=/root/f1tenth_ws
DATA_DIR=$SOURCE_DIR/data
TRAIN_DIR=$DATA_DIR/train_mix_real_car166_8000/car1
TARGET_MODEL=$SOURCE_DIR/src/f1tenth_end2race_ros2/models/ego_norm_100hz_real_car166_8000_k_random_ep100.pth
LOG_FILE=$SOURCE_DIR/training_car166_100hz_k_random_100ep.log

echo "======================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 100Hz Uniform Random k Training (100 Epochs, Full Dataset)"
echo "k Init: Uniform Random in [0.2, 1.6]"
echo "Speed Bounds: min_speed=0.5 m/s, max_speed=2.0 m/s, max_steer=0.28 rad"
echo "Sequence Lengths: 200 400 800 1150 (stride 50)"
echo "Learning Rates: Base=0.0004, k=0.0020, Min=1e-5, Warmup=5.0 Epochs"
echo "Target Model: $TARGET_MODEL"
echo "Dataset: $TRAIN_DIR (8,000 episodes)"
echo "======================================================================"

# Clean up previous checkpoints if any
rm -f "$TARGET_MODEL" "${TARGET_MODEL%.pth}"*.pth

python3 -u $SOURCE_DIR/train_k_random.py \
    --data_path "$TRAIN_DIR" \
    --model_path "$TARGET_MODEL" \
    --k_init_mode uniform \
    --k_random_min 0.2 \
    --k_random_max 1.6 \
    --k_learning_rate 0.0020 \
    --k_min_lr 1e-5 \
    --k_clamp_min 0.05 \
    --k_clamp_max 3.0 \
    --hidden_scale 1 \
    --use_norm \
    --max_steer 0.28 \
    --min_speed 0.5 \
    --max_speed 2.0 \
    --downsample_factor 1 \
    --seq_lengths 200 400 800 1150 \
    --stride 50 \
    --batch_size 32 \
    --num_workers 6 \
    --learning_rate 0.0004 \
    --num_epochs 100 \
    --lr_schedule cosine \
    --warmup_epochs 5.0 \
    --min_lr 1e-6 \
    --speed_input command \
    --mirror_prob 0.5 \
    --start_from_rest \
    --cmd_drop 0.3 \
    --cmd_noise 0.2 \
    --cmd_lag_prob 0.2 \
    --cmd_lag_s 0.5 \
    --save_interval 10 \
    --val_fraction 0.1 \
    --use_amp > "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 100-Epoch 100Hz Uniform Random k Training Completed! Saved to $TARGET_MODEL"

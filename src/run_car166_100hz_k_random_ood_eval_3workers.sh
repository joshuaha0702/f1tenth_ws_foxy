#!/bin/bash
set -e

SOURCE_DIR=/root/f1tenth_ws
DATA_DIR=$SOURCE_DIR/data
MODEL_PATH=${1:-"$SOURCE_DIR/src/f1tenth_end2race_ros2/models/ego_norm_100hz_real_car166_8000_k_random_ep100.pth"}
OOD_ROOT=${2:-"$DATA_DIR/ood_eval_car166_100hz_k_random"}
OOD_OUT_DIR=$OOD_ROOT/episodes
SUMMARY_CSV=${3:-"$DATA_DIR/ood_eval/ood_car166_100hz_k_random_summary.csv"}

mkdir -p "$OOD_OUT_DIR/logs" "$DATA_DIR/ood_eval"

echo "======================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 3-Worker Parallel 100Hz OOD Evaluation (12 Tracks x 10 Episodes)"
echo "Model: $MODEL_PATH"
echo "Model Rate Hz: 100.0"
echo "Output Directory: $OOD_OUT_DIR"
echo "Summary CSV: $SUMMARY_CSV"
echo "======================================================================"

# Clean up fastrtps shared memory
rm -f /dev/shm/fastrtps*

# Worker 1 (Domain 21, Gazebo 11361)
bash -c "
source /opt/ros/foxy/setup.bash
source $SOURCE_DIR/install/setup.bash
export ROS_DOMAIN_ID=21
export GAZEBO_MASTER_URI=http://localhost:11361
export CONFIG_DIR=$SOURCE_DIR/src/f1tenth_lattice_ros2/maps
export EGO=end2race
export EGO_MODEL=$MODEL_PATH
export LEADER=lattice
export HEAD2HEAD=true
export DURATION=12
export EXTRA_LAUNCH_ARGS='headless:=true use_norm:=true model_rate_hz:=100.0 interp_mode:=zoh max_steer:=0.28 min_speed:=0.5 max_speed:=2.0'
bash $SOURCE_DIR/src/f1tenth_lattice_ros2/tools/run_episodes.sh \
    $OOD_OUT_DIR 10 20263001 Simple_n04 Spline_n04 Freeform_n04 Simple_v07
" > "$OOD_ROOT/ood_w1.log" 2>&1 &
PID_W1=$!

# Worker 2 (Domain 22, Gazebo 11362)
bash -c "
source /opt/ros/foxy/setup.bash
source $SOURCE_DIR/install/setup.bash
export ROS_DOMAIN_ID=22
export GAZEBO_MASTER_URI=http://localhost:11362
export CONFIG_DIR=$SOURCE_DIR/src/f1tenth_lattice_ros2/maps
export EGO=end2race
export EGO_MODEL=$MODEL_PATH
export LEADER=lattice
export HEAD2HEAD=true
export DURATION=12
export EXTRA_LAUNCH_ARGS='headless:=true use_norm:=true model_rate_hz:=100.0 interp_mode:=zoh max_steer:=0.28 min_speed:=0.5 max_speed:=2.0'
bash $SOURCE_DIR/src/f1tenth_lattice_ros2/tools/run_episodes.sh \
    $OOD_OUT_DIR 10 20263001 Simple_v15 Spline_s04
bash $SOURCE_DIR/src/f1tenth_lattice_ros2/tools/run_episodes.sh \
    $OOD_OUT_DIR 10 20263101 Freeform_f04 Simple
" > "$OOD_ROOT/ood_w2.log" 2>&1 &
PID_W2=$!

# Worker 3 (Domain 23, Gazebo 11363)
bash -c "
source /opt/ros/foxy/setup.bash
source $SOURCE_DIR/install/setup.bash
export ROS_DOMAIN_ID=23
export GAZEBO_MASTER_URI=http://localhost:11363
export CONFIG_DIR=$SOURCE_DIR/src/f1tenth_lattice_ros2/maps
export EGO=end2race
export EGO_MODEL=$MODEL_PATH
export LEADER=lattice
export HEAD2HEAD=true
export DURATION=12
export EXTRA_LAUNCH_ARGS='headless:=true use_norm:=true model_rate_hz:=100.0 interp_mode:=zoh max_steer:=0.28 min_speed:=0.5 max_speed:=2.0'
bash $SOURCE_DIR/src/f1tenth_lattice_ros2/tools/run_episodes.sh \
    $OOD_OUT_DIR 10 20263001 Simple_augmented monza_track
bash $SOURCE_DIR/src/f1tenth_lattice_ros2/tools/run_episodes.sh \
    $OOD_OUT_DIR 10 20263101 interlagos_track silverstone_track
" > "$OOD_ROOT/ood_w3.log" 2>&1 &
PID_W3=$!

echo "Workers started (PID: W1=$PID_W1, W2=$PID_W2, W3=$PID_W3). Waiting for completion..."
wait $PID_W1
echo "Worker 1 finished."
wait $PID_W2
echo "Worker 2 finished."
wait $PID_W3
echo "Worker 3 finished."

echo "All workers completed. Generating summary..."
python3 -u $SOURCE_DIR/src/f1tenth_lattice_ros2/tools/validate_episodes.py \
    "$OOD_OUT_DIR" --csv "$SUMMARY_CSV"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 100Hz OOD Evaluation Completed! Summary: $SUMMARY_CSV"

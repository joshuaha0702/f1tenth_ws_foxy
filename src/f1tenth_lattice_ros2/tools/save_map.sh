#!/usr/bin/env bash
# ==============================================================================
# Script: save_map.sh
# Location: f1tenth_lattice_ros2/tools/save_map.sh
# Purpose: Create a date-based directory under maps/ and save the SLAM map
#          (.pgm and .yaml only)
# Usage:
#   ./save_map.sh                  # Automatically creates maps/map_YYYYMMDD_HHMMSS/
#   ./save_map.sh <custom_name>    # Creates maps/<custom_name>/
# ==============================================================================

set -e

# 1. Source ROS 2 environment
if [ -f "/opt/ros/foxy/setup.bash" ]; then
    source /opt/ros/foxy/setup.bash
fi

# 2. Locate tools and maps directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAPS_DIR="$(cd "${SCRIPT_DIR}/../maps" 2>/dev/null && pwd)"

if [ -z "${MAPS_DIR}" ] || [ ! -d "${MAPS_DIR}" ]; then
    MAPS_DIR="/home/capstone/f1tenth_ws_foxy/src/f1tenth_lattice_ros2/maps"
    mkdir -p "${MAPS_DIR}"
fi

# 3. Create folder based on current date/time (or user-provided name)
DATE_TAG="$(date +%Y%m%d_%H%M%S)"
MAP_NAME="${1:-map_${DATE_TAG}}"
TARGET_DIR="${MAPS_DIR}/${MAP_NAME}"

echo "============================================================"
echo "[1/3] Creating directory: ${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

BASE_NAME="${MAP_NAME}_map"
MAP_PREFIX="${TARGET_DIR}/${BASE_NAME}"

echo "[2/3] Saving map files (.pgm and .yaml)..."

# Priority 1: Check if slam_toolbox service is active
if ros2 service list 2>/dev/null | grep -q "/slam_toolbox/save_map"; then
    echo " -> Found slam_toolbox service! Saving via /slam_toolbox/save_map..."
    ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '${MAP_PREFIX}'}}" >/dev/null 2>&1

# Priority 2: Fallback to subscribing to /map topic via save_map.py
elif ros2 topic list 2>/dev/null | grep -q "^/map$"; then
    echo " -> Subscribing to /map topic via save_map.py..."
    python3 "${SCRIPT_DIR}/save_map.py" "${MAP_PREFIX}" /map
else
    echo "[ERROR] Neither /slam_toolbox/save_map service nor /map topic is active."
    exit 1
fi

# 4. Clean up any extra files to guarantee only .pgm and .yaml exist
find "${TARGET_DIR}" -type f ! -name "*.pgm" ! -name "*.yaml" -delete 2>/dev/null || true

echo "============================================================"
echo "[3/3] Map saved successfully!"
echo "Target directory: ${TARGET_DIR}"
ls -lh "${TARGET_DIR}"
echo "============================================================"

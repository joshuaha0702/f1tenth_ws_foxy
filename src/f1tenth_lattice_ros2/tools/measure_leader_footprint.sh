#!/bin/bash
# 두 차량을 직선 위에 2.5 m 간격으로 스폰하고 car1 LiDAR 에 잡히는 car2 의 빔/bin 수를 측정 (컨테이너 안에서 실행)
source /opt/ros/foxy/setup.bash; source /root/f1tenth_ws/install/setup.bash; cd /root/f1tenth_ws
label=${1:-measure}; gap=${2:-2.5}; static=${3:-1}
y2=$(python3 -c "print(16.0 - $gap)")
# static=1: 두 차를 End2Race 로 지정하되 모델 경로를 없는 파일로 주어 에이전트가 즉시 종료 → 구동 명령 없음 → 정지 상태 측정
extra=""; [ "$static" = 1 ] && extra="ego:=end2race leader:=end2race ego_model:=/nonexistent.pth leader_model:=/nonexistent.pth"
setsid ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py map:=Simple head2head:=true headless:=true record:=false \
  x:=6.4 y:=16.0 yaw_deg:=-90 x2:=6.4 y2:=$y2 yaw_deg2:=-90 $extra > /tmp/meas_launch_$label.log 2>&1 &
LP=$!
for i in $(seq 60); do ros2 topic list 2>/dev/null | grep -q "/car2/odom" && break; sleep 2; done; sleep 6
echo "--- $label (spawn gap $gap m, static=$static; LiDAR→셸 뒷면 거리 ≈ gap-0.265-0.115)"
python3 src/f1tenth_lattice_ros2/tools/measure_leader_footprint.py --seconds 4 --spawn1 6.4 16.0 -90 --spawn2 6.4 $y2 -90
kill -INT -- -$LP 2>/dev/null; sleep 5
for p in $(pgrep -f "gzserver|gzclient|agent_node|lattice_planner|pure_pursuit|episode_manager|ackermann_to_twist|robot_state_pub|joint_state|static_transform|ros2 launch f1tenth"); do [ "$p" != "$$" ] && kill -9 $p 2>/dev/null; done
sleep 2; echo "gz left: $(pgrep -c gzserver)"

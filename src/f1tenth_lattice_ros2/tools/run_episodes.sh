#!/bin/bash
# Run head-to-head episodes on several maps, one map after another.
#
# Run inside the container after building the workspace:
#   run_episodes.sh <output root> <episodes per map> <seed base> <map> [<map> ...]
#
# Each map uses <config dir>/<map>/lattice_config.yaml and a copy of
# <config dir>/<map>/episodes.yaml with num_episodes, random_seed (seed base + the
# map's trailing number) and output.base_dir (<output root>/<map>) replaced.
# Launch logs go to <output root>/logs/<map>.log. An existing <output root>/<map>
# folder is replaced, and a launch where a car fails to spawn is retried (3 tries).
#
# Environment overrides:
#   CONFIG_DIR  directory holding <map>/{lattice_config,episodes}.yaml (default: installed maps/)
#   EGO         lattice (default, data collection) or end2race (model evaluation)
#   EGO_MODEL   End2Race weights for EGO=end2race
#   LEADER      lattice (default) or end2race for car2
#   LEADER_MODEL  End2Race weights for LEADER=end2race
#   DURATION    sequence_duration_sec override (seconds)
#   ROS_DOMAIN_ID / GAZEBO_MASTER_URI  isolate this run from other ROS/Gazebo users
# (no "set -u": ROS setup scripts reference unset variables)
out_root=$1; episodes=$2; seed_base=$3; shift 3

source /opt/ros/foxy/setup.bash
source /root/f1tenth_ws/install/setup.bash
maps_dir=/root/f1tenth_ws/install/f1tenth_lattice_ros2/share/f1tenth_lattice_ros2/maps
config_dir=${CONFIG_DIR:-$maps_dir}
ego=${EGO:-lattice}
leader=${LEADER:-lattice}
mkdir -p "$out_root/logs"

stop_launch() {
    kill -INT -- -$launch_pid 2>/dev/null
    for _ in $(seq 30); do kill -0 $launch_pid 2>/dev/null || break; sleep 1; done
    kill -KILL -- -$launch_pid 2>/dev/null
    pkill -KILL -f 'gzserver|gzclient' 2>/dev/null   # container-local leftovers only
    sleep 2
}

for map in "$@"; do
    idx=$(( 10#$(echo "$map" | grep -o '[0-9]*$' || echo 0) ))
    cfg="$out_root/logs/${map}_episodes.yaml"
    sed -e "s/^num_episodes: .*/num_episodes: $episodes/" \
        -e "s/^random_seed: .*/random_seed: $((seed_base + idx))/" \
        -e "s#^  base_dir: .*#  base_dir: $out_root/$map#" \
        "$config_dir/$map/episodes.yaml" > "$cfg"
    if [ -n "$DURATION" ]; then
        sed -i "s/^sequence_duration_sec: .*/sequence_duration_sec: $DURATION/" "$cfg"
    fi
    log="$out_root/logs/$map.log"
    echo "[$(date +%T)] $map: start ($episodes episodes, seed $((seed_base + idx)), ego $ego, leader $leader)"
    # Foxy rejects empty launch arguments, so pass model paths only when set.
    model_args=()
    [ -n "$EGO_MODEL" ] && model_args+=("ego_model:=$EGO_MODEL")
    [ -n "$LEADER_MODEL" ] && model_args+=("leader_model:=$LEADER_MODEL")

    # A spawn_entity call occasionally fails (seen once: a DDS shared-memory error left
    # car1 unspawned and every episode recorded car2 alone). Relaunch until both cars exist.
    for attempt in 1 2 3; do
        rm -rf "$out_root/$map"
        setsid ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
            map:="$map" config:="$config_dir/$map/lattice_config.yaml" \
            head2head:=true headless:=true record:=true episodes:="$cfg" \
            ego:="$ego" leader:="$leader" "${model_args[@]}" > "$log" 2>&1 &
        launch_pid=$!
        spawned=0
        for _ in $(seq 60); do
            spawned=$(grep -ac 'Successfully spawned entity' "$log")
            [ "$spawned" -ge 2 ] && break
            kill -0 $launch_pid 2>/dev/null || break
            sleep 1
        done
        [ "$spawned" -ge 2 ] && break
        echo "[$(date +%T)] $map: only $spawned/2 cars spawned (attempt $attempt) — relaunching"
        stop_launch
    done
    # The launch keeps running after the episode manager exits; stop it ourselves.
    until grep -aqE 'All episodes done|episode loop crashed' "$log" || ! kill -0 $launch_pid 2>/dev/null; do
        sleep 5
    done
    stop_launch
    # The bag recorder can fail silently (seen once: every bag from one episode on was
    # empty or had no metadata.yaml), so report broken bags right away.
    broken=0
    for bag in "$out_root/$map"/*/ep*; do
        [ -d "$bag" ] || continue
        count=$(grep -m1 'message_count:' "$bag/metadata.yaml" 2>/dev/null | grep -o '[0-9]*')
        [ "${count:-0}" -ge 100 ] || broken=$((broken + 1))
    done
    echo "[$(date +%T)] $map: done — clean=$(grep -ac '] CLEAN' "$log") collision=$(grep -ac '] COLLISION' "$log") crashed=$(grep -ac 'loop crashed' "$log") broken_bags=$broken"
done

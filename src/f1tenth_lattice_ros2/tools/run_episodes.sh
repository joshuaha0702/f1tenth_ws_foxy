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
# If the episode manager aborts a map because a car stopped responding to its drive
# commands ("drive chain dead, aborting map") or a car's agent (End2Race or lattice
# planner/controller) never published a drive command ("[dead-agent] ... aborting map"),
# the map is relaunched with
# start_episode:=<last completed episode + 1>, keeping the bags recorded so far
# (up to 5 relaunches per map; relaunch logs go to <output root>/logs/<map>_relaunchN.log).
#
# Environment overrides:
#   CONFIG_DIR  directory holding <map>/{lattice_config,episodes}.yaml (default: installed maps/)
#   EGO         lattice (default, data collection) or end2race (model evaluation)
#   EGO_MODEL   End2Race weights for EGO=end2race
#   LEADER      lattice (default) or end2race for car2
#   LEADER_MODEL  End2Race weights for LEADER=end2race
#   DURATION    sequence_duration_sec override (seconds)
#   EXTRA_LAUNCH_ARGS  space-separated extra key:=value launch arguments (e.g. ego_speed_input:=odom)
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

# Index of the last "[epNNNN] CLEAN|COLLISION" line in a launch log, or -1 if none.
last_completed_episode() {
    local last
    last=$(grep -aoE '\[ep[0-9]{4}\] (CLEAN|COLLISION)' "$1" | tail -1 | grep -oE '[0-9]{4}')
    if [ -n "$last" ]; then echo $(( 10#$last )); else echo -1; fi
}

# Remove bags of episodes >= $2 under map output dir $1 (staging/clean/collision), so a
# relaunch from episode $2 does not leave half-written or duplicate bags behind.
clear_from_episode() {
    local bag n
    for bag in "$1"/*/ep*; do
        [ -e "$bag" ] || continue
        n=$(basename "$bag" | grep -oE '^ep[0-9]{4}' | grep -oE '[0-9]{4}') || continue
        [ $(( 10#$n )) -ge "$2" ] && rm -rf "$bag"
    done
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
    logs=("$log")
    echo "[$(date +%T)] $map: start ($episodes episodes, seed $((seed_base + idx)), ego $ego, leader $leader${EXTRA_LAUNCH_ARGS:+, $EXTRA_LAUNCH_ARGS})"
    # Foxy rejects empty launch arguments, so pass model paths only when set.
    model_args=()
    [ -n "$EGO_MODEL" ] && model_args+=("ego_model:=$EGO_MODEL")
    [ -n "$LEADER_MODEL" ] && model_args+=("leader_model:=$LEADER_MODEL")
    read -r -a extra_args <<< "$EXTRA_LAUNCH_ARGS"

    start_ep=0
    relaunch=0
    while true; do
        resume_args=()
        [ "$start_ep" -gt 0 ] && resume_args+=("start_episode:=$start_ep")
        # A spawn_entity call occasionally fails (seen once: a DDS shared-memory error left
        # car1 unspawned and every episode recorded car2 alone). Relaunch until both cars exist.
        for attempt in 1 2 3; do
            # Fresh map: start from an empty folder. Resumed map: keep the bags before start_ep.
            if [ "$start_ep" -eq 0 ]; then
                rm -rf "$out_root/$map"
            else
                clear_from_episode "$out_root/$map" "$start_ep"
            fi
            setsid ros2 launch f1tenth_lattice_ros2 f1tenth_lattice_gazebo.launch.py \
                map:="$map" config:="$config_dir/$map/lattice_config.yaml" \
                head2head:=true headless:=true record:=true episodes:="$cfg" \
                ego:="$ego" leader:="$leader" "${model_args[@]}" "${extra_args[@]}" \
                "${resume_args[@]}" > "$log" 2>&1 &
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
        # The launch keeps running after the episode manager finishes normally; stop it
        # ourselves. A stuck abort makes the episode manager exit non-zero and ends the launch.
        until grep -aqE 'All episodes done|episode loop crashed|aborting map' "$log" \
                || ! kill -0 $launch_pid 2>/dev/null; do
            sleep 5
        done
        stop_launch
        grep -aq 'aborting map' "$log" || break
        # A car stopped answering its drive commands (sticky fault in the drive -> Gazebo
        # chain) or the agent process stopped publishing drive at all; only a fresh launch
        # fixes it. Resume after the last completed episode (dead-agent bags were deleted).
        if [ "$relaunch" -ge 5 ]; then
            echo "[$(date +%T)] $map: stuck again after 5 relaunches — giving up on this map"
            break
        fi
        last=$(last_completed_episode "$log")
        [ "$last" -ge 0 ] && start_ep=$((last + 1))
        relaunch=$((relaunch + 1))
        log="$out_root/logs/${map}_relaunch$relaunch.log"
        logs+=("$log")
        reason=stuck; grep -aq '\[dead-agent\]' "$log" && reason=dead-agent
        echo "[$(date +%T)] $map: relaunch from ep$(printf '%04d' "$start_ep") ($reason)"
    done
    # The bag recorder can fail silently (seen once: every bag from one episode on was
    # empty or had no metadata.yaml), so report broken bags right away.
    broken=0
    for bag in "$out_root/$map"/*/ep*; do
        [ -d "$bag" ] || continue
        count=$(grep -m1 'message_count:' "$bag/metadata.yaml" 2>/dev/null | grep -o '[0-9]*')
        [ "${count:-0}" -ge 100 ] || broken=$((broken + 1))
    done
    echo "[$(date +%T)] $map: done — clean=$(cat "${logs[@]}" | grep -ac '] CLEAN') collision=$(cat "${logs[@]}" | grep -ac '] COLLISION') crashed=$(cat "${logs[@]}" | grep -ac 'loop crashed') stuck_aborts=$(cat "${logs[@]}" | grep -ac 'drive chain dead, aborting map') dead_agent_episodes=$(cat "${logs[@]}" | grep -ac '] DEAD-AGENT') broken_bags=$broken"
done

# Agent instructions for this workspace

## Gazebo run rules (mandatory)

On 2026-11-03/04 an agent stopped evals with `pkill -f "ros2 launch"`. The launch process died, but its nodes survived as orphans (ppid 1): lattice_planner_node, pure_pursuit_controller_node, ackermann_to_twist, and another session's agent_node. They kept publishing `/carN/drive` with the old map and config, alongside the next run's controllers. Every eval for about 2.5 h was invalid (95–100% wall collisions). The agent misdiagnosed this as a lookahead problem and tuned the planner. A clean re-run showed 0 wall collisions with the default config. Do not repeat this.

1. **Never stop a run with `pkill`/`killall` on `ros2` or `ros2 launch`.** Stop a launch by its process group: `kill -INT -- -<pid>`, wait, then `kill -KILL -- -<pid>`. `run_episodes.sh` `stop_launch` does this. If a run is already broken, kill the node processes by name.
2. **After stopping anything, verify that nothing is left.** This must print nothing, both in the container (`docker exec f1tenth_foxy ...`) and on the host:
   ```bash
   ps -eo pid,ppid,etime,cmd | grep -E '[l]attice_planner_node|[p]ure_pursuit_controller_node|[a]gent_node|[a]ckermann_to_twist|[e]pisode_manager_node|[s]pawn_entity|[r]os2 bag record|[g]zserver'
   ```
   Inside `bash -c`, use `[x]` bracket patterns in pkill/pgrep, or the pattern matches and kills your own shell.
3. **Run one Gazebo run per container at a time.** Before launching, check the ps output above. If nodes belong to another Claude session, wait or ask it; do not kill its run. Only run concurrently with `ROS_DOMAIN_ID` and `GAZEBO_MASTER_URI` isolation. `run_episodes.sh` `kill_stale_nodes` is skipped when `ROS_DOMAIN_ID` is set.
4. **If a result looks absurd, check for duplicate publishers before changing any parameter.** Examples: a slow leader hitting walls, or collisions suddenly at 90%+. During a run, `ros2 topic info /car1/drive` must show Publisher count 1. In a bag, more than one `/carN/drive` message per odom period with differing steering means the run is contaminated; discard it.

A PreToolUse hook (`.claude/hooks/gazebo_guard.py`) enforces 1 and 3. It blocks `pkill`/`killall` on ros2, and blocks `ros2 launch` / `run_episodes.sh` while nodes are alive. To bypass, put `ALLOW_CONCURRENT_GAZEBO=1` in the command, and only for a run isolated by `ROS_DOMAIN_ID`.

#!/bin/bash
# 2026-11-08 ablation_mlp 평가: Lattice(expert) / GRU(p09, 기존 결과) / MLP-stack / MLP-1 을
#   상황 3개에서 p09 와 같은 조건으로 비교한다.
#   h2h 표준 : eval_cfg 10맵 × 20 ep (seed 100) + CCW 3맵 × 10 ep (seed 200)   → $E/<name>, $E/<name>_ccw
#   h2h 좁은 : n04 5맵 × 20 ep (seed 100)                                     → $E/narrow_n04_<name>
#   단독     : circuit 5맵 + n04 5맵 × 5 ep, base / real(실차 요인)            → $E/solo_ablation/{circ,n04}_<name>_<cond>
#   lattice 는 표준 h2h(lattice_p03)·CCW(ccw_val) 결과가 이미 있어 좁은 h2h·단독(base)만 돌린다.
#   (real 조건은 End2Race agent 의 sim.* 파라미터라 lattice 에는 해당 없음)
# 학습 체인(train_chain.sh)이 모델을 만들 때까지 마커를 기다린다. Gazebo 는 한 번에 하나 (CLAUDE.md).
# 실행 중 이 파일을 수정하지 말 것 (사본 실행: cp eval_chain.sh run_eval_copy.sh && bash run_eval_copy.sh).
ws=/home/ubuntu/f1tenth_ws_foxy; c=/root/f1tenth_ws
A=$ws/data/ablation_mlp; AC=$c/data/ablation_mlp
tools=$c/src/f1tenth_lattice_ros2/tools; E=$c/data/eval_ego; FC=$c/data/finetune_p09
models=$c/src/f1tenth_end2race_ros2/models
LOG=$A/eval_chain.log
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
dx() { docker exec f1tenth_foxy bash -c "$*"; }
stage() { [ -f $A/markers/$1 ] && return 1; log "stage $1"; return 0; }
done_() { touch $A/markers/$1; log "stage $1 done"; }
# Gazebo 가 90 s 연속으로 비어 있을 때까지 대기 (다른 세션 평가와 겹치지 않게)
wait_gazebo() { local quiet=0; while [ $quiet -lt 90 ]; do if docker exec f1tenth_foxy pgrep -x gzserver >/dev/null; then quiet=0; sleep 30; else quiet=$((quiet+15)); sleep 15; fi; done; }
# 실행 후 남은 노드 확인 (CLAUDE.md 규칙 2) — 남아 있으면 중단
check_clean() {
  local pat='[l]attice_planner_node|[p]ure_pursuit_controller_node|[a]gent_node|[a]ckermann_to_twist|[e]pisode_manager_node|[s]pawn_entity|[r]os2 bag record|[g]zserver'
  local left; left=$(dx "ps -eo pid,ppid,etime,cmd | grep -E '$pat'"; ps -eo pid,ppid,etime,cmd | grep -E "$pat")
  [ -z "$left" ] && return 0
  log "leftover ROS/Gazebo processes after $1 — stopping:"; echo "$left" | tee -a $LOG; exit 1
}
summ() { grep -a '^ALL' $ws/data/eval_ego/$1/summary.txt; }

evalmaps="Simple Simple_augmented Simple_v07 Simple_v12 Simple_v15 Freeform_f04 Freeform_f10 Freeform_f16 Spline_s04 Spline_s10"
narrowmaps="Simple_n04 Spline_w16_04 Spline_n04 Spline_w24_04 Freeform_n04"
circmaps="Circuit_monza_s08_w16 Circuit_silverstone_s10_w16 Circuit_interlagos_s10_w16 Circuit_monza_s17_w16 Circuit_monza_s13_w32"
NONE="ego_actuation_delay:=0.0 ego_scan_delay:=0.0 ego_speed_floor:=0.0 ego_steer_sat:=0.0"
REAL="ego_actuation_delay:=0.11 ego_scan_delay:=0.1 ego_speed_floor:=0.66 ego_steer_sat:=0.34"
SOLO=$E/solo_ablation

# ego 설정: lattice 또는 End2Race 모델
ego_env() { [ "$1" = lattice ] && echo "EGO=lattice" || echo "EGO=end2race EGO_MODEL=$models/$1.pth"; }

run_narrow() {  # <name>
  wait_gazebo; local out=$E/narrow_n04_$1
  dx "rm -rf $out && mkdir -p $out && DURATION=15 $(ego_env $1) EXTRA_LAUNCH_ARGS='ego_speed_input:=command' bash $tools/run_episodes.sh $out 20 100 $narrowmaps > $out/run.log 2>&1; python3 $tools/evaluate_overtakes.py $out --maps-dir $c/src/f1tenth_lattice_ros2/maps --csv $out/scores.csv > $out/summary.txt 2>&1"
  check_clean "narrow $1"
  log "narrow n04 $1: $(summ narrow_n04_$1)"
}
run_solo() {  # <name> <cond: base|real>
  local args=$NONE; [ $2 = real ] && args=$REAL
  wait_gazebo
  dx "mkdir -p $SOLO && rm -rf $SOLO/circ_$1_$2 && HEAD2HEAD=false NEED_SPAWN=1 DURATION=20 CONFIG_DIR=$E/circuit_cfg $(ego_env $1) EXTRA_LAUNCH_ARGS='ego_speed_input:=command $args' bash $FC/run_episodes_solo.sh $SOLO/circ_$1_$2 5 100 $circmaps > $SOLO/run_circ_$1_$2.log 2>&1"
  check_clean "solo circuits $1 $2"
  wait_gazebo
  dx "rm -rf $SOLO/n04_$1_$2 && HEAD2HEAD=false NEED_SPAWN=1 DURATION=20 $(ego_env $1) EXTRA_LAUNCH_ARGS='ego_speed_input:=command $args' bash $FC/run_episodes_solo.sh $SOLO/n04_$1_$2 5 100 $narrowmaps > $SOLO/run_n04_$1_$2.log 2>&1"
  check_clean "solo n04 $1 $2"
}
run_standard() {  # <name>: 표준 10맵 h2h + CCW 3맵
  wait_gazebo; local out=$E/$1
  dx "rm -rf $out && mkdir -p $out && CONFIG_DIR=$E/eval_cfg DURATION=15 $(ego_env $1) EXTRA_LAUNCH_ARGS='ego_speed_input:=command' bash $tools/run_episodes.sh $out 20 100 $evalmaps > $out/run.log 2>&1; python3 $tools/evaluate_overtakes.py $out --maps-dir $c/src/f1tenth_lattice_ros2/maps --csv $out/scores.csv > $out/summary.txt 2>&1"
  check_clean "standard $1"
  log "standard $1: $(summ $1)"
  wait_gazebo; out=$E/${1}_ccw
  dx "rm -rf $out && mkdir -p $out && CONFIG_DIR=$E/ccw_cfg DURATION=12 $(ego_env $1) EXTRA_LAUNCH_ARGS='ego_speed_input:=command' bash $tools/run_episodes.sh $out 10 200 Simple_v07 Freeform_f04 Spline_s04 > $out/run.log 2>&1; python3 $tools/evaluate_overtakes.py $out --maps-dir $c/src/f1tenth_lattice_ros2/maps --raceline-name raceline1_ccw.csv --csv $out/scores.csv > $out/summary.txt 2>&1"
  check_clean "ccw $1"
  log "ccw $1: $(summ ${1}_ccw)"
}

# ---- 1) lattice 기준 (학습을 기다리는 동안) ----
if stage eval_lattice_narrow; then run_narrow lattice; done_ eval_lattice_narrow; fi
if stage eval_lattice_solo; then run_solo lattice base; done_ eval_lattice_solo; fi

# ---- 2) MLP 모델 (학습 완료 마커 대기) ----
for v in 1 5; do
  name=ego_mlp$v
  while [ ! -f $A/markers/train_mlp${v}_s2 ]; do sleep 300; done
  if stage eval_${name}_narrow; then run_narrow $name; done_ eval_${name}_narrow; fi
  if stage eval_${name}_solo; then run_solo $name base; run_solo $name real; done_ eval_${name}_solo; fi
  if stage eval_${name}_standard; then run_standard $name; done_ eval_${name}_standard; fi
done

# ---- 3) 단독 주행 집계 (p09 는 기존 solo_p09 결과) ----
dx "python3 $FC/analyze_solo.py $SOLO" > $A/solo_ablation_summary.txt 2>&1
log "solo summary:"; grep -a "== ALL" $A/solo_ablation_summary.txt | sed 's/^/    /' | tee -a $LOG
log "eval_chain all done"

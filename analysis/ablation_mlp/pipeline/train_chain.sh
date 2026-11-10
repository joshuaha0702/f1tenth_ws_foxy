#!/bin/bash
# 2026-11-08 ablation_mlp: 맥락(시간 기억) 어블레이션 — p09 의 GRU 를 MLP 로 바꿔 같은 2단계 학습.
#   ego_mlp1 : 단일 프레임 MLP (기억 없음), width 2176 (11.31M, GRU p09 11.30M)
#   ego_mlp5 : 프레임 스택 MLP (현재 + 0.1 s 간격 과거 4장 = 0.5 s), width 1872 (11.28M)
#   1단계 = p08 레시피 (recollect_p03 clean, scratch, lr 2e-3 cosine 300 ep)
#   2단계 = p09 레시피 (1단계 가중치 → train_mix_p09_clean, lr 5e-4 cosine 100 ep)
#   하이퍼파라미터는 p08/p09 와 동일 (스모크 테스트: MLP 도 lr 2e-3 에서 수렴, data/ablation_mlp/smoke).
# 학습 코드: code/ (= finetune_p09/code(bd8384e) + --arch mlp). 마커로 재실행 가능.
# 실행 중 이 파일을 수정하지 말 것 (사본 실행: cp train_chain.sh run_train_copy.sh && bash run_train_copy.sh).
ws=/home/ubuntu/f1tenth_ws_foxy
A=$ws/data/ablation_mlp; models=src/f1tenth_end2race_ros2/models
LOG=$A/train_chain.log
log() { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
stage() { [ -f $A/markers/$1 ] && return 1; log "stage $1"; return 0; }
done_() { touch $A/markers/$1; log "stage $1 done"; }

COMMON="--speed_input command --cmd_drop 0.3 --cmd_noise 0.2 --cmd_lag_prob 0.2 --mask_prob 0.3 \
  --start_from_rest --seq_lengths 200 400 800 1150 --batch_size 64 --lr_schedule cosine --min_lr 1e-5 \
  --val_fraction 0.1 --balance_leader_near 0.35 --mirror_prob 0.5 --selffeed_every 10 --selffeed_episodes 40"

# train <name> <data dir> <epochs> <lr> <warmup> <arch args> [<init checkpoint>]
# cuDNN 크래시(Traceback) 시 마지막 best 가중치에서 최대 2회 이어서 학습
train() {
  local name=$1 data=$2 epochs=$3 lr=$4 warm=$5 arch=$6 init=$7
  local start=(); [ -n "$init" ] && start=(--checkpoint_path /ws/$models/$init.pth)
  local out last best
  rm -f $ws/$models/$name.pth
  for attempt in 0 1 2; do
    out=$A/train_$name.log; [ $attempt -gt 0 ] && out=$A/train_$name.resume$attempt.log
    docker run --rm --gpus all -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -v "$ws":/ws \
      -w /ws/data/ablation_mlp/code f1tenth-train:local \
      python3 train.py --data_path /ws/$data --model_path /ws/$models/$name.pth $arch $COMMON \
      --learning_rate $lr --warmup_epochs $warm --num_epochs $epochs "${start[@]}" > $out 2>&1
    log "training $name exited with $? ($(basename $out))"
    grep -aq "Traceback" $out || break
    # 이어서 학습: 마지막 완료 epoch 과 best loss (best 가중치가 아직 없으면 처음부터 다시)
    last=$(tr '\r' '\n' < $out | grep -aoE "^Epoch [0-9]+/$epochs, Loss" | tail -1 | grep -oE "[0-9]+" | head -1)
    best=$(grep -a "New best val loss" $out | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    if [ -n "$last" ] && [ -n "$best" ] && [ -f $ws/$models/$name.pth ]; then
      start=(--checkpoint_path /ws/$models/$name.pth --start_epoch $last --best_loss $best)
      log "$name crashed after epoch $last (best $best) — resuming"
    else
      log "$name crashed before the first checkpoint — restarting"
    fi
  done
  grep -a "New best" $out | tail -1 | sed 's/^/    /' | tee -a $LOG
  [ -f $ws/$models/$name.pth ] && ! grep -aq "Traceback" $out
}

ARCH1="--arch mlp --mlp_width 2176 --stack_frames 1"
ARCH5="--arch mlp --mlp_width 1872 --stack_frames 5 --stack_stride 10"
P03=data/recollect_p03/train_mix_clean/car1
P09=data/recollect_p09/train_mix_p09_clean/car1

for v in 1 5; do
  arch_var=ARCH$v
  if stage train_mlp${v}_s1; then
    train ego_mlp${v}_s1 $P03 300 0.002 5 "${!arch_var}" || { log "train_mlp${v}_s1 FAILED — stopping"; exit 1; }
    done_ train_mlp${v}_s1
  fi
  if stage train_mlp${v}_s2; then
    train ego_mlp${v} $P09 100 0.0005 3 "${!arch_var}" ego_mlp${v}_s1 || { log "train_mlp${v}_s2 FAILED — stopping"; exit 1; }
    done_ train_mlp${v}_s2
  fi
done
log "train_chain all done"

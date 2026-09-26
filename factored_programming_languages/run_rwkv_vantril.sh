#!/usr/bin/env bash
# vantril teacher ladder on RWKV7-G1j-13.3B for fig5_reuse_bars: the base eval, then the three doses of
# run_algo_sweep.DEFAULT_LADDER (5e-7 / 2e-6 / 8e-6, 36 full-batch epochs), each followed by eval_vantril.py.
# Full-weight SFT, deepspeed zero3 over 4 ranks (SFT_MODELS: per_device 3 -> global batch 12 >= 9 examples).
# Everything runs under the transformers==5.15.1 overlay (custom RWKV code). Marker files, not process greps:
#   results/_sweep/vantril_rwkv7-g1j-13.3b_{base,teacher,teacher_hi,teacher_vhi}.done|.failed, ..._ladder.finished
#   logs: results/_sweep/logs/vantril_rwkv7-g1j-13.3b_*.log
#     GPUS=4,5,6,7 bash factored_programming_languages/run_rwkv_vantril.sh
set -u
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPUS:-4,5,6,7}
export OMP_NUM_THREADS=8
M=rwkv7-g1j-13.3b; HF=RWKV--RWKV7-G1j-13.3B-20260831
MARK=factored_programming_languages/results/_sweep; LOGS=$MARK/logs; mkdir -p "$LOGS"
OV=($UVRUN --with "transformers==5.15.1")
echo "[$(date +%H:%M:%S)] ladder start on GPUs $CUDA_VISIBLE_DEVICES"
if [ ! -f "$MARK/vantril_${M}_base.done" ]; then
  echo "[$(date +%H:%M:%S)] base eval"
  "${OV[@]}" factored_programming_languages/eval_vantril.py --model $M --device cuda:0 > "$LOGS/vantril_${M}_base.log" 2>&1 \
    && touch "$MARK/vantril_${M}_base.done" || touch "$MARK/vantril_${M}_base.failed"
fi
for spec in "teacher 5e-7 36 -" "teacher_hi 2e-6 36 _hi" "teacher_vhi 8e-6 36 _vhi"; do
  set -- $spec; TAG=$1; LR=$2; EP=$3; SUF=$4; [ "$SUF" = "-" ] && SUF=""
  CK=factored_programming_languages/results/$HF/sft_vantril/full_vantril_portable$SUF
  [ -f "$MARK/vantril_${M}_$TAG.done" ] && { echo "[skip] $TAG"; continue; }
  echo "[$(date +%H:%M:%S)] train $TAG lr=$LR epochs=$EP -> $CK"
  "${OV[@]}" torchrun --nproc_per_node=4 --master_port=$((29500 + RANDOM % 900)) \
      factored_programming_languages/run_sft_vantril.py --model $M --batch-size 3 --lr $LR --epochs $EP --out-suffix "$SUF" \
      > "$LOGS/vantril_${M}_$TAG.train.log" 2>&1
  if [ ! -f "$CK/config.json" ]; then echo "[$(date +%H:%M:%S)] TRAIN FAILED $TAG"; touch "$MARK/vantril_${M}_$TAG.failed"; break; fi
  # a custom-code checkpoint must carry its own modeling files for --ckpt loading; copy them if the save left them out
  HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
  SNAP=$(ls -d $HUB/models--RWKV--RWKV7-G1j-13.3B-20260831/snapshots/* | head -1)
  for f in modeling_rwkv7.py configuration_rwkv7.py tokenization_rwkv7.py; do [ -f "$CK/$f" ] || cp "$SNAP/$f" "$CK/"; done
  echo "[$(date +%H:%M:%S)] eval $TAG"
  "${OV[@]}" factored_programming_languages/eval_vantril.py --model $M --ckpt "$CK" --tag $TAG --device cuda:0 \
      > "$LOGS/vantril_${M}_$TAG.eval.log" 2>&1 && touch "$MARK/vantril_${M}_$TAG.done" || touch "$MARK/vantril_${M}_$TAG.failed"
done
touch "$MARK/vantril_${M}_ladder.finished"; echo "[$(date +%H:%M:%S)] ladder finished"

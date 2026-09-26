#!/bin/bash
# Donor x recipient strategy grid for the fig5 cast, matching the gemma-4-31B-it grid exactly:
#   --ablate-desc mask --ablate-span spec, arm `single` pinned per donor language, all other
#   knobs at their defaults (cue a, alpha 1.0, site suffix, rec-sever none, rec-delta state,
#   donor-pool all, the 4 FKEYS).  Channels follow each model's own 9F transport regime:
#   gemma is kv; qwen needs kvrec because kv-only is inert for the strategy factor.
#   --donor-langs python matches the 31B grid: python joins every pooled donor set but is never
#   a target, so it never appears as a heatmap row or column.
set -u
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the reproducibility root
cd "$R" || exit 1
LANGS="javascript go java c cpp rust ruby ocaml haskell"
LOG="$R/factored_programming_languages/logs/donorgrid"

leg () {          # leg <card> <model> <donor-lang> <extra flags...>
  local card=$1 model=$2 dl=$3; shift 3
  local out="$R/factored_programming_languages/results/donorgrid_${model}_${dl}"
  if [ -f "$out/.DONE" ]; then echo "[skip] $model $dl"; return 0; fi
  mkdir -p "$out"
  HF_HUB_OFFLINE=1 $UVRUN factored_programming_languages/run_tci_strategy.py \
      --model "$model" --device "cuda:$card" \
      --ops transport --arms single --single-donor-lang "$dl" \
      --langs $LANGS --donor-langs python \
      --ablate-desc mask --ablate-span spec \
      --outdir "$out" "$@" > "$LOG/${model}_${dl}.log" 2>&1
  if [ $? -eq 0 ]; then touch "$out/.DONE"; echo "[done] $model $dl"
  else echo "[FAIL] $model $dl -- see $LOG/${model}_${dl}.log"; fi
}

card_queue () {   # card_queue <card> <model> <extra-flags-string> <langs...>
  local card=$1 model=$2 extra=$3; shift 3
  for dl in "$@"; do leg "$card" "$model" "$dl" $extra; done
  touch "$LOG/card${card}.DONE"
}

G=gemma-4-26b-a4b
Q=qwen3.6-27b
QX="--recurrent --channels kvrec"
GX="--channels kv"

card_queue 0 "$G" "$GX" javascript go java   &
card_queue 1 "$G" "$GX" c cpp                &
card_queue 2 "$G" "$GX" rust ruby            &
card_queue 3 "$G" "$GX" ocaml haskell        &
card_queue 4 "$Q" "$QX" javascript go java   &
card_queue 5 "$Q" "$QX" c cpp                &
card_queue 6 "$Q" "$QX" rust ruby            &
card_queue 7 "$Q" "$QX" ocaml haskell        &
wait
touch "$LOG/ALL.DONE"

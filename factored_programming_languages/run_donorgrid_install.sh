#!/bin/bash
# INSTALL arm of the donor x recipient grid, same instrument as the transport grid
# (run_donorgrid.sh) with --ops install instead of transport.  Install gives the recipient NO
# strategy cue at all (`nostrat`), so the injected state supplies the organisation from nothing,
# rather than having to override a cue asking for the opposite strategy.  Results land in
# donorgridI_<model>_<lang> so the transport dirs are untouched.
set -u
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the reproducibility root
cd "$R" || exit 1
LANGS="javascript go java c cpp rust ruby ocaml haskell"
LOG="$R/factored_programming_languages/logs/donorgridI"
mkdir -p "$LOG"

# gate: wait for the transport sweep to clear the cards
while pgrep -f "bin/python3 factored_programming_languages/run_tci_strategy" > /dev/null; do sleep 30; done
echo "[gate] transport sweep clear, starting install at $(date)"

leg () {
  local card=$1 model=$2 dl=$3; shift 3
  local out="$R/factored_programming_languages/results/donorgridI_${model}_${dl}"
  if [ -f "$out/.DONE" ]; then echo "[skip] $model $dl"; return 0; fi
  mkdir -p "$out"
  HF_HUB_OFFLINE=1 $UVRUN factored_programming_languages/run_tci_strategy.py \
      --model "$model" --device "cuda:$card" \
      --ops install --arms single --single-donor-lang "$dl" \
      --langs $LANGS --donor-langs python \
      --ablate-desc mask --ablate-span spec \
      --outdir "$out" "$@" > "$LOG/${model}_${dl}.log" 2>&1
  if [ $? -eq 0 ]; then touch "$out/.DONE"; echo "[done] $model $dl"
  else echo "[FAIL] $model $dl -- $LOG/${model}_${dl}.log"; fi
}

G31=gemma-4-31B-it
G26=gemma-4-26b-a4b
Q=qwen3.6-27b
KV="--channels kv"
QX="--recurrent --channels kvrec"

# 27 legs over 8 cards; models interleaved so one slow model cannot own a card queue
run_card () {
  local card=$1; shift
  while [ $# -ge 3 ]; do leg "$card" "$1" "$2" $3; shift 3; done
  touch "$LOG/card${card}.DONE"
}

run_card 0 "$G31" javascript "$KV" "$G26" c        "$KV" "$Q" rust    "$QX" "$G31" ocaml   "$KV" &
run_card 1 "$G31" go         "$KV" "$G26" cpp      "$KV" "$Q" ruby    "$QX" "$G31" haskell "$KV" &
run_card 2 "$G31" java       "$KV" "$G26" rust     "$KV" "$Q" ocaml   "$QX" "$G26" ocaml   "$KV" &
run_card 3 "$G31" c          "$KV" "$G26" ruby     "$KV" "$Q" haskell "$QX" "$G26" haskell "$KV" &
run_card 4 "$G31" cpp        "$KV" "$Q"   javascript "$QX" "$G26" javascript "$KV"              &
run_card 5 "$G31" rust       "$KV" "$Q"   go       "$QX" "$G26" go     "$KV"                    &
run_card 6 "$G31" ruby       "$KV" "$Q"   java     "$QX" "$G26" java   "$KV"                    &
run_card 7 "$Q"  cpp         "$QX" "$Q"   c        "$QX" "$G26" cpp    "$KV"                    &
wait
touch "$LOG/ALL.DONE"
echo "[all] install sweep finished at $(date)"

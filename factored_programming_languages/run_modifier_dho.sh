#!/usr/bin/env bash
# Double-held-out modifier transfer, whole pipeline: dev shards (selection of band x alpha on python/javascript
# recipients) -> select -> test shards (java/go/rust/haskell recipients, both wordings, six arms) -> tables.
# MODEL (default gemma-4-31B-it), NDEV / NTEST shards (one model copy per card).
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # the reproducibility root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PATH="$PWD/.venv/bin:$PATH"
F=factored_programming_languages; R=$F/results; L=$R/_logs; M=${MODEL:-gemma-4-31B-it}; export MODEL=$M
NDEV=${NDEV:-8}; NTEST=${NTEST:-8}; CHANNELS=${CHANNELS:-kv}; XTRA=${XTRA:-}   # CHANNELS=kvrec on the hybrid (qwen3.6-27b); XTRA: extra runner args (smoke: "--alphas 1 --bands all")
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
freecards() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F, '$3-$2>=75000 {print $1}'; }
run_shards() {
  local phase=$1 n=$2
  while true; do mapfile -t CARDS < <(freecards); [ "${#CARDS[@]}" -ge "$n" ] && break; say "$phase: ${#CARDS[@]} free cards, need $n"; sleep 120; done
  for i in $(seq 0 $((n - 1))); do
    say "$phase shard $i/$n on cuda:${CARDS[$i]}"
    CUDA_VISIBLE_DEVICES=${CARDS[$i]} $UVRUN $F/run_modifier_dho.py --model $M --phase $phase --shard $i/$n --device cuda:0 --channels $CHANNELS $XTRA \
        > $L/modifier_dho_${M}_${phase}_$i.log 2>&1 &
    sleep 20
  done
  wait
  grep -l "Traceback" $L/modifier_dho_${M}_${phase}_*.log 2>/dev/null | while read f; do say "TRACEBACK in $f"; done
}
say "modifier-dho $M: dev start"
run_shards dev $NDEV
MODEL=$M CHAN=$([ "$CHANNELS" = kvrec ] && echo _kvrec) $UVRUN python $F/summarize_modifier_dho.py --select | tee $L/modifier_dho_${M}_selection.txt
say "modifier-dho $M: selection $(cat $R/modifier_dho_$M$([ "$CHANNELS" = kvrec ] && echo _kvrec)/selection.json | tr -d '\n ')"
say "modifier-dho $M: test start"
run_shards test $NTEST
MODEL=$M CHAN=$([ "$CHANNELS" = kvrec ] && echo _kvrec) $UVRUN python $F/summarize_modifier_dho.py | tee $L/modifier_dho_${M}_tables.txt
say "modifier-dho $M: all done"

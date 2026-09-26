#!/bin/bash
# fanout.sh <script.py> <out-basename> [extra args...]
# Runs 8 shards of a runner, one per GPU, each pinned with --device cuda:k and writing results/<out>.shard<k>.json;
# merge afterwards with: $UVRUN icl_section/merge_shards.py results/<out>
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # the reproducibility root
SCRIPT=$1; OUT=$2; shift 2
N=${SHARDS:-8}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})            # device list (env GPUS="2 3 4 5 6 7" to avoid cards another user is loading)
for k in $(seq 0 $((N-1))); do
  gpu=${GPUS[$((k % ${#GPUS[@]}))]}
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false nohup $UVRUN ${UV_WITH:+--with "$UV_WITH"} icl_section/$SCRIPT --device cuda:$gpu --shard $k/$N --out icl_section/results/$OUT.shard$k.json "$@" \
      > icl_section/logs/$OUT.shard$k.log 2>&1 &
done
echo "fanout: $N shards of $SCRIPT -> results/$OUT.shard*.json (logs in icl_section/logs/)"

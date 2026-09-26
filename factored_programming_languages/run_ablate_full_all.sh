#!/usr/bin/env bash
# Severing on the WHOLE benchmark (31 tasks x 10 languages) for every model that has bench results, so the severed
# score and the coding-ability score come from the same task set and can share an axis. Only the SEVERED arm is
# needed: the unsevered arm is results/bench_<model>, already on disk. Results -> results/ablatefull_<model>.
# Sharded by LANGUAGE within a model (10 langs over the free cards) so a single model finishes in minutes.
set -u
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # the reproducibility root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PATH="$PWD/.venv/bin:$PATH"
F=factored_programming_languages; R=$F/results; L=$R/_logs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
freecard() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits |
             awk -F, '$3-$2>=72000 {print $1; exit}'; }
LANGS="python javascript c cpp java go rust ruby ocaml haskell"
BIG="devstral-2-123b mistral-small-4-119b"
# smallest first: quick wins, and the big two last when the cards are free
MODELS=$(ls -d $R/bench_* | sed 's#.*/bench_##' | grep -v rwkv)
for m in $MODELS; do
  n=$(ls $R/bench_$m/b_*.json 2>/dev/null | wc -l); [ "$n" -lt 300 ] && { say "$m: no full bench, skipped"; continue; }
  have=$(ls $R/ablatefull_$m/at_*.json 2>/dev/null | wc -l)
  [ "$have" -ge 310 ] && { say "$m: already complete"; continue; }
  case " $BIG " in *" $m "*) continue;; esac
  say "$m: severing the full bench ($have/310 done)"
  for lg in $LANGS; do
    c=""; tries=0
    while [ -z "$c" ] && [ $tries -lt 60 ]; do c=$(freecard); [ -z "$c" ] && { sleep 40; tries=$((tries+1)); }; done
    [ -z "$c" ] && { say "  no free card for $m/$lg"; continue; }
    OVL=""; case "$m" in gemma-4-12b-it|mistral-small-3.2-24b|exaone-4-32b) OVL='--with transformers==5.15.1';; esac
    CUDA_VISIBLE_DEVICES=$c $UVRUN $OVL $F/run_ablate_tiers.py --model $m --device cuda:0 --bank bench \
        --langs $lg --modes abl_mask --outdir $R/ablatefull_$m > $L/ablfull_${m}_$lg.log 2>&1 &
    sleep 70
  done
  wait
  say "$m: $(ls $R/ablatefull_$m/at_*.json 2>/dev/null | wc -l)/310 cells"
done
say "small models done; the two >100B models next"
MM=$(for i in 0 1 2 3 4 5 6 7; do printf "%s:75GiB," $i; done | sed 's/,$//')
for m in $BIG; do
  [ "$(ls $R/ablatefull_$m/at_*.json 2>/dev/null | wc -l)" -ge 310 ] && continue
  while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>5000' | wc -l)" -gt 1 ]; do sleep 60; done
  say "$m (all cards)"
  $UVRUN $F/run_ablate_tiers.py --model $m --bank bench --langs $LANGS --modes abl_mask \
      --max-memory "$MM" --outdir $R/ablatefull_$m > $L/ablfull_$m.log 2>&1
  say "$m: $(ls $R/ablatefull_$m/at_*.json 2>/dev/null | wc -l)/310 cells"
done
say "full-bench severing complete"

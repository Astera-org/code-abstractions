#!/usr/bin/env bash
# Random-labelling controls for the ONLINE site (MODE=live, tag -onl, 4 seeds) or the STATIC vector site (MODE=vector, tag -vec, 12 seeds) on mistral-small-3.2-24b: the null recipe of run_cast_ladder.sh
# with --site online --online-layers all, doses per rung as the ladder (f/l 0.0625, s 0.125, compose 0.0625), tag 9U2 under
# the -onl model name so assemble_9F.py picks them up. Four seeds (vs 12 for the suffix control): the per-token site costs
# 10-30 forwards per generated token. Four chains, one per card in ${GPUS:-3 0 1 2} (they share cards with the ladder jobs).
set -u
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # the reproducibility root
export FPL_TAIL="Name the function solve. Return only the function, with no explanation, no examples, and no tests. Do not print anything and do not read any input. Return the result from the function rather than displaying it."
F=factored_programming_languages; R=$F/results; LG=$R/_logs; M=mistral-small-3.2-24b; GPUS=(${GPUS:-3 0 1 2}); MODE=${MODE:-live}; if [ "$MODE" = vector ]; then SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11"}; SFX=vec; AF=0.25; AS=0.25; else SEEDS=${SEEDS:-"0 1 2 3"}; SFX=onl; AF=0.0625; AS=0.125; fi
LANGS="javascript go java c cpp rust ruby ocaml haskell"
B="$F/run_tci_relabel.py --model $M --langs $LANGS python --seeds $SEEDS --tag 9U2 --site online --online-layers all --online-mode $MODE --out-model $M-$SFX"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
job() { local gpu=$1 name=$2; shift 2; say "start $name on gpu $gpu"; CUDA_VISIBLE_DEVICES=$gpu $UVRUN --with "transformers==5.15.1" $B "$@" --device cuda:0 > $LG/$name.log 2>&1; grep -q "^\[done\]" $LG/$name.log && say "done  $name" || say "NO-DONE-MARKER $name"; }
( job ${GPUS[0]} rel${SFX}_tF_$M --alpha $AF --bank function --op transport --axis shape  --ablate-desc zero --ablate-span desc
  job ${GPUS[0]} rel${SFX}_iF_$M --alpha $AF --bank function --op install --axis shape  --ablate-desc none ) &
( job ${GPUS[1]} rel${SFX}_tL_$M --alpha $AF --bank function --op transport --axis colour --ablate-desc zero --ablate-span lang
  job ${GPUS[1]} rel${SFX}_iL_$M --alpha $AF --bank function --op install --axis colour --ablate-desc none ) &
( job ${GPUS[2]} rel${SFX}_tS_$M --alpha $AS --bank strategy --op transport --ablate-desc zero --ablate-span cue
  job ${GPUS[2]} rel${SFX}_iS_$M --alpha $AS --bank strategy --op install --ablate-desc none ) &
( job ${GPUS[3]} rel${SFX}_xFL_$M --alpha $AF --bank function --op compose  --ablate-desc none --match-norm
  job ${GPUS[3]} rel${SFX}_x3_$M --alpha $AF --bank function --op compose3 --ablate-desc none --match-norm ) &
wait; touch $LG/RELABEL_${MODE}_DONE.marker; say "$MODE relabel controls finished"

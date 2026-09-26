#!/usr/bin/env bash
# Random-labelling controls for mistral-small-3.2-24b, SUFFIX site, exactly the null recipe of run_cast_ladder.sh (tag 9U2 so
# assemble_9F.py symlinks them under the set name). Four chains, two per card, on GPUs ${GPUS:-3 7}.
set -u
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # the reproducibility root
export FPL_TAIL="Name the function solve. Return only the function, with no explanation, no examples, and no tests. Do not print anything and do not read any input. Return the result from the function rather than displaying it."
F=factored_programming_languages; R=$F/results; LG=$R/_logs; M=mistral-small-3.2-24b; GPUS=(${GPUS:-3 7})
LANGS="javascript go java c cpp rust ruby ocaml haskell"; SEEDS="0 1 2 3 4 5 6 7 8 9 10 11"
B="$F/run_tci_relabel.py --model $M --langs $LANGS python --seeds $SEEDS --tag 9U2 --alpha 1"
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
job() { local gpu=$1 name=$2; shift 2; say "start $name on gpu $gpu"; CUDA_VISIBLE_DEVICES=$gpu $UVRUN --with "transformers==5.15.1" $B "$@" --device cuda:0 > $LG/$name.log 2>&1; grep -q "^DONE (\|^\[done\]\|\[relabel\] .*done\|all seeds done" $LG/$name.log && say "done  $name" || say "NO-DONE-MARKER $name"; }
( job ${GPUS[0]} rel_tF_$M --bank function --op transport --axis shape  --ablate-desc zero --ablate-span desc
  job ${GPUS[0]} rel_iF_$M --bank function --op install --axis shape  --ablate-desc none ) &
( job ${GPUS[0]} rel_tL_$M --bank function --op transport --axis colour --ablate-desc zero --ablate-span lang
  job ${GPUS[0]} rel_iL_$M --bank function --op install --axis colour --ablate-desc none ) &
( job ${GPUS[1]} rel_tS_$M --bank strategy --op transport --ablate-desc zero --ablate-span cue
  job ${GPUS[1]} rel_iS_$M --bank strategy --op install --ablate-desc none ) &
( job ${GPUS[1]} rel_xFL_$M --bank function --op compose  --ablate-desc none --match-norm
  job ${GPUS[1]} rel_x3_$M --bank function --op compose3 --ablate-desc none --match-norm ) &
wait; touch $LG/RELABEL_suffix_DONE.marker; say "relabel controls finished"

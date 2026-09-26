#!/usr/bin/env bash
# The nine-language TCI ladder on mistral-small-3.2-24b (authors, 2026-09-25: the counter-example model), one site per call:
#   bash run_mistral_ladder.sh suffix ""            -> the standard 9M2 sources + unmasked transport (tcispan9w_*_bnone_)
#   bash run_mistral_ladder.sh online "0.125 0.25"  -> the same rungs with --site online (per-token counterfactual), dose per rung:
#                                                      transport f/l at $A_T, transport s at $A_S, install/compose at $A_I
# Result dirs carry the model name with a site tag so assemble_9F.py / fig_tci_ladder4.py can treat each site as its own "model":
#   suffix: <dir>_mistral-small-3.2-24b          online: <dir>_mistral-small-3.2-24b-onl
# Two jobs share a card (24B bf16 = 48 GB). Chains are static; a marker LADDER_<site>_DONE lands in results/_logs when all end.
set -u
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # the reproducibility root
export FPL_TAIL="Name the function solve. Return only the function, with no explanation, no examples, and no tests. Do not print anything and do not read any input. Return the result from the function rather than displaying it."
SITE=$1; GPUS=(${GPUS:-4 5 6}); A_T=${A_T:-0.25}; A_S=${A_S:-0.25}; A_I=${A_I:-0.125}
F=factored_programming_languages; R=$F/results; LG=$R/_logs; M=mistral-small-3.2-24b
LANGS="javascript go java c cpp rust ruby ocaml haskell"
ARMS_T="none delta exact rand prompted prompted_sev"
ARMS_S="none delta exact rand single prompted prompted_sev mean wrongs"
ARMS_C3="none delta exact rand mean wrong prompted prompted_sev"
if [ "$SITE" = online ]; then TAG="$M-onl"; ST="--site online --online-layers ${LAYERS:-all}"
elif [ "$SITE" = vector ]; then TAG="$M-vec"; ST="--site online --online-layers ${LAYERS:-all} --online-mode vector"   # frozen pooled generation vector
else TAG="$M"; ST=""; A_T=1; A_S=1; A_I=1; fi
OV=($UVRUN --with "transformers==5.15.1")
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
job() {  # job <gpu> <outname> <runner> args...
  local gpu=$1 out=$2 runner=$3; shift 3
  say "start $out on gpu $gpu"
  CUDA_VISIBLE_DEVICES=$gpu "${OV[@]}" $F/$runner --model $M --device cuda:0 --langs $LANGS "$@" --outdir $R/$out > $LG/$out.log 2>&1
  grep -q "^\[done\]\|^DONE" $LG/$out.log && say "done  $out" || say "NO-DONE-MARKER $out (see $LG/$out.log)"
}
NG=${#GPUS[@]}; G0=${GPUS[0]}; G1=${GPUS[$((1 % NG))]}; G2=${GPUS[$((2 % NG))]}; G3=${GPUS[$((3 % NG))]}; G4=${GPUS[$((4 % NG))]}; G5=${GPUS[$((5 % NG))]}   # six chains over the given cards
# chain A (gpu 0): install f/l + compose, unsevered                       | chain B (gpu 0): transport f masked -> l masked -> f unmasked
( job $G0 tci9U1_$TAG run_tci_severed.py --ablate-desc none --alpha $A_I $ST --ops install compose ) &
( job $G3 tcispan9w_f_bmask_$TAG run_tci_severed.py --ablate-desc mask --ablate-span spec --ops transport --alpha $A_T $ST --arms $ARMS_T --no-lfactor
  job $G3 tcispan9w_l_bmask_$TAG run_tci_severed.py --ablate-desc mask --ablate-span spec --ops transport --alpha $A_T $ST --arms $ARMS_T --only-lfactor
  job $G3 tcispan9w_f_bnone_$TAG run_tci_severed.py --ablate-desc none --ops transport --alpha $A_T $ST --arms $ARMS_T --no-lfactor ) &
# chain C (gpu 1): install s                                            | chain D (gpu 1): transport s masked -> l unmasked
( job $G1 tcis9U1_$TAG run_tci_strategy.py --ablate-desc none --alpha $A_S $ST --donor-langs python --ops install --arms $ARMS_S ) &
( job $G4 tcispan9w_s_bmask_$TAG run_tci_strategy.py --donor-langs python --ablate-desc mask --ablate-span spec --ops transport --alpha $A_S $ST --arms $ARMS_T
  job $G4 tcispan9w_l_bnone_$TAG run_tci_severed.py --ablate-desc none --ops transport --alpha $A_T $ST --arms $ARMS_T --only-lfactor ) &
# chain E (gpu 2): compose3 (match-norm)                                | chain F (gpu 2): transport s unmasked -> relabel controls
( job $G2 tcic3m9U1_$TAG run_tci_compose3.py --ablate-desc none --alpha $A_I $ST --donor-langs python --arms $ARMS_C3 --match-norm ) &
( job $G5 tcispan9w_s_bnone_$TAG run_tci_strategy.py --donor-langs python --ablate-desc none --ops transport --alpha $A_S $ST --arms $ARMS_T
  job $G5 tcim9U1_$TAG run_tci_severed.py --ablate-desc none --alpha $A_I $ST --ops compose --match-norm ) &
wait
touch $LG/LADDER_${SITE}_DONE.marker; say "ladder ($SITE) all chains finished"

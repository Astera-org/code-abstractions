#!/usr/bin/env bash
# The transport / install / compose ladder of Fig. 3 for ONE cast model, padded-prompt alignment (the runners'
# default): every prompt is front-padded to one length so the shared tail sits at the same absolute K/V rows.
#   transport rungs   whole-specification attention mask after prefill (--ablate-desc mask --ablate-span spec; the
#                     hybrid qwen3.6-27b uses `masknative`: recurrent state left intact, native recurrent contrast)
#   install/compose   words intact, alpha 1 (tci9U1 / tcim9U1 / tcis9U1 / tcic3m9U1; the hybrid adds its recurrent
#                     channel: --recurrent --channels kv kvrec --rec-sever none --rec-delta state)
#   controls          random-labelling nulls under the ladder's own regime (run_tci_relabel.py, tag 9F): transport
#                     nulls sever the same span as the rung (desc / lang / cue), install and compose nulls unsevered;
#                     the figure reads them with --relabel-tag 9F (assemble_cast_ladder.sh)
#
#   bash factored_programming_languages/run_cast_ladder.sh <model> <card A> <card B>     (two lanes per card)
#   then: bash factored_programming_languages/assemble_cast_ladder.sh
# Smoke: LANGS="javascript c" SEEDS="0" CELLS=2 XTRA="--max-new-tokens 96" bash run_cast_ladder.sh pythia-70m 0 0
set -u
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # the reproducibility root
export FPL_TAIL="${FPL_TAIL:-Name the function solve. Return only the function, with no explanation, no examples, and no tests. Do not print anything and do not read any input. Return the result from the function rather than displaying it.}"
M=$1; GA=$2; GB=$3
F=factored_programming_languages; R=$F/results; LG=$R/_logs/ladder; mkdir -p $LG
LANGS=${LANGS:-"javascript go java c cpp rust ruby ocaml haskell"}; LA="--langs $LANGS"; XTRA=${XTRA:-}; OV=${OV:-}; CELLS=${CELLS:-16}
ARMS_T="none delta exact rand prompted prompted_sev"
ARMS_S="none delta exact rand single prompted prompted_sev mean wrongs"
ARMS_C3="none delta exact rand mean wrong prompted prompted_sev"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11"}
# hybrid (qwen3.6-27b): transport in the masknative regime, install/compose in the native regime
case "$M" in
  qwen*) QT="--rec-sever none --recurrent --channels kv kvrec --rec-delta state"; QI="--recurrent --channels kv kvrec --rec-sever none --rec-delta state"; TB=bmasknative;;
  *)     QT=""; QI=""; TB=bmask;;
esac
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
job() {  # job <gpu> <outname> <runner> args...
  local gpu=$1 out=$2 runner=$3; shift 3
  if grep -qs "^DONE (\|^\[done\]\|\[relabel\] .*done\|all seeds done" $LG/$out.log; then say "skip  $out (done)"; return; fi
  say "start $out on gpu $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $UVRUN $OV $F/$runner --model $M --device cuda:0 "$@" $XTRA > $LG/$out.log 2>&1
  grep -q "^DONE (\|^\[done\]\|\[relabel\] .*done\|all seeds done" $LG/$out.log && say "done  $out" || say "NO-DONE-MARKER $out (see $LG/$out.log)"
}
BF="run_tci_relabel.py --langs $LANGS python --seeds $SEEDS --tag 9F --alpha 1 --cells $CELLS $QI"      # the random-labelling nulls
# lane 1 (card A): install f/l + compose (words intact, alpha 1); compose match-norm; F/L install + compose nulls
( job $GA tci9U1_$M run_tci_severed.py $LA --ablate-desc none --alpha 1 $QI --ops install compose --outdir $R/tci9U1_$M
  job $GA tcim9U1_$M run_tci_severed.py $LA --ablate-desc none --alpha 1 $QI --ops compose --match-norm --outdir $R/tcim9U1_$M
  job $GA rel_iF_$M $BF --bank function --op install --axis shape  --ablate-desc none
  job $GA rel_iL_$M $BF --bank function --op install --axis colour --ablate-desc none
  job $GA rel_xFL_$M $BF --bank function --op compose  --ablate-desc none --match-norm
  job $GA rel_x3_$M $BF --bank function --op compose3 --ablate-desc none --match-norm ) &
# lane 2 (card A): transport f, l under the whole-spec mask; F/L transport nulls
( job $GA tcispan9w_f_${TB}_$M run_tci_severed.py $LA --ablate-desc mask $QT --ablate-span spec --ops transport --alpha 1 --arms $ARMS_T --no-lfactor --outdir $R/tcispan9w_f_${TB}_$M
  job $GA tcispan9w_l_${TB}_$M run_tci_severed.py $LA --ablate-desc mask $QT --ablate-span spec --ops transport --alpha 1 --arms $ARMS_T --only-lfactor --outdir $R/tcispan9w_l_${TB}_$M
  job $GA rel_tF_$M $BF --bank function --op transport --axis shape  --ablate-desc zero --ablate-span desc
  job $GA rel_tL_$M $BF --bank function --op transport --axis colour --ablate-desc zero --ablate-span lang ) &
# lane 3 (card B): install s (words intact); S nulls
( job $GB tcis9U1_$M run_tci_strategy.py $LA --ablate-desc none $QI --donor-langs python --ops install --arms $ARMS_S --outdir $R/tcis9U1_$M
  job $GB rel_iS_$M $BF --bank strategy --op install --ablate-desc none
  job $GB rel_tS_$M $BF --bank strategy --op transport --ablate-desc zero --ablate-span cue ) &
# lane 4 (card B): compose3 (match-norm); transport s under the whole-spec mask
( job $GB tcic3m9U1_$M run_tci_compose3.py $LA --ablate-desc none $QI --donor-langs python --arms $ARMS_C3 --match-norm --outdir $R/tcic3m9U1_$M
  job $GB tcispan9w_s_${TB}_$M run_tci_strategy.py $LA --donor-langs python --ablate-desc mask $QT --ablate-span spec --ops transport --arms $ARMS_T --outdir $R/tcispan9w_s_${TB}_$M ) &
wait
touch $LG/LADDER_${M}_DONE.marker; say "ladder ($M) all lanes finished"

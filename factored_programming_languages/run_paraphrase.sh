#!/usr/bin/env bash
# Paraphrase robustness of the function / language rungs (appendix app:paraphrase), for ONE model.
# Four wording conditions (descriptions: 0 = canonical, 1-3 = tasks_atlas_paraphrases.PARA[f]):
#   pavg   --donor-para 0,1,2,3 --recipient-para 0        donors pooled over all four wordings
#   pdisj  --donor-para 1,2,3   --recipient-para 0        donors never use the recipient's wording
#   rpara  --donor-para 0       --recipient-para rotate   recipient (and its prompted asks) paraphrased
#   both   --donor-para 0,1,2,3 --recipient-para rotate
# each through the three jobs of the published ladder (run_cast_ladder.sh): transport f and transport lambda under the
# whole-specification mask, install + compose words intact. Outputs results/tcipara_<cond>_<f|l|ic>_<model>; the
# canonical-wording baseline is the cast ladder's own tcispan9w_* / tci9U1_* sets. RWKV-7 runs its words-intact ladder
# (run_rwkv_tci.py, set para<cond>Rpl, overlay transformers==5.15.1) on card A only.
#
#   bash factored_programming_languages/run_paraphrase.sh <model> <card A> <card B>
#   COMPOSE3=1 ...          also the compose3 rung with pooled donor wordings (pavg, pdisj; run for gemma-4-31B-it)
#   then: uv run factored_programming_languages/para_table.py; uv run factored_programming_languages/figures/fig_para_ladder.py
# Smoke: CONDS="pavg" LANGS="javascript c" XTRA="--max-new-tokens 96" bash run_paraphrase.sh pythia-70m 0 0
set -u
UVRUN=${UVRUN:-"uv run"}
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # the reproducibility root
export FPL_TAIL="${FPL_TAIL:-Name the function solve. Return only the function, with no explanation, no examples, and no tests. Do not print anything and do not read any input. Return the result from the function rather than displaying it.}"
M=$1; GA=$2; GB=${3:-$2}
F=factored_programming_languages; R=$F/results; LG=$R/_logs/paraphrase; mkdir -p $LG
LANGS=${LANGS:-"javascript go java c cpp rust ruby ocaml haskell"}; XTRA=${XTRA:-}
CONDS=${CONDS:-"pavg pdisj rpara both"}
ARMS_T="none delta exact rand prompted prompted_sev"
ARMS_C3="none delta exact rand mean wrong prompted prompted_sev"
declare -A WORD=([pavg]="--donor-para 0,1,2,3 --recipient-para 0" [pdisj]="--donor-para 1,2,3 --recipient-para 0"
                 [rpara]="--donor-para 0 --recipient-para rotate" [both]="--donor-para 0,1,2,3 --recipient-para rotate")
case "$M" in qwen*) HY="--recurrent --channels kv kvrec --rec-sever none --rec-delta state";; *) HY="";; esac
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
job() {  # job <gpu> <outname> <overlay or ""> <runner> args...
  local gpu=$1 out=$2 ov=$3 runner=$4; shift 4
  if grep -qs "^DONE (\|^\[done\]" $LG/$out.log; then say "skip  $out (done)"; return; fi
  say "start $out on gpu $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $UVRUN $ov $F/$runner --device cuda:0 "$@" $XTRA > $LG/$out.log 2>&1
  grep -q "^DONE (\|^\[done\]" $LG/$out.log && say "done  $out" || say "NO-DONE-MARKER $out (see $LG/$out.log)"
}

if [[ "$M" == rwkv* ]]; then
  for c in $CONDS; do
    job $GA tcipara${c}Rpl_$M "--with transformers==5.15.1" run_rwkv_tci.py --langs $LANGS --patch-point prelast \
        --set para${c}Rpl --factors F L ${WORD[$c]}
  done
  say "paraphrase ladder ($M) finished"; exit 0
fi

# card A: transport f and transport lambda under the whole-spec mask; card B: install + compose, words intact
( for c in $CONDS; do
    job $GA tcipara_${c}_f_$M "" run_tci_severed.py --model $M --langs $LANGS --ablate-desc mask --ablate-span spec --ops transport \
        --alpha 1 --arms $ARMS_T --no-lfactor $HY ${WORD[$c]} --outdir $R/tcipara_${c}_f_$M
    job $GA tcipara_${c}_l_$M "" run_tci_severed.py --model $M --langs $LANGS --ablate-desc mask --ablate-span spec --ops transport \
        --alpha 1 --arms $ARMS_T --only-lfactor $HY ${WORD[$c]} --outdir $R/tcipara_${c}_l_$M
  done ) &
( for c in $CONDS; do
    job $GB tcipara_${c}_ic_$M "" run_tci_severed.py --model $M --langs $LANGS --ablate-desc none --alpha 1 --ops install compose \
        $HY ${WORD[$c]} --outdir $R/tcipara_${c}_ic_$M
  done
  if [ "${COMPOSE3:-0}" = 1 ]; then      # optional: compose3 with pooled donor wordings (the recipient is the carrier)
    for c in pavg pdisj; do case " $CONDS " in *" $c "*) ;; *) continue;; esac
      job $GB tcipara_${c}_c3_$M "" run_tci_compose3.py --model $M --langs $LANGS --ablate-desc none --donor-langs python \
          --arms $ARMS_C3 --match-norm $HY ${WORD[$c]} --outdir $R/tcipara_${c}_c3_$M
    done
  fi ) &
wait
touch $LG/PARAPHRASE_${M}_DONE.marker; say "paraphrase ladder ($M) all lanes finished"

#!/usr/bin/env bash
# Smoke test: run every pipeline family on a tiny subset (small public models, two languages, one function pair,
# one seed, short generations) and render every figure script on the result. It checks that the code runs end to end
# from scratch; the numbers it produces are meaningless. Runs on ONE GPU in well under two hours (the SFT step needs
# deepspeed). Everything it writes stays inside this package (results/, figures/, smoke/, toy_model/{mlruns,results,logs,figures,paper_tables}).
#
#   bash smoke_test.sh [family ...]        families: setup bench sever pretrained tci grid icl sft variance geometry
#                                                    modifier subsets online setupfig toy figures   (default: all)
#                                          opt-in, not in the default list: paraphrase  (bash smoke_test.sh paraphrase)
#   GPU=3 bash smoke_test.sh               pick the card (default 0)
#   UVRUN=/path/to/.venv/bin/python bash smoke_test.sh     run with an existing environment instead of `uv run`
#
# Language toolchains: the grading step compiles C/C++/Go/Rust/Java/OCaml/Haskell/Ruby/JavaScript programs; a
# missing toolchain makes those cells grade as compile failures but does not stop the smoke test (see README.md).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source env.sh
export UVRUN=${UVRUN:-"uv run"}
export HF_HOME=${HF_HOME:-$PWD/smoke/hf_cache}          # small checkpoints are downloaded here unless HF_HOME is set
GPU=${GPU:-0}; export CUDA_VISIBLE_DEVICES=$GPU
F=factored_programming_languages; I=icl_section; R=$F/results; LOG=smoke/logs; mkdir -p $LOG smoke
M=pythia-70m                                             # the smoke model for every attention-only pipeline
MH=qwen3.5-0.8b                                          # the hybrid (linear + full attention) smoke model for the ICL stages
PAIR="digit_prod-digit_sum"
say() { echo "[$(date '+%H:%M:%S')] $*"; }
STATUS=()
run() {   # run <name> <command...>  : log to smoke/logs/<name>.log, record pass/fail
  local name=$1; shift; say "$name: $*" | cut -c1-200
  if "$@" > $LOG/$name.log 2>&1; then STATUS+=("PASS  $name"); else STATUS+=("FAIL  $name  (see $LOG/$name.log)"); say "  FAILED: $name"; fi
}
py() { $UVRUN "$@"; }

setup() {
  run setup.runners py $F/verify_runners.py
  run setup.corpora py $F/generate_algo_corpora.py
}
bench() { run bench py $F/run_bench.py --model $M --device cuda:0 --shard 0/31 --max-new-tokens 96; }
sever() { run sever.flow py $F/run_ablate_tiers.py --model $M --device cuda:0 --bank flow --langs python c --modes none abl_rand abl_mask --max-new-tokens 96 --outdir $R/ablateflow_$M; }
pretrained() { run pretrained py $F/run_completion_ablate.py --model $M --device cuda:0 --langs c --tasks digit_sum --modes none abl_mask --layout godoc --width xwide --max-new-tokens 64; }
tci() {   # the real driver (padded alignment), on two languages, one relabel seed, two relabel cells, short generations
  run tci.ladder env LANGS="javascript c" SEEDS="0" CELLS=2 XTRA="--max-new-tokens 96" bash $F/run_cast_ladder.sh $M $GPU $GPU
  run tci.assemble env CAST="$M" MODELS5="$M=$M (smoke)" RWKV_SET="" bash $F/assemble_cast_ladder.sh
}
grid() {
  run grid.donor py $F/run_tci_strategy.py --model $M --device cuda:0 --ops transport --arms single --single-donor-lang c --langs javascript c --donor-langs python --ablate-desc mask --ablate-span spec --max-new-tokens 96 --outdir $R/donorgrid_${M}_c
  run grid.donorI py $F/run_tci_strategy.py --model $M --device cuda:0 --ops install --arms single --single-donor-lang c --langs javascript c --donor-langs python --ablate-desc mask --ablate-span spec --max-new-tokens 96 --outdir $R/donorgridI_${M}_c
  run grid.prompted py $F/run_tci_strategy.py --model $M --device cuda:0 --ops transport --arms none prompted --langs javascript c --donor-langs python --ablate-desc mask --ablate-span spec --max-new-tokens 96 --outdir $R/tcish_$M
  run grid.xlang py $F/run_tci_strategy.py --model $M --device cuda:0 --ops transport install --arms none delta exact xlang --xlang-donors c --langs javascript c --donor-langs python --ablate-desc mask --ablate-span spec --max-new-tokens 96 --outdir $R/tcisx_$M
}
icl() {
  run icl.stageA py $I/run_stageA.py --model $MH --device cuda:0 --funcs thulep --langs python javascript --max-new-tokens 64 --out $I/results/stageA_$MH.json
  run icl.stageB py $I/run_stageB.py --model $MH --device cuda:0 --funcs thulep --langs python javascript --stageA $I/results/stageA_$MH.json --out $I/results/stageB_$MH.json
  run icl.verbatim py $I/test_verbatim_recall.py --model $MH --device cuda:0 --funcs thulep
  run icl.copy py $I/test_copy_control.py --model $MH --device cuda:0 --funcs thulep
}
sft() {
  run sft.train py -m torch.distributed.run --nproc_per_node=1 --master_port=29777 $F/run_sft_vantril.py --model $M --batch-size 9 --epochs 2 --lr 5e-7
  CK=$R/EleutherAI--$M/sft_vantril/full_vantril_portable
  run sft.eval_base py $F/eval_vantril.py --model $M --device cuda:0 --langs python javascript c --max-new-tokens 64
  run sft.eval_teacher py $F/eval_vantril.py --model $M --ckpt $CK --tag teacher --device cuda:0 --langs python javascript c --max-new-tokens 64
  run sft.algo_base py $F/eval_algo.py --algo kanex --model $M --device cuda:0 --langs python javascript --max-new-tokens 64
  run sft.algo_teacher py $F/eval_algo.py --algo kanex --model $M --ckpt $CK --tag teacher --device cuda:0 --langs python javascript --max-new-tokens 64
}
variance() {
  run variance.extract py $F/extract_fxl20.py --model $M --device cuda:0 --suffix fxl20
  run variance.factorial py $F/analysis_factorial.py --model-dir $R/EleutherAI--$M--fxl20
  run variance.bands py $F/certify_bands.py --model $M
}
geometry() {
  run geometry.steps py $F/run_step_algebra.py --model $M --device cuda:0 --langs python javascript --max-new-tokens 64
  run geometry.rays py $F/run_greedy_walk.py --model $M --device cuda:0 --rays --ray-radii 0.5 1.0 --ray-randoms 1 --max-new-tokens 64 --outdir $R/rays_$M
  run geometry.rayscores py $F/run_greedy_walk.py --model $M --device cuda:0 --score-rays --score-radii 0 0.5 1.0 --max-new-tokens 64 --outdir $R/rayscores_$M/shard0
  for l in python javascript; do run geometry.atlas_$l py $F/run_atlas_harvest.py --model $M --device cuda:0 --lang $l; done
  run geometry.atlas py $F/atlas_geometry.py --model $M
}
modifier() {
  run modifier.dev py $F/run_modifier_dho.py --model $M --device cuda:0 --phase dev --alphas 1 --bands all --max-new-tokens 64
  MODEL=$M run modifier.select py $F/summarize_modifier_dho.py --select
  run modifier.test py $F/run_modifier_dho.py --model $M --device cuda:0 --phase test --max-new-tokens 64
}
subsets() { run subsets py $F/run_subsets.py --model $M --device cuda:0 --subsets 1 --sizes 0 2 --suffixes 40 --max-new-tokens 64; }
online() { run online py $F/run_tci_severed.py --model $M --device cuda:0 --langs javascript c --site online --online-layers all --ops install --alpha 0.0625 --ablate-desc none --max-new-tokens 64 --outdir $R/tci9U1_$M-onl; }
setupfig() { run setupfig py $F/paper_figures/figA0_setup.py; }
toy() {   # the co-authors' toy in its own environment (toy_model/): tests, a 200-step training of both models, a reduced suite, the figures
  [ -n "${TOYRUN:-}" ] || run toy.env bash -c "cd toy_model && uv sync"                 # toy_model/.venv (skipped when TOYRUN names an interpreter)
  run toy.tests bash -c "cd toy_model && JAX_PLATFORMS=cpu ${TOYRUN:-uv run python} -m pytest -q tests"
  run toy.train env STEPS=200 SEEDS=1 GPU=0 bash toy_model/run_toy.sh train           # GPU=0: CUDA_VISIBLE_DEVICES already selects the card
  run toy.suites env SEEDS=1 GPU=0 SUITE_ARGS="--n-recipients 3 --recipient-pool 12 --min-function-certainty 0.95 --n-donor-paths 32" bash toy_model/run_toy.sh suites
  run toy.figures env SEEDS=1 bash toy_model/run_toy.sh figures
}
paraphrase() {   # OPT-IN (not in the default FAMS): the paraphrase driver on one condition, then its table and figure
  run para.ladder env CONDS="pavg" LANGS="javascript c" XTRA="--max-new-tokens 96" bash $F/run_paraphrase.sh $M $GPU $GPU
  run para.table py $F/para_table.py
  run para.figure py $F/figures/fig_para_ladder.py
}
figures() {   # every paper figure script on the smoke results (sparse figures are expected)
  run fig2 py $F/paper_figures/fig5_scatter.py --ablmode abl_mask --by-attn --age-color --min-cells 5
  run fig16 py $F/figures/fig_bench_lcb.py
  run fig19 py $F/paper_figures/fig_sever_base_vs_instruct.py --pairs "$M=$M=$M (smoke)"
  run fig17a py $F/figures/fig_subsets_surface.py --model $M
  run fig17b py $F/figures/fig_clause_difficulty.py --model $M
  run fig8 py $F/figures/fig_variance_depth.py
  run fig9 py $F/paper_figures/fig_geometry.py --model $M
  run fig10 py $F/paper_figures/fig_modifier_dho.py --models "$M=$M (smoke)=modifier_dho_$M/test/*.json"
  run fig4b py $F/paper_figures/fig5_reuse_bars.py --only-a --models $M
  run fig18 py $F/paper_figures/fig5b_algo_model.py --models $M --algos kanex
  run fig6.grid py $F/paper_figures/fig_donor_recipient.py --model $M --op transport --differs-figure
  run fig6 py $F/paper_figures/fig_differs_summary.py --models "$M=$M (smoke)"
  run fig7 py $F/paper_figures/fig_differs_examples.py --models "$M=$M (smoke)"
  run fig5 py $F/paper_figures/fig4_transport_structure.py --model $M --cell digit_sum --lang javascript --want iter --donor-cell digit_prod --donor-lang c
  run fig4a.count py $I/count_install.py --models "$MH=stageB_$MH.json"
  run fig4a py $I/fig_install.py --models "$MH=$MH (smoke)"
  run fig4a.table py $I/table_install.py --models "$MH=$MH (smoke)"
  run table1 py $I/table_samelang.py --tex --models "$MH=$MH (smoke)=copy_control_$MH.json=verbatim_recall_$MH.json" --funcs thulep thulep thulep thulep
  run table4 py $F/table_bench_tasks.py
  run fig20 py $F/figures/fig_tci_ladder4.py --set 9U1 --equal --out fig_tci_ladder4_smoke_online --no-set-label --models "$M-onl=$M online (smoke)"
}

FAMS=${@:-setup bench sever pretrained tci grid icl sft variance geometry modifier subsets online setupfig toy figures}
for fam in $FAMS; do say "== $fam"; $fam; done
echo; echo "SMOKE TEST SUMMARY"; printf '  %s\n' "${STATUS[@]}"
printf '%s\n' "${STATUS[@]}" > smoke/SUMMARY.txt

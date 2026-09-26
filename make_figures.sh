#!/usr/bin/env bash
# Regenerate every figure and table of the paper from scratch, one target per figure family.
#
#   bash make_figures.sh <family> [<family> ...]        e.g.  bash make_figures.sh setup bench sever fig2
#   bash make_figures.sh all                             everything, in dependency order (many GPU-days)
#
# Families, in dependency order (README.md has the figure -> family map and the compute estimates):
#   setup      environment checks, execution-runner positive control, SFT corpora
#   bench      in-house 310-cell benchmark, 26 models + RWKV                       (Fig 2 x-axis, Fig 16)
#   sever      description masked after prefill, whole benchmark + flow bank        (Fig 2, Fig 16, Fig 19 instruct arm)
#   fig2       render Fig 2 and Fig 16
#   pretrained completion-layout severing of the pretrained twins                   (Fig 19)
#   tci        transport / install / compose ladders, 4 models + RWKV               (Fig 3)
#   grid       single-donor strategy grids + the matched triplet                    (Fig 5, 6, 7)
#   icl        in-context installation stages + same-language tests                 (Fig 4a, Table 1)
#   sft        Python-only fine-tuning ladders + evaluation                         (Fig 4b, Fig 18, Tables 4-6)
#   variance   fxl20 activation extraction + factorial decomposition                (Fig 8)
#   geometry   step algebra, rays, atlas                                            (Fig 9)
#   modifier   double-held-out modifier transfer, 3 models + RWKV                   (Fig 10)
#   subsets    demand subsets x suffix length                                       (Fig 17)
#   mistral    Mistral-3.2 four-site ladder                                         (Fig 20)
#   setupfig   the prompt-layout figure                                             (Fig 15)
#   paraphrase paraphrased descriptions on the ladder rungs, needs tci first        (Appendix app:paraphrase)
#   toy        the minimal programming process (toy_model/run_toy.sh, own environment)  (Fig 12, 13, Table 3)
#
# Every command runs from the package root. GPU assignment is through CUDA_VISIBLE_DEVICES / the drivers'
# card arguments; edit the CARD* variables below for your machine. Nothing here is parallelised across
# machines: the paper's runs took roughly the GPU-hours quoted in README.md on 8x H100.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source env.sh
export UVRUN=${UVRUN:-"uv run"}
OV='--with transformers==5.15.1'                 # per-model dependency overlay (see README.md)
F=factored_programming_languages; I=icl_section; R=$F/results
CAST="gemma-4-31B-it gemma-4-26b-a4b qwen3.6-27b pythia-6.9b-tulu"
CAST3="gemma-4-31B-it gemma-4-26b-a4b qwen3.6-27b"
BENCH_MODELS="gemma-4-31B-it gemma-4-26b-a4b qwen3.6-27b gemma-4-12b-it qwen3.6-35b-a3b gemma-3-27b-it qwen3.8-27b qwen3-30b-2507 qwen3.5-35b-a3b qwen3-coder-30b devstral-2-123b glm-4-32b-instruct gemma-3-12b-it mistral-small-4-119b qwen3-32b qwen3-30b-a3b mistral-small-24b-2501 exaone-4-32b mistral-small-3.2-24b qwen2.5-14b dscoder-v2-lite ministral-3-8b mistral-nemo-12b gemma-2-9b-it mistral-7b-v0.3 pythia-6.9b-tulu"
OVERLAY_MODELS="gemma-4-12b-it mistral-small-3.2-24b exaone-4-32b"      # need transformers==5.15.1
BIG_MODELS="devstral-2-123b mistral-small-4-119b"                        # do not fit one 80 GB card: --max-memory over all cards
LANGS10="python javascript c cpp java go rust ruby ocaml haskell"
LANGS9="javascript go java c cpp rust ruby ocaml haskell"
CARD=${CARD:-0}; CARDB=${CARDB:-1}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
ovl() { case " $OVERLAY_MODELS " in *" $1 "*) echo "$OV";; esac; }        # overlay flag for a model, or nothing

setup() {
  say "python: $($UVRUN python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)')"
  say "execution runners (positive control -- every language must grade a hand-written digit_sum as correct)"
  $UVRUN $F/verify_runners.py
  say "SFT corpora (writes data/<algo>_portable_{fence,bare}.json; asserts the shipped vantril corpus is reproduced)"
  $UVRUN $F/generate_algo_corpora.py
  say "nonce functions (writes icl_section/nonce_functions.json; the shipped file was produced by this command)"
  $UVRUN $I/nonce.py --n 12 --seed 0 --out $I/nonce_functions.json.regenerated
}

bench() {   # 31 tasks x 10 languages, greedy, reasoning off, execution-scored -> results/bench_<model>
  for m in $BENCH_MODELS; do
    case " $BIG_MODELS " in *" $m "*) $UVRUN $(ovl $m) $F/run_bench.py --model $m --max-memory "0:75GiB,1:75GiB,2:75GiB,3:75GiB,4:75GiB,5:75GiB,6:75GiB,7:75GiB";;
      *) CUDA_VISIBLE_DEVICES=$CARD $UVRUN $(ovl $m) $F/run_bench.py --model $m --device cuda:0;; esac
  done
  CUDA_VISIBLE_DEVICES=$CARD $UVRUN $OV $F/run_rwkv_bench.py --device cuda:0 --langs $LANGS10
}

sever() {   # attention to the description masked after prefill: whole bench (Fig 2) and the 4-task flow bank (Fig 16, 19)
  for m in $BENCH_MODELS; do
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $(ovl $m) $F/run_ablate_tiers.py --model $m --device cuda:0 --bank flow \
        --langs $LANGS10 --modes none abl_rand abl_mask --outdir $R/ablateflow_$m
  done
  bash $F/run_ablate_full_all.sh          # results/ablatefull_<model>: 310 cells under abl_mask for every model with a bench
}

fig2() {
  $UVRUN $F/paper_figures/fig5_scatter.py --ablmode abl_mask --by-attn --age-color --full     # Fig 2 (a,b) -> paper_figures/fig5_scatter_abl_mask_attn_age_full.png
  $UVRUN $F/figures/fig_bench_lcb.py                                                             # Fig 16 -> figures/fig_bench_lcb.png
}

pretrained() {   # Fig 19: pretrained twins through the completion layout; the instruct arm is results/ablateflow_<m> from `sever`
  for m in gemma-4-12b-pt gemma-4-26b-a4b-pt gemma-4-31B qwen3.5-35b-a3b-pt; do
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/run_completion_ablate.py --model $m --device cuda:0 --langs c go haskell \
        --tasks digit_sum digit_prod sq_digit_sum even_digits --modes none abl_mask --layout godoc --width xwide
  done
  $UVRUN $F/paper_figures/fig_sever_base_vs_instruct.py --instrument abl_mask   # -> paper_figures/fig_sever_base_vs_instruct.png
}

tci() {   # Fig 3: padded-prompt alignment (the runners' default). Transport under the whole-specification mask (--ablate-desc mask
          # --ablate-span spec; `masknative` on the hybrid), install/compose words-intact (9U1), random-labelling nulls (tag 9F).
  for m in $CAST; do bash $F/run_cast_ladder.sh $m $CARD $CARDB; done
  # RWKV-7: no KV cache, so the ladder runs words-intact after prefill; python-anchored language install, patch before the last token
  CUDA_VISIBLE_DEVICES=$CARD $UVRUN $OV $F/run_rwkv_tci.py --device cuda:0 --set 9Rpl --l-anchor python --patch-point prelast --langs $LANGS9
  bash $F/assemble_cast_ladder.sh          # assemble_9F.py --set 9M2 per model, then fig_tci_ladder4.py -> figures/fig_tci_ladder4_9M2_5m.png (+ _sevnorm)
}

grid() {   # Fig 5, 6, 7: single-donor strategy transport / install grids
  bash $F/run_donorgrid.sh; bash $F/run_donorgrid_install.sh      # gemma-4-26b-a4b + qwen3.6-27b (edit G/Q/cards for the 31B)
  for dl in $LANGS9; do
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/run_tci_strategy.py --model gemma-4-31B-it --device cuda:0 --ops transport --arms single \
        --single-donor-lang $dl --langs $LANGS9 --donor-langs python --ablate-desc mask --ablate-span spec --outdir $R/donorgrid_gemma-4-31B-it_$dl
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/run_tci_strategy.py --model gemma-4-31B-it --device cuda:0 --ops install --arms single \
        --single-donor-lang $dl --langs $LANGS9 --donor-langs python --ablate-desc mask --ablate-span spec --outdir $R/donorgridI_gemma-4-31B-it_$dl
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $OV $F/run_rwkv_tci.py --device cuda:0 --set 9Rpl --factors S --ops transport install --single-donor-lang $dl --langs $LANGS9
  done
  # the matched triplet of Fig 5: prompted arms (tcish_) and a Haskell-only xlang donor arm (tcisx_) on the 31B
  CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/run_tci_strategy.py --model gemma-4-31B-it --device cuda:0 --ops transport --arms none prompted \
      --langs $LANGS9 haskell --donor-langs python --ablate-desc mask --ablate-span spec --outdir $R/tcish_gemma-4-31B-it
  CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/run_tci_strategy.py --model gemma-4-31B-it --device cuda:0 --ops transport install --arms none delta exact xlang \
      --xlang-donors haskell --langs $LANGS9 --donor-langs python --ablate-desc mask --ablate-span spec --outdir $R/tcisx_gemma-4-31B-it
  for m in gemma-4-31B-it gemma-4-26b-a4b qwen3.6-27b rwkv7-g1j-13.3b; do for op in transport install; do
    $UVRUN $F/paper_figures/fig_donor_recipient.py --model $m --op $op --differs-figure; done; done
  $UVRUN $F/paper_figures/fig_differs_summary.py        # Fig 6  -> paper_figures/fig_differs_grids.png
  $UVRUN $F/paper_figures/fig_differs_examples.py       # Fig 7  -> paper_figures/fig_differs_examples.png
  $UVRUN $F/paper_figures/fig4_transport_structure.py   # Fig 5  -> paper_figures/fig4_transport_structure.png
}

icl() {   # Fig 4a + Table 1. Stage A (compile map) -> Stage B (installation) -> same-language tests; RWKV through its own runner.
  for m in $CAST3; do
    [ "$m" = gemma-4-31B-it ] && tag="" || tag="_$m"
    # one process per stage here; the paper sharded each stage over 8 cards with
    #   GPUS="0 1 2 3 4 5 6 7" $I/fanout.sh run_stageA.py stageA$tag --model $m   (then $I/merge_shards.py results/stageA$tag)
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $I/run_stageA.py --model $m --device cuda:0 --out $I/results/stageA$tag.json
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $I/run_stageB.py --model $m --device cuda:0 --funcs thulep vantril kanex faromin --stageA $I/results/stageA$tag.json --out $I/results/stageB$tag.json
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $I/test_verbatim_recall.py --model $m --device cuda:0 --funcs thulep vantril kanex faromin
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $I/test_copy_control.py --model $m --device cuda:0 --funcs thulep vantril kanex faromin
  done
  CUDA_VISIBLE_DEVICES=$CARD $UVRUN $OV $I/run_stage_rwkv.py --stage A --device cuda:0 --out $I/results/stageA_rwkv.json
  CUDA_VISIBLE_DEVICES=$CARD $UVRUN $OV $I/run_stage_rwkv.py --stage B --device cuda:0 --funcs thulep vantril kanex faromin --out $I/results/stageB_rwkv.json
  CUDA_VISIBLE_DEVICES=$CARD $UVRUN $OV $I/test_samelang_rwkv.py --device cuda:0 --funcs thulep vantril kanex faromin
  $UVRUN $I/pick_four.py $I/results/stageA.json           # the selection rule (the balanced pick used in the paper is the same four on every model)
  $UVRUN $I/count_install.py && $UVRUN $I/fig_install.py && $UVRUN $I/table_install.py   # Fig 4a -> icl_section/figures/fig_install.png
  $UVRUN $I/table_samelang.py --tex                                                    # Table 1 -> stdout (LaTeX)
}

sft() {   # Fig 4b, Fig 18, Tables 4-6: full-weight Python-only teachers at three doses, evaluated on the nine held-out languages
  for m in gemma-4-31B-it gemma-4-26b-a4b qwen3.6-27b pythia-6.9b-tulu; do
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/eval_vantril.py --model $m --device cuda:0                      # base arm
    for spec in "teacher 5e-7 36 " "teacher_hi 2e-6 36 _hi" "teacher_vhi 8e-6 36 _vhi"; do set -- $spec
      TXT=""; case $m in gemma-4*) TXT="--text-only";; esac                                              # gemma-4 under ZeRO-3: see README.md
      $UVRUN torchrun --nproc_per_node=8 $F/run_sft_vantril.py --model $m --batch-size 2 --lr $2 --epochs $3 --out-suffix "$4" $TXT
      CK=$R/$(python3 -c "import yaml;print(yaml.safe_load(open('models.yaml'))['$m']['hf_name'].replace('/','--'))")/sft_vantril/full_vantril_portable$4
      CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/eval_vantril.py --model $m --ckpt $CK --tag $1 --device cuda:0
    done
  done
  bash $F/run_rwkv_vantril.sh                                                                             # RWKV-7 ladder (4 ranks, overlay)
  $UVRUN $F/run_algo_sweep.py --lane big --algos thulep kanex faromin                                     # 3 more computations x 3 doses, big models
  $UVRUN $F/run_algo_sweep.py --lane pythia --algos thulep kanex faromin
  $UVRUN $F/run_algo_sweep.py --lane rwkv --algos thulep kanex faromin
  $UVRUN $F/summarize_algos.py
  $UVRUN $F/paper_figures/fig5_reuse_bars.py --only-a                                                     # Fig 4b -> paper_figures/fig5_reuse_bars_a.png
  $UVRUN $F/paper_figures/fig5b_algo_model.py --algos vantril thulep kanex faromin                         # Fig 18 -> paper_figures/fig5b_algo_model_4algos.png
}

variance() {   # Fig 8: per-layer ANOVA of the fxl20 corpus (20 functions x {loop, rec} x {fwd, rev} x 10 languages)
  for m in $CAST; do
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/extract_fxl20.py --model $m --device cuda:0 --suffix fxl20     # -> results/<hf-name>--fxl20/
    HF=$(python3 -c "import yaml;print(yaml.safe_load(open('models.yaml'))['$m']['hf_name'].replace('/','--'))")
    $UVRUN $F/analysis_factorial.py --model-dir $R/$HF--fxl20                                            # -> .../analysis/factorial_fxl20.json
  done
  $UVRUN $F/certify_bands.py --all
  $UVRUN $F/figures/fig_variance_depth.py                          # -> figures/fig_variance_depth.png
}

geometry() {   # Fig 9 (gemma-4-31B-it): (a) step algebra, (b,c) rays, (d) atlas alignments
  CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/run_step_algebra.py --model gemma-4-31B-it --device cuda:0
  CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/run_greedy_walk.py --model gemma-4-31B-it --device cuda:0 --rays --outdir $R/rays_gemma-4-31B-it   # the scorer and the figure read results/rays_<model>
  CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/run_greedy_walk.py --model gemma-4-31B-it --device cuda:0 --score-rays --outdir $R/rayscores_gemma-4-31B-it/shard0
  for l in python javascript java go rust haskell; do CUDA_VISIBLE_DEVICES=$CARD $UVRUN $F/run_atlas_harvest.py --model gemma-4-31B-it --device cuda:0 --lang $l; done
  $UVRUN $F/atlas_geometry.py                                       # -> results/atlas_gemma-4-31B-it/alignments.json
  $UVRUN $F/paper_figures/fig_geometry.py                           # -> paper_figures/fig_geometry.png
}

modifier() {   # Fig 10: double-held-out modifier transfer (dev phase selects band x dose on python/javascript; test on java/go/rust/haskell)
  MODEL=gemma-4-31B-it bash $F/run_modifier_dho.sh
  MODEL=gemma-4-26b-a4b bash $F/run_modifier_dho.sh
  MODEL=qwen3.6-27b CHANNELS=kvrec bash $F/run_modifier_dho.sh
  for l in python javascript java go rust haskell; do
    CUDA_VISIBLE_DEVICES=$CARD $UVRUN $OV $F/run_rwkv_tiers.py --device cuda:0 --tiers modifier --langs $l --outdir $R/rwkv_tiers6; done
  $UVRUN $F/paper_figures/fig_modifier_dho.py                       # -> paper_figures/fig_modifier_dho.png
}

subsets() {   # Fig 17 (gemma-4-31B-it): 10 random subsets of the 11 demands at every size x 12 suffix lengths, severed and intact arms
  for s in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$((s % 8)) $UVRUN $F/run_subsets.py --model gemma-4-31B-it --device cuda:0 --shard $s --nshards 8 > $F/logs/subsets_$s.log 2>&1 &
  done; wait
  $UVRUN $F/figures/fig_subsets_surface.py; $UVRUN $F/figures/fig_clause_difficulty.py      # -> figures/fig_subsets_surface.png, fig_clause_difficulty.png
}

mistral() {   # Fig 20: suffix site, suffix + masking, online dynamic, online static on mistral-small-3.2-24b (overlay: transformers==5.15.1)
  GPUS="$CARD $CARDB" bash $F/run_mistral_ladder.sh suffix
  GPUS="$CARD $CARDB" bash $F/run_mistral_ladder.sh online
  GPUS="$CARD $CARDB" bash $F/run_mistral_ladder.sh vector
  GPUS="$CARD $CARDB" bash $F/run_mistral_relabel.sh
  GPUS="$CARD $CARDB $CARD $CARDB" MODE=live bash $F/run_mistral_relabel_online.sh
  GPUS="$CARD $CARDB $CARD $CARDB" MODE=vector bash $F/run_mistral_relabel_online.sh
  bash $F/assemble_mistral_ladders.sh                               # -> figures/fig_tci_ladder4_mistral_four_sites.png
}

setupfig() { $UVRUN $F/paper_figures/figA0_setup.py; }             # Fig 15 -> paper_figures/figA0_setup.png

paraphrase() {   # Appendix app:paraphrase: the f / lambda rungs under four wording conditions; the canonical baseline is `tci`
  for m in $CAST; do bash $F/run_paraphrase.sh $m $CARD $CARDB; done
  COMPOSE3=1 CONDS="pavg pdisj" bash $F/run_paraphrase.sh gemma-4-31B-it $CARD $CARDB     # optional compose3 rung (31B only)
  bash $F/run_paraphrase.sh rwkv7-g1j-13.3b $CARD
  $UVRUN $F/para_table.py                        # -> results/para_table.md, results/para_table.tex (tab:paraphrase)
  $UVRUN $F/figures/fig_para_ladder.py           # -> figures/fig_para_ladder.png
}

toy() {   # Fig 12, 13, Table 3 (the co-authors' code, its own environment: `cd toy_model && uv sync` once; see toy_model/README.md)
  GPU=$CARD bash toy_model/run_toy.sh train      # 2 models x 3 seeds, lean trainer -> toy_model/mlruns
  GPU=$CARD bash toy_model/run_toy.sh suites     # program_transport.py, 4 and 16 blanks, description masked -> toy_model/results/programming_toy_v2_n24_b{4,16}_p777_MASK
  bash toy_model/run_toy.sh figures              # paper_numbers.py (Table 3 = toy_model/paper_tables/toy_table_belief.tex), paper_figure.py -> toy_model/figures/toy_cfs_b{4,16}.png
}

if [ $# -eq 0 ]; then sed -n 2,26p "$0"; exit 0; fi
for fam in "$@"; do
  if [ "$fam" = all ]; then for f in setup bench sever fig2 pretrained tci grid icl sft variance geometry modifier subsets mistral paraphrase setupfig toy; do say "== $f"; $f; done
  else say "== $fam"; $fam; fi
done

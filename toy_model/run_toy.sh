#!/usr/bin/env bash
# Fig. 12, Fig. 13 and Table 3 from scratch with the co-authors' toy-model code (see README.md in this directory):
#   train    the windowed model (8 layers, window 8, rotary) and the dense control (4 layers), three seeds each,
#            with the lean trainer into the local MLflow store ./mlruns              (~30 min per model on one GPU)
#   suites   the paper's intervention suite on every run: suffix-only position-wise edits, description masked
#            after prefill, 4 and 16 appended blanks, 24 certain recipients, 3 designed tests per trial
#            (results/programming_toy_v2_n24_b{4,16}_p777_MASK/<run name>/step_<step>/summary.json)
#   figures  paper_numbers.py (macros, tables incl. Table 3, Bayesian reference ticks) and paper_figure.py
#            (figures/toy_cfs_b4.{png,pdf} = Fig. 12, figures/toy_cfs_b16.{png,pdf} = Fig. 13)
#
#   bash toy_model/run_toy.sh [train|suites|figures|all]         from the package root (default: all)
#
# Runs in the toy model's own environment (toy_model/pyproject.toml + uv.lock; `uv sync` in this directory first).
# Environment: TOYRUN (python launcher; default `uv run python` in this directory), GPU (card index, default 0),
# SEEDS (default "1 2 3"), and for the smoke test STEPS / SUITE_ARGS / BLANKS (see smoke_test.sh).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
TOYRUN=${TOYRUN:-"uv run python"}
GPU=${GPU:-0}; SEEDS=${SEEDS:-"1 2 3"}; BLANKS=${BLANKS:-"4 16"}
STEPS=${STEPS:-}                                  # smoke: training.num_steps override (the paper: the schedule's 50000)
SUITE_ARGS=${SUITE_ARGS:-"--n-recipients 24 --min-function-certainty 0.95 --recipient-pool 90"}
PROCESS=nonergodic_24x_program_call
MODELS="transformer_d64_ctx64_8L_ln_w8_rope transformer_d64_ctx64_4L_ln_rope"    # windowed, dense control
URI=file:./mlruns
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p logs results figures paper_tables

run_id_of() {   # run_id_of <run name>  -> the MLflow run id in the local store (empty if not found)
  $TOYRUN - "$URI" "$1" <<'PY' 2>/dev/null
import sys, mlflow
c = mlflow.MlflowClient(tracking_uri=sys.argv[1])
e = c.get_experiment_by_name("/Shared/programming_toy_lean")
runs = c.search_runs([e.experiment_id], f"attributes.run_name = '{sys.argv[2]}'", order_by=["attributes.start_time DESC"]) if e else []
print(runs[0].info.run_id if runs else "")
PY
}

train() {
  for MODEL in $MODELS; do for S in $SEEDS; do
    NAME="${PROCESS}_${MODEL}_s${S}"
    if [ -n "$(run_id_of $NAME)" ]; then say "skip training $NAME (run exists in $URI)"; continue; fi
    say "training $NAME on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    $TOYRUN training/train_lean.py mlflow=local generative_process=$PROCESS predictive_model=$MODEL \
      training=schedule_50K_warmup_cosine seed=$S weight_init_seed=$S experiment_name=programming_toy_lean run_name=$NAME \
      hydra.run.dir=results/runs/lean_$NAME ${STEPS:+training.num_steps=$STEPS} > logs/lean_$NAME.log 2>&1
    say "  exit $?; $(grep -h 'train_lean\] done' logs/lean_$NAME.log | tail -1)"
  done; done
}

suites() {
  for MODEL in $MODELS; do for S in $SEEDS; do
    NAME="${PROCESS}_${MODEL}_s${S}"; RUN=$(run_id_of $NAME)
    [ -z "$RUN" ] && { say "no trained run for $NAME; skipped"; continue; }
    for B in $BLANKS; do
      OUT=results/programming_toy_v2_n24_b${B}_p777_MASK
      if ls $OUT/$NAME/step_*/summary.json > /dev/null 2>&1; then say "skip suite b$B for $NAME (summary exists)"; continue; fi
      say "suite b$B for $NAME (run $RUN)"
      CUDA_VISIBLE_DEVICES=$GPU $TOYRUN programming_minimal_model/toy/program_transport.py --run-id $RUN --tracking-uri $URI \
        --T 36 --suffix-blanks $B --edit-positions suffix --sever --sever-mode mask --estimators position --alphas 1,2 --n-probes 3 \
        $SUITE_ARGS --seed 777 --device cuda --batch 256 --out $OUT > logs/suite_b${B}_$NAME.log 2>&1
      say "  exit $?"
    done
  done; done
}

figures() {
  SEEDS="$SEEDS" $TOYRUN programming_minimal_model/toy/analysis/paper_numbers.py || return 1   # paper_tables/: toy_numbers.tex, toy_table*.tex (Table 3 = toy_table_belief.tex), toy_references*.json
  W="windowed attention=$(for S in $SEEDS; do printf "%s," ${PROCESS}_transformer_d64_ctx64_8L_ln_w8_rope_s$S; done | sed 's/,$//')"
  D="dense attention (control)=$(for S in $SEEDS; do printf "%s," ${PROCESS}_transformer_d64_ctx64_4L_ln_rope_s$S; done | sed 's/,$//')"
  REF=$($TOYRUN -c "import json;d=json.load(open('paper_tables/toy_references.json'));print(','.join(f'{k}={v}' for k,v in d.items()))")
  # the manuscript's figure style: Times-like text, legend beside the two panels, compact rows, short axis label
  COMMON=(--no-seed-count --severed-ceiling --severed-style row --prompted-label "target prompt (no intervention)"
          --severed-label "target prompt (description masked)" --drop-control mean --drop-control shuffled --neutral-carriers
          --font stix --legend right --row-height 0.42 --hspace 0.25 --xlabel "fraction of trials where all tests pass")
  for B in $BLANKS; do
    $TOYRUN programming_minimal_model/toy/paper_figure.py --panel "=results/programming_toy_v2_n24_b${B}_p777_MASK" \
      --model "$W" --model "$D" "${COMMON[@]}" --reference "$REF" --out figures/toy_cfs_b${B}.png    # b4 = Fig. 12, b16 = Fig. 13
  done
}

case "${1:-all}" in
  train) train;; suites) suites;; figures) figures || exit 1;;
  all) train; suites; figures || exit 1;;
  *) echo "usage: run_toy.sh [train|suites|figures|all]"; exit 1;;
esac

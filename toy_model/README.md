# Toy model: the minimal programming process (Appendix A.6; Fig. 12, Fig. 13, Table 3)

This directory is the co-authors' own code for the toy model, taken from their repository as it stood at submission and
reduced to what the three toy results need. It has its **own locked environment** (`pyproject.toml`, `uv.lock`,
Python 3.12; JAX, transformer_lens, Hydra, a local MLflow file store, and the authors' generative-process library,
vendored under `third_party/genproc` at the commit the results were produced with), separate from the package's
environment:

```
cd toy_model && uv sync && cd ..          # once
bash toy_model/run_toy.sh                 # train -> suites -> figures (GPU=k picks the card)
```

| path | role |
|---|---|
| `training/matrices.py` | the process: `program_call` (a two-mode language machine gating a function machine: `mess3`, soft `no_repeat` or `sum_mod3`, plus the shared `0 1` watcher and the return), registered into the library's HMM registry on import |
| `training/configs/generative_process/nonergodic_24x_program_call.yaml` | the paper's mixture, 8 languages x 3 functions (written by `programming_minimal_model/toy/gen_program_config.py --n-langs 8`); `nonergodic_12x_...` is the 4-language version the unit tests also build |
| `training/configs/predictive_model/transformer_d64_ctx64_8L_ln_w8_rope.yaml`, `..._4L_ln_rope.yaml` | the windowed model (8 layers, window 8) and the dense control (4 layers), d_model 64, rotary positions |
| `training/configs/training/schedule_50K_warmup_cosine.yaml`, `optimizer/pytorch_adam.yaml` | 50,000 steps, batch 512, AdamW 1e-3, 1,500 warm-up steps, cosine to 1e-6, 44 checkpoints |
| `training/train_lean.py` (+ `optimal_loss.py`, `utils.py`, `configs/`) | the trainer; logs the evaluation loss and its excess over the exact observer's optimum on the same batch |
| `programming_minimal_model/toy/program_transport.py` | the intervention suite: realization-matched donor paths, suffix-only position-wise edits (leave-one-context-out, same-context-only, random, wrong value, mean, random labelling), description masked after prefill, designed tests and the belief readout |
| `programming_minimal_model/toy/program_process.py`, `behavioral_checks.py`, `toy_process.py` | codec, exact observer, readouts, cache hooks and the checkpoint loader the suite imports |
| `programming_minimal_model/toy/analysis/paper_numbers.py`, `observer_floor.py` | every number the manuscript quotes as LaTeX macros, the tables (Table 3 = `toy_table_belief.tex`) and the Bayesian reference ticks, written to `paper_tables/` |
| `programming_minimal_model/toy/paper_figure.py` | the figure (`figures/toy_cfs_b4.png` = Fig. 12, `toy_cfs_b16.png` = Fig. 13, + PDF) |
| `tests/` | invariants of the process (agreement with an independent Kronecker construction, the exact observer's readouts) |
| `run_toy.sh` | the exact commands, in order |

Commands behind `run_toy.sh` (from this directory):

```
uv run python training/train_lean.py mlflow=local generative_process=nonergodic_24x_program_call \
    predictive_model=<transformer_d64_ctx64_8L_ln_w8_rope | transformer_d64_ctx64_4L_ln_rope> \
    training=schedule_50K_warmup_cosine seed=S weight_init_seed=S experiment_name=programming_toy_lean \
    run_name=nonergodic_24x_program_call_<model>_s<S>                                  # S = 1, 2, 3
uv run python programming_minimal_model/toy/program_transport.py --run-id <id> --tracking-uri file:./mlruns \
    --T 36 --suffix-blanks <4 | 16> --edit-positions suffix --sever --sever-mode mask --estimators position \
    --alphas 1,2 --n-probes 3 --n-recipients 24 --min-function-certainty 0.95 --recipient-pool 90 --seed 777 \
    --device cuda --batch 256 --out results/programming_toy_v2_n24_b<B>_p777_MASK
uv run python programming_minimal_model/toy/analysis/paper_numbers.py
uv run python programming_minimal_model/toy/paper_figure.py --panel "=results/programming_toy_v2_n24_b<B>_p777_MASK" \
    --model "windowed attention=<3 run names>" --model "dense attention (control)=<3 run names>" \
    --reference <paper_tables/toy_references.json> <style flags in run_toy.sh> --out figures/toy_cfs_b<B>.png
```

Runtime on one H100: about half an hour per training run, roughly an hour per suite (6 runs x 2 blank lengths).
Outputs (`mlruns/`, `results/`, `logs/`, `figures/`, `paper_tables/`) are created by the runs.

Changes from the authors' version: MLflow defaults point at the local file store `./mlruns` (instead of a hosted
workspace); `observer_floor.py` builds the process from the config file instead of loading a stored run;
`paper_numbers.py` writes to `paper_tables/`, reads the belief readout from the masked suites, and skips (with a
warning) the appendix tables and training macros whose inputs are not produced here; two tests of a fourth
generator used only in an unpublished fine-tuning experiment are removed. Fig. 11 and Fig. 14 are hand-drawn.

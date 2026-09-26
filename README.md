# *LLMs program using reusable and composable abstract representations* — code

Everything needed to regenerate the paper's figures and tables. Nothing is pre-computed: the commands below
download the models, run the experiments and render the figures.

## Setup

```
uv sync                       # locked environment (Python 3.10)
source env.sh                 # shared prompt suffix, toolchain path, HF cache
export HF_TOKEN=hf_...        # google/gemma-* models are gated
bash make_figures.sh setup    # checks the execution grader on every language
```

Generated programs are graded by running them, so compilers/runtimes for Python, JavaScript, C, C++, Java, Go,
Rust, Ruby, OCaml and Haskell must be installed. `node`, `gcc`/`g++`, `ruby`, `python3` come from `PATH`; Java,
Go, Rust, OCaml and GHC are looked up under `$FPL_TOOLCHAINS` (default `/opt/toolchains`, see
`factored_programming_languages/execute.py`). `setup` fails if any language is missing.

The paper's runs used 8x H100 (80 GB).

## Reproducing the figures

```
bash make_figures.sh <target> [<target> ...]     # or: bash make_figures.sh all
```

| target | figures / tables | approx. GPU-h |
|---|---|---|
| `bench`, `sever`, `fig2` | Fig. 2, 16 | 120 |
| `pretrained` | Fig. 19 (needs `sever`) | 4 |
| `tci` | Fig. 3 | 40 per model |
| `grid` | Fig. 5, 6, 7 | 60 |
| `icl` | Fig. 4a, Table 1 | 8 per model |
| `sft` | Fig. 4b, 18 | 180 |
| `variance` | Fig. 8 | 1 per model |
| `geometry` | Fig. 9 | 20 |
| `modifier` | Fig. 10 | 25 per model |
| `subsets` | Fig. 17 | 15 |
| `mistral` | Fig. 20 | 60 |
| `setupfig` | Fig. 15 | – |
| `paraphrase` | paraphrase appendix (needs `tci`) | ≈ `tci` |
| `toy` | Fig. 12, 13, Table 3 | ~1 per run |

The toy model (`toy_model/`) has its own environment: run `uv sync` in that directory first, then
`bash toy_model/run_toy.sh` (or the `toy` target). Figures are written to `factored_programming_languages/figures/`,
`factored_programming_languages/paper_figures/`, `icl_section/figures/` and `toy_model/figures/`.

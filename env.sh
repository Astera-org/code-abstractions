# Environment for the reproducibility package. `source env.sh` from the package root.
# Every value can be overridden before sourcing.

# The fixed formatting instruction shared by every prompt (the "tail" / shared suffix sigma of Eq. 4).
export FPL_TAIL="${FPL_TAIL:-Name the function solve. Return only the function, with no explanation, no examples, and no tests. Do not print anything and do not read any input. Return the result from the function rather than displaying it.}"

# Root of the language toolchains used for execution-based grading (see README.md, "Execution toolchains").
export FPL_TOOLCHAINS="${FPL_TOOLCHAINS:-/opt/toolchains}"

# Hugging Face cache and token (gated google/gemma-* repositories need a token that accepted the licence).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# export HF_TOKEN=hf_...

# Allocator setting used for every large-model run in the paper.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# The package root, for scripts that need it.
export REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

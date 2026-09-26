#!/usr/bin/env bash
# Assemble the padded cast sets into 9M2 (assemble_9F.py: transport rungs from the whole-spec-mask runs, install /
# compose rungs from the words-intact 9U1 runs) and, with --relabel-tag 9F, the nulls of run_cast_ladder.sh; render the main cast ladder, figures/fig_tci_ladder4_9M2_5m (+ _sevnorm).
#   bash factored_programming_languages/assemble_cast_ladder.sh
# Smoke: CAST="pythia-70m" MODELS5="pythia-70m=pythia-70m (smoke)" RWKV_SET="" bash .../assemble_cast_ladder.sh
set -u
UVRUN=${UVRUN:-"uv run"}
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # the reproducibility root
F=factored_programming_languages; R=$F/results; mkdir -p $R/_logs
CAST=${CAST:-"gemma-4-31B-it gemma-4-26b-a4b qwen3.6-27b pythia-6.9b-tulu"}
for M in $CAST; do
  case "$M" in qwen*) TB=bmasknative;; *) TB=bmask;; esac
  $UVRUN $F/assemble_9F.py $M --transport-suffix $TB --transport-prefix tcispan9w --install-from 9U1 --set 9M2 --force | tail -1
done
MODELS5=${MODELS5:-"gemma-4-31B-it=gemma-4-31B,gemma-4-26b-a4b=gemma-4-26B-A4B,qwen3.6-27b=Qwen3.6-27B,pythia-6.9b-tulu=pythia-6.9b (control),rwkv7-g1j-13.3b=RWKV7-13.3B"}
RWKV_SET=${RWKV_SET-"--model-set rwkv7-g1j-13.3b=9Rpl"}   # the RWKV column from its words-intact set 9Rpl (run_rwkv_tci.py --set 9Rpl)
$UVRUN $F/figures/fig_tci_ladder4.py --set 9M2 --relabel-tag 9F --models "$MODELS5" $RWKV_SET > $R/_logs/fig_9M2_5m_raw.txt 2>&1
$UVRUN $F/figures/fig_tci_ladder4.py --set 9M2 --sev --relabel-tag 9F --models "$MODELS5" $RWKV_SET > $R/_logs/fig_9M2_5m_sev.txt 2>&1
echo "figures -> $F/figures/fig_tci_ladder4_9M2_5m.png (+ _sevnorm); numbers in $R/_logs/fig_9M2_5m_raw.txt / _sev.txt"

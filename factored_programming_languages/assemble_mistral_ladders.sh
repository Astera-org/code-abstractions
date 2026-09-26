#!/usr/bin/env bash
# Assemble and render the Mistral-3.2 suffix-vs-online ladders (authors, 2026-09-25).
#   sets: 9M2 = transport under the whole-spec attention mask (the paper's regime), 9N2 = transport with the words intact;
#   install/compose rungs are the unsevered 9U1 runs in both. Tags: mistral-small-3.2-24b (suffix site), -onl (online site).
#   Random-labelling seeds (tag 9U2) are symlinked under both set names for every control kind, so the figure finds them.
set -u
UVRUN=${UVRUN:-"uv run"}   # how to run python with the locked environment (`uv run`, or e.g. /path/to/.venv/bin/python)
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # the reproducibility root
F=factored_programming_languages; R=$F/results; M=mistral-small-3.2-24b
for TAG in $M $M-onl $M-vec; do
  $UVRUN $F/assemble_9F.py $TAG --transport-suffix bmask --transport-prefix tcispan9w --install-from 9U1 --set 9M2 --force | tail -1
  $UVRUN $F/assemble_9F.py $TAG --transport-suffix bnone --transport-prefix tcispan9w --install-from 9U1 --set 9N2 --force | tail -1
  for d in $R/tcirel*9U2_${TAG}_s*; do [ -d "$d" ] || continue; b=$(basename $d)
    for SET in 9M2 9N2; do t=$R/${b/9U2/$SET}; [ -e "$t" ] || ln -s $b $t; done; done
  echo "$TAG: $(ls -d $R/tcirel*9M2_${TAG}_s* 2>/dev/null | wc -l) relabel seed dirs under 9M2"
done
OUT=$F/figures
# masked-suffix alias: the same 9M2 sets under a second model key, so one figure can hold both suffix regimes as columns
for d in $R/tci9M2_$M $R/tcim9M2_$M $R/tcis9M2_$M $R/tcic3m9M2_$M $R/tcirel*9M2_${M}_s*; do [ -e "$d" ] || continue
  b=$(basename $d); t=$R/${b/_$M/_$M-mask}; [ -e "$t" ] || ln -s $b $t; done
# the four-site figure (authors, 2026-09-25): suffix, suffix + masking, online dynamic, online static; words intact except column 2
# all-cells denominators are the main figure (authors, 2026-09-25: "the more honest figure"); the addressable-set version is the _sevnorm variant
MODELS4="$M=Mistral-3.2 suffix,$M-mask=Mistral-3.2 suffix + masking,$M-onl=Mistral-3.2 online (dynamic),$M-vec=Mistral-3.2 online (static)"
$UVRUN $F/figures/fig_tci_ladder4.py --set 9N2 --equal --out fig_tci_ladder4_mistral_four_sites --no-set-label --model-set $M-mask=9M2 --models "$MODELS4" > $R/_logs/fig_mistral_four_sites.txt 2>&1
$UVRUN $F/figures/fig_tci_ladder4.py --set 9N2 --sev --equal --out fig_tci_ladder4_mistral_four_sites_sevnorm --no-set-label --model-set $M-mask=9M2 --models "$MODELS4" > $R/_logs/fig_mistral_four_sites_sevnorm.txt 2>&1
$UVRUN $F/figures/fig_tci_ladder4.py --set 9M2 --sev --equal --out fig_tci_ladder4_9M2_mistral_online \
    --models "gemma-4-31B-it=gemma-4-31B,$M=Mistral-3.2 suffix,$M-onl=Mistral-3.2 online" > $R/_logs/fig_9M2_mistral_online.txt 2>&1
$UVRUN $F/figures/fig_tci_ladder4.py --set 9N2 --sev --equal --out fig_tci_ladder4_9N2_mistral_online \
    --models "$M=Mistral-3.2 suffix,$M-onl=Mistral-3.2 online" > $R/_logs/fig_9N2_mistral_online.txt 2>&1
echo "figures -> $OUT/fig_tci_ladder4_mistral_four_sites.png, fig_tci_ladder4_9M2_mistral_online.png, fig_tci_ladder4_9N2_mistral_online.png; numbers in $R/_logs/fig_9?2_mistral_online.txt"

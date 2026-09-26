"""The cross-language installation figure (fig_install.py) as an ICLR-style TeX table, from the same recounted numbers
(results/fig_install_numbers.json). Two variants:
    results/table_install.tex        one row per model (the compact figure)
    results/table_install_split.tex  regular / nonce sub-columns for the two treatment conditions (the split figure)
    uv run icl_section/table_install.py
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = json.loads((HERE / "results" / "fig_install_numbers.json").read_text())
ROWS = [("gemma-4-31B-it", "Gemma-4-31B"), ("gemma-4-26b-a4b", "Gemma-4-26B-A4B"), ("Qwen3.6-27B", "Qwen3.6-27B"), ("RWKV7-13.3B", "RWKV-7 13B")]
if "--models" in sys.argv:                     # --models key=label,... overrides the rows (smoke test)
    ROWS = [tuple(x.split("=", 1)) for x in sys.argv[sys.argv.index("--models") + 1].split(",")]
frac = lambda hn: f"{hn[0]}/{hn[1]}"
pct = lambda hn: f"{frac(hn)} ({100 * hn[0] / hn[1]:.0f}\\%)" if hn[1] else "-"
class _Arms(dict):                      # a missing arm (no addressable cell) reads as 0/0
    def __getitem__(self, k): return self.get(k, [0, 0])
NUM = {m: {k: (_Arms(v) if isinstance(v, dict) and k in ("all", "invented", "nonce") else v) for k, v in d.items()} for m, d in NUM.items()}

CAPTION = (r"Cross-language installation. A state compiled by one example program in language $x$ (Gemma: the 39 K/V rows of the "
           r"name-free ask; Qwen: the rows and the recurrent state; RWKV: the recurrent state) is written into a prompt with no example, "
           r"and the model is asked for language $L \neq x$. \emph{single donor}: one donor's state, over all admissible donors "
           r"(Stage~A carry hits) $\times$ targets; \emph{mean over donors}: the mean state of the function's admissible donors, per target; "
           r"\emph{no example}: the ask alone; \emph{random edit}: the no-example state plus noise with the norm of the mean-donor edit. "
           r"Success = the program executes the function on the held-out battery, graded in $L$. Two regular algorithms and two nonces "
           r"per model; the last three columns count (function, target) cells with at least one admissible donor (38 where two nonce cells have none).")

def table(caption, label, header, rows, spec):
    return "\n".join([r"\begin{table}[t]", r"\caption{" + caption + "}", r"\label{" + label + "}", r"\begin{center}",
                      r"\small\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{" + spec + "}", r"\toprule", *header, r"\midrule",
                      *[" & ".join(r) + r" \\" for r in rows], r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{table}", ""])

# ---- aggregate: one row per model -------------------------------------------------------------------------------------------
hdr = [r"\multicolumn{1}{c}{\bf model} & no example & random edit & mean over donors & single donor \\"]
rows = [[lab, frac(c["none"]), frac(c["rand"]), pct(c["pooled"]), pct(c["single"])] for key, lab in ROWS for c in [NUM[key]["all"]]]
(HERE / "results" / "table_install.tex").write_text(table(CAPTION, "tab:install", hdr, rows, "l cc cc"))

# ---- split: regular / nonce sub-columns for the treatments ----------------------------------------------------------------
cap2 = CAPTION.replace("Two regular algorithms and two nonces per model;", "Two regular algorithms and two nonces per model, "
                       "shown separately for the treatments (the controls are 0 in every sub-cell);")
hdr2 = [r"& \multicolumn{2}{c}{\textsc{controls}} & \multicolumn{2}{c}{\textsc{mean over donors}} & \multicolumn{2}{c}{\textsc{single donor}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"\multicolumn{1}{c}{\bf model} & no example & random edit & regular & nonce & regular & nonce \\"]
rows2 = [[lab, frac(NUM[key]["all"]["none"]), frac(NUM[key]["all"]["rand"]), frac(NUM[key]["invented"]["pooled"]), frac(NUM[key]["nonce"]["pooled"]),
          pct(NUM[key]["invented"]["single"]), pct(NUM[key]["nonce"]["single"])] for key, lab in ROWS]
(HERE / "results" / "table_install_split.tex").write_text(table(cap2, "tab:install_split", hdr2, rows2, "l cc cc cc"))
print("wrote results/table_install.tex, results/table_install_split.tex")

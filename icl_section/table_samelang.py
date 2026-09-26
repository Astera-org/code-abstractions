"""Same-language elicitation, 'demo match' table: how close is the generated program to the example it came from, when the
example is VISIBLE (prompted), SEVERED after prefill (prompted + masked), or STEERED (its compiled rows installed into a
demo-free prompt)? Per model (and per function with --by-func). Identity = same text modulo whitespace; similarity = difflib
ratio on whitespace-normalised text; correct = executes the function. Sources: results/copy_control[_model].json
(full = visible, carry = severed), results/verbatim_recall[_model].json (install = steered), results/samelang_rwkv7.json.
    uv run icl_section/table_samelang.py [--by-func]"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; R = HERE / "results"; BYF = "--by-func" in sys.argv; TEX = "--tex" in sys.argv
MODELS = [("gemma-4-31B-it", "Gemma-4-31B", "copy_control.json", "verbatim_recall.json"), ("gemma-4-26b-a4b", "Gemma-4-26B-A4B", "copy_control_gemma-4-26b-a4b.json", "verbatim_recall_gemma-4-26b-a4b.json"),
          ("qwen3.6-27b", "Qwen3.6-27B", "copy_control_qwen3.6-27b.json", "verbatim_recall_qwen3.6-27b.json"), ("rwkv7", "RWKV-7 13B", "samelang_rwkv7.json", None)]
if "--models" in sys.argv:                     # --models key=label=copy_control.json=verbatim_recall.json,... (smoke test)
    MODELS = [tuple(x.split("=", 3)) for x in sys.argv[sys.argv.index("--models") + 1].split(",")]
# the four functions to tabulate: default = the balanced pick (2026-09-24: same four on every model); `--funcs a b c d` overrides
FUNCS = sys.argv[sys.argv.index("--funcs") + 1: sys.argv.index("--funcs") + 5] if "--funcs" in sys.argv else ["thulep", "vantril", "kanex", "faromin"]


def load(m, cc, vr):
    cells = {}                                                           # (func, lang) -> {regime: record}
    if not (R / cc).exists(): return cells
    C = json.loads((R / cc).read_text())
    for k, v in C.items():
        f, L, arm = k.split("|"); reg = {"full": "visible", "carry": "severed", "visible": "visible", "install": "steered"}[arm]
        cells.setdefault((f, L), {})[reg] = v
    if vr and (R / vr).exists():
        for k, v in json.loads((R / vr).read_text()).items():
            f, L = k.split("|"); cells.setdefault((f, L), {})["steered"] = v
    return cells


import difflib, re
DON = json.loads((HERE / "donors.json").read_text()); norm = lambda c: re.sub(r"\s+", " ", (c or "").strip())


def rescore(f, L, r):
    """name-agnostic: the ask says 'Name the function solve', the example is named after the function, so identity is
    judged after renaming the example's function to solve; similarity likewise (stored outputs are cut at 500-600 chars, so
    long programs can only match up to that cut)"""
    donor = norm(DON[f]["impls"][L][0].replace(f, "solve")); out = norm(r["code"])
    cut = len(r["code"]) >= 500; d = donor[:len(out)] if cut else donor
    return {"identical": out == d, "similarity": difflib.SequenceMatcher(None, out, d).ratio(), "hit": r["hit"]}


def summarize(recs):
    n = len(recs); return (sum(r["identical"] for r in recs), n, sum(r["similarity"] for r in recs) / n, sum(r["hit"] for r in recs)) if n else None


rows = []
for key, name, cc, vr in MODELS:
    cells = load(key, cc, vr)
    if not cells: rows.append((name, None)); continue
    cells = {k: v for k, v in cells.items() if k[0] in FUNCS}          # the result files may hold more functions than the pick
    groups = [(f, {k: v for k, v in cells.items() if k[0] == f}) for f in FUNCS] if BYF else [("all four", cells)]
    for g, cs in groups:
        rows.append((f"{name} · {g}" if BYF else name, {reg: summarize([rescore(k[0], k[1], c[reg]) for k, c in cs.items() if reg in c]) for reg in ("visible", "severed", "steered")}))
if TEX:                                                               # ICLR style: caption ABOVE the table, centred, booktabs
    def tc(s): return "--- & ---" if s is None else f"{s[0]}/{s[1]} & {s[3]}/{s[1]}"
    print(r"""\begin{table}[t]
\caption{Same-language elicitation. The example program in language $L$ is shown and the model is asked for $L$, with the example \emph{visible}, \emph{severed} after prefill (masked from the fence and every generated token), or \emph{installed} (the compiled state written into a prompt with no example; Gemma: the 39 K/V rows, Qwen: rows and recurrent state, RWKV: recurrent state). \emph{ident.}: generations identical to the example (modulo whitespace, after renaming its function to \texttt{solve} as the ask requires); \emph{correct}: executes the function on the held-out battery. Two regular algorithms and two nonces $\times$ ten languages per model. RWKV has no attention to sever.}
\label{tab:samelang}
\begin{center}
\small\setlength{\tabcolsep}{5pt}
\begin{tabular}{l cc cc cc}
\toprule
& \multicolumn{2}{c}{\textsc{visible}} & \multicolumn{2}{c}{\textsc{severed}} & \multicolumn{2}{c}{\textsc{installed}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\multicolumn{1}{c}{\bf model} & ident. & correct & ident. & correct & ident. & correct \\
\midrule""")
    for name, d in rows:
        if d is None: continue
        print(f"{name} & {tc(d['visible'])} & {tc(d['severed'])} & {tc(d['steered'])} " + r"\\")
    print(r"""\bottomrule
\end{tabular}
\end{center}
\end{table}""")
else:
    print("| model | visible: identical / mean sim / correct | severed: identical / mean sim / correct | steered: identical / mean sim / correct |")
    print("|---|---|---|---|")
    for name, d in rows:
        if d is None: print(f"| {name} | running | running | running |"); continue
        cell = lambda s: "—" if s is None else f"{s[0]}/{s[1]} · {s[2]:.2f} · {s[3]}/{s[1]}"
        print(f"| {name} | {cell(d['visible'])} | {cell(d['severed'])} | {cell(d['steered'])} |")

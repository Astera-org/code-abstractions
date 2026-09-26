"""Paraphrase-robustness tables for the function/language ladder rungs (app:paraphrase): per model, the published padded
baseline (run_cast_ladder.sh / run_rwkv_tci.py --set 9Rpl) and the four wording conditions (run_paraphrase.sh). Cell = pooled-donor (delta) successes / cells so far; the
severed-ask reference and the largest control (none, random) are separate columns. Hybrid: the kv+recurrent channel.
Partial runs are shown with their current n. Writes results/para_table.md and results/para_table.tex.
    uv run factored_programming_languages/para_table.py
"""
import glob, json
from collections import defaultdict
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
MODELS = [("gemma-4-31B-it", "Gemma-4-31B"), ("gemma-4-26b-a4b", "Gemma-4-26B-A4B"), ("qwen3.6-27b", "Qwen3.6-27B"),
          ("rwkv7-g1j-13.3b", "RWKV-7 13B"), ("pythia-6.9b-tulu", "Pythia-6.9B")]
CONDS = [("paper", "published (canonical wording)"), ("pavg", "donors: all 4 wordings"), ("pdisj", "donors: 3 paraphrases only"),
         ("rpara", "recipient paraphrased"), ("both", "donors all 4, recipient paraphrased")]
RUNGS = [("transport", "F"), ("transport", "L"), ("install", "F"), ("install", "L"), ("compose", "F")]
RLAB = {("transport", "F"): "transport $f$", ("transport", "L"): "transport $\\lambda$", ("install", "F"): "install $f$",
        ("install", "L"): "install $\\lambda$", ("compose", "F"): "compose $f,\\lambda$"}


def dirs(m, c):
    if m.startswith("rwkv"):
        return [R / (f"tci9Rpl_{m}" if c == "paper" else f"tcipara{c}Rpl_{m}")]
    if c == "paper":
        t = "bmasknative" if m.startswith("qwen") else "bmask"
        return [R / f"tcispan9w_f_{t}_{m}", R / f"tcispan9w_l_{t}_{m}", R / f"tci9U1_{m}"]
    return [R / f"tcipara_{c}_{p}_{m}" for p in ("f", "l", "ic")]


def load(m, c):
    by = defaultdict(lambda: defaultdict(list))            # (op, factor) -> (channel, arm) -> hits
    for d in dirs(m, c):
        for f in d.glob("c_*.json"):
            r = json.loads(f.read_text())
            fac = r.get("factor") or "F"
            if r["arm"] in ("prompted", "prompted_sev"):
                fac = "F"
            by[(r["op"], fac)][(r.get("channel", "kv"), r["arm"])].append(bool(r["hit"]))
    return by


def pick(g, arm):
    for ch in ("kvrec", "kv"):
        if (ch, arm) in g:
            return g[(ch, arm)]
    return None


def frac(v):
    return f"{sum(v)}/{len(v)}" if v else "--"


def main():
    rows = []
    for m, ml in MODELS:
        for c, cl in CONDS:
            by = load(m, c)
            if not by:
                continue
            cells = [frac(pick(by[k], "delta")) if k in by else "--" for k in RUNGS]
            ref = pick(by[("transport", "F")], "prompted_sev") or pick(by[("transport", "F")], "prompted")
            ctrl = [x for k in RUNGS if k in by for arm in ("none", "rand") for x in [pick(by[k], arm)] if x]
            cmax = max((sum(v) / len(v) for v in ctrl), default=None)
            rows.append((ml, cl, cells, frac(ref), "--" if cmax is None else f"{cmax:.2f}"))

    hdr = ["model", "wording"] + [RLAB[k].replace("$", "").replace("\\lambda", "λ").replace("f,λ", "f+λ") for k in RUNGS] + ["severed ask (transport f)", "max control"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    last = None
    for ml, cl, cells, ref, cm in rows:
        md.append(f"| {ml if ml != last else ''} | {cl} | " + " | ".join(cells) + f" | {ref} | {cm} |"); last = ml
    (R / "para_table.md").write_text("\n".join(md) + "\n")
    tex = [r"\begin{table}[t]",
           r"\caption{Robustness of the function and language rungs to paraphrased descriptions. Each function is described in four "
           r"wordings (the original and three paraphrases). \emph{donors: all 4 wordings}: the pooled donor state is averaged over "
           r"all wordings; \emph{3 paraphrases only}: donors never use the recipient's wording; \emph{recipient paraphrased}: the "
           r"recipient (and its prompted asks) uses a paraphrase. Cells: successes of the pooled (leave-one-out) donor state over "
           r"cells; the severed ask is the recipient asked directly with its description masked; controls are no intervention and a "
           r"norm-matched random edit (largest rate shown).}",
           r"\label{tab:paraphrase}", r"\begin{center}", r"\small\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{l l " + "r " * len(RUNGS) + "r r}", r"\toprule",
           r"model & donor / recipient wording & " + " & ".join(RLAB[k] for k in RUNGS) + r" & severed ask & control \\", r"\midrule"]
    last = None
    for ml, cl, cells, ref, cm in rows:
        if last is not None and ml != last:
            tex.append(r"\midrule")
        tex.append(f"{ml if ml != last else ''} & {cl} & " + " & ".join(cells) + f" & {ref} & {cm} \\\\"); last = ml
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{table}", ""]
    (R / "para_table.tex").write_text("\n".join(tex))
    print("\n".join(md))


if __name__ == "__main__":
    main()

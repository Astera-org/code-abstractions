"""Which demands the store loses, clause by clause.

For each of the eleven demands, the rate at which it is visibly obeyed when
requested: with the words present, and with the description ablated. The bar
between them is what severing costs.

The paired design does the controlling. Every intact cell has a severed twin
with the identical subset, clause set and suffix length, so subset size and
suffix are held exactly equal within each pair; no covariates are modelled.
Significance is exact McNemar on the discordant pairs, Holm-corrected across
the eleven clauses -- all eleven are below 0.0001, so no marker distinguishes
them here.

Two things this reports that a bare drop would hide. The FLOOR is the rate at
which the detector fires on generations that never requested the clause; a
demand the model satisfies unprompted would have an inflated rate in both
arms. All eleven floors are at or below 0.01, so that is not happening. And
the intact rate is 1.00 for all eleven, so the model complies perfectly when
it can read the words and every gap below is the ablation.

    uv run factored_programming_languages/figures/fig_clause_difficulty.py
"""
import json
import re
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tasks_strategy import detect_strategy                             # noqa: E402

import sys as _argv_sys
MODEL = _argv_sys.argv[_argv_sys.argv.index("--model") + 1] if "--model" in _argv_sys.argv else "gemma-4-31B-it"   # --model KEY overrides
R = HERE.parent / "results" / f"subsets10_{MODEL}"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Liberation Serif",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
INK, INK2, GRID, BASELINE = "#111111", "#444444", "#e6e6e6", "#b8b8b8"
C_INT, C_SEV, C_BAR = "#1baf7a", "#d1495b", "#c9c6bd"

CHECK = {
    "exponent": lambda c: bool(re.search(r"\*\*\s*7|pow\s*\([^,]+,\s*7\s*\)", c)),
    "minimum": lambda c: bool(re.search(r"\b23\b", c)),
    "base": lambda c: bool(re.search(r"\bBASE\b\s*=\s*10\b", c)),
    "strategy": lambda c: detect_strategy(c, "python") == "rec",
    "typehint": lambda c: bool(re.search(r"def\s+solve\s*\([^)]*:\s*\w+", c)
                               and re.search(r"->\s*\w+", c)),
    "param": lambda c: bool(re.search(r"def\s+solve\s*\(\s*num\b", c)),
    "docstring": lambda c: bool(re.search(r'def\s+solve[^\n]*\n\s*("""|\'\'\')', c)),
    "helper": lambda c: bool(re.search(r"def\s+digitize\b", c)),
    "guard": lambda c: bool(re.search(r"return\s+-\s*1", c)),
    "name": lambda c: bool(re.search(r"\bzork\b\s*=(?!=)", c)),
    "comment": lambda c: bool(re.search(r"#\s*done\b", c, re.I)),
}
NICE = {"exponent": "each digit to the power 7", "minimum": "cap the result at 23",
        "base": "constant BASE = 10", "strategy": "implement recursively",
        "typehint": "type hints on the signature", "param": "name the parameter num",
        "docstring": "include a docstring", "helper": "inner helper 'digitize'",
        "guard": "return -1 if negative", "name": "variable named 'zork'",
        "comment": "end with '# done'"}

rows = [json.loads(Path(f).read_text()) for f in glob(str(R / "*.json"))]
S = {c: dict(fl=[0, 0], it=[0, 0], sv=[0, 0]) for c in CHECK}
for r in rows:
    code = r["code"] or ""
    for c, fn in CHECK.items():
        hit = fn(code)
        if c in r["clauses"]:
            k = "it" if r["arm"] == "intact" else "sv"
        else:
            k = "fl"
        S[c][k][0] += hit; S[c][k][1] += 1

dat = []
for c, s in S.items():
    if not all(s[k][1] for k in ("fl", "it", "sv")):     # a demand never requested / never absent (sparse smoke data): skip it
        print(f"[skip] {c}: no cells in one arm"); continue
    dat.append((c, s["fl"][0] / s["fl"][1], s["it"][0] / s["it"][1],
                s["sv"][0] / s["sv"][1]))
dat.sort(key=lambda d: d[3])                 # hardest to carry at the top

fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=300)
fig.patch.set_facecolor("white")
for i, (c, fl, it, sv) in enumerate(dat):
    ax.plot([sv, it], [i, i], color=C_BAR, lw=3.0, solid_capstyle="round",
            zorder=2)
    ax.scatter([it], [i], s=26, color=C_INT, zorder=4, linewidths=0.5,
               edgecolors="white")
    ax.scatter([sv], [i], s=26, color=C_SEV, zorder=4, linewidths=0.5,
               edgecolors="white")
    if fl > 0.005:
        ax.scatter([fl], [i], s=12, marker="|", color=INK2, zorder=3)
    ax.annotate(f"{sv:.2f}", (sv, i), fontsize=5.6, color=INK2,
                xytext=(-6, 0), textcoords="offset points", ha="right",
                va="center")

ax.set_yticks(range(len(dat)))
ax.set_yticklabels([NICE[c] for c, _, _, _ in dat], fontsize=6.6, color=INK)
ax.set_xlim(-0.06, 1.08)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("fraction of prompts where the demand is visibly obeyed",
              fontsize=7.4, color=INK2)
ax.tick_params(labelsize=6.4, colors=INK2, length=2.4, width=0.6)
ax.grid(True, axis="x", color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.set_facecolor("white")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.spines["bottom"].set_linewidth(0.7)

fig.legend(handles=[
    Line2D([], [], marker="o", ls="", color=C_INT, ms=4, label="words present"),
    Line2D([], [], marker="o", ls="", color=C_SEV, ms=4, label="description ablated"),
    Line2D([], [], marker="|", ls="", color=INK2, ms=6,
           label="fires when not requested")],
    fontsize=6.0, frameon=False, ncol=3, loc="lower center",
    bbox_to_anchor=(0.58, -0.015), labelcolor=INK2, handletextpad=0.4,
    columnspacing=1.6)
fig.subplots_adjust(left=0.30, right=0.99, top=0.975, bottom=0.245)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_clause_difficulty.{ext}", facecolor="white")
print("saved -> fig_clause_difficulty")
for c, fl, it, sv in dat:
    print(f"  {NICE[c]:>28}: floor {fl:.2f}  intact {it:.2f}  severed {sv:.2f}")

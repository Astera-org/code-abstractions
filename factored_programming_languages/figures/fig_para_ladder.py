"""Paraphrase robustness of the function/language ladder rungs, in the style of fig_tci_ladder4 (app:paraphrase).

Columns = models; row blocks = rungs (transport f, transport lambda, install f, install lambda, compose f+lambda); inside
each block one line per wording condition: the published canonical-wording run, donors pooled over all four wordings,
donors restricted to the three paraphrases (never the recipient's wording), recipient paraphrased, and both. Per line:
random control (grey), leave-one-out pooled factor (filled), same-context-only factor (open), and the prompted asks as
ticks (prompted, and for transport the ask with the words severed). All-cells denominators, as the main ladder figure.
Partial runs are drawn with their current n (see para_table.py for the counts). Reads the same directories as para_table.py.
    uv run factored_programming_languages/figures/fig_para_ladder.py [--out NAME]
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from para_table import MODELS, CONDS, load, pick                       # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
INK, INK2 = "#111111", "#444444"
BASELINE, GRID = "#b8b8b8", "#e6e6e6"
COL = {"rand": "#8a8a8a", "delta": "#20507e", "exact": "#2a78d6", "prompted": "#a8531f", "prompted_sev": "#d9a066"}
RUNGS = [("transport", "F", r"$f$"), ("transport", "L", r"$\lambda$"), ("install", "F", r"$f$"), ("install", "L", r"$\lambda$"),
         ("compose", "F", r"$f,\lambda$")]
OPS = [("transport", [0, 1]), ("install", [2, 3]), ("compose", [4])]
CLAB = {"paper": "published", "pavg": "donors: 4 wordings", "pdisj": "donors: paraphrases only",
        "rpara": "recipient paraphrased", "both": "both"}
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "fig_para_ladder"

DATA = {(m, c): load(m, c) for m, _ in MODELS for c, _ in CONDS}
NC = len(CONDS)
rows_per = [len(ix) * (NC + 1) for _, ix in OPS]
fig, axes = plt.subplots(len(OPS), len(MODELS), figsize=(7.2, 7.4), dpi=300, squeeze=False,
                         gridspec_kw={"height_ratios": rows_per, "hspace": 0.12, "wspace": 0.06})
fig.patch.set_facecolor("white")
for j, (m, mlab) in enumerate(MODELS):
    for i, (op, ix) in enumerate(OPS):
        ax = axes[i][j]
        ax.set_facecolor("white")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(axis="x", labelsize=5.6, colors=INK2, length=2, labelbottom=(i == len(OPS) - 1))
        ax.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.set_yticks([])
        y = 0
        ylab = []
        for r in ix:
            opn, fac, flab = RUNGS[r]
            y0 = y
            for c, _ in CONDS:
                g = DATA[(m, c)].get((opn, fac), {})
                vals = {a: pick(g, a) for a in ("rand", "delta", "exact")}
                pr = {a: pick(DATA[(m, c)].get((opn, "F"), {}), a) for a in ("prompted", "prompted_sev")}
                if j == 0:
                    ylab.append((y, CLAB[c]))
                if not any(vals.values()):
                    ax.text(0.0, y, "–", fontsize=5.0, color="#bbbbbb", va="center")
                    y += 1
                    continue
                rate = {a: (sum(v) / len(v) if v else None) for a, v in vals.items()}
                pts = [v for v in rate.values() if v is not None]
                ax.plot([min(pts), max(pts)], [y, y], color=BASELINE, lw=0.9, zorder=2, solid_capstyle="round")
                for a, v in pr.items():
                    if v and not (a == "prompted_sev" and opn != "transport"):
                        pv = sum(v) / len(v)
                        ax.plot([pv, pv], [y - 0.3, y + 0.3], color=COL[a], lw=0.9, zorder=3,
                                linestyle=(0, (1.2, 1.0)) if a == "prompted_sev" else (0, (2.4, 1.2)))
                if rate["rand"] is not None:
                    ax.scatter([rate["rand"]], [y], s=10, color=COL["rand"], zorder=4, linewidths=0)
                if rate["exact"] is not None:
                    ax.scatter([rate["exact"]], [y], s=26, facecolor="none", edgecolor=COL["exact"], linewidth=0.8, zorder=5)
                if rate["delta"] is not None:
                    ax.scatter([rate["delta"]], [y], s=12, color=COL["delta"], zorder=6, linewidths=0)
                y += 1
            if j == 0:
                ax.text(-1.02, (y0 + y - 1) / 2, flab, fontsize=7.0, color="#555555", ha="center", va="center",
                        transform=ax.get_yaxis_transform())
            y += 1
        ax.set_ylim(y - 1, -1)
        if j == 0:
            for yy, lab in ylab:
                ax.text(-0.05, yy, lab, fontsize=4.9, color=INK2, ha="right", va="center", transform=ax.get_yaxis_transform())
            ax.text(-1.2, (y - 2) / 2, op, rotation=90, fontsize=7.2, color=INK, ha="center", va="center",
                    transform=ax.get_yaxis_transform())
        if i == 0:
            ax.set_title(mlab, fontsize=6.8, color=INK, pad=4)
fig.text(0.56, 0.058, "fraction of cells where the injected factor is present in code that runs correctly", fontsize=6.4,
         color=INK2, ha="center")
LH = [Line2D([], [], color=COL["rand"], marker="o", ls="none", markersize=2.9, label="random control"),
      Line2D([], [], color=COL["delta"], marker="o", ls="none", markersize=3.1, label="leave-one-out factor"),
      Line2D([], [], color=COL["exact"], marker="o", ls="none", markerfacecolor="none", markeredgewidth=0.8, markersize=4.4,
             label="same-context-only factor"),
      Line2D([], [], color=COL["prompted"], lw=0.9, ls=(0, (2.4, 1.2)), label="prompted (no intervention)"),
      Line2D([], [], color=COL["prompted_sev"], lw=0.9, ls=(0, (1.2, 1.0)), label="prompted, words severed")]
fig.legend(handles=LH, fontsize=5.6, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.56, 0.012),
           handletextpad=0.4, columnspacing=1.2)
fig.subplots_adjust(left=0.23, right=0.965, top=0.965, bottom=0.095)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"{OUT}.{ext}", facecolor="white")
print(f"saved -> figures/{OUT}.png / .pdf")

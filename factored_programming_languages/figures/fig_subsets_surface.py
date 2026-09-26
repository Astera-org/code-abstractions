"""How many demands survive the ablation, against how much store there is.

Each cell is 10 randomly drawn subsets of that size from a pool of eleven
independent demands, so the count on the y-axis is not confounded with WHICH
demands were asked for -- the earlier fixed ladder always added the same
clauses in the same order, and its easiest three happened to come first.

Colour is the fraction of those subsets whose severed generation honored
every demand it was given. The intact arm honors everything in 1221 of 1224
cells, so the whole surface is caused by the ablation.

Sizes 0 and 11 have only one possible subset and are drawn from n = 1.

    uv run factored_programming_languages/figures/fig_subsets_surface.py
"""
import json
from collections import defaultdict
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                  # noqa: E402

HERE = Path(__file__).resolve().parent
import sys as _argv_sys
MODEL = _argv_sys.argv[_argv_sys.argv.index("--model") + 1] if "--model" in _argv_sys.argv else "gemma-4-31B-it"   # --model KEY overrides
R = HERE.parent / "results" / f"subsets10_{MODEL}"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Liberation Serif",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
INK, INK2, BASELINE = "#111111", "#444444", "#b8b8b8"
# one hue, light to dark: a magnitude, not a category
CMAP = LinearSegmentedColormap.from_list("s", ["#f7f6f2", "#bcd9f2", "#2a78d6",
                                               "#12365f"])

rows = [json.loads(Path(f).read_text()) for f in glob(str(R / "*.json"))]
sev = [r for r in rows if r["arm"] == "severed"]
SUF = sorted({r["suffix_tokens"] for r in sev})
SZ = sorted({r["size"] for r in sev})
D = defaultdict(lambda: [0, 0])
for r in sev:
    d = D[(r["size"], r["suffix_tokens"])]
    d[0] += r["all_honored"]; d[1] += 1

M = [[D[(k, s)][0] / D[(k, s)][1] if D[(k, s)][1] else float("nan")
      for s in SUF] for k in SZ]

fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=300)
fig.patch.set_facecolor("white")
im = ax.imshow(M, cmap=CMAP, vmin=0, vmax=1, aspect="auto", origin="lower")
for i, k in enumerate(SZ):
    for j, s in enumerate(SUF):
        h, n = D[(k, s)]
        if not n:
            continue
        ax.text(j, i, f"{h}/{n}", ha="center", va="center", fontsize=5.0,
                color="white" if M[i][j] > 0.55 else INK2)
ax.set_xticks(range(len(SUF)))
ax.set_xticklabels([str(s) for s in SUF], fontsize=6.2, color=INK2)
ax.set_yticks(range(len(SZ)))
ax.set_yticklabels([str(k) for k in SZ], fontsize=6.2, color=INK2)
ax.set_xlabel("surviving suffix (tokens after the ablated span)",
              fontsize=7.4, color=INK2)
ax.set_ylabel("demands in the prompt", fontsize=7.4, color=INK2)
ax.tick_params(length=2.2, width=0.6, colors=INK2)
for s in ax.spines.values():
    s.set_color(BASELINE); s.set_linewidth(0.6)

cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.015)
cb.set_label("subsets honoring every demand", fontsize=6.6, color=INK2)
cb.ax.tick_params(labelsize=5.8, colors=INK2, length=2)
cb.outline.set_edgecolor(BASELINE)
cb.outline.set_linewidth(0.6)

fig.subplots_adjust(left=0.085, right=0.995, top=0.975, bottom=0.145)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_subsets_surface.{ext}", facecolor="white")
print("saved -> fig_subsets_surface")
it = [r for r in rows if r["arm"] == "intact"]
print(f"  intact ceiling {sum(r['all_honored'] for r in it)}/{len(it)}")

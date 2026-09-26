"""Where in the stack each factor lives -- the fig2 variance-share profile, for this cast.

The 4-factor fxl20 ANOVA with DIRECTION POOLED OUT and the rest renormalised, exactly as
the archive's fig2_it.py does it: every component involving direction is dropped and the
remainder rescaled, which for a balanced design is identical to the 3-factor ANOVA over
cell means pooled across the two directions. Components sum to 1 at every row.

Read it as: the function factor rises to a mid-stack peak and the language factor takes
over late. The marked point is the peak of the combined algorithm+paradigm share.

    uv run factored_programming_languages/figures/fig_variance_depth.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import Patch                                     # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import models as cast                                                    # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE, BASELINE = "#e1e0d9", "#fcfcfb", "#c3c2b7"
# kept identical to the archive's fig2_it.py so the two figures cross-reference
COMPONENTS = ["algorithm", "paradigm", "language", "pair_sum", "residual"]
CCOLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CLABEL = ["function (algorithm)", "strategy (recursion vs iteration)", "language",
          "pairwise interactions", "three-way (residual)"]
NICE = {"gemma-4-31B-it": "gemma-4-31B-it", "qwen3.6-27b": "Qwen3.6-27B",
        "mistral-small-4-119b": "Mistral-Small-4-119B-A6.5B",
        "pythia-6.9b-tulu": "pythia-6.9b-tulu  (control)"}


def reduced(r):
    keep = {"algorithm": r["mains"]["algorithm"],
            "paradigm": r["mains"]["paradigm"],
            "language": r["mains"]["language"],
            "pair_sum": sum(v for k, v in r["pairs"].items() if "direction" not in k),
            "residual": r["threes"]["algorithmxparadigmxlanguage"]}
    tot = sum(keep.values())
    return {k: v / tot for k, v in keep.items()}


data = {}
for m in cast.CAST:
    f = ROOT / "results" / f"{m.dirname}--fxl20" / "analysis" / "factorial_fxl20.json"
    if not f.exists():
        print(f"[skip] {m.key}")
        continue
    j = json.loads(f.read_text())
    red = [reduced(r) for r in j["per_layer"]]
    d = {c: np.array([r[c] for r in red]) for c in COMPONENTS}
    tot = sum(d[c] for c in COMPONENTS)
    assert np.allclose(tot, 1.0, atol=1e-9), m.key
    d["n_layers"] = j["n_rows"] - 1
    try:
        d["band"] = cast.band(m.key)
    except Exception:
        d["band"] = None
    data[m.key] = d
print("models:", list(data))

plt.rcParams["font.family"] = "sans-serif"
fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.4), dpi=300, sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)

for ax, key in zip(axes.ravel(), [m.key for m in cast.CAST if m.key in data]):
    d = data[key]
    L = d["n_layers"]
    x = np.arange(L + 1) / L
    ax.set_facecolor(SURFACE)
    ax.stackplot(x, *[d[c] for c in COMPONENTS], colors=CCOLOR, linewidth=0.35,
                 edgecolor=SURFACE, zorder=2)
    f = d["algorithm"] + d["paradigm"]
    pk = int(np.argmax(f[1:]) + 1)
    ax.scatter([pk / L], [f[pk]], s=17, color=INK, zorder=5,
               edgecolor=SURFACE, linewidth=0.9)
    ax.annotate(f"{f[pk]:.2f}", (pk / L, f[pk]), textcoords="offset points",
                xytext=(5, 5), fontsize=6.6, color=INK, zorder=5)
    ax.set_title(f"{NICE.get(key, key)}   ({L} layers)", fontsize=8.6, color=INK,
                 loc="left", pad=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(colors=INK2, labelsize=7.6, length=3, width=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASELINE)
        ax.spines[sp].set_linewidth(0.8)

for ax in axes[1]:
    ax.set_xlabel("normalised depth", fontsize=8.5, color=INK2)
for ax in axes[:, 0]:
    ax.set_ylabel("share of explained variance", fontsize=8.5, color=INK2)

fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(CCOLOR, CLABEL)],
           fontsize=7.2, frameon=False, ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, -0.005), handlelength=1.3)
fig.suptitle("the function factor peaks mid-stack; language takes over late",
             fontsize=10.5, color=INK, x=0.012, ha="left", y=0.985)
fig.tight_layout(rect=[0, 0.10, 1, 0.965])
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_variance_depth.{ext}", dpi=300, facecolor=SURFACE)
print(f"[saved] {HERE}/fig_variance_depth.png / .pdf")

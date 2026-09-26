"""External validity of the in-house bench: against LiveCodeBench v6.

Our capability axis is a 310-cell in-house suite (31 tasks x 10 languages,
six inputs per cell, all-or-nothing, reasoning off, greedy). This asks whether
it tracks a public benchmark.

Scores supplied externally, one per model, from mixed sources: Google and
Qwen release evals, BenchForge, Zyphra, an EXAONE comparative eval, DataFlow.
Where a source reported both modes, the NON-thinking number is used, because
our bench runs with reasoning off -- that is the matched condition. Three
models (Qwen3-32B, Qwen3-30B-A3B, EXAONE-4.0-32B) also have a reasoning score
30+ points higher; those are recorded in ALT but not drawn, since plotting
both put two marks on one model and invited reading the pair as data.

18 of our 26 models have a verified v6 score. The eight without one sit mostly
at the bottom of our range (including the pythia control at 0.048), so the fit
is unconstrained below about 0.5 on the x-axis.

Spearman is the headline rather than Pearson: the scores come from six
different harnesses, so ranks are more trustworthy than levels.

Points are coloured by the interface taxonomy: the compiler-like models are
the ones that still write the requested function when their description is
ablated (correctness under the norm-matched random ablation, from
results/ablateflow_*). Read from disk rather than hard-coded, so the split
cannot drift from the measurement it is named after.

    uv run factored_programming_languages/figures/fig_bench_lcb.py
"""
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Liberation Serif",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
INK, INK2, MUTED, GRID, BASELINE = "#111111", "#444444", "#8a8a8a", "#e6e6e6", "#b8b8b8"
C_PT, C_ALT, C_FIT = "#2a78d6", "#a8531f", "#8a8a8a"
C_COMP = "#d1495b"          # compiler-like
COMPILER_FLOOR = 0.9        # correctness under abl_rand

OURS = {
    "gemma-4-31B-it": 0.881, "gemma-4-26b-a4b": 0.835, "qwen3.6-27b": 0.832,
    "gemma-4-12b-it": 0.787, "qwen3.6-35b-a3b": 0.768, "gemma-3-27b-it": 0.758,
    "qwen3.8-27b": 0.758, "qwen3-30b-2507": 0.735, "qwen3.5-35b-a3b": 0.723,
    "qwen3-coder-30b": 0.716, "devstral-2-123b": 0.703,
    "glm-4-32b-instruct": 0.703, "gemma-3-12b-it": 0.671,
    "mistral-small-4-119b": 0.671, "qwen3-32b": 0.671, "qwen3-30b-a3b": 0.639,
    "mistral-small-24b-2501": 0.619, "exaone-4-32b": 0.616,
    "mistral-small-3.2-24b": 0.574, "qwen2.5-14b": 0.552,
    "dscoder-v2-lite": 0.513, "ministral-3-8b": 0.513,
    "mistral-nemo-12b": 0.477, "gemma-2-9b-it": 0.406,
    "mistral-7b-v0.3": 0.216, "pythia-6.9b-tulu": 0.048,
}
LCB = {
    "gemma-4-31B-it": 80.0, "gemma-4-26b-a4b": 77.1, "qwen3.6-27b": 83.9,
    "gemma-4-12b-it": 72.0, "qwen3.6-35b-a3b": 80.4, "gemma-3-27b-it": 29.1,
    "qwen3.8-27b": 90.3, "qwen3-30b-2507": 43.2, "qwen3.5-35b-a3b": 74.6,
    "qwen3-coder-30b": 53.2, "devstral-2-123b": 44.8,
    "mistral-small-4-119b": 57.9, "qwen3-32b": 28.0, "qwen3-30b-a3b": 29.0,
    "exaone-4-32b": 43.1, "mistral-small-3.2-24b": 26.9, "qwen2.5-14b": 33.4,
    "dscoder-v2-lite": 25.5,
}
# models whose source also reported a reasoning-mode score: the plotted value
# is the non-thinking one, but the choice moves them a long way
ALT = {"qwen3-32b": 60.1, "qwen3-30b-a3b": 57.4, "exaone-4-32b": 66.7}
# scores the source measured with reasoning ON. Marked because the mode is a
# large effect (the three models in ALT move 30+ points between modes) and
# because these seven turn out to be exactly the compiler-like set -- see the
# printed overlap check below.
THINK = {"gemma-4-31B-it", "gemma-4-26b-a4b", "gemma-4-12b-it",
         "qwen3.8-27b", "qwen3.6-27b", "qwen3.6-35b-a3b", "qwen3.5-35b-a3b"}
LABEL = {
    "gemma-4-31B-it": "gemma-4-31B", "gemma-4-26b-a4b": "gemma-4-26B-A4B",
    "qwen3.6-27b": "Qwen3.6-27B", "gemma-4-12b-it": "gemma-4-12b",
    "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "gemma-3-27b-it": "gemma-3-27b",
    "qwen3.8-27b": "Qwen3.8-27B", "qwen3-30b-2507": "Qwen3-30B-2507",
    "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B", "qwen3-coder-30b": "Qwen3-Coder-30B",
    "devstral-2-123b": "Devstral-2-123B",
    "mistral-small-4-119b": "Mistral-Small-4-119B", "qwen3-32b": "Qwen3-32B",
    "qwen3-30b-a3b": "Qwen3-30B-A3B", "exaone-4-32b": "EXAONE-4.0-32B",
    "mistral-small-3.2-24b": "Mistral-Small-3.2-24B",
    "qwen2.5-14b": "Qwen2.5-14B", "dscoder-v2-lite": "DS-Coder-V2-Lite",
}
# per-model label offsets in points, to keep 18 labels from colliding
OFF = {
    "gemma-4-31B-it": (-8, -7, "right"), "qwen3.6-27b": (5, 3, "left"),
    "qwen3.8-27b": (5, 2, "left"), "qwen3.6-35b-a3b": (5, 3, "left"),
    "gemma-4-26b-a4b": (5, -4, "left"), "gemma-4-12b-it": (5, -3, "left"),
    "qwen3.5-35b-a3b": (-5, 4, "right"), "mistral-small-4-119b": (-5, 4, "right"),
    "qwen3-coder-30b": (5, 2, "left"), "devstral-2-123b": (-6, 4, "right"),
    "qwen3-30b-2507": (5, -4, "left"), "gemma-3-27b-it": (5, -2, "left"),
    "qwen3-32b": (5, -4, "left"), "qwen3-30b-a3b": (-5, -6, "right"),
    "exaone-4-32b": (-5, 4, "right"), "mistral-small-3.2-24b": (5, -5, "left"),
    "qwen2.5-14b": (-5, 4, "right"), "dscoder-v2-lite": (5, 5, "left"),
}


def _bench_rate(m):
    """pass rate from results/bench_<m>/b_*.json (31 x 10 cells) when the run is on disk; else the value recorded in OURS"""
    fs = glob.glob(str(HERE.parent / "results" / f"bench_{m}" / "b_*.json"))
    if len(fs) < 300:
        return OURS.get(m)
    ok = sum(bool(json.loads(Path(f).read_text()).get("hit")) for f in fs)
    return ok / len(fs)


OURS = {m: _bench_rate(m) for m in OURS}


def abl_rand(m):
    """correctness under the norm-matched random description ablation"""
    d = HERE.parent / "results" / f"ablateflow_{m}"
    if not d.is_dir():
        return None
    by = {}
    for f in d.glob("at_*.json"):
        r = json.loads(f.read_text())
        by.setdefault(r["mode"], {})[(r["task"], r["lang"])] = \
            (r.get("impl") == r["task"])
    if "none" not in by or "abl_rand" not in by:
        return None
    cells = [k for k in by["none"] if k in by["abl_rand"]]
    return sum(by["abl_rand"][k] for k in cells) / len(cells) if cells else None


def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    r, i = [0.0] * len(v), 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
            j += 1
        for k in range(i, j + 1):
            r[s[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pear(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


ms = [m for m in OURS if m in LCB]
X = [OURS[m] for m in ms]
Y = [LCB[m] for m in ms]
rho = pear(rank(X), rank(Y))
r = pear(X, Y)

fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# least-squares line, for the eye only
n = len(X); mx = sum(X) / n; my = sum(Y) / n
b = sum((a - mx) * (c - my) for a, c in zip(X, Y)) / sum((a - mx) ** 2 for a in X)
xs = [min(X) - 0.02, max(X) + 0.02]
ax.plot(xs, [my + b * (v - mx) for v in xs], color=C_FIT, lw=0.8,
        ls=(0, (4, 3)), zorder=2)

ABL = {m: abl_rand(m) for m in ms}
COMP = {m for m in ms if (ABL[m] or 0) >= COMPILER_FLOOR}
for m in ms:
    c = C_COMP if m in COMP else C_PT
    star = m in THINK
    ax.scatter([OURS[m]], [LCB[m]], s=95 if star else 26,
               marker="*" if star else "o", color=c,
               linewidths=0.5, edgecolors="white", zorder=6 if star else 5)
    dx, dy, ha = OFF.get(m, (5, 3, "left"))
    ax.annotate(LABEL[m], (OURS[m], LCB[m]), fontsize=5.2, color=INK2,
                xytext=(dx, dy), textcoords="offset points", va="center",
                ha=ha)


ax.set_xlabel("in-house bench pass rate  (310 cells)", fontsize=7.4, color=INK2)
ax.set_ylabel("LiveCodeBench v6  (%)\nstars: measured with reasoning ON", fontsize=7.4, color=INK2,
              linespacing=1.6)
ax.set_xlim(0.46, 0.94)
ax.set_ylim(15, 98)
ax.tick_params(labelsize=6.4, colors=INK2, length=2.4, width=0.6)
ax.grid(True, color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(BASELINE)
    ax.spines[s].set_linewidth(0.7)
ax.text(0.03, 0.95, f"Spearman $\\rho$ = {rho:+.2f}\nPearson $r$ = {r:+.2f}\n"
        f"n = {len(ms)} of 26", transform=ax.transAxes, fontsize=6.6,
        color=INK, va="top", linespacing=1.5)

fig.legend(handles=[
    Line2D([], [], marker="o", ls="", color=C_COMP, ms=4.2,
           label="compiler-like (survives description ablation)"),
    Line2D([], [], marker="o", ls="", color=C_PT, ms=4, label="re-reader"),
    Line2D([], [], marker="*", ls="", color=INK2, ms=7,
           label="LiveCodeBench score measured with reasoning ON")],
    fontsize=6.0, frameon=False, ncol=3, loc="lower center",
    bbox_to_anchor=(0.56, -0.012), labelcolor=INK2, handletextpad=0.4,
    columnspacing=1.8)
fig.subplots_adjust(left=0.125, right=0.995, top=0.97, bottom=0.185)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_bench_lcb.{ext}", facecolor="white")
print(f"saved -> fig_bench_lcb   n={len(ms)}  rho={rho:+.3f}  r={r:+.3f}")
print(f"compiler-like ({len(COMP)}): " + ", ".join(sorted(COMP)))
print(f"reasoning-on   ({len(THINK & set(ms))}): " + ", ".join(sorted(THINK & set(ms))))
print(f"OVERLAP: compiler-like == reasoning-on ? "
      f"{COMP == (THINK & set(ms))}")
off = [m for m in ms if m not in THINK]
xo = [OURS[m] for m in off]; yo = [LCB[m] for m in off]
ro, rhoo = pear(xo, yo), pear(rank(xo), rank(yo))
to = rhoo * math.sqrt((len(off) - 2) / (1 - rhoo ** 2))
print(f"\nREASONING-OFF ONLY (mode-matched to our bench): n={len(off)}  "
      f"rho={rhoo:+.3f}  r={ro:+.3f}  t={to:+.2f}")
print("  " + ", ".join(sorted(off)))
print("abl_rand per model: " + ", ".join(
    f"{m}={ABL[m]:.2f}" if ABL[m] is not None else f"{m}=n/a" for m in sorted(ms, key=lambda k: -(ABL[k] or 0))))
missing = sorted(set(OURS) - set(LCB))
print(f"no v6 score ({len(missing)}): {', '.join(missing)}")

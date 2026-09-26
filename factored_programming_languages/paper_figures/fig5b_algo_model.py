"""Five invented computations taught in Python only, across four models.

Each bar is the nine held-out languages at that model's BEST dose for that
computation -- the learning rate, of 5e-7 / 2e-6 / 8e-6, where the most
languages ran; ties go to the lower rate. Which rate that was is NOT drawn:
it varies by cell, and it belongs in the appendix rather than on the bars.
Running this file prints it per row, and summarize_algos.py prints the whole
ladder each bar was picked from.

Within a bar, one hue per model and a light-to-dark ramp for how far the
taught computation got:

    solid   ran in that language and matched the reference on all eight probes
    mid     the procedure is in the generation, in the target language's own
            shape, but it does not run
    dotted  the procedure is there as a bare Python body instead
    blank   the computation never arrived
    hatch   the checkpoint can no longer write that language AT ALL -- it now
            fails digit_sum, a function it could write before training. Dose
            damage, so a miss there is not evidence about the computation.

The caret is the untouched base model's own score on the same computation
described in plain words with the invented name removed. It is the ceiling the
bar should be read against: bars short of a low caret are the model's limit,
bars short of a full caret are a failure to transfer.

Structural detection and its three controls are algorithm_present_algo.py.

    uv run factored_programming_languages/paper_figures/fig5b_algo_model.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import Patch                                   # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from style import INK, INK2, GRID, BASELINE, label as _label, colour as _colour   # noqa: E402
from algorithm_present_algo import (algorithm_present,                 # noqa: E402
                                    looks_like_python, HELD)
from algos import ALGOS                                                # noqa: E402

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Liberation Serif",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
R = HERE.parent / "results"
MODELS = ["gemma-4-31B-it", "gemma-4-26b-a4b", "qwen3.6-27b",
          "rwkv7-g1j-13.3b", "pythia-6.9b-tulu"]
if "--models" in sys.argv:                     # --models k1,k2,... overrides the row set (smoke test)
    MODELS = sys.argv[sys.argv.index("--models") + 1].split(",")
TAGS = ("teacher", "teacher_hi", "teacher_vhi")
# RWKV-7 is not a style.CAST member; name and hue local to this figure (as in fig5_reuse_bars.py)
EXTRA = {"rwkv7-g1j-13.3b": ("RWKV-7 13B", "#b3261e")}


def label(m):
    return EXTRA[m][0] if m in EXTRA else _label(m)


def colour(m):
    return EXTRA[m][1] if m in EXTRA else _colour(m)


# `--algos a b ...` restricts the rows (e.g. the balanced four of the ICL section) and suffixes the file name _<n>algos
ALGO_LIST = sys.argv[sys.argv.index("--algos") + 1:] if "--algos" in sys.argv else list(ALGOS)
SUFFIX = f"_{len(ALGO_LIST)}algos" if "--algos" in sys.argv else ""
LR = {"teacher": "5e-7", "teacher_hi": "2e-6", "teacher_vhi": "8e-6"}
EMPTY = "#f4f3ee"


def cell(algo, model, tag):
    """-> (ran, arrived-in-target-shape, arrived-as-Python, dead, n) or None"""
    d = R / f"eval{algo}_{model}_{tag}"
    if not d.is_dir():
        return None
    c = dict(ran=0, tgt=0, py=0, dead=0, n=0)
    for l in HELD:
        f = d / f"{algo}_{l}.json"
        if not f.exists():
            continue
        c["n"] += 1
        r = json.loads(f.read_text())
        fr = d / f"renders_{l}.json"
        rend = json.loads(fr.read_text()).get("hit") if fr.exists() else True
        if r.get("hit"):
            c["ran"] += 1
        elif not rend:
            c["dead"] += 1
        elif algorithm_present(algo, r.get("code")):
            c["py" if looks_like_python(r.get("code")) else "tgt"] += 1
    return c if c["n"] else None


def best(algo, model):
    """the dose where the most languages ran; ties to the LOWER dose, which is
    the one that has taken less damage"""
    out = None
    for t in TAGS:
        c = cell(algo, model, t)
        if c and (out is None or c["ran"] > out[1]["ran"]):
            out = (t, c)
    return out


def native(algo, model):
    d = R / f"eval{algo}_{model}_base"
    if not d.is_dir():
        return None
    return sum(json.loads((d / f"native_{l}.json").read_text()).get("hit")
               for l in HELD if (d / f"native_{l}.json").exists())


# rows: 5 algorithms x 4 models, grouped, with a gap between algorithms
ROWS, y = [], 0.0
for algo in ALGO_LIST:
    for m in MODELS:
        b = best(algo, m)
        if b:
            ROWS.append((y, algo, m, b[0], b[1], native(algo, m)))
            y += 1.0
    y += 0.65

fig, ax = plt.subplots(figsize=(6.6, 1.3 + 0.165 * len(ROWS)), dpi=300)   # was 4.6 in for 20 rows
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for y, algo, m, tag, c, nat in ROWS:
    n, col = c["n"], colour(m)
    ax.barh(y, n, height=0.62, color=EMPTY, edgecolor=GRID, lw=0.5, zorder=2)
    if c["dead"]:
        ax.barh(y, c["dead"], left=n - c["dead"], height=0.62,
                facecolor="white", edgecolor=BASELINE, lw=0.5, hatch="///",
                zorder=3)
    ax.barh(y, c["ran"] + c["tgt"] + c["py"], height=0.62, color=col,
            alpha=0.18, edgecolor="none", zorder=4)
    if c["py"]:
        # dots, in the model's own hue, over just this segment. Two pale tints
        # of one hue are hard to tell apart at bar height, and the pale tint
        # also has to be told apart from the empty remainder -- texture does
        # that where a third alpha step would not. Dots, not the diagonal used
        # for dose damage, so the two textures never read as the same thing.
        ax.barh(y, c["py"], left=c["ran"] + c["tgt"], height=0.62,
                facecolor="none", edgecolor=col, hatch="....", lw=0.0,
                zorder=5)
    ax.barh(y, c["ran"] + c["tgt"], height=0.62, color=col, alpha=0.42,
            edgecolor="none", zorder=6)
    ax.barh(y, c["ran"], height=0.62, color=col, zorder=7)
    if nat is not None:
        # a tick spanning the bar's own height, not a marker floating above it:
        # above the bar it reads as belonging to the row before
        ax.plot([nat, nat], [y - 0.36, y + 0.36], color=INK2, lw=1.0,
                solid_capstyle="butt", zorder=9)
    extra = "".join(t for t in (f"+{c['tgt']}" if c["tgt"] else "",
                                f"+{c['py']}p" if c["py"] else "") if t)
    ax.annotate(f"{c['ran']}/{n}{('  ' + extra) if extra else ''}",
                (n, y), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=6.2, color=INK, zorder=8)

ax.set_yticks([r[0] for r in ROWS])
ax.set_yticklabels([label(r[2]) for r in ROWS], fontsize=6.2, color=INK)
ax.tick_params(axis="y", length=0, pad=4)
# the algorithm name gets its OWN column to the left of the model names --
# sharing the column put "vantril" on top of "gemma-4-26B-A4B"
for algo in ALGO_LIST:
    ys = [r[0] for r in ROWS if r[1] == algo]
    ax.text(-0.235, (min(ys) + max(ys)) / 2, algo,
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=7.8, color=INK, fontweight="bold")

ax.set_xlim(0, 9)
ax.set_ylim(max(r[0] for r in ROWS) + 0.75, min(r[0] for r in ROWS) - 0.75)
ax.set_xticks(range(0, 10, 3))
ax.set_xlabel("held-out languages, of nine", fontsize=7.4, color=INK2)
ax.tick_params(axis="x", labelsize=6.4, colors=INK2, length=2.4, width=0.6)
ax.grid(True, axis="x", color=GRID, lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.spines["bottom"].set_linewidth(0.7)

fig.legend(handles=[
    Patch(facecolor=INK2, label="ran correctly"),
    Patch(facecolor=INK2, alpha=0.42, label="arrived in the target language"),
    Patch(facecolor=INK2, alpha=0.18, edgecolor=INK2, hatch="....",
          label="arrived as Python"),
    Patch(facecolor=EMPTY, edgecolor=GRID, label="did not arrive"),
    Patch(facecolor="white", edgecolor=BASELINE, hatch="///",
          label="cannot write this language at all"),
    Line2D([], [], marker="|", ls="", color=INK2, ms=7, mew=1.2,
           label="untouched base, described in plain words")],
    fontsize=6.0, frameon=False, ncol=3, labelcolor=INK2, loc="lower center",
    bbox_to_anchor=(0.60, 0.002), handletextpad=0.5, columnspacing=1.5)
fig.subplots_adjust(left=0.275, right=0.935, top=0.985, bottom=0.76 / fig.get_size_inches()[1])   # margin in inches, not a fraction
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig5b_algo_model{SUFFIX}.{ext}", facecolor="white")
print(f"saved -> paper_figures/fig5b_algo_model{SUFFIX}.png / .pdf")
for y, algo, m, tag, c, nat in ROWS:
    print(f"  {algo:>8} {label(m):>18} {LR[tag]:>6}  ran {c['ran']}/{c['n']}"
          f"  +tgt {c['tgt']}  +py {c['py']}  dead {c['dead']}  native {nat}/9")

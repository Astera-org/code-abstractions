"""Severing the description: pretrained twins vs their instruction-tuned twins.

--instrument abl_mask (default, authors, 2026-09-18) cuts the description's
COLUMNS out of the attention mask during generation, prefill left intact, which
is the instrument the main figures use: nothing is substituted for the rows, so
they take no share of the softmax. --instrument abl_rand is the original, which
replaced those KV rows with norm-matched random vectors. Either way generation
keeps only the rows computed after the description. Instruct twins are measured through their chat template
(results/ablateflow_<m>, the published instrument); pretrained twins through
the completion layout that held their conditioner best and gave them the most
room, `godoc` with the 55-row generic suffix (results/cabl_<m>_godoc_xwide,
completion.py). A cell counts only if the uncut run is correct; the score is
the fraction of those cells still correct under the ablation.

  A  per lineage: instruct (filled) vs pretrained (hollow), the gap drawn.
     Qwen3.5 is a hybrid: KV ablation leaves its recurrent trace of the
     description intact, so its pair is not a severing and is drawn muted.
  B  the pretrained prompt itself (digit_sum, Go): the request comment is the
     span that is cut (masked, or randomised under --instrument abl_rand); the generic doc comment and the opened
     signature are the rows the model is left with; generation starts after
     the brace. The by-task pooling (canonical survives, twists do not) is
     printed to the console and lives in the writeup.

    uv run factored_programming_languages/paper_figures/fig_sever_base_vs_instruct.py
"""
import argparse
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

from style import INK, INK2, MUTED, GRID, BASELINE                     # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from completion import build                                           # noqa: E402
from tasks_function import FDESC                                       # noqa: E402
from matplotlib.patches import FancyBboxPatch                          # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"
_ap = argparse.ArgumentParser()
_ap.add_argument("--instrument", default="abl_mask", choices=["abl_mask", "abl_rand"])
_a, _ = _ap.parse_known_args()
ABL = _a.instrument
# say what was actually done to the description: masked, not severed (authors, 2026-09-18)
CUT, UNCUT = ("masked", "unmasked") if ABL == "abl_mask" else ("severed", "unsevered")
LAYOUT = "godoc_xwide"
PAIRS = [("gemma-4-12b-pt", "gemma-4-12b-it", "gemma-4-12b", False),
         ("gemma-4-26b-a4b-pt", "gemma-4-26b-a4b", "gemma-4-26B-A4B", False),
         ("gemma-4-31B", "gemma-4-31B-it", "gemma-4-31B", False),
         ("qwen3.5-35b-a3b-pt", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B", True)]
if "--pairs" in sys.argv:                      # --pairs pt=it=label[=hybrid],... overrides the lineages (smoke test)
    PAIRS = [(p[0], p[1], p[2], len(p) > 3 and p[3] == "hybrid") for p in (x.split("=") for x in sys.argv[sys.argv.index("--pairs") + 1].split(","))]
TASKS = [("digit_sum", "digit sum"), ("digit_prod", "digit product"),
         ("sq_digit_sum", "sum of squared digits"), ("even_digits", "count of even digits")]
INSTRUCT, PRETRAINED = "#20507e", "#a8531f"        # the compiler/re-reader pair used in fig5


def base_cells(m):
    cells = collections.defaultdict(dict)
    for f in glob.glob(str(R / f"cabl_{m}_{LAYOUT}" / "*.json")):
        j = json.loads(Path(f).read_text())
        cells[(j["task"], j["lang"])][j["mode"]] = j["intact"]
    return {k: (v["none"], v[ABL]) for k, v in cells.items()
            if "none" in v and ABL in v}


def inst_cells(m):
    cells = collections.defaultdict(dict)
    for f in glob.glob(str(R / f"ablateflow_{m}" / "at_*.json")):
        j = json.loads(Path(f).read_text())
        cells[(j["task"], j["lang"])][j["mode"]] = j["impl"] == j["task"]
    return {k: (v["none"], v[ABL]) for k, v in cells.items()
            if "none" in v and ABL in v}


def survival(cells, task=None):
    """(uncut correct, severed correct, cells) over ALL cells of the bank: a
    cell the model cannot do uncut counts as a miss on both, so the gap from
    1 to the unsevered mark is the conditioner and the gap from unsevered to
    severed is the ablation. (No cell is correct severed but wrong uncut.)"""
    ks = [(u, s) for (t, _), (u, s) in cells.items() if task is None or t == task]
    return sum(u for u, _ in ks), sum(s for _, s in ks), len(ks)


data = [(lab, hyb, survival(inst_cells(i)), survival(base_cells(b))) for b, i, lab, hyb in PAIRS]
pooled_i = collections.defaultdict(lambda: [0, 0])
pooled_b = collections.defaultdict(lambda: [0, 0])
for b, i, lab, hyb in PAIRS:
    if hyb:
        continue
    ci, cb = inst_cells(i), base_cells(b)
    for t, _ in TASKS:
        _, k, n = survival(ci, t); pooled_i[t][0] += k; pooled_i[t][1] += n
        _, k, n = survival(cb, t); pooled_b[t][0] += k; pooled_b[t][1] += n

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
W, H = 5.5, 2.0                                   # ICLR text width, as fig5_scatter
fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, H), dpi=300,
                             gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.32})
fig.patch.set_facecolor("white")


def marks(axis, x, pts, muted=False):
    """pts: list of (value, count, n, kind); kinds: inst_sev, pre_unsev, pre_sev"""
    ys = [v for v, _, _, _ in pts]
    axis.plot([x, x], [min(ys), max(ys)], color=BASELINE if muted else INK2, lw=0.9, zorder=2)
    style = {"inst_sev": dict(marker="o", facecolors=MUTED if muted else INSTRUCT,
                              edgecolors="white", linewidths=0.6),
             "pre_unsev": dict(marker="s", facecolors="white",
                               edgecolors=MUTED if muted else PRETRAINED, linewidths=1.2),
             "pre_sev": dict(marker="o", facecolors="white",
                             edgecolors=MUTED if muted else PRETRAINED, linewidths=1.2)}
    for v, _, _, kind in pts:
        axis.scatter([x], [v], s=30, zorder=4, **style[kind])
    # counts beside the marks in ink, one label per distinct height
    col = MUTED if muted else INK2
    byy = {}
    for v, k, n, _ in pts:
        byy.setdefault(round(v, 6), []).append(f"{k}/{n}")
    for v, labs in byy.items():
        axis.text(x + 0.13, v, ", ".join(dict.fromkeys(labs)), fontsize=5.4, color=col,
                  va="center")


for axis in (ax,):
    axis.set_facecolor("white")
    for s in ("top", "right"):
        axis.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axis.spines[s].set_color(BASELINE)
        axis.spines[s].set_linewidth(0.7)
    axis.tick_params(labelsize=6.2, colors=INK2, length=2.5, width=0.6)
    axis.set_ylim(-0.04, 1.08)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(["0", ".25", ".50", ".75", "1"])
    axis.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    axis.set_axisbelow(True)

# ---- A: per lineage --------------------------------------------------------
for x, (lab, hyb, (ui, si, ni), (ub, sb, nb)) in enumerate(data):
    marks(ax, x, [(si / ni, si, ni, "inst_sev"), (ub / nb, ub, nb, "pre_unsev"),
                  (sb / nb, sb, nb, "pre_sev")], muted=hyb)
ax.set_xticks(range(len(data)))
ax.set_xticklabels([d[0] for d in data], fontsize=6.0, rotation=18, ha="right")
ax.set_xlim(-0.5, len(data) - 0.2)
ax.set_ylabel("cells implementing the function", fontsize=6.4, color=INK2)
ax.set_title("(a)", fontsize=7.0, color=INK, loc="left", pad=4)

# ---- B: the pretrained prompt, with the ablated span marked -----------------
bx.set_axis_off()
bx.set_xlim(0, 1)
bx.set_ylim(0, 1)
text, (d0, d1) = build("go", FDESC["digit_sum"], "xwide", "godoc")
head, comment, tail = text[:d0], text[d0:d1], text[d1:]
FS = 4.8


def wrap(line, width=42):
    """display-only reflow at spaces; the prompt itself is one line"""
    out, cur = [], ""
    for w in line.split(" "):
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    return out + [cur]


LH = 0.068
rows = []                                  # (text, kind) kind in desc/suffix
for ln in wrap(comment.rstrip("\n")):
    rows.append((ln, "desc"))
for ln in tail.rstrip("\n").split("\n"):
    for w in wrap(ln):
        rows.append((w, "suffix"))
y0 = 0.86
XT, XR = 0.02, 0.84                        # text left edge; right edge of the span box
ys = {}
for i, (ln, kind) in enumerate(rows):
    y = y0 - i * LH
    ys.setdefault(kind, []).append(y)
    bx.text(XT, y, ln, fontsize=FS, family="monospace", color=INK, va="center")

# the ablated span: a tinted box behind the request comment, labelled above it
dy = ys["desc"]
bx.add_patch(FancyBboxPatch((XT - 0.008, min(dy) - LH * 0.5), XR - XT + 0.016,
                            (max(dy) - min(dy)) + LH,
                            boxstyle="round,pad=0.004,rounding_size=0.008",
                            facecolor="#f5e3d8", edgecolor=PRETRAINED, lw=0.7, zorder=0))
# the kept suffix: a bracket to the right of the text, labelled below the block
sy = ys["suffix"]
xb = XR + 0.05
bx.plot([xb, xb], [min(sy) - LH * 0.45, max(sy) + LH * 0.45], color=INK2, lw=0.9)
for yy in (min(sy) - LH * 0.45, max(sy) + LH * 0.45):
    bx.plot([xb - 0.02, xb], [yy, yy], color=INK2, lw=0.9)
bx.text(xb + 0.025, (min(sy) + max(sy)) / 2, "suffix, kept", fontsize=5.5, color=INK2,
        rotation=90, ha="left", va="center")
bx.set_title("(b)", fontsize=7.0, color=INK, loc="left", pad=4)

handles = [Line2D([], [], marker="o", ls="", ms=5.2, color=INSTRUCT,
                  label=f"instruction-tuned, {CUT}"),
           Line2D([], [], marker="s", ls="", ms=5.0, markerfacecolor="white",
                  markeredgecolor=PRETRAINED, markeredgewidth=1.2, label=f"pretrained, {UNCUT}"),
           Line2D([], [], marker="o", ls="", ms=5.2, markerfacecolor="white",
                  markeredgecolor=PRETRAINED, markeredgewidth=1.2, label=f"pretrained, {CUT}")]
ax.legend(handles=handles, fontsize=5.6, frameon=False, loc="lower right",
          handletextpad=0.3, borderaxespad=0.2, labelspacing=0.35)

out = Path(__file__).parent / ("fig_sever_base_vs_instruct"
                               + ("" if ABL == "abl_rand" else f"_{ABL}"))
fig.savefig(f"{out}.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
fig.savefig(f"{out}.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.02)
print(f"saved -> {out}.png / .pdf")
for lab, hyb, (ui, si, ni), (ub, sb, nb) in data:
    print(f"  {lab:16s} instruct {CUT} {si}/{ni}  pretrained {UNCUT} {ub}/{nb} {CUT} {sb}/{nb}"
          + ("  (hybrid, KV-only)" if hyb else ""))
for t, tlab in TASKS:
    print(f"  {tlab:22s} instruct {pooled_i[t][0]}/{pooled_i[t][1]}  pretrained {pooled_b[t][0]}/{pooled_b[t][1]}")

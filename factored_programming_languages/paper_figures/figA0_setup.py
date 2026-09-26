"""Setup figure, ICLR style: (a) the prompt family, (b) the formatting
text, (c) the position layout (padding prepended). Designed at exactly
\\textwidth = 5.5in so it is included 1:1 and the point sizes are real;
serif (Times-family) for labels to match the paper body, Courier-family
mono for verbatim prompt text.

    uv run factored_programming_languages/paper_figures/figA0_setup.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle                               # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Liberation Serif", "Times New Roman",
                   "DejaVu Serif"],
    "font.monospace": ["Liberation Mono", "Nimbus Mono PS", "Courier New",
                       "DejaVu Sans Mono"],
    "mathtext.fontset": "stix",
})
MONO = dict(family="monospace")

INK, MUTED, GRID = "#111111", "#666666", "#bbbbbb"
C_LANG_F, C_LANG_E = "#dce8f7", "#3a6ea8"
C_FN_F, C_FN_E = "#fbe4d5", "#c06a35"
C_ST_F, C_ST_E = "#eae2f5", "#6f56a0"
C_PAD_F = "#f0f0ec"

W, H = 5.5, 3.62
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")


def chip(x, y, w, txt, fc, ec):
    ax.add_patch(Rectangle((x, y - 0.095), w, 0.20, facecolor=fc,
                           edgecolor=ec, lw=0.7))
    ax.annotate(txt, (x + w / 2, y + 0.004), fontsize=6.2, color=INK,
                ha="center", va="center")


def dash(x, y, w):
    ax.annotate("—", (x + w / 2, y + 0.004), fontsize=6.5, color=MUTED,
                ha="center", va="center")


# ---- (a) the prompt family -------------------------------------------------------------
TOP = H - 0.20
HX0 = 3.58
ax.annotate("(a)", (0.06, TOP), fontsize=9, color=INK, va="center",
            fontweight="bold")
COLX = {"lang": (0.94, 0.44), "fn": (1.46, 0.90), "st": (2.44, 0.92)}
for key, lab in (("lang", "language"), ("fn", "function"), ("st", "strategy")):
    ax.annotate(lab, (COLX[key][0] + COLX[key][1] / 2, TOP),
                fontsize=6.2, color=MUTED, ha="center", va="center",
                style="italic")
ax.annotate("example", (HX0, TOP), fontsize=6.2,
            color=MUTED, va="center", style="italic")

ROWS = [
    ("$\\mathrm{full}(A,\\lambda)$", True, True, False,
     ["Write a function in Go that returns the sum",
      "of the digits of a non-negative integer."]),
    ("$\\mathrm{nofn}(\\lambda)$", True, False, False,
     ["Write a function in Go."]),
    ("$\\mathrm{nolang}(A)$", False, True, False,
     ["Write a function that returns the sum of",
      "the digits of a non-negative integer."]),
    ("$\\mathrm{carrier}$", False, False, False, ["Write a function."]),
    ("$\\mathrm{strat}(A,\\lambda,s)$", True, True, True,
     ["Write a function in Go that returns the sum",
      "of the digits of a non-negative integer.",
      "Use recursion; do not use loops."]),
]
HX = 3.58
y = TOP - 0.34
for name, hl, hf, hs, head in ROWS:
    n = len(head)
    yc = y - 0.0675 * (n - 1)
    ax.annotate(name, (0.06, yc + 0.004), fontsize=7.2, color=INK,
                va="center")
    (chip if hl else dash)(*(COLX["lang"][0], yc, COLX["lang"][1]),
                           *(("Go", C_LANG_F, C_LANG_E) if hl else ()))
    (chip if hf else dash)(*(COLX["fn"][0], yc, COLX["fn"][1]),
                           *(("sum of the digits", C_FN_F, C_FN_E) if hf else ()))
    (chip if hs else dash)(*(COLX["st"][0], yc, COLX["st"][1]),
                           *(("recursion, no loops", C_ST_F, C_ST_E) if hs else ()))
    for j, ln in enumerate(head):
        ax.annotate(ln, (HX, y - 0.135 * j), fontsize=5.2, color="#333333",
                    va="center", **MONO)
    y -= 0.155 + 0.135 * n

# ---- (b) the formatting text -----------------------------------------------------------
y += 0.12
ax.annotate("(b)", (0.06, y), fontsize=9, color=INK, va="center",
            fontweight="bold")
BW, BH = 4.78, 0.26
ax.add_patch(Rectangle((0.30, y - 0.18 - BH), BW, BH, facecolor="#f8f7f4",
                       edgecolor=GRID, lw=0.7))
ax.annotate("Name the function solve. Return only the function, with no "
            "explanation, no examples, and no tests.",
            (0.38, y - 0.18 - BH / 2), fontsize=5.6, color=INK, va="center",
            **MONO)

# ---- the position layout (part of b) ---------------------------------------------------
YBOX = y - 0.18 - BH
SEGS = [
    ("chat markup", 0.62, "#f0f0f0", GRID, False),
    ("padding", 0.72, C_PAD_F, MUTED, False),
    ("description", 1.30, C_FN_F, C_FN_E, False),
    ("formatting instructions", 1.18, C_LANG_F, C_LANG_E, False),
    ("markup", 0.48, "#f0f0f0", GRID, False),
    ("generated tokens \u2026", 0.84, "none", MUTED, True),
]
BY = YBOX - 0.52
x = 0.30
CXF = None
for lab, w, fc, ec, dashed in SEGS:
    ax.add_patch(Rectangle((x, BY - 0.135), w - 0.035, 0.29, facecolor=fc,
                           edgecolor=ec, lw=0.7,
                           linestyle=(0, (3, 2)) if dashed else "solid"))
    ax.annotate(lab, (x + (w - 0.035) / 2, BY + 0.004), fontsize=6.0,
                color=INK if not dashed else MUTED, ha="center", va="center")
    if lab.startswith("formatting"):
        CXF = x + (w - 0.035) / 2
    x += w + (0.05 if lab == "markup" else 0.0)
ax.add_patch(FancyArrowPatch((CXF, YBOX - 0.02), (CXF, BY + 0.17),
                             arrowstyle="-|>", mutation_scale=7,
                             color="#555555", lw=0.7))

fig.savefig(Path(__file__).parent / "figA0_setup.png",
            facecolor="white", bbox_inches="tight", pad_inches=0.01)
fig.savefig(Path(__file__).parent / "figA0_setup.pdf",
            facecolor="white", bbox_inches="tight", pad_inches=0.01)
print("saved -> paper_figures/figA0_setup.png / .pdf")

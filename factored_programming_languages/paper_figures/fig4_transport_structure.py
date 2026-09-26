"""One matched triplet: transport preserves implementation structure.

The minimal argument. A Haskell donor realizes iteration as a fold; a Java
recipient that received the donor-derived state writes a stream reduction;
the same Java recipient, simply prompted for iteration, writes a while loop.
Same function, same language, same coarse strategy in both Java programs --
they differ only in which one received the donor state.

    Haskell syntax does not appear in Java   -> not surface copying
    both Java outputs are iterative          -> not just the strategy label
    the donor's organization survives        -> an implementation abstraction

Every program is a verbatim generation; three fragments are inked and the
rest is faded so the reader compares the constructions, not whole programs.
Long lines are wrapped after a paren and deep continuation indents clipped
(whitespace only). The layout is computed in inches from the text it holds,
so the figure is exactly as wide as its three code columns.

    uv run factored_programming_languages/paper_figures/fig4_transport_structure.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import FancyBboxPatch                          # noqa: E402

from style import INK, INK2, MUTED, GRID, save                         # noqa: E402
from tasks_strategy import detect_strategy                             # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"
import sys as _argv_sys
# --model KEY --cell FN --lang L --want iter|rec --donor-cell FN --donor-lang L override the paper's triplet
def _opt(flag, default):
    return _argv_sys.argv[_argv_sys.argv.index(flag) + 1] if flag in _argv_sys.argv else default
MODEL = _opt("--model", "gemma-4-31B-it")
CELL, LANG, WANT = _opt("--cell", "digit_sum"), _opt("--lang", "java"), _opt("--want", "iter")
DONOR_CELL = _opt("--donor-cell", "digit_prod")          # a held-out function, as the delta uses
DONOR_LANG = _opt("--donor-lang", "haskell")
FADE = "#a8a6a0"

# the three fragments the argument turns on
HL = {"donor": [r"foldl'"],
      "steered": [r"IntStream", r"\.map\(", r"\.sum\(\)"],
      "prompted": [r"\bwhile\b"]}


def rec(path):
    return json.loads((R / path).read_text())


def fit(line, maxw=52):
    """display-only reflow; whitespace is the only thing changed"""
    line = re.sub(r"^ {12,}", " " * 8, line)
    if len(line) <= maxw:
        return [line]
    ind = len(line) - len(line.lstrip())
    cut = line.rfind("(", 0, maxw)
    if cut < 0:
        cut = line.rfind(" ", 0, maxw)
        if cut < 0:
            return [line]
    return [line[:cut + 1], " " * (ind + 4) + line[cut + 1:].lstrip()]


def lines_of(code, cap=7):
    out = []
    for ln in (code or "").splitlines():
        if ln.strip():
            out += fit(ln)
    return out[:cap] or ["(no program)"]          # a cell with no generation (sparse smoke data) still lays out


donor = rec(f"tcish_{MODEL}/c_transport_{DONOR_CELL}_{DONOR_LANG}_{WANT}"
            f"_kv_prompted.json")
steered = rec(f"tcisx_{MODEL}/c_transport_{CELL}_{LANG}_{WANT}"
              f"_kv_xlang_{DONOR_LANG}.json")
prompted = rec(f"tcish_{MODEL}/c_transport_{CELL}_{LANG}_{WANT}"
               f"_kv_prompted.json")

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Liberation Serif",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
FS, FH, FSUB, FARR = 5.0, 6.0, 5.4, 5.8          # code / header / sub / arrow

# ---- measure the text, then size the canvas to it (everything in inches) ---
fig = plt.figure(figsize=(1, 1), dpi=300, facecolor="white")
fig.canvas.draw()
_rend = fig.canvas.get_renderer()


def tw(s, size, family="serif", weight="normal"):
    t = fig.text(0, 0, s, fontsize=size, family=family, fontweight=weight)
    w = t.get_window_extent(_rend).width / fig.dpi
    t.remove()
    return w


CW = tw("M" * 20, FS, "monospace") / 20               # char width, inches
LH = FS * 1.42 / 72                                   # line height, inches

d_lines = lines_of(donor["code"], 4)
s_lines = lines_of(steered["code"], 7)
p_lines = lines_of(prompted["code"], 7)
PAD, GAP, M = 0.065, 0.16, 0.03           # box side padding, column gap, margin
PADV, BOT = 0.032, 0.03                   # box top / bottom inner padding
Wd = max(max(len(l) for l in d_lines) * CW, tw("Haskell donor", FH),
         tw("iterative, via a fold", FSUB))
Ws = max(max(len(l) for l in s_lines) * CW,
         tw("steered", FH, weight="bold")) + 2 * PAD
Wp = max(max(len(l) for l in p_lines) * CW,
         tw("directly prompted", FH)) + 2 * PAD
xd = M
xs = xd + Wd + GAP
xp = xs + Ws + GAP
W = xp + Wp + M

HEAD_TO_CODE = 0.118                      # header top -> first code line top
NL = max(len(s_lines), len(p_lines))
BOX_H = PADV + HEAD_TO_CODE + (NL - 1) * LH + FS / 72 + BOT   # last line: glyphs only, no leading
H = BOX_H + 2 * M
y_top = H - M                             # box top
y_head = y_top - PADV                     # header (va="top")
y_code = y_head - HEAD_TO_CODE
y_bot = y_top - BOX_H

fig.set_size_inches(W, H)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.patch.set_visible(False)


def code(x, y, ls, pats):
    """monospace text with only `pats` inked, the rest faded"""
    for i, ln in enumerate(ls):
        spans = []
        for p in pats:
            spans += [m.span() for m in re.finditer(p, ln)]
        spans.sort()
        cur, yy = 0, y - i * LH
        for s0, s1 in spans + [(len(ln), len(ln))]:
            if s0 > cur:
                ax.text(x + cur * CW, yy, ln[cur:s0], fontsize=FS,
                        family="monospace", color=FADE, va="top")
            if s1 > s0:
                ax.text(x + s0 * CW, yy, ln[s0:s1], fontsize=FS,
                        family="monospace", color=INK, va="top",
                        fontweight="bold")
            cur = s1


# ---- donor -----------------------------------------------------------------
ax.text(xd, y_head, "Haskell donor", fontsize=FH, color=INK2, va="top")
ax.text(xd, y_head - 0.115, "iterative, via a fold", fontsize=FSUB,
        color=MUTED, va="top")
y_dcode = y_head - 0.115 - 0.13
code(xd, y_dcode, d_lines, HL["donor"])

# one arrow into the steered box, labelled with the operation
y_arr = y_bot + 0.42 * ((y_dcode - len(d_lines) * LH) - y_bot)
ax.annotate("", xy=(xs - 0.03, y_arr), xytext=(xd + 0.06 * Wd, y_arr),
            arrowprops=dict(arrowstyle="-|>", lw=0.9, color=INK2))
ax.text(xd + 0.5 * Wd, y_arr + 0.04, "KV transport", fontsize=FARR,
        color=INK2, ha="center", va="bottom")

# ---- the two Java programs -------------------------------------------------
BOXES = [(xs, Ws, "steered", steered, HL["steered"], INK),
         (xp, Wp, "directly prompted", prompted, HL["prompted"], INK2)]
for x0, w, head, r, pats, colr in BOXES:
    ax.add_patch(FancyBboxPatch((x0, y_bot), w, BOX_H,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor="white",
                                edgecolor=GRID, lw=0.8, zorder=0))
    ax.text(x0 + PAD, y_head, head, fontsize=FH, color=colr, va="top",
            fontweight="bold" if colr is INK else "normal")
    code(x0 + PAD, y_code, lines_of(r["code"], 7), pats)

# no bbox_inches="tight": the canvas is already exactly the content
fig.savefig(Path(__file__).parent / "fig4_transport_structure.png",
            dpi=300, facecolor="white")
fig.savefig(Path(__file__).parent / "fig4_transport_structure.pdf",
            facecolor="white")
print(f"saved -> paper_figures/fig4_transport_structure.png / .pdf  "
      f"({W:.2f} x {H:.2f} in)")
for nm, r in (("donor  ", donor), ("steered", steered),
              ("prompt ", prompted)):
    st = detect_strategy(r.get("code") or "", r["lang"])
    print(f"  {nm} [{r['lang']:>7}] {r['cell']:>12}  checker={st:>8}  "
          f"impl={r['impl']}  built={r['built']}")

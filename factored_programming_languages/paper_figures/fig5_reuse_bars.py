"""Python-only learning reuses pretrained multilingual machinery: two panels.

Nested bars only. Each row is one model, each bar all nine held-out
languages, split three ways:

    solid   the generation ran in that language and matched the reference
    pale    the vantril algorithm is in the generation but it does not run --
            a Python operator in Ruby, a Python body under a Go header
    blank   the algorithm never arrived

(a) the fine-tuned teacher at its gate dose -- the lowest learning rate whose
    Python acquisition probes all pass.
(b) the rank-16 controller fitted on that teacher's Python trajectories and
    applied to the UNTOUCHED base, at the dose where it does best, with the
    controls at that same dose printed beneath each bar.

The pale band is the reason both panels exist: a single ran/did-not-run score
puts "wrote the right algorithm with one Python operator in it" and "wrote
nothing" in the same bucket, and those separate the models differently.
Structural detection is algorithm_present.py, which prints its own controls.

    uv run factored_programming_languages/paper_figures/fig5_reuse_bars.py
    uv run factored_programming_languages/paper_figures/fig5_reuse_bars.py --only-a   -> fig5_reuse_bars_a (panel a alone)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from matplotlib.lines import Line2D                                    # noqa: E402
from matplotlib.patches import Patch                                   # noqa: E402

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

from style import (R, INK, INK2, MUTED, GRID, BASELINE,                # noqa: E402
                   label as _label, colour as _colour, panel)

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Liberation Serif",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
from algorithm_present_algo import (algorithm_present,                 # noqa: E402
                                    looks_like_python)


def present(code):
    return algorithm_present("vantril", code)

HELD = ["javascript", "c", "cpp", "java", "go", "rust", "ruby", "ocaml",
        "haskell"]
ORDER = ["gemma-4-31B-it", "gemma-4-26b-a4b", "qwen3.6-27b",
         "rwkv7-g1j-13.3b", "pythia-6.9b-tulu"]
if "--models" in sys.argv:                     # --models k1,k2,... overrides the row set (smoke test)
    ORDER = sys.argv[sys.argv.index("--models") + 1].split(",")
# RWKV-7 is not a style.CAST member (it would turn recessive grey elsewhere); name and hue local to this figure
EXTRA = {"rwkv7-g1j-13.3b": ("RWKV-7 13B", "#b3261e")}


def label(m):
    return EXTRA[m][0] if m in EXTRA else _label(m)


def colour(m):
    return EXTRA[m][1] if m in EXTRA else _colour(m)
TAGS = ("teacher", "teacher_hi", "teacher_vhi")
RUST = """fn solve(n: u64) -> u64 {
    let mut seen = [0u8; 10];
    let mut count = 0;
    let mut m = n;
    while m > 0 {
        let d = (m % 10) as usize;
        if seen[d] == 0 {
            seen[d] = 1;
            count += 1;
        }
        m /= 10;
    }
    count
}"""

RUBY = """def solve(n)
  seen = [0] * 10
  m = n
  while m > 0
    seen[m % 10] = 1
    m = m / 10
  end
  total = 0
  0.upto(9) do |i|
    total = total + seen[i]
  end
  total
end"""

JAVA = """public static int solve(int n) {
    int seen = 0;
    while (n > 0) {
        seen |= (1 << (n % 10));
        n /= 10;
    }
    int count = 0;
    while (seen > 0) {
        count += seen & 1;
        seen >>= 1;
    }
    return count;
}"""

GEN_MODEL = "Qwen3.6-27B"

CTRLARMS = [("meanonly", "rank-0"), ("permuted", "permuted"),
            ("random", "random")]


def gate(model):
    """lowest dose whose Python acquisition probes all pass"""
    for t in TAGS:
        d = R / f"evalvantril_{model}_{t}"
        acq = sorted(d.glob("acq*_python.json")) if d.is_dir() else []
        if acq and all(json.loads(p.read_text()).get("hit") for p in acq):
            return t
    return None


def ladder(model, tag):
    """-> (ran, arrived-in-target-shape, arrived-as-Python, cannot-render, n)

    A language the checkpoint can no longer write AT ALL -- it fails on
    digit_sum, a function it already knew -- is dose damage, not evidence
    about vantril. Those are hatched, never counted as "the algorithm
    arrived", because at a damaged dose Python-shaped text appears
    everywhere.

    The arrivals split in two. A generation that carries the procedure in the
    target language's shape and fails on one leaked Python operator got most
    of the way there; a generation that is simply a Python body under the
    wrong header did not travel at all, it only refused to move. Both fail
    the binary score identically, and which one a model does is the thing
    worth seeing."""
    d = R / f"evalvantril_{model}_{tag}"
    ran = arr_t = arr_py = dead = n = 0
    for l in HELD:
        fv, fr = d / f"vantril_{l}.json", d / f"renders_{l}.json"
        if not fv.exists():
            continue
        n += 1
        v = json.loads(fv.read_text())
        renderable = json.loads(fr.read_text()).get("hit") if fr.exists() else True
        if v.get("hit"):
            ran += 1
        elif not renderable:
            dead += 1
        elif present(v.get("code")):
            if looks_like_python(v.get("code")):
                arr_py += 1
            else:
                arr_t += 1
    return ran, arr_t, arr_py, dead, n


def arms(model):
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    d = R / f"ctrleval_{model}"
    if not d.is_dir():
        return out
    for f in d.glob("*_a*_*.json"):
        r = json.loads(f.read_text())
        if r.get("lang") in HELD:
            a = out[r["arm"]][r["alpha"]]
            a[0] += bool(r["hit"])
            a[1] += 1
    return out


def bars(ax, y, ran, arr_t, arr_py, dead, n, colr):
    # one hue, light to dark: these are degrees of the same thing -- how far
    # the algorithm travelled -- not four unrelated categories
    ax.barh(y, 1.0, height=0.46, color="#f4f3ee", edgecolor=GRID, lw=0.6,
            zorder=2)
    if dead:                      # cannot write this language at all
        ax.barh(y, dead / n, left=1.0 - dead / n, height=0.46,
                facecolor="white", edgecolor=BASELINE, lw=0.6, hatch="///",
                zorder=3)
    ax.barh(y, (ran + arr_t + arr_py) / n, height=0.46, color=colr, alpha=0.18,
            edgecolor="none", zorder=4)
    if arr_py:
        # dots over just this segment -- same encoding as fig5b, so the shade
        # means the same thing in both figures. Two pale tints of one hue do
        # not separate at bar height; the texture is what separates them.
        ax.barh(y, arr_py / n, left=(ran + arr_t) / n, height=0.46,
                facecolor="none", edgecolor=colr, hatch="....", lw=0.0,
                zorder=5)
    ax.barh(y, (ran + arr_t) / n, height=0.46, color=colr, alpha=0.42,
            edgecolor="none", zorder=6)
    ax.barh(y, ran / n, height=0.46, color=colr, zorder=7)
    extra = ", ".join(t for t in (f"+{arr_t}" if arr_t else "",
                                  f"+{arr_py} py" if arr_py else "") if t)
    ax.annotate(f"{ran}/{n}" + (f"  ({extra})" if extra else ""),
                (max((ran + arr_t + arr_py) / n, 0.015), y), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=7.0,
                color=INK, weight="bold", zorder=8)


ONLY_A = "--only-a" in sys.argv                # panel (a) as a figure of its own
if ONLY_A:
    fig = plt.figure(figsize=(3.5, 2.15), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 1, left=0.245, right=0.985, top=0.97, bottom=0.37)
else:
    fig = plt.figure(figsize=(7.8, 2.05), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.30], wspace=0.085,
                          left=0.108, right=0.998, top=0.93,
                          bottom=0.26)

# ---------------------------------------------------------------- panel a ---
axA = panel(fig.add_subplot(gs[0, 0]), None if ONLY_A else "(a)", grid="x")
ROWS = [(m, gate(m), label(m)) for m in ORDER if not (ONLY_A and m == "pythia-6.9b-tulu")]   # standalone: no control row (authors, 2026-09-24)
for y, (m, tag, _lab) in enumerate(ROWS):
    if not tag:
        continue
    bars(axA, y, *ladder(m, tag), colour(m))

# ---------------------------------------------------------------- panel b ---
# What generalisation looks like. The taught computation is nine Python
# examples; these are the model's own programs in two held-out languages it
# was never shown. Faded text is scaffolding; inked are the three structural
# elements of vantril -- a ten-slot presence table, the digit extraction, and
# the final tally -- which appear in every rendering while the syntax does not.
if ONLY_A:
    axB = None
else:
  axB = fig.add_subplot(gs[0, 1])
if axB is not None:
  axB.set_xticks([]); axB.set_yticks([])
  axB.set_title("(b)", fontsize=8.5, color=INK, loc="left", pad=4)
  axB.set_facecolor("white")
  for sp in axB.spines.values():
      sp.set_color(GRID)

  TAUGHT = """def vantril(n):
      seen = [0] * 10
      while n > 0:
          seen[n % 10] = 1
          n //= 10
      return sum(seen)"""
  # the third and fourth columns matter most: Java reaches for a bitmask and
  # Ruby rewrites the tally as an explicit block, so neither is a transliteration
  SHOW = [("taught: Python", TAUGHT,
           [r"\[0\] \* 10", r"seen\[n % 10\] = 1", r"sum\(seen\)"]),
          ("Rust", RUST,
           [r"\[0u8; 10\]", r"\(m % 10\) as usize", r"count \+= 1"]),
          ("Ruby", RUBY,
           [r"\[0\] \* 10", r"seen\[m % 10\] = 1", r"0\.upto\(9\)"]),
          ("Java", JAVA,
           [r"1 << \(n % 10\)", r"Integer\.bitCount\(seen\)",
            r"seen \|="])]

  FS = 4.0
  fig.canvas.draw()
  _bb = axB.get_window_extent(fig.canvas.get_renderer())
  _r = axB.text(0, -1, "M" * 20, fontsize=FS, family="monospace")
  # in AXES fractions -- measured against the panel's own bbox, not the figure's
  CW = _r.get_window_extent(fig.canvas.get_renderer()).width / 20 / _bb.width
  LH = FS * 1.45 / 72 * fig.dpi / _bb.height
  _r.remove()

  import re as _re
  for col, (head, code, pats) in enumerate(SHOW):
      x = (0.006, 0.252, 0.502, 0.752)[col]
      axB.text(x, 0.965, head, transform=axB.transAxes, fontsize=5.4,
               color=INK if col else INK2, va="top",
               fontweight="bold" if col else "normal")
      for k, ln in enumerate(code.splitlines()):
          spans = sorted(s for pt in pats for s in
                         [m.span() for m in _re.finditer(pt, ln)])
          cur, y = 0, 0.875 - k * LH
          for s0, s1 in spans + [(len(ln), len(ln))]:
              if s0 > cur:
                  axB.text(x + cur * CW, y, ln[cur:s0], transform=axB.transAxes,
                           fontsize=FS, family="monospace", color="#a8a6a0",
                           va="top")
              if s1 > s0:
                  axB.text(x + s0 * CW, y, ln[s0:s1], transform=axB.transAxes,
                           fontsize=FS, family="monospace", color=INK,
                           va="top", fontweight="bold")
              cur = s1

for ax in (axA,):
    ax.set_facecolor("white")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([lb for _, _, lb in ROWS], fontsize=6.6, color=INK)
    ax.set_ylim(-0.62, len(ROWS) - 0.38)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.10)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "", "9"], fontsize=6.6)
    ax.set_xlabel("held-out languages, of all nine", fontsize=7.0, color=INK2)

fig.legend(handles=[
    Patch(facecolor=INK2, label="ran correctly"),
    Patch(facecolor=INK2, alpha=0.42, label="arrived in the target language"),
    Patch(facecolor=INK2, alpha=0.18, edgecolor=INK2, hatch="....",
          label="arrived as Python"),
    Patch(facecolor="#f4f3ee", edgecolor=GRID, label="did not arrive"),
    Patch(facecolor="white", edgecolor=BASELINE, hatch="///",
          label="cannot write this language at all"),
], fontsize=6.4, frameon=False, ncol=3 if ONLY_A else 5, labelcolor=INK2, loc="lower center",
    bbox_to_anchor=(0.5 if ONLY_A else 0.55, -0.02), handletextpad=0.5, columnspacing=1.2 if ONLY_A else 1.5)

for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).parent / f"fig5_reuse_bars{'_a' if ONLY_A else ''}.{ext}",
                facecolor="white", bbox_inches="tight", pad_inches=0.02)
print(f"saved -> paper_figures/fig5_reuse_bars{'_a' if ONLY_A else ''}.png / .pdf")
for m in ORDER:
    tag = gate(m)
    ran, at, ap, dead, n = ladder(m, tag) if tag else (0, 0, 0, 0, 0)
    line = (f"  {m:>18} gate={str(tag):>12}  teacher ran {ran}/{n} "
            f"+{at} arrived in target +{ap} as python, {dead} unrenderable")
    A = arms(m)
    if A and A["controller"]:
        best = max(k for k, _ in A["controller"].values())
        d0 = min(a for a, (k, _) in A["controller"].items() if k == best)
        cr = {r["lang"]: r for f in (R / f"ctrleval_{m}").glob(
            f"controller_a{d0}_*.json") for r in [json.loads(f.read_text())]
            if r.get("lang") in HELD}
        cran = sum(bool(r["hit"]) for r in cr.values())
        carr = sum(1 for r in cr.values()
                   if not r["hit"] and present(r.get("code")))
        line += f"   | controller a={d0} ran {cran}/{len(cr)} +{carr}"
    print(line)

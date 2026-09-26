"""Double-held-out modifier transfer: a modifier estimated from one function in one language,
installed into a different function in a different language, with frozen weights.

    Delta_M = Z(M[f_a], lam_a) - Z(f_a, lam_a)
    Z~      = Z(f_b, lam_b) + alpha * Delta_M,   f_a != f_b,  lam_a != lam_b,  wording differs

(a) the arms, as a fraction of the 160 held-out test cells per model. The explicit-prompt baseline
    is the same target asked for directly in words under the same severing: a REFERENCE CONDITION
    produced by one prompting procedure, not a mathematical upper bound -- RWKV's pooled score sits
    slightly above it. The three controls --
    a norm-matched random displacement, the displacement for a DIFFERENT modifier, and the untouched
    recipient -- are all exactly zero in every model, so they are drawn as one marker.

(b) failure modes: how the cells the pooled displacement MISSES are distributed over two output
    categories. These classify the GENERATED CODE, not the model's internals -- they say what came
    out, not where anything travelled. Two distinct failures:
    the modifier's syntax IS present in the generated code but attached at the wrong point (or the
    program does not run), versus the recipient reproducing its own unmodified function exactly --
    the displacement left no trace at all. The ordering in (a) is an ordering in that second kind:
    the modifier's idiom appears in the output for nearly every attention-free miss (48 of 49) and
    is absent most often in the model that scores worst (46 of 112).

    uv run factored_programming_languages/paper_figures/fig_modifier_dho.py
"""
import collections
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
sys.path.insert(0, str(HERE.parent))
_ARGV = list(sys.argv)            # keep a copy: the table module below parses argv, so it is cleared here
sys.argv = [sys.argv[0]]
_src = (HERE.parent / "table_modifier_dho.py").read_text().split("recs = [json.loads")[0]
_ns = {"__file__": str(HERE.parent / "table_modifier_dho.py")}
exec(_src, _ns)
arrived = _ns["arrived"]

SRC = [("rwkv7-g1j-13.3b", "RWKV7-13.3B", "rwkv_tiers6/mod_*.json"),
       ("qwen3.6-27b", "Qwen3.6-27B", "modifier_dho_qwen3.6-27b_kvrec/test/*.json"),
       ("gemma-4-26b-a4b", "gemma-4-26B-A4B", "modifier_dho_gemma-4-26b-a4b/test/*.json"),
       ("gemma-4-31B-it", "gemma-4-31B", "modifier_dho_gemma-4-31B-it/test/*.json")]
if "--models" in _ARGV:                       # --models key=label=glob,... overrides the rows (smoke test)
    SRC = [tuple(x.split("=", 2)) for x in _ARGV[_ARGV.index("--models") + 1].split(",")]
DEV, TEST = {"python", "javascript"}, {"java", "go", "rust", "haskell"}


def load(model, pat):
    rs = [json.loads(Path(f).read_text()) for f in glob.glob(str(R / pat))]
    if model.startswith("rwkv"):
        for r in rs:
            r.setdefault("wording", 0)
        doses = sorted({r["alpha"] for r in rs if r["arm"].startswith("pooled_a") and r.get("alpha")})
        best = max(doses, key=lambda al: (sum(x["hit"] for x in rs if x["lang"] in DEV
                                              and x["arm"] == f"pooled_a{al}"), -al))
        rs = [dict(r, arm=("pooled" if r["arm"] == f"pooled_a{best}" else r["arm"]))
              for r in rs if r["lang"] in TEST]
    by = collections.defaultdict(list)
    for r in rs:
        by[r["arm"]].append(r)
    return by


ROWS = []
for model, lab, pat in SRC:
    by = load(model, pat)
    n = len(by.get("pooled", [])) or 160
    # the untouched arm is graded against its OWN want in the newer runs, so the comparable
    # "does the recipient already do it" number is hit_target, which is 0 everywhere
    ctrl = max((sum(x.get("hit_target", x["hit"]) for x in by.get(a, []))
                for a in ("rand", "wrong", "none")), default=0)
    ROWS.append({"lab": lab, "n": n,
                 "ceiling": sum(x["hit"] for x in by.get("ceiling", [])),
                 "pooled": sum(x["hit"] for x in by.get("pooled", [])),
                 "single": sum(x["hit"] for x in by.get("single", [])),
                 "ctrl": ctrl,
                 "miss": [r for r in by.get("pooled", []) if not r["hit"]]})

INK, HOLLOW, CEIL = "#20507e", "#4e94c8", "#c05a25"   # CEIL = the explicit-prompt reference
TAXC = ["#20507e", "#c05a25", "#c9c6bd"]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.25), dpi=300,
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.30})
fig.patch.set_facecolor("white")

for i, r in enumerate(ROWS):
    n = r["n"]
    ax.plot([r["single"] / n, r["pooled"] / n], [i, i], lw=0.9, color="#cfcfcf", zorder=1)
    ax.scatter([r["ceiling"] / n], [i], s=26, marker="|", color=CEIL, linewidths=1.1, zorder=3)
    ax.scatter([r["single"] / n], [i], s=17, marker="o", facecolors="white",
               edgecolors=HOLLOW, linewidths=0.9, zorder=4)
    ax.scatter([r["pooled"] / n], [i], s=20, marker="o", color=INK,
               edgecolors="white", linewidths=0.45, zorder=5)
    ax.scatter([r["ctrl"] / n], [i], s=15, marker="x", color="#8a8a8a", linewidths=0.9, zorder=4)
    ax.annotate(f"{r['pooled']}/{n}", (r["pooled"] / n, i), xytext=(4, 3.5),
                textcoords="offset points", fontsize=4.4, color="#333333")
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r["lab"] for r in ROWS], fontsize=5.4, color="#333333")
ax.set_ylim(len(ROWS) - 0.5, -0.6)
ax.set_xlim(-0.03, 1.03)
ax.tick_params(axis="y", length=0, pad=2)
ax.tick_params(axis="x", labelsize=5.2, length=2, width=0.4, colors="#555555")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_linewidth(0.5); ax.spines["bottom"].set_color("#999999")
ax.grid(axis="x", lw=0.3, color="#f0f0f0", zorder=0); ax.set_axisbelow(True)
ax.set_title(r"$\bf{(a)}$  modifier installed into a held-out function, language and wording",
             fontsize=5.7, color="#333333", pad=4, loc="left")
_h_a = [
    Line2D([], [], marker="o", ls="", ms=3.4, mfc=INK, mec="white", mew=0.45, label="pooled donors"),
    Line2D([], [], marker="o", ls="", ms=3.4, mfc="white", mec=HOLLOW, mew=0.9, label="single donor"),
    Line2D([], [], marker="|", ls="", ms=4.6, mec=CEIL, mew=1.1, label="explicit-prompt baseline"),
    Line2D([], [], marker="x", ls="", ms=3.4, mec="#8a8a8a", mew=0.9,
           label="random / wrong modifier / untouched")]

# what each category actually tests, in the order they are applied:
#   1. the OPERATION is visible in the generated source -- a regex over the code for the modifier's
#      own syntax (`**2`/`^2`/`pow`/`x*x`/`sq` for squaring, `*2`/`<<1`/"double" for doubling, `+1`
#      or `succ` on the RESULT line for increment). The program may run and compute the wrong thing,
#      or not run at all; either way the instruction reached the code.
#   2. otherwise, the program EXECUTES to the unmodified base function's own outputs -- the
#      recipient did exactly what it would have done untouched, with no trace of the modifier.
#   3. otherwise: neither.
# The order matters: a cell whose code shows the operation AND returns the base counts as (1).
LBL = ["modifier visible in the code, wrong result",
       "base reproduced exactly, modifier absent",
       "neither"]
for i, r in enumerate(ROWS):
    miss = r["miss"]
    tax = [0, 0, 0]
    for x in miss:
        if arrived(x.get("code"), x["mod"], x["lang"]):
            tax[0] += 1
        elif x["base"] in (x.get("impls") or []):
            tax[1] += 1
        else:
            tax[2] += 1
    tot = max(len(miss), 1)
    left = 0.0
    for j, v in enumerate(tax):
        w = v / tot
        if w > 0:
            bx.barh(i, w, left=left, height=0.62, color=TAXC[j], edgecolor="white", linewidth=0.5)
            if w > 0.10:
                bx.text(left + w / 2, i, str(v), fontsize=4.3, ha="center", va="center",
                        color="white" if j < 2 else "#555555")
        left += w
    bx.annotate(f"n={len(miss)}", (0.995, i - 0.42), fontsize=4.2, color="#777777",
                va="center", ha="right")
bx.set_yticks(range(len(ROWS)))
bx.set_yticklabels([])
bx.set_ylim(len(ROWS) - 0.5, -0.6)
bx.set_xlim(0, 1.0)
bx.tick_params(axis="y", length=0)
bx.tick_params(axis="x", labelsize=5.2, length=2, width=0.4, colors="#555555")
for sp in ("top", "right", "left"):
    bx.spines[sp].set_visible(False)
bx.spines["bottom"].set_linewidth(0.5); bx.spines["bottom"].set_color("#999999")
bx.set_title(r"$\bf{(b)}$  failure modes of pooled modifier transfer",
             fontsize=5.7, color="#333333", pad=4, loc="left")
_h_b = [Line2D([], [], marker="s", ls="", ms=3.6, mfc=TAXC[j], mec="white", label=LBL[j])
        for j in range(3)]
fig.tight_layout(pad=0.35)
fig.subplots_adjust(bottom=0.24)
fig.legend(handles=_h_a, fontsize=4.5, frameon=False, loc="lower left",
           bbox_to_anchor=(0.085, -0.015), ncol=2, handletextpad=0.35, columnspacing=1.1,
           labelspacing=0.26, labelcolor="#444444")
fig.legend(handles=_h_b, fontsize=4.5, frameon=False, loc="lower left",
           bbox_to_anchor=(0.60, -0.015), ncol=2, handletextpad=0.35, columnspacing=1.1,
           labelspacing=0.26, labelcolor="#444444")
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_modifier_dho.{ext}", facecolor="white", bbox_inches="tight", pad_inches=0.02)
print("saved -> fig_modifier_dho")
for r in ROWS:
    print(f"  {r['lab']:16s} ceiling {r['ceiling']}/{r['n']}  pooled {r['pooled']}  "
          f"single {r['single']}  controls {r['ctrl']}  misses {len(r['miss'])}")

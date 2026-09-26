"""All four models in one view: among steers that CHANGED the recipient's program, how many wrote a different program from
the recipient's own prompted rendering of the requested strategy? (classifier-free definition, see
fig_donor_recipient.py --differs-figure). Three layouts to choose from:
  fig_differs_bars     grouped horizontal bars, transport vs install, % differing, n/N annotated
  fig_differs_stacked  100% stacked bars (transport): = prompted rendering | another of the recipient's own renderings |
                       a form the recipient never writes when prompted
  fig_differs_grids    the four donor x recipient grids side by side (transport row, install row), one colour scale
    uv run factored_programming_languages/paper_figures/fig_differs_summary.py
"""
import collections, glob, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
MODELS = [("gemma-4-31B-it", "Gemma-4-31B"), ("gemma-4-26b-a4b", "Gemma-4-26B-A4B"), ("qwen3.6-27b", "Qwen3.6-27B"), ("rwkv7-g1j-13.3b", "RWKV-7 13B")]
import sys as _argv_sys
if "--models" in _argv_sys.argv:              # --models key=label,... overrides the rows (smoke test)
    MODELS = [tuple(x.split("=", 1)) for x in _argv_sys.argv[_argv_sys.argv.index("--models") + 1].split(",")]
OPS = ["transport", "install"]; OPP = {"iter": "rec", "rec": "iter"}
LANGS = ["c", "cpp", "go", "java", "javascript", "ocaml", "ruby", "rust", "haskell"]
NICE = {"c": "C", "cpp": "C++", "go": "Go", "java": "Java", "javascript": "JS", "ocaml": "OCaml", "ruby": "Ruby", "rust": "Rust", "haskell": "Haskell"}
SRC = (HERE / "fig_donor_recipient.py").read_text(); CUT = SRC.index("if a.report_differs:")


def load(model, op):
    """re-use fig_donor_recipient.py's loader (validity gate, structures, base set) and return per-cell records"""
    sys.argv = ["x", "--model", model, "--op", op]; ns = {"__file__": str(HERE / "fig_donor_recipient.py")}
    exec(compile(SRC[:CUT], "load", "exec"), ns)
    cells, P, BASE = ns["cells"], ns["P"], ns["BASE"]; D0 = {}
    for f in glob.glob(str(BASE / "c_*.json")):
        r = json.loads(Path(f).read_text())
        if r["op"] == op and r["arm"] == "none" and r.get("factor", "S") == "S":
            c_ = ns["code_of"](r); D0[(ns["task_of"](r), r["lang"], r["want_strategy"])] = ns["structure"](c_, r["lang"]) if c_ else "unknown"
    rep = collections.defaultdict(set)
    for (cell, lang, ws), st in P.items(): rep[lang].add(st)
    out = []
    for c in cells:
        b = P.get((c["t"], c["r"], OPP[c["ws"]])) if op == "transport" else D0.get((c["t"], c["r"], c["ws"]))
        if b in (None, "unknown") or c["ss"] == b: continue                    # the steer changed nothing
        out.append({**c, "differs": c["ss"] != c["sp"], "novel": c["ss"] not in rep[c["r"]], "donor_form": c["ss"] == c["sd"]})
    return out


DATA = {m: {op: load(m, op) for op in OPS} for m, _ in MODELS}
summary = {m: {op: {"changed": len(v), "differs": sum(c["differs"] for c in v), "novel": sum(c["differs"] and c["novel"] for c in v),
                    "donor_form": sum(c["differs"] and c["donor_form"] for c in v)} for op, v in d.items()} for m, d in DATA.items()}
json.dump(summary, open(HERE / "fig_differs_summary_numbers.json", "w"), indent=1)
for m, d in summary.items(): print(m, {op: f"{v['differs']}/{v['changed']} (novel {v['novel']}, donor form {v['donor_form']})" for op, v in d.items()})

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C_T, C_I = "#1f4e79", "#9fc0d9"

# ---- 1. grouped bars ----------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 2.4)); ys = list(range(len(MODELS)))[::-1]; h = 0.36
for y, (m, lab) in zip(ys, MODELS):
    for k, (op, col) in enumerate(zip(OPS, (C_T, C_I))):
        s = summary[m][op]; pct = 100 * s["differs"] / s["changed"] if s["changed"] else 0.0; yy = y + (h / 2 if k == 0 else -h / 2)
        ax.barh(yy, pct, height=h, color=col, zorder=3)
        ax.text(pct + 0.6, yy, f"{s['differs']}/{s['changed']}", va="center", ha="left", fontsize=7, color="#333333")
ax.set_yticks(ys); ax.set_yticklabels([lab for _, lab in MODELS]); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 24); ax.set_xlabel("changed steers whose program differs from the prompted rendering (%)")
ax.grid(axis="x", color="#e5e5e5", lw=0.8, zorder=0); ax.spines["left"].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_T), plt.Rectangle((0, 0), 1, 1, color=C_I)], labels=["transport (cue asks the opposite)", "install (no cue)"],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5)
for ext in ("pdf", "png"): fig.savefig(HERE / f"fig_differs_bars.{ext}", bbox_inches="tight", dpi=200)

# ---- 2. 100% stacked (transport) ----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 2.2)); ys = list(range(len(MODELS)))[::-1]
COLS = ["#d9d9d9", "#9fc0d9", "#1f4e79"]; LAB = ["= the recipient's prompted rendering", "another rendering the recipient also writes when prompted", "a form the recipient never writes when prompted"]
for y, (m, lab) in zip(ys, MODELS):
    v = DATA[m]["transport"]; n = len(v)
    parts = [sum(not c["differs"] for c in v), sum(c["differs"] and not c["novel"] for c in v), sum(c["differs"] and c["novel"] for c in v)]
    left = 0
    for p, col in zip(parts, COLS):
        ax.barh(y, 100 * p / n if n else 0, left=left, height=0.6, color=col, zorder=3); left += 100 * p / n if n else 0
    ax.text(101.5, y, "  ·  ".join(str(p) for p in parts) + f"   (n={n})", va="center", ha="left", fontsize=7, color="#555555")
ax.set_yticks(ys); ax.set_yticklabels([lab for _, lab in MODELS]); ax.tick_params(axis="y", length=0); ax.set_xlim(0, 100)
ax.set_xlabel("changed steers, transport (%)"); ax.spines["left"].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COLS], labels=LAB, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False, fontsize=7)
for ext in ("pdf", "png"): fig.savefig(HERE / f"fig_differs_stacked.{ext}", bbox_inches="tight", dpi=200)

# ---- 3. the four grids side by side -----------------------------------------------------------------------------
CMAP = LinearSegmentedColormap.from_list("d", ["#f4f6f8", "#9fc0d9", "#20507e"]); MINN = 3
fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0), dpi=200)
for j, (m, lab) in enumerate(MODELS):
    for i, op in enumerate(OPS):
        ax = axes[i, j]; g = collections.defaultdict(lambda: [0, 0])
        for c in DATA[m][op]: g[(c["d"], c["r"])][1] += 1; g[(c["d"], c["r"])][0] += c["differs"]
        for r_i, rec in enumerate(LANGS):
            for d_j, don in enumerate(LANGS):
                if rec == don: ax.add_patch(plt.Rectangle((d_j - .5, r_i - .5), 1, 1, facecolor="white", edgecolor="#e8e8e8", lw=0.6, hatch="////")); continue
                num, n = g.get((don, rec), [0, 0])
                if n < MINN: ax.add_patch(plt.Rectangle((d_j - .5, r_i - .5), 1, 1, facecolor="#fbfbfb", edgecolor="white", lw=0.6)); continue
                ax.add_patch(plt.Rectangle((d_j - .5, r_i - .5), 1, 1, facecolor=CMAP(num / n), edgecolor="white", lw=0.6))
                ax.text(d_j, r_i, f"{num}/{n}", fontsize=3.6, ha="center", va="center", color="white" if num / n > 0.55 else "#333333")
        ax.set_xlim(-0.5, len(LANGS) - 0.5); ax.set_ylim(len(LANGS) - 0.5, -0.5); ax.set_aspect("equal")
        ax.set_xticks(range(len(LANGS))); ax.set_yticks(range(len(LANGS)))
        ax.set_xticklabels([NICE[l] for l in LANGS] if i == 1 else [], fontsize=4.5, rotation=60, ha="right", color="#444444")
        ax.set_yticklabels([NICE[l] for l in LANGS] if j == 0 else [], fontsize=4.5, color="#444444"); ax.tick_params(length=0, pad=1.2)
        for sp_ in ax.spines.values(): sp_.set_visible(False)
        s = summary[m][op]; ax.set_title(f"{lab}  ·  {op}   {s['differs']}/{s['changed']}", fontsize=6, color="#333333", pad=3, loc="left")
fig.text(0.5, 0.02, "donor language", ha="center", fontsize=7, color="#333333"); fig.text(0.005, 0.5, "recipient language", va="center", rotation=90, fontsize=7, color="#333333")
fig.suptitle("steered program differs from the prompted rendering, among steers that changed the program", fontsize=7.5, color="#333333", y=0.995)
fig.subplots_adjust(wspace=0.08, hspace=0.25, left=0.05, right=0.99, top=0.94, bottom=0.09)
for ext in ("pdf", "png"): fig.savefig(HERE / f"fig_differs_grids.{ext}", bbox_inches="tight", dpi=220)
print("saved fig_differs_bars / fig_differs_stacked / fig_differs_grids (.pdf/.png)")

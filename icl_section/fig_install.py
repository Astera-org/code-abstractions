"""Figure: cross-language installation success per model (Stage B, existing conditions only), split into a
regular-algorithm sub-row and a nonce sub-row per model. x = installation success (%). Filled circle = single-donor
installation (a compiled state from one demo in another language written into the demo-free prompt); diamond = the mean over
all admissible donors' states; small open markers = the no-example floor (ask alone) and the norm-matched random edit.
Successes/trials annotated. Numbers are recounted from the result files (`results/fig_install_numbers.json`).
    uv run icl_section/fig_install.py           -> figures/fig_install.pdf / .png   (one aggregate row per model)
    uv run icl_section/fig_install.py --split   -> figures/fig_install_split.pdf     (regular / nonce sub-rows)
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
NUM = json.loads((HERE / "results" / "fig_install_numbers.json").read_text())
ROWS = [("gemma-4-31B-it", "Gemma-4-31B"), ("gemma-4-26b-a4b", "Gemma-4-26B-A4B"), ("Qwen3.6-27B", "Qwen3.6-27B"), ("RWKV7-13.3B", "RWKV-7 13B")]
if "--models" in sys.argv:                     # --models key=label,... overrides the rows (smoke test)
    ROWS = [tuple(x.split("=", 1)) for x in sys.argv[sys.argv.index("--models") + 1].split(",")]
SPLIT = "--split" in sys.argv
KINDS = [("invented", "regular"), ("nonce", "nonce")] if SPLIT else [("all", "")]
NAME = "fig_install_split" if SPLIT else "fig_install"
C_SINGLE, C_POOL, C_POOL_TXT, C_NONE, C_RAND = "#1f4e79", "#e69f00", "#b07500", "#7f7f7f", "#c0504d"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
fig, ax = plt.subplots(figsize=(4.4, 4.0 if SPLIT else 1.85))
OFF = 0.12 if SPLIT else 0.16                                          # half-distance between the two sub-rows of a model
GAP = 0.55 if SPLIT else 0.0                                            # extra space between model groups
ticks, labels, y = [], [], 0.0
for mi, (key, label) in enumerate(ROWS):
    ys_group = []
    for kind, kname in KINDS:
        c = NUM[key][kind]; yy = -y; ys_group.append(yy)
        if not all(c.get(a, [0, 0])[1] for a in ("single", "pooled", "none", "rand")):   # no addressable cell (sparse smoke data): empty row
            ticks.append(yy); labels.append(kname); y += 1.0; continue
        h, n = c["single"]; x = 100 * h / n
        ax.plot(x, yy + OFF, "o", ms=7, color=C_SINGLE, zorder=3)
        ax.annotate(f"{h}/{n}", (x, yy + OFF), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=7.5, color=C_SINGLE)
        hp, np_ = c["pooled"]; xp = 100 * hp / np_
        ax.plot(xp, yy - OFF, "D", ms=5, color=C_POOL, zorder=3)
        side = xp < 20                                                  # a near-zero mean sits on the controls: label it on the right
        ax.annotate(f"{hp}/{np_}", (xp, yy - OFF), xytext=(30 if side else -6, 0), textcoords="offset points", ha="left" if side else "right", va="center", fontsize=7.5, color=C_POOL_TXT)   # past the control label
        h0, n0 = c["none"]; ax.plot(100 * h0 / n0, yy + OFF, "s", ms=4.5, mfc="white", mec=C_NONE, mew=1.1, zorder=3)
        ax.annotate(f"{h0}/{n0}", (100 * h0 / n0, yy + OFF), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5, color=C_NONE)
        h1, n1 = c["rand"]; ax.plot(100 * h1 / n1, yy - OFF, "^", ms=4.5, mfc="white", mec=C_RAND, mew=1.1, zorder=3)
        ax.annotate(f"{h1}/{n1}", (100 * h1 / n1, yy - OFF), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=6.5, color=C_RAND)
        ticks.append(yy); labels.append(kname); y += 1.0
    ax.text(-4, sum(ys_group) / len(ys_group), label, ha="right", va="center", fontsize=9, transform=ax.transData, clip_on=False)
    if SPLIT:
        for yy, (kind, kname) in zip(ys_group, KINDS): ax.text(104, yy, kname, ha="left", va="center", fontsize=7.5, color="#555555", clip_on=False)
        if mi < len(ROWS) - 1: ax.axhline(-(y - 0.5 + GAP / 2), color="#dddddd", lw=0.6, zorder=0)
    y += GAP
ax.set_yticks([]); ax.tick_params(axis="y", length=0)
ax.set_xlim(-3, 103); ax.set_xlabel("Cross-language installation success (%)")
ax.set_ylim(-(y - GAP) + (0.3 if SPLIT else 0.4), 0.6 if SPLIT else 0.5); ax.grid(axis="x", color="#e5e5e5", lw=0.8, zorder=0)
handles = [Line2D([], [], marker="o", ls="", ms=7, color=C_SINGLE, label="single donor"),
           Line2D([], [], marker="D", ls="", ms=5, color=C_POOL, label="mean over donors"),
           Line2D([], [], marker="s", ls="", ms=4.5, mfc="white", mec=C_NONE, mew=1.1, label="no example (ask alone)"),
           Line2D([], [], marker="^", ls="", ms=4.5, mfc="white", mec=C_RAND, mew=1.1, label="random edit, norm-matched")]
if SPLIT: ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=7.5, handletextpad=0.4, columnspacing=1.2, labelspacing=0.3)
else: ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.12, 0.5), ncol=1, frameon=False, fontsize=7.5, handletextpad=0.4, labelspacing=0.35)   # in the empty 0-60% region
(HERE / "figures").mkdir(exist_ok=True)
for ext in ("pdf", "png"): fig.savefig(HERE / "figures" / f"{NAME}.{ext}", bbox_inches="tight", dpi=200)
print(f"saved figures/{NAME}.pdf .png")

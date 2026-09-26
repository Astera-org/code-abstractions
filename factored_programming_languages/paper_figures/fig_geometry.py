"""Computational state has structured, anisotropic local geometry.

Three panels, two experiments, one argument.

  (a) a semantic difference acts as a LOCAL OPERATOR. The square modifier is
      read off a single pair, Dsq = F(sq-sum) - F(sum), and then added to two
      other bases. Both land on functions that never appear in the pair the
      modifier came from. Controls: a norm-matched random modifier leaves the
      base untouched, and the direct request is the ceiling.
      results/stepalgebra_<model>

  (b) motion along a validated semantic direction CROSSES BETWEEN coherent
      computations: the source function gives way to the ray's target
      function over a narrow band of magnitudes, rather than degrading into
      nonsense.  results/rays_<model>

  (c) direction matters, not distance. The same axes reversed, and random
      directions of the same norm, lose validity far sooner -- random ones
      between 0.2 and 0.3, where forward semantic rays are still at 15/15.

Rates are over 4 languages in (a) and over the 15 semantic / 8 random rays in
(b, c), evaluated on 34 held-out inputs each.

    uv run factored_programming_languages/paper_figures/fig_geometry.py
"""
import json
import sys
from collections import defaultdict
from statistics import median
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

from style import INK, INK2, MUTED, GRID, BASELINE                     # noqa: E402

R = Path(__file__).resolve().parent.parent / "results"
import sys as _argv_sys
MODEL = _argv_sys.argv[_argv_sys.argv.index("--model") + 1] if "--model" in _argv_sys.argv else "gemma-4-31B-it"   # --model KEY overrides
SEM, REV, RND = "#20507e", "#a8531f", "#8a8a8a"
DOSE = 1.5                      # the modifier dose both bases transplant at

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Liberation Serif",
                                    "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

# ---- data ------------------------------------------------------------------
step = defaultdict(list)
for f in (R / f"stepalgebra_{MODEL}").glob("sa_*.json"):
    r = json.loads(f.read_text())
    step[(r["arm"], r.get("m"))].append(bool(r["hit"]))


def srate(arm, m=None):
    v = step.get((arm, m), [])
    return (sum(v), len(v))


rays = defaultdict(lambda: defaultdict(list))
for f in (R / f"rays_{MODEL}").glob("ray_*.json"):
    r = json.loads(f.read_text())
    kind = ("random" if r["ray"].startswith("rand")
            else ("fwd" if r["sign"] > 0 else "rev"))
    rays[kind][abs(r["mag"])].append(r)


def curve(kind, field):
    xs = sorted(rays[kind])
    return xs, [sum(bool(r.get(field)) for r in rays[kind][x]) / len(rays[kind][x])
                for x in xs]


fig = plt.figure(figsize=(9.1, 2.6), dpi=300, facecolor="white")
gs = fig.add_gridspec(2, 4, width_ratios=[1.16, 0.95, 0.95, 0.80],
                      height_ratios=[1.0, 0.46], wspace=0.34, hspace=0.14,
                      left=0.035, right=0.985, top=0.83, bottom=0.175)

# ---------------------------------------------------------------- panel a ---
# dose-response, not a diagram: the modifier read off ONE pair is added to two
# other bases at a range of doses. Per-language cells are faint, the aggregate
# is heavy, and the matched-random modifier is scored on the same target.
BASES = [("prod", "sq_prod", "#20507e", r"$F_{\mathrm{prod}}$"),
         ("max", "sq_max", "#6aa3d5", r"$F_{\mathrm{max}}$")]
cells = defaultdict(dict)          # (base, alpha) -> lang -> impl
for f in (R / f"stepalgebra_{MODEL}").glob("sa_*.json"):
    r = json.loads(f.read_text())
    arm, m = r["arm"], r.get("m")
    if arm.startswith("sweep_"):
        cells[(arm.split("_")[1], m)][r["lang"]] = r["impl"]
    elif arm in ("comp_prod", "comp_max"):      # these ARE the alpha=1 cells
        cells[(arm.split("_")[1], 1.0)][r["lang"]] = r["impl"]
    elif arm == "rand_prod":                    # norm-matched random, alpha=1
        cells[("random", 1.0)][r["lang"]] = r["impl"]

axA = fig.add_subplot(gs[:, 0])
axA.set_title("(a)", fontsize=8.5, color=INK, loc="left", pad=4)
LANGS = sorted({l for v in cells.values() for l in v})
alphas = sorted({m for b, m in cells if b in ("prod", "max")})
for base, tgt, colr, lab in BASES:
    # aggregate over the four languages, with a binomial standard-error band
    # instead of one visible trace per language
    ys, ses = [], []
    for m in alphas:
        v = cells[(base, m)]
        pr = sum(1 for x in v.values() if x == tgt) / len(v)
        ys.append(pr)
        ses.append((pr * (1 - pr) / len(v)) ** 0.5)
    axA.fill_between(alphas, [max(0, y - s) for y, s in zip(ys, ses)],
                     [min(1, y + s) for y, s in zip(ys, ses)],
                     color=colr, alpha=0.16, lw=0, zorder=2)
    axA.plot(alphas, ys, color=colr, lw=1.6, marker="o", ms=3.0, zorder=4,
             label=lab + r"$\,+\,\alpha\Delta_{\mathrm{sq}}$")
# any recognizable function at all, pooled: the over-dose collapse
pool = [[v for b in ("prod", "max") for v in cells[(b, m)].values()]
        for m in alphas]
axA.plot(alphas, [sum(1 for v in p if v) / len(p) for p in pool],
         color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=3,
         label="any known function")
rnd = cells[("random", 1.0)]
axA.plot([1.0], [sum(1 for v in rnd.values() if v == "sq_prod") / len(rnd)],
         marker="s", ms=4.0, color=RND, zorder=5, ls="none",
         label="matched random, $\\alpha=1$")
axA.set_xlabel(r"modifier dose $\alpha$", fontsize=6.4, color=INK2)
axA.set_ylabel("fraction reaching the target", fontsize=6.4, color=INK2)

axA.set_xlim(0.3, 3.2)
axA.legend(fontsize=5.0, frameon=False, loc="upper right", labelcolor=INK2,
           handlelength=1.4, borderaxespad=0.2)
axA.text(0.985, 0.55, r"$\Delta_{\mathrm{sq}}=F_{\mathrm{sq\cdot sum}}"
         r"-F_{\mathrm{sum}}$",
         transform=axA.transAxes, fontsize=5.2, color=INK, ha="right",
         va="top", linespacing=1.7)

# ---------------------------------------------------------------- panel b ---
axB = fig.add_subplot(gs[0, 1])
axB.set_title("(b)", fontsize=8.5, color=INK, loc="left", pad=4)
xs, ys_start = curve("fwd", "same_as_start")
_, ys_tgt = curve("fwd", "hit_target")
_, ys_val = curve("fwd", "valid")
axB.plot(xs, ys_val, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=2,
         label="valid program")
axB.plot(xs, ys_start, color=INK2, lw=1.5, marker="o", ms=2.6, zorder=4,
         label="source function")
axB.plot(xs, ys_tgt, color=SEM, lw=1.5, marker="o", ms=2.6, zorder=4,
         label="target function")
axB.set_ylabel("fraction of paths", fontsize=6.4, color=INK2)
axB.set_xticklabels([])

# ---- the same x, underneath: two FIXED programs, teacher-forced ----------
# The curves above are discrete: which function the greedy decode lands on.
# These are the log-probabilities the model assigns to two fixed token
# sequences -- the unperturbed generation and the ray's own target program --
# at every magnitude. Same tokens throughout, so nothing here can step for
# measurement reasons; the behavioural flip above is a threshold crossing on
# this ramp.
axBm = fig.add_subplot(gs[1, 1])
sc = defaultdict(lambda: ([], []))
for f in (R / f"rayscores_{MODEL}").glob("shard*/rayscores.json"):
    for r in json.loads(f.read_text())["rows"]:
        sc[r["mag"]][0].append(r["source_mean"])
        sc[r["mag"]][1].append(r["target_mean"])
sx = sorted(sc)
for idx, (colr, lab) in enumerate(((INK2, "source program"),
                                   (SEM, "target program"))):
    ys = [sum(sc[m][idx]) / len(sc[m][idx]) for m in sx]
    se = [(sum((v - mu) ** 2 for v in sc[m][idx])
           / (len(sc[m][idx]) * (len(sc[m][idx]) - 1))) ** 0.5
          for m, mu in zip(sx, ys)]
    axBm.fill_between(sx, [a - b for a, b in zip(ys, se)],
                      [a + b for a, b in zip(ys, se)], color=colr, alpha=0.16,
                      lw=0, zorder=2)
    axBm.plot(sx, ys, color=colr, lw=1.5, zorder=4, label=lab)
axBm.set_xlabel("distance along the semantic path", fontsize=6.4,
                color=INK2)
axBm.set_ylabel("log prob / token", fontsize=6.4, color=INK2)
axBm.legend(fontsize=5.0, frameon=False, loc="lower left", labelcolor=INK2,
            handlelength=1.4, borderaxespad=0.2)
axB.legend(fontsize=5.0, frameon=False, loc="center right",
           bbox_to_anchor=(0.78, 0.40), labelcolor=INK2, handlelength=1.4,
           borderaxespad=0.2)

# ---------------------------------------------------------------- panel c ---
axC = fig.add_subplot(gs[:, 2])
axC.set_title("(c)", fontsize=8.5, color=INK, loc="left", pad=4)
for kind, colr, lab in (("fwd", SEM, "semantic direction"),
                        ("rev", REV, "same axes, reversed"),
                        ("random", RND, "random direction")):
    x, y = curve(kind, "valid")
    axC.plot(x, y, color=colr, lw=1.5, marker="o", ms=2.6, label=lab)
axC.set_xlabel("perturbation magnitude", fontsize=6.4, color=INK2)
axC.set_ylabel("valid-program rate", fontsize=6.4, color=INK2)
axC.legend(fontsize=5.0, frameon=False, loc="upper right",
           bbox_to_anchor=(0.80, 0.50), labelcolor=INK2, handlelength=1.4,
           borderaxespad=0.2)

# ---------------------------------------------------------------- panel d ---
# Geometry, measured rather than projected. Every cosine is computed in the
# full 13.5M-dimensional state, so nothing here depends on a 2-D layout:
#   a modifier read off one base points the same way read off another;
#   square and cube are nearly the SAME axis, differing in how far they go;
#   neither aligns with what separates the base functions from each other.
axD = fig.add_subplot(gs[:, 3])
axD.set_title("(d)", fontsize=8.5, color=INK, loc="left", pad=4)
al = json.loads((R / f"atlas_{MODEL}" / "alignments.json").read_text())
groups = list(al["groups"].items())
rng = __import__("random").Random(0)
for y, (name, vals) in enumerate(groups):
    colr = SEM if y < 2 else MUTED
    axD.scatter([v for v in vals],
                [y + (rng.random() - 0.5) * 0.26 for _ in vals],
                s=9, color=colr, alpha=0.75, linewidths=0, zorder=4)
    mu = sum(vals) / len(vals)
    axD.plot([mu, mu], [y - 0.28, y + 0.28], color=INK, lw=1.4, zorder=5)
    axD.annotate(f"{mu:+.2f}", (mu, y - 0.34), fontsize=5.4, color=INK,
                 ha="center", va="bottom", zorder=6)
axD.axvline(0, color=BASELINE, lw=0.9, zorder=1)
axD.set_yticks(range(len(groups)))
axD.set_yticklabels([n for n, _ in groups], fontsize=5.4, color=INK2)
axD.set_ylim(-0.65, len(groups) - 0.35)
axD.invert_yaxis()
axD.set_xlim(-0.6, 1.0)
axD.set_xlabel("cosine between state displacements", fontsize=6.4, color=INK2)

for ax in (axA, axB, axBm, axC, axD):
    if ax not in (axBm, axD):
        ax.set_ylim(-0.04, 1.06)
    if ax not in (axA, axD):        # (d) is a cosine axis, and negative
        ax.set_xlim(0, 1.58)        # cosines are half of what it shows
    ax.tick_params(labelsize=5.6, colors=INK2, length=2.2, width=0.6)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.7)

for ext in ("png", "pdf"):
    fig.savefig(Path(__file__).parent / f"fig_geometry.{ext}",
                facecolor="white", bbox_inches="tight", pad_inches=0.02)
print("saved -> paper_figures/fig_geometry.png / .pdf")
for arm, m in (("sweep_prod", DOSE), ("sweep_max", DOSE), ("comp_prod", None),
               ("comp_max", None), ("rand_prod", None), ("ceil_sqprod", None),
               ("base_prod", None)):
    h, n = srate(arm, m)
    print(f"  (a) {arm:>12} m={str(m):>4}: {h}/{n}")
for kind in ("fwd", "rev", "random"):
    x, y = curve(kind, "valid")
    _, t = curve(kind, "hit_target")
    n = len(rays[kind][x[0]])
    print(f"  ({kind:>6}) n={n} per magnitude | valid "
          + " ".join(f"{v:.2f}" for v in y))
    print(f"           {'':>12} target " + " ".join(f"{v:.2f}" for v in t))

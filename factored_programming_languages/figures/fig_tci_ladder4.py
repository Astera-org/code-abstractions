"""fig_tci_ladder4 with the full control set on every rung.

The ladder is a one-dimensional layout -- position along the row IS the score --
so the four controls, which all sit near zero, would pile on top of each other
at the left end. They are fanned vertically within each factor row instead: the
claim (pooled factor) and its ceiling (exact) stay on the row's own line, and
the controls sit as a small comb just above and below it. Reading a row left to
right still gives control -> intervention -> ask; reading it top to bottom now
also shows that nothing lacking the factor's identity moves off the floor.

Original docstring follows.

Transport / Install / Compose ladders for all THREE factors.

fig_tci_ladder.py with the strategy factor added: each of transport and
install carries an f, a lambda and an s line, and compose carries the joint
f+lambda line plus (when its run exists) the three-factor f+lambda+s line.

f and lambda come from results/tci_<model> (run_tci_severed.py); s comes from
results/tcis_<model> (run_tci_strategy.py). The s cells are restricted to the
ones where the untouched severed recipient does NOT already emit the target
strategy -- otherwise half the install cells are degenerate and even the
random control scores 0.50.

Original docstring follows.

Transport / Install / Compose ladders: control -> intervention -> ask.

3 rows (the three operations) x 4 columns (the continuation cast + the
pythia control). Each panel is one ladder: a line spanning from the
control score to the best available score, with a marker per arm --

    ctrl    norm-matched random delta (the floor)
    LOO     leave-one-out pooled factor (the held-out claim)
    exact   the non-held-out contrast for this very cell
    prompted (severed)  asked in the prompt, then the description
            ablated exactly as in the intervention arms -- what the
            compiled store carries on its own
    prompted  asked in the prompt, words intact (the ceiling)

Score = fraction of cells where the code runs under the requested
language's own runner AND implements the target function -- so every
row asks both that the injected factor arrived and that the
recipient's own factor survived. Arms absent from the results are
skipped, so this renders on partial data.

    uv run factored_programming_languages/figures/fig_tci_ladder.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402

import sys                                                              # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tasks_strategy import detect_strategy                              # noqa: E402

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"

# --nine: read the nine-recipient-language reruns (tci9_/tcim9_/tcis9_/tcic3m9_,
# 2026-09-03/04, Python donor-only) instead of the four-language originals, and
# write the figure under a _9lang suffix so both versions coexist. The relabel
# controls (tcirel*) draw on the whole bank and are shared.
import sys as _sys
NINE = "--nine" in _sys.argv
SEV = "--sev" in _sys.argv          # normalise to the severed-ask ceiling (see load)
_S = "9" if NINE else ""
_TAG = ("_9lang" if NINE else "")
if "--set" in _sys.argv:            # e.g. --set 9L: the long-tail variant's directories
    _S = _sys.argv[_sys.argv.index("--set") + 1]
    _TAG = f"_{_S}"
_TAG += "_sevnorm" if SEV else ""
# --model-set MODEL=SET (repeatable): read ONE model from a different result
# set, e.g. pythia from its steel-man run (--model-set pythia-6.9b-tulu=9A:
# every row patched, words intact). If that set's severed runner recorded
# ablate_desc "none", the intervention arms kept the recipient's words, so
# under --sev the ceiling for that model is the plain prompted ask rather
# than the severed one (which would be a different condition from its arms).
OVERRIDE = {}
for _i, _a in enumerate(_sys.argv):
    if _a == "--model-set":
        _m, _, _st = _sys.argv[_i + 1].partition("=")
        OVERRIDE[_m] = _st


def set_of(model):
    return OVERRIDE.get(model, _S)


def _meta(model):
    try:
        return json.loads((R / f"tci{set_of(model)}_{model}" / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


def words_intact(model):
    """did the intervention arms of this model's set leave the words alone?"""
    return _meta(model).get("ablate_desc") == "none"


def ceil_of(model, op):
    """the ceiling arm for one rung: a merged set (assemble_9F.py) names it per
    operation -- transport severed (prompted_sev), install/compose unsevered
    (prompted); otherwise the set's single regime decides"""
    c = _meta(model).get("ceiling")
    if isinstance(c, dict) and op in c:
        return c[op]
    return "prompted" if words_intact(model) else "prompted_sev"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
INK, INK2 = "#111111", "#444444"
BASELINE, GRID = "#b8b8b8", "#e6e6e6"
ARMS = [("rand", "random control", "#8a8a8a", "o"),
        ("delta", "leave-one-out factor", "#20507e", "o"),
        ("exact", "same-context-only factor", "#2a78d6", "o"),   # was "exact factor" (authors, 2026-09-24)
        ("prompted", "prompted (no intervention)", "#a8531f", "D"),
        ("prompted_sev", "prompted, factor's words severed", "#d9a066", "D")]
# the controls, fanned vertically off the row's own line: (arm, label, marker,
# size, y offset). Every one asks "does something WITHOUT the factor's identity
# do this too?" -- so they belong on the same rung as the claim they check.
CTRL = [("wrong", "wrong value", "X", 11, -0.19),
        ("mean", "mean over values", "s", 8, 0.0),
        ("relabel", "random labelling", "D", 8, 0.19)]
CCOL = "#8a8a8a"
CANON = {"wrongf": "wrong", "wrongl": "wrong", "wrongs": "wrong",
         "meanf": "mean", "meanl": "mean"}
OPS = [("transport", "transport"), ("install", "install"),
       ("compose", "compose")]
MODELS = [("gemma-4-31B-it", "gemma-4-31B"),
          ("gemma-4-26b-a4b", "gemma-4-26B-A4B"),
          ("qwen3.6-27b", "Qwen3.6-27B"),
          ("pythia-6.9b-tulu", "pythia-6.9b (control)")]
# --models key=Label,key=Label ... : override the cast (2026-09-14, to add the attention-free RWKV-7 column).
# Pair it with --model-set key=SET when that model's cells live in a different result set.
if "--models" in _sys.argv:
    MODELS = [tuple(x.split("=", 1)) for x in _sys.argv[_sys.argv.index("--models") + 1].split(",")]
    _TAG += "_" + str(len(MODELS)) + "m"
# --out NAME writes somewhere else. Querying the printed numbers used to mean re-rendering with
# throwaway labels, which silently overwrote the real figure -- twice (authors, 2026-09-18).
_OUT = (_sys.argv[_sys.argv.index("--out") + 1] if "--out" in _sys.argv
        else f"fig_tci_ladder4{_TAG}")


NOCH = ("prompted", "prompted_sev")   # no intervention: channel-free; prompted_sev is drawn only in the raw figure
# one ladder per FACTOR: the function factor injected alone (CFS_F), the
# language factor alone (CFS_L), and for compose both together (F+L).
LINES = {"install": [("F", r"$f$"), ("L", r"$\lambda$"), ("S", r"$s$")],
         "transport": [("F", r"$f$"), ("L", r"$\lambda$"), ("S", r"$s$")],
         "compose": [("FL", r"$f,\lambda$"), ("FLS", r"$f,\lambda,s$")]}
YOFF = {1: [0.0], 2: [0.45, -0.45], 3: [0.66, 0.0, -0.66]}


def _cutword():
    """what was done to the description, from any loaded set's meta: masked or severed"""
    for mk, _ in MODELS:
        m = _meta(mk).get("transport_regime") or {}
        if m.get("ablate_desc") == "mask":
            return "masked"
    return "severed"


def load(model):
    """per (op, factor, arm): the score for THAT factor's claim --
    function execution for F, building in the target language for L,
    both for the joint F+L line. Hybrids use the kvrec channel.

    Records are first collected per CELL so that --sev can restrict every rung
    to the cells whose prompted_sev arm -- the target asked for directly in the
    prompt, then its specification ablated exactly as the intervention arms
    ablate the recipient's -- still produces the target. That arm is what the
    compiled store carries unsteered, so it is the ceiling an injected factor
    can be held to; cells the model cannot complete unsteered-and-severed are
    left out of every arm's denominator. Without --sev nothing is filtered and
    the numbers are the classic ones."""
    S = set_of(model)
    d = R / f"tci{S}_{model}"
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    free = defaultdict(lambda: defaultdict(list))
    if not d.is_dir():
        return by, free
    # compose is read from the NORM-MATCHED run: summing factors inflates the
    # delta's norm, and at alpha=1 the composed arm was over-dosed relative to
    # the single-contrast arms it is compared against
    dm = R / f"tcim{S}_{model}"
    files = [f for f in d.glob("c_*.json")
             if not (dm.is_dir() and f.name.startswith("c_compose"))]
    files += list(dm.glob("c_compose_*.json")) if dm.is_dir() else []
    recs = defaultdict(dict)            # (op, fac, cell, lang) -> {(arm, ch): val}
    sev_ok = {}                         # (op, fac, cell, lang) -> prompted_sev claim
    for f in files:
        r = json.loads(f.read_text())
        fac = r.get("factor", "F")
        if r["op"] == "compose":     # compose patches both factors at once;
            fac = "FL"               # the runner tags those records "F"
        hit_fn = bool(r.get("hit_fn", r["hit"] if fac != "L" else False))
        if fac == "L":
            hit_fn = bool(r.get("hit_fn", False))
        built = bool(r.get("built"))
        val = {"F": hit_fn, "L": built and hit_fn,
               "FL": hit_fn and built}[fac]
        ch = r.get("channel", "kv")
        ck = (r["op"], fac, r["cell"], r["lang"])
        CEIL = ceil_of(model, r["op"])
        if r["arm"] == CEIL:                    # the ceiling of this rung
            sev_ok[ck] = val
            if fac == "F":       # the language claim: the ceiling ask still builds in lam
                sev_ok[(r["op"], "L", r["cell"], r["lang"])] = built
        recs[ck][(r["arm"], ch)] = val          # prompted and prompted_sev stay as references
        if r["arm"] in NOCH and fac == "F":     # one prompted generation answers both
            recs[(r["op"], "L", r["cell"], r["lang"])][(r["arm"], ch)] = built and hit_fn
    for ck, arms in recs.items():
        if SEV and not sev_ok.get(ck, False):
            continue
        for (arm, ch), val in arms.items():
            if arm in NOCH:
                free[(ck[0], ck[1])][arm].append(val)
            else:
                by[(ck[0], ck[1])][ch][CANON.get(arm, arm)].append(val)
    # ---- the strategy factor, from the separate run ----------------------
    ds = R / f"tcis{S}_{model}"
    if ds.is_dir():
        cells = defaultdict(dict)
        for f in ds.glob("c_*.json"):
            r = json.loads(f.read_text())
            code = r.get("code") or ""
            r["strategy"] = (detect_strategy(code, r["lang"]) if code
                             else "unknown")
            r["hit_strat"] = r["strategy"] == r["want_strategy"]
            r["hit"] = bool(r["hit_strat"] and r["impl"] == r["cell"]
                            and r["built"])
            cells[(r["op"], r["cell"], r["lang"], r["want_strategy"])][
                (r["arm"], r.get("channel", "kv"))] = r
        for key, v in cells.items():
            base = v.get(("none", "kv"))
            if base is None or base["hit_strat"]:
                continue          # nothing to move in this cell
            sev = v.get((ceil_of(model, key[0]), "kv"))
            if SEV and not (sev is not None and sev["hit"]):
                continue
            for (arm, ch), r in v.items():
                if arm in NOCH:
                    free[(key[0], "S")][arm].append(r["hit"])
                else:
                    by[(key[0], "S")][ch][CANON.get(arm, arm)].append(r["hit"])
    # ---- the three-factor compose, from its own run ----------------------
    dc = R / f"tcic3m{S}_{model}"         # norm-matched three-factor compose
    if dc.is_dir():
        cells3 = defaultdict(dict)
        for f in dc.glob("c_*.json"):
            r = json.loads(f.read_text())
            code = r.get("code") or ""
            strat_ = detect_strategy(code, r["lang"]) if code else "unknown"
            val = bool(strat_ == r["want_strategy"] and r["impl"] == r["cell"]
                       and r["built"])
            cells3[(r["cell"], r["lang"], r["want_strategy"])][
                (r["arm"], r.get("channel", "kv"))] = val
        for key, v in cells3.items():
            if SEV and not v.get((ceil_of(model, "compose"), "kv"), False):
                continue
            for (arm, ch), val in v.items():
                if arm in NOCH:
                    free[("compose", "FLS")][arm].append(val)
                else:
                    by[("compose", "FLS")][ch][CANON.get(arm, arm)].append(val)
    # the relabel controls run on a fake grid and have no severed-ask twin;
    # they are the same in both normalisations
    # relabel controls: the set's own tagged runs when they exist (9F ran every
    # kind under its regime), otherwise the shared untagged originals
    _ST = set_of(model)
    # --relabel-tag TAG: read every null kind from tag TAG (e.g. 9F, the controls run under the ladder's own regime by
    # run_9F_relabel.sh) instead of the set's own tag; without it the lookup below is unchanged
    _RT = _sys.argv[_sys.argv.index("--relabel-tag") + 1] if "--relabel-tag" in _sys.argv else None
    for pref, key in (("tcirel", ("transport", "F")), ("tcirelc", ("transport", "L")),
                      ("tcirels", ("transport", "S")), ("tcireli", ("install", "F")),
                      ("tcirelic", ("install", "L")), ("tcirelsi", ("install", "S")),
                      ("tcirelx", ("compose", "FL")), ("tcirelx3", ("compose", "FLS"))):
        dirs = sorted(R.glob(f"{pref}{_RT or _ST}_{model}_s*")) if (_RT or _ST) else []
        # the 9F set draws only its own controls (run under the 9F regime); older
        # sets fall back to the shared untagged originals for the non-strategy kinds
        if not dirs and _RT is None and _ST != "9F" and pref not in ("tcirels", "tcirelsi"):
            dirs = sorted(R.glob(f"{pref}_{model}_s*"))
        for dr in dirs:
            for f in dr.glob("c_*.json"):
                r = json.loads(f.read_text())
                if r["arm"] == "delta":
                    by[key][r.get("channel", "kv")]["relabel"].append(
                        bool(r["hit"]))
    out = defaultdict(dict)
    for key, chans in by.items():
        best = next((c for c in ("kvrec", "kv", "rec") if c in chans), None)
        if best:
            out[key] = dict(chans[best])
    return out, free


W, H = 5.8, 2.86
if "--width" in _sys.argv:           # override the base width (authors, 2026-09-18: "squeeze the
    W = float(_sys.argv[_sys.argv.index("--width") + 1])   # figure a bit tighter")
if "--height" in _sys.argv:
    H = float(_sys.argv[_sys.argv.index("--height") + 1])
EQUAL = "--equal" in _sys.argv        # every column full width (no narrow control panel), e.g. a suffix-vs-online pair
if EQUAL:
    W *= len(MODELS) / 3.46
elif len(MODELS) != 4:               # keep per-panel width constant when the cast changes
    W *= (len(MODELS) - 1 + 0.46) / 3.46
# compose carries two lines where transport and install carry three, so it
# does not need an equal share of the height
fig, axes = plt.subplots(len(OPS), len(MODELS), figsize=(W, H), dpi=300,
                         sharex=True, squeeze=False,
                         gridspec_kw={"height_ratios":
                                      [len(LINES[o]) for o, _ in OPS],
                                      "width_ratios":
                                      ([1.0] * len(MODELS) if EQUAL else
                                       [1.0] * (len(MODELS) - 1) + [0.46]
                                       if len(MODELS) != 4 else [1.0, 1.0, 1.0, 0.46])})
fig.patch.set_facecolor("white")
summary = []
LOADED = {mk: load(mk) for mk, _ in MODELS}
CUTWORD = _cutword()
for ci, (mk, mlab) in enumerate(MODELS):
    data, free = LOADED[mk]
    for ri, (op, olab) in enumerate(OPS):
        ax = axes[ri, ci]
        ax.set_facecolor("white")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.set_xlim(-0.04, 1.04)
        # fit the row to its lines instead of a fixed +-1.05: compose
        # carries two, the others three
        _pad = max(YOFF[len(LINES[op])]) + 0.50
        ax.set_ylim(-_pad, _pad)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=5.6, colors=INK2, length=2,
                       width=0.6)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        lines = LINES[op]
        ys = YOFF[len(lines)]
        drew = False
        for (fac, flab), y in zip(lines, ys):
            arms = data.get((op, fac), {})
            fr = free.get((op, fac), {})
            for arm, lab, col, mkr in ARMS:          # prompted references
                if arm not in NOCH or arm not in fr:
                    continue
                if arm == "prompted_sev" and (SEV or op != "transport"):
                    continue      # under --sev it is the denominator; elsewhere the rungs are unsevered
                v = float(np.mean(fr[arm]))
                ax.plot([v, v], [y - 0.26, y + 0.26], color=col, lw=0.9,
                        ls=(0, (3, 2)) if arm == "prompted" else (0, (1.2, 1.6)), zorder=3, alpha=0.9)
                summary.append((mk, op, fac, arm, v, len(fr[arm])))
            if flab and ci == 0:        # notation on the left, every row
                ax.text(-0.05, y, flab, fontsize=6.2, color="#666666",
                        ha="right", va="center")
            vals = {arm: (float(np.mean(h)), len(h))
                    for arm in ("rand", "delta", "exact")
                    if (h := arms.get(arm))}
            if not vals:
                continue
            drew = True
            lo = vals.get("rand", (0.0, 0))[0]
            hi = max(v for v, _ in vals.values())
            ax.plot([lo, hi], [y, y], color=BASELINE, lw=1.0, zorder=2,
                    solid_capstyle="round")
            # nested markers: the LOO dot sits inside the larger open
            # exact ring, so coincident values stay legible
            for arm, lab, col, mkr in ARMS:
                if arm not in vals:
                    continue
                v, n = vals[arm]
                if arm == "exact":       # larger, open: LOO sits inside it
                    ax.scatter([v], [y], s=34, marker="o", facecolor="none",
                               edgecolor=col, linewidth=0.9, zorder=4)
                elif arm == "delta":
                    ax.scatter([v], [y], s=11, marker="o", color=col,
                               zorder=6)
                else:
                    ax.scatter([v], [y], s=14, marker=mkr, color=col,
                               edgecolor="white", linewidth=0.6, zorder=5)
                summary.append((mk, op, fac, arm, v, n))
            for carm, _cl, cmk, csz, cdy in CTRL:      # the control comb
                h = arms.get(carm)
                if not h:
                    continue
                ax.scatter([float(np.mean(h))], [y + cdy], s=csz, marker=cmk,
                           color=CCOL, linewidths=0.0, zorder=3)
            d_ = vals.get("delta")
            if d_:
                ax.annotate(f"{d_[0]:.2f}", (d_[0], y), fontsize=4.6,
                            color="#20507e", ha="center", va="bottom",
                            xytext=(0, 3), textcoords="offset points")
        if not drew:
            # an empty panel: no data yet, or -- under --sev -- no cell whose
            # severed ask completes (pythia: 0 of 360), so nothing to normalise to
            _has = (R / f"tci{set_of(mk)}_{mk}").is_dir()
            ax.text(0.5, 0, f"no cell completes the {CUTWORD} ask" if (SEV and _has) else "not run",
                    ha="center", va="center", fontsize=5.4, color="#999999")
        if ri == 0:
            if mk in OVERRIDE:
                # a model whose meta says severing is unavailable gets no subtitle: in an all-recurrent
                # model there are no per-token rows to ablate, so "words intact" names a distinction that does
                # not exist there rather than a condition we chose (2026-09-14).
                sub = ("" if _meta(mk).get("severing") == "unavailable"
                       else "\nall rows patched, words intact" if words_intact(mk)
                       else "" if "--no-set-label" in _sys.argv     # the column label already says what the set is
                       else f"\nset {OVERRIDE[mk]}")
                ax.set_title(mlab + sub, fontsize=6.0 if sub else 6.8, color=INK, pad=3 if sub else 4)
            else:
                ax.set_title(mlab, fontsize=6.8, color=INK, pad=4)
        if ci == 0:
            # y=0 is the axes' BOTTOM edge, so the row labels were sitting
            # under their rows rather than beside them
            ax.text(-0.27, 0.5, olab, transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=7.0, color=INK)
        ax.set_xticks([0, 0.5, 1.0])
        if ri != len(OPS) - 1:
            ax.set_xticklabels([])

fig.text(0.56, 0.232,
         # the conditioning ("among cells the unsteered ask, severed for transport, completes")
         # is the same in both modes and lives in the caption text, not on the axis
         # (authors, 2026-09-18: "get rid of everything after correctly")
         "fraction of cells where the injected factor is present in code that runs correctly",
         fontsize=6.6, color=INK2, ha="center")
_LH = []
for arm, l, c, m in ARMS:
    if arm == "prompted_sev" and SEV:
        continue                # the denominator of this figure, not a rung
    if arm in NOCH:             # drawn as dashed rules, not markers
        _LH.append(Line2D([], [], color=c, lw=0.9,
                          ls=(0, (3, 2)) if arm == "prompted" else (0, (1.2, 1.6)), label=l))
    elif arm == "exact":
        _LH.append(Line2D([], [], marker="o", linestyle="none",
                          markerfacecolor="none", markeredgecolor=c,
                          markeredgewidth=0.9, markersize=4.8, label=l))
    elif arm == "delta":
        _LH.append(Line2D([], [], color=c, marker="o", linestyle="none",
                          markersize=2.9, label=l))
    else:
        _LH.append(Line2D([], [], color=c, marker=m, linestyle="none",
                          markersize=3.4, label=l))
_LH += [Line2D([], [], color=CCOL, marker=m, linestyle="none",
               markersize=3.2, label=l) for _a, l, m, _s, _d in CTRL]
fig.legend(handles=_LH,
           fontsize=5.8, frameon=False, ncol=4 if SEV else 5, loc="lower center",
           bbox_to_anchor=(0.56, 0.105), handletextpad=0.3,
           columnspacing=1.15)
fig.subplots_adjust(left=0.145, right=0.997, top=0.90, bottom=0.295,
                    hspace=0.20, wspace=0.10)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"{_OUT}.{ext}", facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
print("saved -> fig_tci_ladder4")
for mk, op, fac, arm, v, n in summary:
    print(f"  {mk:>18} {op:>9} {fac:>3} {arm:>13}: {v:.2f} (n={n})")

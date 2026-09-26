"""Does the donor state make the recipient write something PROMPTING does not produce?

`fig4_transport_structure` shows one matched triplet: a Haskell fold donor, a Java recipient that
writes a stream reduction after receiving the donor state, and the same Java recipient writing a
while loop when merely prompted. This sweeps that comparison over every ordered language pair.

Each cell of the grid forces the single-donor arm to take its donor from one language
(`run_tci_strategy.py --single-donor-lang`) and compares three generations:

    donor form     the donor cell's own prompted generation
    steered form   the recipient after receiving the donor-derived state
    prompted form  the same recipient asked for the same strategy in words

(a) how often the steered form differs from the recipient's prompted default -- i.e. whether the
    intervention changes the shape of the program at all. The diagonal is empty by construction: a
    recipient's own language is never in its donor pool.
    The column set off on the right is the POOLED held-out S factor (`delta`, averaged over the
    other languages) from the 9F run, scored identically. It moves the form less often than a
    single donor does (31% vs 44%), which is what averaging over donors is expected to do to
    donor-specific idioms -- and it is why the pooled arm cannot appear in (b): with no single
    donor there is no donor form to match.

(b) the cells that can discriminate -- where the donor's form differs from the prompted default, so
    asking in words and sending the state predict different programs. Asking in words lands on the
    donor's form in none of them, by construction; that is what makes the cell discriminating. The
    comparison the panel draws is therefore against a language-matched shuffle: within each
    recipient language, how often a form drawn from that language's own distribution of steered
    outputs would coincide with the donor's. Steering exceeds both.

The claim this supports is not that the donor's organisation reliably transports -- it does not, the
recipient's default wins about half the time. It is that the donor state is not a slower way of
saying the strategy out loud: it moves the program off the prompted default in half of these cells,
and onto the donor's specific form more often than the recipient language's own output distribution
would deliver by chance.

Structure labels come from `code_structure.structure` (pipeline / helper_acc / recursion /
loop_acc). A cell is scorable for (a) as soon as the RECIPIENT'S PROMPTED DEFAULT has a label:
576 of 576 (100%). An unlabelled steered program matched none of the four patterns, so it is not
the pattern the default matched, and it counts as differing (12 cells). Dropping those instead would make the denominator depend on the
steered outcome. Comparing the raw text rather than the structure is the wrong granularity: only
12% of steered generations are textually identical to the prompted one, so text comparison calls
88% of cells "different" on variable names and whitespace. Two earlier losses are fixed and both
mattered: generations holding more
than one fenced block were dropped by `extract_code` (125 cells, 115 recoverable from the first
block), and `_self_recursive` was eating typed definition lines, so `fn solve(n: u64)` recursions
labelled `unknown` -- which is why Rust read as an inert recipient. Recursion is also detected in
inner workers now, not only in `solve` itself: a JS `const countEvens = (num) => ... countEvens(...)`
matched nothing before, and an inner worker taking an extra argument is the hand-written fold, so
it reads `helper_acc` rather than falling through a name whitelist. With those fixed the labelled
set is no longer selected on the steer having landed: `hit_strat` is 54% over all cells and 55%
over scorable ones, against 54% vs 70% before.

    uv run factored_programming_languages/paper_figures/fig_donor_recipient.py \
        --model gemma-4-31B-it --op transport
"""
import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                  # noqa: E402

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["STIXGeneral", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})
HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
sys.path.insert(0, str(HERE.parent))
from code_structure import structure                                   # noqa: E402
from elicit import find_containers                                    # noqa: E402


def code_of(r):
    """`extract_code` returns None when a generation holds more than one fenced block, which is
    what a rambling answer looks like: a solution, then "Wait, ..." and a second attempt. Under a
    masked description the model rambles often -- 125 of 576 grid cells -- and dropping those cells
    silently conditions the denominator on the steer having landed. Take the FIRST block, the same
    rule `regrade_subsets.py` uses."""
    if r.get("code"):
        return r["code"]
    cs = find_containers(r.get("text") or "")
    return sorted(cs, key=lambda c: c[2][0])[0][1].strip() if cs else None

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-31B-it")
ap.add_argument("--op", default="transport", choices=["transport", "install"],
                help="transport: the recipient's cue asks for the OTHER strategy and the state "
                     "must override it. install: the recipient gets no cue and the state supplies "
                     "the organisation from nothing.")
ap.add_argument("--draws", type=int, default=20000)
ap.add_argument("--report-differs", action="store_true",
                help="print, among successful steers (recipient writes the requested strategy), how many programs "
                     "differ in structure from the recipient's own PROMPTED rendering of that strategy; then exit")
ap.add_argument("--differs-figure", action="store_true",
                help="draw ONE grid: among successful steers (recipient writes the requested strategy), the share whose "
                     "program differs in structure from the recipient's own prompted rendering of that strategy "
                     "-> fig_donor_recipient_<model>_<op>_differs; then exit")
ap.add_argument("--all-cells", action="store_true",
                help="do NOT condition on the generation implementing the requested function. "
                     "Off by default: an output that computes the wrong thing has no organisation "
                     "worth attributing to the donor, and including those cells manufactured the "
                     "whole hand-written-fold effect (gemma-4-26b-a4b +12.5%% -> -0.6%%).")
a, _ = ap.parse_known_args()
MODEL, OP = a.model, a.op
PREFIX = "donorgrid" if OP == "transport" else "donorgridI"
LANGS = ["c", "cpp", "go", "java", "javascript", "ocaml", "ruby", "rust", "haskell"]
NICE = {"c": "C", "cpp": "C++", "go": "Go", "java": "Java", "javascript": "JS",
        "ocaml": "OCaml", "ruby": "Ruby", "rust": "Rust", "haskell": "Haskell"}
MINN = 3                                    # below this a cell is drawn as too thin to read

def task_of(r):
    """the recipient's task: 9F records carry it in `cell`; the RWKV 9R set writes `cell` as "<B>:<A>-<strategy>" and the
    task in `want`"""
    return r.get("want") or r["cell"]


def fn_ok(r):
    """the generation implements the requested function (validity gate). Derived from `impl` so it means the same thing in
    every set: the RWKV runner's `hit_fn` on S cells is the COMBINED strategy-and-function claim."""
    return r.get("impl") == task_of(r) if "impl" in r else bool(r.get("hit_fn"))


def strat_ok(r):
    return (r.get("strategy") == r.get("want_strategy")) if "strategy" in r else bool(r.get("hit_strat"))


# the recipient's own prompted arms: the assembled strategy set of the ladder (9F / 9M2, RWKV 9R); --base-dir overrides
BASE = next((R / d for d in (f"tcis9F_{MODEL}", f"tci9R_{MODEL}", f"tcis9M2_{MODEL}", f"tcis9U1_{MODEL}") if (R / d).exists()), R / f"tcis9F_{MODEL}")
if "--base-dir" in sys.argv: BASE = Path(sys.argv[sys.argv.index("--base-dir") + 1])
P, POOLED = {}, []
for f in glob.glob(str(BASE / "c_*.json")):
    r = json.loads(Path(f).read_text())
    if r["op"] != OP or r.get("factor", "S") != "S":
        continue
    if r["arm"] == "prompted":
        _c = code_of(r)
        if _c:
            P[(task_of(r), r["lang"], r["want_strategy"])] = structure(_c, r["lang"])
    elif r["arm"] == "delta":            # the pooled, held-out S factor (run_tci_strategy.py:32)
        POOLED.append(r)

if not glob.glob(str(R / f"{PREFIX}_{MODEL}_*" / "c_*.json")):
    sys.exit(f"no grid data for {MODEL} / {OP} (expected {PREFIX}_{MODEL}_<lang>/)")

cells, RAW, HIT, NVALID = [], 0, [0, 0], 0
for f in glob.glob(str(R / f"{PREFIX}_{MODEL}_*" / "c_*.json")):
    r = json.loads(Path(f).read_text())
    if not r.get("donor"):
        continue
    RAW += 1
    HIT[1] += 1; HIT[0] += strat_ok(r)
    df, dl, _ = r["donor"]
    sd = P.get((df, dl, r["want_strategy"]))
    sp = P.get((task_of(r), r["lang"], r["want_strategy"]))
    _c = code_of(r)
    ss = structure(_c, r["lang"]) if _c else "unknown"
    # A cell is scorable for (a) as soon as the recipient's PROMPTED DEFAULT has a label: an
    # unlabelled steered program matched none of the four patterns, so it is not the pattern the
    # default matched, and it counts as differing. Dropping it instead would make the denominator
    # depend on the steered outcome, which is what let the old denominator track hit_strat.
    if sp is None or sp == "unknown":
        continue
    # VALIDITY GATE. `hit_fn` is true when the generation actually implements the requested
    # function under the recipient language's runner. A broken generation still parses as some
    # structure -- often a named helper with an extra argument -- so leaving those cells in
    # attributes the donor an organisation the program does not have.
    if fn_ok(r):
        NVALID += 1
    elif not a.all_cells:
        continue
    cells.append({"d": dl, "r": r["lang"], "sd": sd, "sp": sp, "ss": ss,
                  "t": task_of(r), "ws": r["want_strategy"],
                  "hit": strat_ok(r)})

if a.report_differs:
    hits = [c for c in cells if c["hit"]]
    same = sum(c["ss"] == c["sp"] for c in hits); asdonor = sum(c["ss"] != c["sp"] and c["ss"] == c["sd"] for c in hits)
    other = len(hits) - same - asdonor
    print(f"{MODEL} / {OP}: valid cells {len(cells)}, successful steers (recipient writes the requested strategy) {len(hits)}")
    print(f"  steered program has the SAME structure as the recipient prompted in words for that strategy: {same}/{len(hits)} = {same/len(hits):.0%}")
    print(f"  DIFFERS from the prompted rendering: {len(hits)-same}/{len(hits)} = {(len(hits)-same)/len(hits):.0%}"
          f"  (of which = the donor's own form {asdonor}, some third form {other})")
    dd = [c for c in hits if c["sd"] is not None and c["sd"] != c["sp"]]
    if dd: print(f"  where the donor's form differs from the prompted one ({len(dd)} hits): steered = donor form {sum(c['ss']==c['sd'] for c in dd)}, "
                 f"= prompted form {sum(c['ss']==c['sp'] for c in dd)}, other {sum(c['ss']!=c['sd'] and c['ss']!=c['sp'] for c in dd)}")
    by = collections.defaultdict(lambda: [0, 0])
    for c in hits: by[c["r"]][1] += 1; by[c["r"]][0] += c["ss"] != c["sp"]
    print("  differs-from-prompted by recipient language: " + "  ".join(f"{NICE[l]} {by[l][0]}/{by[l][1]}" for l in LANGS if by[l][1]))
    ph = [r for r in POOLED if strat_ok(r) and (a.all_cells or fn_ok(r))]
    pd = sum(structure(code_of(r), r["lang"]) != P.get((task_of(r), r["lang"], r["want_strategy"])) for r in ph if code_of(r))
    print(f"  pooled (held-out mean) arm: successful steers {len(ph)}, differ from prompted {pd}/{len(ph)}")
    sys.exit(0)

# NUMERATOR = the recipient actually writes the REQUESTED strategy (`hit_strat`), not merely a
# form unlike the prompted default. Under transport the cue asks for the opposite strategy, so a
# steer that does nothing leaves the recipient obeying the cue -- which differs from the prompted
# arm and used to be drawn as a hit. On the gemmas 88-94% of "differs" cells were failed steers,
# i.e. the map was brightest where transport worked worst.
grid = collections.defaultdict(lambda: [0, 0])          # (donor, recip) -> [arrived, n]
for c in cells:
    g = grid[(c["d"], c["r"])]
    g[1] += 1
    g[0] += c["hit"]

pool = collections.defaultdict(lambda: [0, 0])          # recipient -> [differs, n]
for r in POOLED:
    sp = P.get((task_of(r), r["lang"], r["want_strategy"]))
    _c = code_of(r)
    ss = structure(_c, r["lang"]) if _c else "unknown"
    if sp is None or sp == "unknown":
        continue
    if not a.all_cells and not fn_ok(r):
        continue
    g = pool[r["lang"]]
    g[1] += 1
    g[0] += strat_ok(r)

CMAP = LinearSegmentedColormap.from_list("d", ["#f4f6f8", "#9fc0d9", "#20507e"])


def draw_grid(ax, gmap, pmap, title):
    """donor (columns) x recipient (rows) grid + the pooled column; each cell `num/den`, colour = num/den on a 0-1 scale"""
    for i, rec in enumerate(LANGS):
        for j, don in enumerate(LANGS):
            if rec == don:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#ffffff", edgecolor="#e8e8e8", lw=0.8, hatch="////")); continue
            num, n = gmap.get((don, rec), [0, 0])
            if n < MINN:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#fbfbfb", edgecolor="white", lw=0.8))
                ax.text(j, i, "·", fontsize=5, color="#c0c0c0", ha="center", va="center"); continue
            frac = num / n
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=CMAP(frac), edgecolor="white", lw=0.8))
            ax.text(j, i, f"{num}/{n}", fontsize=4.0, ha="center", va="center", color="white" if frac > 0.55 else "#333333")
    PX = len(LANGS) + 0.55
    for i, rec in enumerate(LANGS):
        num, n = pmap.get(rec, [0, 0])
        if n < MINN:
            ax.add_patch(plt.Rectangle((PX - .5, i - .5), 1, 1, facecolor="#fbfbfb", edgecolor="white", lw=0.8))
            ax.text(PX, i, "·", fontsize=5, color="#c0c0c0", ha="center", va="center"); continue
        frac = num / n
        ax.add_patch(plt.Rectangle((PX - .5, i - .5), 1, 1, facecolor=CMAP(frac), edgecolor="white", lw=0.8))
        ax.text(PX, i, f"{num}/{n}", fontsize=4.0, ha="center", va="center", color="white" if frac > 0.55 else "#333333")
    ax.plot([PX - 0.80, PX - 0.80], [-0.5, len(LANGS) - 0.5], color="#cccccc", lw=0.6, zorder=4)
    ax.set_xlim(-0.5, PX + 0.5); ax.set_ylim(len(LANGS) - 0.5, -0.5)
    ax.set_xticks(list(range(len(LANGS))) + [PX]); ax.set_yticks(range(len(LANGS)))
    ax.set_xticklabels([NICE[l] for l in LANGS] + ["pooled"], fontsize=4.8, rotation=45, ha="right", color="#444444")
    ax.set_yticklabels([NICE[l] for l in LANGS], fontsize=4.8, color="#444444"); ax.tick_params(length=0, pad=1.5)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    ax.set_xlabel("donor language", fontsize=5.6, color="#333333", labelpad=2); ax.set_ylabel("recipient language", fontsize=5.6, color="#333333", labelpad=2)
    ax.set_title(title, fontsize=5.8, color="#333333", pad=4, loc="left")


if a.differs_figure:
    # Classifier-free success (the authors, 2026-09-24): a steer SUCCEEDED when the program is not the recipient's prompted
    # rendering of the CUE's (opposite) strategy -- the state overrode the cue. `hit_strat` was not used because the strategy
    # classifier scores a fold written as a recursive accumulator helper as "not iteration", which removed exactly the
    # Haskell -> Java cells where the steered program differs from the prompted one.
    # numerator: overriding steers whose program's structure differs from the recipient's PROMPTED rendering of the
    # REQUESTED strategy; denominator: overriding steers (validity-gated as above).
    OPP = {"iter": "rec", "rec": "iter"}
    # what the recipient would have written had the state not transferred: under TRANSPORT the prompted rendering of the
    # cue's (opposite) strategy; under INSTALL (no cue) its unsteered default -- the 9F run's `none` arm for that cell.
    D0 = {}
    for f in glob.glob(str(BASE / "c_*.json")):
        r = json.loads(Path(f).read_text())
        if r["op"] == OP and r["arm"] == "none" and r.get("factor", "S") == "S":
            _c = code_of(r); D0[(task_of(r), r["lang"], r["want_strategy"])] = structure(_c, r["lang"]) if _c else "unknown"
    def baseline(cell, lang, ws):
        return P.get((cell, lang, OPP[ws])) if OP == "transport" else D0.get((cell, lang, ws))
    g2 = collections.defaultdict(lambda: [0, 0]); p2 = collections.defaultdict(lambda: [0, 0])
    for c in cells:
        b = baseline(c["t"], c["r"], c["ws"])
        if b in (None, "unknown") or c["ss"] == b: continue                # the steer changed nothing -> not a success
        g = g2[(c["d"], c["r"])]; g[1] += 1; g[0] += c["ss"] != c["sp"]    # differs from BOTH the baseline and the prompted rendering
    for r in POOLED:
        if not a.all_cells and not fn_ok(r): continue
        sp = P.get((task_of(r), r["lang"], r["want_strategy"])); b = baseline(task_of(r), r["lang"], r["want_strategy"]); _c = code_of(r)
        if sp in (None, "unknown") or b in (None, "unknown"): continue
        ss = structure(_c, r["lang"]) if _c else "unknown"
        if ss == b: continue
        g = p2[r["lang"]]; g[1] += 1; g[0] += ss != sp
    fig, ax = plt.subplots(figsize=(4.2, 2.65), dpi=300); fig.patch.set_facecolor("white")
    ttl = ("steered program differs from the prompted rendering  |  steer changed the program"
           + ("  (baseline: prompted for the cue's strategy)" if OP == "transport" else "  (baseline: unsteered default)") + f"   [{MODEL} · {OP}]")
    draw_grid(ax, g2, p2, ttl)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_donor_recipient_{MODEL}_{OP}_differs.{ext}", facecolor="white", bbox_inches="tight", pad_inches=0.03)
    tn = sum(v[1] for v in g2.values()); td = sum(v[0] for v in g2.values()); pn = sum(v[1] for v in p2.values()); pd_ = sum(v[0] for v in p2.values())
    print(f"saved -> fig_donor_recipient_{MODEL}_{OP}_differs   ({td}/{tn} = {(td / tn if tn else 0):.0%} of overriding single-donor steers differ from the prompted rendering; pooled {pd_}/{pn})")
    sys.exit(0)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.65), dpi=300,
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.30})
fig.patch.set_facecolor("white")

for i, rec in enumerate(LANGS):
    for j, don in enumerate(LANGS):
        if rec == don:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#ffffff",
                                       edgecolor="#e8e8e8", lw=0.8, hatch="////"))
            continue
        hits, n = grid.get((don, rec), [0, 0])
        if n < MINN:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#fbfbfb",
                                       edgecolor="white", lw=0.8))
            ax.text(j, i, "·", fontsize=5, color="#c0c0c0", ha="center", va="center")
            continue
        frac = hits / n
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=CMAP(frac),
                                   edgecolor="white", lw=0.8))
        ax.text(j, i, f"{hits}/{n}", fontsize=4.0, ha="center", va="center",
                color="white" if frac > 0.55 else "#333333")
PX = len(LANGS) + 0.55                                  # pooled column, set off by a gap
for i, rec in enumerate(LANGS):
    hits, n = pool.get(rec, [0, 0])
    if n < MINN:
        ax.add_patch(plt.Rectangle((PX - .5, i - .5), 1, 1, facecolor="#fbfbfb",
                                   edgecolor="white", lw=0.8))
        ax.text(PX, i, "·", fontsize=5, color="#c0c0c0", ha="center", va="center")
        continue
    frac = hits / n
    ax.add_patch(plt.Rectangle((PX - .5, i - .5), 1, 1, facecolor=CMAP(frac),
                               edgecolor="white", lw=0.8))
    ax.text(PX, i, f"{hits}/{n}", fontsize=4.0, ha="center", va="center",
            color="white" if frac > 0.55 else "#333333")
ax.plot([PX - 0.80, PX - 0.80], [-0.5, len(LANGS) - 0.5], color="#cccccc", lw=0.6, zorder=4)

ax.set_xlim(-0.5, PX + 0.5); ax.set_ylim(len(LANGS) - 0.5, -0.5)
ax.set_xticks(list(range(len(LANGS))) + [PX]); ax.set_yticks(range(len(LANGS)))
ax.set_xticklabels([NICE[l] for l in LANGS] + ["pooled"], fontsize=4.8, rotation=45,
                   ha="right", color="#444444")
ax.set_yticklabels([NICE[l] for l in LANGS], fontsize=4.8, color="#444444")
ax.tick_params(length=0, pad=1.5)
for sp_ in ax.spines.values():
    sp_.set_visible(False)
ax.set_xlabel("donor language", fontsize=5.6, color="#333333", labelpad=2)
ax.set_ylabel("recipient language", fontsize=5.6, color="#333333", labelpad=2)
ax.set_title(r"$\bf{(a)}$  recipient writes the requested strategy"
             f"      [{MODEL} · {OP}]",
             fontsize=5.8, color="#333333", pad=4, loc="left")

# ---- (b) what the recipient writes, split by the DONOR's own form -------------------------
# The three-way split (default / donor's form / a third form) hides the effect: a recipient given
# a functional donor rarely writes the donor's literal form, it writes the donor's ORGANISATION in
# its own idiom -- helper_acc, the fold written out by hand. That lands in "a third form". So the
# panel shows the four structures directly. Rows are paired: only recipient cells holding BOTH a
# canonical-form donor and an off-canonical one, so recipient, task and requested strategy are
# fixed inside a cell and only the donor's form moves.
CANON = {"iter": "loop_acc", "rec": "recursion"}
bycell = collections.defaultdict(list)
for c in cells:
    bycell[(c["t"], c["ws"], c["r"])].append(c)

rows_d = {"prompted": collections.Counter(), "canon": collections.Counter(),
          "off": collections.Counter()}
NPAIR = 0
for (t, ws, r), g in bycell.items():
    canon = CANON[ws]
    A = [c for c in g if c["sd"] == canon]
    B = [c for c in g if c["sd"] not in (None, "unknown", canon)]
    if not A or not B:
        continue
    NPAIR += 1
    for c in A:
        rows_d["canon"][c["ss"]] += 1
        rows_d["prompted"][c["sp"]] += 1
    for c in B:
        rows_d["off"][c["ss"]] += 1
        rows_d["prompted"][c["sp"]] += 1

_groups = []
for (t, ws, r_) , g in bycell.items():
    _c = CANON[ws]
    if any(x["sd"] == _c for x in g) and any(
            x["sd"] not in (None, "unknown", _c) for x in g):
        _groups.append([(x["sd"] == _c, x["ss"]) for x in g
                        if x["sd"] not in (None, "unknown")])


def _stat(gs):
    A = H = B = K = 0
    for g in gs:
        for is_canon, ss in g:
            if is_canon:
                A += 1; H += (ss == "helper_acc")
            else:
                B += 1; K += (ss == "helper_acc")
    return (K / B - H / A) if A and B else 0.0


SHIFT = _stat(_groups)
random.seed(0); _ge = 0
for _ in range(a.draws):
    _sh = []
    for g in _groups:
        fl = [x for x, _ in g]; random.shuffle(fl)
        _sh.append([(f_, ss) for f_, (_, ss) in zip(fl, g)])
    if _stat(_sh) >= SHIFT:
        _ge += 1
PVAL = _ge / a.draws

FORMS = [("loop_acc", "mutable loop", "#c05a25"),
         ("recursion", "plain recursion", "#e6b089"),
         ("helper_acc", "hand-written fold", "#7fb3e3"),
         ("pipeline", "pipeline", "#20507e"),
         ("unknown", "unlabelled", "#d9d7d0")]
BROWS = [("donor writes an\noff-canonical form", "off", True),
         ("donor writes the\ncanonical form", "canon", True),
         ("asked in words\n(prompting)", "prompted", False)]
for y, (name, key, solid) in enumerate(BROWS):
    cnt = rows_d[key]; n = sum(cnt.values())
    left = 0.0
    for fk, _lab, col in FORMS:
        w = cnt[fk] / n
        if w <= 0:
            continue
        bx.barh(y, w, left=left, height=0.54, color=col, edgecolor="white", linewidth=0.6,
                alpha=1.0 if solid else 0.5)
        if w > 0.075:
            bx.text(left + w / 2, y, f"{w:.0%}", fontsize=4.5, ha="center", va="center",
                    color="white" if fk in ("loop_acc", "pipeline") else "#4a4a4a")
        left += w
    bx.text(1.012, y, f"n={n}", fontsize=4.3, color="#888888", va="center", ha="left")
bx.set_xlim(0, 1); bx.set_ylim(-0.55, len(BROWS) - 0.45)
bx.set_yticks(range(len(BROWS)))
bx.set_yticklabels([r[0] for r in BROWS], fontsize=4.6, color="#444444", linespacing=1.15)
bx.tick_params(axis="y", length=0, pad=1.5)
bx.tick_params(axis="x", labelsize=5.0, length=2, width=0.4, colors="#555555")
for sp_ in ("top", "right", "left"):
    bx.spines[sp_].set_visible(False)
bx.spines["bottom"].set_linewidth(0.5); bx.spines["bottom"].set_color("#999999")
bx.set_title(r"$\bf{(b)}$  what the recipient writes, by the donor's own form",
             fontsize=5.8, color="#333333", pad=4, loc="left")
bx.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=3.6, mfc=c, mec="white", label=l)
                   for _, l, c in FORMS],
          fontsize=4.5, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.055),
          handletextpad=0.35, labelspacing=0.25, labelcolor="#444444", ncol=2, columnspacing=0.8)
fig.tight_layout(pad=0.35)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fig_donor_recipient_{MODEL}_{OP}.{ext}", facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
tot = sum(g[1] for g in grid.values()); dif = sum(g[0] for g in grid.values())
print(f"saved -> fig_donor_recipient_{MODEL}_{OP}   ({tot} valid cells over {len(grid)} pairs, "
      f"{dif}/{tot} = {dif/tot:.0%} write the requested strategy)")
print(f"  (b) paired on {NPAIR} recipient cells holding both donor kinds; "
      f"hand-written fold shift {SHIFT:+.1%}, permutation p={PVAL:.4f}")
for _k, _n in (("prompted", "prompted"), ("canon", "donor canonical"), ("off", "donor off-canon")):
    _c = rows_d[_k]; _t = sum(_c.values())
    print(f"    {_n:<16} " + "  ".join(f"{f} {_c[f]/_t:.0%}" for f, _, _ in FORMS if _c[f]))
_ph = sum(v[0] for v in pool.values()); _pn = sum(v[1] for v in pool.values())
_kh = sum(c["hit"] for c in cells)
_uk = sum(c["ss"] == "unknown" for c in cells)
print(f"  cells kept {len(cells)}/{RAW} = {len(cells)/RAW:.0%}  "
      f"(implements the requested function: {NVALID}/{RAW} = {NVALID/RAW:.0%})")
print(f"  of those, steered form unlabelled: {_uk} -- counted as differing")
print(f"  hit_strat: all {HIT[0]}/{HIT[1]} = {HIT[0]/HIT[1]:.0%}   "
      f"labelled {_kh}/{len(cells)} = {_kh/len(cells):.0%}")
print(f"  pooled column: {_ph}/{_pn} = {_ph/_pn:.0%} write the requested strategy"
      f"   (single-donor grid {dif}/{tot} = {dif/tot:.0%})")

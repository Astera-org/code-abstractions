"""Examples figure: for each model, two TRANSPORT cells where the steered program differs from the recipient's own prompted
rendering of the requested strategy. Each example is a row of three verbatim generations:
    donor program  (the donor language's own prompted program for the donor task, requested strategy)
    steered        (the recipient after receiving the single-donor state; cue asked for the opposite strategy)
    prompted       (the same recipient asked for the requested strategy in words)
Selection is automatic and stated: among changed cells whose steered structure differs from the prompted one, prefer forms
the recipient never writes when prompted, then shorter programs; at most one example per recipient language per model; the
first example per model is a functional donor into an imperative recipient where one exists.
Layout follows the paper's figure conventions (style.py palette, sans labels, 5.5 in text width): one column-header row,
bold panel letters per model, thin rules between rows, the steered column set on a light band, no boxes.
    uv run factored_programming_languages/paper_figures/fig_differs_examples.py [--max-lines 14] [--width 46]
"""
import argparse, collections, glob, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; R = HERE.parent / "results"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from style import INK, INK2, MUTED, GRID, SURFACE                        # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--max-lines", type=int, default=14); ap.add_argument("--width", type=int, default=46)
A, _ = ap.parse_known_args()
MODELS = [("gemma-4-31B-it", "Gemma-4-31B"), ("gemma-4-26b-a4b", "Gemma-4-26B-A4B"), ("qwen3.6-27b", "Qwen3.6-27B"), ("rwkv7-g1j-13.3b", "RWKV-7 13B")]
import sys as _argv_sys
if "--models" in _argv_sys.argv:              # --models key=label,... overrides the rows (smoke test)
    MODELS = [tuple(x.split("=", 1)) for x in _argv_sys.argv[_argv_sys.argv.index("--models") + 1].split(",")]
NICE = {"c": "C", "cpp": "C++", "go": "Go", "java": "Java", "javascript": "JavaScript", "ocaml": "OCaml", "ruby": "Ruby", "rust": "Rust", "haskell": "Haskell", "python": "Python"}
OPP = {"iter": "rec", "rec": "iter"}; SNAME = {"iter": "iteration", "rec": "recursion"}
SRC = (HERE / "fig_donor_recipient.py").read_text(); CUT = SRC.index("if a.report_differs:")


def load(model):
    sys.argv = ["x", "--model", model, "--op", "transport"]; ns = {"__file__": str(HERE / "fig_donor_recipient.py")}
    exec(compile(SRC[:CUT], "load", "exec"), ns)
    cells, P, BASE, code_of, task_of = ns["cells"], ns["P"], ns["BASE"], ns["code_of"], ns["task_of"]
    PROG = {}
    for f in glob.glob(str(BASE / "c_*.json")):
        r = json.loads(Path(f).read_text())
        if r["op"] == "transport" and r["arm"] == "prompted" and r.get("factor", "S") == "S":
            c_ = code_of(r)
            if c_: PROG[(task_of(r), r["lang"], r["want_strategy"])] = c_
    STEER = {}
    for f in glob.glob(str(R / f"donorgrid_{model}_*" / "c_*.json")):
        r = json.loads(Path(f).read_text())
        if r.get("donor"):
            c_ = code_of(r)
            if c_: STEER[(task_of(r), r["lang"], r["want_strategy"], r["donor"][1])] = (c_, r["donor"][0])
    rep = collections.defaultdict(set)
    for (t, l, w), st in P.items(): rep[l].add(st)
    cands = []
    for c in cells:
        cue = P.get((c["t"], c["r"], OPP[c["ws"]]))
        if cue in (None, "unknown") or c["ss"] == cue or c["ss"] == c["sp"]: continue
        key = (c["t"], c["r"], c["ws"], c["d"])
        if key not in STEER or (c["t"], c["r"], c["ws"]) not in PROG: continue
        steered, dtask = STEER[key]; prompted = PROG[(c["t"], c["r"], c["ws"])]; donor = PROG.get((dtask, c["d"], c["ws"]))
        if donor is None: continue
        n = max(len(x.strip().splitlines()) for x in (donor, steered, prompted))
        cands.append({"novel": c["ss"] not in rep[c["r"]], "lines": n, "t": c["t"], "r": c["r"], "ws": c["ws"], "d": c["d"], "dtask": dtask,
                      "ss": c["ss"], "sp": c["sp"], "sd": c["sd"], "donor": donor, "steered": steered, "prompted": prompted})
    FUNC = {"haskell", "ocaml"}
    cands.sort(key=lambda x: (not x["novel"], x["lines"]))
    picked, seen = [], set()
    first = [x for x in cands if x["d"] in FUNC and x["r"] not in FUNC]
    if first: picked.append(first[0]); seen.add(first[0]["r"])
    for x in cands:
        if x["r"] in seen or (picked and x["d"] == picked[0]["d"] and x["r"] == picked[0]["r"]): continue
        picked.append(x); seen.add(x["r"])
        if len(picked) == 2: break
    return picked


def wrap(code, width, maxl):
    out = []
    for ln in code.strip().splitlines():
        ln = ln.replace("\t", "    ").rstrip()
        while len(ln) > width:
            cut = ln.rfind(" ", 0, width); cut = cut if cut > width // 2 else width
            out.append(ln[:cut]); ln = "  " + ln[cut:].lstrip()
        out.append(ln)
    if len(out) > maxl: out = out[:maxl - 1] + ["…"]
    return out


EX = {m: load(m) for m, _ in MODELS}
for m, ex in EX.items():
    for x in ex: print(f"{m:<16} {x['r']:<10} task {x['t']:<13} {x['ws']:<4} donor {x['d']}/{x['dtask']:<12} forms donor={x['sd']} steered={x['ss']} prompted={x['sp']} novel={x['novel']} lines={x['lines']}")

# ---- layout (units: one code line = 1) ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"]})
W_IN = 5.5; U = 0.082                                                     # ICLR text width; inches per line unit
FS_CODE, FS_LAB, FS_HEAD, FS_PANEL = 4.1, 4.9, 5.4, 6.4
X0 = [0.005, 0.338, 0.671]; CW = 0.325                                    # three columns (figure fraction)
HEADS = ["Donor program", "Recipient after the donor state", "Recipient asked in words"]
rows = [(m, lab, x) for m, lab in MODELS for x in EX[m]]
prep = []
for m, lab, x in rows:
    L = {k: wrap(x[k], A.width, A.max_lines) for k in ("donor", "steered", "prompted")}
    prep.append((m, lab, x, L, max(len(v) for v in L.values()) + 1.9))       # label line + code lines + breathing
HEADER_H, PANEL_H, ROW_GAP = 1.6, 1.5, 0.35
total = HEADER_H + sum(h + ROW_GAP for *_, h in prep) + PANEL_H * len(MODELS) + 0.3
fig = plt.figure(figsize=(W_IN, total * U), dpi=300, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, total)
ax.add_patch(plt.Rectangle((X0[1] - 0.006, 0), CW + 0.012, total, facecolor=SURFACE, edgecolor="none", zorder=0))   # band under the steered column
y = total - 0.2
for x0, h in zip(X0, HEADS):
    ax.text(x0 + 0.004, y, h, fontsize=FS_HEAD, color=INK, va="top", weight="bold")
y -= HEADER_H
ax.plot([0, 1], [y + 0.25, y + 0.25], color=INK, lw=0.6, zorder=2)
for i, (m, lab, x, L, h) in enumerate(prep):
    if i == 0 or prep[i - 1][0] != m:
        letter = "abcd"[[k for k, _ in MODELS].index(m)]
        ax.text(0.005, y - 0.15, f"({letter})", fontsize=FS_PANEL, color=INK, va="top", weight="bold")
        ax.text(0.035, y - 0.15, lab, fontsize=FS_PANEL, color=INK, va="top")
        y -= PANEL_H
    labs = [f"{NICE[x['d']]} · {x['dtask']} · {SNAME[x['ws']]}", f"{NICE[x['r']]} · {x['t']} · {NICE[x['d']]} state",
            f"{NICE[x['r']]} · {x['t']} · {SNAME[x['ws']]} in words"]
    for x0, lab_, kind in zip(X0, labs, ("donor", "steered", "prompted")):
        ax.text(x0 + 0.004, y - 0.1, lab_, fontsize=FS_LAB, color=INK2 if kind == "steered" else MUTED, va="top")
        for k, ln in enumerate(L[kind]):
            ax.text(x0 + 0.004, y - 1.25 - k, ln, fontsize=FS_CODE, family="monospace", color=INK, va="top")
    y -= h
    last_of_group = (i == len(prep) - 1) or prep[i + 1][0] != m
    ax.plot([0, 1], [y - ROW_GAP / 2, y - ROW_GAP / 2], color=INK if last_of_group else GRID, lw=0.5 if last_of_group else 0.4, zorder=2)
    y -= ROW_GAP
for ext in ("pdf", "png"): fig.savefig(HERE / f"fig_differs_examples.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=300, facecolor="white")
print(f"saved fig_differs_examples.pdf/.png  ({W_IN:.1f} x {total * U:.1f} in)")

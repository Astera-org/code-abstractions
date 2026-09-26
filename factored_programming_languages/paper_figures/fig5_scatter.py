"""Condensed fig5: scatter, ICLR-sized. Coding ability (y) vs compiler
index (x), one point per model; only anchor models are labeled. Same data
as fig5_interface_spectrum: mask circles, mean-ablation diamonds for the
hybrids the mask cannot sever (--ablmode swaps every model onto one
ablation instrument, suffixing the filename).

    uv run factored_programming_languages/paper_figures/fig5_scatter.py [--ablmode abl_0]
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})
ap = argparse.ArgumentParser()
ap.add_argument("--ablmode", default=None, choices=["abl_kv", "abl_rand", "abl_0", "abl_mask", "abl_maskboth"])
ap.add_argument("--full", action="store_true",
                help="severing measured on the WHOLE benchmark (results/ablatefull_<model>) against the same "
                     "tasks as the coding-ability axis, rather than the 4-task flow bank")
ap.add_argument("--age-color", action="store_true",
                help="colour each point by the model's release date (HF repo creation, results/"
                     "model_release_dates.json) instead of by compiler/re-reader band; adds a colourbar and "
                     "suffixes the filename _age. Marker SHAPE still carries the attention class, so the legend "
                     "keeps saying what kind of model each point is.")
ap.add_argument("--min-cells", type=int, default=310,
                help="cells a model needs in results/ablatefull_<model> and results/bench_<model> to be drawn "
                     "(the paper: the full 310-cell benchmark; lower it only for a smoke test)")
ap.add_argument("--by-attn", action="store_true",
                help="marker by attention architecture (full dense / interleaved local-global / "
                     "linear+full hybrid), read from the cached HF config; suffixes the filename _attn")
_a = ap.parse_args()
ABLMODE, BY_ATTN, AGE, FULL = _a.ablmode, _a.by_attn, _a.age_color, _a.full
MIN_CELLS = _a.min_cells
import datetime as _dt                                                 # noqa: E402
import matplotlib.dates as _mdates                                     # noqa: E402
AGES = {}
if AGE:
    _p = Path(__file__).resolve().parent.parent / "data" / "model_release_dates.json"   # HF repo creation dates, curated by hand
    AGES = {k: _dt.date.fromisoformat(v) for k, v in json.loads(_p.read_text()).items() if v}

R = Path(__file__).resolve().parent.parent / "results"
COMPILER, REREADER, MID = "#20507e", "#a8531f", "#8a8a8a"

import sys                                                             # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recipes import RECIPES                                            # noqa: E402
import os                                                              # noqa: E402
# the HF hub cache (config.json of each model is read for its attention architecture)
HF_HUB = os.environ.get("HF_HUB_CACHE") or str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")


def hybrid(m):
    try:
        hf = RECIPES[m]["hf_name"].replace("/", "--")
        f = glob.glob(f"{HF_HUB}/models--{hf}"
                      f"/snapshots/*/config.json")
        if not f:
            return False
        tc = json.loads(Path(f[0]).read_text())
        tc = tc.get("text_config", tc)
        return "linear_attention" in (tc.get("layer_types") or [])
    except Exception:
        return False


def _text_config(m):
    key = m if m in RECIPES else next((k for k in RECIPES if k.lower() == m.lower()), None)
    if key is None:
        return None
    hf = RECIPES[key]["hf_name"].replace("/", "--")
    f = glob.glob(f"{HF_HUB}/models--{hf}/snapshots/*/config.json")
    if not f:
        return None
    tc = json.loads(Path(f[0]).read_text())
    return tc.get("text_config", tc)


def attn_class(m):
    """'hybrid' = linear-attention layers interleaved with full attention (qwen3.5/3.6/3.8);
    'local_global' = sliding-window layers interleaved with global ones (gemma-2/3/4, EXAONE-4);
    'full' = every layer attends to the whole context"""
    tc = _text_config(m) or {}
    lt = tc.get("layer_types") or []
    if "linear_attention" in lt:
        return "hybrid"
    if "sliding_attention" in lt or tc.get("sliding_window_pattern"):
        return "local_global"
    if tc.get("model_type", "").startswith(("gemma2", "gemma3", "gemma4")):
        return "local_global"                 # older gemma configs carry only sliding_window
    if tc.get("sliding_window") and tc.get("use_sliding_window", True):
        return "local_global"
    return "full"


ATTN_MARKER = {"full": "o", "local_global": "s", "hybrid": "D"}
ATTN_LABEL = {"full": "full attention", "local_global": "local / global attention", "hybrid": "linear + full (hybrid)"}


def flowcut(m):
    d = R / f"flow_cut_{m}"
    if not d.is_dir():
        return None
    by = {}
    for f in d.glob("flow_*.json"):
        r = json.loads(f.read_text())
        by.setdefault(r["cond"], {})[r["cell"]] = bool(r["intact"])
    if "none" not in by or "gen" not in by:
        return None
    base = {k for k, v in by["none"].items() if v}
    g = [by["gen"][k] for k in base if k in by["gen"]]
    return sum(g) / len(g) if g else None


def ablate_ci(m, mode):
    if FULL:
        d = R / f"ablatefull_{m}"
        cells = [json.loads(f.read_text()) for f in d.glob("at_*.json")] if d.is_dir() else []
        if len(cells) < MIN_CELLS:
            return None
        sev = {(r["task"], r["lang"]): (r.get("impl") == r["task"]) for r in cells}
        ok = {}
        for f in glob.glob(str(R / f"bench_{m}" / "b_*.json")):
            r = json.loads(Path(f).read_text())
            ok[(r["task"], r["lang"])] = bool(r.get("pass"))
        # ABSOLUTE pass rate on the shared task set, not a retention rate (authors, 2026-09-17: an all-recurrent
        # model's score under severing "should be the same as its coding performance"). Both axes are then the
        # same quantity measured with and without the mask, so y = x means severing costs nothing.
        keep = [k for k in ok if k in sev]
        return sum(sev[k] for k in keep) / len(keep) if keep else None
    d = R / f"ablateflow_{m}"
    if not d.is_dir():
        return None
    by = {}
    for f in d.glob("at_*.json"):
        r = json.loads(f.read_text())
        by.setdefault(r["mode"], {})[(r["task"], r["lang"])] = \
            (r.get("impl") == r["task"])
    if "none" not in by or mode not in by:
        return None
    if ABLMODE:
        # absolute correctness under the ablation (unconditioned): the
        # y-axis for the ablation-instrument variants
        cells = [k for k in by["none"] if k in by[mode]]
        return sum(by[mode][k] for k in cells) / len(cells) if cells else None
    base = {k for k, v in by["none"].items() if v}
    g = [by[mode][k] for k in base if k in by[mode]]
    return sum(g) / len(g) if g else None


def bench(m):
    fs = glob.glob(str(R / f"bench_{m}" / "b_*.json"))
    if len(fs) < min(300, MIN_CELLS):
        return None
    ok = sum(1 for f in fs
             if json.loads(Path(f).read_text()).get("pass") in (True, 1))
    return ok / len(fs)


# Models with NO severing condition (authors, 2026-09-14). An all-recurrent model keeps no per-token cache, so
# there are no rows an ablation could remove: the state is the only thing carried past prefill, and "correctness
# under ablation" equals plain correctness by construction rather than by measurement. Such a model is drawn with
# its own marker at y = its own coding ability -- the value severing is defined to leave untouched -- and is NOT
# given fabricated ablation cells.
STRUCTURAL = {"rwkv7-g1j-13.3b": "no per-token cache: severing undefined"}

prof = set(json.loads((R / "cfs_profile.json").read_text())) if (R / "cfs_profile.json").exists() else set()
extra = {d.name[6:] for d in R.glob("bench_*") if d.is_dir()}
pts = []
for m in sorted(prof | extra):
    bn = bench(m)
    if bn is None:
        continue
    hyb = hybrid(m)
    if m in STRUCTURAL:
        ci, dia = (bn if (ABLMODE and FULL) else 1.0), False
    elif ABLMODE:
        ci, dia = ablate_ci(m, ABLMODE), True
    elif hyb:
        ci, dia = ablate_ci(m, "abl_kv"), True
    else:
        ci, dia = flowcut(m), False
    if ci is None:
        continue
    pts.append((m, ci, bn, dia))

# wide single-column layout: ability (well spread) on the LONG axis,
# compiler index (bimodal, tolerates compression) on the short one
LABELS = {
    "gemma-4-31B-it": ("gemma-4-31B", (-0.005, -0.055), "right"),
    "gemma-4-26b-a4b": ("gemma-4-26B-A4B", (0.006, -0.02), "left"),
    "qwen3.6-27b": ("Qwen3.6-27B", (0.000, -0.13), "center"),
    "qwen3.8-27b": ("Qwen3.8-27B", (-0.006, 0.005), "right"),
    "qwen3-coder-30b": ("Qwen3-Coder", (-0.004, -0.11), "right"),
    "qwen3-30b-2507": ("Qwen3-30B-2507", (0.006, 0.0), "left"),
    "mistral-small-4-119b": ("Mistral-Small-4", (-0.006, 0.0), "right"),
    "gemma-3-27b-it": ("gemma-3-27b", (-0.006, -0.045), "right"),
    "devstral-2-123b": ("Devstral-2", (-0.006, 0.0), "right"),
    "pythia-6.9b-tulu": ("pythia (control)", (0.007, 0.0), "left"),
    "rwkv7-g1j-13.3b": ("RWKV7-13.3B (no attention)", (-0.007, 0.0), "right"),
}

# label anchors for the ablation variants: the compiler-like cluster is
# too tight to label in place, so names sit in the empty band below it
# with leader lines -- a left stack for the four leftmost points and a
# center-right stack for the three rightmost. Slot order within each
# stack is chosen so no two leaders cross.
# slots retuned 2026-09-17 for the absolute y-axis: the labelled cluster now sits at y 0.36-0.53 and the whole
# upper half of the panel is empty, so both stacks live above the data. Within a stack the slot order matches
# the target order in y, which is what keeps leaders from crossing each other.
ANCHOR = {   # (label, text anchor, alignment)
    "qwen3.6-35b-a3b": ("Qwen3.6-35B-A3B", (0.470, 0.97), "right"),
    "qwen3.5-35b-a3b": ("Qwen3.5-35B-A3B", (0.470, 0.85), "right"),
    "qwen3.8-27b":     ("Qwen3.8-27B",     (0.470, 0.73), "right"),
    "gemma-4-12b-it":  ("gemma-4-12B",     (0.470, 0.61), "right"),
    "exaone-4-32b":    ("EXAONE-4.0-32B",  (0.470, 0.45), "right"),
    "gemma-4-31B-it":  ("gemma-4-31B",     (0.950, 0.88), "right"),
    "qwen3.6-27b":     ("Qwen3.6-27B",     (0.950, 0.76), "right"),
    "gemma-4-26b-a4b": ("gemma-4-26B-A4B", (0.950, 0.64), "right"),
}
INPLACE = {  # isolated points: labelled where they sit, no leader
    "pythia-6.9b-tulu": ("pythia (control)", (0.020, 0.0), "left"),
}

W, H = 5.5, 1.55
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#999999")
ax.tick_params(labelsize=6.5, colors="#444444", length=2.5)

seen_attn = set()
seen_structural = set()
if AGE:
    import matplotlib.cm as _cm
    _ds = [_mdates.date2num(AGES[m]) for m, _, _, _ in pts if m in AGES]
    CMAP = _cm.get_cmap("viridis")
    NORM = plt.Normalize(min(_ds), max(_ds)) if _ds else plt.Normalize(0, 1)
for m, ci, bn, dia in pts:
    col = COMPILER if ci >= 0.7 else (REREADER if ci <= 0.3 else MID)
    if AGE and m in AGES:
        col = CMAP(NORM(_mdates.date2num(AGES[m])))
    if m in STRUCTURAL:      # y is a property of the architecture, not a measurement: open star, no fill
        ax.scatter([bn], [ci], s=52, marker="*", color="white", zorder=5.4)   # halo: leaders pass behind
        ax.scatter([bn], [ci], s=46, marker="*", facecolors="none",
                   edgecolors=col, linewidths=0.9, zorder=6)
        seen_structural.add(m)
    elif BY_ATTN:
        ac = attn_class(m); seen_attn.add(ac)
        mk = ATTN_MARKER[ac]; sz = {"o": 27, "s": 24, "D": 22}[mk]
        ax.scatter([bn], [ci], s=sz, marker=mk,
                   color=col, edgecolors="white", linewidths=0.6, zorder=5)
    else:
        mk, sz = ("D" if dia else "o"), (24 if dia else 27)
        ax.scatter([bn], [ci], s=sz, marker=mk,
                   color=col, edgecolors="white", linewidths=0.6, zorder=5)
    if ABLMODE and m in ANCHOR:                 # spread out, leader lines
        lab, (ax_, ay_), ha = ANCHOR[m]
        ax.annotate(lab, xy=(bn, ci), xytext=(ax_, ay_), fontsize=5.6,
                    color="#333333", ha=ha, va="center", zorder=3,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#bbbbbb",
                                    shrinkA=1.5, shrinkB=2.5,
                                    connectionstyle="arc3,rad=0.0"))
    elif ABLMODE:
        if m not in INPLACE:
            continue
        lab, (dx, dy), ha = INPLACE[m]
        ax.annotate(lab, (bn + dx, ci + dy), fontsize=5.6, color="#333333",
                    ha=ha, va="center", zorder=6)
    elif m in LABELS:
        lab, (dx, dy), ha = LABELS[m]
        ax.annotate(lab, (bn + dx, ci + dy), fontsize=5.8, color="#333333",
                    ha=ha, va="center", zorder=6)

ax.set_xlim(0, 0.95)
ax.set_ylim(-0.04, 1.04)
ax.set_xlabel("coding ability  (bench pass rate)", fontsize=7.0,
              color="#222222")
ax.set_ylabel("correctness under ablation" if ABLMODE else "compiler index",
              fontsize=7.0, color="#222222")
if BY_ATTN:
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker=ATTN_MARKER[k], ls="", ms=4.2, mfc="#777777", mec="white", mew=0.5, label=ATTN_LABEL[k])
          for k in ("full", "local_global", "hybrid") if k in seen_attn]
    if seen_structural:
        hs.append(Line2D([], [], marker="*", ls="", ms=6.5, mfc="none", mec="#777777", mew=0.9,
                         label=("no attention (masking inert)" if ABLMODE == "abl_mask" else "no attention (severing inert)")))
    ax.legend(handles=hs, fontsize=5.4, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99),
              handletextpad=0.4, borderaxespad=0.2, labelcolor="#333333")
if AGE:                     # a scale for the release-date colouring
    _sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM); _sm.set_array([])
    _cb = fig.colorbar(_sm, ax=ax, fraction=0.022, pad=0.010)
    _cb.ax.tick_params(labelsize=5.0, length=2, width=0.4)
    _cb.outline.set_linewidth(0.4)
    _cb.ax.yaxis.set_major_locator(_mdates.YearLocator())
    _cb.ax.yaxis.set_major_formatter(_mdates.DateFormatter("%Y"))
    _cb.set_label("model release", fontsize=5.8, labelpad=2)
fig.tight_layout(pad=0.4)
suf = (f"_{ABLMODE}" if ABLMODE else "") + ("_attn" if BY_ATTN else "") + ("_age" if AGE else "") + ("_full" if FULL else "")
fig.savefig(Path(__file__).parent / f"fig5_scatter{suf}.png",
            facecolor="white", bbox_inches="tight", pad_inches=0.02)
fig.savefig(Path(__file__).parent / f"fig5_scatter{suf}.pdf",
            facecolor="white", bbox_inches="tight", pad_inches=0.02)
print(f"saved -> fig5_scatter{suf} ({len(pts)} models)")

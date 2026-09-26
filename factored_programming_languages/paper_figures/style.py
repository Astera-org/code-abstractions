"""Shared visual conventions for the main-paper figures (spec.md).

The spec mandates cross-figure conventions; putting them here rather than in each
figure is what makes them actually hold:

  * the recurring mechanistic cast is coloured identically in every figure
  * 1.0 is a visible reference wherever a normalised score is plotted, and
    values above 1 are NEVER clipped
  * a gated model (no addressable interface) is drawn on a separate shelf with
    a hollow marker -- never as a score of zero
  * channels have fixed encodings, and ONE-SHOT (suffix/span) is visually
    distinct from CONTINUOUS (online) delivery, because they do not support the
    same mechanistic claim

    from style import CAST, C, gated_marker, ...
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = Path(__file__).resolve().parent

# ---- ink -------------------------------------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE, BASELINE = "#e1e0d9", "#fcfcfb", "#c3c2b7"

# ---- the recurring cast (spec: "make these easy to track across figures") ---
CAST = {
    "qwen3-32b":             ("Qwen3-32B",        "#2a78d6"),
    "qwen3.6-27b":           ("Qwen3.6-27B",      "#7b3fb5"),
    "gemma-4-31B-it":        ("gemma-4-31B",      "#1baf7a"),
    "gemma-4-26b-a4b":       ("gemma-4-26B-A4B",  "#0e7c86"),
    "mistral-small-3.2-24b": ("Mistral-3.2",      "#eb6834"),
    "pythia-6.9b-tulu":      ("pythia (control)", "#565551"),
}
OTHER = "#b9b7ae"          # non-cast models: present but recessive


def label(m):
    return CAST[m][0] if m in CAST else m


def colour(m):
    return CAST[m][1] if m in CAST else OTHER


def is_cast(m):
    return m in CAST


# ---- channels --------------------------------------------------------------
# one-shot vs continuous is a MECHANISTIC distinction, not a cosmetic one
CHANNEL = {
    "suffix": ("#2a78d6", "o", "one-shot"),
    "span":   ("#4a3aa7", "o", "one-shot"),
    "online": ("#eb6834", "^", "continuous"),
    "override": ("#898781", "s", "—"),
}
ONESHOT = ("suffix", "span")


def channel_of(tag):
    """'span@1.0' -> 'span'"""
    return (tag or "").split("@")[0]


def is_oneshot(tag):
    return channel_of(tag) in ONESHOT


# ---- data ------------------------------------------------------------------
def cfs_profile():
    return json.loads((R / "cfs_profile.json").read_text())


def strategy_profile():
    return json.loads((R / "strategy_profile.json").read_text())


def best_site(sites):
    """the site with a reportable CFS at the highest addressability"""
    best = None
    for s, e in sites.items():
        if e.get("CFS") is not None and (best is None or e["A_s"] > best[1]["A_s"]):
            best = (s, e)
    return best


def scored_and_gated(key="CFS"):
    """-> (list of (model, entry) with a score, list of (model, best A_s) gated)

    Gated = probed, but no site clears the addressability floor. The spec is
    explicit that these must never be drawn as zero."""
    prof = cfs_profile()
    scored, gated = [], []
    for m, sites in prof.items():
        b = best_site(sites)
        if b and b[1].get(key) is not None:
            scored.append((m, b[1]))
        elif sites:
            gated.append((m, max(e.get("A_s", float("-inf")) for e in sites.values())))
    return scored, gated


# ---- drawing helpers -------------------------------------------------------
def figure(w, h):
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    return fig


def panel(ax, title=None, grid="both"):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    if grid in ("both", "x"):
        ax.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
    if grid in ("both", "y"):
        ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=3)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=7)
    return ax


def unity(ax, v=1.0, horizontal=False, text=None):
    """the visible 1.0 reference the spec asks for"""
    (ax.axhline if horizontal else ax.axvline)(
        v, color=BASELINE, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if text:
        if horizontal:
            ax.annotate(text, (0.01, v), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points",
                        fontsize=6.5, color=MUTED)
        else:
            ax.annotate(text, (v, 0.99), xycoords=("data", "axes fraction"),
                        xytext=(3, -3), textcoords="offset points",
                        fontsize=6.5, color=MUTED, va="top")


def gated_legend_handle(lab="not addressable (A_s < floor)"):
    return Line2D([], [], marker="o", ls="", ms=6, mfc="none",
                  markeredgecolor=INK2, label=lab)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor=SURFACE,
                bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", facecolor=SURFACE, bbox_inches="tight")
    print(f"saved -> paper_figures/{name}.png / .pdf")

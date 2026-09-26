"""Orthogonal energy decomposition of the fxl20 direction-crossed corpus.

Four factors of the fxl20 corpus (Fig. 8):

    algorithm (20) x paradigm {loop, rec} x direction {forward, reverse} x language (10)

    sum_cells ||h - mu||^2 = 4 mains + 6 pairwise + 4 three-way + 1 four-way

All terms are reported as fractions of total energy (effect sizes, not tests). The
FUNCTIONAL SHARE -- what Figure 2's peak dot and rise-region band are computed from --
is the sum of the algorithm + paradigm + direction main effects.

Noise correction as in the 3-factor script: null cells from a 2-way rename-seed split;
the difference of two half-means carries 4x the noise variance of a cell mean, so the
null decomposition is divided by 4. With the fxl20 corpus's 2 seeds each half is a
single program, which makes the floor itself noisier but leaves the /4 exact.

Usage (from the package root):
    uv run factored_programming_languages/analysis_factorial.py \
        --model-dir factored_programming_languages/results/google--gemma-4-31B-it--fxl20
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FACTORS = ["algorithm", "paradigm", "direction", "language"]
AXES = {name: ax for ax, name in enumerate(FACTORS)}


def decompose(X):
    """Orthogonal energy partition of a balanced 4-factor design.

    X (n_a, n_p, n_d, n_l, dim) cell means.
    Returns dicts of energy FRACTIONS: mains, pairs, threes, plus the four-way rest.
    """
    Xc = X - X.mean(axis=(0, 1, 2, 3), keepdims=True)
    total = float((Xc ** 2).sum())

    def frac(E):
        return float((E ** 2).sum() * (Xc.size / E.size)) / total

    eff = {}
    for name, ax in AXES.items():
        other = tuple(a for a in range(4) if a != ax)
        eff[name] = Xc.mean(axis=other, keepdims=True)
    mains = {n: frac(eff[n]) for n in FACTORS}

    pair_eff, pairs = {}, {}
    for a, b in itertools.combinations(FACTORS, 2):
        other = tuple(ax for ax in range(4) if ax not in (AXES[a], AXES[b]))
        E = Xc.mean(axis=other, keepdims=True) - eff[a] - eff[b]
        pair_eff[f"{a}x{b}"] = E
        pairs[f"{a}x{b}"] = frac(E)

    threes = {}
    for trio in itertools.combinations(FACTORS, 3):
        other = tuple(ax for ax in range(4) if ax != AXES[trio[0]]
                      and ax != AXES[trio[1]] and ax != AXES[trio[2]])
        E = Xc.mean(axis=other, keepdims=True)
        for n in trio:
            E = E - eff[n]
        for a, b in itertools.combinations(trio, 2):
            E = E - pair_eff[f"{a}x{b}"]
        threes["x".join(trio)] = frac(E)

    four = 1.0 - sum(mains.values()) - sum(pairs.values()) - sum(threes.values())
    return mains, pairs, threes, four, total


def load(model_dir):
    """Group means reshaped to (n_layers, n_a, n_p, n_d, n_l, dim)."""
    z = np.load(Path(model_dir) / "group_means.npz", allow_pickle=True)
    g_task = np.array([str(t) for t in z["group_tasks"]])
    g_lang = np.array([str(l) for l in z["group_languages"]])
    algos = list(dict.fromkeys(t.split("__")[0] for t in g_task))
    strats = list(dict.fromkeys(t.split("__")[1] for t in g_task))
    paradigms = list(dict.fromkeys(s.split("_")[0] for s in strats))
    directions = list(dict.fromkeys(s.split("_")[1] for s in strats))
    langs = list(dict.fromkeys(g_lang))
    ai = {a: i for i, a in enumerate(algos)}
    pi = {p: i for i, p in enumerate(paradigms)}
    di = {d: i for i, d in enumerate(directions)}
    li = {l: i for i, l in enumerate(langs)}

    M = z["mean_pool"]                       # (n_store, n_groups, dim)
    n_store, n_groups, dim = M.shape
    cells = np.full((n_store, len(algos), len(paradigms), len(directions),
                     len(langs), dim), np.nan, dtype=np.float64)
    for gi in range(n_groups):
        a, s = g_task[gi].split("__")
        p, d = s.split("_")
        cells[:, ai[a], pi[p], di[d], li[g_lang[gi]]] = M[:, gi]
    assert not np.isnan(cells).any(), "unfilled cell: corpus is not balanced"
    return cells, algos, paradigms, directions, langs, z


def noise_cells(z, algos, paradigms, directions, langs):
    """Null cells from a 2-way seed split (difference of half-means, /4 downstream)."""
    V = z["variant_mean_pool"]                # (n_store, n_programs, dim)
    p_task = np.array([str(t) for t in z["program_tasks"]])
    p_lang = np.array([str(l) for l in z["program_languages"]])
    p_var = np.asarray(z["program_variants"]).astype(int)
    seeds = sorted(set(p_var.tolist()))
    assert len(seeds) >= 2, "need >= 2 rename seeds for a noise floor"
    h1, h2 = seeds[::2], seeds[1::2]
    ai = {a: i for i, a in enumerate(algos)}
    pi = {p: i for i, p in enumerate(paradigms)}
    di = {d: i for i, d in enumerate(directions)}
    li = {l: i for i, l in enumerate(langs)}
    n_store, _, dim = V.shape
    out = np.zeros((n_store, len(algos), len(paradigms), len(directions),
                    len(langs), dim))
    for a_s in set(p_task.tolist()):
        a, s = a_s.split("__")
        p, d = s.split("_")
        for lang in langs:
            m = (p_task == a_s) & (p_lang == lang)
            r1 = np.flatnonzero(m & np.isin(p_var, h1))
            r2 = np.flatnonzero(m & np.isin(p_var, h2))
            out[:, ai[a], pi[p], di[d], li[lang]] = \
                (V[:, r1].astype(np.float64).mean(1)
                 - V[:, r2].astype(np.float64).mean(1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="4-factor ANOVA of the fxl20 corpus")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells, algos, paradigms, directions, langs, z = load(args.model_dir)
    n_store = cells.shape[0]
    print(f"{len(algos)} algorithms x {len(paradigms)} paradigms x "
          f"{len(directions)} directions x {len(langs)} languages, "
          f"{n_store} rows, dim={cells.shape[-1]}")

    null = noise_cells(z, algos, paradigms, directions, langs)

    rows = []
    print(f"\n{'row':>4} {'A':>6} {'P':>6} {'D':>6} {'L':>6} | {'func':>6} "
          f"{'pairs':>6} {'3way':>6} {'4way':>6} | {'noise':>6}")
    for L in range(n_store):
        mains, pairs, threes, four, total = decompose(cells[L])
        _, _, _, _, ntot = decompose(null[L])
        noise_frac = (ntot / 4.0) / total
        func = mains["algorithm"] + mains["paradigm"] + mains["direction"]
        rows.append({"layer": L, "total": total, "noise_fraction": noise_frac,
                     "mains": mains, "pairs": pairs, "threes": threes,
                     "four_way": four, "functional_share": func})
        print(f"{L:>4} {mains['algorithm']:>6.3f} {mains['paradigm']:>6.3f} "
              f"{mains['direction']:>6.3f} {mains['language']:>6.3f} | "
              f"{func:>6.3f} {sum(pairs.values()):>6.3f} "
              f"{sum(threes.values()):>6.3f} {four:>6.3f} | {noise_frac:>6.3f}")

    peak = max((r for r in rows if r["layer"] > 0),
               key=lambda r: r["functional_share"])
    print(f"\nfunctional-share peak (algorithm+paradigm+direction, embedding row "
          f"excluded): {peak['functional_share']:.3f} at row {peak['layer']} "
          f"of {n_store - 1}")

    out = args.out or str(Path(args.model_dir) / "analysis" / "factorial_fxl20.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {"factors": {"algorithm": algos, "paradigm": paradigms,
                     "direction": directions, "language": langs},
         "n_rows": n_store, "per_layer": rows,
         "functional_peak_row": peak["layer"],
         "functional_peak_share": peak["functional_share"]}, indent=1))
    print(f"[saved] {out}")

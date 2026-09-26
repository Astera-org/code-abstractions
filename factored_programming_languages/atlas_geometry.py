"""Geometry of the function bank: is a modifier the same displacement everywhere?

Panel (a) of fig_geometry shows CAUSALLY that a displacement read off one pair
transplants onto other bases. This is the geometric counterpart, on the same
kind of object: the leave-one-out absolute factor

    F(f) = mean over languages of  state(f, lang) - state(nofn, lang)

harvested over the aligned suffix, exactly as run_step_algebra defines it, but
from the 42-function atlas so every function in the bank is available.

Two modifier families are present over the same three bases:

    square   digit_sum -> sq_digit_sum,  digit_prod -> sq_prod,  max_digit -> sq_max
    cube     digit_sum -> cube_sum,      digit_prod -> cube_prod, max_digit -> cube_max

If a modifier is a reusable displacement, its three instances should point the
same way regardless of which base they start from, and should NOT align with
the base-to-base differences that separate the functions themselves.

The full state is 60 layers x {keys, values} x 16 heads x 30 rows x 256 dims.
Averaging over the 30 suffix rows makes the arithmetic cheap; __main__ checks
that reduction against the all-rows cosines before anything is reported.

    uv run factored_programming_languages/atlas_geometry.py
"""
import os
# a 104-core box with several of these in parallel: torch's default thread
# count per process drove the load average past 400 and the work stopped
# making progress. Cap before torch is imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import torch
torch.set_num_threads(8)
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
import sys as _argv_sys
MODEL = _argv_sys.argv[_argv_sys.argv.index("--model") + 1] if "--model" in _argv_sys.argv else "gemma-4-31B-it"   # --model KEY overrides
BASES = ["digit_sum", "digit_prod", "max_digit"]
FAMILY = {"square": {"digit_sum": "sq_digit_sum", "digit_prod": "sq_prod",
                     "max_digit": "sq_max"},
          "cube": {"digit_sum": "cube_sum", "digit_prod": "cube_prod",
                   "max_digit": "cube_max"}}


def langs():
    return sorted(p.name[7:-3] for p in
                  (R / f"atlas_{MODEL}").glob("states_*.pt"))


def flat(state, rowmean=True):
    """one vector per function: layers x {keys, values} x heads x dims"""
    out = []
    for i in sorted(state):
        for n in sorted(state[i]):
            v = state[i][n].float()
            out.append(v.mean(1).reshape(-1) if rowmean else v.reshape(-1))
    return torch.cat(out)


def factors(rowmean=True, only=None):
    """-> {function: F(f)} averaged over languages, F as defined above"""
    acc, n = {}, 0
    for lang in langs():
        d = torch.load(R / f"atlas_{MODEL}" / f"states_{lang}.pt",
                       map_location="cpu", weights_only=False)["states"]
        base = flat(d["nofn"], rowmean)
        for f, st in d.items():
            if f == "nofn" or (only is not None and f not in only):
                continue
            v = flat(st, rowmean) - base
            acc[f] = v if f not in acc else acc[f] + v
        n += 1
        del d
    return {f: v / n for f, v in acc.items()}


def cos(x, y):
    return float((x @ y) / (x.norm() * y.norm() + 1e-9))


def alignments(F):
    """within-family, cross-family, and modifier-vs-base-difference cosines"""
    step = {fam: {b: F[tgt[b]] - F[b] for b in BASES if b in F and tgt[b] in F}
            for fam, tgt in FAMILY.items()}
    out = {}
    for fam, d in step.items():
        ks = sorted(d)
        out[f"{fam} within"] = [cos(d[a], d[b]) for i, a in enumerate(ks)
                                for b in ks[i + 1:]]
    if len(step) == 2:
        s, c = step["square"], step["cube"]
        out["square vs cube"] = [cos(s[b], c[b]) for b in s if b in c]
    diffs = [F[a] - F[b] for i, a in enumerate(BASES) for b in BASES[i + 1:]
             if a in F and b in F]
    out["modifier vs base difference"] = [
        cos(v, dv) for d in step.values() for v in d.values() for dv in diffs]
    return out


CACHE = R / f"atlas_{MODEL}" / "factors_rowmean.pt"


def cached_factors():
    """row-mean F(f) for all 42 functions, computed once and cached: the
    source files are ~13 GB in total and the figure should not re-read them"""
    if CACHE.exists():
        return torch.load(CACHE, map_location="cpu", weights_only=False)
    F = factors(rowmean=True)
    torch.save(F, CACHE)
    return F


if __name__ == "__main__":
    need = set(BASES) | {v for d in FAMILY.values() for v in d.values()}

    # validate the reduction on ONE language: if row-means preserve the
    # alignment structure there, they preserve it everywhere
    one = langs()[0]
    d = torch.load(R / f"atlas_{MODEL}" / f"states_{one}.pt",
                   map_location="cpu", weights_only=False)["states"]
    for rm in (False, True):
        base = flat(d["nofn"], rm)
        F1 = {f: flat(d[f], rm) - base for f in need}
        a = alignments(F1)
        print(f"  [{one}] {'row-mean' if rm else 'all rows'}: " + "   ".join(
            f"{k} {sum(v) / len(v):+.3f}" for k, v in a.items()), flush=True)
    del d

    F = cached_factors()
    print(f"\n[factors] {len(F)} functions, dim {F[BASES[0]].numel()}"
          f"  -> {CACHE}")
    AL = alignments(F)
    for k, v in AL.items():
        print(f"  {k:>30}: mean {sum(v) / len(v):+.3f}   "
              + " ".join(f"{x:+.2f}" for x in v))
    # the alignment groups fig_geometry.py panel (d) draws
    import json
    (R / f"atlas_{MODEL}" / "alignments.json").write_text(json.dumps(
        {"model": MODEL, "dim": int(F[BASES[0]].numel()), "groups": {k: [float(x) for x in v] for k, v in AL.items()}}, indent=1))
    print(f"saved -> {R / f'atlas_{MODEL}' / 'alignments.json'}")

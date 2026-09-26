"""Certify each model's steering band from its own fxl20 variance curve.

The rule (adopted 2026-08-12, validated against the retired 5-band steering sweeps):

  function band   on rows 1..L of the functional share (algorithm + paradigm, direction
                  pooled out), take thr = min + 0.5*(max - min), start at the argmax and
                  walk outward in both directions while share >= thr.
  language band   the same half-rise rule applied to the LATE RECOVERY of the language
                  share, searched only over rows argmin .. floor(0.9*L). The truncation
                  is deliberate: the last rows are contaminated by the next-token
                  prediction, which is trivially language-specific.

Bands are 1-indexed layer-output ROWS: row r is the output of layer r-1, row 0 the
embedding. No steering runs are involved -- a band comes entirely from the model's own
extraction, which is what makes it a prediction rather than a fit.

Reimplemented here rather than copied from the archive's fig2_it.py, which couples band
certification to figure rendering through a MODELS list and a BASE_TITLE dict and raises
KeyError before its own --bands-only exit if either registration is missed.

    uv run factored_programming_languages/certify_bands.py --model pythia-6.9b-tulu
    uv run factored_programming_languages/certify_bands.py --all
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import models as cast                                                    # noqa: E402


def half_rise(share, lo=1, hi=None):
    """(start, end) of the contiguous run around the argmax that stays above
    min + 0.5*(max - min), searched within [lo, hi]."""
    hi = len(share) - 1 if hi is None else hi
    seg = {r: share[r] for r in range(lo, hi + 1)}
    top = max(seg, key=seg.get)
    thr = min(seg.values()) + 0.5 * (max(seg.values()) - min(seg.values()))
    a = b = top
    while a - 1 >= lo and seg[a - 1] >= thr:
        a -= 1
    while b + 1 <= hi and seg[b + 1] >= thr:
        b += 1
    return a, b


def bands_for(anova_path):
    d = json.load(open(anova_path))
    per = {p["layer"]: p for p in d["per_layer"]}
    L = max(per)
    fshare = {r: per[r]["functional_share"] for r in per}
    lshare = {r: per[r]["mains"]["language"] for r in per}
    fn = half_rise(fshare, 1, L)
    # language: search the late recovery only
    lo = min(range(1, L + 1), key=lambda r: lshare[r])
    lang = half_rise(lshare, max(lo, 1), int(0.9 * L))
    return {"function": list(fn), "language": list(lang),
            "functional_peak_row": d.get("functional_peak_row"), "n_rows": L + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--from-dir", default=None,
                    help="certify from an explicit factorial_fxl20.json instead "
                         "(used to validate the rule against known bands)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.from_dir:
        print(json.dumps(bands_for(a.from_dir), indent=1))
        return

    out = Path(a.out or cast.BAND_FILE)
    have = json.load(open(out)) if out.exists() else {}
    lang_out = out.with_name("langbands_fxl20.json")
    have_l = json.load(open(lang_out)) if lang_out.exists() else {}

    todo = cast.CAST if a.all else [cast.get(a.model)]
    for m in todo:
        p = Path(a.results) / f"{m.dirname}--fxl20" / "analysis" / "factorial_fxl20.json"
        if not p.exists():
            print(f"  {m.key:<22} SKIP -- no {p}")
            continue
        b = bands_for(p)
        old = have.get(m.band_title)
        flag = ""
        if old and old != b["function"]:
            flag = f"   [WARN] band changed {old} -> {b['function']}"
        have[m.band_title] = b["function"]
        have_l[m.band_title] = b["language"]
        print(f"  {m.key:<22} function {str(b['function']):<10} "
              f"language {str(b['language']):<10} peak row {b['functional_peak_row']}"
              f" of {b['n_rows'] - 1}{flag}")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(have, open(out, "w"), indent=1, sort_keys=True)
    json.dump(have_l, open(lang_out, "w"), indent=1, sort_keys=True)
    print(f"\n[saved] {out}\n[saved] {lang_out}")


if __name__ == "__main__":
    main()

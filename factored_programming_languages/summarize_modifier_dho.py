"""Double-held-out modifier transfer: select (band, alpha) on the dev phase, tabulate the test phase.
    MODEL=gemma-4-31B-it uv run factored_programming_languages/summarize_modifier_dho.py --select   # writes selection.json
    MODEL=gemma-4-31B-it uv run factored_programming_languages/summarize_modifier_dho.py            # test tables
"""
import collections
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("MODEL", "gemma-4-31B-it")
MODE = os.environ.get("MODE", "sever")                 # sever | install -> results dir suffix
CHAN = os.environ.get("CHAN", "")          # "_kvrec" when the hybrid's recurrent channel was patched too
D = ROOT / "results" / (f"modifier_dho_{MODEL}" + ("_install" if MODE == "install" else "") + CHAN)
PREMISE = "none" if MODE == "sever" else "base"
BANDS = ["all", "mid", "low", "high"]
MODS = ["post_sq", "pre_sq", "post_inc", "post_dbl"]
BASES = ["sum", "prod", "max", "min", "altsum"]


def load(phase):
    return [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(D / phase / "*.json")))]


def rate(rs, k):
    return f"{sum(bool(r[k]) for r in rs)}/{len(rs)}" if rs else "-"


if "--select" in sys.argv:
    recs = load("dev")
    none = [r for r in recs if r["arm"] == PREMISE]
    print(f"dev ({MODE}): {len(recs)} records; premise arm '{PREMISE}' reproduces f_b {rate(none, 'kept_base')}, language ok {rate(none, 'lang_ok')}")
    tab = collections.defaultdict(list)
    for r in recs:
        if r["arm"].startswith("pooled_"):
            tab[(r["band"], r["alpha"])].append(r)
    print(f"pooled-donor arm ({MODE}) on dev recipients (python, javascript): target hits with language preserved / cells")
    print("| band \\ alpha | " + " | ".join(f"{al}" for al in sorted({k[1] for k in tab})) + " |")
    for band in [b for b in BANDS if any(k[0] == b for k in tab)]:
        cols = [tab[(band, al)] for al in sorted({k[1] for k in tab})]
        print(f"| {band} | " + " | ".join(f"{sum(r['hit'] for r in c)}/{len(c)} (valid {sum(r['status'] == 'ok' for r in c)}, whole {sum(r.get('whole_hit', r['hit']) for r in c)})" for c in cols) + " |")
    best = max(tab, key=lambda k: (sum(r["hit"] for r in tab[k]), -k[1], -BANDS.index(k[0])))
    sel = {"band": best[0], "alpha": best[1], "dev_hits": sum(r["hit"] for r in tab[best]), "dev_cells": len(tab[best]),
           "rule": "max target hits (language preserved) on dev; ties -> smaller alpha, then band order all, mid, low, high"}
    (D / "selection.json").write_text(json.dumps(sel, indent=1))
    print("selected:", sel)
    sys.exit(0)

recs = load("test")
if not recs:
    print("no test records yet"); sys.exit(0)
sel = json.loads((D / "selection.json").read_text()) if (D / "selection.json").exists() else {}
print(f"test ({MODE}): {len(recs)} records at band {sel.get('band')} alpha {sel.get('alpha')} (selected on dev, {sel.get('dev_hits')}/{sel.get('dev_cells')})")
ARMS = [x for x in ["none", "base", "single", "pooled", "rand", "wrong", "ceil_install", "ceiling"] if any(r["arm"] == x for r in recs)]
by = collections.defaultdict(list)
for r in recs:
    by[r["arm"]].append(r)
print("\n| arm | implements M[f_b] in lam_b | keeps f_b | other grid fn | valid program | language == lam_b | multi-block outputs | whole-output hit |")
print("|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    rs = by[arm]
    other = sum(1 for r in rs if r["impls"] and r["want"] not in r["impls"] and not r["kept_base"])
    print(f"| {arm} | {rate(rs, 'hit')} | {rate(rs, 'kept_base')} | {other}/{len(rs)} | "
          f"{sum(r['status'] == 'ok' for r in rs)}/{len(rs)} | {rate(rs, 'lang_ok')} | "
          f"{sum(r.get('n_blocks', 1) > 1 for r in rs)}/{len(rs)} | {rate(rs, 'whole_hit') if rs and 'whole_hit' in rs[0] else '-'} |")
addr = {(r["base"], r["mod"], r["lang"], r["wording"]) for r in by[PREMISE] if r["kept_base"]}
print(f"\naddressable cells (premise arm '{PREMISE}' reproduces f_b): {len(addr)} of {len(by[PREMISE])}")
for arm in [x for x in ["single", "pooled", "rand", "wrong", "ceil_install"] if x in by]:
    rs = [r for r in by[arm] if (r["base"], r["mod"], r["lang"], r["wording"]) in addr]
    print(f"  {arm:7s} on addressable: {rate(rs, 'hit')}")


def cut(field, values):
    print(f"\npooled / single / ceiling hits by {field}:")
    for v in values:
        p = [r for r in by["pooled"] if r[field] == v]; s = [r for r in by["single"] if r[field] == v]
        c = [r for r in by["ceiling"] if r[field] == v]; n = [r for r in by[PREMISE] if r[field] == v]
        print(f"  {str(v):11s} pooled {rate(p, 'hit'):7s} single {rate(s, 'hit'):7s} ceiling {rate(c, 'hit'):7s} "
              f"premise-keeps-base {rate(n, 'kept_base')}" + (f" install-ceiling {rate([r for r in by['ceil_install'] if r[field] == v], 'hit')}" if "ceil_install" in by else ""))


cut("mod", MODS)
cut("base", BASES)
cut("lang", sorted({r["lang"] for r in recs}))
cut("wording", [0, 1])
print("\nwhere squaring acts (separable bases sum / altsum; pooled arm): what landed")
for base in ["sum", "altsum"]:
    for m in ["post_sq", "pre_sq"]:
        rs = [r for r in by["pooled"] if r["base"] == base and r["mod"] == m]
        c = collections.Counter((r["impl"] or ("invalid" if r["status"] != "ok" else "other")) for r in rs)   # impl joins coincident keys with |
        print(f"  Delta_{m} on {base}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
print("\nwrong-modifier arm: what it installed")
c = collections.Counter()
for r in by["wrong"]:
    lab = "target M[f_b]" if r["hit"] else ("wrong M'[f_b]" if f"{r['wrong_mod']}:{r['base']}" in r["impls"] else
                                             ("f_b" if r["kept_base"] else ("other" if r["impls"] else "invalid")))
    c[lab] += 1
print("  " + ", ".join(f"{k} {v}" for k, v in c.most_common()))

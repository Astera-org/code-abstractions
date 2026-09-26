"""Control for test 2: with the L-program VISIBLE (plain in-context example), then the fence for L, does the model copy the
example verbatim? Also the carry arm (example masked). 4 functions x 10 languages x {full, carry}.
    uv run icl_section/test_copy_control.py --device cuda:1"""
import argparse, difflib, json, re, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "factored_programming_languages"))   # shared modules (algos, recipes, elicit, code_style, languages, tasks_function, entry)
from stage_common import Rig, DON
from tasks_function import LANGS
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gemma-4-31B-it"); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--funcs", nargs="+", default=["thulep", "vantril", "lufemun", "vupovuth"])
a = ap.parse_args(); rig = Rig(a.model, a.device); norm = lambda c: re.sub(r"\s+", " ", (c or "").strip()); OUT = HERE / "results" / ("copy_control.json" if a.model == "gemma-4-31B-it" else f"copy_control_{a.model}.json")
res = json.loads(OUT.read_text()) if OUT.exists() else {}   # merge: new functions join the old cells
for f in a.funcs:
    for L in LANGS:
        donor = DON[f]["impls"][L][0]
        with torch.no_grad():
            c = rig.model(input_ids=torch.tensor([rig.donor_ids(f, L, 0)], device=rig.dev), use_cache=True).past_key_values
            full = rig.generate_from_cache(c, L)                          # example visible
        carry = rig.swap_and_generate(f, L, 0, None, L)                   # example masked from fence + generation
        for arm, txt in (("full", full), ("carry", carry)):
            g = rig.grade(txt, f, L); code = g.get("code") or ""
            res[f"{f}|{L}|{arm}"] = {"hit": g["hit"], "identical": norm(code) == norm(donor), "similarity": difflib.SequenceMatcher(None, norm(code), norm(donor)).ratio(), "code": code[:500]}
        r1, r2 = res[f"{f}|{L}|full"], res[f"{f}|{L}|carry"]
        print(f"  {f:<9}{L:<11} full: hit={r1['hit']!s:<5} identical={r1['identical']!s:<5} sim={r1['similarity']:.2f} | carry: hit={r2['hit']!s:<5} identical={r2['identical']!s:<5} sim={r2['similarity']:.2f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
for arm in ("full", "carry"):
    rr = [v for k, v in res.items() if k.endswith("|" + arm)]
    print(f"{arm}: identical {sum(v['identical'] for v in rr)}/{len(rr)}, mean similarity {sum(v['similarity'] for v in rr)/len(rr):.2f}, hits {sum(v['hit'] for v in rr)}/{len(rr)}")

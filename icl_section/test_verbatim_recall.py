"""Test 2 (authors): if the compiled rows stored the donor PROGRAM, installing the rows written by a demo in language L and
asking for L again should reproduce the demo text. Same-language install: for each function and each language, the rows
compiled from the L-program demo (name-free ask) are installed into the neutral prompt and the model is asked for L. Compare
the output with the donor program: exact match (mod whitespace), difflib similarity, structure; and grade it.
    uv run icl_section/test_verbatim_recall.py --device cuda:2
"""
import argparse, difflib, json, re, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "factored_programming_languages"))
from stage_common import Rig, DON                                        # noqa: E402
from tasks_function import LANGS                                         # noqa: E402
from code_structure import structure                                     # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gemma-4-31B-it"); ap.add_argument("--device", default="cuda:2")
ap.add_argument("--funcs", nargs="+", default=["thulep", "vantril", "lufemun", "vupovuth"]); ap.add_argument("--out", default=None)
a = ap.parse_args(); a.out = a.out or str(HERE / "results" / ("verbatim_recall.json" if a.model == "gemma-4-31B-it" else f"verbatim_recall_{a.model}.json")); rig = Rig(a.model, a.device); Lr = len(rig.neutral_ids()); norm = lambda c: re.sub(r"\s+", " ", (c or "").strip())
res = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}   # merge: new functions join the old cells
for f in a.funcs:
    for L in LANGS:
        donor = DON[f]["impls"][L][0]
        txt = rig.install_and_generate(rig.suffix_kv(f, L, 0, Lr), L); g = rig.grade(txt, f, L); code = g.get("code") or ""
        r = {"hit": g["hit"], "which": g["which"], "status": g["status"], "identical": norm(code) == norm(donor), "similarity": difflib.SequenceMatcher(None, norm(code), norm(donor)).ratio(),
             "structure_out": structure(code, L) if code else "unknown", "structure_donor": structure(donor, L), "len_out": len(norm(code)), "len_donor": len(norm(donor)), "code": code[:600], "donor": donor[:600]}
        res[f"{f}|{L}"] = r
        print(f"  {f:<9}{L:<11} hit={r['hit']!s:<5} identical={r['identical']!s:<5} sim={r['similarity']:.2f}  structure out/donor {r['structure_out']}/{r['structure_donor']}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
n = len(res); print(f"\n{sum(r['identical'] for r in res.values())}/{n} identical to the donor program; mean similarity {sum(r['similarity'] for r in res.values())/n:.2f}; "
      f"same structure as donor {sum(r['structure_out']==r['structure_donor'] for r in res.values())}/{n}; hits {sum(r['hit'] for r in res.values())}/{n}")

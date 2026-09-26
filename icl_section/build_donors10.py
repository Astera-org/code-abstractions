"""Donor table for the 10-demo-language design (2026-09-23, the authors): one execution-verified program per language per
function. Every prompt is (demo language x, target language L) with x != L.
    python : the function's definition (invented: the bank's first program; nonce: the generator's first program)
    other 9: the model's OWN rendering, execution-verified — the first unmasked-arm hit in the 3-demo-language Stage A
             (stageA_3lang.json; donor order python0, python1, python2, java0, java1, go0, go1), `solve` renamed to the
             function's name; fallback: a verified self-rendering from the 3-language Stage C baseline arm.
Every program is re-verified here against the function's reference outputs on its battery before being written.
    uv run icl_section/build_donors10.py -> icl_section/donors.json  (source of each program recorded)
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "factored_programming_languages"))   # shared modules (algos, recipes, elicit, code_style, languages, tasks_function, entry)
import execute as S                                                     # noqa: E402
from entry import find_entry                                            # noqa: E402
from tasks_function import LANGS                                        # noqa: E402
OLD = json.loads((HERE / "donors_3lang.json").read_text())
A = json.loads((HERE / "results" / "stageA_3lang.json").read_text())
C = json.loads((HERE / "results" / "stageC_3lang.json").read_text())
ORDER = ["python0", "python1", "python2", "java0", "java1", "go0", "go1"]
NM = json.loads((HERE / "results" / "icl_nonce.json").read_text())


def verify(code, lang, want, battery):
    S.BATTERY = battery; entry = find_entry(lang, code)
    if not entry: return False
    for form in S.FORMS:
        st, got = S.RUNNERS[lang](code, entry, form)
        if st == "ok" and got == want: return True
    return False


donors = {}; missing = []
for f, d in OLD.items():
    want, bat = d["want"], d["battery"]; impls = {}; src = {}
    cands = {"python": [("definition", d["impls"]["python"][0])]}
    for L in LANGS:
        if L == "python": continue
        cands[L] = [(f"stageA_3lang:{x}|full", re.sub(r"\bsolve\b", f, A[f"{f}|{x}|{L}|full"]["code"]))
                    for x in ORDER if A.get(f"{f}|{x}|{L}|full", {}).get("hit")]
        cands[L] += [(f"stageC_3lang:{k}", re.sub(r"\bsolve\b", f, v["code"])) for k, v in C.items()
                     if k.startswith(f + "->") and "|none|" in k and k.endswith("|" + L) and v.get("hit") and v.get("code")]
        # further fallbacks: the nonce matrix's unmasked renderings (ordinary ICL, name visible), then Stage C swap hits
        cands[L] += [(f"icl_nonce:{f}|full|{L}", re.sub(r"\bsolve\b", f, it["code"])) for it in NM.get(f"{f}|full|{L}", [])
                     if it.get("hit") and it.get("code")]
        cands[L] += [(f"stageC_3lang:{k}", re.sub(r"\bsolve\b", f, v["code"])) for k, v in C.items()
                     if k.endswith(f"->{f}|swap|{L}") and v.get("hit") and v.get("code")]
    for L in LANGS:
        ok = next(((s, c) for s, c in cands[L] if verify(c, L, want, bat)), None)
        if ok is None: missing.append((f, L)); continue
        impls[L] = [ok[1]]; src[L] = ok[0]
    donors[f] = {"kind": d["kind"], "want": want, "battery": bat, "impls": impls, "source": src}
    print(f"  {f:<9} {len(impls)}/10 languages verified;", " ".join(f"{L}:{src[L].split(':')[0].replace('_3lang','')}" for L in LANGS if L in src))
print("MISSING:", missing or "none")
(HERE / "donors.json").write_text(json.dumps(donors, indent=1)); print(f"-> donors.json ({sum(len(v['impls']) for v in donors.values())} programs)")

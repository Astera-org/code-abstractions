"""Recount the cross-language installation numbers (fig_install.py / table_install.py) from the Stage B result files.
Per model: the four functions of the pick, split into regular ("invented") and nonce; arms none / pooled / rand / wrong
(hit = renders the asked function) and the single-donor cells (keys f|x|L with x a demo language); hybrid channel arms on
Qwen. Output: results/fig_install_numbers.json  {model: {"all"|"invented"|"nonce": {arm: [hits, trials]}, "functions": {...}}}.
    uv run icl_section/count_install.py [--pick balanced|own] [--out results/fig_install_numbers.json]
"""
import argparse, sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; R = HERE / "results"
DON = json.loads((HERE / "donors.json").read_text()); KIND = {f: v["kind"] for f, v in DON.items()}
FILES = {"gemma-4-31B-it": "stageB.json", "gemma-4-26b-a4b": "stageB_gemma-4-26b-a4b.json", "Qwen3.6-27B": "stageB_qwen3.6-27b.json", "RWKV7-13.3B": "stageB_rwkv.json"}
OWN = {"gemma-4-31B-it": ["thulep", "vantril", "lufemun", "vupovuth"], "gemma-4-26b-a4b": ["thulep", "vantril", "kanex", "lufemun"],
       "Qwen3.6-27B": ["thulep", "vantril", "kanex", "faromin"], "RWKV7-13.3B": ["thulep", "vantril", "lufemun", "vupovuth"]}
BAL = ["thulep", "vantril", "kanex", "faromin"]
ARMS = ("none", "pooled", "single", "wrong", "rand", "pooled_kv", "pooled_rec")
if "--models" in sys.argv:                     # --models key=stageB_file,... overrides the model set (smoke test)
    FILES = dict(x.split("=", 1) for x in sys.argv[sys.argv.index("--models") + 1].split(","))
    sys.argv = [x for i, x in enumerate(sys.argv) if x != "--models" and (i == 0 or sys.argv[i - 1] != "--models")]
ap = argparse.ArgumentParser(); ap.add_argument("--pick", choices=["balanced", "own"], default="balanced"); ap.add_argument("--out", default=str(R / "fig_install_numbers.json"))
a = ap.parse_args(); out = {}
for m, fn in FILES.items():
    B = json.loads((R / fn).read_text()); funcs = BAL if a.pick == "balanced" else OWN[m]
    rec = {k: {} for k in ("all", "invented", "nonce")}
    for f in funcs:
        for k, v in B.items():
            p = k.split("|")
            if p[0] != f or not isinstance(v, dict) or v.get("hit") is None: continue
            arm = p[1] if p[1] in ARMS else "single"
            for grp in ("all", KIND[f]):
                h = rec[grp].setdefault(arm, [0, 0]); h[0] += bool(v["hit"]); h[1] += 1
    missing = [f for f in funcs if not any(k.startswith(f + "|") for k in B)]
    rec["functions"] = {"invented": [f for f in funcs if KIND[f] == "invented"], "nonce": [f for f in funcs if KIND[f] == "nonce"]}
    if missing: rec["missing"] = missing
    out[m] = rec
    print(f"{m:<16} " + "  ".join(f"{arm} {rec['all'][arm][0]}/{rec['all'][arm][1]}" for arm in ARMS if arm in rec["all"]) + (f"   MISSING {missing}" if missing else ""))
Path(a.out).write_text(json.dumps(out, indent=1)); print("wrote", a.out)

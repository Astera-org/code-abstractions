"""Stage A table + the selection rule (stated before the numbers; authors, 2026-09-23): rank functions by coverage of
the CARRY arm — greedy rollouts with the example shown, the language-free ask prefilled, then attention to the example
masked from the fence and every generated token — over all (demo language, target language) cells with demo != target;
take the top two invented algorithms and the top two nonces. Ties: higher unmasked (full) coverage, then name.
    uv run icl_section/pick_four.py [results/stageA.json]"""
import json, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "stageA.json"
path = path if path.exists() else HERE / path                        # accept paths relative to the CWD or to icl_section/
A = json.loads(path.read_text()); DON = json.loads((HERE / "donors.json").read_text())
L10 = ["python", "javascript", "c", "cpp", "java", "go", "rust", "ruby", "ocaml", "haskell"]
carry = defaultdict(lambda: [0, 0]); full = defaultdict(lambda: [0, 0]); floor = defaultdict(lambda: [0, 0])
by_demo = defaultdict(lambda: defaultdict(lambda: [0, 0])); by_target = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for k, v in A.items():
    f, x, L, arm = k.split("|")
    if arm == "floor": floor[f][0] += v["hit"]; floor[f][1] += 1
    elif arm == "full": full[f][0] += v["hit"]; full[f][1] += 1
    else:
        carry[f][0] += v["hit"]; carry[f][1] += 1
        by_demo[f][x][0] += v["hit"]; by_demo[f][x][1] += 1; by_target[f][L][0] += v["hit"]; by_target[f][L][1] += 1
rank = lambda f: (-carry[f][0] / max(carry[f][1], 1), -full[f][0] / max(full[f][1], 1), f)
rows = sorted(DON, key=rank)
print(f"{'kind':<9}{'function':<10}{'carry':>8}{'full':>8}{'floor':>7}   carry by demo language")
for f in rows:
    print(f"{DON[f]['kind']:<9}{f:<10}{carry[f][0]:>4}/{carry[f][1]:<4}{full[f][0]:>3}/{full[f][1]:<4}{floor[f][0]:>2}/{floor[f][1]:<3}   "
          + " ".join(f"{x[:4]}:{by_demo[f][x][0]}/{by_demo[f][x][1]}" for x in L10))
print("\ncarry hits by target language (out of 9 demo languages each):")
print(f"{'function':<10}" + "".join(f"{L[:5]:>7}" for L in L10))
for f in rows: print(f"{f:<10}" + "".join(f"{by_target[f][L][0]:>7}" for L in L10))
pick = {kind: [f for f in rows if DON[f]["kind"] == kind][:2] for kind in ("invented", "nonce")}
print(f"\nSELECTED (top-2 carry coverage per class; ties by full coverage): invented {pick['invented']}, nonce {pick['nonce']}")
(HERE / "results" / "stageA_pick.json").write_text(json.dumps(pick, indent=1))

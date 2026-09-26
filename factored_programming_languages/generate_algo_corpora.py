"""Write the 9-example Python-only SFT corpora, one pair of files per algorithm (vantril, thulep, kanex, faromin).

3 paraphrases x 3 portable implementations = 9 examples, in the two assistant
styles the SFT plumbing supports: `fence` (```python ... ```) for the chat
models and `bare` for pythia.

vantril is regenerated as a CONSISTENCY CHECK, not to overwrite: the existing
data/vantril_portable_*.json are the validated files behind the published
result, and this script asserts it reproduces them exactly before writing
anything new. If that assert fires, the generator and the validated corpus
have drifted and the new corpora are not comparable to the old one.

    uv run factored_programming_languages/generate_algo_corpora.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from algos import ALGOS, PARAPHRASES                                    # noqa: E402

DATA = HERE / "data"


def build(algo, style):
    spec = ALGOS[algo]
    out = []
    for para in PARAPHRASES[style]:
        for src in spec["impls"]:
            body = f"```python\n{src}\n```" if style == "fence" else src
            out.append({"user": para.format(desc=spec["desc"]),
                        "assistant": body})
    return out


# ---- consistency check against the validated vantril corpus ---------------
for style in ("fence", "bare"):
    ref = DATA / f"vantril_portable_{style}.json"
    if not ref.exists():
        sys.exit(f"missing {ref}")
    have = json.loads(ref.read_text())
    mine = build("vantril", style)
    if have != mine:
        for i, (a, b) in enumerate(zip(have, mine)):
            if a != b:
                print(f"first difference at example {i}:")
                print(f"  validated user: {a['user']!r}")
                print(f"  generated user: {b['user']!r}")
                print(f"  validated asst: {a['assistant']!r}")
                print(f"  generated asst: {b['assistant']!r}")
                break
        sys.exit(f"generator does not reproduce {ref.name} -- refusing to "
                 f"write corpora that would not be comparable")
print("[check] generator reproduces both validated vantril corpora exactly")

# ---- write the three other algorithms of the paper (Fig. 18) -------------
for algo in ("thulep", "kanex", "faromin"):
    for style in ("fence", "bare"):
        p = DATA / f"{algo}_portable_{style}.json"
        p.write_text(json.dumps(build(algo, style), indent=1))
        print(f"  wrote {p.relative_to(HERE)}  ({len(build(algo, style))} examples)")

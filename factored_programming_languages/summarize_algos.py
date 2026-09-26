"""The five-algorithm reuse surface: every language, every dose, nothing gated.

No dose is selected and no language is dropped. A single gated number was
hiding the shape of the thing: morvex on gemma-4-31B acquires perfectly at
8e-6 and transfers to nothing, while at 2e-6 it transfers to five languages
and takes no damage at all -- and the gate rule, which looks only at Python,
picked the first. Both rows are the result, so both rows are printed.

Three views:

  1. the full ladder, one row per (algorithm, model, dose)
  2. transfer against dose, as a matrix
  3. per-language detail, one grid per algorithm  (--langs)

Columns:

    S_py   acquisition -- mean battery agreement over the three HELD-OUT
           Python paraphrases (never one of the three trained on). `gate`
           marks the lowest dose where all three are exact hits.
    py     did the taught language itself come out right
    ran    held-out languages that ran and matched, of nine
    +tgt   arrived: the procedure is in the generation, in the target
           language's shape, but it does not run
    +py    arrived as a bare Python body instead
    dead   languages where the checkpoint now fails digit_sum -- a function it
           knew BEFORE any fine-tuning. Dose damage: a miss there is not
           evidence about the taught computation either way.
    nat    the untouched base's own score on this computation in these nine
           languages, described in plain words with the invented name removed.
           The ceiling any reuse number should be read against.

    uv run factored_programming_languages/summarize_algos.py
    uv run factored_programming_languages/summarize_algos.py --langs
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from algos import ALGOS                                                  # noqa: E402
from algorithm_present_algo import (algorithm_present, looks_like_python,  # noqa: E402
                                    gate_of, HELD)

R = ROOT / "results"
MODELS = ["gemma-4-31B-it", "gemma-4-26b-a4b", "qwen3.6-27b",
          "pythia-6.9b-tulu"]
TAGS = ("teacher", "teacher_hi", "teacher_vhi")
LR = {"teacher": "5e-7", "teacher_hi": "2e-6", "teacher_vhi": "8e-6"}
SHORT = {"gemma-4-31B-it": "gemma-31B", "gemma-4-26b-a4b": "gemma-26B",
         "qwen3.6-27b": "qwen3.6-27B", "pythia-6.9b-tulu": "pythia"}
# ran / arrived in target shape / arrived as python / dose damage / nothing
MARK = {"ran": "#", "tgt": "+", "py": "p", "dead": "x", "no": "."}
# explicit, not l[:4]: javascript and java both truncate to "java"
ABBR = {"javascript": "js", "c": "c", "cpp": "cpp", "java": "java", "go": "go",
        "rust": "rust", "ruby": "ruby", "ocaml": "ml", "haskell": "hs"}


def per_lang(algo, model, tag):
    """-> {lang: one of ran/tgt/py/dead/no}, or {} if the cell has not run"""
    d = R / f"eval{algo}_{model}_{tag}"
    if not d.is_dir():
        return {}
    out = {}
    for l in HELD:
        f = d / f"{algo}_{l}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        fr = d / f"renders_{l}.json"
        rend = json.loads(fr.read_text()).get("hit") if fr.exists() else True
        if r.get("hit"):
            out[l] = "ran"
        elif not rend:
            out[l] = "dead"
        elif algorithm_present(algo, r.get("code")):
            out[l] = "py" if looks_like_python(r.get("code")) else "tgt"
        else:
            out[l] = "no"
    return out


def row(algo, model, tag):
    pl = per_lang(algo, model, tag)
    if not pl:
        return None
    d = R / f"eval{algo}_{model}_{tag}"
    sf = d / "summary.json"
    s = json.loads(sf.read_text()) if sf.exists() else {}
    fp = d / f"{algo}_python.json"
    c = {k: sum(v == k for v in pl.values()) for k in MARK}
    return dict(s_py=s.get("S_python_heldout"),
                py=json.loads(fp.read_text()).get("hit") if fp.exists() else None,
                ran=c["ran"], at=c["tgt"], ap=c["py"], dead=c["dead"],
                n=len(pl), lang=pl)


def native(algo, model):
    d = R / f"eval{algo}_{model}_base"
    if not d.is_dir():
        return None
    return sum(json.loads((d / f"native_{l}.json").read_text()).get("hit")
               for l in HELD if (d / f"native_{l}.json").exists())


# ------------------------------------------------------------ 1. the ladder --
print(f"{'algo':>8} {'model':>12} {'dose':>6} {'S_py':>6} {'py':>3} "
      f"{'ran':>6} {'+tgt':>5} {'+py':>4} {'dead':>5} {'nat':>5}")
for algo in ALGOS:
    for m in MODELS:
        g, nat, seen = gate_of(algo, m), native(algo, m), False
        for t in TAGS:
            r = row(algo, m, t)
            if not r:
                continue
            seen = True
            sp = "-" if r["s_py"] is None else f"{r['s_py']:.2f}"
            print(f"{algo:>8} {SHORT[m]:>12} {LR[t]:>6} {sp:>6} "
                  f"{('y' if r['py'] else 'n'):>3} {r['ran']:>3}/{r['n']:<2} "
                  f"{r['at']:>5} {r['ap']:>4} {r['dead']:>5} "
                  f"{(str(nat) + '/9') if nat is not None else '-':>5}"
                  + ("   gate" if t == g else ""))
        if seen:
            print()

# ----------------------------------------------------- 2. transfer vs dose --
print("\nheld-out languages that ran, of nine, at each dose "
      "(x = how many the checkpoint can no longer write)")
print(f"{'':>8} {'':>12}" + "".join(f"{LR[t]:>12}" for t in TAGS) + f"{'nat':>8}")
for algo in ALGOS:
    for m in MODELS:
        cells, seen = [], False
        for t in TAGS:
            r = row(algo, m, t)
            if not r:
                cells.append("-")
                continue
            seen = True
            cells.append(f"{r['ran']}/{r['n']}"
                         + (f" x{r['dead']}" if r["dead"] else ""))
        if seen:
            nat = native(algo, m)
            print(f"{algo:>8} {SHORT[m]:>12}"
                  + "".join(f"{c:>12}" for c in cells)
                  + f"{(str(nat) + '/9') if nat is not None else '-':>8}")

# --------------------------------------------------------- 3. per language --
if "--langs" in sys.argv:
    key = "   ".join(f"{MARK[k]} {k}" for k in ("ran", "tgt", "py", "dead", "no"))
    print(f"\nper language   ({key})")
    hdr = "".join(f"{ABBR[l]:>5}" for l in HELD)
    for algo in ALGOS:
        print(f"\n{algo}\n{'':>12}{'':>6}{hdr}")
        for m in MODELS:
            for t in TAGS:
                r = row(algo, m, t)
                if not r:
                    continue
                print(f"{SHORT[m]:>12}{LR[t]:>6}"
                      + "".join(f"{MARK[r['lang'].get(l, 'no')]:>5}"
                                for l in HELD))

"""Did the ALGORITHM arrive, separately from whether it rendered -- for each
of the five invented computations.

algorithm_present.py answers this for vantril alone, with a hand-written
structural rule. The same question across five algorithms needs five rules,
and a hand-written structural rule is exactly the sort of instrument that
quietly mislabels things, so each one is stated as

    key           the step that SEPARATES this computation from its neighbours.
                  Mandatory. A digit loop is never evidence on its own --
                  every one of the five has it, and so does digit_sum.
    corroborate   the program reaches the decimal digits at all.

and each is then checked against three controls before any number is printed:

    fires on every generation that RAN and matched the reference (want all)
    fires on no untaught base generation for the same name    (want none)
    fires on no CORRECT digit_sum program                     (want none)

The digit_sum control is the load-bearing one. Those 606 programs have the
digit loop that corroborates all five rules and none of the separating steps,
so a rule that fires there is matching the corroboration rather than the
computation. The base control is weaker: under an invented name the model
often writes nothing coherent, so silence there is cheap.

All five rules currently sit at 170/170, 0/36, 0/606. Every clause here was
written against a miss or a false positive the controls surfaced -- matching
variable NAMES in particular did not survive contact with the data, since
models write high/low, h/l, max_digit/min_digit and hi/lo for the same thing.
Re-run this file after adding any generations; the rules are fitted to what
models actually wrote, so new data can break them.

    uv run factored_programming_languages/algorithm_present_algo.py
"""
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from algos import ALGOS                                                  # noqa: E402

R = ROOT / "results"
HELD = ["javascript", "c", "cpp", "java", "go", "rust", "ruby", "ocaml",
        "haskell"]

# Shared corroboration: the program reaches the decimal digits. Rendering the
# number as a string counts as much as mod-10 does -- `length . nub . show $ n`
# extracts digits every bit as much as a divide-by-ten loop, and the original
# rule scored that correct Haskell program as "algorithm absent".
DIGITS = (r"%\s*10|`mod`\s*10|\bmod\s+10\b|/\s*10|/=\s*10|div\s+10|//\s*10|"
          r"\bshow\b|\bstr\s*\(|toString|to_string|\bString\s*\(|\.chars\b|"
          r"\bdigits\b")

# There is deliberately no second corroborating clause. The original vantril
# rule also required a tally word (sum/count/total/++), on the reasoning that a
# digit loop alone is not evidence. But the separating step below already
# carries the discrimination, and the tally clause only produced false
# negatives: it rejected the canonical thulep program, which accumulates
# nothing (it tracks hi and lo and subtracts), and the canonical krendal
# program, whose accumulator is spelled `even = even + d`. Dropping it takes
# detection on known-correct programs from 45/47 to 47/47 while both controls
# stay at zero -- so the clause was costing accuracy and buying nothing.

# the separating step, one per computation
KEY = {
    # a slot FLAGGED rather than accumulated, or a set/uniq/bitmask -- the
    # dedup that distinguishes counting distinct digits from summing them
    # `.(d) <- true` is OCaml's array store; without it a correct OCaml
    # presence-table program scored as "algorithm absent"
    # one added per digit: `d + 1` on the digit itself, or the digit sum plus a digit count. Haskell spells it
    # `succ`/`(+1)`. Bare `+= 1` loop counters are NOT accepted (that is every C string walk).
    "kanex": [r"(?:%\s*10|\bd\b|\bdigit\w*|\bc\b|\bch\b)\s*\)?\s*\+\s*1\b|1\s*\+\s*\(?\s*(?:\w+\s*%\s*10|\bd\b|\bdigit)|"
              r"\+\s*(?:count|cnt|len|length|size|n_?digits|digits)\b|(?:count|cnt|len|length|size|n_?digits)\s*\+|"
              r"\bsucc\b|\(\s*\+\s*1\s*\)|\bd\s*\+\s*1\b|\bi\s*\+\s*1\b|\+\s*1\s*\)"],
    # the dedup clause of vantril (a slot flagged, a set, a bitmask) AND the +1 map of kanex
    "faromin": [r"\[[^\]]*\]\s*=\s*(?:1|true|True)\b|"
                r"\.\(\s*[^)]*\)\s*<-\s*(?:true|1)\b|\.(?:add|insert)\s*\(|"
                r"\bnub\b|\buniq\b|\bdistinct\b|\bSet\b|\bset\s*\(|\bHashSet\b|"
                r"\bmap\[[^\]]*\]bool\b|\|=?\s*\(?\s*1\s*<<|bitCount|popcount|countOnes",
              r"(?:%\s*10|\bd\b|\bdigit\w*|\bc\b|\bch\b)\s*\)?\s*\+\s*1\b|1\s*\+\s*\(?\s*(?:\w+\s*%\s*10|\bd\b|\bdigit)|"
              r"\+\s*(?:count|cnt|len|length|size|n_?digits|digits)\b|(?:count|cnt|len|length|size|n_?digits)\s*\+|"
              r"\bsucc\b|\(\s*\+\s*1\s*\)|\bd\s*\+\s*1\b|\bi\s*\+\s*1\b|\+\s*1\s*\)"],
    "vantril": [r"\[[^\]]*\]\s*=\s*(?:1|true|True)\b|"
                r"\.\(\s*[^)]*\)\s*<-\s*(?:true|1)\b|\.(?:add|insert)\s*\(|"
                r"\bnub\b|\buniq\b|\bdistinct\b|\bSet\b|\bset\s*\(|\bHashSet\b|"
                r"\bmap\[[^\]]*\]bool\b|\|=?\s*\(?\s*1\s*<<|bitCount|popcount|"
                r"countOnes"],
    # position parity: only every other digit is added. The last alternative is
    # the toggle-flag idiom -- some variable is tested against 1 and set to 0
    # in the same body -- which models write as often as they write `% 2`.
    "morvex": [r"%\s*2\b|`mod`\s*2|\bmod\s+2\b|\beven\b|\bodd\b|"
               r"step_by\s*\(\s*2\s*\)|(?:\+=|\+)\s*2\b|"
               r"1\s*-\s*(?:take|flip|toggle|parity)|"
               r"!\s*(?:take|flip|toggle)|not\s+(?:take|flip|toggle)|"
               r"\b(\w+)\s*==?\s*1\b[\s\S]{0,200}?\b\1\s*=\s*0\b"],
    # BOTH extremes tracked, and subtracted from one another. Matching on
    # variable NAMES does not survive contact with the data -- models write
    # high/low, h/l, max_digit/min_digit, hi/lo -- so the first two clauses
    # also accept the structure itself: `d > X` followed by `X = d`, and the
    # same with `<`. That is what tracking a running maximum IS.
    "thulep": [r"(\w+)\s*>\s*(\w+)[\s\S]{0,120}?\b\2\s*(?::?=|<-)\s*\1\b|"
               r"\bmax\w*|\bMath\.max|\bmaximum\b|\bhi\b|\bhighest\b|"
               r"\blargest\b|\bbiggest\b|\bmx\b",
               r"(\w+)\s*<\s*(\w+)[\s\S]{0,120}?\b\2\s*(?::?=|<-)\s*\1\b|"
               r"\bmin\w*|\bMath\.min|\bminimum\b|\blo\b|\blowest\b|"
               r"\bsmallest\b|\bmn\b",
               r"\w\s*-\s*\w"],
    # per-digit sign alternation AND an absolute value. `sign = -sign` is
    # required rather than any negation: a digit_sum that handles negative
    # input writes `sign = -1` and `if n < 0`, which tripped both clauses and
    # was the one false positive in the whole control set. Dividing by 100
    # counts as alternation too -- it consumes a digit PAIR and subtracts.
    "krendal": [r"sign\s*=\s*-\s*sign|\*=\s*-\s*1|=\s*-\s*sign|%\s*2\b|"
                r"`mod`\s*2|\bmod\s+2\b|\beven\b|\bodd\b|\balternat|"
                r"/=?\s*100\b|div\s+100\b",
                r"\babs\b|\bfabs\b|\.abs\b|\bnegate\b|<\s*0\b|0\s*-\s*\w"],
    # a per-digit COUNT (incremented, not flagged) and an exactly-one test
    "pilnoth": [r"\[[^\]]*\]\s*(?:\+=|\+\+)|"
                r"\[[^\]]*\]\s*=\s*\w*\s*\[[^\]]*\]\s*\+\s*1|"
                r"\.(?:count|tally|frequency|freq)|\bcounts\b|\btally\b",
                r"==\s*1\b|=\s*1\s*(?:then|->|\))|\.count\b[^\n]*1"],
}


def strip(code):
    """line comments strip to end of LINE. With re.S they strip to end of file,
    so a Python `//` integer division would delete the rest of the program."""
    src = re.sub(r"--[^\n]*|//[^\n]*|#[^\n]*", " ", code)
    return re.sub(r"\(\*.*?\*\)", " ", src, flags=re.S)


def algorithm_present(algo, code):
    if not code or not code.strip():
        return False
    if algo not in KEY:                  # no rule written (dorvath): never claim the algorithm arrived
        return False
    src = strip(code)
    if not all(re.search(p, src) for p in KEY[algo]):
        return False
    return bool(re.search(DIGITS, src))


def looks_like_python(code):
    """Is this body Python, whatever language was asked for?

    Parsing it decides, not the shared `detect_language` regex, which reads
    Ruby's `def solve(n)` as Python and would put every correct Ruby program
    in the wrong bucket. Ruby's `end` and its missing colon are both syntax
    errors to the Python parser, so the check separates the two; the control
    in __main__ verifies that on the real Ruby generations.

    Used only to SPLIT the arrived-but-did-not-run band: a Python body under a
    C header is a different failure from a target-language program with one
    Python operator left in it, and averaging them hides which one a model does.
    """
    if not code or not code.strip():
        return False
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               for n in ast.walk(tree))


def cells(algo, model, tag):
    d = R / f"eval{algo}_{model}_{tag}"
    if not d.is_dir():
        return {}
    return {l: json.loads((d / f"{algo}_{l}.json").read_text())
            for l in HELD if (d / f"{algo}_{l}.json").exists()}


def live_langs(algo, model, tag):
    """the held-out languages this checkpoint can still write AT ALL.

    A checkpoint is only an instrument for "did the taught computation reach
    Rust" while it can still write Rust. `renders` asks it for digit_sum -- a
    function it knew before any fine-tuning -- so a language that fails there
    is a broken probe, and a miss in that language is not evidence about the
    taught computation. Those languages leave the denominator.
    """
    d = R / f"eval{algo}_{model}_{tag}"
    out = []
    for l in HELD:
        f = d / f"renders_{l}.json"
        if not (d / f"{algo}_{l}.json").exists():
            continue
        if not f.exists() or json.loads(f.read_text()).get("hit"):
            out.append(l)
    return out


def gate_of(algo, model, tags=("teacher", "teacher_hi", "teacher_vhi")):
    """the LOWEST dose whose three held-out Python acquisition probes all pass.

    Reported as an ANNOTATION, not used to select what gets shown. Collapsing
    a dose ladder to one gated number hides the thing the ladder measures:
    morvex on gemma-4-31B acquires perfectly at 8e-6 and transfers to nothing
    (all nine languages fail digit_sum there, so the checkpoint cannot write
    them at all), while at 2e-6 it transfers to five with no damage. Both
    facts are the result. summarize_algos.py prints every dose.

    Derived, never hard-coded: hard-coding it once put pythia on teacher_vhi,
    where the algorithm reaches 9/9 languages, instead of its actual gate.
    """
    for tag in tags:
        d = R / f"eval{algo}_{model}_{tag}"
        acq = sorted(d.glob("acq*_python.json")) if d.is_dir() else []
        if acq and all(json.loads(q.read_text()).get("hit") for q in acq):
            return tag
    return None


if __name__ == "__main__":
    MODELS = ["gemma-4-31B-it", "gemma-4-26b-a4b", "qwen3.6-27b",
              "pythia-6.9b-tulu"]
    # The digit_sum control is the sharp one. Every `renders` generation that
    # ran and matched is a CORRECT digit-sum program: it has the digit loop and
    # the tally that corroborate all five rules, and none of the separating
    # steps. A rule that fires there is matching the corroboration, not the
    # computation. The base control is weaker -- under an invented name the
    # model often writes nothing coherent -- so both are reported.
    ds = [r for f in R.glob("eval*/renders_*.json")
          for r in [json.loads(f.read_text())] if r.get("hit")]
    print(f"{'algo':>9} {'fires on correct':>17} {'fires on base':>14} "
          f"{'fires on digit_sum':>19}   verdict")
    for algo in ALGOS:
        pos = [r for f in R.glob(f"eval{algo}_*/{algo}_*.json")
               for r in [json.loads(f.read_text())] if r.get("hit")]
        neg = [r for m in MODELS for r in cells(algo, m, "base").values()]
        fp = sum(algorithm_present(algo, r.get("code")) for r in pos)
        fn = sum(algorithm_present(algo, r.get("code")) for r in neg)
        fd = sum(algorithm_present(algo, r.get("code")) for r in ds)
        good = (not pos or fp == len(pos)) and fn == 0 and fd == 0
        print(f"{algo:>9} {fp:>12}/{len(pos):<4} {fn:>10}/{len(neg):<4} "
              f"{fd:>14}/{len(ds):<4}   " + ("ok" if good else "RULE NEEDS WORK"))
    print()

    print(f"{'algo':>9} {'model':>18} {'gate':>12}  {'present':>8}  "
          f"{'runs':>6}   arrived but did not render")
    for algo in ALGOS:
        for m in MODELS:
            tag = gate_of(algo, m)
            if not tag:
                print(f"{algo:>9} {m:>18} {'no gate':>12}  "
                      f"{'-':>8}  {'-':>6}   (acquisition never passed)")
                continue
            rs = cells(algo, m, tag)
            if not rs:
                continue
            alg = {l: algorithm_present(algo, r.get("code")) for l, r in rs.items()}
            hit = {l: bool(r["hit"]) for l, r in rs.items()}
            gap = [l for l in rs if alg[l] and not hit[l]]
            print(f"{algo:>9} {m:>18} {tag:>12}  "
                  f"{sum(alg.values()):>5}/{len(rs):<2}  {sum(hit.values()):>3}/{len(rs):<2}"
                  f"   " + (", ".join(gap) if gap else "-"))

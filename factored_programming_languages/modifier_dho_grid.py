"""Grid, references and grader shared by run_modifier_dho.py and regrade_modifier_dho.py (double-held-out modifier
transfer). Kept separate so saved generations can be re-graded without loading a model.

Grading rule (2026-09-13, after the first dev cells): the model's answer is its FIRST fenced block. Under the
intervention gemma-4-31B often emits a correct first implementation and then argues with the (severed) prompt in
prose -- "Wait, the prompt asks for the product of the digits ..." -- and emits a second block; `extract_code`
returns None for outputs with more than one container, which turned those into no_entry. The whole-output grading
is kept alongside as whole_* for transparency. This is the first-emitted-implementation rule of
result_writeups/selectors_and_pointers.md (experiment E).
"""
import itertools
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from elicit import extract_code                                            # noqa: E402
from entry import find_entry, detect_language                          # noqa: E402
import execute as S                                                        # noqa: E402

BASES = ["sum", "prod", "max", "min", "altsum"]
MODS = ["post_sq", "pre_sq", "post_inc", "post_dbl"]
LANGS = ["python", "javascript", "java", "go", "rust", "haskell"]
DEV_LANGS = ["python", "javascript"]
TEST_LANGS = ["java", "go", "rust", "haskell"]
BANDS = {"all": (0.0, 1.0), "low": (0.0, 0.5), "mid": (0.25, 0.75), "high": (0.5, 1.0)}
ALPHAS = [0.5, 1.0, 1.5, 2.0]
BATTERY = [7, 1234, 5, 90, 406, 2468, 3917, 8052]
S.BATTERY = BATTERY


def _ds(n):
    return [int(c) for c in str(n)]


def _alt(ds):
    return sum(d if i % 2 == 0 else -d for i, d in enumerate(ds))


AGG = {"sum": sum, "prod": math.prod, "max": max, "min": min, "altsum": _alt}


def ref(base, mod):
    def f(n):
        ds = _ds(n)
        if mod == "pre_sq":
            ds = [d * d for d in ds]
        v = AGG[base](ds)
        return {"post_sq": v * v, "post_inc": v + 1, "post_dbl": 2 * v}.get(mod, v)
    return f


NP_BASE = {"sum": "the sum of the digits", "prod": "the product of the digits", "max": "the largest digit",
           "min": "the smallest digit", "altsum": "the alternating sum of the digits"}
NP_PRESQ = {"sum": "the sum of the squares of the digits", "prod": "the product of the squares of the digits",
            "max": "the largest of the squares of the digits", "min": "the smallest of the squares of the digits",
            "altsum": "the alternating sum of the squares of the digits"}
CLAUSE = {"altsum": " (first digit minus the second, plus the third, and so on)"}
CLAUSE_PRESQ = {"altsum": " (first squared digit minus the second, plus the third, and so on)"}


def key(base, mod=None):
    return base if mod is None else f"{mod}:{base}"


def desc(base, mod, w):
    if mod == "pre_sq":
        np_, cl = NP_PRESQ[base], CLAUSE_PRESQ.get(base, "")
    else:
        core = NP_BASE[base]
        np_ = {None: core, "post_sq": f"the square of {core}", "post_inc": f"one more than {core}",
               "post_dbl": f"twice {core}"}[mod]
        cl = CLAUSE.get(base, "")
    return (f"returns {np_} of a non-negative integer{cl}" if w == 0
            else f"takes a non-negative integer and returns {np_} of that number{cl}")


FUNCS = [(b, None) for b in BASES] + [(b, m) for m in MODS for b in BASES]
WANTV = {key(b, m): [ref(b, m)(x) for x in BATTERY] for b, m in FUNCS}
COINCIDE = {frozenset({key(b, "post_sq"), key(b, "pre_sq")}) for b in ("prod", "max", "min")}   # extensionally equal
_coll = {frozenset({k1, k2}) for k1, k2 in itertools.combinations(WANTV, 2) if WANTV[k1] == WANTV[k2]}
assert _coll == COINCIDE, f"unexpected reference collisions on the battery: {_coll ^ COINCIDE}"

FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def first_block(txt):
    m = FENCE.search(txt or "")
    if m:
        return m.group(1).strip()
    code, _ = extract_code(txt or "", "auto")
    return code


def run_code(code, lam_b):
    det = detect_language(code) if code else "unknown"
    entry = find_entry(det, code) if det != "unknown" else None
    got, st = None, "no_entry"
    if code and entry and det in S.RUNNERS:
        for form in S.FORMS:
            st, got = S.RUNNERS[det](code, entry, form)
            if st == "ok" and got is not None:
                break
    impls = [k for k, v in WANTV.items() if got is not None and list(got) == v]   # >1 only for the coincident pairs
    return {"lang_detect": det, "lang_ok": det == lam_b, "status": st, "outputs": got,
            "impls": impls, "impl": "|".join(impls) if impls else None}


def grade(txt, lam_b, want_key, base_key):
    code = first_block(txt)
    g = run_code(code, lam_b)
    w_code, _ = extract_code(txt or "", "auto")
    w = g if w_code == code else run_code(w_code, lam_b)
    return {"code": code, "text": txt, "n_blocks": len(FENCE.findall(txt or "")), **g,
            "hit": want_key in g["impls"] and g["lang_ok"], "kept_base": base_key in g["impls"],
            "whole_code": w_code, "whole_status": w["status"], "whole_impl": w["impl"],
            "whole_hit": want_key in w["impls"] and w["lang_ok"]}

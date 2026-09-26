"""Procedurally generated nonce functions: novel COMPOSITIONS of digit primitives the model already has.

A nonce is (select, map, reduce) over the decimal digits of n. The model knows every primitive; the composition and
the name are new (three-tiers: novel composition of known steps compiles; novel steps only transcribe). Each nonce
comes with an executable reference, two verified Python implementations (loop and functional), a templated
specification for step-3 recipients, and a random pronounceable name.

Battery = false-friend guard: chosen so that on it every accepted nonce differs from (i) each of the five invented
algorithms, (ii) each bank function, (iii) every other accepted nonce, and (iv) each of its own ONE-STEP
simplifications (a component replaced by its default) -- silent regression to a canonical neighbour cannot pass.

    uv run icl_section/nonce.py --n 12 --seed 0        -> icl_section/nonce_functions.json
"""
import argparse, itertools, json, math, random, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "factored_programming_languages"))   # shared modules
from algos import ALGOS                                                  # noqa: E402
import execute as S                                                      # noqa: E402

# ---- primitives -------------------------------------------------------------------------------------------------
# digits: most-significant first. Positions for parity are counted FROM THE RIGHT (last digit = position 0), as the
# invented algorithms do.
SELECT = {
    "all":      ("all digits of n",                       lambda d: d,
                 "digits", "digits"),
    "even_pos": ("digits of n at even positions counting from the right (last digit is position 0)",
                 lambda d: [x for i, x in enumerate(reversed(d)) if i % 2 == 0][::-1],
                 "[x for i, x in enumerate(reversed(digits)) if i % 2 == 0]", "[x for i, x in enumerate(digits[::-1]) if i % 2 == 0]"),
    "odd_pos":  ("digits of n at odd positions counting from the right (last digit is position 0)",
                 lambda d: [x for i, x in enumerate(reversed(d)) if i % 2 == 1][::-1],
                 "[x for i, x in enumerate(reversed(digits)) if i % 2 == 1]", "[x for i, x in enumerate(digits[::-1]) if i % 2 == 1]"),
    "even_val": ("even digits of n",                      lambda d: [x for x in d if x % 2 == 0],
                 "[x for x in digits if x % 2 == 0]", "list(filter(lambda x: x % 2 == 0, digits))"),
    "odd_val":  ("odd digits of n",                       lambda d: [x for x in d if x % 2 == 1],
                 "[x for x in digits if x % 2 == 1]", "list(filter(lambda x: x % 2 == 1, digits))"),
    "distinct": ("distinct digits of n",                  lambda d: sorted(set(d)),
                 "sorted(set(digits))", "sorted(set(digits))"),
    "once":     ("digits that occur exactly once in n",   lambda d: [x for x in d if d.count(x) == 1],
                 "[x for x in digits if digits.count(x) == 1]", "[x for x in digits if digits.count(x) == 1]"),
    "gt4":      ("digits of n greater than 4",            lambda d: [x for x in d if x > 4],
                 "[x for x in digits if x > 4]", "list(filter(lambda x: x > 4, digits))"),
    "nonzero":  ("nonzero digits of n",                   lambda d: [x for x in d if x != 0],
                 "[x for x in digits if x != 0]", "list(filter(None, digits))"),
}
MAP = {
    "id":     ("",                              lambda x: x,          "x"),
    "square": ("the squares of",                lambda x: x * x,      "x * x"),
    "succ":   ("one plus each of",              lambda x: x + 1,      "x + 1"),
    "compl":  ("nine minus each of",            lambda x: 9 - x,      "9 - x"),
    "mod3":   ("each of ... taken modulo 3",    lambda x: x % 3,      "x % 3"),
}
REDUCE = {
    "sum":     ("the sum of",                        lambda v: sum(v),                          "sum(v)"),
    "prod":    ("the product of (1 if there are none)", lambda v: math.prod(v),                 "math.prod(v)"),
    "count":   ("the number of",                     lambda v: len(v),                          "len(v)"),
    "max":     ("the largest of (0 if there are none)", lambda v: max(v) if v else 0,           "max(v) if v else 0"),
    "range":   ("the largest minus the smallest of (0 if there are none)",
                lambda v: (max(v) - min(v)) if v else 0,                                        "(max(v) - min(v)) if v else 0"),
    "altsum":  ("the absolute value of the alternating sum (first plus, second minus, ...) of",
                lambda v: abs(sum(x if i % 2 == 0 else -x for i, x in enumerate(v))),
                "abs(sum(x if i % 2 == 0 else -x for i, x in enumerate(v)))"),
    "ndistinct": ("the number of distinct values among",  lambda v: len(set(v)),                "len(set(v))"),
}
DEFAULT = {"select": "all", "map": "id", "reduce": "sum"}


def ref(sel, mp, rd):
    def f(n):
        d = [int(c) for c in str(abs(n))]
        return REDUCE[rd][1]([MAP[mp][1](x) for x in SELECT[sel][1](d)])
    return f


def describe(sel, mp, rd):
    sel_txt = SELECT[sel][0]
    the = "" if sel == "all" else "the "                       # "all digits of n", never "the all digits"
    if mp == "mod3":
        obj = f"{the}{sel_txt}, each taken modulo 3"
    elif mp == "id":
        obj = f"{the}{sel_txt}"
    else:
        obj = f"{MAP[mp][0]} {the}{sel_txt}"
    return f"computes {REDUCE[rd][0]} {obj}"


def impl_loop(name, sel, mp, rd):
    return f"""def {name}(n):
    import math
    digits = [int(c) for c in str(abs(n))]
    selected = {SELECT[sel][2]}
    v = []
    for x in selected:
        v.append({MAP[mp][2]})
    return {REDUCE[rd][2]}"""


def impl_func(name, sel, mp, rd):
    return f"""def {name}(n):
    import math
    digits = list(map(int, str(abs(n))))
    v = [{MAP[mp][2]} for x in {SELECT[sel][3]}]
    return {REDUCE[rd][2]}"""


# substrings a random name must not contain (slurs, hate symbols, real groups, profanity) -- 2026-09-23 after the
# seed-0 set produced "nazifun", "taluban" and "fukith"
BAD_SUBSTRINGS = ("nazi", "talib", "talu", "taleb", "fuk", "fuc", "shit", "kkk", "isis", "rape", "cunt", "dick", "nig", "fag", "slut",
                  "porn", "hitl", "kill", "dead", "sex", "anal", "ass", "tit", "cock", "dam", "hell", "jew", "arab",
                  "gay", "homo", "retar", "spic", "chink", "gook", "kike", "wop", "paki", "coon", "pedo", "nazis")


def nonce_name(rng):
    C, V = "bdfgklmnprstvz", "aeiou"
    while True:
        w = "".join(rng.choice(C) + rng.choice(V) for _ in range(rng.choice([2, 3]))) + rng.choice(["x", "l", "n", "th", "sk"])
        if 5 <= len(w) <= 8 and not any(b in w for b in BAD_SUBSTRINGS):
            return w


# ---- competitors the battery must separate every nonce from ---------------------------------------------------
BANK = {"digit_sum": lambda n: sum(int(c) for c in str(n)), "digit_prod": lambda n: math.prod(int(c) for c in str(n)),
        "even_digits": lambda n: sum(1 for c in str(n) if int(c) % 2 == 0),
        "sq_digit_sum": lambda n: sum(int(c) ** 2 for c in str(n))}
COMPETITORS = {**{a: ALGOS[a]["ref"] for a in ALGOS}, **BANK}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "nonce_functions.json"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    BATTERY = [7, 777, 1002, 112358, 90, 2468, 100, 9999, 305, 4441]      # algos' battery + two extra separators
    S.BATTERY = BATTERY


    def outs(f):
        return [f(n) for n in BATTERY]


    def simplifications(sel, mp, rd):
        for comp, default in DEFAULT.items():
            t = {"select": sel, "map": mp, "reduce": rd}
            if t[comp] != default:
                t[comp] = default
                yield (t["select"], t["map"], t["reduce"])


    triples = [t for t in itertools.product(SELECT, MAP, REDUCE) if t != ("all", "id", "sum")]
    rng.shuffle(triples)
    comp_outs = {k: outs(f) for k, f in COMPETITORS.items()}
    accepted, used_names = [], set(COMPETITORS)
    for sel, mp, rd in triples:
        f = ref(sel, mp, rd); o = outs(f)
        if len(set(o)) < 5 or o.count(0) > 5:                    # too constant to grade
            continue
        if any(o == c for c in comp_outs.values()):                # equals a known function on the battery
            continue
        if any(o == n["want"] for n in accepted):                  # equals an accepted nonce
            continue
        if any(o == outs(ref(*s)) for s in simplifications(sel, mp, rd)):   # a one-step simplification passes
            continue
        if all(v == 0 for v in o[:8]):
            continue
        name = nonce_name(rng)
        while name in used_names:
            name = nonce_name(rng)
        used_names.add(name)
        impls = [impl_loop(name, sel, mp, rd), impl_func(name, sel, mp, rd)]
        ok = True
        for code in impls:                                         # execution-verify both renderings
            code_solve = code.replace(f"def {name}(", "def solve(")
            st, got = None, None
            for form in S.FORMS:
                st, got = S.run_python(code_solve, "solve", form)
                if st == "ok" and got:
                    break
            if not (st == "ok" and got == o):
                print(f"[reject] {name} {sel}/{mp}/{rd}: rendering failed verification ({st}, {got} vs {o})")
                ok = False
        if not ok:
            continue
        accepted.append({"name": name, "select": sel, "map": mp, "reduce": rd, "desc": describe(sel, mp, rd),
                         "impls": impls, "want": o})
        if len(accepted) >= a.n:
            break

    # minimum separation among accepted + competitors on the battery
    allo = {**comp_outs, **{n["name"]: n["want"] for n in accepted}}
    mind = min(sum(x != y for x, y in zip(allo[p], allo[q])) for p, q in itertools.combinations(allo, 2))
    Path(a.out).write_text(json.dumps({"battery": BATTERY, "functions": accepted}, indent=1))
    print(f"[nonce] {len(accepted)} functions accepted of {len(triples)} triples; battery {BATTERY}; "
          f"min pairwise battery disagreements (nonces + competitors) = {mind}")
    for n in accepted:
        print(f"  {n['name']:<9} {n['select']:>8}/{n['map']:<6}/{n['reduce']:<9} {n['desc']}")
        print(f"            want {n['want']}")

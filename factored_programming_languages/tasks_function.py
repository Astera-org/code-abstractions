"""The function x language factorial for the counterfactual transplant experiment.

Four integer->integer digit functions with the SAME signature and mutually distinct
outputs on a fixed battery, so "which function did it implement" is decided by running
the code, not by reading it. The function is always named `solve`, so nothing here can
be carried by a lexical function-name representation.

    battery      7     1234   5      90     406    2468
    digit_sum    7     10     5      9      10     20
    even_digits  0     2      0      1      3      4
    digit_prod   7     24     5      0      0      384
    sq_digit_sum 49    30     25     81     52     120

Every pair of rows differs on at least four of the six inputs.
"""
import math

from languages import LANGS, DISPLAY                                # noqa: F401

BATTERY = [7, 1234, 5, 90, 406, 2468]

FUNCS = [
    ("digit_sum", "returns the sum of the digits of a non-negative integer",
     lambda n: sum(int(c) for c in str(n))),
    ("even_digits", "returns how many of the digits of a non-negative integer are even",
     lambda n: sum(1 for c in str(n) if int(c) % 2 == 0)),
    ("digit_prod", "returns the product of the digits of a non-negative integer",
     lambda n: math.prod(int(c) for c in str(n))),
    ("sq_digit_sum", "returns the sum of the squares of the digits of a non-negative integer",
     lambda n: sum(int(c) ** 2 for c in str(n))),
]
FKEYS = [f[0] for f in FUNCS]
FDESC = {k: d for k, d, _ in FUNCS}
FREF = {k: r for k, _, r in FUNCS}
WANT = {k: [FREF[k](v) for v in BATTERY] for k in FKEYS}

# The shared tail: identical across every prompt, so the aligned suffix of the
# padded set (the patch site) stays long. FPL_TAIL overrides it for a whole
# process -- used by the long-tail variant of the TCI battery, which doubles
# the suffix with two content-free sentences (results/*9L_*).
import os as _os
TAIL = _os.environ.get("FPL_TAIL") or (
        "Name the function solve. Return only the function, with no explanation, "
        "no examples, and no tests.")


def prompt(lang, fkey):
    return f"Write a function in {DISPLAY[lang]} that {FDESC[fkey]}. {TAIL}"


def which_function(outputs):
    """the function key whose reference the executed outputs reproduce, or None"""
    if not outputs:
        return None
    for k in FKEYS:
        if outputs == WANT[k]:
            return k
    return None

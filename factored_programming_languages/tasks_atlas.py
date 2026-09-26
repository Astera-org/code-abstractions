"""Function set for the atlas experiments: the 31 bench tasks plus the
algebra-grid extensions, each with a description and an executable reference.

BIGBAT is the common input battery for BEHAVIOR vectors (reference outputs
only -- no model involvement): behavioural similarity between functions is
computed from these vectors and compared against factor-space geometry.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks_bench as TB                                                   # noqa: E402

_ds = lambda n: [int(c) for c in str(n)]


def _pers(n, op):
    c = 0
    while n >= 10:
        n = op(_ds(n))
        c += 1
    return c


def _happy(n):
    c = 0
    while n not in (1, 89):
        n = sum(d * d for d in _ds(n))
        c += 1
    return c


def _alt(n, p):
    s, sign = 0, 1
    for d in _ds(n):
        s += sign * d ** p
        sign = -sign
    return s


EXTRA = {
    "sq_prod": ("returns the product of the squares of the digits of a "
                "non-negative integer",
                lambda n: math.prod(d * d for d in _ds(n))),
    "max_digit": ("returns the largest digit of a non-negative integer",
                  lambda n: max(_ds(n))),
    "sq_max": ("returns the largest of the squares of the digits of a "
               "non-negative integer", lambda n: max(d * d for d in _ds(n))),
    "cube_sum": ("returns the sum of the cubes of the digits of a "
                 "non-negative integer", lambda n: sum(d ** 3 for d in _ds(n))),
    "cube_prod": ("returns the product of the cubes of the digits of a "
                  "non-negative integer",
                  lambda n: math.prod(d ** 3 for d in _ds(n))),
    "cube_max": ("returns the largest of the cubes of the digits of a "
                 "non-negative integer", lambda n: max(d ** 3 for d in _ds(n))),
    "alt_sq_sum": ("returns the alternating sum of the squares of the digits "
                   "of a positive integer, starting positive from the most "
                   "significant digit", lambda n: _alt(n, 2)),
    "alt_cube_sum": ("returns the alternating sum of the cubes of the digits "
                     "of a positive integer, starting positive from the most "
                     "significant digit", lambda n: _alt(n, 3)),
    "add_pers": ("returns how many times the digits of a non-negative integer "
                 "must be summed repeatedly until a single digit remains",
                 lambda n: _pers(n, sum)),
    "mult_pers": ("returns how many times the digits of a positive integer "
                  "must be multiplied together repeatedly until a single "
                  "digit remains", lambda n: _pers(n, math.prod)),
    "happy89": ("returns how many times the sum-of-squared-digits map must be "
                "applied to a positive integer before reaching 1 or 89",
                _happy),
}

DESC = {n: d for n, d, _, _ in TB.BENCH}
REF = {n: r for n, _, r, _ in TB.BENCH}
for k, (d, r) in EXTRA.items():
    assert k not in DESC, k
    DESC[k] = d
    REF[k] = r

# behaviour battery: reference outputs only, CPU
BIGBAT = [2, 5, 7, 19, 29, 44, 90, 163, 305, 406, 1234, 2468]


def behavior_vector(name):
    return [REF[name](x) for x in BIGBAT]


FKEYS = sorted(DESC)
assert len(FKEYS) == len(TB.BENCH) + len(EXTRA)

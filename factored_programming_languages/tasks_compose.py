"""Prompt variants for the causal composition experiment.

The composition test steers an UNDERSPECIFIED carrier prompt with factor deltas
estimated from partially-specified prompts:

    full(A, lam)   "Write a function in {lam} that {desc_A}. {TAIL}"     the 4x10 grid
    nofn(lam)      "Write a function in {lam}. {TAIL}"                   function slot empty
    nolang(A)      "Write a function that {desc_A}. {TAIL}"              language slot empty
    carrier        "Write a function. {TAIL}"                            both empty

Factor deltas are leave-one-out so the composed cell (A, lam) is genuinely held out:

    f_A   = E_{lam' != lam}[ M(full(A, lam')) - M(nofn(lam')) ]
    l_lam = E_{A'  != A  }[ M(full(A', lam)) - M(nolang(A')) ]

and the intervention on the carrier is  M += a_f * f_A + a_l * l_lam.  Success is the
model writing function A (scored by execution against the battery) in language lam
(scored by building under lam's toolchain), neither of which the carrier names.

All prompts share TAIL verbatim, so the aligned suffix of the padded set stays long.
"""
from tasks_function import FDESC, TAIL                                   # noqa: F401
from languages import DISPLAY


def full(lang, fkey):
    return f"Write a function in {DISPLAY[lang]} that {FDESC[fkey]}. {TAIL}"


def nofn(lang):
    return f"Write a function in {DISPLAY[lang]}. {TAIL}"


def nolang(fkey):
    return f"Write a function that {FDESC[fkey]}. {TAIL}"


CARRIER = f"Write a function. {TAIL}"

"""The canonical ten programming languages, in one place.

Fixed for the paper: csharp is out, cpp and ruby are in.
Every experiment in this directory uses exactly this set, so an N/10 is always the same
denominator. Execution toolchains exist for all ten (see execute.py).

EXTENSION SWITCH (a side experiment not used in the paper). Setting
FPL_EXTRA_LANGS="fsharp,clojure,scala,elixir" in the environment appends those
languages to every table below for the whole process. A process that does not
set it sees exactly the ten, so no existing run, figure or control changes.
"""
import os as _os

LANGS = ["python", "javascript", "c", "cpp", "java", "go", "rust", "ruby", "ocaml",
         "haskell"]

# how the language is named inside a prompt
DISPLAY = {"python": "Python", "javascript": "JavaScript", "c": "C", "cpp": "C++",
           "java": "Java", "go": "Go", "rust": "Rust", "ruby": "Ruby",
           "ocaml": "OCaml", "haskell": "Haskell"}

# the noun phrase for a sequence argument, in each language's idiom
COLL = {"python": "a list of integers", "javascript": "an array of integers",
        "c": "an integer array", "cpp": "a vector of integers",
        "java": "an integer array", "go": "a slice of integers",
        "rust": "a slice of integers", "ruby": "an array of integers",
        "ocaml": "a list of integers", "haskell": "a list of integers"}

# markdown fence tag, for the V3 elicitation rung and for extraction
FENCE = {"python": "python", "javascript": "javascript", "c": "c", "cpp": "cpp",
         "java": "java", "go": "go", "rust": "rust", "ruby": "ruby",
         "ocaml": "ocaml", "haskell": "haskell"}

# paradigm family, used only for reporting -- never for gating an experiment
FAMILY = {"python": "script", "javascript": "script", "ruby": "script",
          "c": "imperative", "cpp": "imperative", "java": "imperative",
          "go": "imperative", "rust": "imperative",
          "ocaml": "functional", "haskell": "functional"}

# ---- the four extension languages ------------------------------------------
_EXTRA = {
    "fsharp":  ("F#", "a list of integers", "fsharp", "functional"),
    "clojure": ("Clojure", "a vector of integers", "clojure", "functional"),
    "scala":   ("Scala", "a list of integers", "scala", "functional"),
    "elixir":  ("Elixir", "a list of integers", "elixir", "functional"),
}
EXTRA_LANGS = [l for l in _os.environ.get("FPL_EXTRA_LANGS", "").replace(",", " ").split()
               if l]
assert all(l in _EXTRA for l in EXTRA_LANGS), f"unknown extra language in {EXTRA_LANGS}"
for _l in EXTRA_LANGS:
    LANGS.append(_l)
    DISPLAY[_l], COLL[_l], FENCE[_l], FAMILY[_l] = _EXTRA[_l]

# two-way paradigm split for the donor-pool arms of the extension (all /
# within-family / cross-family). Rust and Scala are the two ambiguous
# assignments; the runners take --families to override this table, and the
# assignment used is written into every result file.
PARADIGM = {"python": "imperative", "javascript": "imperative", "c": "imperative",
            "cpp": "imperative", "java": "imperative", "go": "imperative",
            "rust": "imperative", "ruby": "imperative",
            "ocaml": "functional", "haskell": "functional",
            "fsharp": "functional", "clojure": "functional",
            "scala": "functional", "elixir": "functional"}


def parse_families(spec):
    """'imperative:c,cpp,...;functional:haskell,...' -> {lang: family};
    None/'' -> PARADIGM restricted to LANGS"""
    if not spec:
        return {l: PARADIGM[l] for l in LANGS}
    out = {}
    for part in spec.split(";"):
        fam, _, ls = part.partition(":")
        for l in ls.replace(",", " ").split():
            out[l] = fam.strip()
    missing = [l for l in LANGS if l not in out]
    assert not missing, f"--families leaves {missing} unassigned"
    return out


def donor_pool(lam, pool, fams, langs=None, always=()):
    """the languages a pooled factor for recipient `lam` is averaged over.
    all: every other language (the ten/nine-language design); family: the
    other languages of lam's own family; xfamily: the other family. `always`
    (e.g. python) joins the pool whatever its family; lam itself never does."""
    langs = list(LANGS if langs is None else langs)
    if pool == "all":
        keep = [l for l in langs if l != lam]
    elif pool == "family":
        keep = [l for l in langs if l != lam and fams[l] == fams[lam]]
    elif pool == "xfamily":
        keep = [l for l in langs if l != lam and fams[l] != fams[lam]]
    else:
        raise ValueError(pool)
    for l in always:
        if l in langs and l != lam and l not in keep:
            keep.append(l)
    return keep


assert set(DISPLAY) == set(COLL) == set(FENCE) == set(FAMILY) == set(LANGS)
assert len(LANGS) == 10 + len(EXTRA_LANGS)

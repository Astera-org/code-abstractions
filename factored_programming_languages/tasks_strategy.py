"""Prompts and a structural checker for the recursion-vs-iteration factor
(the strategy factor of Fig. 3, 5, 6).

Prompt shape mirrors tasks_compose.full() so the padded set, the aligned
suffix and every existing harvest/patch helper keep working:

    strat(lang, fkey, s, cue)  "Write a function in {lang} that {desc}. {CUE}. {TAIL}"

CUE FAMILIES (section 6.1): three per strategy, and family "c" for each avoids
the literal words recursion/recursive/iterative/iteration/loop entirely, so the
pooled strategy delta cannot be an average of the recipient's own cue tokens.
The evaluation holds the recipient's cue family out of the donor estimator.

STRATEGY CHECKER (section 11.3): structural, not keyword-spotting on the prompt.
  recursive  the entry function (or a helper it defines) calls itself, and the
             core computation is not carried by a loop
  iterative  a loop / fold / library aggregate carries the computation and there
             is no self-call
For OCaml/Haskell, where "iterative" has no loop keyword in idiomatic code, the
accepted iterative constructions are declared here in advance (folds, library
aggregates, tail-recursive accumulator helpers are NOT counted as iterative --
they self-call, so they grade as recursive; this is stated rather than inferred).
"""
import re

from tasks_function import FDESC, TAIL
from languages import DISPLAY

STRATEGIES = ["rec", "iter"]

# cue families; "c" avoids the literal strategy words on both sides
CUES = {
    "rec": {
        "a": "Use recursion; do not use loops",
        "b": "Implement it recursively rather than with any loop construct",
        "c": "Have the function call itself on a smaller input until it bottoms out; "
             "do not use for, while, or any repetition construct",
    },
    "iter": {
        "a": "Use an iterative loop; do not use recursion",
        "b": "Implement it iteratively with a loop rather than recursively",
        "c": "Keep an accumulator and walk the input with a repetition construct; "
             "the function must never call itself",
    },
}
CUE_FAMILIES = ["a", "b", "c"]


def strat(lang, fkey, s, cue="a"):
    return (f"Write a function in {DISPLAY[lang]} that {FDESC[fkey]}. "
            f"{CUES[s][cue]}. {TAIL}")


def stratonly(s, cue="a"):
    """strategy slot ONLY -- no function description, no language. The
    empty-slot prompt the three-factor compose needs: its contrast against
    the bare carrier is the strategy factor with nothing else attached."""
    return f"Write a function. {CUES[s][cue]}. {TAIL}"


def nostrat(lang, fkey):
    """strategy slot empty -- the baseline for an absolute strategy factor"""
    return f"Write a function in {DISPLAY[lang]} that {FDESC[fkey]}. {TAIL}"


# ---------------------------------------------------------------- checker ---
# loop constructs that carry a computation, per language
LOOPS = {
    "python": [r"\bfor\b", r"\bwhile\b"],
    "javascript": [r"\bfor\b", r"\bwhile\b", r"\bdo\b\s*\{", r"\.reduce\(",
                   r"\.map\(", r"\.filter\(", r"\.forEach\(", r"\.split\("],
    "c": [r"\bfor\b", r"\bwhile\b", r"\bdo\b\s*\{"],
    "cpp": [r"\bfor\b", r"\bwhile\b", r"\bdo\b\s*\{"],
    "java": [r"\bfor\b", r"\bwhile\b", r"\bdo\b\s*\{"],
    "go": [r"\bfor\b"],
    "rust": [r"\bfor\b", r"\bwhile\b", r"\bloop\b", r"\.iter\(", r"\.map\(",
             r"\.fold\(", r"\.sum\(", r"\.product\(", r"\.filter\(",
             r"\.chars\(", r"\.count\(", r"\.rev\("],
    "ruby": [r"\bwhile\b", r"\buntil\b", r"\bfor\b", r"\.each\b", r"\.times\b",
             r"\.upto\b", r"\.step\b", r"\.inject\b", r"\.reduce\b", r"\.sum\b",
             r"\.map\b", r"\.count\b"],
    # declared in advance: idiomatic non-self-calling repetition in ML/Haskell
    "ocaml": [r"\bfor\b", r"\bwhile\b", r"List\.fold_left", r"List\.fold_right",
              r"List\.iter", r"List\.map", r"List\.filter", r"Array\.fold_left",
              r"Seq\.fold_left"],
    "haskell": [r"\bfoldl'?\b", r"\bfoldr\b", r"\bsum\b", r"\bproduct\b",
                r"\blength\b", r"\bmap\b", r"\bfilter\b", r"\bzipWith\b",
                r"\biterate\b", r"\bscanl\b", r"\breplicate\b", r"\bmapM_\b"],
    # declared in advance 2026-09-04 for the extension languages (experiments/
    # family_pooling_extension.md), on the same rule as ocaml/haskell: loops
    # where the language has them, else folds and library aggregates; a
    # self-calling accumulator helper is NOT iterative -- it self-calls.
    "fsharp": [r"\bfor\b", r"\bwhile\b",
               r"\b(?:List|Seq|Array|String)\.(?:fold|foldBack|reduce|iter|iteri|map|mapi|"
               r"filter|sum|sumBy|length|collect|choose|scan|unfold|countBy)\b"],
    # loop/recur is Clojure's loop; recur WITHOUT loop re-enters the fn (see
    # detect_strategy). doseq/dotimes/for are its other repetition forms.
    "clojure": [r"\(loop\b", r"\(doseq\b", r"\(dotimes\b", r"\(while\b", r"\(for\b",
                r"\(reduce\b", r"\(reductions\b", r"\(map\b", r"\(mapv\b",
                r"\(filter\b", r"\(filterv\b", r"\(keep\b", r"\(apply\s+[+*]",
                r"\(count\b", r"\(frequencies\b", r"\(iterate\b", r"\(transduce\b"],
    "scala": [r"\bfor\b", r"\bwhile\b", r"\.foldLeft\b", r"\.foldRight\b", r"\.fold\b",
              r"\.reduce\b", r"\.reduceLeft\b", r"\.map\b", r"\.filter\b", r"\.sum\b",
              r"\.product\b", r"\.count\b", r"\.foreach\b", r"\.scanLeft\b",
              r"\.flatMap\b", r"\.collect\b", r"\bIterator\.iterate\b"],
    # Elixir has no loop statement: every non-self-calling repetition is a
    # comprehension or an Enum/Stream/List aggregate
    "elixir": [r"\bfor\s", r"\bEnum\.\w+", r"\bStream\.\w+", r"\bList\.fold[lr]\b",
               r"\bInteger\.digits\b", r"\bString\.(?:graphemes|codepoints|to_charlist)\b"],
}
# comment syntax to strip before structural checks
COMMENTS = {
    "python": [r"#.*"], "ruby": [r"#.*"],
    "javascript": [r"//.*", r"/\*.*?\*/"], "c": [r"//.*", r"/\*.*?\*/"],
    "cpp": [r"//.*", r"/\*.*?\*/"], "java": [r"//.*", r"/\*.*?\*/"],
    "go": [r"//.*", r"/\*.*?\*/"], "rust": [r"//.*", r"/\*.*?\*/"],
    "ocaml": [r"\(\*.*?\*\)"], "haskell": [r"--.*", r"\{-.*?-\}"],
    "fsharp": [r"//.*", r"\(\*.*?\*\)"], "clojure": [r";.*"],
    "scala": [r"//.*", r"/\*.*?\*/"], "elixir": [r"#.*"],
}
# how a function definition is introduced, for finding locally defined helpers
DEFS = {
    "python": r"def\s+([A-Za-z_]\w*)\s*\(",
    "javascript": r"(?:function\s+([A-Za-z_$][\w$]*)\s*\(|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:function|\())",
    "c": r"^[A-Za-z_][\w \*]*\b([A-Za-z_]\w*)\s*\([^;]*\)\s*\{",
    "cpp": r"^[A-Za-z_][\w \*:<>]*\b([A-Za-z_]\w*)\s*\([^;]*\)\s*\{",
    "java": r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{",
    "go": r"func\s+([A-Za-z_]\w*)\s*\(",
    "rust": r"fn\s+([A-Za-z_]\w*)\s*(?:<[^>]*>)?\s*\(",
    "ruby": r"def\s+([A-Za-z_]\w*[!?]?)",
    # FUNCTION bindings only: "let f x =" / "let rec f x =" / "let f = fun".
    # Plain value bindings ("let sum = ref 0") are not definitions -- counting
    # them made every use of a local variable look like a recursive call, so
    # an ordinary while-loop scored as recursion.
    "ocaml": r"let\s+(?:rec\s+)?([A-Za-z_]\w*)\s+(?=[A-Za-z_(])"
             r"|let\s+(?:rec\s+)?([A-Za-z_]\w*)\s*=\s*fun(?:ction)?\b",
    # top-level AND indented (where/let) bindings, but only FUNCTION ones:
    # a name followed by at least one argument before the "=". "acc = ..." is
    # a value binding, and a where-bound helper is where Haskell puts its
    # recursion, so missing it made tail-recursive answers read as unknown.
    "haskell": r"^\s*([a-z_]\w*)\s*::"
               r"|^\s*([a-z_]\w*)\s+[^=\s|][^=\n]*=(?!=)",
    # extension languages (2026-09-04). fsharp as ocaml, FUNCTION bindings
    # only; clojure `(defn f` / `(def f (fn`; scala `def f`; elixir `def`/`defp`
    # (multi-clause definitions are consecutive same-name headers, which
    # _bodies already merges into one definition).
    "fsharp": r"let\s+(?:rec\s+)?(?:inline\s+)?(?:private\s+)?([A-Za-z_]\w*)\s+(?=[A-Za-z_(])"
              r"|let\s+(?:rec\s+)?([A-Za-z_]\w*)\s*=\s*fun\b",
    "clojure": r"\(defn-?\s+([^\s\[\]()\"]+)|\(def\s+([^\s\[\]()\"]+)\s+\(fn\b",
    "scala": r"def\s+([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*[(:=]",
    "elixir": r"defp?\s+([a-z_]\w*[?!]?)",
}


# words a definition pattern can match but which never name a function
KEYWORDS = {
    "haskell": {"let", "where", "in", "do", "if", "then", "else", "case",
                "of", "module", "import", "data", "type", "newtype",
                "instance", "class", "main"},
    "ocaml": {"let", "in", "rec", "and", "if", "then", "else", "match",
              "with", "fun", "function"},
    "fsharp": {"let", "in", "rec", "and", "if", "then", "else", "match",
               "with", "fun", "function", "mutable", "inline", "private"},
    "clojure": {"if", "let", "loop", "recur", "fn", "do", "when", "cond", "case"},
    "scala": {"if", "else", "for", "while", "match", "case", "def", "val", "var",
              "return", "yield", "new"},
    "elixir": {"do", "end", "fn", "when", "if", "else", "case", "cond", "with", "for",
               "def", "defp", "defmodule"},
}


def _name_re(lang, name):
    """`name` as a whole identifier. Clojure identifiers may contain - ? ! * <
    > = +, where \\b does not mark the boundary, so the boundary is 'not
    another symbol character' there."""
    if lang == "clojure":
        return rf"(?<![\w*+!?<>=/.-]){re.escape(name)}(?![\w*+!?<>=/.-])"
    return rf"\b{re.escape(name)}\b"


def _strip_comments(code, lang):
    out = code
    for pat in COMMENTS.get(lang, []):
        out = re.sub(pat, " ", out, flags=re.S | re.M)
    return out


def defined_names(code, lang):
    """every function name defined in this snippet (entry + helpers)"""
    pat = DEFS.get(lang)
    if not pat:
        return []
    names = []
    for m in re.finditer(pat, code, flags=re.M):
        for g in m.groups():
            if g:
                names.append(g)
    return names


def _def_occurrences(src, lang, name):
    """how many times `name` appears as part of a DEFINITION header (not a call)"""
    pat = DEFS.get(lang)
    if not pat:
        return 0
    n = 0
    for m in re.finditer(pat, src, flags=re.M):
        if name in [g for g in m.groups() if g]:
            n += 1
    return n


def _bodies(src, lang):
    """-> [(name, body)] where body runs from a definition header to the
    start of the next one. Definitions are sequential in every language here,
    so this is the extent a self-call must occur inside."""
    pat = DEFS.get(lang)
    if not pat:
        return []
    marks = []
    for m in re.finditer(pat, src, flags=re.M):
        name = next((g for g in m.groups() if g), None)
        if name and name not in KEYWORDS.get(lang, ()):
            marks.append((m.start(), m.end(), name))
    # consecutive headers for the SAME name are one definition: a Haskell type
    # signature followed by its equation, or a multi-clause definition. The
    # body starts after the last of them, else the equation would look like a
    # call of the name its own signature just declared.
    runs = []
    for s0, e0, name in marks:
        if runs and runs[-1][2] == name:
            runs[-1] = (runs[-1][0], e0, name)
        else:
            runs.append((s0, e0, name))
    out = []
    for i, (s0, e0, name) in enumerate(runs):
        end = runs[i + 1][0] if i + 1 < len(runs) else len(src)
        out.append((name, src[e0:end]))
    return out


def detect_strategy(code, lang):
    """-> 'rec' | 'iter' | 'mixed' | 'unknown'  (structural, comments stripped)

    rec   a defined function's own name occurs inside its OWN body -- i.e. it
          calls itself. Scoping this to the body matters: a demo main() that
          calls solve() is not solve recursing, and counting it as one made
          every C/C++ answer carrying a driver score as 'mixed'.
    iter  a loop or a declared iterative construction is present, no self-call
    mixed both signals (e.g. a loop inside a recursive helper)
    """
    if not code or not code.strip():
        return "unknown"
    src = _strip_comments(code, lang)
    self_call = False
    for name, body in _bodies(src, lang):
        if len(name) < 2:                   # single letters: too collision-prone
            continue
        if re.search(_name_re(lang, name), body):
            self_call = True
            break
    # Clojure: `recur` re-enters the nearest enclosing `loop` -- a loop -- or,
    # with no loop present, the enclosing fn itself: self-recursion without
    # the name. Declared in advance with the LOOPS table above.
    if lang == "clojure" and re.search(r"\(recur\b", src) and not re.search(r"\(loop\b", src):
        self_call = True
    has_loop = any(re.search(p, src) for p in LOOPS.get(lang, []))
    if self_call and has_loop:
        return "mixed"
    if self_call:
        return "rec"
    if has_loop:
        return "iter"
    return "unknown"

"""How a generation is ORGANISED, one level finer than iterative-vs-recursive.

`detect_strategy` separates iteration from recursion, which is the label the strategy factor is
scored against. It cannot see the difference the transport-structure argument turns on: a Java
`while` loop and a Java stream reduction are both iterative, but only one of them looks like the
Haskell fold that produced it.

Five labels, conservative -- anything that does not match cleanly is `unknown` rather than guessed,
and the unknown rate is reported wherever this is used:

    pipeline    the digits become a sequence that is mapped/filtered/folded: comprehensions,
                streams, iterator chains, sum/foldl/reduce/inject over a collection
    helper_acc  a NAMED inner helper carrying an accumulator argument (`go acc`, `aux acc`,
                `loop acc`) -- the fold written out by hand
    recursion   the entry function calls itself, with no accumulator helper
    loop_acc    a while/for loop updating a mutable accumulator
    unknown     none of the above matched

Precedence is helper_acc > pipeline > recursion > loop_acc: a hand-written fold is a fold even
though its helper recurses, and a comprehension inside a loop body is still a pipeline.
"""
import re

_PIPE = (
    r"\.stream\(\)|IntStream|\.mapToInt\(|\.reduce\(|\.collect\(",          # java
    r"\.map\(|\.filter\(|\.reduce\(|\.fold\(|\.sum\(\)|\.iter\(\)",         # rust / js
    r"\[[^\]]*\bfor\b[^\]]*\]|\bsum\s*\(|\bmap\s*\(|\bfilter\s*\(",        # python comprehension
    r"\bsum\s*\[|\bfoldr\b|\bfoldl'?\b|\bmap\b\s|\bfilter\b\s|\.\s*sum\b",  # haskell
    r"\[[^\]]*\|[^\]]*<-",                                              # haskell/list comprehension
    r"List\.fold_left|List\.map|List\.filter|\bfold_left\b",                # ocaml
    r"\.inject\(|\.each_char|\.chars\b|\.digits\b|\.sum\b|\.reduce\b|\.count\(",  # ruby
    r"std::accumulate|std::transform|ranges::",                             # c++
)
_HELPER = (
    r"\b(?:go|aux|loop|helper|iter|rec)\b\s+\w*\s*(?:acc|a|s|sum|total)\b",
    r"\bfn\s+(?:go|aux|loop|helper)\s*\(",
    r"\bdef\s+(?:go|aux|loop|helper)\s*\(",
    r"\blet\s+rec\s+(?:go|aux|loop|helper)\b",
    r"\b(?:static\s+)?\w+\s+(?:go|aux|loop|helper)\s*\([^)]*\bacc\b",
)
_LOOP = r"\bwhile\b|\bfor\b|\bloop\b\s*[\{]|\bloop\s+do\b|\.each\b|\bforeach\b"


def _self_recursive(code, entry="solve"):
    # Drop DECLARATIONS only -- Haskell `solve :: Int -> Int`, OCaml `val solve : ...`. The older
    # rule dropped any line with `solve` and a colon, which also ate typed definition lines
    # (`fn solve(n: u64)`, `def solve(n: int) -> int:`) and left those recursions labelled unknown.
    body = re.sub(rf"^\s*(?:val\s+)?{entry}\s*::?[^\n]*$", "", code, flags=re.M)
    calls = len(re.findall(rf"\b{entry}\s*[\(\s]", body))
    return calls >= 2                     # the definition plus at least one call to itself


# A function definition whose NAME we can trust: an explicit definition keyword, or a binding to a
# lambda/arrow. Java/C `TYPE NAME(` is deliberately excluded -- it matches casts and declarations.
# (pattern, arity-style). Parameter lists are comma-separated in the c-family and the scripting
# languages, but whitespace-separated in ocaml -- counting commas there reports arity 1 for
# `let rec go n acc =` and turns a hand-written fold into plain recursion.
_DEFN = (
    (r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)", "comma"),               # js
    (r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:function\s*)?\(([^)]*)\)\s*=>",
     "comma"),                                                                   # js arrow
    (r"\bdef\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", "comma"),                        # python / ruby
    (r"\bfn\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", "comma"),                         # rust
    (r"\bfunc\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", "comma"),                       # go
    # ocaml. `rec` is optional because the models sometimes omit it on a self-calling let; the
    # empty-argument guard below keeps plain bindings (`let digit = x mod 10 in`) out.
    (r"\blet\s+(?:rec\s+)?([A-Za-z_][\w\x27]*)\s+([^=\n]*?)=(?!=)", "space"),   # ocaml
    # c / c++ helper with no access modifier: `int solve_recursive(int n, int acc) {` at the
    # start of a line. The leading token must look like a type, never a control keyword.
    (r"^[ \t]*(?!return\b|else\b|if\b|while\b|for\b|switch\b|case\b)"
     r"[A-Za-z_]\w*[\w \t\*&<>:]*?\b([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{", "comma"),
    # java / c++ helpers: anchored on an access modifier AND a body brace, so casts, declarations
    # and plain calls cannot match
    (r"\b(?:public|private|protected|static)[\w\s<>\[\]]*?\b([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{",
     "comma"),
)


_EQ = r"^[ \t]+([A-Za-z_][\w\x27]*)[ \t]+([^=\n]*?)=(?!=)"          # indented `name args = ...`
_EQ_SKIP = {"let", "in", "where", "then", "else", "if", "case", "of", "do", "return"}


_EQ_LANGS = {"haskell", "ocaml"}


def _worker_equations(code, entry="solve", lang=None):
    """name and arity of a self-recursive WHERE/LET-bound equation.

    ML-family only. `name args = ...` is also the shape of a c declaration -- `int digit = n % 10;`
    parses as the function `int` taking `digit` -- which relabelled 743 c-family loops as recursion
    before this guard."""
    if lang not in _EQ_LANGS:
        return None, 0
    for m in re.finditer(_EQ, code, re.M):
        nm, args = m.group(1), m.group(2)
        if nm == entry or nm in _EQ_SKIP or not args.strip():
            continue
        if len(re.findall(rf"(?<![\w\x27]){re.escape(nm)}(?![\w\x27])", code)) < 3:
            continue                   # needs >=2 defining equations or a definition plus a call
        return nm, len(args.split())
    return None, 0


def _recursive_worker(code, entry="solve", lang=None):
    """the name and arity of a SELF-RECURSIVE function, entry or inner.

    `_self_recursive` only ever asks whether the entry calls itself, so a computation delegated to
    a named inner function -- a JS `const countEvens = (num) => ... countEvens(...)` -- matched no
    pattern at all and came out `unknown`."""
    for pat, style in _DEFN:
        for m in re.finditer(pat, code, re.M):     # ^ in the c-family pattern needs MULTILINE
            nm, params = m.group(1), m.group(2)
            if not params.strip():                 # `let digit = ...` is a binding, not a function
                continue
            n_paren = len(re.findall(rf"\b{re.escape(nm)}\s*\(", code))
            n_bare = len(re.findall(rf"(?<![\w\x27]){re.escape(nm)}(?![\w\x27])", code))
            if n_paren < 2 and n_bare < 3:
                continue                       # defined but never called -> not recursive
            arity = (len([x for x in params.split(",") if x.strip()]) if style == "comma"
                     else len(params.split()))
            return nm, arity
    return _worker_equations(code, entry, lang)


def structure(code, lang, entry="solve"):
    if not code or not code.strip():
        return "unknown"
    c = code
    if any(re.search(p, c) for p in _HELPER):
        return "helper_acc"
    if any(re.search(p, c) for p in _PIPE):
        return "pipeline"
    if _self_recursive(c, entry):
        return "recursion"
    nm, arity = _recursive_worker(c, entry, lang)
    if nm is not None:
        # an inner worker carrying an extra argument IS the hand-written fold; one argument is
        # plain recursion that happens to live in a helper
        return "helper_acc" if nm != entry and arity >= 2 else "recursion"
    if re.search(_LOOP, c):
        return "loop_acc"
    return "unknown"

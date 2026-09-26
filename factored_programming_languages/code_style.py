"""Reading generated code: which function is it, is it recursive, is it degenerate.

Extracted from the archive's run_style_axis.py / run_style_paradigm.py so this directory
imports nothing from another experiment dir (both of those pull in torch and transformers
at module scope, which a pure text classifier has no business doing).

Extended to the canonical ten: cpp and ruby rules added, csharp dropped.
"""
import re

# ---- from run_style_paradigm.py ----
def _hs_cond(cond, c):
    # translate the two TASKS conditions to Haskell
    return cond.replace("@c@ % 2 == 0", "even @c@").replace("@c@", c)

def _ml_cond(cond, c):
    # translate the two TASKS conditions to OCaml (mod operator, single = for int eq)
    return cond.replace("@c@ % 2 == 0", "@c@ mod 2 = 0").replace("@c@", c)

_PIPE_RE = {
    "java":    re.compile(r"\.stream\(|\.map\(|\.filter\(|\.reduce\(|\.sum\(|\.collect\("),
    "csharp":  re.compile(r"\.Where\(|\.Select\(|\.Sum\(|\.Count\(|\.Aggregate\(|\.Average\("),
    "python":  re.compile(r"\[[^\]\n]*\bfor\b[^\]\n]*\bin\b[^\]\n]*\]|\bsum\([^\n]*\bfor\b|\bmap\(|\bfilter\("),
    # functional pipeline incl. point-free composition: "sum . map (^2)", "length . filter even"
    "haskell": re.compile(r"\bfold[lr]?\b|\bmap\b|\bfilter\b|\bsum\b\s*\.|\.\s*(map|filter)\b|\[[^\]\n]*\|[^\]\n]*<-"),
    "ocaml":   re.compile(r"List\.(fold_left|fold_right|map|filter)|\bfold_left\b"),
    # added 2026-08-14 for the confound-breaking languages (rust / javascript).
    # Iterator-adaptor chains are the fold pole in both; `.iter()`/`.into_iter()` is
    # the entry point of every rust chain and `.reduce(`/`.map(`/`.filter(` the js one.
    "rust":    re.compile(r"\.iter\(\)|\.into_iter\(\)|\.iter_mut\(\)|\.fold\(|"
                          r"\.map\(|\.filter\(|\.sum::|\.sum\(\)|\.count\(\)|"
                          r"\.collect\(|\.chunks\(|\.rev\(\)"),
    "javascript": re.compile(r"\.reduce\(|\.reduceRight\(|\.map\(|\.filter\(|"
                             r"\.flatMap\(|\.every\(|\.some\("),
    # added 2026-08-15 for the held-out languages. ruby's fold pole is the
    # Enumerable chain; cpp's is <numeric>/<algorithm>. C and Go have NO idiomatic
    # collection pipeline at all -- no map/filter/reduce in the standard library --
    # so their fold pole is genuinely empty and the pattern must never fire, or every
    # loop in them would be misread as a pipeline. `(?!)` never matches.
    "ruby":   re.compile(r"\.map\b|\.select\b|\.reject\b|\.reduce\b|\.inject\b|"
                         r"\.sum\b|\.filter\b|\.each_with_object\b|\.collect\b"),
    "cpp":    re.compile(r"std::accumulate|std::transform|std::for_each|"
                         r"std::reduce|std::count_if|ranges::"),
    "c":      re.compile(r"(?!)"),
    "go":     re.compile(r"(?!)"),
    # added 2026-09-04 for the extension languages (family_pooling_extension.md).
    # The fold pole is the collection pipeline of each: F#'s List/Seq/Array
    # module functions and |> chains, Clojure's reduce/map/filter/apply forms,
    # Scala's collection methods, Elixir's Enum/Stream modules and |> chains.
    "fsharp": re.compile(r"\b(?:List|Seq|Array|String)\.(?:fold|foldBack|reduce|map|mapi|"
                         r"filter|sum|sumBy|iter|length|collect|choose|scan)\b|\|>"),
    "clojure": re.compile(r"\((?:reduce|map|mapv|filter|filterv|apply|reductions|"
                          r"frequencies|keep|transduce|count)\b"),
    "scala":  re.compile(r"\.(?:foldLeft|foldRight|fold|reduce|reduceLeft|map|filter|sum|"
                         r"product|count|foreach|flatMap|scanLeft|collect)\b"),
    "elixir": re.compile(r"\bEnum\.\w+|\bStream\.\w+|\|>"),
}

def first_unit(text, lang):
    out = []
    for i, ln in enumerate(text.split("\n")):
        s = ln.strip()
        if out and s.startswith(("#", "//", "--", "(*")):
            break
        if out and ((lang == "python" and (s.startswith("def ") or (s and not ln[:1].isspace())))
                    or (lang in ("java", "csharp") and s.startswith(("public ", "private ", "static ")))
                    or (lang == "haskell" and re.match(r"^\w+\s*::", s))
                    or (lang == "ocaml" and i > 0 and re.match(r"^(let|;;)", s))
                    # a NEW top-level item (column 0) ends the first unit
                    or (lang == "rust" and not ln[:1].isspace()
                        and re.match(r"^(pub\s+)?fn\b", s))
                    or (lang == "javascript" and not ln[:1].isspace()
                        and re.match(r"^(export\s+)?(async\s+)?"
                                     r"(function\b|const\s|let\s|var\s|class\b)", s))
                    # 2026-08-15: a new top-level definition ends the unit
                    or (lang == "go" and not ln[:1].isspace()
                        and re.match(r"^(func|type|var|const)\b", s))
                    or (lang == "ruby" and not ln[:1].isspace()
                        and re.match(r"^(def|class|module)\b", s))
                    # 2026-09-04 extension languages: a new top-level definition
                    or (lang == "fsharp" and not ln[:1].isspace() and i > 0
                        and re.match(r"^(let|module|type|open)\b", s))
                    or (lang == "clojure" and not ln[:1].isspace() and i > 0
                        and re.match(r"^\((?:defn-?|def|ns)\b", s))
                    or (lang == "scala" and not ln[:1].isspace() and i > 0
                        and re.match(r"^(def|object|class|val|var|@main|import)\b", s))
                    or (lang == "elixir" and not ln[:1].isspace() and i > 0
                        and re.match(r"^(defmodule|def|defp)\b", s))
                    or (lang in ("c", "cpp") and not ln[:1].isspace()
                        and re.match(r"^[\w:][\w:<>,\s&*]*\b\w+\s*\(", s)
                        and not s.startswith(("return", "if", "for", "while",
                                              "else", "}")))):
            break
        out.append(ln)
        if len(out) >= 12:
            break
    return "\n".join(out)

def is_degenerate(text):
    from collections import Counter
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    # repeated CONTENT line >=3x (ignore structural brace/paren/punctuation-only lines like "}")
    content = [l for l in lines if l.strip(" \t{}()[];,")]
    if content and Counter(content).most_common(1)[0][1] >= 3:
        return True
    if sum(bool(re.match(r"^[\w.]+\s*(\(?\w*\)?\s*)?=\s*-?\d+\s*$", l)) for l in lines) >= 4:  # enumeration
        return True
    return False

# ---- from run_style_axis.py ----
def self_recursive_anywhere(code, lang):
    """Does ANY function defined in the emitted block call itself?

    Added 2026-08-14.  The elicitation prompt asks for one function, but gemma very
    often answers a recursion request with a PUBLIC WRAPPER that delegates to a
    PRIVATE RECURSIVE HELPER:

        public int sumOfSquares(int[] nums) { return sumOfSquaresRec(nums, 0); }
        private int sumOfSquaresRec(int[] nums, int i) { ...  sumOfSquaresRec(nums, i+1); }

    `first_unit` truncates at the next `public`/`private`/`def`/... line, so the
    classifier only ever saw the WRAPPER, whose body contains no self-call, and
    scored genuine recursion as "other".  This scans every definition region --
    from a definition line to the next one -- for a call to that definition's own
    name, which is recursion wherever it sits in the block.
    """
    lines = (code or "").split("\n")
    defs = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if lang in ("java", "csharp", "c", "cpp", "go") and (
                s.endswith(";") or s.startswith(("return", "if", "for", "while",
                                                 "}", "else"))):
            continue                      # a call, not a signature
        # 2026-08-15 BUGFIX. javascript's ENTRY_RE accepts `const|let|var NAME =`,
        # which is how a function expression is written -- but also how an ordinary
        # local variable is declared. A `const first = ...` inside a function body was
        # being recorded as a new definition, which SPLIT the scan region and hid the
        # self-call that came after it, scoring genuine recursion as "other". Seen on
        # 2 of 4 javascript recursion elicitations. A const/let/var line only defines a
        # function if it binds one.
        if lang == "javascript" and re.match(r"^\s*(?:const|let|var)\s", ln) \
                and not re.search(r"=>|\bfunction\b", ln):
            continue
        m = re.match(ENTRY_RE[lang], ln)
        if m and m.group(1) not in ("if", "return", "let", "in", "for", "while"):
            defs.append((i, m.group(1)))
    for j, (i, name) in enumerate(defs):
        end = defs[j + 1][0] if j + 1 < len(defs) else len(lines)
        body = "\n".join(lines[i + 1:end])
        if lang == "elixir":            # `def f(n), do: ... f(...)` is ONE line
            body = re.sub(rf"^\s*defp?\s+{re.escape(name)}\b", "", lines[i]) + "\n" + body
        if re.search(rf"\b{re.escape(name)}\s*\(", body):
            return True
        if lang in ("ocaml", "fsharp") and re.search(rf"\b{re.escape(name)}\s+\w", body):
            return True      # ML applies by juxtaposition: `f xs`, not `f(xs)`
        if lang == "clojure" and re.search(rf"\({re.escape(name)}\s", body):
            return True      # Lisp call: `(f xs)`
    return False

def classify_recursion(code, lang, fn):
    txt = first_unit(code, lang)
    fold = bool(_PIPE_RE[lang].search(txt))
    body = "\n".join(txt.split("\n")[1:])                       # drop the signature line
    if lang == "elixir" and fn:      # the unit opens with `defmodule`, so the def
        body = re.sub(rf"\bdefp?\s+{re.escape(fn)}\b", "", body)   # header is in the body
    rec = bool(re.search(rf"\b{re.escape(fn)}\s*\(", body)) if fn else False     # self-CALL (not a same-named var)
    # ML-family structural recursion WITHOUT a textual self-call: a Haskell cons
    # pattern `(x:xs)` or an OCaml `match ... with`.  LANGUAGE-GATED (2026-08-14):
    # `\(\s*\w+\s*:\s*\w+\s*\)` matches ANY `(name: Type)` parameter list, so in
    # Rust every `fn f(xs: &[i32])`-style signature -- and any type-annotated
    # single-argument python/typescript signature -- would score "recursion"
    # unconditionally.  Only haskell and ocaml need the fallback; every other
    # language in the panel writes recursion as an explicit self-call.
    if lang in ("haskell", "ocaml"):
        rec = rec or bool(re.search(r"\(\s*\w+\s*:\s*\w+\s*\)|\bmatch\b[\s\S]*\bwith\b", txt))  # cons / match
    if lang == "fsharp":
        # F# requires `let rec` exactly when a binding refers to itself, and
        # applies by juxtaposition like OCaml
        rec = rec or bool(re.search(r"\blet\s+rec\b", txt))
        rec = rec or (bool(fn) and bool(re.search(rf"\b{re.escape(fn)}\s+[\w(]", body)))
    if lang == "clojure":
        # a Lisp self-call is `(f ...)`; `recur` outside a `loop` re-enters the
        # enclosing fn, which is self-recursion without the name
        rec = rec or (bool(fn) and bool(re.search(rf"\({re.escape(fn)}\s", body)))
        rec = rec or (bool(re.search(r"\(recur\b", txt)) and not re.search(r"\(loop\b", txt))
    if lang == "ocaml":
        # OCaml recursion that neither clause above can see (2026-08-14): the model
        # very often writes `let rec f = function | [] -> 0 | x :: xs -> ... f xs`,
        # which has NO `match ... with`, no Haskell-style parenthesised cons `(x:xs)`
        # -- OCaml's cons is the bare infix `x :: xs` -- and no parenthesised
        # self-call, because OCaml applies functions by juxtaposition (`f xs`).
        # Three OCaml-only signals: the `rec` keyword itself (the compiler requires
        # it exactly when the binding refers to itself), a `function`/`match` arm
        # that destructures with `::`, and a juxtaposition self-call `f xs`.
        rec = rec or bool(re.search(r"\blet\s+rec\b", txt))
        rec = rec or bool(re.search(r"\|[^\n|]*::[^\n|]*->", txt))
        rec = rec or (bool(fn) and bool(re.search(rf"\b{re.escape(fn)}\s+\w", body)))
    # wrapper + recursive helper, which first_unit truncates away (see above)
    rec = rec or self_recursive_anywhere(code, lang)
    if rec and not fold:
        return "recursion"
    if fold and not rec:
        return "fold"
    if rec and fold:
        return "mixed"
    return "other"

ENTRY_RE = {
    "java":   r"^\s*(?:public\s+|private\s+|internal\s+|protected\s+)*(?:static\s+)?[\w\[\]<>,\s]+?\b(\w+)\s*\(",
    "csharp": r"^\s*(?:public\s+|private\s+|internal\s+|protected\s+)*(?:static\s+)?[\w\[\]<>,\s]+?\b(\w+)\s*\(",
    "python": r"^\s*def\s+(\w+)\s*\(",
    "haskell": r"^(\w+)\s*::",
    "ocaml":  r"^\s*let\s+(?:rec\s+)?(\w+)",
    # added 2026-08-14. rust pattern copied from polyglot_runners.DEF_PATTERNS;
    # javascript covers `function f(`, `const f = (`, `let/var f = function(`.
    "rust":   r"^\s*(?:pub\s+)?(?:unsafe\s+)?fn\s+(\w+)\s*[(<]",
    "javascript": r"^\s*(?:export\s+(?:default\s+)?)?(?:(?:async\s+)?function\s*\*?\s*"
                  r"|(?:const|let|var)\s+)(\w+)\s*[(=]",
    # added 2026-08-15 for the held-out languages. go and ruby have a single
    # unambiguous definition keyword; c and cpp use the java-style "return-type name("
    # shape, so their patterns must not fire on call sites -- the callers of
    # ENTRY_RE already skip lines starting with return/if/for/while (extended to
    # c/cpp/go below), and requiring a type token before the name does the rest.
    "go":     r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(",
    "ruby":   r"^\s*def\s+(?:self\.)?(\w+)",
    "c":      r"^\s*(?:static\s+|inline\s+|extern\s+|const\s+|unsigned\s+|long\s+)*"
              r"\w+(?:\s*\*)*\s+(\w+)\s*\(",
    "cpp":    r"^\s*(?:static\s+|inline\s+|constexpr\s+|const\s+|unsigned\s+|long\s+)*"
              r"[\w:]+(?:\s*<[^>]*>)?(?:\s*[&*])*\s+(\w+)\s*\(",
    # added 2026-09-04, extension languages. fsharp mirrors ocaml (`let [rec] f`,
    # also `let inline`/`let private`); clojure is `(defn f` or `(def f (fn`;
    # scala has the single keyword `def`; elixir `def`/`defp` (names may end
    # in ? or !).
    "fsharp": r"^\s*let\s+(?:rec\s+)?(?:inline\s+)?(?:private\s+)?(\w+)",
    "clojure": r"^\s*\((?:defn-?\s+|def\s+(?=\S+\s+\(fn\b))([^\s\[\]()\"]+)",
    "scala":  r"^\s*(?:(?:private|protected|final|override|implicit|inline)\s+)*def\s+(\w+)",
    "elixir": r"^\s*defp?\s+(\w+[?!]?)",
}

def detect_entry(lang, code, fallback):
    for ln in (code or "").split("\n"):
        m = re.match(ENTRY_RE[lang], ln)
        if m and m.group(1) not in ("if", "return", "let", "in", "for", "while"):
            return m.group(1)
    return fallback

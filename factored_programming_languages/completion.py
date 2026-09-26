"""Completion-format prompts, so a PRETRAINED checkpoint can be measured on the
same instrument as an instruction-tuned one.

A pretrained model has no chat template and does not follow "Write a function
in Go that ...". It does continue code. So the same request is posed as a
completion. Two layouts:

  layout="example"  (the first attempt, kept so its results stay reproducible)

    <one-shot example, a different and arithmetic-neutral function>

    <comment: Write a function in LANG that DESC.>
    <one contentless generic line>
    <signature, opened>

  layout="file"

    <file header: package main / #include / module Main where>

    <comment: Write a function in LANG that DESC.>
    <signature, opened>
        <generic docstring inside the body, ~31 tokens>

  layout="godoc"    (the current default)

    <file header>

    <comment: Write a function in LANG that DESC.>
    <generic doc comment, ~23 tokens, where Godoc / Doxygen put it>
    <signature, opened>

  layout="minex"

    <primer: a zero-argument function returning a string>

    <comment: Write a function in LANG that DESC.>
    <generic doc comment>
    <signature, opened>

  layout="primer"

    <primer: a completed exercise with a real loop body, on strings>

    <comment: Write a function in LANG that DESC.>
    <generic doc comment>
    <signature, opened>

Why the second layout exists (results/cabl_*, results/cflow_* on the gemma-4
pretrained twins, 2026-09-03): the one-shot example is the only coherent code
left in context once the description is ablated, and the base model either
copies it verbatim or lets its predicate bleed into the task (`n % 10 < 0`,
`filter (< 0)`). About 14 of 24 mask-cut misses were the example, not the
model. Ablating the example too randomises ~75 of ~99 rows (the chat arm
ablates 34 of 53) and the outputs degrade to garbage. So the example goes: a
file header establishes "this is a source file" without offering a function
body to copy, and the opened body with a docstring commits the model to
writing code (the empty-body failure of the first pilot was with a BARE
signature, `def solve(n):`, nothing opened after it).

Why the third and fourth exist (results/cabl_*_file, same day): the in-body
docstring is a LINE COMMENT in C and Go, and a base model at "comment inside a
body, newline" keeps writing comments -- "The function should return ...",
"Do not use ...", Input/Output examples -- or stubs the body ("// Your code
here", "return 0"). Uncut C/Go cells failed to compile on three of the four
models. Haskell was fine, because its generation point is after `solve n =`
and an expression is the only legal continuation; Python's docstring is a
closed string literal, which is why the pilot never saw this. So the doc
comment moves to where Godoc and Doxygen put it, before the signature, and
the opened brace is the last thing in the prompt: the first body line is the
model's. `minex` hedges the other way: it keeps a one-shot primer so the
"comment, signature, body" pattern is demonstrated, but the primer takes no
argument and returns a string, so nothing in it can be pasted into an
int -> int body or leak a predicate into one.

Why the fifth exists (results/cabl_*_godoc and _minex, same day): in Go the
26b and 31B base models STUB the body under both -- "// TODO: Implement this
function", "return 0" -- 10 of 16 uncut Go cells. A doc comment, an opened
`func solve(n int) int {` and no completed function in sight reads as an
exercise template, and a template gets a stub. The one-line minex primer did
not break that; the original example layout (a completed exercise with a
multi-line body) never stubbed. So `primer` keeps a completed exercise but
makes it one whose body cannot be pasted into ours: the longest string in a
list -- a loop and an accumulator over strings, no arithmetic, no predicate
on an integer. If it leaks, a max-over-digits function would come out, which
is not in the bank and is recognisable.

The description span is the request comment alone. Everything after it is the
suffix, which is what survives the cut. In the file / godoc / minex layouts
the suffix is the signature plus the generic doc comment, sized to match the
31 rows the chat prompt has after its description (TAIL + template); the first
pilot bracketed the width the pointer needs at 5 < w* <= 20, so 21 (the
example layout) sits at the edge and 31 does not. wide=False drops the doc
comment, leaving the signature (~8 rows): the narrow control; "xwide" is the
next rung (~55 rows).

Harvesting differs by language and nothing else does: Python and Haskell end a
definition at the first column-0 line, brace languages end one line later, at
the closing brace. `program()` drops everything before the first blank line
(the example, the primer or the header), because the runners supply their own
package / include / module lines.
"""

# one worked example per language: a sign test, sharing no operation with
# digit_sum / digit_prod / even_digits / sq_digit_sum
EXAMPLE = {
    "python": "# Write a function in Python that returns 1 if an integer is "
              "negative and 0 otherwise.\n"
              "def solve(n):\n"
              "    if n < 0:\n"
              "        return 1\n"
              "    return 0\n",
    "c": "// Write a function in C that returns 1 if an integer is negative "
         "and 0 otherwise.\n"
         "int solve(int n) {\n"
         "    if (n < 0) {\n"
         "        return 1;\n"
         "    }\n"
         "    return 0;\n"
         "}\n",
    "go": "// Write a function in Go that returns 1 if an integer is negative "
          "and 0 otherwise.\n"
          "func solve(n int) int {\n"
          "\tif n < 0 {\n"
          "\t\treturn 1\n"
          "\t}\n"
          "\treturn 0\n"
          "}\n",
    "haskell": "-- Write a function in Haskell that returns 1 if an integer "
               "is negative and 0 otherwise.\n"
               "solve :: Int -> Int\n"
               "solve n = if n < 0 then 1 else 0\n",
    "javascript": "// Write a function in JavaScript that returns 1 if an integer "
                  "is negative and 0 otherwise.\n"
                  "function solve(n) {\n"
                  "    if (n < 0) {\n"
                  "        return 1;\n"
                  "    }\n"
                  "    return 0;\n"
                  "}\n",
    "java": "// Write a function in Java that returns 1 if an integer is negative "
            "and 0 otherwise.\n"
            "static int solve(int n) {\n"
            "    if (n < 0) {\n"
            "        return 1;\n"
            "    }\n"
            "    return 0;\n"
            "}\n",
    "cpp": "// Write a function in C++ that returns 1 if an integer is negative "
           "and 0 otherwise.\n"
           "int solve(int n) {\n"
           "    if (n < 0) {\n"
           "        return 1;\n"
           "    }\n"
           "    return 0;\n"
           "}\n",
    "rust": "// Write a function in Rust that returns 1 if an integer is negative "
            "and 0 otherwise.\n"
            "fn solve(n: i64) -> i64 {\n"
            "    if n < 0 {\n"
            "        return 1;\n"
            "    }\n"
            "    0\n"
            "}\n",
    "ruby": "# Write a function in Ruby that returns 1 if an integer is negative "
            "and 0 otherwise.\n"
            "def solve(n)\n"
            "  return 1 if n < 0\n"
            "  0\n"
            "end\n",
    "ocaml": "(* Write a function in OCaml that returns 1 if an integer is "
             "negative and 0 otherwise. *)\n"
             "let solve n = if n < 0 then 1 else 0\n",
}
# minex layout: a primer that demonstrates "comment, signature, body" and
# nothing else -- no argument, no arithmetic, no predicate, a string result
MINEX = {
    "python": "# Write a function in Python that returns the string \"hello\".\n"
              "def greet():\n"
              "    return \"hello\"\n",
    "c": "// Write a function in C that returns the string \"hello\".\n"
         "const char *greet(void) {\n"
         "    return \"hello\";\n"
         "}\n",
    "go": "// Write a function in Go that returns the string \"hello\".\n"
          "func greet() string {\n"
          "\treturn \"hello\"\n"
          "}\n",
    "haskell": "-- Write a function in Haskell that returns the string \"hello\".\n"
               "greet :: String\n"
               "greet = \"hello\"\n",
    "javascript": "// Write a function in JavaScript that returns the string \"hello\".\n"
                  "function greet() {\n"
                  "    return \"hello\";\n"
                  "}\n",
    "java": "// Write a function in Java that returns the string \"hello\".\n"
            "static String greet() {\n"
            "    return \"hello\";\n"
            "}\n",
    "cpp": "// Write a function in C++ that returns the string \"hello\".\n"
           "std::string greet() {\n"
           "    return \"hello\";\n"
           "}\n",
    "rust": "// Write a function in Rust that returns the string \"hello\".\n"
            "fn greet() -> String {\n"
            "    String::from(\"hello\")\n"
            "}\n",
    "ruby": "# Write a function in Ruby that returns the string \"hello\".\n"
            "def greet\n"
            "  \"hello\"\n"
            "end\n",
    "ocaml": "(* Write a function in OCaml that returns the string \"hello\". *)\n"
             "let greet () = \"hello\"\n",
}
# primer layout: a completed exercise with a real loop body, on strings
PRIMER = {
    "python": "# Write a function in Python that returns the longest string in a "
              "list.\n"
              "def longest(items):\n"
              "    best = \"\"\n"
              "    for s in items:\n"
              "        if len(s) > len(best):\n"
              "            best = s\n"
              "    return best\n",
    "c": "// Write a function in C that returns the longest string in an array of "
         "strings.\n"
         "const char *longest(const char **items, int count) {\n"
         "    const char *best = \"\";\n"
         "    for (int i = 0; i < count; i++) {\n"
         "        if (strlen(items[i]) > strlen(best)) {\n"
         "            best = items[i];\n"
         "        }\n"
         "    }\n"
         "    return best;\n"
         "}\n",
    "go": "// Write a function in Go that returns the longest string in a list.\n"
          "func longest(items []string) string {\n"
          "\tbest := \"\"\n"
          "\tfor _, s := range items {\n"
          "\t\tif len(s) > len(best) {\n"
          "\t\t\tbest = s\n"
          "\t\t}\n"
          "\t}\n"
          "\treturn best\n"
          "}\n",
    "haskell": "-- Write a function in Haskell that returns the longest string in a "
               "list.\n"
               "longest :: [String] -> String\n"
               "longest = foldr (\\s best -> if length s > length best then s "
               "else best) \"\"\n",
    "javascript": "// Write a function in JavaScript that returns the longest string "
                  "in a list.\n"
                  "function longest(items) {\n"
                  "    let best = \"\";\n"
                  "    for (const s of items) {\n"
                  "        if (s.length > best.length) {\n"
                  "            best = s;\n"
                  "        }\n"
                  "    }\n"
                  "    return best;\n"
                  "}\n",
    "java": "// Write a function in Java that returns the longest string in a list.\n"
            "static String longest(List<String> items) {\n"
            "    String best = \"\";\n"
            "    for (String s : items) {\n"
            "        if (s.length() > best.length()) {\n"
            "            best = s;\n"
            "        }\n"
            "    }\n"
            "    return best;\n"
            "}\n",
    "cpp": "// Write a function in C++ that returns the longest string in a list.\n"
           "std::string longest(const std::vector<std::string> &items) {\n"
           "    std::string best = \"\";\n"
           "    for (const auto &s : items) {\n"
           "        if (s.size() > best.size()) {\n"
           "            best = s;\n"
           "        }\n"
           "    }\n"
           "    return best;\n"
           "}\n",
    "rust": "// Write a function in Rust that returns the longest string in a list.\n"
            "fn longest(items: &[String]) -> String {\n"
            "    let mut best = String::new();\n"
            "    for s in items {\n"
            "        if s.len() > best.len() {\n"
            "            best = s.clone();\n"
            "        }\n"
            "    }\n"
            "    best\n"
            "}\n",
    "ruby": "# Write a function in Ruby that returns the longest string in a list.\n"
            "def longest(items)\n"
            "  best = \"\"\n"
            "  items.each do |s|\n"
            "    best = s if s.length > best.length\n"
            "  end\n"
            "  best\n"
            "end\n",
    "ocaml": "(* Write a function in OCaml that returns the longest string in a "
             "list. *)\n"
             "let longest items =\n"
             "  List.fold_left (fun best s -> "
             "if String.length s > String.length best then s else best) \"\" items\n",
}
# file / godoc layouts: a header that says "source file", nothing about any task
HEADER = {
    "python": "#!/usr/bin/env python3\n",
    "c": "#include <stdio.h>\n",
    "go": "package main\n",
    "haskell": "module Main where\n",
    "javascript": "'use strict';\n",
    "java": "import java.util.*;\n",
    "cpp": "#include <string>\n",
    "rust": "#![allow(dead_code)]\n",
    "ruby": "#!/usr/bin/env ruby\n",
    "ocaml": "(* solve.ml *)\n",
}
COMMENT = {"python": "#", "c": "//", "go": "//", "haskell": "--",
           "javascript": "//", "java": "//", "cpp": "//", "rust": "//",
           "ruby": "#", "ocaml": "(*"}
# OCaml has no line comment, so its comment needs closing. Everything else ends
# at the newline (authors, 2026-09-18: the four-task bank in all ten languages).
COMMENT_END = {"ocaml": " *)"}
# the contentless line of the example layout: room for the description to
# compile into, saying nothing about the task (~13 rows with marker + newline)
GENERIC = "Implements exactly the behaviour described in the comment above."
# the in-body docstring of the file layout: the same sentence plus one more
# that only restates the signature, sized so signature + docstring is ~31 rows
# on the gemma-4 tokenizer (c 32, go 33, haskell 32, python 30), the width the
# chat prompt has after its description
GENERIC_BODY = ("Implements exactly the behaviour described in the comment "
                "above. Takes a single integer argument and returns an integer.")
# the next rung of the width ladder (~55 rows with the signature): two more
# sentences that still say nothing about the task
GENERIC_BODY_X = (GENERIC_BODY + " Uses no helper functions, no input or output, "
                  "and no global state. The body below is the complete "
                  "implementation.")
# how the function is opened for the model
SIG = {
    "python": "def solve(n):\n",
    "c": "int solve(int n) {\n",
    "go": "func solve(n int) int {\n",
    "haskell": "solve :: Int -> Int\nsolve n =",
    "javascript": "function solve(n) {\n",
    # the java runner wraps whatever it is given in `public class Main`, so a
    # bare static method is what the completion should open (execute._raw_java)
    "java": "static int solve(int n) {\n",
    "cpp": "int solve(int n) {\n",
    "rust": "fn solve(n: i64) -> i64 {\n",
    "ruby": "def solve(n)\n",
    "ocaml": "let solve n =",
}
# the in-body docstring, per language; Haskell has no body comment before the
# expression, so its line sits between the type signature and the equation
_DOC_WRAP = {"python": ('    """', '"""\n'), "c": ("    // ", "\n"),
             "go": ("\t// ", "\n"), "haskell": ("-- ", "\n"),
             "javascript": ("    // ", "\n"), "java": ("    // ", "\n"),
             "cpp": ("    // ", "\n"), "rust": ("    // ", "\n"),
             "ruby": ("  # ", "\n"), "ocaml": ("  (* ", " *)\n")}


def body_doc(lang, width):
    text = {"wide": GENERIC_BODY, "xwide": GENERIC_BODY_X}[width]
    pre, post = _DOC_WRAP[lang]
    return pre + text + post


CLOSER = {"c": "}", "go": "}", "javascript": "}", "java": "}", "cpp": "}",
          "rust": "}", "ruby": "end"}
DISPLAY = {"python": "Python", "c": "C", "go": "Go", "haskell": "Haskell",
           "javascript": "JavaScript", "java": "Java", "cpp": "C++",
           "rust": "Rust", "ruby": "Ruby", "ocaml": "OCaml"}
LAYOUTS = ("example", "file", "godoc", "minex", "primer")
WIDTHS = ("narrow", "wide", "xwide")          # xwide: not for the example layout
DOC_TEXT = {"wide": GENERIC_BODY, "xwide": GENERIC_BODY_X}


def _width(wide):
    """the runners pass either a bool (wide / narrow) or a width name"""
    if isinstance(wide, str):
        assert wide in WIDTHS, wide
        return wide
    return "wide" if wide else "narrow"


def build(lang, desc, wide=True, layout="example"):
    """-> (text, description span as (start, end) char offsets)

    wide: True/"wide" keeps the generic line (example layout) or the in-body
    docstring (file layout); False/"narrow" drops it, leaving only the
    signature after the description -- the arm expected to hold nothing, which
    shows the width matters rather than the format; "xwide" (file layout only)
    is the next rung of the ladder, ~55 rows.
    """
    assert layout in LAYOUTS, layout
    width = _width(wide)
    c = COMMENT[lang]
    head = {"example": EXAMPLE, "minex": MINEX,
            "primer": PRIMER}.get(layout, HEADER)[lang] + "\n"
    ce = COMMENT_END.get(lang, "")
    comment = f"{c} Write a function in {DISPLAY[lang]} that {desc}.{ce}\n"
    lo = len(head)
    hi = lo + len(comment)
    if layout == "example":
        assert width != "xwide", "xwide is not defined for the example layout"
        tail = (f"{c} {GENERIC}{ce}\n" if width == "wide" else "") + SIG[lang]
    elif layout == "file":
        doc = body_doc(lang, width) if width != "narrow" else ""
        if lang == "haskell":
            tsig, eq = SIG[lang].split("\n")
            tail = tsig + "\n" + doc + eq
        else:
            tail = SIG[lang] + doc
    else:                           # godoc, minex, primer: doc comment, then signature
        doc = f"{c} {DOC_TEXT[width]}{ce}\n" if width != "narrow" else ""
        tail = doc + SIG[lang]
    return head + comment + tail, (lo, hi)


# a Haskell generation is over at the first column-0 line that is not part of
# the answer: a main, a new exercise, a redefinition of solve, or a
# module/import line. Everything else at column 0 -- a helper's type signature
# and equations, a comment introducing it -- is kept, because the models write
# `product (digits n)` and define `digits` next, and the chat-format instrument
# keeps the whole code block, helpers included.
_HS_STOP = ("main", "-- Write a function", "solve", "import ", "module ")


def _hs_stop(ln):
    return bool(ln) and not ln.startswith((" ", "\t")) and ln.startswith(_HS_STOP)


def finished(lang, generated):
    """has the generation run past the end of the answer? (shared by the
    runners' decode loops and by harvest, so they cannot disagree)"""
    lines = generated.split("\n")
    if lang == "haskell":
        return any(_hs_stop(ln) for ln in lines[1:])
    # a brace language ends one line after the first column-0 line, at the
    # closing brace; python and the rest end at the first column-0 line
    return len(lines) > 2 and any(
        ln and not ln.startswith((" ", "\t")) and ln.strip() != CLOSER.get(lang)
        for ln in lines[1:])


def harvest(lang, generated):
    """the code the model wrote, trimmed at the end of the answer"""
    lines = generated.split("\n")
    keep = []
    if lang == "haskell":
        for i, ln in enumerate(lines):
            if i and _hs_stop(ln):
                break
            keep.append(ln)
        # drop trailing blank and comment-only lines
        while keep and (not keep[-1].strip() or keep[-1].lstrip().startswith("--")):
            keep.pop()
        return "\n".join(keep)
    for i, ln in enumerate(lines):
        if i and ln and not ln.startswith((" ", "\t")):
            if lang in CLOSER and ln.strip() == CLOSER[lang]:
                keep.append(ln)
            break
        keep.append(ln)
    return "\n".join(keep)


def program(lang, desc, generated, wide=True, layout="example"):
    """the full compilable program: what we prompted with plus what it wrote"""
    text, _ = build(lang, desc, wide, layout)
    # drop the one-shot example (it defines `solve` too and would collide),
    # the primer, or the file header (the runners supply their own)
    body = text.split("\n\n", 1)[1]
    return body + harvest(lang, generated)

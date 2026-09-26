"""Entry-point and language detection for generated programs (find_entry, detect_language), shared by every runner and grader."""
import re
from code_style import detect_entry
LANG_SIG = [
    ("haskell", r"(?m)^\s*solve\s*::|^\s*\w+\s*::\s*\w"),
    ("ocaml", r"\blet\s+rec\b|\blet\s+solve\b"),
    ("rust", r"\bfn\s+solve\s*\("),
    ("go", r"\bfunc\s+solve\s*\("),
    # ruby before python: both open a function with `def solve(...)`, but ruby's header takes no trailing colon
    # and the body closes with a bare `end`. Without this, idiomatic ruby (`def solve(n)` / `.to_s.chars.map(&:to_i)`
    # / `end`) was reported as python (2026-09-14).
    ("ruby", r"(?ms)^\s*def\s+\w+(?:\s*\([^)]*\))?\s*$.*?^\s*end\s*$"),
    ("python", r"(?m)^\s*def\s+solve\s*\("),
    ("csharp", r"\busing\s+System\b|\bpublic\s+static\s+int\s+Solve\b"),
    ("java", r"\b(public|private|static)\b[^\n]*\bsolve\s*\([^\n]*\)\s*\{"),
    ("c", r"#include|\bint\s+solve\s*\(\s*int\b"),
    ("javascript", r"\bfunction\s+solve\s*\(|\bconst\s+solve\s*=|=>"),
]


def detect_language(code):
    for name, pat in LANG_SIG:
        if re.search(pat, code or ""):
            return name
    return "unknown"


def find_entry(lang, code):
    if code and re.search(r"\bsolve\b", code):
        return "solve"
    return detect_entry(lang, code, None) if code else None

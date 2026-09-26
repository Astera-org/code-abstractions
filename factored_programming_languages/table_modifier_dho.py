"""Per-slice outcome table for the double-held-out modifier transfer (authors, 2026-09-14: "split it out per slice, and
within each slice differentiate 'perfect' from 'made it, but incorrect'").

Outcome classes for one cell of an arm (first-block grading):
  perfect     the program implements M[f_b] in lam_b
  arrived     not perfect, but the modifier's operation is present in the emitted code (comments stripped):
              x2: `* 2` / `2 *` / `<< 1` / double / twice;  +1: `+ 1` / `1 +` / succ in a RESULT line (return,
              the Haskell definition line, or the trailing Rust expression), so loop counters do not count;
              squaring (either placement): `** 2` / `^ 2` / pow / `x * x`
  base        outputs reproduce f_b and no modifier operation is present
  other       a valid program computing something else, no modifier operation
  invalid     did not compile / run / no entry, no modifier operation
    MODEL=gemma-4-31B-it uv run factored_programming_languages/table_modifier_dho.py [--arm pooled|single] [--examples]
"""
import collections
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = os.environ.get("MODEL", "gemma-4-31B-it")
MODE = os.environ.get("MODE", "sever")
ARM = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "pooled"
MODS = ["post_sq", "pre_sq", "post_inc", "post_dbl"]
BASES = ["sum", "prod", "max", "min", "altsum"]
LANGS = ["java", "go", "rust", "haskell"]
CLASSES = ["perfect", "arrived", "base", "other", "invalid"]


def strip_comments(code):
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
    code = re.sub(r"\{-.*?-\}", " ", code, flags=re.S)
    return "\n".join(re.split(r"//|#(?!\[)|--", ln, maxsplit=1)[0] for ln in code.splitlines())


def result_lines(code, lang):
    lines = [ln for ln in code.splitlines() if ln.strip()]
    out = [ln for ln in lines if re.search(r"\breturn\b", ln)]
    if lang == "haskell":
        out += [ln for ln in lines if re.match(r"\s*solve\b.*=", ln)]
    if lang == "rust" and lines:
        out.append(lines[-1] if not lines[-1].strip().startswith("}") else (lines[-2] if len(lines) > 1 else ""))
    return "\n".join(out) or "\n".join(lines[-2:])


def arrived(code, mod, lang):
    c = strip_comments(code or "").lower()
    if mod == "post_dbl":
        return bool(re.search(r"\*\s*2(?![\d.])|(?<![\w.])2\s*\*|<<\s*1\b|\bdouble\b|\btwice\b", c))
    if mod == "post_inc":
        r = result_lines(c, lang)
        return bool(re.search(r"\+\s*1(?![\d.])|(?<![\w.])1\s*\+|\bsucc\b", r))
    return bool(re.search(r"\*\*\s*2|\^\s*2|\bpow\b|\.pow\(|\b(\w+)\s*\*\s*\1\b|\bsq(?:uare)?\w*\b", c))


def classify(r):
    if r["hit"]:
        return "perfect"
    if arrived(r["code"], r["mod"], r["lang"]):
        return "arrived"
    if r["kept_base"]:
        return "base"
    return "other" if r["status"] == "ok" else "invalid"


recs = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "results" / (f"modifier_dho_{MODEL}" + ("_install" if MODE == "install" else "")) / "test" / "*.json"))]
rs = [r for r in recs if r["arm"] == ARM]
for r in rs:
    r["cls"] = classify(r)


def row(sel):
    c = collections.Counter(r["cls"] for r in sel)
    return " | ".join(str(c[k]) for k in CLASSES) + f" | {len(sel)}"


hdr = "| slice | " + " | ".join(CLASSES) + " | n |\n|---|" + "---|" * (len(CLASSES) + 1)
print(f"arm = {ARM}, {len(rs)} cells (first-block grading)\n")
print("## modifier x base (8 cells each: 4 languages x 2 wordings)\n" + hdr)
for m in MODS:
    for b in BASES:
        print(f"| {m} on {b} | " + row([r for r in rs if r["mod"] == m and r["base"] == b]) + " |")
print("\n## modifier x language (10 cells each)\n" + hdr)
for m in MODS:
    for l in LANGS:
        print(f"| {m} / {l} | " + row([r for r in rs if r["mod"] == m and r["lang"] == l]) + " |")
print("\n## base x language (8 cells each)\n" + hdr)
for b in BASES:
    for l in LANGS:
        print(f"| {b} / {l} | " + row([r for r in rs if r["base"] == b and r["lang"] == l]) + " |")
print("\n## marginals\n" + hdr)
for m in MODS:
    print(f"| {m} | " + row([r for r in rs if r["mod"] == m]) + " |")
for b in BASES:
    print(f"| {b} | " + row([r for r in rs if r["base"] == b]) + " |")
for l in LANGS:
    print(f"| {l} | " + row([r for r in rs if r["lang"] == l]) + " |")
for w in (0, 1):
    print(f"| wording {w} | " + row([r for r in rs if r["wording"] == w]) + " |")
print(f"| all | " + row(rs) + " |")
if "--examples" in sys.argv:
    print("\n## 'arrived' examples (result lines)")
    for r in [r for r in rs if r["cls"] == "arrived"][:12]:
        print(f"--- {r['mod']} on {r['base']} / {r['lang']} w{r['wording']}: status={r['status']} impl={r['impl']}")
        print("    " + "\n    ".join(result_lines(strip_comments(r["code"] or ""), r["lang"]).splitlines()[:4]))

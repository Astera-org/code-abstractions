"""Every one of the canonical ten builds and runs a known-correct program.

The transplant scores which function was implemented by EXECUTING the generation, so a
broken runner is indistinguishable from a model that failed. This pins each runner
against a hand-written digit_sum before any GPU time is spent.

    uv run factored_programming_languages/verify_runners.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import execute as X                                                      # noqa: E402
from tasks_function import BATTERY, WANT                                 # noqa: E402
from languages import LANGS                                              # noqa: E402

GOOD = {
    "python": "def solve(n):\n    return sum(int(c) for c in str(n))\n",
    "javascript": ("function solve(n) {\n"
                   "  return String(n).split('').reduce((a, c) => a + Number(c), 0);\n}\n"),
    "c": "int solve(int n){int s=0;while(n>0){s+=n%10;n/=10;}return s;}\n",
    "cpp": "int solve(int n){int s=0;while(n>0){s+=n%10;n/=10;}return s;}\n",
    "java": ("public static int solve(int n) {\n"
             "    int s = 0;\n    while (n > 0) { s += n % 10; n /= 10; }\n"
             "    return s;\n}\n"),
    "go": ("func solve(n int) int {\n\ts := 0\n\tfor n > 0 {\n\t\ts += n % 10\n"
           "\t\tn /= 10\n\t}\n\treturn s\n}\n"),
    "rust": ("fn solve(n: i64) -> i64 {\n    let mut n = n;\n    let mut s = 0;\n"
             "    while n > 0 { s += n % 10; n /= 10; }\n    s\n}\n"),
    "ruby": "def solve(n)\n  n.to_s.chars.map(&:to_i).sum\nend\n",
    "ocaml": "let rec solve n = if n = 0 then 0 else (n mod 10) + solve (n / 10)\n",
    "haskell": ("solve :: Int -> Int\nsolve 0 = 0\n"
                "solve n = n `mod` 10 + solve (n `div` 10)\n"),
    # extension languages (FPL_EXTRA_LANGS), 2026-09-04
    "fsharp": ("let rec solve (n: int) : int =\n"
               "    if n = 0 then 0 else n % 10 + solve (n / 10)\n"),
    "clojure": ("(defn solve [n]\n"
                "  (if (zero? n) 0 (+ (mod n 10) (solve (quot n 10)))))\n"),
    "scala": ("object Solution {\n  def solve(n: Int): Int =\n"
              "    if (n == 0) 0 else n % 10 + solve(n / 10)\n}\n"),
    "elixir": ("defmodule Solution do\n  def solve(0), do: 0\n"
               "  def solve(n), do: rem(n, 10) + solve(div(n, 10))\nend\n"),
}
# a wrong program in each: must build, and must NOT grade as any bank function
WRONG = {
    "fsharp": "let solve (n: int) : int = n + 1\n",
    "clojure": "(defn solve [n] (inc n))\n",
    "scala": "def solve(n: Int): Int = n + 1\n",
    "elixir": "defmodule S do\n  def solve(n), do: n + 1\nend\n",
}


def main():
    X.BATTERY = BATTERY
    want = WANT["digit_sum"]
    print(f"battery {BATTERY} -> digit_sum {want}\n")
    bad = []
    for lang in LANGS:
        st, got = "no_runner", None
        for form in X.FORMS:
            st, got = X.RUNNERS[lang](GOOD[lang], "solve", form)
            if st == "ok":
                break
        ok = st == "ok" and got == want
        print(f"  {lang:<11} {st:<13} {'PASS' if ok else 'FAIL ' + str(got)}")
        if not ok:
            bad.append(lang)
    # the wrong-program half of the control, through the memoised scoring path
    # the runners use (scoring.run_forms): builds, but names no bank function
    from scoring import run_forms
    from tasks_function import which_function
    for lang in LANGS:
        if lang not in WRONG:
            continue
        st1, got1 = X.RUNNERS[lang](WRONG[lang], "solve", 1)   # it does build and run
        st, got = run_forms(lang, WRONG[lang], "solve")        # the sweep names nothing
        # (run_forms returns the LAST form's status when no form names a bank
        # function; the extra-argument forms fail on a one-argument function,
        # so that status is a failure by design and is not what is checked)
        ok = st1 == "ok" and which_function(got1) is None and which_function(got) is None
        print(f"  {lang:<11} wrong-program form1={st1:<13} sweep_impl={which_function(got)} "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            bad.append(lang + ":wrong")
    print()
    if bad:
        print(f"FAILED: {bad}")
        return 1
    print(f"all {len(LANGS)} runners pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The execution grader: every generated program is built and run in a real toolchain against a fixed
battery of inputs, and scored at three levels. Every runner rebinds BATTERY before grading (README.md).

Levels, from loosest to strictest:
  (a) CLEAN FLIP   - the recursion classifier labels the first unit "recursion" and the
                     unit is not degenerate.
  (b) COMPILES     - the emitted function builds in a real toolchain
                     (javac / dotnet / ghc / ocamlopt / python import).
  (c) TASK-OK      - it builds AND returns the reference answers of the task it was
                     asked for on a fixed battery of integer lists.
Entry points are detected with code_style.detect_entry. If a one-argument
call does not build, a two-argument (list, index) call is tried once -- recursion often
introduces an index accumulator -- and `call_form` records which was used.

Toolchains: $FPL_TOOLCHAINS (default /opt/toolchains; temurin JDK 21, .NET 8 SDK, GHC 9.6.6,
OCaml 5.2.0); python runs in a subprocess with the repo interpreter.

Imported by every runner (`import execute as S`); verify_runners.py is the positive control.
"""
import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from elicit import extract_code                                          # noqa: E402
from code_style import (classify_recursion, detect_entry, first_unit,     # noqa: E402
                        is_degenerate)

# Root of the user-space toolchains (JDK 21, GHC, OCaml, rustup, ...). Override with
# FPL_TOOLCHAINS; see README.md "Execution toolchains" for the expected layout.
T = Path(os.environ.get("FPL_TOOLCHAINS", "/opt/toolchains"))
JAVAC, JAVA = str(T / "jdk-21.0.7+6" / "bin" / "javac"), str(T / "jdk-21.0.7+6" / "bin" / "java")
GHC = str(T / ".ghcup" / "bin" / "ghc")
GHC_LDFLAG = f"-optl-L{T / 'lib'}"
OCAMLOPT = str(T / "ocaml-5.2.0" / "bin" / "ocamlopt")
DOTNET = str(T / "dotnet" / "dotnet")
LIB_ENV = {**os.environ,
           "LD_LIBRARY_PATH": ":".join([str(T / "jdk-21.0.7+6" / "lib"),
                                        os.environ.get("LD_LIBRARY_PATH", "")])}
DOTNET_ENV = {**os.environ, "DOTNET_ROOT": str(T / "dotnet"),
              "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1",
              "DOTNET_CLI_HOME": "/tmp/dotnet_home",
              "PATH": str(T / "dotnet") + ":" + os.environ.get("PATH", "")}

# the four generation tasks of the panel, in run_style_axis.py's GEN_PROMPTS order
BATTERY = [[1, 2, 3], [-2, 5, 0, 7], [4], [3, -1, 2, -4, 6], [10, -10, 5], []]
REF = [lambda v: sum(x * x for x in v),
       lambda v: sum(x for x in v if x > 0),
       lambda v: sum(1 for x in v if x % 2 == 0),
       lambda v: sum(x ** 3 for x in v)]

# SCALAR tasks (indices 4-7), added 2026-08-14 for the scalar-vs-array patch test in
# run_patch_interchange.py. The battery entries are plain positive integers rather
# than lists; every literal builder below emits a bare (parenthesised) number for
# them, so the same runners execute both task families. `execute` swaps BATTERY for
# SBATTERY for the duration of a scalar call and restores it, so nothing that scores
# the array tasks changes.
SCALAR_TI0 = 4
SBATTERY = [7, 1234, 5, 90, 406, 2468]
SREF = [lambda n: sum(int(c) for c in str(n)),
        lambda n: sum(1 for c in str(n) if int(c) % 2 == 0),
        lambda n: math.prod(int(c) for c in str(n)),
        lambda n: sum(int(c) ** 2 for c in str(n))]


def _lit(v, open_, sep, close_):
    """array literal for a list battery entry, a parenthesised number for a scalar
    one (`(7)` is a valid expression in every language of the panel)."""
    return (f"({v})" if isinstance(v, int)
            else open_ + sep.join(str(x) for x in v) + close_)


def _run(cmd, timeout=120, env=None, cwd=None, via_bash=False):
    """Run a command, returning CompletedProcess, or None if it timed out.

    The child gets its OWN SESSION so a timeout can kill the whole tree. This is
    load-bearing, not tidiness: `go run` compiles to /tmp/go-build*/exe/main and
    execs it as a CHILD, and `bash -c` likewise wraps a grandchild. Killing only
    the direct child -- which is all subprocess.run's timeout does -- leaves any
    generated program that loops forever spinning at 100% CPU, reparented to
    init, until the box is rebooted. Graded code loops often enough that these
    accumulated to 24 orphans burning ~23 of 104 cores over three days.
    """
    if via_bash:
        import shlex
        cmd = ["bash", "-c", shlex.join(cmd)]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=env, cwd=cwd, start_new_session=True)
    except FileNotFoundError as e:   # a toolchain binary is missing: grade as a build failure, do not crash the run
        return subprocess.CompletedProcess(cmd, 127, "", f"toolchain binary not found: {e.filename} (set FPL_TOOLCHAINS; see README.md)")
    try:
        out, err = p.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, p.returncode, out, err)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            p.kill()
        try:                      # reap, so the killed tree leaves no zombie
            p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        return None


def _ints(out):
    try:
        return [int(x) for x in out.split()]
    except ValueError:
        return None


# Recursion introduces an accumulator argument, and which one it is varies:
# form 1 = f(list); 2 = f(list, 0) (index counting up); 3 = f(list, n) and
# 4 = f(list, n-1) (index counting down, the shape a "helper(arr, n)" recursion
# uses). Without 3/4 a correct length-counter recursion scores task-FAIL.
FORMS = (1, 2, 3, 4)


def _arg(form, v, sep):
    """the extra call argument for `v` under `form`, prefixed by `sep`."""
    if form == 1:
        return ""
    # a scalar battery entry has no index to count: the extra argument a scalar
    # digit-recursion introduces is an ACCUMULATOR (0 for a sum, 1 for a product).
    n = ({2: 0, 3: 1, 4: 0}[form] if isinstance(v, int)
         else {2: 0, 3: len(v), 4: len(v) - 1}[form])
    return sep + (f"({n})" if n < 0 else str(n))


# ---- generous imports (authors, 2026-09-17: "if they're there, that's fine, if they're not there, we add
# them"). Our prompts say "return only the function", so a model that omits an import has followed the
# instruction rather than made an error. Each driver builds the generation AS WRITTEN first; only if that
# fails does it retry with the language's preamble prepended. Retry-on-failure is strictly additive:
# nothing that passes today can start failing, and code that already imports what it needs never sees it.
# (haskell instead widens HS_MODS -- it imports only names the code references and does not define, so a
# model-written `sort` never becomes ambiguous.)
PRELUDE = {
    "python": ("import math, itertools, functools, collections, re, string, heapq, bisect\n"
               "from collections import Counter, defaultdict, deque\n"
               "from functools import reduce, lru_cache\n"
               "from itertools import accumulate, chain, groupby, permutations, combinations\n"),
    "c": "#include <math.h>\n#include <ctype.h>\n#include <limits.h>\n",
    "cpp": ("#include <cmath>\n#include <map>\n#include <set>\n#include <unordered_map>\n"
            "#include <unordered_set>\n#include <sstream>\n#include <climits>\n#include <cctype>\n"),
    "java": "import java.math.*;\nimport java.util.function.*;\n",
    "rust": ("#[allow(unused_imports)] use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet};\n"
             "#[allow(unused_imports)] use std::cmp::{min, max};\n"),
    "ruby": "require 'set'\n",
    "ocaml": "",
}
HS_MODS_WIDE = {
    "Data.List": ("sort", "sortBy", "sortOn", "foldl'", "foldl1'", "nub", "group", "groupBy", "genericLength",
                  "partition", "transpose", "intercalate", "isPrefixOf", "isSuffixOf", "isInfixOf", "maximumBy",
                  "minimumBy", "tails", "inits", "subsequences", "permutations", "unfoldr", "zipWith3"),
    "Data.Char": ("digitToInt", "isDigit", "ord", "chr", "isSpace", "isAlpha", "isUpper", "isLower", "toUpper",
                  "toLower", "isAlphaNum", "intToDigit"),
    "Data.Maybe": ("fromMaybe", "mapMaybe", "catMaybes", "isJust", "isNothing", "fromJust"),
    "Data.Ord": ("comparing",),
    "Data.Bits": ("shiftR", "shiftL", "xor", "popCount"),
}


# ---- per-language build+run; each returns (status, outputs) --------------------
# status: "ok" (built and ran), "compile_fail", "run_fail", "timeout", "no_entry"
def run_python(code, entry, form, _retry=True):
    calls = "\n".join(f"print({entry}({v!r}{_arg(form, v, ', ')}))" for v in BATTERY)
    src = code + "\n\nimport sys\n" + calls + "\n"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.py").write_text(src)
        r = _run([sys.executable, "m.py"], cwd=td, timeout=40)
    if _retry and (r is None or r.returncode != 0):        # missing import? try again with the preamble
        return run_python(PRELUDE["python"] + code, entry, form, _retry=False)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return ("compile_fail" if "SyntaxError" in r.stderr else "run_fail"), None
    return "ok", _ints(r.stdout)


def _is_static(code, entry):
    """A model-written method may be an instance method; call it accordingly."""
    m = re.search(rf"(?m)^[^\n]*\b{re.escape(entry)}\s*\(", code)
    return bool(m and "static" in m.group(0))


_CLASS_RE = re.compile(r"\A\s*(?:public\s+|final\s+|abstract\s+|static\s+)*class\s+"
                       r"\w+\s*\{(.*)\}\s*\Z", re.S)


def _unwrap_class(code):
    """Steered java/C# generations often arrive already wrapped in their own
    `public class Main { ... }`. Wrapping that again nests a class where a method
    is expected and the file will not build, which has nothing to do with the
    code the model wrote."""
    m = _CLASS_RE.match(code)
    return m.group(1) if m else code


def _raw_java(code, entry, form):
    code = _unwrap_class(code)
    lits = [_lit(v, "new int[]{", ",", "}") for v in BATTERY]
    recv = "" if _is_static(code, entry) else "new Main()."
    calls = " ".join(f"System.out.println({recv}{entry}({L}{_arg(form, v, ', ')}));"
                     for v, L in zip(BATTERY, lits))
    src = ("import java.util.*;\nimport java.util.stream.*;\n"
           f"public class Main {{\n{code}\n"
           f"public static void main(String[] a) {{ {calls} }}\n}}\n")
    with tempfile.TemporaryDirectory() as td:
        Path(td, "Main.java").write_text(src)
        c = _run([JAVAC, "Main.java"], cwd=td, env=LIB_ENV, via_bash=True)
        if c is None:
            return "timeout", None
        if c.returncode != 0:
            return "compile_fail", None
        r = _run([JAVA, "-cp", td, "Main"], timeout=40, env=LIB_ENV, via_bash=True)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return "run_fail", None
    return "ok", _ints(r.stdout)


CS_PROJ = Path("/tmp/pvs_csproj")


def _cs_init():
    if (CS_PROJ / "app" / "app.csproj").exists():
        return
    CS_PROJ.mkdir(parents=True, exist_ok=True)
    _run([DOTNET, "new", "console", "-o", "app", "--force"], cwd=str(CS_PROJ),
         env=DOTNET_ENV, timeout=300)


def run_csharp(code, entry, form):
    _cs_init()
    code = _unwrap_class(code)
    lits = [_lit(v, "new int[]{", ",", "}") for v in BATTERY]
    recv = "" if _is_static(code, entry) else "new P()."
    calls = "\n".join(f"    Console.WriteLine({recv}{entry}({L}{_arg(form, v, ', ')}));"
                      for v, L in zip(BATTERY, lits))
    # Main sits INSIDE P: C# members are private by default, and a model that
    # writes `int f(int[] a)` with no modifier is not reachable from outside.
    src = ("using System;\nusing System.Linq;\nusing System.Collections.Generic;\n"
           f"public class P {{\n{code}\n"
           f"  public static void Main() {{\n{calls}\n  }}\n}}\n")
    Path(CS_PROJ, "app", "Program.cs").write_text(src)
    r = _run([DOTNET, "run", "--project", "app"], cwd=str(CS_PROJ), env=DOTNET_ENV,
             timeout=180)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return ("compile_fail" if ("error CS" in r.stdout or "error CS" in r.stderr)
                else "run_fail"), None
    return "ok", _ints(r.stdout)


HS_MODS = {"Data.List": ("sort", "sortBy", "sortOn", "foldl'", "nub", "group",
                         "groupBy", "genericLength", "partition"),
           "Data.Char": ("digitToInt", "isDigit", "ord", "chr", "isSpace")}


def run_haskell(code, entry, form, _mods=None):
    """Import only the stdlib names the generation actually references and does not
    define itself (a model-written `sort` would otherwise become ambiguous). On failure the caller retries with
    HS_MODS_WIDE, which covers far more of Data.List / Char / Maybe / Ord / Bits (generous imports)."""
    defined = set(re.findall(r"(?m)^(\w+)", code))
    imports = ""
    for mod, names in (_mods or HS_MODS).items():
        need = [w for w in names if w not in defined
                and re.search(rf"(?<![\w']){re.escape(w)}(?![\w'])", code)]
        if need:
            imports += f"import {mod} ({', '.join(need)})\n"
    lits = [_lit(v, "[", ",", "]") for v in BATTERY]
    calls = "\n".join(f"    print ({entry} {L}{_arg(form, v, ' ')})"
                      for v, L in zip(BATTERY, lits))
    src = f"{imports}{code}\nmain :: IO ()\nmain = do\n{calls}\n"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "M.hs").write_text(src)
        c = _run(["bash", "-c", f"{GHC} -O0 -v0 {GHC_LDFLAG} {td}/M.hs -o {td}/mh"],
                 timeout=180)
        if c is None:
            return "timeout", None
        if c.returncode != 0:
            return "compile_fail", None
        r = _run([f"{td}/mh"], timeout=40)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return "run_fail", None
    return "ok", _ints(r.stdout)


def run_ocaml(code, entry, form):
    lits = [_lit(v, "[", ";", "]") for v in BATTERY]
    calls = "\n".join(f'let () = Printf.printf "%d\\n" ({entry} {L}'
                      f"{_arg(form, v, ' ')})" for v, L in zip(BATTERY, lits))
    src = f"{code}\n{calls}\n"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.ml").write_text(src)
        c = _run(["bash", "-c", f"{OCAMLOPT} -w -a -o {td}/mo {td}/m.ml"], timeout=180)
        if c is None:
            return "timeout", None
        if c.returncode != 0:
            return "compile_fail", None
        r = _run([f"{td}/mo"], timeout=40)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return "run_fail", None
    return "ok", _ints(r.stdout)


# ---- rust / javascript: added 2026-08-14 for the confound-breaking languages ----
# rustc is the rustup stable toolchain under $FPL_TOOLCHAINS (its
# $ORIGIN-relative RPATH does not resolve under a uv-launched subprocess, so the lib
# dir must be exported inside a shell -- the via_bash workaround polyglot_runners.py
# documents).  node is the system interpreter; there is no pinned node in toolchains.
_RSBINS = sorted((T / "rustup" / "toolchains").glob("*/bin/rustc"))
RUSTC = str(_RSBINS[-1]) if _RSBINS else "rustc"
RUST_ENV = {**os.environ, "LD_LIBRARY_PATH": ":".join(
    [*(str(p.parent.parent / "lib") for p in _RSBINS[-1:]),
     os.environ.get("LD_LIBRARY_PATH", "")])}
NODE = os.environ.get("FPL_NODE") or __import__("shutil").which("node") or "/usr/bin/node"

_RS_MAIN = re.compile(r"(?m)^\s*fn\s+main\s*\(")


def _strip_block(code, m):
    """remove the brace-balanced item that `m` starts (used for a model-written main)."""
    i = code.find("{", m.end() - 1)
    if i < 0:
        return code[:m.start()]
    depth = 0
    for j in range(i, len(code)):
        depth += (code[j] == "{") - (code[j] == "}")
        if depth == 0:
            return code[:m.start()] + code[j + 1:]
    return code[:m.start()]


def _rs_borrows(code, entry):
    """Does the entry take its list BY REFERENCE (&[i32] / &Vec<i32>) or BY VALUE
    (Vec<i32>)?  `&vec![..]` coerces to &[T]; an owned Vec must be passed bare."""
    m = re.search(rf"\bfn\s+{re.escape(entry)}\s*(?:<[^>]*>)?\s*\(([^)]*)\)", code)
    if not m:
        return True
    first = m.group(1).split(",")[0]
    ty = first.split(":", 1)[1] if ":" in first else first
    return ty.strip().startswith("&")


def _raw_rust(code, entry, form):
    if _RS_MAIN.search(code):                      # our driver supplies main
        code = _strip_block(code, _RS_MAIN.search(code))
    for amp in (["&", ""] if _rs_borrows(code, entry) else ["", "&"]):
        lits = [(f"({v})" if isinstance(v, int) else
                 f"{amp}vec![" + ",".join(str(x) for x in v) + "]") for v in BATTERY]
        calls = "\n".join(f'    println!("{{}}", {entry}({L}{_arg(form, v, ", ")}));'
                          for v, L in zip(BATTERY, lits))
        src = f"{code}\nfn main() {{\n{calls}\n}}\n"
        with tempfile.TemporaryDirectory() as td:
            Path(td, "m.rs").write_text(src)
            c = _run([RUSTC, "-O", "-A", "warnings", "m.rs", "-o", "m"], cwd=td,
                     env=RUST_ENV, timeout=180, via_bash=True)
            if c is None:
                return "timeout", None
            if c.returncode != 0:
                continue                           # try the other calling convention
            r = _run([f"{td}/m"], timeout=40)
        if r is None:
            return "timeout", None
        if r.returncode != 0:
            return "run_fail", None
        return "ok", _ints(r.stdout)
    return "compile_fail", None


def run_javascript(code, entry, form):
    lits = [_lit(v, "[", ",", "]") for v in BATTERY]
    calls = "\n".join(f"console.log({entry}({L}{_arg(form, v, ', ')}));"
                      for v, L in zip(BATTERY, lits))
    src = code + "\n" + calls + "\n"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.js").write_text(src)
        r = _run([NODE, "m.js"], cwd=td, timeout=40)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return ("compile_fail" if "SyntaxError" in r.stderr else "run_fail"), None
    return "ok", _ints(r.stdout)


# ---- held-out languages, added 2026-08-15 -------------------------------------
# go and c are the targets of the out-of-sample transfer prediction. The scaffold
# handling mirrors polyglot_runners.run_go (strip the model's package/import/main,
# re-emit an import block containing only packages the remaining code references,
# because go refuses to compile an unused import).
GO = str(T / "go" / "bin" / "go")
GO_ENV = {**os.environ, "GOCACHE": str(T / "gocache"), "GOPATH": str(T / "gopath")}
_C_MAIN = re.compile(r"(?m)^\s*int\s+main\s*\(")


def run_go(code, entry, form):
    written = re.findall(r'(?m)^\s*import\s+"([^"]+)"\s*$', code)
    for blk in re.findall(r"(?m)^\s*import\s*\(([^)]*)\)", code):
        written += re.findall(r'"([^"]+)"', blk)
    code = re.sub(r"^package\s+\w+\s*$", "", code, flags=re.M)
    code = re.sub(r"^import\s*\([^)]*\)\s*$", "", code, flags=re.M)
    code = re.sub(r'^import\s+"[^"]*"\s*$', "", code, flags=re.M)
    code = re.sub(r"^func main\(\) \{.*?^\}", "", code, flags=re.M | re.S)
    imports = sorted({"fmt"} | {p for p in written
                                if re.search(rf'\b{re.escape(p.split("/")[-1])}\.',
                                             code)})
    imp = "import (\n" + "\n".join(f'\t"{p}"' for p in imports) + "\n)"
    lits = [(str(v) if isinstance(v, int)
             else "[]int{" + ",".join(str(x) for x in v) + "}") for v in BATTERY]
    calls = "\n".join(f"\tfmt.Println({entry}({L}{_arg(form, v, ', ')}))"
                      for v, L in zip(BATTERY, lits))
    src = f"package main\n\n{imp}\n\n{code}\nfunc main() {{\n{calls}\n}}\n"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "main.go").write_text(src)
        r = _run([GO, "run", "main.go"], env=GO_ENV, cwd=td, timeout=120)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return ("compile_fail" if "main.go:" in r.stderr else "run_fail"), None
    return "ok", _ints(r.stdout)


def _raw_c(code, entry, form):
    if _C_MAIN.search(code):                       # our driver supplies main
        code = _strip_block(code, _C_MAIN.search(code))
    decls, calls = [], []
    for i, v in enumerate(BATTERY):
        if isinstance(v, int):
            lit = str(v)
        else:
            decls.append(f"    int a{i}[] = {{{','.join(str(x) for x in v)}}};")
            lit = f"a{i}"
        calls.append(f'    printf("%lld\\n", (long long)'
                     f'{entry}({lit}{_arg(form, v, ", ")}));')
    src = ("#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n"
           + code + "\nint main(void){\n" + "\n".join(decls + calls)
           + "\n    return 0;\n}\n")
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.c").write_text(src)
        c = _run(["gcc", "-O1", "-w", "m.c", "-o", "m"], cwd=td, timeout=120)
        if c is None:
            return "timeout", None
        if c.returncode != 0:
            return "compile_fail", None
        r = _run([f"{td}/m"], timeout=40)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return "run_fail", None
    return "ok", _ints(r.stdout)



# ---- cpp and ruby: added for the canonical ten (csharp dropped) ---------------
# Same contract as the runners above: (code, entry, form) -> (status, outputs), reading
# the module-global BATTERY, so the FORMS sweep works for them too. Written natively
# rather than wrapping polyglot_runners, whose runners take an explicit test list and
# have no notion of `form` -- the extra accumulator/index argument a recursive rewrite
# introduces would have been unreachable.
def _raw_cpp(code, entry, form):
    if _C_MAIN.search(code):                       # our driver supplies main
        code = _strip_block(code, _C_MAIN.search(code))
    decls, calls = [], []
    for i, v in enumerate(BATTERY):
        if isinstance(v, int):
            lit = str(v)
        else:
            decls.append(f"    std::vector<int> a{i} = {{{','.join(str(x) for x in v)}}};")
            lit = f"a{i}"
        calls.append(f'    std::printf("%lld\\n", (long long)'
                     f'{entry}({lit}{_arg(form, v, ", ")}));')
    src = ("#include <cstdio>\n#include <vector>\n#include <string>\n"
           "#include <algorithm>\n#include <numeric>\nusing namespace std;\n"
           + code + "\nint main(){\n" + "\n".join(decls + calls)
           + "\n    return 0;\n}\n")
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.cpp").write_text(src)
        c = _run(["g++", "-std=c++17", "-O1", "-w", "m.cpp", "-o", "m"],
                 cwd=td, timeout=180)
        if c is None:
            return "timeout", None
        if c.returncode != 0:
            return "compile_fail", None
        r = _run([f"{td}/m"], timeout=40)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return "run_fail", None
    return "ok", _ints(r.stdout)


RUBY = str(T / "ruby-inst" / "bin" / "ruby")


def _raw_ruby(code, entry, form):
    calls = "\n".join(
        f"puts {entry}({_lit(v, '[', ',', ']')}{_arg(form, v, ', ')})"
        for v in BATTERY)
    src = code + "\n\n" + calls + "\n"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.rb").write_text(src)
        r = _run([RUBY, "m.rb"], cwd=td, timeout=40)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        # ruby reports a syntax error before running anything
        return ("compile_fail" if "syntax error" in r.stderr else "run_fail"), None
    return "ok", _ints(r.stdout)


# ---- extension languages, added 2026-09-04 (experiments/family_pooling_extension.py)
# fsharp / clojure / scala / elixir. Same contract as every runner above:
# (code, entry, form) -> (status, outputs) over the module-global BATTERY. They
# are reachable only for a language that is in languages.LANGS, i.e. only when
# FPL_EXTRA_LANGS is set, so the ten-language experiments never call them.
# Toolchains (all user-space, under $FPL_TOOLCHAINS): dotnet fsi
# (the .NET 8 SDK the C# runner uses), clojure-1.11.4 jars on the JDK,
# scala-cli 1.16 / Scala 3.8.4 on the JDK, Erlang/OTP 26.2.5 (Hex prebuilt) +
# Elixir 1.17.3.
import textwrap

CLJ_CP = ":".join(str(j) for j in sorted((T / "clojure").glob("*.jar")))
SCALA_CLI = str(T / "scala" / "scala-cli")
SCALA_ENV = {**LIB_ENV, "JAVA_HOME": str(T / "jdk-21.0.7+6"),      # LIB_ENV: libjli.so
             "COURSIER_CACHE": str(T / "scala" / "coursier-cache"),
             "SCALA_CLI_HOME": str(T / "scala" / "home"),
             "PATH": str(T / "jdk-21.0.7+6" / "bin") + ":" + os.environ.get("PATH", "")}
ELIXIR = str(T / "elixir" / "bin" / "elixir")
ELIXIR_ENV = {**os.environ,
              "PATH": ":".join([str(T / "erlang" / "OTP-26.2.5.4" / "bin"),
                                str(T / "elixir" / "bin"), os.environ.get("PATH", "")])}


def run_fsharp(code, entry, form):
    """dotnet fsi on a script: code, then one printfn per battery entry. F# is
    offside-sensitive, so the block is dedented; a model-written [<EntryPoint>]
    main must be the LAST declaration, so the attribute is dropped (the function
    stays, unused); a `namespace` cannot hold top-level statements, so it is
    dropped too; a nested `module M =` is opened so the entry is in scope."""
    code = textwrap.dedent(code)
    code = re.sub(r"(?m)^\s*\[<\s*EntryPoint\s*>\]\s*$", "", code)
    code = re.sub(r"(?m)^namespace\s+[\w.]+\s*$", "", code)
    opens = "".join(f"open {m}\n" for m in re.findall(r"(?m)^module\s+([\w.]+)\s*=\s*$", code))
    lits = [_lit(v, "[", ";", "]") for v in BATTERY]
    calls = "\n".join(f'printfn "%d" ({entry} {L}{_arg(form, v, " ")})'
                      for v, L in zip(BATTERY, lits))
    src = f"{code}\n{opens}{calls}\n"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.fsx").write_text(src)
        r = _run([DOTNET, "fsi", "--quiet", "--exec", "m.fsx"], cwd=td, env=DOTNET_ENV,
                 timeout=180)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return ("compile_fail" if ("error FS" in r.stdout or "error FS" in r.stderr)
                else "run_fail"), None
    return "ok", _ints(r.stdout)


def run_clojure(code, entry, form):
    """clojure.main on a script. `(7)` is a CALL in Clojure, so scalar battery
    entries are emitted bare; a vector literal for list entries."""
    lits = [(str(v) if isinstance(v, int) else "[" + " ".join(str(x) for x in v) + "]")
            for v in BATTERY]
    calls = "\n".join(f"(println ({entry} {L}{_arg(form, v, ' ')}))"
                      for v, L in zip(BATTERY, lits))
    src = f"{code}\n{calls}\n"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.clj").write_text(src)
        r = _run([JAVA, "-cp", CLJ_CP, "clojure.main", "m.clj"], cwd=td, env=LIB_ENV,
                 timeout=120, via_bash=True)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return ("compile_fail" if "Syntax error" in r.stderr else "run_fail"), None
    return "ok", _ints(r.stdout)


_SC_OBJ = re.compile(r"\A\s*(?:case\s+)?object\s+(\w+)\b")
_SC_CLS = re.compile(r"\A\s*(?:case\s+)?class\s+(\w+)\b")


def run_scala(code, entry, form):
    """scala-cli (Scala 3, no build server) on one file. The model's code is
    kept as written -- `object X { def solve }`, `object X:` (indentation
    syntax), a bare top-level def, or `class X` -- and the driver is a separate
    top-level `@main def run`, selected explicitly so a model-written main does
    not compete with it."""
    if m := _SC_OBJ.match(code):
        recv = f"{m.group(1)}."
    elif m := _SC_CLS.match(code):
        recv = f"(new {m.group(1)})."
    else:
        recv = ""
    lits = [_lit(v, "List(", ",", ")") for v in BATTERY]
    calls = "\n".join(f"  println({recv}{entry}({L}{_arg(form, v, ', ')}))"
                      for v, L in zip(BATTERY, lits))
    src = f"{code}\n\n@main def run(): Unit = {{\n{calls}\n}}\n"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "M.scala").write_text(src)
        r = _run([SCALA_CLI, "run", "--server=false", "--jvm", "system", "--scala", "3.8.4",
                  "--main-class", "run", "M.scala"], cwd=td, env=SCALA_ENV, timeout=300,
                 via_bash=True)          # the JDK's libjli.so resolves only via a shell
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return ("compile_fail" if ("[error]" in r.stderr or "[error]" in r.stdout
                                   or "Compilation failed" in r.stderr) else "run_fail"), None
    return "ok", _ints(r.stdout)


def run_elixir(code, entry, form):
    """`elixir m.exs`. Functions live in modules: the call is qualified with the
    module that defines the entry (the last `defmodule` opened before the
    entry's first `def`); bare `def`s are wrapped in one. Two schedulers, so a
    BEAM start does not spin up one thread per core for a six-number script."""
    mods = [(m.start(), m.group(1)) for m in re.finditer(r"defmodule\s+([\w.]+)", code)]
    if not mods:
        code, mods = f"defmodule Sol do\n{code}\nend", [(0, "Sol")]
    d = re.search(rf"\bdefp?\s+{re.escape(entry)}\b", code)
    pos = d.start() if d else len(code)
    mod = ([n for s0, n in mods if s0 < pos] or [mods[0][1]])[-1]
    lits = [_lit(v, "[", ",", "]") for v in BATTERY]
    calls = "\n".join(f"IO.puts({mod}.{entry}({L}{_arg(form, v, ', ')}))"
                      for v, L in zip(BATTERY, lits))
    src = f"{code}\n{calls}\n"
    with tempfile.TemporaryDirectory() as td:
        Path(td, "m.exs").write_text(src)
        r = _run([ELIXIR, "--erl", "+S 2:2", "m.exs"], cwd=td, env=ELIXIR_ENV, timeout=90)
    if r is None:
        return "timeout", None
    if r.returncode != 0:
        return ("compile_fail" if re.search(r"\*\* \((Compile|Syntax|TokenMissing|"
                                            r"MismatchedDelimiter)Error\)", r.stderr)
                else "run_fail"), None
    return "ok", _ints(r.stdout)


# the canonical ten (the four extension languages are reachable only through
# languages.LANGS, i.e. only when FPL_EXTRA_LANGS names them)

def run_c(code, entry, form):
    """as written first; on failure retry once with PRELUDE["c"] prepended (generous imports)"""
    st, got = _raw_c(code, entry, form)
    if st in ("compile_fail", "run_fail") and PRELUDE.get("c"):
        st2, got2 = _raw_c(PRELUDE["c"] + code, entry, form)
        if st2 == "ok":
            return st2, got2
    return st, got

def run_cpp(code, entry, form):
    """as written first; on failure retry once with PRELUDE["cpp"] prepended (generous imports)"""
    st, got = _raw_cpp(code, entry, form)
    if st in ("compile_fail", "run_fail") and PRELUDE.get("cpp"):
        st2, got2 = _raw_cpp(PRELUDE["cpp"] + code, entry, form)
        if st2 == "ok":
            return st2, got2
    return st, got

def run_java(code, entry, form):
    """as written first; on failure retry once with PRELUDE["java"] prepended (generous imports)"""
    st, got = _raw_java(code, entry, form)
    if st in ("compile_fail", "run_fail") and PRELUDE.get("java"):
        st2, got2 = _raw_java(PRELUDE["java"] + code, entry, form)
        if st2 == "ok":
            return st2, got2
    return st, got

def run_rust(code, entry, form):
    """as written first; on failure retry once with PRELUDE["rust"] prepended (generous imports)"""
    st, got = _raw_rust(code, entry, form)
    if st in ("compile_fail", "run_fail") and PRELUDE.get("rust"):
        st2, got2 = _raw_rust(PRELUDE["rust"] + code, entry, form)
        if st2 == "ok":
            return st2, got2
    return st, got

def run_ruby(code, entry, form):
    """as written first; on failure retry once with PRELUDE["ruby"] prepended (generous imports)"""
    st, got = _raw_ruby(code, entry, form)
    if st in ("compile_fail", "run_fail") and PRELUDE.get("ruby"):
        st2, got2 = _raw_ruby(PRELUDE["ruby"] + code, entry, form)
        if st2 == "ok":
            return st2, got2
    return st, got

def run_haskell_generous(code, entry, form):
    """as written (narrow import set) first; on failure retry with the wide one"""
    st, got = run_haskell(code, entry, form)
    if st in ("compile_fail", "run_fail"):
        st2, got2 = run_haskell(code, entry, form, _mods=HS_MODS_WIDE)
        if st2 == "ok":
            return st2, got2
    return st, got


RUNNERS = {"python": run_python, "javascript": run_javascript, "c": run_c,
           "cpp": run_cpp, "java": run_java, "go": run_go, "rust": run_rust,
           "ruby": run_ruby, "ocaml": run_ocaml, "haskell": run_haskell_generous,
           "fsharp": run_fsharp, "clojure": run_clojure, "scala": run_scala,
           "elixir": run_elixir}

_cache = {}


def execute(lang, code, entry, ti):
    """(status, task_ok, call_form). Tries a 1-arg call, then the three
    (list, index) conventions of FORMS."""
    key = (lang, code, entry, ti)
    if key in _cache:
        return _cache[key]
    global BATTERY
    saved, scalar = BATTERY, ti >= SCALAR_TI0
    if scalar:
        BATTERY = SBATTERY                    # the runners read it at call time
    try:
        want = [(SREF[ti - SCALAR_TI0] if scalar else REF[ti])(v) for v in BATTERY]
        best = ("no_entry", False, None)
        if code and entry:
            for form in FORMS:
                st, got = RUNNERS[lang](code, entry, form)
                if st == "ok":
                    best = ("ok", got == want, form)
                    if got == want:
                        break
                elif best[0] in ("no_entry", "compile_fail"):
                    best = (st, False, form)
    finally:
        BATTERY = saved
    _cache[key] = best
    return best


def load_prompt_rollouts(d, pat):
    """run_steer_promptvec_strategy.py: one json per cell. Unsteered baselines are
    deduplicated: a dose-extension run re-emits the same greedy baseline under its
    own tag."""
    seen = set()
    for p in sorted(Path(d).glob(pat)):
        r = json.load(open(p))
        if r["arm"] == "none":
            if (r["lang"], r["ti"]) in seen:
                continue
            seen.add((r["lang"], r["ti"]))
        yield {"file": p.name, "src": "prompt", "lang": r["lang"], "ti": r["ti"],
               "space": r["space"], "band": r["band"], "arm": r["arm"],
               "pole": r["pole"], "alpha": r["alpha"],
               "completion": r["completion"], "damage_nats": r.get("damage_nats"),
               "disp": r.get("disp") or {}}


def load_program_rollouts(files):
    """run_style_axis.py: one json per (style, band, dose pair), rollouts[lang] =
    four records with a baseline and four steered arms. donor_A = the java-donor
    ->recursion arm, the panel's headline; native_A = the target's own axis."""
    ARMS = {"baseline": ("none", "-", 0.0), "donor_A": ("donor", "rec", None),
            "donor_B": ("donor", "fold", None), "native_A": ("own", "rec", None),
            "native_B": ("own", "fold", None)}
    for f in files:
        s = json.load(open(f))
        aa, ab = s["alpha_a"], s["alpha_b"]
        dmg = s.get("damage_donor") or {}
        for lang, rs in s["rollouts"].items():
            for ti, rec in enumerate(rs):
                for k, (arm, pole, al) in ARMS.items():
                    if k not in rec:
                        continue
                    yield {"file": f"{Path(f).name}::{lang}[{ti}].{k}",
                           "src": "program", "lang": lang, "ti": ti,
                           "space": "std" if s.get("standardize") else "raw",
                           "band": s["band"], "arm": arm, "pole": pole,
                           "alpha": (al if al is not None
                                     else (aa if pole == "rec" else ab)),
                           "completion": rec[k],
                           "damage_nats": (dmg.get(lang) if arm != "none" else None),
                           "disp": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--glob", default="pvs_*.json")
    ap.add_argument("--program", nargs="*", default=[],
                    help="run_style_axis.py result json(s) to score with the SAME "
                         "three levels, for the head-to-head table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    if a.dir:
        rows += list(load_prompt_rollouts(a.dir, a.glob))
    if a.program:
        rows += list(load_program_rollouts(a.program))
    print(f"[scoring] {len(rows)} rollouts", flush=True)

    recs = []
    for i, r in enumerate(rows):
        lang, ti = r["lang"], r["ti"]
        code, info = extract_code(r["completion"], "auto")
        unit = first_unit(code, lang) if code else ""
        entry = detect_entry(lang, code, None) if code else None
        cls = classify_recursion(code, lang, entry) if (code and entry) else "other"
        degen = is_degenerate(unit) if unit else True
        st, task_ok, form = execute(lang, code, entry, ti)
        recs.append({
            "file": r["file"], "src": r["src"], "lang": lang, "ti": ti,
            "space": r["space"],
            "band": r["band"], "arm": r["arm"], "pole": r["pole"],
            "alpha": r["alpha"], "entry": entry, "cls": cls, "degenerate": degen,
            "clean_rec": bool(cls == "recursion" and not degen),
            "clean_fold": bool(cls == "fold" and not degen),
            "exec_status": st, "compiles": bool(st == "ok"), "task_ok": bool(task_ok),
            "call_form": form,
            "compliant": bool(code is not None and info["n_containers"] <= 1
                              and info["prose_chars"] == 0),
            "damage_nats": r.get("damage_nats"),
            "disp_frac": r["disp"].get("mean_frac"),
            "disp_abs": r["disp"].get("mean_abs")})
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(rows)}]", flush=True)

    out = Path(a.out)
    json.dump(recs, open(out, "w"), indent=1)
    print(f"[saved] {out}")
    report(recs)


LANG_ORDER = ("java", "csharp", "python", "haskell", "ocaml")


def report(recs):
    def flip(r):
        return r["clean_fold"] if r["pole"] == "fold" else r["clean_rec"]

    grp = defaultdict(list)
    for r in recs:
        grp[(r["src"], r["space"], r["band"], r["arm"], r["pole"], r["alpha"])].append(r)
    hdr = (f"{'src':<8} {'space':<5} {'band':<5} {'arm':<10} {'pole':<5} {'a':>4} "
           f"{'n':>3} {'clean':>7} {'compile':>8} {'taskOK':>8} {'clean&task':>11} "
           f"{'damage':>7} {'|dh|/|h|':>8}   per lang clean|clean&task "
           f"(java csharp python haskell ocaml)")
    print("\n" + hdr)
    print("-" * len(hdr))
    for k in sorted(grp, key=lambda k: (k[0], k[1], k[2], k[3] != "own",
                                        k[3], k[4], k[5])):
        g = grp[k]
        n = len(g)
        per = []
        for L in LANG_ORDER:
            gg = [r for r in g if r["lang"] == L]
            per.append(f"{sum(flip(r) for r in gg)}|"
                       f"{sum(flip(r) and r['task_ok'] for r in gg)}/{len(gg)}"
                       if gg else "-")
        dm = [r["damage_nats"] for r in g if r["damage_nats"] is not None]
        df = [r["disp_frac"] for r in g if r["disp_frac"] is not None]
        both = sum(flip(r) and r["task_ok"] for r in g)
        print(f"{k[0]:<8} {k[1]:<5} {k[2]:<5} {k[3]:<10} {k[4]:<5} {k[5]:>4} {n:>3} "
              f"{sum(flip(r) for r in g):>3}/{n:<3} "
              f"{sum(r['compiles'] for r in g):>4}/{n:<3} "
              f"{sum(r['task_ok'] for r in g):>4}/{n:<3} {both:>6}/{n:<4} "
              f"{(sum(dm) / len(dm) if dm else float('nan')):>7.2f} "
              f"{(sum(df) / len(df) if df else float('nan')):>8.3f}   "
              + " ".join(f"{x:>7}" for x in per))


if __name__ == "__main__":
    main()

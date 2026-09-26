"""The 31 tasks of the coding-ability bench (Appendix A.10 table): name, specification and the six battery inputs,
as Markdown (default) or LaTeX (--tex). Reads tasks_bench.BENCH, the same table run_bench.py prompts from.
    uv run factored_programming_languages/table_bench_tasks.py [--tex]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks_bench import BENCH                                             # noqa: E402

TEX = "--tex" in sys.argv
rows = [(name, desc, battery) for name, desc, _ref, battery in BENCH]
if TEX:
    print(r"\begin{tabular}{l p{8cm} l}" "\n" r"\toprule" "\n" r"task & specification & battery \\" "\n" r"\midrule")
    for name, desc, battery in rows:
        print(f"\\texttt{{{name.replace('_', r'_')}}} & {desc} & {', '.join(map(str, battery))} \\\\")
    print(r"\bottomrule" "\n" r"\end{tabular}")
else:
    print("| task | specification | battery |\n|---|---|---|")
    for name, desc, battery in rows:
        print(f"| `{name}` | {desc} | {', '.join(map(str, battery))} |")
print(f"\n{len(rows)} tasks", file=sys.stderr)

"""Number of demands, with WHICH demands averaged out.

run_grid varied the count along a fixed nested sequence, so rung k always
meant the same k clauses. Count and identity were confounded: rung 6 needed a
much longer suffix than rung 7, which is a fact about one particular set of
six, not about six. Here each size is sampled instead -- several different
random subsets of that size, drawn from a pool of eleven independent demands
-- so identity averages out and the count is what varies.

Independence matters for the sampling to be valid: every clause has to make
sense on its own, in any combination. That is why the cap constant is
BASE = 10 for digit extraction rather than "the cap in a constant", which
would be incoherent whenever the cap clause was not drawn.

Arms: one severed draw and one intact generation per (subset, suffix), so
every bar carries its own unsevered ceiling rather than borrowing one.

    uv run factored_programming_languages/run_subsets.py --shard 0 --nshards 8
"""
import argparse
import itertools
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import torch
torch.set_num_threads(6)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from elicit import extract_code, render_prompt                            # noqa: E402
from recipes import RECIPES                                               # noqa: E402
from tasks_compose import TAIL                                            # noqa: E402
from tasks_strategy import detect_strategy                                # noqa: E402
import prompt_memory as pm                                                # noqa: E402
import loader                                                             # noqa: E402

ANCHOR = "Write a function"
SENT_ORDER = ["exponent", "minimum", "base", "strategy", "typehint", "param",
              "docstring", "helper", "guard", "name", "comment"]
CLAUSES = {
    "exponent": ", each raised to the power 7",
    "minimum": ", capping the result at 23",
    "base": ", using a constant named BASE set to 10 for the digit extraction",
    "strategy": ", implemented recursively",
    "typehint": ", with type hints on the signature",
    "param": ", naming the parameter num",
    "docstring": ", including a one-line docstring",
    "helper": ", using an inner helper function named digitize",
    "guard": ", returning -1 if the input is negative",
    "name": ", saving the result to a variable named zork before returning it",
    "comment": ", ending the function with a comment that reads # done",
}
POOL = list(CLAUSES)                     # 11 independent demands
SUFFIXES = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]   # -- may be overridden below
TESTS = [0, 5, 9, 10, 47, 123, 909, 1234]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-31B-it")
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--max-new-tokens", type=int, default=700)
ap.add_argument("--subsets", type=int, default=10,
                help="random subsets sampled per size (sizes 0 and 11 have "
                     "only one possible subset)")
ap.add_argument("--suffixes", nargs="+", type=int, default=None,
                help="override the suffix ladder. The default tops out at 550, which is where the "
                     "k=10 cells only just begin to capture every demand (1/10 at 507, 2/10 at "
                     "556) -- the saturation point is past the edge of the default range "
                     "(authors, 2026-09-19).")
ap.add_argument("--sizes", nargs="+", type=int, default=None,
                help="only these subset sizes, instead of 0..len(pool)")
ap.add_argument("--demands", nargs="+", default=None, choices=list(CLAUSES),
                help="restrict the pool to these demands. The eleven are not equally carryable -- "
                     "`comment` survives the cut in 1%% of prompts and `helper` in 41%% against ~60%% "
                     "for the rest -- so a mixed pool measures the law through a denominator that "
                     "changes meaning with k. Passing the reliably-carried ones makes the pool "
                     "homogeneous (authors, 2026-09-18).")
ap.add_argument("--seed", type=int, default=17)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--sever", default="rand", choices=["rand", "mask"],
                help="rand = the description's KV rows replaced by norm-matched random vectors "
                     "(the original). mask = its COLUMNS cut from the attention mask during "
                     "generation, prefill intact -- the instrument the main figures use. Masking "
                     "is deterministic, so a severed cell needs one pass, not a draw per cell "
                     "(authors, 2026-09-18). Writes results/subsets10_<model>_mask.")
ap.add_argument("--kmax", type=int, default=None,
                help="largest subset size to run; the default keeps every size up to the pool. "
                     "--kmax 10 drops the all-demands row, which has exactly one possible subset.")
ap.add_argument("--arms", nargs="+", default=["severed", "intact"], choices=["severed", "intact"],
                help="the intact arm has no intervention, so under a new instrument it can be "
                     "reused from the earlier directory instead of regenerated")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

outdir = Path(a.outdir or ROOT / "results"
              / (f"subsets10_{a.model}" + ("" if a.sever == "rand" else f"_{a.sever}")))
outdir.mkdir(parents=True, exist_ok=True)

# the same subsets on every shard: seeded once, never re-drawn
rng = random.Random(a.seed)
if a.suffixes:
    SUFFIXES = list(a.suffixes)
    print(f"[suffix ladder] {SUFFIXES}", flush=True)
if a.demands:
    POOL = [d for d in POOL if d in set(a.demands)]
    print(f"[pool] restricted to {len(POOL)}: {POOL}", flush=True)
SUBSETS = []                             # (size, subset_id, tuple of clauses)
_SIZES = a.sizes if a.sizes else range(len(POOL) + 1)
for k in _SIZES:
    allc = list(itertools.combinations(POOL, k))
    rng.shuffle(allc)
    picks = allc[:a.subsets]
    for j, s in enumerate(picks):
        SUBSETS.append((k, j, tuple(s)))
print(f"[subsets] {len(SUBSETS)} across sizes 0..{len(POOL)}", flush=True)

NATURAL = [
    " This request comes from a routine batch of programming exercises.",
    " The batch is assembled automatically from a larger pool.",
    " Items in the pool are unrelated to one another.",
    " Each item is self-contained and carries its own description.",
    " There is no shared context between items.",
    " No information has been withheld from the description above.",
    " The wording has already been reviewed and is final.",
    " Items vary in length and in how many details they mention.",
    " Some are a single clause and others run to several.",
    " That variation reflects how the pool was built.",
    " It does not indicate difficulty or priority.",
    " Responses are collected and stored without further processing.",
    " They are not compared against one another.",
    " The order of items in the batch is arbitrary.",
    " Nothing depends on which item came before this one.",
    " The batch may be regenerated at any time.",
    " Identifiers attached to items are internal and carry no meaning.",
    " Timing information is recorded separately.",
    " No feedback is returned to the author of the item.",
    " The pool is refreshed periodically from the same sources.",
    " Duplicate items are removed before the batch is formed.",
    " Formatting of the description is normalised in advance.",
    " Whitespace and punctuation are standardised.",
    " Line breaks in the original are not preserved.",
    " The description is presented exactly as stored.",
    " Any trailing notes were stripped during preparation.",
    " This paragraph is boilerplate attached to every item.",
    " It is identical across the batch and carries no instruction.",
    " It exists to keep the surrounding material a fixed length.",
    " Nothing in it should be treated as part of the request.",
    " The request itself appears above this paragraph.",
    " Reading further will not add anything to it.",
    " The remainder of this note is administrative.",
    " Records are retained for a fixed period and then discarded.",
    " No further action follows from this text.",
    " Storage is handled by a separate system.",
    " That system has no access to the content of responses.",
    " Counts are aggregated once per day.",
    " The aggregation is not inspected by a person.",
    " Errors in preparation are logged and ignored.",
    " The pipeline has no retry behaviour.",
    " Items are dropped rather than repaired.",
    " This has no bearing on the request above.",
    " The note ends here.",
    " Nothing follows it.",
    " Batches are numbered in the order they are created.",
    " Numbering restarts when the pool is rebuilt.",
    " There is no relationship between numbering and content.",
    " Preparation runs on a fixed schedule.",
    " The schedule is unrelated to when responses are read.",
    " Older batches are archived without review.",
    " Archives are not consulted during preparation.",
    " Each item is prepared independently of the others.",
    " Preparation does not inspect the text of the request.",
    " No summary of the request is produced.",
    " No index of terms is maintained.",
    " Requests are stored as plain text.",
    " Encoding is fixed and does not vary between items.",
    " Character sets are normalised at ingestion.",
    " Ingestion happens once per item.",
    " Items are never modified after ingestion.",
    " Corrections are handled by creating a new item.",
    " The original is retained unchanged.",
    " This note is appended after ingestion.",
    " It is not part of the stored request.",
    " Its length is fixed by configuration.",
]

recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]
from transformers import AutoTokenizer                                    # noqa: E402
tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]),
                                    **recipe.get("tokenizer_kwargs", {}))
_L = loader.load(a.model, device=a.device, max_memory=a.max_memory, attn="sdpa")
model, dev, n_layers = _L.model, _L.device, _L.n_layers
gc = model.generation_config
stop = set()
for e in ([gc.eos_token_id] if isinstance(gc.eos_token_id, int)
          else (gc.eos_token_id or [])):
    stop.add(int(e))
if tok.eos_token_id is not None:
    stop.add(int(tok.eos_token_id))
stop_strings = recipe.get("stop_strings") or []


def rendered(sub, pad):
    s = ("Write a function in Python that returns the sum of the digits of a "
         "non-negative integer")
    for k in SENT_ORDER:
        if k in sub:
            s += CLAUSES[k]
    return render_prompt(tok, recipe, s + ". " + TAIL + pad)


def desc_rows(r):
    o = tok(r, add_special_tokens=False,
            return_offsets_mapping=True)["offset_mapping"]
    c0, c1 = r.find(ANCHOR), r.find(TAIL)
    if c0 < 0 or c1 <= c0:
        return None
    c0 += len(ANCHOR)
    rows = [i for i, (s_, e) in enumerate(o) if s_ < c1 and e > c0 and e > s_]
    return (min(rows), max(rows) + 1) if rows else None


def suffix_len(pad):
    r = rendered((), pad)
    return len(tok(r, add_special_tokens=False)["input_ids"]) - desc_rows(r)[1]


PADS, SUFLEN = {}, {}
for tgt in SUFFIXES:
    s, i = "", 0
    while tgt and suffix_len(s) < tgt:
        s += NATURAL[i % len(NATURAL)]
        i += 1
    PADS[tgt], SUFLEN[tgt] = s, suffix_len(s)
print("[suffix] " + "  ".join(f"{t}->{v}" for t, v in SUFLEN.items()), flush=True)

with torch.inference_mode():
    _probe = model(input_ids=tok(rendered((), ""), add_special_tokens=False,
                                 return_tensors="pt")["input_ids"].to(dev),
                   use_cache=True)
KINDS = pm.profile(_probe.past_key_values, n_layers)
del _probe
KV_L = [i for i in range(n_layers) if KINDS[i] == pm.KV]


def ablate(cache, win, gen):
    d0, d1 = win
    for i in KV_L:
        layer = cache.layers[i]
        for attr in ("keys", "values"):
            t = getattr(layer, attr, None)
            if t is None or t.shape[2] < d1:
                continue
            w = t[:, :, d0:d1, :]
            r = torch.randn(w.shape, generator=gen,
                            dtype=torch.float32).to(w.device, w.dtype)
            t[:, :, d0:d1, :] = r * (
                w.float().norm(dim=-1, keepdim=True).to(w.dtype)
                / (r.float().norm(dim=-1, keepdim=True).to(w.dtype) + 1e-6))


def _cycling(out, window=60, maxp=12):
    if len(out) < window:
        return False
    w = out[-window:]
    return any(all(w[i] == w[i - p] for i in range(p, window))
               for p in range(1, maxp + 1))


@torch.inference_mode()
def generate(r, gen):
    """gen is None for the intact arm; otherwise this is the severed arm"""
    ids = tok(r, add_special_tokens=False,
              return_tensors="pt")["input_ids"].to(dev)
    L = ids.shape[1]
    cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    amask = None
    if gen is not None and a.sever == "mask":
        d0, d1 = desc_rows(r)
        amask = torch.ones(1, L, dtype=torch.long, device=dev)
        amask[:, d0:min(d1, L - 1)] = 0        # prefill untouched: "cannot re-read"
    elif gen is not None:
        ablate(cache, desc_rows(r), gen)
    step = model(input_ids=ids[:, L - 1:], past_key_values=cache,
                 use_cache=True, attention_mask=amask)
    logits, cache = step.logits[:, -1], step.past_key_values
    out, degen = [], False
    for _n in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in stop:
            break
        out.append(nxt)
        if _n % 20 == 0 and _cycling(out):
            degen = True
            break
        txt = tok.decode(out[-24:], skip_special_tokens=True)
        if stop_strings and any(s_ in txt for s_ in stop_strings):
            break
        if amask is not None:
            amask = torch.cat([amask, torch.ones(1, 1, dtype=amask.dtype, device=dev)], 1)
        step = model(input_ids=torch.tensor([[nxt]], device=dev),
                     past_key_values=cache, use_cache=True, attention_mask=amask)
        logits, cache = step.logits[:, -1], step.past_key_values
    return tok.decode(out, skip_special_tokens=True), degen


def reference(sub):
    e = 7 if "exponent" in sub else 1
    cap = 23 if "minimum" in sub else None
    v = [sum(int(d) ** e for d in str(n)) for n in TESTS]
    return [min(x, cap) if cap is not None else x for x in v]


HARNESS = ("\nimport json\n"
           "print(json.dumps([solve(n) for n in %r]))\n" % (TESTS,))


def runs_correct(code, sub):
    if not code or "def solve" not in code:
        return False, "no solve"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code + HARNESS)
        path = fh.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=15)
    except subprocess.TimeoutExpired:
        os.unlink(path); return False, "timeout"
    os.unlink(path)
    if p.returncode != 0:
        return False, (p.stderr.strip().splitlines() or ["error"])[-1][:70]
    try:
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return False, "unparseable"
    return got == reference(sub), f"got {got[:3]}"


CHECK = {
    "exponent": lambda c: bool(re.search(r"\*\*\s*7|pow\s*\([^,]+,\s*7\s*\)", c)),
    "minimum": lambda c: bool(re.search(r"\b23\b", c)),
    "base": lambda c: bool(re.search(r"\bBASE\b\s*=\s*10\b", c)),
    "strategy": lambda c: detect_strategy(c, "python") == "rec",
    "typehint": lambda c: bool(re.search(r"def\s+solve\s*\([^)]*:\s*\w+", c)
                               and re.search(r"->\s*\w+", c)),
    "param": lambda c: bool(re.search(r"def\s+solve\s*\(\s*num\b", c)),
    "docstring": lambda c: bool(re.search(r'def\s+solve[^\n]*\n\s*("""|\'\'\')', c)),
    "helper": lambda c: bool(re.search(r"def\s+digitize\b", c)),
    "guard": lambda c: bool(re.search(r"return\s+-\s*1", c)),
    "name": lambda c: bool(re.search(r"\bzork\b\s*=(?!=)", c)),
    "comment": lambda c: bool(re.search(r"#\s*done\b", c, re.I)),
}

WORK = []
for k, j, sub in SUBSETS:
    if a.kmax is not None and k > a.kmax:
        continue
    for sf in SUFFIXES:
        for _arm in ("severed", "intact"):
            if _arm in a.arms:
                WORK.append((k, j, sub, sf, _arm))
mine = [w for i, w in enumerate(WORK) if i % a.nshards == a.shard]
print(f"[shard {a.shard}/{a.nshards}] {len(mine)} of {len(WORK)} cells",
      flush=True)

n_done = 0
for k, j, sub, sf, arm in mine:
    fp = outdir / f"k{k:02d}_j{j}_s{sf:03d}_{arm}.json"
    if fp.exists():
        continue
    r = rendered(sub, PADS[sf])
    gen = None
    if arm == "severed":
        gen = torch.Generator().manual_seed(50021 + k * 733 + j * 97 + sf)
    txt, degen = generate(r, gen)
    code, _ = extract_code(txt, "python")
    if not code:
        # extract_code refuses when a generation holds more than one fenced block, which is what a
        # rambling answer looks like: a solution, then "Wait, the ..." and a second attempt. Those
        # cells stored code="" and scored zero demands honored, concentrated in the low-suffix
        # corner -- which made a graded ramp look like a collapse (authors, 2026-09-19).
        _m = re.search(r"```(?:[a-zA-Z]*)?\s*\n(.*?)\n```", txt or "", re.S)
        code = _m.group(1).strip() if _m and "def solve" in _m.group(1) else ""
    code = code or ""
    hon = {c: CHECK[c](code) for c in sub}
    val, why = runs_correct(code, sub)
    fp.write_text(json.dumps(
        {"size": k, "subset_id": j, "clauses": list(sub),
         "suffix_target": sf, "suffix_tokens": SUFLEN[sf], "arm": arm,
         "honored": hon, "n_honored": sum(hon.values()),
         "all_honored": all(hon.values()) if hon else True,
         "valid": val, "why": why, "degenerate": degen,
         "code": code, "text": txt}, indent=1))
    n_done += 1
    print(f"[k{k:>2} j{j} suffix{sf:>4} {arm:>7}] "
          f"honored={sum(hon.values())}/{len(hon)} valid={val!s:>5}"
          f"{' CYCLE' if degen else ''}", flush=True)

print(f"[done] shard {a.shard}: {n_done} generations -> {outdir}", flush=True)

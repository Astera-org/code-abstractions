"""Tier signatures on a modern hybrid Qwen via MEAN-ABLATION, not mask cuts.

The attention-mask cut is blind on qwen3.6's 48 linear-attention layers, so
the tier battery is re-run with cache surgery after prefill instead:

  abl_kv    description rows of every KV layer replaced by the MEAN over the
            five task prompts at those positions (same language, aligned by
            padding): WHICH task the description names is destroyed, "a
            description of this shape lives here" survives.
  abl_full  abl_kv + the recurrent/conv states of the 48 linear layers
            replaced by their across-task mean (position-free states, so the
            whole state is averaged) -- without this the linear layers carry
            the description through untouched.

Tasks: digit_sum, sq_digit_sum (tier 1), happy4/happy89 (tier 2 parameter
probe), dsl1 (tier 3). Grading: behavioural reference match + stop-literal /
DSL markers, raw text stored.

    uv run factored_programming_languages/run_ablate_tiers.py --model qwen3.6-27b
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from elicit import extract_code, render_prompt                             # noqa: E402
from recipes import RECIPES                                                # noqa: E402
from tasks_function import BATTERY                                         # noqa: E402
from tasks_compose import TAIL                                             # noqa: E402
from languages import DISPLAY                                              # noqa: E402
import execute as S                                                        # noqa: E402
import prompt_memory as pm                                                 # noqa: E402
import loader                                                              # noqa: E402
from entry import find_entry, detect_language                          # noqa: E402

TASKS = {
    "digit_sum": "returns the sum of the digits of a non-negative integer",
    "sq_digit_sum": "returns the sum of the squares of the digits of a "
                    "non-negative integer",
    "happy4": "returns how many times the sum-of-squared-digits map must be "
              "applied to a positive integer before reaching 1 or 4",
    "happy89": "returns how many times the sum-of-squared-digits map must be "
               "applied to a positive integer before reaching 1 or 89",
    # 8 is NOT on the unhappy cycle: a faithful implementation loops forever
    # on most inputs. Probes compliance with a broken deviant spec (uncut) and
    # whether the ablation refill depends on the erased value (it cannot,
    # under slot-filling: prediction = 4, same as the happy89 recipient)
    "happy8": "returns how many times the sum-of-squared-digits map must be "
              "applied to a positive integer before reaching 1 or 8",
    "dsl1": "returns the sum of the digits of a non-negative integer, written "
            "ONLY in the following invented language and not in the language "
            "named earlier: functions are declared 'fn name(x) => ... end'; "
            "assignment is 'let x <- expr'; while-loops are 'as long as "
            "cond: ... end'; return is 'give expr'; integer division and "
            "modulo are the operators 'div' and 'mod'",
}
# the ablation MEAN is computed over UNRELATED distractor descriptions --
# using the tier tasks themselves leaks content (they are mutually related:
# dsl1 is digit-sum in disguise, happy4/happy89 are near-twins, and the first
# run showed exactly that: happy89 "regressed" to its twin happy4)
DISTRACT = {
    "d_collatz": "returns the number of steps of the map sending even numbers "
                 "to their half and odd numbers to three times themselves "
                 "plus one, applied to a positive integer until it reaches 1",
    "d_phi": "returns Euler's totient of a positive integer",
    "d_lpf": "returns the largest prime factor of an integer greater than 1",
    "d_gray": "returns the number of ones in the binary reflected Gray code "
              "of a non-negative integer",
    "d_mount": "returns how many integers from 1 to n have decimal digits "
               "that strictly increase and then strictly decrease, with both "
               "parts non-empty",
}
_ds = lambda n: [int(c) for c in str(n)]


def _happy(stop):
    def f(n):
        c = 0
        while n not in stop:
            n = sum(d * d for d in _ds(n))
            c += 1
        return c
    return f


REFV = {"digit_sum": [sum(_ds(x)) for x in BATTERY],
        "sq_digit_sum": [sum(d * d for d in _ds(x)) for x in BATTERY],
        "happy4": [_happy({1, 4})(x) for x in BATTERY],
        "happy89": [_happy({1, 89})(x) for x in BATTERY]}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen3.6-27b")
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--langs", nargs="+", default=["python", "java"])
ap.add_argument("--modes", nargs="+",
                default=["none", "abl_kv", "abl_full", "abl_rand"])
ap.add_argument("--bank", default="tiers", choices=["tiers", "flow", "bench"],
                help="flow = the standard flow-cut bank (4 canonical tasks). bench = the WHOLE benchmark "
                     "(tasks_bench.BENCH, 31 tasks), so the severed score is measured on exactly the task set the "
                     "coding-ability axis uses and the two are directly comparable -- the unsevered arm is then "
                     "results/bench_<model>, already on disk (authors, 2026-09-17).")
ap.add_argument("--max-new-tokens", type=int, default=380)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()
S.BATTERY = BATTERY
BATT = {}                                   # per-task battery (BENCH carries its own; the tiers bank shares one)
if a.bank in ("flow", "bench"):
    from tasks_bench import BENCH as _BENCH
    _by = {n: (d, r, b) for n, d, r, b in _BENCH}
    keep = (("digit_sum", "sq_digit_sum", "digit_prod", "even_digits") if a.bank == "flow" else tuple(_by))
    TASKS = {n: _by[n][0] for n in keep}
    BATT = {n: _by[n][2] for n in keep}
    REFV = {n: [_by[n][1](x) for x in BATT[n]] for n in TASKS}

recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]
from transformers import AutoTokenizer                                     # noqa: E402
tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]),
                                    **recipe.get("tokenizer_kwargs", {}))
ids_of = lambda t: tok(render_prompt(tok, recipe, t),                      # noqa: E731
                       add_special_tokens=False)["input_ids"]

raw = {(f, l): f"Write a function in {DISPLAY[l]} that {d}. {TAIL}"
       for f, d in {**TASKS, **DISTRACT}.items() for l in a.langs}
target = max(len(ids_of(u)) for u in raw.values())
FILL = ["\n", " ", ".", "-", "#", "*", ". ", "- ", "; "]


def _pad_one(u):
    base = len(ids_of(u))
    for c1 in FILL:
        if len(ids_of(c1 + u)) == base and c1 * 3 + u != u \
                and len(ids_of(c1 * 3 + u)) == base:
            continue
        for n in range(400):
            m = len(ids_of(c1 * n + u))
            if m == target:
                return c1 * n + u
            if abs(m - target) <= 2:
                for c2 in FILL:
                    if c2 != c1 and len(ids_of(c1 * n + c2 + u)) == target:
                        return c1 * n + c2 + u
            if m > target + 8:
                break
    return None


padded = {k: _pad_one(u) for k, u in raw.items()}
assert all(padded.values()), "padding failed"

# union description window per language (absolute rows; padding right-aligns
# the tail so the desc region is positionally comparable across tasks)
DESC_WIN, LANG_WIN = {}, {}
for l in a.langs:
    lo, hi = 10 ** 9, 0
    for f, dsc in {**TASKS, **DISTRACT}.items():
        u = padded[(f, l)]
        rend = render_prompt(tok, recipe, u)
        offs = tok(rend, add_special_tokens=False,
                   return_offsets_mapping=True)["offset_mapping"]
        core = f"Write a function in {DISPLAY[l]} that "
        c0 = rend.find(core)
        t1 = c0 + len(core)
        d1 = t1 + len(dsc)
        idx = [i for i, (s, e) in enumerate(offs) if s < d1 and e > t1 and e > s]
        lo, hi = min(lo, min(idx)), max(hi, max(idx) + 1)
    DESC_WIN[l] = (lo, hi)
    # the language span: the tokens of the language NAME inside "Write a function in <Lang> that ..."
    u0 = padded[(list(TASKS)[0], l)]
    rend0 = render_prompt(tok, recipe, u0)
    offs0 = tok(rend0, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    nm0 = DISPLAY[l]
    c_ = rend0.find(f"Write a function in {nm0}") + len("Write a function in ")
    idxl = [i for i, (s_, e) in enumerate(offs0) if s_ < c_ + len(nm0) and e > c_ and e > s_]
    LANG_WIN[l] = (min(idxl), max(idxl) + 1)
    print(f"[desc window] {l}: rows {lo}..{hi}   [lang window] rows {LANG_WIN[l][0]}..{LANG_WIN[l][1]}", flush=True)

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
enc_of = lambda t: tok(render_prompt(tok, recipe, t), return_tensors="pt",  # noqa: E731
                       add_special_tokens=False).to(dev)["input_ids"]

with torch.inference_mode():
    _probe = model(input_ids=enc_of(padded[("digit_sum", a.langs[0])]),
                   use_cache=True)
KINDS = pm.profile(_probe.past_key_values, n_layers)
del _probe
KV_L = [i for i in range(n_layers) if KINDS[i] == pm.KV]
RC_L = [i for i in range(n_layers) if KINDS[i] == pm.RECURRENT]
print(f"[cache] {pm.describe(KINDS)}", flush=True)

LFULL = target


def harvest(key):
    with torch.inference_mode():
        c = model(input_ids=enc_of(padded[key]), use_cache=True).past_key_values
    return {i: pm.harvest(c, i, KINDS[i], LFULL) for i in range(n_layers)
            if KINDS[i] is not None}


STORE = {k: harvest(k) for k in padded}
print(f"[harvest] {len(STORE)} states", flush=True)


@torch.inference_mode()
def generate(task, lam, mode):
    """mode: none | abl_kv | abl_full | abl_rand | abl_0 (noise / zeros instead of the distractor mean --
    instrument comparison, the authors) | abl_mask / abl_masklang / abl_maskboth (authors, 2026-09-17: sever by cutting
    the ATTENTION PATH to those columns rather than editing their content -- a 2D attention mask with zeros there,
    applied at prefill and at every decode step. Cleaner than content edits: a zeroed row still occupies a
    position and participates in the softmax, whereas a masked column is simply unreachable. transformers builds
    one mask per layer type, so this covers sliding-window layers too; verified no query row is left with nothing
    to attend to.)"""
    ids = enc_of(padded[(task, lam)])
    L = ids.shape[1]
    am = None
    if "mask" in mode:
        spans = []
        if mode.endswith("mask") or mode.endswith("maskboth"):
            spans.append(DESC_WIN[lam])
        if mode.endswith("masklang") or mode.endswith("maskboth"):
            spans.append(LANG_WIN[lam])
        am = torch.ones(1, L, dtype=torch.long, device=dev)
        for d0_, d1_ in spans:
            am[:, d0_:min(d1_, L)] = 0
    # PREFILL IS UNMASKED on purpose. The KV-ablation modes let the tail rows absorb the description and then
    # destroy those rows, so what they test is "cannot RE-READ". Masking during prefill instead means the tail
    # never saw the description at all -- a strictly stronger condition ("never read"), which on gemma-4-31B turns
    # 12/12 correct into 12/12 run_fail. Keep the two separable: abl_mask* mask only the decode steps;
    # abl_premask* mask the prefill too.  (authors, 2026-09-17)
    pre = am[:, :L - 1] if (am is not None and mode.startswith("abl_premask")) else None
    cache = model(input_ids=ids[:, :L - 1], use_cache=True,
                  **({"attention_mask": pre} if pre is not None else {})).past_key_values
    if "mask" in mode:
        step = model(input_ids=ids[:, L - 1:], past_key_values=cache, use_cache=True, attention_mask=am)
        logits, cache = step.logits[:, -1], step.past_key_values
        out = []
        for _ in range(a.max_new_tokens):
            nxt = int(logits.argmax(-1))
            if nxt in stop:
                break
            out.append(nxt)
            if stop_strings and any(t_ in tok.decode(out[-24:], skip_special_tokens=True) for t_ in stop_strings):
                break
            am = torch.cat([am, torch.ones(1, 1, dtype=am.dtype, device=am.device)], dim=1)
            o = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True,
                      attention_mask=am)
            logits, cache = o.logits[:, -1], o.past_key_values
        txt = tok.decode(out, skip_special_tokens=True)
        for t_ in stop_strings:
            txt = txt.split(t_)[0]
        return txt
    if mode != "none":
        d0, d1 = DESC_WIN[lam]
        d1 = min(d1, L - 1)
        gen = torch.Generator(device="cpu").manual_seed(0)
        for i in KV_L:
            layer = cache.layers[i]
            for nm, attr in (("keys", "keys"), ("values", "values")):
                t = getattr(layer, attr)
                if mode == "abl_0":
                    t[:, :, d0:d1, :] = 0
                    continue
                if mode == "abl_rand":
                    orig = t[:, :, d0:d1, :].float().cpu()
                    r = torch.randn(orig.shape, generator=gen)
                    r = r * (orig.norm(dim=-1, keepdim=True)
                             / (r.norm(dim=-1, keepdim=True) + 1e-8))
                    t[:, :, d0:d1, :] = r.to(t.device, t.dtype)
                    continue
                mu = torch.stack([STORE[(f, lam)][i][nm][:, d0:d1, :].float()
                                  for f in DISTRACT]).mean(0)
                t[:, :, d0:d1, :] = mu.to(t.device, t.dtype)
        if mode == "abl_full":
            for i in RC_L:
                layer = cache.layers[i]
                for nm, attr in (("conv", "conv_states"),
                                 ("recurrent", "recurrent_states")):
                    st = getattr(layer, attr, None)
                    if st is None:
                        continue
                    items = ([(f"{nm}.{k}", k) for k in st] if isinstance(st, dict)
                             else [(nm, None)])
                    for dk, k in items:
                        if isinstance(st, dict) and st[k] is None:
                            continue
                        mu = torch.stack([STORE[(f, lam)][i][dk].float()
                                          for f in DISTRACT
                                          if dk in STORE[(f, lam)][i]]).mean(0)
                        tgt = st[k] if isinstance(st, dict) else st
                        # harvest keeps the batch dim for recurrent state
                        if mu.dim() == tgt.dim() and mu.shape[0] == 1 \
                                and tgt.shape[0] != 1:
                            mu = mu[0]
                        if isinstance(st, dict):
                            st[k] = mu.to(tgt.device, tgt.dtype)
                        else:
                            setattr(layer, attr, mu.to(tgt.device, tgt.dtype))
    step = model(input_ids=ids[:, L - 1:], past_key_values=cache, use_cache=True)
    logits, cache = step.logits[:, -1], step.past_key_values
    out = []
    for _ in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in stop:
            break
        out.append(nxt)
        if stop_strings and any(t_ in tok.decode(out[-24:], skip_special_tokens=True)
                                for t_ in stop_strings):
            break
        o = model(input_ids=torch.tensor([[nxt]], device=dev),
                  past_key_values=cache, use_cache=True)
        logits, cache = o.logits[:, -1], o.past_key_values
    txt = tok.decode(out, skip_special_tokens=True)
    for t_ in stop_strings:
        txt = txt.split(t_)[0]
    return txt


_LIT = re.compile(r"(?<![\d.])(4|8|89)(?![\d.])")
_DSL = re.compile(r"\blet\b.*<-|\bgive\b|fn \w+\(.*\) =>")


def grade(txt, task, lam):
    code, _ = extract_code(txt, "auto")
    det = detect_language(code)
    entry = find_entry(det, code) if det != "unknown" else None
    got, st = None, "no_entry"
    if code and entry and det in S.RUNNERS:
        for form in S.FORMS:
            st, got = S.RUNNERS[det](code, entry, form)
            if st == "ok" and got is not None:
                break
    if BATT:
        # per-task batteries: another task's reference was computed on DIFFERENT inputs, so cross-task
        # attribution is meaningless here -- only ask whether this task's own reference is reproduced.
        impl = task if (got is not None and list(got) == REFV[task]) else None
    else:
        impl = next((k for k, v in REFV.items() if got is not None
                     and list(got) == v), None)
    return {"code": code, "text": txt, "lang_detect": det, "status": st,
            "outputs": got, "impl": impl,
            "lits": ",".join(sorted(set(_LIT.findall(code or "")))),
            "dsl": bool(_DSL.search(code or ""))}


outdir = Path(a.outdir or ROOT / "results" / f"ablatetiers_{a.model}")
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "meta.json").write_text(json.dumps(
    {"model": a.model, "tasks": list(TASKS), "distractors": list(DISTRACT), "desc_win": DESC_WIN,
     "kv_layers": len(KV_L), "recurrent_layers": len(RC_L)}, indent=1))
for lam in a.langs:
    for task in TASKS:
        for mode in a.modes:
            fp = outdir / f"at_{task}_{lam}_{mode}.json"
            if fp.exists():
                continue
            if BATT:                      # the bench bank carries a per-task battery; grade against that one
                S.BATTERY = BATT[task]
            txt = generate(task, lam, mode)
            g = grade(txt, task, lam)
            print(f"  {task:13s} {lam:7s} {mode:9s} impl={str(g['impl']):13s} "
                  f"{g['status']:12s} lits={g['lits'] or '-':6s} "
                  f"dsl={g['dsl']}", flush=True)
            fp.write_text(json.dumps(
                {"model": a.model, "task": task, "lang": lam, "mode": mode,
                 **g}, indent=1))
print(f"saved -> {outdir}", flush=True)

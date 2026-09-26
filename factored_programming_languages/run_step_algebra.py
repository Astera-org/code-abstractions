"""Does the activation space contain STEPS, or only task names?

A 2x3 function grid: bases {digit sum, digit product, largest digit} x
{plain, squared-digits}. The squared column's prod and max members are in NO
bank -- novel but compositional. Two readouts:

  geometry   are the three squaring-step vectors F(sq_X) - F(X) mutually
             aligned, far above cross-function baseline cosines?
  causal     install F(digit_prod) + Dsq_sum -- the squaring step estimated
             from the SUM pair -- into nofn(lam): does it produce the product
             of squared digits, a computation neither donor performs? Same
             for max. Ceiling: direct install of the described sq_digit_prod.
             Controls: base alone, norm-matched random step.

If the pointer is an atomic name, the arithmetic lands nowhere; if the space
encodes the steps, it lands on the novel composition.

    uv run factored_programming_languages/run_step_algebra.py
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from elicit import extract_code, render_prompt                             # noqa: E402
from recipes import RECIPES                                                # noqa: E402
from tasks_function import BATTERY                                         # noqa: E402
from tasks_compose import TAIL, nofn                                       # noqa: E402
from languages import DISPLAY                                              # noqa: E402
import execute as S                                                        # noqa: E402
import prompt_memory as pm                                                 # noqa: E402
import loader                                                              # noqa: E402
from entry import find_entry, detect_language                          # noqa: E402

GRID = {
    "sum": "returns the sum of the digits of a non-negative integer",
    "sq_sum": "returns the sum of the squares of the digits of a non-negative "
              "integer",
    "prod": "returns the product of the digits of a non-negative integer",
    "sq_prod": "returns the product of the squares of the digits of a "
               "non-negative integer",
    "max": "returns the largest digit of a non-negative integer",
    "sq_max": "returns the largest of the squares of the digits of a "
              "non-negative integer",
}
_ds = lambda n: [int(c) for c in str(n)]
REF = {
    "sum": lambda n: sum(_ds(n)),
    "sq_sum": lambda n: sum(d * d for d in _ds(n)),
    "prod": lambda n: math.prod(_ds(n)),
    "sq_prod": lambda n: math.prod(d * d for d in _ds(n)),
    "max": lambda n: max(_ds(n)),
    "sq_max": lambda n: max(d * d for d in _ds(n)),
}
WANTV = {k: [f(x) for x in BATTERY] for k, f in REF.items()}
assert len({tuple(v) for v in WANTV.values()}) == 6, "grid refs collide"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-31B-it")
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--langs", nargs="+", default=["python", "javascript", "java", "go"])
ap.add_argument("--max-new-tokens", type=int, default=320)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()
S.BATTERY = BATTERY

recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]
from transformers import AutoTokenizer                                     # noqa: E402
tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]),
                                    **recipe.get("tokenizer_kwargs", {}))
ids_of = lambda t: tok(render_prompt(tok, recipe, t),                      # noqa: E731
                       add_special_tokens=False)["input_ids"]

raw = {}
for lam in a.langs:
    raw[("nofn", lam)] = nofn(lam)
    for f, d in GRID.items():
        raw[(f, lam)] = f"Write a function in {DISPLAY[lam]} that {d}. {TAIL}"
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
seqs = [ids_of(u) for u in padded.values()]
K = 0
while K < target and len({s[-1 - K] for s in seqs}) == 1:
    K += 1
LFULL = target
W0, W1 = LFULL - K, LFULL
print(f"[site suffix] {W0}..{W1} of {LFULL} (K={K})", flush=True)

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
    _probe = model(input_ids=enc_of(padded[("nofn", a.langs[0])]), use_cache=True)
KINDS = pm.profile(_probe.past_key_values, n_layers)
del _probe
PATCHED = pm.select(KINDS, "both")


def harvest(key):
    with torch.inference_mode():
        c = model(input_ids=enc_of(padded[key]), use_cache=True).past_key_values
    return {i: pm.harvest(c, i, KINDS[i], LFULL) for i in PATCHED}


STORE = {k: harvest(k) for k in padded}
print(f"[harvest] {len(STORE)} states", flush=True)


def F(f, lam):
    """LOO absolute factor of grid function f, held-out language lam"""
    oth = [l for l in a.langs if l != lam]
    out = {}
    for i in PATCHED:
        names = STORE[(f, oth[0])][i].keys()
        out[i] = {n: torch.stack([STORE[(f, l)][i][n].float()
                                  - STORE[("nofn", l)][i][n].float()
                                  for l in oth]).mean(0) for n in names}
    return out


def add(d1, d2, s2=1.0):
    return {i: {n: d1[i][n] + s2 * d2[i][n] for n in d1[i]} for i in d1}


def sub(d1, d2):
    return add(d1, d2, -1.0)


def rand_matched(delta, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for i in delta:
        out[i] = {}
        for n, v in delta[i].items():
            r = torch.randn(v.shape, generator=g)
            w, rw = v[:, W0:W1, :], r[:, W0:W1, :]
            out[i][n] = r * (w.norm() / (rw.norm() + 1e-9))
    return out


# ---- geometry: are the squaring-step vectors aligned? -------------------------
def _flat(d):
    return torch.cat([d[i][n][:, W0:W1, :].reshape(-1)
                      for i in sorted(d) for n in sorted(d[i])])


def cos(x, y):
    return float((x @ y) / (x.norm() * y.norm() + 1e-9))


print("\n[geometry] squaring-step vector alignment (suffix window, per held-out lang)")
step_cos, base_cos = [], []
for lam in a.langs:
    D = {b: _flat(sub(F(f"sq_{b}", lam), F(b, lam)))
         for b in ("sum", "prod", "max")}
    X = {b: _flat(F(b, lam)) for b in ("sum", "prod", "max")}
    sc = [cos(D["sum"], D["prod"]), cos(D["sum"], D["max"]), cos(D["prod"], D["max"])]
    bc = [cos(D["sum"], X["prod"] - X["sum"]), cos(D["prod"], X["max"] - X["prod"]),
          cos(D["max"], X["sum"] - X["max"])]
    step_cos += sc
    base_cos += bc
    print(f"  {lam:11s} step-step cos {['%.3f' % c for c in sc]}   "
          f"step-vs-basechange {['%.3f' % c for c in bc]}")
print(f"  MEAN step-step {sum(step_cos)/len(step_cos):.3f}   "
      f"baseline {sum(base_cos)/len(base_cos):.3f}")


def grade(txt):
    code, _ = extract_code(txt, "auto")
    det = detect_language(code)
    entry = find_entry(det, code) if det != "unknown" else None
    got, st = None, "no_entry"
    if code and entry and det in S.RUNNERS:
        for form in S.FORMS:
            st, got = S.RUNNERS[det](code, entry, form)
            if st == "ok" and got is not None:
                break
    impl = next((k for k, v in WANTV.items() if got is not None and list(got) == v),
                None)
    return {"code": code, "text": txt, "lang_detect": det, "status": st,
            "outputs": got, "impl": impl}


@torch.inference_mode()
def generate(rec_key, delta):
    ids = enc_of(padded[rec_key])
    L = ids.shape[1]
    cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    hi = min(W1, L - 1)
    if delta is not None:
        for i in PATCHED:
            if KINDS[i] != pm.KV:
                continue
            pm.patch(cache, i, pm.KV, delta[i], 1.0, lo=W0, hi=hi, s0=W0, s1=hi)
    step = model(input_ids=ids[:, L - 1:], past_key_values=cache, use_cache=True)
    if delta is not None and W1 >= L - 1:
        for i in PATCHED:
            if KINDS[i] != pm.KV:
                continue
            pm.patch(cache, i, pm.KV, delta[i], 1.0, lo=L - 1, hi=L,
                     s0=L - 1, s1=L)
    logits = step.logits[:, -1]
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


outdir = Path(a.outdir or ROOT / "results" / f"stepalgebra_{a.model}")
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "meta.json").write_text(json.dumps(
    {"model": a.model, "grid": GRID, "wantv": WANTV, "K": K,
     "geometry": {"step_step": step_cos, "baseline": base_cos}}, indent=1))

print("\n[causal] step transplant into nofn(lam)")
for lam in a.langs:
    Dsq_sum = sub(F("sq_sum", lam), F("sum", lam))
    ARMS = [
        ("none", None, None),
        ("base_prod", F("prod", lam), "prod"),
        ("comp_prod", add(F("prod", lam), Dsq_sum), "sq_prod"),
        ("comp_max", add(F("max", lam), Dsq_sum), "sq_max"),
        ("ceil_sqprod", F("sq_prod", lam), "sq_prod"),
        ("rand_prod", add(F("prod", lam), rand_matched(Dsq_sum,
                                                       a.langs.index(lam))),
         "prod"),
    ]
    for name, delta, want in ARMS:
        fp = outdir / f"sa_{name}_{lam}.json"
        if fp.exists():
            continue
        txt = generate(("nofn", lam), delta)
        g = grade(txt)
        hit = g["impl"] == want if want else g["impl"] is None
        fp.write_text(json.dumps(
            {"model": a.model, "arm": name, "lang": lam, "want": want,
             "hit": hit, **g}, indent=1))
        print(f"  [{lam}] {name:12s} impl={str(g['impl']):8s} want={str(want):8s} "
              f"{g['status']:10s}" + ("   HIT" if hit and want else ""), flush=True)
# ---- modifier dose sweep -------------------------------------------------------
# alpha=1 alone cannot distinguish "base-dependent attachment" from
# under-dosing (the LOO-CAA lesson). Sweep m in F(base) + m*Dsq for both bases
# and all languages; the already-hit cells double as an over-dose check.
print("\n[sweep] modifier dose m", flush=True)
for lam in a.langs:
    Dsq_sum = sub(F("sq_sum", lam), F("sum", lam))
    for base, want in (("max", "sq_max"), ("prod", "sq_prod")):
        for m in (0.5, 1.5, 2.0, 3.0):
            fp = outdir / f"sa_sweep_{base}_m{m}_{lam}.json"
            if fp.exists():
                continue
            txt = generate(("nofn", lam), add(F(base, lam), Dsq_sum, m))
            g = grade(txt)
            fp.write_text(json.dumps(
                {"model": a.model, "arm": f"sweep_{base}", "m": m, "lang": lam,
                 "want": want, "hit": g["impl"] == want, **g}, indent=1))
            print(f"  [{lam}] {base}+{m}*Dsq  impl={str(g['impl']):8s} "
                  f"{g['status']:10s}" + ("   HIT" if g["impl"] == want else ""),
                  flush=True)
print(f"saved -> {outdir}", flush=True)

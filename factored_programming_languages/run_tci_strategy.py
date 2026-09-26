"""Strategy (recursion vs iteration) as a third factor, in the SEVERED regime.

The strategy analogue of run_tci_severed.py, replacing run_strategy.py's
cue-span design. What changed, and why:

  site       the ALIGNED SUFFIX, not the cue span. A span patch is
             positional, so run_strategy.py had to exclude every recurrent
             layer -- on qwen3.6-27b that is 48 of 64 layers, i.e. the
             model was never actually intervened on. The suffix site plus
             a whole-state recurrent patch reaches the same interface the
             function and language factors are measured at.
  severed    the recipient's ENTIRE specification (language, function
             description AND strategy cue) is randomly ablated post-prefill,
             so the words channel specifies nothing and an observed strategy
             can only have come from the injected state.
  channels   kv for dense models; kv, kvrec and rec for hybrids.
  grading    executed under the REQUESTED language's runner, never by
             detection; the strategy read is the structural checker
             tasks_strategy.detect_strategy on the same code.

  INSTALL    recipient nostrat(lam, A)   ("... that <desc>. <tail>", no cue)
             delta  LOO mean over f' != A, lam' != lam, cue' != cue of
                    state(strat(lam', f', s, cue')) - state(nostrat(lam', f'))
             hit    writes A, in lam, using strategy s
  TRANSPORT  recipient strat(lam, A, s_other, cue)  (cue asks the OTHER
             strategy; after ablation only its compiled store carries it)
             delta  LOO mean of state(strat(lam', f', s, cue'))
                                 - state(strat(lam', f', s_other, cue'))
             hit    flips to strategy s, keeps A and lam

Arms per cell: none (untouched severed recipient), delta (the held-out
pooled factor), exact (this cell's own contrast, the ceiling), rand
(norm-matched random delta), prompted (no intervention, words intact --
just ask for s).

    uv run factored_programming_languages/run_tci_strategy.py \
        --model gemma-4-31B-it --device cuda:0 --ops install transport
"""
import argparse
import json
import sys
from pathlib import Path

import os
# 8 jobs x 104 default torch threads thrashes a 104-core box (load 428,
# GPUs idle). Cap threads: the per-cell delta math is small.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import torch
torch.set_num_threads(6)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from elicit import extract_code, render_prompt                            # noqa: E402
from recipes import RECIPES                                               # noqa: E402
from tasks_function import LANGS, FKEYS, BATTERY, which_function          # noqa: E402
from tasks_strategy import (CUES, STRATEGIES, CUE_FAMILIES, strat, nostrat,     # noqa: E402
                            detect_strategy)
from tasks_compose import TAIL                                            # noqa: E402
import execute as S                                                       # noqa: E402
import prompt_memory as pm                                                # noqa: E402
from scoring import run_forms, Deferred                                   # noqa: E402
import loader                                                             # noqa: E402
from entry import find_entry                                            # noqa: E402

S.BATTERY = BATTERY          # mandatory rebind (memory: execute-battery-rebind)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--ops", nargs="+", default=["install", "transport"],
                choices=["install", "transport"])
ap.add_argument("--langs", nargs="+", default=["javascript", "go", "java", "c"])
ap.add_argument("--funcs", nargs="+", default=FKEYS)
ap.add_argument("--strategies", nargs="+", default=STRATEGIES,
                help="target strategies to install/transport INTO the cell")
ap.add_argument("--cue", default="a", help="recipient cue family; donors use "
                                           "the other families only")
ap.add_argument("--xlang-donors", nargs="+", default=None,
                help="add one arm per donor LANGUAGE: the delta is built from "
                     "that language's prompts alone (held-out functions and "
                     "cue families), then injected into the recipient's "
                     "language. haskell -> java is the cross-idiom case.")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--site", default="suffix", choices=["suffix", "all", "online"],
                help="rows patched: the aligned suffix (the compiled store; default) "
                     "or ALL rows of the padded prompt incl. the description -- the "
                     "interface a re-reading model uses. Pair `all` with "
                     "--ablate-desc none to steel-man a re-reader: its words stay, "
                     "and the pooled delta is written over them.")
ap.add_argument("--online-mode", default="live", choices=["live", "vector"],
                help="--site online: live = per-token counterfactual; vector = pooled generation vector, frozen (online_site.py)")
ap.add_argument("--online-layers", default="band", choices=["band", "all"],
                help="--site online: steer the certified band (models.band) or every layer")
ap.add_argument("--ablate-desc", default="rand", choices=["rand", "zero", "none", "mask", "bias"],
                help="rand/zero overwrite the severed rows; bias leaves the cache intact and "
                     "subtracts --desc-bias from those rows' attention logits at the launch step "
                     "and every decode step (inf = masked out)")
ap.add_argument("--desc-bias", type=float, default=0.0)
ap.add_argument("--ablate-span", default="spec", choices=["spec", "cue"],
                help="rows the severing covers on a strategy prompt: spec = language, description "
                     "AND cue; cue = the strategy cue sentence only (the words that conflict with "
                     "the injected strategy), language and description stay readable")
ap.add_argument("--rec-sever", default="none", choices=["none", "skip"],
                help="hybrids: also sever the RECURRENT channel of the recipient by "
                     "replaying the recorded rule inputs with the specification steps "
                     "skipped (k=v=0, decay only) -- the counterpart of KV zeros")
ap.add_argument("--rec-delta", default="state", choices=["state", "suffix"],
                help="hybrids: recurrent part of every delta from FINAL states (published "
                     "kvrec) or from skip-replayed states (suffix-only, 2026-09-05)")
ap.add_argument("--donor-pool", default="all", choices=["all", "family", "xfamily"],
                help="which languages the pooled (held-out) factor is averaged over "
                     "for a recipient: every other language (the ten/nine-language "
                     "design), the other languages of its own paradigm family, or the "
                     "languages of the other family (a side experiment; unused in the paper)")
ap.add_argument("--families", default=None,
                help="paradigm assignment 'imperative:c,cpp,...;functional:haskell,...'; "
                     "default languages.PARADIGM. Written into meta.json.")
ap.add_argument("--pool-always", nargs="*", default=[],
                help="languages that join every pool whatever their family (never the "
                     "recipient itself), e.g. python")
ap.add_argument("--recurrent", action="store_true",
                help="hybrids: add kvrec / rec arms")
ap.add_argument("--arms", nargs="+",
                default=["none", "delta", "exact", "rand", "single",
                         "prompted", "xlang"])
ap.add_argument("--limit", type=int, default=None,
                help="smoke test: stop after this many generations")
ap.add_argument("--donor-langs", nargs="+", default=None,
                help="languages that join every pooled donor set but are never "
                     "targets (python: the carrier default). Makes the pool "
                     "match run_tci_severed.py, which pools over all of LANGS.")
ap.add_argument("--single-donor-lang", default=None,
                help="force the SINGLE-donor arm to take its donor from this language instead of "
                     "whatever happens to be first in the donor list. Sweeping it gives a donor x "
                     "recipient grid; the diagonal is empty by construction, since a recipient's "
                     "own language is never in its donor pool (authors, 2026-09-18).")
ap.add_argument("--channels", nargs="+", default=None,
                choices=["kv", "kvrec", "rec"],
                help="intervention channels to run (default: kv, plus kvrec "
                     "and rec on hybrids with --recurrent). The channel-free "
                     "none/prompted arms are always written, tagged kv.")
ap.add_argument("--max-new-tokens", type=int, default=340)
ap.add_argument("--strip-think", action="store_true")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]
from transformers import AutoTokenizer                                    # noqa: E402
tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]),
                                    **recipe.get("tokenizer_kwargs", {}))
ids_of = lambda t: tok(render_prompt(tok, recipe, t),                     # noqa: E731
                       add_special_tokens=False)["input_ids"]

LANGS_T = [l for l in a.langs if l in LANGS]
XLANG = [l for l in (a.xlang_donors or []) if l in LANGS]
DONOR = [l for l in (a.donor_langs or []) if l in LANGS]
LANGS_P = list(dict.fromkeys(LANGS_T + XLANG + DONOR))   # prompt set covers all
from languages import parse_families, donor_pool                          # noqa: E402
FAMS = parse_families(a.families)


def POOL(lam):
    """the donor languages of the pooled S factor for recipient lam"""
    oth = donor_pool(lam, a.donor_pool, FAMS, LANGS_P, a.pool_always)
    assert oth, f"empty donor pool for {lam} under --donor-pool {a.donor_pool}"
    return oth


print(f"[pool] targets {LANGS_T}; donor-only {DONOR}; xlang {XLANG}", flush=True)

# ---- prompt set: every (f, lam, strategy, cue) + the cue-empty variants -----
raw = {}
for f in a.funcs:
    for l in LANGS_P:
        raw[("nos", f, l, "-")] = nostrat(l, f)
        for s in STRATEGIES:
            for c in CUE_FAMILIES:
                raw[("s", f, l, s, c)] = strat(l, f, s, c)

target = max(len(ids_of(u)) for u in raw.values())
FILL = ["\n", " ", ".", "-", "#", "*", ". ", "- ", "; "]


def _pad_one(u):
    base = len(ids_of(u))
    for c1 in FILL:
        if len(ids_of(c1 + u)) == base and c1 * 3 + u != u \
                and len(ids_of(c1 * 3 + u)) == base:
            continue
        for n in range(600):
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
assert all(padded.values()), \
    f"padding failed: {[k for k, v in padded.items() if not v]}"
seqs = [ids_of(u) for u in padded.values()]
K = 0
while K < min(len(s_) for s_ in seqs) and len({s_[-1 - K] for s_ in seqs}) == 1:
    K += 1
PLEN = {k: len(s) for k, s in zip(padded, seqs)}
LFULL = target
W0, W1 = LFULL - K, LFULL
if a.site == "all":                 # every row, description included; the
    W0, K = 0, LFULL                    # harvest already keeps all LFULL rows
assert K >= 10, f"shared suffix only {K} tokens"
print(f"[site] {len(padded)} prompts, suffix rows {W0}..{W1} of {LFULL} "
      f"(K={K})", flush=True)

# ---- model ------------------------------------------------------------------
_L = loader.load(a.model, device=a.device, max_memory=a.max_memory, attn="sdpa")
from online_site import OnlineSite                                   # noqa: E402
ONLINE = None
_ONL = {"printed": False}
if a.site == "online":                  # residual-stream site (see online_site.py); the K/V patch is skipped
    import models as _cast
    _rows = (1, len(_L.layers)) if a.online_layers == "all" else _cast.band(a.model)   # all layers beat the band 23/24 (steering_everywhere.md)
    ONLINE = OnlineSite(_L.layers, _rows, mode=a.online_mode)
    print(f"[online] residual site on layers {ONLINE.band[0]}..{ONLINE.band[-1]} ({len(ONLINE.band)} layers), alpha {a.alpha}", flush=True)
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

# ---- attention-logit bias on the severed rows (--ablate-desc bias), ported
# from run_tci_severed.py: pre-hook on every text self-attention module ----
BIAS = {"on": False, "beta": 0.0, "cols": None, "fired": 0, "shape": None}


def _bias_hook(module, args, kwargs):
    if not BIAS["on"] or BIAS["cols"] is None:
        return None
    hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
    if hs.shape[1] != 1:                       # prefill: leave the mask alone
        return None
    m = kwargs.get("attention_mask")
    if m is None:
        pkv = kwargs.get("past_key_values")
        n = pkv.layers[module.layer_idx].keys.shape[2] + 1
        sw = getattr(module.config, "sliding_window", None)
        assert not (getattr(module, "layer_type", None) == "sliding_attention" and sw and n > sw), \
            "sliding layer beyond its window with no mask from the model"
        m = torch.zeros(1, 1, 1, n, dtype=hs.dtype, device=hs.device)
    elif m.dtype == torch.bool:
        m = torch.zeros(m.shape, dtype=hs.dtype, device=m.device).masked_fill(~m, torch.finfo(hs.dtype).min)
    else:
        m = m.to(hs.dtype).clone()
    c0, c1 = BIAS["cols"]
    if BIAS["beta"] == float("inf"):
        m[..., c0:c1] = torch.finfo(hs.dtype).min
    else:
        m[..., c0:c1] = m[..., c0:c1] - BIAS["beta"]
    BIAS["fired"] += 1
    BIAS["shape"] = tuple(m.shape)
    kwargs["attention_mask"] = m
    return args, kwargs


_ATTN = [(n, m) for n, m in model.named_modules() if n.endswith(("self_attn", "attention")) and hasattr(m, "layer_idx")]
if any("language_model" in n for n, _ in _ATTN):
    _ATTN = [(n, m) for n, m in _ATTN if "language_model" in n]
_ATTN = [m for _, m in _ATTN]
for _m in _ATTN:
    _m.register_forward_pre_hook(_bias_hook, with_kwargs=True)
_PLUMB = {}

_any = next(iter(padded))
with torch.inference_mode():
    _probe = model(input_ids=enc_of(padded[_any]), use_cache=True)
KINDS = pm.profile(_probe.past_key_values, n_layers)
del _probe
PATCHED = pm.select(KINDS, "both")
KV_L = [i for i in PATCHED if KINDS[i] == pm.KV]
RC_L = [i for i in range(n_layers) if KINDS[i] != pm.KV]
print(f"[cache] {pm.describe(KINDS)} | kv {len(KV_L)} rec {len(RC_L)}",
      flush=True)
print(f"[bias] hook on {len(_ATTN)} text self-attention modules (active only under --ablate-desc bias)", flush=True)
if a.ablate_desc == "bias":
    assert len(_ATTN) == len(KV_L), f"hooked {len(_ATTN)} attention modules but {len(KV_L)} KV layers"
HYBRID = bool(RC_L) and a.recurrent

# ---- recurrent replay (ported from run_ablate_both.py / run_tci_severed.py) ----
REC = {"on": False, "inputs": {}}
if RC_L and (a.rec_sever == "skip" or a.rec_delta == "suffix"):
    LIN = {i: getattr(layer, "linear_attn", None) for i, layer in enumerate(_L.layers)}
    LIN = {i: m for i, m in LIN.items() if m is not None}
    assert sorted(LIN) == RC_L, (sorted(LIN)[:5], RC_L[:5])

    def _record(i, m):
        orig = m.chunk_gated_delta_rule

        def fn(query, key, value, g, beta, **kw):
            if REC["on"]:
                REC["inputs"][i] = {"query": query.detach(), "key": key.detach(),
                                    "value": value.detach(), "g": g.detach(),
                                    "beta": beta.detach()}
            return orig(query, key, value, g=g, beta=beta, **kw)
        m.chunk_gated_delta_rule = fn

    for _i, _m in LIN.items():
        _record(_i, _m)
    import importlib
    _mt = model.config.model_type.replace("_text", "")
    _rec_rule = importlib.import_module(
        f"transformers.models.{_mt}.modeling_{_mt}").torch_recurrent_gated_delta_rule
    REC["on"], REC["inputs"] = True, {}
    with torch.inference_mode():
        _c0 = model(input_ids=enc_of(padded[next(iter(padded))])[:, :-1],
                    use_cache=True).past_key_values
    REC["on"] = False
    _worst = 0.0
    for _i in RC_L:
        _S0 = _c0.layers[_i].recurrent_states.float()
        _q, _k, _v, _g, _b = (REC["inputs"][_i][x] for x in ("query", "key", "value", "g", "beta"))
        _, _S1 = _rec_rule(_q, _k, _v, _g, _b, initial_state=None, output_final_state=True,
                           use_qk_l2norm_in_kernel=True)
        _worst = max(_worst, float((_S1.to(_S0) - _S0).norm() / (_S0.norm() + 1e-8)))
    print(f"[replay check] {len(RC_L)} layers, worst relative error {_worst:.2e}", flush=True)
    assert _worst < 2e-2, "replay does not reproduce the cached recurrent state"
    del _c0


@torch.inference_mode()
def replay_skip(inp, w0=None, w1=None):
    q, k, v, g, b = (inp[x] for x in ("query", "key", "value", "g", "beta"))
    if w0 is not None and w1 is not None and w1 > w0:
        k, v = k.clone(), v.clone()
        k[:, w0:w1] = 0
        v[:, w0:w1] = 0
    _, St = _rec_rule(q, k, v, g, b, initial_state=None, output_final_state=True,
                      use_qk_l2norm_in_kernel=True)
    return St


ANCHOR = "Write a function"


def _tail_start(offsets, c1):
    """the spec window's right edge: the start of the token that owns character c1 (TAIL's first character). That
    token carries the space before "Name", so it belongs to the shared tail, not to the specification."""
    return next((s for s, e in offsets if s <= c1 < e), c1)


def desc_rows(key):
    """rows of the recipient's ENTIRE specification: language, function
    description and -- for the strategy prompts -- the cue, i.e. everything
    between the frame and the shared tail."""
    r = render_prompt(tok, recipe, padded[key])
    o = tok(r, add_special_tokens=False,
            return_offsets_mapping=True)["offset_mapping"]
    c0, c1 = r.find(ANCHOR), r.find(TAIL)
    if c0 < 0 or c1 <= c0:
        return None
    # right edge as in the published runs: the token that owns the tail's first character (it carries the space before
    # "Name") is admitted, so one tail token is severed with the specification; _tail_start would exclude it.
    c0 += len(ANCHOR)
    if a.ablate_span == "cue" and key[0] == "s":      # the cue sentence alone
        cue = CUES[key[3]][key[4]]
        c0 = r.find(cue)
        assert c0 > 0, cue
        c1 = c0 + len(cue) + 1                         # + its period
    rows = [i for i, (s_, e) in enumerate(o) if s_ < c1 and e > c0 and e > s_]
    return (min(rows), max(rows) + 1) if rows else None



def collect_rec(c):
    """recurrent matrix + convolution buffer of every linear layer, float32 on CPU"""
    rec = {}
    for i in RC_L:
        layer = c.layers[i]
        d = {}
        for nm, attr in (("conv", "conv_states"), ("recurrent", "recurrent_states")):
            st = getattr(layer, attr, None)
            if st is None:
                continue
            if isinstance(st, dict):
                for k, v in st.items():
                    if v is not None:
                        d[f"{nm}.{k}"] = v.detach().float().cpu().clone()
            else:
                d[nm] = st.detach().float().cpu().clone()
        if d:
            rec[i] = d
    return rec


def harvest(key):
    """KV rows of the whole prompt (LFULL rows, launch slot included) and the recurrent
    states as they stand at the RECIPIENT's patch time -- after L-1 tokens, before the
    launch token. Harvesting after the launch token (the earlier convention) left every
    recurrent contrast one step ahead of the state it was added to (fixed 2026-09-09)."""
    want_replay = bool(RC_L) and a.rec_delta == "suffix"
    ids = enc_of(padded[key])
    L = ids.shape[1]
    REC["on"], REC["inputs"] = want_replay, {}
    with torch.inference_mode():
        c = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    REC["on"] = False
    out = {}
    if RC_L:
        native = collect_rec(c)
        if want_replay:                          # suffix-only states, built beside the native ones
            win = desc_rows(key)
            suf = {}
            with torch.inference_mode():
                for i in RC_L:
                    d = dict(native[i])
                    assert "recurrent" in d, f"layer {i}: recurrent state not a plain tensor: {list(d)}"
                    St = replay_skip(REC["inputs"][i], *(win or (None, None)))
                    d["recurrent"] = St.detach().float().cpu().clone()
                    suf[i] = d
            out["_rec"] = suf
        else:
            out["_rec"] = native
    with torch.inference_mode():                 # the launch token completes the KV rows
        model(input_ids=ids[:, L - 1:], past_key_values=c, use_cache=True)
    out.update({i: pm.harvest(c, i, KINDS[i], LFULL) for i in PATCHED})
    return out


STORE = {k: harvest(k) for k in sorted(padded)}
print(f"[harvest] {len(STORE)} prompt states", flush=True)
for _k in sorted(padded):                 # the severed window on one prompt, in words (the fix of 2026-09-25 is visible here)
    _w = desc_rows(_k)
    if _w is None:
        continue
    _ids = ids_of(padded[_k])
    print(f"[window] {_k}: rows {_w[0]}..{_w[1]} of {len(_ids)} severed = {tok.decode(_ids[_w[0]:_w[1]])!r}; "
          f"next = {tok.decode(_ids[_w[1]:_w[1] + 3])!r}; tail rows {len(_ids) - K}..{len(_ids)} = {tok.decode(_ids[len(_ids) - K:])!r}; "
          f"window/tail overlap {max(0, _w[1] - (len(_ids) - K))} row(s)", flush=True)
    break


def mean_delta(pairs):
    if ONLINE is not None:                 # online site: a recipe of context pairs, evaluated per token (online_site.py)
        return {"_groups": [list(pairs)]}
    out = {}
    for i in PATCHED:
        names = STORE[pairs[0][0]][i].keys()
        out[i] = {n: torch.stack([STORE[x][i][n].float()
                                  - STORE[y][i][n].float()
                                  for x, y in pairs]).mean(0) for n in names}
    if RC_L:
        rec = {}
        for i in RC_L:
            if i not in STORE[pairs[0][0]].get("_rec", {}):
                continue
            keys = STORE[pairs[0][0]]["_rec"][i].keys()
            rec[i] = {k: torch.stack([STORE[x]["_rec"][i][k]
                                      - STORE[y]["_rec"][i][k]
                                      for x, y in pairs]).mean(0)
                      for k in keys}
        out["_rec"] = rec
    return out


def rand_matched(delta, seed):
    if "_groups" in delta:
        return {**delta, "_rand": seed}
    g = torch.Generator().manual_seed(seed)
    out = {}
    for i in PATCHED:
        out[i] = {}
        for n, v in delta[i].items():
            r = torch.randn(v.shape, generator=g)
            w, rw = v[:, W0:W1, :], r[:, W0:W1, :]
            out[i][n] = r * (w.norm() / (rw.norm() + 1e-9))
    if "_rec" in delta:
        rec = {}
        for i, d in delta["_rec"].items():
            rec[i] = {}
            for k, v in d.items():
                r = torch.randn(v.shape, generator=g)
                rec[i][k] = r * (v.norm() / (r.norm() + 1e-9))
        out["_rec"] = rec
    return out


_gabl = torch.Generator().manual_seed(7)
def ablate_desc(cache, win):
    if win is None or a.ablate_desc in ("none", "mask", "bias"):
        return
    d0, d1 = win
    for i in KV_L:
        layer = cache.layers[i]
        for attr in ("keys", "values"):
            t = getattr(layer, attr, None)
            if t is None or t.shape[2] < d1:
                continue
            w = t[:, :, d0:d1, :]
            if a.ablate_desc == "zero":
                t[:, :, d0:d1, :] = 0
            else:
                r = torch.randn(w.shape, generator=_gabl,
                                dtype=torch.float32).to(w.device, w.dtype)
                t[:, :, d0:d1, :] = r * (
                    w.float().norm(dim=-1, keepdim=True).to(w.dtype)
                    / (r.float().norm(dim=-1, keepdim=True).to(w.dtype)
                       + 1e-6))


def patch_rec(cache, delta):
    if "_rec" not in delta:
        return
    for i, d in delta["_rec"].items():
        layer = cache.layers[i]
        for nm, attr in (("conv", "conv_states"),
                         ("recurrent", "recurrent_states")):
            st = getattr(layer, attr, None)
            if st is None:
                continue
            if isinstance(st, dict):
                for k in list(st):
                    key = f"{nm}.{k}"
                    if st[k] is None or key not in d:
                        continue
                    add, tgt = d[key], st[k]
                    if add.dim() == tgt.dim() + 1 and add.shape[0] == 1:
                        add = add[0]
                    st[k] = (tgt.float() + a.alpha
                             * add.to(tgt.device).float()).to(tgt.dtype)
            elif nm in d:
                add = d[nm]
                if add.dim() == st.dim() + 1 and add.shape[0] == 1:
                    add = add[0]
                setattr(layer, attr,
                        (st.float() + a.alpha
                         * add.to(st.device).float()).to(st.dtype))


def kv_site_patch(cache, i, delta, lo, hi, ctx, L):
    """site rows lo:hi of the recipient (its own rows) += alpha * delta. The store shares the recipient's
    absolute rows."""
    pm.patch(cache, i, pm.KV, delta[i], a.alpha, lo=lo, hi=hi, s0=lo, s1=hi)


@torch.inference_mode()
def generate(rec_key, delta, channel="kv", ablate=True):
    ids = enc_of(padded[rec_key])
    L = ids.shape[1]
    REC["on"], REC["inputs"] = (ablate and a.rec_sever == "skip" and bool(RC_L)), {}
    cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    REC["on"] = False
    win = None
    if ablate:
        win = desc_rows(rec_key)
        ablate_desc(cache, win)
        if a.rec_sever == "skip" and RC_L and win is not None and a.ablate_desc != "none":
            d0, d1 = win[0], min(win[1], L - 1)
            for i in RC_L:
                St = replay_skip(REC["inputs"][i], d0, d1)
                tgt = cache.layers[i].recurrent_states
                tgt.copy_(St.to(tgt.device, tgt.dtype))
    # attention mask for the generation steps: the cue's rows are excluded from
    # attention rather than overwritten -- they take no softmax share and relay
    # nothing. Prefill above stays unmasked, so the tail rows are written as
    # they always are and the condition is "cannot re-read" (run_tci_severed.py).
    amask = None
    if ablate and a.ablate_desc == "mask" and win is not None:
        amask = torch.ones(1, L, dtype=torch.long, device=dev)
        amask[:, win[0]:min(win[1], L - 1)] = 0
    S0, S1 = W0, W1
    hi = min(S1, L - 1)
    if delta is not None and channel in ("kv", "kvrec") and a.site != "online":
        for i in KV_L:
            kv_site_patch(cache, i, delta, S0, hi, L - 1, L)
    if delta is not None and channel in ("rec", "kvrec"):
        patch_rec(cache, delta)
    if ablate and a.ablate_desc == "bias" and win is not None:
        BIAS.update(on=True, beta=a.desc_bias, cols=(win[0], min(win[1], L - 1)), fired=0)
    if ONLINE is not None and delta is not None:
        assert "_groups" in delta, "online site: delta without context groups"
        return ONLINE.generate_online(model, tok, enc_of, padded, cache, ids, amask, delta, a.alpha, stop, stop_strings, a.max_new_tokens, dev)
    step = model(input_ids=ids[:, L - 1:], past_key_values=cache,
                 use_cache=True, attention_mask=amask)
    if delta is not None and channel in ("kv", "kvrec") and S1 >= L - 1 and a.site != "online":
        for i in KV_L:                      # launch slot, patched exactly once
            kv_site_patch(cache, i, delta, L - 1, L, L, L)
    logits, cache = step.logits[:, -1], step.past_key_values
    out = []
    for _ in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in stop:
            break
        out.append(nxt)
        txt = tok.decode(out[-24:], skip_special_tokens=True)
        if stop_strings and any(s_ in txt for s_ in stop_strings):
            break
        if amask is not None:
            amask = torch.cat([amask, torch.ones(1, 1, dtype=amask.dtype, device=dev)], 1)
        step = model(input_ids=torch.tensor([[nxt]], device=dev),
                     past_key_values=cache, use_cache=True, attention_mask=amask)
        logits, cache = step.logits[:, -1], step.past_key_values
    if BIAS["on"]:
        if not _PLUMB.get("bias"):
            _PLUMB["bias"] = True
            print(f"[bias] beta={BIAS['beta']} cols={BIAS['cols']} fired={BIAS['fired']} over "
                  f"{len(out) + 1} steps, mask {BIAS['shape']}", flush=True)
            assert BIAS["fired"] == len(_ATTN) * (len(out) + 1), (BIAS["fired"], len(_ATTN), len(out) + 1)
        BIAS["on"] = False
    return tok.decode(out, skip_special_tokens=True)


def score(txt, lam):
    """execute under the REQUESTED language's runner, never by detection;
    the strategy is read structurally off the same code"""
    if a.strip_think and "</think>" in txt:
        txt = txt.split("</think>")[-1]
    code, _ = extract_code(txt, "auto")
    entry = find_entry(lam, code) if code else None
    st, got = run_forms(lam, code, entry)     # memoised calling-form sweep
    return {"code": code, "entry": entry, "status": st,
            "impl": which_function(got), "built": st == "ok",
            "strategy": detect_strategy(code, lam) if code else "unknown"}


outdir = Path(a.outdir or ROOT / "results" / f"tcis_{a.model}")
outdir.mkdir(parents=True, exist_ok=True)
CHANNELS = a.channels or (["kv"] + (["kvrec", "rec"] if HYBRID else []))
NOCH_ARMS = ("none", "prompted", "prompted_sev")   # channel-free: written ONCE,
                                     # tagged kv, whatever --channels says
DEF = Deferred(workers=2)            # compile/run off the GPU's critical path
OTHER = {"rec": "iter", "iter": "rec"}
n_done = 0

for op in a.ops:
    for A in a.funcs:
        for lam in LANGS_T:
            for s in a.strategies:
                sb = OTHER[s]
                # the pooled delta draws on every language in the
                # prompt set except the recipient's own
                donors = [(f, l, c) for f in a.funcs if f != A
                          for l in POOL(lam)
                          for c in CUE_FAMILIES if c != a.cue]
                assert donors, ("no held-out donors: need >=2 functions and "
                                ">=2 languages in the grid")
                rec_key = (("nos", A, lam, "-") if op == "install"
                           else ("s", A, lam, sb, a.cue))
                d0 = donors[0]
                if a.single_donor_lang:
                    _d = [d for d in donors if d[1] == a.single_donor_lang]
                    if not _d:            # recipient's own language: no such donor exists
                        continue
                    d0 = _d[0]
                delta = exact = single = rnd = means = wrongs = None
                xl = {}
                # deltas are only built if an arm actually needs them: the
                # validity grid runs the prompted arm alone
                need = bool({"delta", "exact", "rand", "single", "mean",
                             "wrongs"} & set(a.arms))
                want_xl = "xlang" in a.arms and XLANG
                if need or want_xl:
                    if op == "install":
                        delta = mean_delta([(("s", f, l, s, c),
                                             ("nos", f, l, "-"))
                                            for f, l, c in donors])
                        exact = mean_delta([(("s", A, lam, s, a.cue),
                                             ("nos", A, lam, "-"))])
                        # ONE held-out donor, not the pooled mean
                        single = mean_delta([(("s", d0[0], d0[1], s, d0[2]),
                                              ("nos", d0[0], d0[1], "-"))])
                        # mean over the strategy values: strategy-shaped
                        # state carrying neither identity. Readable here
                        # because the movable-cell filter keeps almost only
                        # rec-targets (the untouched recipient is iterative
                        # in 31/32 cells), so "did it install rec" is a real
                        # question. On TRANSPORT the same construction is
                        # identically zero -- the two directed deltas are
                        # exact negatives -- so it is built for install only.
                        means = mean_delta([(("s", f, l, s2, c),
                                             ("nos", f, l, "-"))
                                            for s2 in STRATEGIES
                                            for f, l, c in donors])
                        # wrong value: the LOO estimate for the OTHER
                        # strategy, installed into the same empty slot.
                        wrongs = mean_delta([(("s", f, l, sb, c),
                                              ("nos", f, l, "-"))
                                             for f, l, c in donors])
                    else:
                        delta = mean_delta([(("s", f, l, s, c),
                                             ("s", f, l, sb, c))
                                            for f, l, c in donors])
                        exact = mean_delta([(("s", A, lam, s, a.cue),
                                             ("s", A, lam, sb, a.cue))])
                        single = mean_delta([(("s", d0[0], d0[1], s, d0[2]),
                                              ("s", d0[0], d0[1], sb, d0[2]))])
                        # wrong value. With two strategies the other value's
                        # LOO estimate is exactly -delta, so this is an
                        # anti-dose into a recipient the severed cue already
                        # holds at sb (untouched, it emits sb in 31/32 cells).
                        # Its score against the target is 0 by construction:
                        # it tests that the factor is SIGN-sensitive, not that
                        # it is value-specific -- a binary factor has no third
                        # value to draw an independent contrast from.
                        wrongs = mean_delta([(("s", f, l, sb, c),
                                              ("s", f, l, s, c))
                                             for f, l, c in donors])
                        # mean over the strategy values. With two strategies
                        # the two directed deltas are exact negatives, so this
                        # is the ZERO vector: it reproduces `none` by
                        # construction rather than measuring anything.
                        # Generated regardless, so the figure carries a
                        # measured point instead of an unexplained gap.
                        means = mean_delta([(("s", f, l, s2, c),
                                             ("s", f, l, OTHER[s2], c))
                                            for s2 in STRATEGIES
                                            for f, l, c in donors])
                    rnd = rand_matched(delta, 0)
                if want_xl:      # one delta per DONOR LANGUAGE
                    for ld in XLANG:
                        pairs = [(("s", f, ld, s, c), ("s", f, ld, sb, c))
                                 if op == "transport" else
                                 (("s", f, ld, s, c), ("nos", f, ld, "-"))
                                 for f in a.funcs if f != A
                                 for c in CUE_FAMILIES if c != a.cue]
                        xl[ld] = mean_delta(pairs)

                best_ch = CHANNELS[1] if len(CHANNELS) > 1 else CHANNELS[0]
                for ch in CHANNELS:
                    arms = [("none", None), ("delta", delta),
                            ("exact", exact), ("rand", rnd),
                            ("single", single), ("prompted", None),
                            ("prompted_sev", None)]
                    if means is not None:
                        arms.append(("mean", means))
                    if wrongs is not None:
                        arms.append(("wrongs", wrongs))
                    if ch == best_ch:      # donor-language sweep, one channel
                        arms += [(f"xlang_{ld}", xl[ld]) for ld in XLANG]
                    for arm, dd in arms:
                        if arm.startswith("xlang_"):
                            if "xlang" not in a.arms:
                                continue
                        elif arm not in a.arms:
                            continue
                        if arm in NOCH_ARMS and ch != CHANNELS[0]:
                            continue      # channel-free references, once
                        cht = "kv" if arm in NOCH_ARMS else ch
                        tag = f"{op}_{A}_{lam}_{s}_{cht}_{arm}"
                        fp = outdir / f"c_{tag}.json"
                        if fp.exists():
                            continue
                        if arm == "prompted":
                            # ask for exactly what we want, words intact
                            txt = generate(("s", A, lam, s, a.cue), None, "kv",
                                           ablate=False)
                        elif arm == "prompted_sev":
                            # ask, then sever the ask: the compiled store alone
                            txt = generate(("s", A, lam, s, a.cue), None, "kv",
                                           ablate=True)
                        else:
                            txt = generate(rec_key, dd, ch)

                        def _emit(sc, fp=fp, tag=tag, op=op, A=A, lam=lam, s=s,
                                  sb=sb, cht=cht, arm=arm, rec_key=rec_key,
                                  n_don=(1 if arm == "single" else len(donors)),
                                  donor=(list(d0) if arm == "single" else
                                         (arm.split("_", 1)[1]
                                          if arm.startswith("xlang_") else None)),
                                  txt=txt):
                            global n_done
                            rec = {"op": op, "factor": "S", "cell": A, "lang": lam,
                                   "want_strategy": s, "from_strategy":
                                       sb if op == "transport" else None,
                                   "cue": a.cue, "channel": cht, "arm": arm,
                                   "recipient": list(rec_key),
                                   "n_donors": n_don, "donor": donor,
                                   "donor_pool": a.donor_pool,
                                   "pool_langs": POOL(lam),
                                   "alpha": a.alpha,
                                   "ablate_desc": a.ablate_desc, "desc_bias": a.desc_bias,
                                   "ablate_span": a.ablate_span,
                                   "hit_strat": sc["strategy"] == s,
                                   "hit_fn": sc["impl"] == A,
                                   "hit": (sc["strategy"] == s
                                           and sc["impl"] == A and sc["built"]),
                                   **sc, "text": txt}
                            fp.write_text(json.dumps(rec, indent=1))
                            n_done += 1
                            print(f"[{tag}] strat={sc['strategy']} "
                                  f"impl={sc['impl']} built={sc['built']} "
                                  f"hit={rec['hit']}", flush=True)
                        DEF.submit(score, (txt, lam), _emit)
                    DEF.flush()
                    if a.limit and n_done >= a.limit:
                        print(f"[limit] stopping at {n_done}", flush=True)
                        sys.exit(0)

DEF.flush()
(outdir / "meta.json").write_text(json.dumps(
    {"model": a.model, "langs": LANGS_T, "donor_only": DONOR, "funcs": a.funcs,
     "cue": a.cue, "strategies": a.strategies, "xlang": XLANG, "alpha": a.alpha,
     "ablate_desc": a.ablate_desc, "desc_bias": a.desc_bias, "ablate_span": a.ablate_span,
     "site": [W0, W1], "align": "pad", "K": K, "site_name": a.site, "online": (ONLINE.info() if ONLINE is not None else None), "channels": CHANNELS,
     "rec_sever": a.rec_sever, "rec_delta": a.rec_delta,
     "donor_pool": a.donor_pool, "families": FAMS, "pool_always": a.pool_always},
    indent=1))
print(f"[done] {n_done} cells -> {outdir}", flush=True)

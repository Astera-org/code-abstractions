"""Random-relabelling control: is CFS measuring THIS grid, or any grid?

The bank is 40 prompts = 4 functions x 10 languages, and CFS_F ~ 1 says the
model's geometry agrees with that carving. This asks whether it would agree
with any carving. Invent two fake factors -- "shape" (4 levels) and "colour"
(10 levels) -- assign the 40 prompts to that grid at random, and run the same
battery treating shape as the factor:

    recipient   the prompt at (shape s0, colour c), description ablated
    exact       state(s1, c) - state(s0, c)          a genuine state swap
    delta       mean over c' != c of state(s1, c') - state(s0, c')
    rand        norm-matched noise

A hit means the generation implements the TARGET prompt's function in the
TARGET prompt's language -- the same execution grading as the real battery.

Prediction: exact still works, because swapping one prompt's state for
another's does not care how the prompts are labelled. The pooled arm should
collapse, because it averages arrows that each change a different function AND
a different language. That is the "addressable but not factored" cell the
table does not currently contain, and it is what makes CFS ~ 1 a finding
rather than what the pipeline always returns.

    uv run factored_programming_languages/run_tci_relabel.py \
        --model gemma-4-31B-it --device cuda:0 --seed 0
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
from tasks_compose import full, nofn, nolang, CARRIER, TAIL                     # noqa: E402
from tasks_function import FDESC                                          # noqa: E402
import execute as S                                                       # noqa: E402
import prompt_memory as pm                                                # noqa: E402
from scoring import run_forms, Deferred                                   # noqa: E402
import loader                                                             # noqa: E402
from entry import find_entry                                            # noqa: E402
from tasks_strategy import detect_strategy                               # noqa: E402

S.BATTERY = BATTERY          # mandatory rebind (memory: execute-battery-rebind)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--seed", type=int, default=0,
                help="which random labelling; the null distribution is over "
                     "seeds")
ap.add_argument("--seeds", nargs="+", type=int, default=None,
                help="run several labellings in one process (the model and the "
                     "prompt store are loaded once); overrides --seed")
ap.add_argument("--tag", default="",
                help="inserted into the results dir name: <sfx><tag>_<model>_s<seed>, "
                     "so a rerun on a new bank does not overwrite the originals")
ap.add_argument("--bank", default="function",
                choices=["function", "strategy"],
                help="which prompt bank to re-carve. 'function' is the 40 "
                     "full(f, lang) prompts, a 4 x 10 grid. 'strategy' is the "
                     "96 strat(lang, f, s, cue) prompts, re-carved as 2 fake "
                     "levels x 48 fake contexts, where a context is a "
                     "(function, language, cue) triple -- the same shape as "
                     "the real strategy arm, which is a 2-level factor pooled "
                     "over contexts.")
ap.add_argument("--op", default="transport",
                choices=["transport", "install", "compose", "compose3"],
                help="which operation to run the null for. transport moves "
                     "between two cells of the fake grid; install writes a "
                     "cell into an empty slot; compose sums a shape factor "
                     "and a colour factor into the bare carrier.")
ap.add_argument("--axis", default="shape", choices=["shape", "colour"],
                help="which fake factor to treat as THE factor. 'shape' is a "
                     "4-level factor pooled over the 9 other colours -- the "
                     "null matched to the real f arm. 'colour' is a 10-level "
                     "factor pooled over the 3 other shapes -- matched to the "
                     "real lambda arm (othf = 3 functions), and the harder "
                     "null of the two, because a 3-donor pooled estimate is "
                     "much noisier than a 9-donor one.")
ap.add_argument("--cells", type=int, default=16,
                help="shape-pairs x colours to run, matched to the real "
                     "battery's 16 transport cells")
ap.add_argument("--langs", nargs="+", default=["javascript", "go", "java", "c"],
                help="target languages (python excluded: carrier default)")
ap.add_argument("--funcs", nargs="+", default=FKEYS)
ap.add_argument("--pairs", nargs="+",
                default=["digit_sum:digit_prod", "digit_prod:digit_sum",
                         "sq_digit_sum:even_digits", "even_digits:sq_digit_sum"],
                help="transport A:B (write A into a recipient asking for B)")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--match-norm", action="store_true",
                help="rescale the COMPOSED delta so its norm over the patch "
                     "window equals the single-contrast exact delta's. "
                     "Summing factors inflates the norm (~1.4x for three), "
                     "and the usable window is narrow, so an unmatched sum is "
                     "over-dosed and the comparison to exact is unfair.")
ap.add_argument("--ablate-desc", default="rand", choices=["rand", "zero", "none"])
ap.add_argument("--ablate-span", default="spec", choices=["spec", "desc", "lang", "cue"],
                help="rows the severing covers on the RECIPIENT prompt: spec = its whole specification; "
                     "desc = the description sentence only; lang = the language name only; cue = the "
                     "strategy cue only -- the same span the corresponding real rung severs, so the null "
                     "runs the real rung's procedure on a random carving (9F: shape->desc, colour->lang, "
                     "strategy->cue)")
ap.add_argument("--rec-sever", default="none", choices=["none", "skip"],
                help="hybrids: also sever the RECURRENT channel of the recipient (skip replay)")
ap.add_argument("--rec-delta", default="state", choices=["state", "suffix"],
                help="hybrids: recurrent delta from FINAL states or from skip-replayed states")
ap.add_argument("--recurrent", action="store_true",
                help="hybrids: add kvrec / rec arms")
ap.add_argument("--only-base", action="store_true",
                help="generate ONLY none/delta/rand (complement of "
                     "--skip-base, so two processes can split a model)")
ap.add_argument("--skip-base", action="store_true",
                help="generate only the arms added later (exact, prompted, "
                     "L cells) so a completion pass can run "
                     "on a free card alongside the base grid without "
                     "duplicating its none/delta/rand work")
ap.add_argument("--channels", nargs="+", default=None,
                choices=["kv", "kvrec", "rec"],
                help="intervention channels to run (default: kv, plus kvrec "
                     "and rec on hybrids with --recurrent). The channel-free "
                     "none/prompted arms are always written, tagged kv.")
ap.add_argument("--max-new-tokens", type=int, default=340)
ap.add_argument("--strip-think", action="store_true")
ap.add_argument("--outdir", default=None)
ap.add_argument("--out-model", default=None, help="name used in place of --model in the default result-dir names (e.g. <model>-onl)")
ap.add_argument("--site", default="suffix", choices=["suffix", "online"],
                help="suffix = K/V rows of the aligned suffix; online = per-token residual counterfactual (online_site.py)")
ap.add_argument("--online-mode", default="live", choices=["live", "vector"],
                help="--site online: live = per-token counterfactual; vector = pooled generation vector, frozen (online_site.py)")
ap.add_argument("--online-layers", default="band", choices=["band", "all"])
a = ap.parse_args()

recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]
from transformers import AutoTokenizer                                    # noqa: E402
tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]),
                                    **recipe.get("tokenizer_kwargs", {}))
ids_of = lambda t: tok(render_prompt(tok, recipe, t),                     # noqa: E731
                       add_special_tokens=False)["input_ids"]

# ---- prompt set: the compose grid + the empty-slot variants ----------------
raw = {}
if a.bank == "strategy":
    from tasks_strategy import (STRATEGIES, CUE_FAMILIES, strat,    # noqa: E402
                                nostrat)
    STRAT_CTX = [(f, l, c) for f in FKEYS for l in a.langs
                 for c in CUE_FAMILIES]
    for f, l, c in STRAT_CTX:
        for s in STRATEGIES:
            raw[("st", f, l, f"{s}:{c}")] = strat(l, f, s, c)
    for f, l, _c in STRAT_CTX:          # strategy-absent, 16 distinct
        raw[("nos", f, l, "-")] = nostrat(l, f)
    raw[("carrier", "-", "-")] = CARRIER
else:
    for f in FKEYS:
        for l in LANGS:
            raw[("full", f, l)] = full(l, f)
    for l in LANGS:
        raw[("nofn", "-", l)] = nofn(l)
    for f in FKEYS:
        raw[("nolang", f, "-")] = nolang(f)
    raw[("carrier", "-", "-")] = CARRIER

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
assert all(padded.values()), \
    f"padding failed: {[k for k, v in padded.items() if not v]}"
seqs = [ids_of(u) for u in padded.values()]
K = 0
while K < min(len(s) for s in seqs) and len({s[-1 - K] for s in seqs}) == 1:
    K += 1
PLEN = {k: len(s) for k, s in zip(padded, seqs)}
LFULL = target
W0, W1 = LFULL - K, LFULL
assert K >= 10, f"shared suffix only {K} tokens"
print(f"[site] suffix rows {W0}..{W1} of {LFULL} (K={K})", flush=True)

# ---- model ------------------------------------------------------------------
_L = loader.load(a.model, device=a.device, max_memory=a.max_memory, attn="sdpa")
from online_site import OnlineSite                                   # noqa: E402
ONLINE = None
if a.site == "online":
    import models as _cast
    _rows = (1, len(_L.layers)) if a.online_layers == "all" else _cast.band(a.model)
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

with torch.inference_mode():
    _probe = model(input_ids=enc_of(padded[("carrier", "-", "-")]),
                   use_cache=True)
KINDS = pm.profile(_probe.past_key_values, n_layers)
del _probe
PATCHED = pm.select(KINDS, "both")
KV_L = [i for i in PATCHED if KINDS[i] == pm.KV]
RC_L = [i for i in range(n_layers) if KINDS[i] != pm.KV]
print(f"[cache] {pm.describe(KINDS)} | kv {len(KV_L)} rec {len(RC_L)}",
      flush=True)
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
    """rows of the recipient's ENTIRE specification -- everything between
    the frame ("Write a function") and the shared tail. For full(lam, f)
    that is the language name AND the function description; for nofn it
    is the language name; for nolang the function description. Ablating
    the whole span is what makes the severed regime consistent: the words
    channel then specifies nothing and the state is the only source."""
    if key[0] == "carrier":
        return None
    r = render_prompt(tok, recipe, padded[key])
    o = tok(r, add_special_tokens=False,
            return_offsets_mapping=True)["offset_mapping"]
    c0 = r.find(ANCHOR)
    c1 = r.find(TAIL)
    if c0 < 0 or c1 <= c0:
        return None
    # right edge as in the published runs: the token that owns the tail's first character (it carries the space before
    # "Name") is admitted, so one tail token is severed with the specification; _tail_start would exclude it.
    c0 += len(ANCHOR)
    if a.ablate_span in ("desc", "lang") and key[0] == "full":
        from languages import DISPLAY
        lead = f" in {DISPLAY[key[2]]} that "
        assert r.startswith(lead, c0), r[c0:c0 + 40]
        if a.ablate_span == "desc":
            c0 += len(lead)                                  # the description sentence
        else:
            c0 += len(" in "); c1 = c0 + len(DISPLAY[key[2]])   # the language name alone
    elif a.ablate_span == "cue" and key[0] == "st":
        from tasks_strategy import CUES
        s_, cue = key[3].split(":")
        c0 = r.find(CUES[s_][cue]); assert c0 > 0, key
        c1 = c0 + len(CUES[s_][cue]) + 1
    rows = [i for i, (s_, e) in enumerate(o)
            if s_ < c1 and e > c0 and e > s_]
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


NEED = set(raw)
STORE = {k: harvest(k) for k in sorted(NEED)}
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
    """mean over pairs of (KV(x) - KV(y)); also recurrent if present"""
    if ONLINE is not None:                 # online site: a recipe of context pairs, evaluated per token
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


def add_deltas(d1, d2):
    if "_groups" in d1:
        return {"_groups": d1["_groups"] + d2["_groups"]}
    out = {i: {n: d1[i][n] + d2[i][n] for n in d1[i]} for i in PATCHED}
    if "_rec" in d1 and "_rec" in d2:
        out["_rec"] = {i: {k: d1["_rec"][i][k] + d2["_rec"][i][k]
                           for k in d1["_rec"][i]}
                       for i in d1["_rec"] if i in d2["_rec"]}
    return out


def win_norm(d):
    """norm over the patched suffix window only -- the part that is applied"""
    return float(torch.cat([d[i][n][:, W0:W1, :].reshape(-1)
                            for i in sorted(k for k in d if isinstance(k, int))
                            for n in sorted(d[i])]).norm())


def rescale_to(d, target):
    s = target / (win_norm(d) + 1e-9)
    out = {i: {n: v * s for n, v in d[i].items()}
           for i in d if isinstance(i, int)}
    if "_rec" in d:
        out["_rec"] = {i: {k: v * s for k, v in dd.items()}
                       for i, dd in d["_rec"].items()}
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
    if win is None or a.ablate_desc == "none":
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
    """add the recurrent/conv part of delta into the cache in place"""
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
                    add = d[key]
                    tgt = st[k]
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
def generate(rec_key, delta, channel="kv"):
    ids = enc_of(padded[rec_key])
    L = ids.shape[1]
    REC["on"], REC["inputs"] = (a.rec_sever == "skip" and bool(RC_L)), {}
    cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    REC["on"] = False
    win = desc_rows(rec_key)
    ablate_desc(cache, win)
    if a.rec_sever == "skip" and RC_L and win is not None and a.ablate_desc != "none":
        d0, d1 = win[0], min(win[1], L - 1)
        for i in RC_L:
            St = replay_skip(REC["inputs"][i], d0, d1)
            tgt = cache.layers[i].recurrent_states
            tgt.copy_(St.to(tgt.device, tgt.dtype))
    S0, S1 = W0, W1
    hi = min(S1, L - 1)
    if delta is not None and channel in ("kv", "kvrec") and a.site != "online":
        for i in KV_L:
            kv_site_patch(cache, i, delta, S0, hi, L - 1, L)
    if delta is not None and channel in ("rec", "kvrec"):
        patch_rec(cache, delta)
    if ONLINE is not None and delta is not None:
        assert "_groups" in delta, "online site: delta without context groups"
        return ONLINE.generate_online(model, tok, enc_of, padded, cache, ids, None, delta, a.alpha, stop, stop_strings, a.max_new_tokens, dev)
    step = model(input_ids=ids[:, L - 1:], past_key_values=cache,
                 use_cache=True)
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
        step = model(input_ids=torch.tensor([[nxt]], device=dev),
                     past_key_values=cache, use_cache=True)
        logits, cache = step.logits[:, -1], step.past_key_values
    return tok.decode(out, skip_special_tokens=True)


def score(txt, lam):
    """execute under the REQUESTED language's runner, never by detection"""
    if a.strip_think and "</think>" in txt:
        txt = txt.split("</think>")[-1]
    code, _ = extract_code(txt, "auto")
    entry = find_entry(lam, code) if code else None
    st, got = run_forms(lam, code, entry)     # memoised calling-form sweep
    return {"code": code, "entry": entry, "status": st,
            "impl": which_function(got), "built": st == "ok"}


def run_seed(seed):
    """one random labelling: build the fake grid, its cases, run them"""
    global n_done
    a.seed = seed
    _sfx = ("tcirel" if a.axis == "shape" else "tcirelc") if a.op == "transport" \
        else {"install": ("tcireli" if a.axis == "shape" else "tcirelic"),
              "compose": "tcirelx", "compose3": "tcirelx3"}[a.op]
    if a.bank == "strategy":
        _sfx = {"transport": "tcirels", "install": "tcirelsi"}[a.op]
    outdir = Path(a.outdir or ROOT / "results" / f"{_sfx}{a.tag}_{a.out_model or a.model}_s{a.seed}")
    outdir.mkdir(parents=True, exist_ok=True)
    CHANNELS = a.channels or (["kv"] + (["kvrec", "rec"] if HYBRID else []))
    NOCH_ARMS = ("none", "prompted")     # channel-free references: written ONCE,
                                         # tagged kv, whatever --channels says
    DEF = Deferred(workers=2)            # compile/run off the GPU's critical path
    n_done = 0

    # ---- the fake grid ---------------------------------------------------------
    import random as _random
    if a.bank == "strategy":
        # 2 fake levels x 48 fake contexts, mirroring the real strategy arm (a
        # 2-level factor pooled over (function, language, cue) contexts). The
        # WHOLE bank is scattered, so each arrow changes a different combination
        # of function, language, strategy and cue. The cheaper alternative --
        # randomly flipping which strategy counts as the target in each context --
        # does not work: with two levels those arrows are sign-reversed copies of
        # one another and the pooled average cancels to the zero vector, giving
        # another degenerate arm rather than a labelling null.
        CELLS = [k for k in raw if k[0] == "st"]
        _NOS = [k for k in raw if k[0] == "nos"]
        SHAPES = len(STRATEGIES)
        COLOURS = len(CELLS) // SHAPES
    else:
        CELLS = [(f, l) for f in FKEYS for l in LANGS]
        SHAPES, COLOURS = 4, len(LANGS)
    _random.Random(a.seed).shuffle(CELLS)
    grid = {}                       # (shape, colour) -> (function, language)
    for idx, fl in enumerate(CELLS):
        grid[(idx // COLOURS, idx % COLOURS)] = fl
    print(f"[relabel] seed {a.seed}: 4 shapes x {COLOURS} colours over "
          f"{len(CELLS)} prompts", flush=True)

    assert not (a.bank == "strategy" and a.op in ("compose", "compose3")), \
        "compose on the strategy bank needs a 3-axis carving: use --op compose3 "\
        "on the function bank"


    def KEY(cell):
        """prompt-set key for a fake-grid cell. For the strategy bank the grid
        already holds full prompt keys; for the function bank it holds (f, lang)."""
        g = grid[cell]
        return g if a.bank == "strategy" else ("full",) + g


    CARRIER_KEY = ("carrier", "-", "-")
    # Install needs "the cell at colour c with its shape removed", but colour is a
    # random group of four unrelated prompts, so nothing can lack it. Scatter the
    # ten empty-slot prompts across the ten colours instead: no assignment is
    # canonical, so this one is part of the labelling and moves with the seed.
    if a.bank == "strategy":
        # strategy-absent prompts scattered over the fake contexts. There are only
        # 16 distinct nostrat prompts for 48 contexts, so the assignment repeats;
        # no assignment is canonical either way, and this one moves with the seed.
        _es = [_NOS[i % len(_NOS)] for i in range(COLOURS)]
        _random.Random(a.seed + 4441).shuffle(_es)
        EMPTY = {c: _es[c] for c in range(COLOURS)}
        EMPTY_L = {}
    else:
        _empt = [("nofn", "-", l) for l in LANGS]
        _random.Random(a.seed + 9973).shuffle(_empt)
        EMPTY = {c: _empt[c] for c in range(COLOURS)}
        # the OTHER empty family: language-absent (nolang), scattered over the
        # fake shapes. Install's lambda-analogue removes the colour, not the
        # shape, so its recipient must be empty in the colour direction.
        _emptL = [("nolang", f, "-") for f in FKEYS]
        _random.Random(a.seed + 7717).shuffle(_emptL)
        EMPTY_L = {s: _emptL[s] for s in range(SHAPES)}

    _pp = ([(0, 1), (1, 0)] if SHAPES == 2
           else [(0, 1), (1, 0), (2, 3), (3, 2)][:max(1, a.cells // 4)])
    _cols = a.cells // SHAPES
    CASES = []                  # (tag stem, recipient key, target cell, delta)
    if a.op == "transport" and a.axis == "shape":     # 4 levels, 9 donors  (~ f)
        for s0, s1 in _pp:
            for c in list(range(COLOURS))[:a.cells // max(1, len(_pp))]:
                CASES.append((f"rel_{s0}{s1}_c{c}", KEY((s0, c)), KEY((s1, c)),
                              mean_delta([(KEY((s1, cc)), KEY((s0, cc)))
                                          for cc in range(COLOURS) if cc != c])))
    elif a.op == "transport":                    # 10 levels, 3 donors (~ lambda)
        for c0, c1 in _pp:
            for s in list(range(SHAPES))[:a.cells // max(1, len(_pp))]:
                CASES.append((f"relc_{c0}{c1}_s{s}", KEY((s, c0)),
                              KEY((s, c1)),
                              mean_delta([(("full",) + grid[(ss, c1)],
                                           ("full",) + grid[(ss, c0)])
                                          for ss in range(SHAPES) if ss != s])))
    elif a.op == "install" and a.axis == "shape":
        for s in range(SHAPES):
            for c in range(_cols):
                CASES.append((f"reli_s{s}_c{c}", EMPTY[c], KEY((s, c)),
                              mean_delta([(KEY((s, cc)), EMPTY[cc])
                                          for cc in range(COLOURS) if cc != c])))
    elif a.op == "install":                     # colour axis (~ lambda)
        for s in range(SHAPES):
            for c in range(_cols):
                CASES.append((f"relic_s{s}_c{c}", EMPTY_L[s], KEY((s, c)),
                              mean_delta([(KEY((ss, c)), EMPTY_L[ss])
                                          for ss in range(SHAPES) if ss != s])))
    elif a.op == "compose":                      # into the bare carrier
        for s in range(SHAPES):
            for c in range(_cols):
                shp = mean_delta([(("full",) + grid[(s, cc)], CARRIER_KEY)
                                  for cc in range(COLOURS) if cc != c])
                col = mean_delta([(("full",) + grid[(ss, c)], CARRIER_KEY)
                                  for ss in range(SHAPES) if ss != s])
                CASES.append((f"relx_s{s}_c{c}", CARRIER_KEY, KEY((s, c)),
                              add_deltas(shp, col)))

    if a.op == "compose3":
        # A 3-axis random carving, mirroring the real f x lambda x s compose:
        # 4 x 4 x 2 cells drawn from the bank, three factors each measured against
        # the carrier and each held out from the cell being written, then summed.
        _c3 = CELLS[:32]
        assert len(_c3) == 32, "compose3 needs >= 32 prompts in the bank"
        g3 = {}
        for _i, _fl in enumerate(_c3):
            g3[(_i // 8, (_i // 2) % 4, _i % 2)] = _fl
        K3 = lambda c3: (c3 if a.bank == "strategy" else ("full",) + g3[c3])  # noqa
        if a.bank != "strategy":
            K3 = lambda c3: ("full",) + g3[c3]                                # noqa
        _n = 0
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    if _n >= a.cells:
                        continue
                    _n += 1
                    A_ = mean_delta([(K3((i, jj, kk)), CARRIER_KEY)
                                     for jj in range(4) for kk in range(2)
                                     if (jj, kk) != (j, k)])
                    B_ = mean_delta([(K3((ii, j, kk)), CARRIER_KEY)
                                     for ii in range(4) for kk in range(2)
                                     if (ii, kk) != (i, k)])
                    C_ = mean_delta([(K3((ii, jj, k)), CARRIER_KEY)
                                     for ii in range(4) for jj in range(4)
                                     if (ii, jj) != (i, j)])
                    CASES.append((f"relx3_{i}{j}{k}", CARRIER_KEY, K3((i, j, k)),
                                  add_deltas(add_deltas(A_, B_), C_)))

    for stem, rec_key, tkey, delta in CASES:
        if a.bank == "strategy":
            _, tf, tl, _sc = tkey
            ts = _sc.split(":")[0]           # the target's strategy
        else:
            _, tf, tl = tkey
            ts = None
        exact = mean_delta([(tkey, rec_key)])
        rnd = rand_matched(delta, 0)
        for ch in CHANNELS:
            for arm, dd in (("none", None), ("delta", delta),
                            ("exact", exact), ("rand", rnd)):
                if arm in NOCH_ARMS and ch != CHANNELS[0]:
                    continue                  # channel-free reference, once
                cht = "kv" if arm in NOCH_ARMS else ch
                fp = outdir / f"c_{stem}_{cht}_{arm}.json"
                if fp.exists():
                    continue
                txt = generate(rec_key, dd, ch)

                def _emit(sc, fp=fp, stem=stem, rec_key=rec_key, tf=tf, tl=tl,
                          ts=ts, cht=cht, arm=arm, txt=txt):
                    global n_done
                    # the strategy bank asks the same conjunction the real s arm
                    # does: the target prompt's function, in its language, with
                    # its strategy
                    got_s = (detect_strategy(sc["code"] or "", tl)
                             if a.bank == "strategy" else None)
                    hit = bool(sc["impl"] == tf and sc["built"]
                               and (ts is None or got_s == ts))
                    rec = {"seed": a.seed, "axis": a.axis, "op": a.op,
                           "bank": a.bank, "want_strategy": ts, "strategy": got_s,
                           "stem": stem,
                           "recipient": list(rec_key), "target": [tf, tl],
                           "channel": cht, "arm": arm,
                           "hit": hit, "hit_fn": sc["impl"] == tf, **sc, "text": txt}
                    fp.write_text(json.dumps(rec, indent=1))
                    n_done += 1
                    print(f"  [{stem}_{cht}_{arm}] -> {tf}/{tl}  impl={sc['impl']} "
                          f"hit={rec['hit']}", flush=True)
                DEF.submit(score, (txt, tl), _emit)   # graded in the TARGET's language
        DEF.flush()

for _sd in (a.seeds or [a.seed]):
    run_seed(_sd)                      # each seed flushes its own deferred scores
print(f"[done] seeds {a.seeds or [a.seed]}", flush=True)

"""Transport / Compose / Install at the suffix, in the SEVERED regime.

The campaign's three canonical operations, re-run under what the
abstraction campaign established: suffix KV surgery at alpha=1 with the
recipient's function description RANDOMLY ABLATED post-prefill (the
words channel is the veto; see abstraction.md). Four models: the
continuation cast plus the pythia control (expected to fail -- it is a
re-reader, flow-cut index 0.00). For hybrid models (qwen3.6-27b) the
recurrent/conv states are a second intervention channel, run as extra
arms, because its pointer does not live in the suffix rows alone.

  INSTALL    recipient nofn(lam)      ("Write a function in LANG.")
             delta  f_A = LOO mean over lam' != lam of
                    KV(full(lam', A)) - KV(nofn(lam'))
             hit    writes function A
  COMPOSE    recipient CARRIER        ("Write a function.")
             delta  f_A + l_lam, both leave-one-out
             hit    function A in language lam (both slots filled)
  TRANSPORT  recipient full(lam, B)   (occupied by function B)
             delta  LOO mean of KV(full(lam', A)) - KV(full(lam', B))
             hit    writes A instead of B, in lam
             ONLY this op has a description to ablate; --ablate-desc rand
             severs it post-prefill (recipient's own spec rows).

Arms per cell: none (untouched recipient), delta, rand (norm-matched
random delta). Hybrids add kvrec (KV + recurrent) and rec (recurrent
only). Grading executes the generated code under the REQUESTED
language's runner (never by detection -- see memory note).

    uv run factored_programming_languages/run_tci_severed.py \
        --model gemma-4-31B-it --device cuda:0 --ops install compose transport
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

S.BATTERY = BATTERY          # mandatory rebind (memory: execute-battery-rebind)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--ops", nargs="+", default=["install", "compose", "transport"],
                choices=["install", "compose", "transport"])
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
ap.add_argument("--donor-para", default="0",
                help="wordings the donor prompts are pooled over, comma list of paraphrase indices: 0 = the canonical "
                     "description, 1-3 = tasks_atlas_paraphrases.PARA[f][i-1]. Every donor contrast (x, y) becomes the mean "
                     "over these wordings of (x@p, y@p); prompts without a description are unchanged")
ap.add_argument("--recipient-para", default="0", choices=["0", "1", "2", "3", "rotate"],
                help="wording of the recipient and of the prompted asks: 0 canonical, 1-3 a fixed paraphrase, rotate = "
                     "paraphrase 1 + (lang index + function index) mod 3, so every paraphrase appears")
ap.add_argument("--site", default="suffix", choices=["suffix", "all", "afterdesc", "passage", "online"],
                help="rows patched: the aligned suffix (the compiled store; default) "
                     "or ALL rows of the padded prompt incl. the description -- the "
                     "interface a re-reading model uses. Pair `all` with "
                     "--ablate-desc none to steel-man a re-reader: its words stay, "
                     "and the pooled delta is written over them.")
ap.add_argument("--online-mode", default="live", choices=["live", "vector"],
                help="--site online: live = per-token counterfactual; vector = pooled generation vector, frozen (online_site.py)")
ap.add_argument("--online-layers", default="band", choices=["band", "all"],
                help="--site online: steer the certified band (models.band) or every layer")
ap.add_argument("--sever-tail", action="store_true",
                help="with --filler between: also zero the TAIL rows after prefill whenever the "
                     "description is severed (not in the words-intact `prompted` ceiling), so the "
                     "passage rows are the only carrier; site `passage` steers them alone. "
                     "Sites afterdesc = passage + tail rows, passage = passage rows only")
ap.add_argument("--desc-bias", type=float, default=0.0,
                help="with --ablate-desc bias: constant subtracted from the attention logits of "
                     "the recipient's specification rows in every KV layer at the launch step and "
                     "every decode step (prefill untouched). 0 = words intact, inf = masked out. "
                     "The cache is never modified: the words are attenuated, not overwritten")
ap.add_argument("--ablate-desc", default="rand", choices=["rand", "zero", "none", "mask", "bias"],
                help="how the recipient's specification rows are severed: rand/zero overwrite "
                     "the KV rows; mask leaves the cache intact and excludes the rows from "
                     "attention (attention_mask 0) at the launch step and every decode step -- "
                     "a masked row is absent, with no softmax share and no relayed context")
ap.add_argument("--ablate-span", default="spec", choices=["spec", "desc", "lang"],
                help="rows the severing covers on a full prompt: spec = language name AND "
                     "function description (the ladders' convention); desc = the function "
                     "description only, the language phrase stays readable (the convention of "
                     "run_distance_sever.py, so its numbers are comparable); lang = the language "
                     "name only (for the language-factor cells: the words that conflict with the "
                     "injected language, the description stays readable)")
ap.add_argument("--rec-sever", default="none", choices=["none", "skip"],
                help="hybrids: also sever the RECURRENT channel of the recipient by "
                     "replaying each linear layer's recorded rule inputs with the "
                     "specification steps skipped (k=v=0, decay only) -- the recurrent "
                     "counterpart of KV zeros (run_ablate_both.py, 2026-09-05)")
ap.add_argument("--rec-delta", default="state", choices=["state", "suffix"],
                help="hybrids: build the recurrent part of every delta from the FINAL "
                     "states (published kvrec) or from skip-replayed states, so both "
                     "sides of the difference never wrote their specification and the "
                     "delta is what the suffix steps write differently ('suffix-only')")
ap.add_argument("--arms", nargs="+", default=None,
                help="restrict the F-factor arms (none delta exact rand wrongf meanf "
                     "prompted prompted_sev); default all")
ap.add_argument("--no-lfactor", action="store_true", help="skip the language-factor cells")
ap.add_argument("--guidance", nargs="*", type=float, default=None,
                help="logit guidance (authors, 2026-09-08): for each g, decode the INTACT recipient from "
                     "lp0 + g*(lpD - lp0), where lp0 are the log-probs of the plain recipient and lpD of "
                     "the same recipient with the state displacement applied at alpha 1 (two caches, both "
                     "fed the chosen token). g=1 is ordinary decoding of the intervened recipient. Adds arms "
                     "delta_g<g>/exact_g<g>/rand_g<g> on the last channel. Requires --ablate-desc none")
ap.add_argument("--guidance-tau", type=float, default=0.1,
                help="plausibility floor: only tokens with pD >= tau * max pD (branch WITH the "
                     "displacement) are eligible for the guided argmax")
ap.add_argument("--readout-offset", nargs="*", type=float, default=None,
                help="hybrids, transport (authors, 2026-09-08): persistent recurrent-READOUT offset. For each "
                     "alpha given, arms ro_delta/ro_exact/ro_rand: the tail KV contrast as usual (channel kv, "
                     "recurrent state untouched) plus, at the launch step and every decode step, "
                     "alpha * q_t^T dR added to every linear layer's state read, dR = the NATIVE recurrent "
                     "contrast (matrix only), using the rule's own l2-normalised, scaled query; dR is never "
                     "written into the stored state")
ap.add_argument("--rec-contrast-arms", action="store_true",
                help="hybrids, transport: build the recurrent contrast three ways as separate kvrec arms -- "
                     "from NATIVE (normal-prefill) donor states (delta_native/exact_native), from "
                     "skip-replayed donor states (delta_suffix/exact_suffix), and the suffix contrast "
                     "rescaled per layer and tensor to the native one's norm (*_suffix_nm); plus "
                     "rand_native. The KV part is identical in all. Meant with --rec-sever none. "
                     "The first exact_native cell prints an [endpoint] check against the target's prefill")
ap.add_argument("--only-lfactor", action="store_true", help="run ONLY the language-factor cells")
ap.add_argument("--desc-band", default=None,
                help="lo:hi -- ALSO write the delta into the rows BEFORE the suffix (frame + "
                     "specification), but only in KV layers lo <= i < hi; the suffix rows are "
                     "patched in every layer as usual. The words-intact mechanistic test: does "
                     "the override act above the lexical band? (2026-09-07)")
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
ap.add_argument("--filler", default="none", choices=["none", "between"],
                help="between: put the ~1,500-token unrelated passage (filler_text.py) between "
                     "every prompt's specification and the tail, so the specification sits "
                     "more than a local-attention window behind the tail and everything "
                     "generated; the site stays the tail rows")
ap.add_argument("--passage-rows", default="own", choices=["own", "carrier", "mask"],
                help="with --filler between: own = the recipient's passage rows as prefilled "
                     "(they were written with the specification in view and relay it); "
                     "carrier = replace them, before steering, with the bare carrier prompt's "
                     "rows for the same positions (spec-free, in-distribution), so the tail "
                     "is the only carrier the local layers can read; mask = exclude the passage "
                     "rows from attention during generation (see --ablate-desc mask)")
a = ap.parse_args()
DESC_BAND = tuple(int(x) for x in a.desc_band.split(":")) if a.desc_band else None
from languages import parse_families, donor_pool, DISPLAY                 # noqa: E402
FAMS = parse_families(a.families)


def POOL(lam):
    """the donor languages of the pooled F factor for recipient lam"""
    oth = donor_pool(lam, a.donor_pool, FAMS, LANGS, a.pool_always)
    assert oth, f"empty donor pool for {lam} under --donor-pool {a.donor_pool}"
    return oth



recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]
from transformers import AutoTokenizer                                    # noqa: E402
tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]),
                                    **recipe.get("tokenizer_kwargs", {}))
ids_of = lambda t: tok(render_prompt(tok, recipe, t),                     # noqa: E731
                       add_special_tokens=False)["input_ids"]

# ---- prompt set: the compose grid + the empty-slot variants ----------------
raw = {}
for f in FKEYS:
    for l in LANGS:
        raw[("full", f, l)] = full(l, f)
for l in LANGS:
    raw[("nofn", "-", l)] = nofn(l)
for f in FKEYS:
    raw[("nolang", f, "-")] = nolang(f)
raw[("carrier", "-", "-")] = CARRIER
# ---- paraphrased wordings (--donor-para / --recipient-para) -------------------
from tasks_atlas_paraphrases import PARA                                  # noqa: E402
DPARA = [int(x) for x in a.donor_para.split(",")]
assert DPARA and all(0 <= x <= 3 for x in DPARA), a.donor_para
DESCP = lambda f, p: FDESC[f] if p == 0 else PARA[f][p - 1]              # noqa: E731


def RPARA(key):
    """the recipient wording for a prompt key"""
    if a.recipient_para != "rotate":
        return int(a.recipient_para)
    li = LANGS.index(key[2]) if key[2] in LANGS else 0
    return 1 + (li + FKEYS.index(key[1])) % 3


def V(key, p):
    """the variant of a prompt key in wording p (keys without a description are unchanged)"""
    return key if p == 0 or key[0] not in ("full", "nolang") else tuple(key[:3]) + (p,)


def RV(key):
    return V(key, RPARA(key)) if key[0] in ("full", "nolang") else key


_PNEED = set(x for x in DPARA if x) | ({1, 2, 3} if a.recipient_para == "rotate" else ({int(a.recipient_para)} - {0}))
for _p in sorted(_PNEED):
    for f in FKEYS:
        for l in LANGS:
            raw[("full", f, l, _p)] = f"Write a function in {DISPLAY[l]} that {DESCP(f, _p)}. {TAIL}"
        raw[("nolang", f, "-", _p)] = f"Write a function that {DESCP(f, _p)}. {TAIL}"
if _PNEED:
    print(f"[para] donor wordings {DPARA}, recipient wording {a.recipient_para}; {len(raw)} prompts", flush=True)
FILLER = None
if a.filler == "between":
    from filler_text import make_filler, insert_between, rows_from            # noqa: E402
    FILLER = make_filler(tok)
    raw = {k: insert_between(u, FILLER, TAIL) for k, u in raw.items()}
    print(f"[filler] {len(tok(FILLER, add_special_tokens=False)['input_ids'])} tokens "
          f"between specification and tail", flush=True)

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
if a.site == "all":                 # every row, description included; the
    W0, K = 0, LFULL                    # harvest already keeps all LFULL rows
assert K >= 10, f"shared suffix only {K} tokens"
P0 = T0 = None
if FILLER is not None:                 # rows: [pad][spec][P0: passage][T0: tail]
    _t = {k: rows_from(render_prompt(tok, recipe, u), tok, TAIL)[0] for k, u in padded.items()}
    _p = {k: rows_from(render_prompt(tok, recipe, u), tok, FILLER)[0] for k, u in padded.items()}
    assert len(set(_t.values())) == 1 and len(set(_p.values())) == 1, \
        f"tail/passage rows differ across prompts: {set(_t.values())} {set(_p.values())}"
    T0, P0 = next(iter(_t.values())), next(iter(_p.values()))
    assert 0 < P0 < T0 and LFULL - T0 >= 10 and T0 - P0 > 1024, (P0, T0, LFULL)
    print(f"[filler] passage rows {P0}..{T0} ({T0 - P0} > window 1024), tail rows {T0}..{LFULL}", flush=True)
    W0, W1 = {"suffix": (T0, LFULL), "afterdesc": (P0, LFULL), "passage": (P0, T0),
              "all": (0, LFULL)}[a.site]
    K = W1 - W0
else:
    assert a.site in ("suffix", "all", "online") and not a.sever_tail, "afterdesc/passage/--sever-tail need --filler between"
print(f"[site] suffix rows {W0}..{W1} of {LFULL} (K={K})", flush=True)

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

# ---- attention-logit bias on the specification rows (--ablate-desc bias) --
# A forward pre-hook on every self-attention module. While BIAS["on"], for
# single-token steps (launch + decode) it hands the module an additive float
# mask with -beta on the specification columns; beta = inf masks them out.
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
            "sliding layer beyond its window with no mask from the model -- cannot rebuild the window"
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
if any("language_model" in n for n, _ in _ATTN):        # multimodal: text tower only
    _ATTN = [(n, m) for n, m in _ATTN if "language_model" in n]
_ATTN = [m for _, m in _ATTN]
for _m in _ATTN:
    _m.register_forward_pre_hook(_bias_hook, with_kwargs=True)

with torch.inference_mode():
    _probe = model(input_ids=enc_of(padded[("carrier", "-", "-")]),
                   use_cache=True)
KINDS = pm.profile(_probe.past_key_values, n_layers)
del _probe
PATCHED = pm.select(KINDS, "both")
KV_L = [i for i in PATCHED if KINDS[i] == pm.KV]
RC_L = [i for i in range(n_layers) if KINDS[i] != pm.KV]
print(f"[bias] hook on {len(_ATTN)} text self-attention modules (active only under --ablate-desc bias)", flush=True)
if a.ablate_desc == "bias":
    assert len(_ATTN) == len(KV_L), f"hooked {len(_ATTN)} attention modules but {len(KV_L)} KV layers"
print(f"[cache] {pm.describe(KINDS)} | kv {len(KV_L)} rec {len(RC_L)}",
      flush=True)
HYBRID = bool(RC_L) and a.recurrent

# ---- recurrent replay (ported from run_ablate_both.py) -------------------------
REC = {"on": False, "inputs": {}}
if RC_L and (a.rec_sever == "skip" or a.rec_delta == "suffix" or a.rec_contrast_arms or a.readout_offset):
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
    _l2norm = importlib.import_module(f"transformers.models.{_mt}.modeling_{_mt}").l2norm

    # ---- persistent readout offset (--readout-offset): r_t += alpha * q_t^T dR ----
    RO = {"on": False, "alpha": 1.0, "delta": {}, "fired": 0}

    def _readout(i, m):
        orig = m.recurrent_gated_delta_rule          # the decode-time rule (single tokens)

        def fn(query, key, value, g, beta, initial_state, output_final_state,
               use_qk_l2norm_in_kernel=False):
            out, st = orig(query, key, value, g=g, beta=beta, initial_state=initial_state,
                           output_final_state=output_final_state,
                           use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
            if RO["on"] and i in RO["delta"]:
                q = _l2norm(query, dim=-1, eps=1e-6) if use_qk_l2norm_in_kernel else query
                q = q.transpose(1, 2).float() * (1.0 / q.shape[-1] ** 0.5)     # [B, H, S, dk]
                dR = RO["delta"][i].to(q.device, torch.float32)                # [B, H, dk, dv]
                add = torch.einsum("bhsk,bhkv->bhsv", q, dR) * RO["alpha"]     # [B, H, S, dv]
                out = out + add.transpose(1, 2).to(out.dtype)                   # out: [B, S, H, dv]
                RO["fired"] += 1
            return out, st
        m.recurrent_gated_delta_rule = fn

    if a.readout_offset:
        for _i, _m in LIN.items():
            _readout(_i, _m)
        print(f"[readout] persistent offset hook on {len(LIN)} linear layers (alphas {a.readout_offset})", flush=True)
    # positive control: an unmodified replay must reproduce the cached state
    REC["on"], REC["inputs"] = True, {}
    with torch.inference_mode():
        _c0 = model(input_ids=enc_of(padded[("carrier", "-", "-")])[:, :-1],
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
    """final recurrent state from the recorded rule inputs with steps w0:w1
    skipped (k=v=0: they neither write nor erase; decay still applies)"""
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
    channel then specifies nothing and the state is the only source.

    KNOWN, DELIBERATELY KEPT (matches the published runs): the right edge bleeds exactly one token into the tail. The tail
    begins at the char offset of TAIL, but the token carrying its first word
    owns the space before it, so `s_ < c1` admits it -- on gemma-4-31B the
    severed prompt reads " the function solve. Return only ..." with "Name"
    removed. Every instrument and every run in this family shares the window,
    so comparisons are unaffected; tightening it mid-campaign would not be."""
    if key[0] == "carrier":
        return None
    r = render_prompt(tok, recipe, padded[key])
    o = tok(r, add_special_tokens=False,
            return_offsets_mapping=True)["offset_mapping"]
    c0 = r.find(ANCHOR)
    c1 = r.find(FILLER) if FILLER else r.find(TAIL)   # the spec ends where the passage starts
    if c0 < 0 or c1 <= c0:
        return None
    # right edge as in the published runs: the token that owns the tail's first character (it carries the space before
    # "Name") is admitted, so one tail token is severed with the specification; _tail_start would exclude it.
    c0 += len(ANCHOR)
    if a.ablate_span == "desc" and key[0] == "full":
        lead = f" in {DISPLAY[key[2]]} that "
        assert r.startswith(lead, c0), r[c0:c0 + 40]
        c0 += len(lead)
    elif a.ablate_span == "lang" and key[0] == "full":
        assert r.startswith(" in ", c0), r[c0:c0 + 40]
        c0 += len(" in ")
        c1 = c0 + len(DISPLAY[key[2]])          # the language name alone
        assert r.startswith(DISPLAY[key[2]], c0), r[c0:c0 + 40]
    rows = [i for i, (s_, e) in enumerate(o)
            if s_ < c1 and e > c0 and e > s_]
    return (min(rows), max(rows) + 1) if rows else None



def collect_rec(c):
    """recurrent matrix + convolution buffer of every linear layer, float32 on CPU"""
    rec = {}
    for i in RC_L:
        layer = c.layers[i]
        d = {}
        for nm, attr in (("conv", "conv_states"),
                         ("recurrent", "recurrent_states")):
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
    states as they stand at the RECIPIENT's patch time, i.e. after L-1 tokens, before the
    launch token. A recipient is patched at exactly that step, so donor differences must be
    taken there; harvesting after the launch token (the earlier convention) left every
    recurrent contrast one step ahead of the state it was added to."""
    want_replay = bool(RC_L) and (a.rec_delta == "suffix" or a.rec_contrast_arms)
    ids = enc_of(padded[key])
    L = ids.shape[1]
    REC["on"], REC["inputs"] = want_replay, {}
    with torch.inference_mode():
        c = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    REC["on"] = False
    out = {}
    if RC_L:
        out["_rec_native"] = collect_rec(c)      # normal prefill at L-1: matrix + conv buffer
        if want_replay:
            # suffix-only recurrent states: the same L-1 steps replayed with this prompt's
            # own specification steps skipped, so a donor-minus-baseline difference never
            # contains a specification's writes. Built beside the native states, the
            # cache itself is left untouched so the launch row below is computed normally.
            win = desc_rows(key)
            suf = {}
            with torch.inference_mode():
                for i in RC_L:
                    d = dict(out["_rec_native"][i])
                    assert "recurrent" in d, f"layer {i}: recurrent state not a plain tensor: {list(d)}"
                    St = replay_skip(REC["inputs"][i], *(win or (None, None)))
                    d["recurrent"] = St.detach().float().cpu().clone()
                    suf[i] = d
            out["_rec_suffix"] = suf
        out["_rec"] = out["_rec_suffix"] if a.rec_delta == "suffix" else out["_rec_native"]
    with torch.inference_mode():                 # the launch token: completes the KV rows
        model(input_ids=ids[:, L - 1:], past_key_values=c, use_cache=True)
    out.update({i: pm.harvest(c, i, KINDS[i], LFULL) for i in PATCHED if KINDS[i] == pm.KV})
    for i in PATCHED:                            # recurrent kinds: pm.harvest's own copy is not used
        if KINDS[i] != pm.KV:
            out[i] = pm.harvest(c, i, KINDS[i], LFULL)
    # a sliding-window layer keeps only its last rows: remember how many it
    # dropped, so every row index below can be an absolute position
    out["_off"] = {i: LFULL - c.layers[i].keys.shape[2] for i in KV_L}
    return out


NEED = set()
for f in FKEYS:
    for l in LANGS:
        NEED.add(("full", f, l))
for l in LANGS:
    NEED.add(("nofn", "-", l))
for f in FKEYS:
    NEED.add(("nolang", f, "-"))
NEED.add(("carrier", "-", "-"))
NEED |= {k for k in raw if len(k) == 4}                   # paraphrased wordings
STORE = {k: harvest(k) for k in sorted(NEED, key=str)}
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
_off0 = STORE[("carrier", "-", "-")]["_off"]
_slide = [i for i in KV_L if _off0[i]]
print(f"[cache] {len(_slide)} of {len(KV_L)} kv layers are sliding-window: they hold the last "
      f"{LFULL - _off0[_slide[0]] if _slide else LFULL} of {LFULL} rows", flush=True)
if _slide:
    assert W0 >= _off0[_slide[0]], "the site must lie inside the sliding window"


def OFF(d, i):
    """rows layer i dropped in the store/delta d (0 for a full-attention layer)"""
    return d.get("_off", {}).get(i, 0)


def mean_delta(pairs, rec_key="_rec"):
    """mean over pairs of (KV(x) - KV(y)); also recurrent if present.
    Every pair is expanded over the donor wordings (--donor-para): (x, y) -> (x@p, y@p) for p in DPARA."""
    pairs = [(V(x, p), V(y, p)) for x, y in pairs for p in DPARA]
    if ONLINE is not None:                 # online site: a recipe of context pairs, evaluated per token (online_site.py)
        return {"_groups": [list(pairs)]}
    out = {}
    for i in PATCHED:
        names = STORE[pairs[0][0]][i].keys()
        out[i] = {n: torch.stack([STORE[x][i][n].float()
                                  - STORE[y][i][n].float()
                                  for x, y in pairs]).mean(0) for n in names}
    out["_off"] = STORE[pairs[0][0]].get("_off", {})
    if RC_L:
        rec = {}
        for i in RC_L:
            if i not in STORE[pairs[0][0]].get(rec_key, {}):
                continue
            keys = STORE[pairs[0][0]][rec_key][i].keys()
            rec[i] = {k: torch.stack([STORE[x][rec_key][i][k]
                                      - STORE[y][rec_key][i][k]
                                      for x, y in pairs]).mean(0)
                      for k in keys}
        out["_rec"] = rec
    return out


def add_deltas(d1, d2):
    if "_groups" in d1:
        return {"_groups": d1["_groups"] + d2["_groups"]}
    out = {i: {n: d1[i][n] + d2[i][n] for n in d1[i]} for i in PATCHED}
    out["_off"] = d1.get("_off", {})
    if "_rec" in d1 and "_rec" in d2:
        out["_rec"] = {i: {k: d1["_rec"][i][k] + d2["_rec"][i][k]
                           for k in d1["_rec"][i]}
                       for i in d1["_rec"] if i in d2["_rec"]}
    return out


def win_norm(d):
    """norm over the patched suffix window only -- the part that is applied"""
    if "_groups" in d:
        return 1.0
    return float(torch.cat([d[i][n][:, W0 - OFF(d, i):W1 - OFF(d, i), :].reshape(-1)
                            for i in sorted(k for k in d if isinstance(k, int))
                            for n in sorted(d[i])]).norm())


def rescale_to(d, target, ref=None):
    if "_groups" in d:
        return {**d, "_match": (ref or {}).get("_groups")}
    s = target / (win_norm(d) + 1e-9)
    out = {i: {n: v * s for n, v in d[i].items()}
           for i in d if isinstance(i, int)}
    out["_off"] = d.get("_off", {})
    if "_rec" in d:
        out["_rec"] = {i: {k: v * s for k, v in dd.items()}
                       for i, dd in d["_rec"].items()}
    return out


def rescale_rec_to(d, ref):
    """d with its recurrent/conv part rescaled, per layer and tensor, to ref's norms"""
    out = {i: d[i] for i in d if isinstance(i, int)}
    out["_off"] = d.get("_off", {})
    out["_rec"] = {i: {k: v * (ref["_rec"][i][k].norm() / (v.norm() + 1e-9))
                       for k, v in dd.items()} for i, dd in d["_rec"].items()}
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
            o = OFF(delta, i)
            w, rw = v[:, W0 - o:W1 - o, :], r[:, W0 - o:W1 - o, :]
            out[i][n] = r * (w.norm() / (rw.norm() + 1e-9))
    out["_off"] = delta.get("_off", {})
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


def ablate_desc(cache, win, ctx):
    """ablate the rows at absolute positions win; ctx = tokens prefilled so far.
    Rows a sliding-window layer has already dropped are outside its window and
    are left alone (the layer cannot read them anyway)."""
    if win is None or a.ablate_desc in ("none", "mask", "bias"):
        return
    d0, d1 = win
    for i in KV_L:
        layer = cache.layers[i]
        oc = ctx - layer.keys.shape[2]
        z0, z1 = max(d0, oc) - oc, d1 - oc
        if z1 <= z0:
            continue
        for attr in ("keys", "values"):
            t = getattr(layer, attr, None)
            if t is None:
                continue
            w = t[:, :, z0:z1, :]
            if a.ablate_desc == "zero":
                t[:, :, z0:z1, :] = 0
            else:
                r = torch.randn(w.shape, generator=_gabl,
                                dtype=torch.float32).to(w.device, w.dtype)
                t[:, :, z0:z1, :] = r * (
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


def kv_patch(cache, i, delta, alpha, lo, hi, ctx, L=None):
    """M[lo:hi] += alpha * delta[lo:hi], rows addressed by ABSOLUTE position.
    ctx = tokens prefilled into the cache so far. A sliding-window layer holds
    only its last rows, so its row = position - (ctx - rows held); the
    harvested delta has its own offset (_off). Returns the rows patched."""
    layer = cache.layers[i]
    oc, od = ctx - layer.keys.shape[2], OFF(delta, i)
    a0, a1 = max(lo, oc, od), hi
    if a1 <= a0:
        return 0
    for name in ("keys", "values"):
        t = getattr(layer, name)
        d = delta[i][name].to(t.device)
        t[:, :, a0 - oc:a1 - oc, :] = (t[:, :, a0 - oc:a1 - oc, :].float()
                                       + alpha * d[:, a0 - od:a1 - od, :]).to(t.dtype)
    return a1 - a0


_PLUMB = {"done": False}


def endpoint_check(cache, target_key, L):
    """exact-arm endpoint: every PATCHED tensor of the steered recipient should equal the
    target's own (identically severed) prefill; the retained unpatched rows differ by
    construction (different padding/frame positions). Printed once."""
    with torch.inference_mode():
        tids = enc_of(padded[target_key])
        Lt = tids.shape[1]
        tc = model(input_ids=tids[:, :Lt - 1], use_cache=True).past_key_values
        ablate_desc(tc, desc_rows(target_key), Lt - 1)
    rel = lambda p_, t_: ((p_.float() - t_.float()).norm() / (t_.float().norm() + 1e-9)).item()   # noqa: E731
    tail, head = [], []
    for i in KV_L:
        for attr in ("keys", "values"):
            p_, t_ = getattr(cache.layers[i], attr), getattr(tc.layers[i], attr)
            oc = (L - 1) - p_.shape[2]
            tail.append(rel(p_[:, :, W0 - oc:], t_[:, :, W0 - oc:]))
            head.append(rel(p_[:, :, :W0 - oc], t_[:, :, :W0 - oc]))
    recd = {}
    for i in RC_L:
        for attr in ("recurrent_states", "conv_states"):
            p_, t_ = getattr(cache.layers[i], attr, None), getattr(tc.layers[i], attr, None)
            if p_ is None:
                continue
            if isinstance(p_, dict):
                for k in p_:
                    if p_[k] is not None:
                        recd.setdefault(f"{attr}.{k}", []).append(rel(p_[k], t_[k]))
            else:
                recd.setdefault(attr, []).append(rel(p_, t_))
    print(f"[endpoint] exact_native vs the target's severed prefill ({len(KV_L)} kv, {len(RC_L)} rec layers): "
          f"KV tail rows rel diff max {max(tail):.2e} (mean {sum(tail) / len(tail):.2e}); "
          + f"unpatched rows 0..{W0} rel diff mean {sum(head) / len(head):.3f}; "
          + "; ".join(f"{k} rel diff max {max(v):.2e}" for k, v in recd.items())
          + "; first logits are computed at the launch step AFTER the patch", flush=True)


@torch.inference_mode()
def generate(rec_key, delta, channel="kv", check=None, readout=None, ro_alpha=1.0):
    rec_key = RV(rec_key)                    # the recipient's wording (--recipient-para)
    ids = enc_of(padded[rec_key])
    L = ids.shape[1]
    REC["on"], REC["inputs"] = (a.rec_sever == "skip" and bool(RC_L)), {}
    cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    REC["on"] = False
    win = desc_rows(rec_key)
    ablate_desc(cache, win, L - 1)
    if a.sever_tail and a.ablate_desc != "none":   # the tail rows go too: passage = sole carrier
        for i in KV_L:
            layer = cache.layers[i]
            oc = (L - 1) - layer.keys.shape[2]
            z0, z1 = max(T0, oc) - oc, (L - 1) - oc
            if z1 > z0:
                layer.keys[:, :, z0:z1, :] = 0
                layer.values[:, :, z0:z1, :] = 0
    if a.rec_sever == "skip" and RC_L and win is not None and a.ablate_desc != "none":
        d0, d1 = win[0], min(win[1], L - 1)
        for i in RC_L:
            St = replay_skip(REC["inputs"][i], d0, d1)
            tgt = cache.layers[i].recurrent_states
            tgt.copy_(St.to(tgt.device, tgt.dtype))
    if FILLER is not None and a.passage_rows == "carrier" and rec_key != ("carrier", "-", "-"):
        car = STORE[("carrier", "-", "-")]
        for i in KV_L:                          # rows by absolute position, see kv_patch
            layer = cache.layers[i]
            oc, od = (L - 1) - layer.keys.shape[2], OFF(car, i)
            a0, a1 = max(P0, oc, od), W0
            if a1 <= a0:
                continue
            for attr in ("keys", "values"):
                t = getattr(layer, attr)
                t[:, :, a0 - oc:a1 - oc, :] = car[i][attr][:, a0 - od:a1 - od, :].to(t.device, t.dtype)
    # attention mask for the generation steps: rows excluded from attention are
    # absent -- unlike zeroed rows they take no softmax share, unlike replaced
    # rows they relay nothing. The prefill above stays unmasked so the tail
    # rows are written as they always are.
    amask = None
    if (a.ablate_desc == "mask" and win is not None) or (FILLER is not None and a.passage_rows == "mask"):
        amask = torch.ones(1, L, dtype=torch.long, device=dev)
        if a.ablate_desc == "mask" and win is not None:
            amask[:, win[0]:min(win[1], L - 1)] = 0
        if FILLER is not None and a.passage_rows == "mask" and rec_key != ("carrier", "-", "-"):
            amask[:, P0:W0] = 0
    S0, S1 = W0, W1
    hi = min(S1, L - 1)
    if delta is not None and channel in ("kv", "kvrec") and a.site != "online":
        got = {i: kv_patch(cache, i, delta, a.alpha, S0, hi, L - 1, L) for i in KV_L}
        if not _PLUMB["done"]:               # fail fast if a layer kind misses the site
            _PLUMB["done"] = True
            by = {}
            for i in KV_L:
                by.setdefault("sliding" if OFF(delta, i) else "global", set()).add(got[i])
            print(f"[plumb] site rows patched per layer kind: {by} (site {S0}..{hi})", flush=True)
            assert all(v == {hi - S0} for v in by.values()), by
        if DESC_BAND and W0 > 0:          # the rows before the suffix, band-limited
            for i in KV_L:
                if DESC_BAND[0] <= i < DESC_BAND[1]:
                    kv_patch(cache, i, delta, a.alpha, 0, W0, L - 1)
    if delta is not None and channel in ("rec", "kvrec"):
        patch_rec(cache, delta)
    if check is not None and not _PLUMB.get("endpoint"):
        _PLUMB["endpoint"] = True
        endpoint_check(cache, check, L)
    if readout is not None:
        RO.update(on=True, alpha=ro_alpha, fired=0,
                  delta={i: readout["_rec"][i]["recurrent"] for i in RC_L if "recurrent" in readout["_rec"].get(i, {})})
    if a.ablate_desc == "bias" and win is not None:
        BIAS.update(on=True, beta=a.desc_bias, cols=(win[0], min(win[1], L - 1)), fired=0)
    if ONLINE is not None and delta is not None:
        assert "_groups" in delta, "online site: delta without context groups"
        return ONLINE.generate_online(model, tok, enc_of, padded, cache, ids, amask, delta, a.alpha, stop, stop_strings, a.max_new_tokens, dev)
    step = model(input_ids=ids[:, L - 1:], past_key_values=cache,
                 use_cache=True, attention_mask=amask)
    if delta is not None and channel in ("kv", "kvrec") and S1 >= L - 1 and a.site != "online":
        for i in KV_L:                      # launch slot, patched exactly once
            kv_patch(cache, i, delta, a.alpha, L - 1, L, L, L)
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
    if readout is not None:
        if not _PLUMB.get("readout"):
            _PLUMB["readout"] = True
            print(f"[readout] alpha={ro_alpha} layers={len(RO['delta'])} fired={RO['fired']} over {len(out) + 1} steps", flush=True)
            assert RO["fired"] == len(RO["delta"]) * (len(out) + 1), (RO["fired"], len(RO["delta"]), len(out) + 1)
        RO["on"] = False
    if BIAS["on"]:
        if not _PLUMB.get("bias"):
            _PLUMB["bias"] = True
            print(f"[bias] beta={BIAS['beta']} cols={BIAS['cols']} fired={BIAS['fired']} over "
                  f"{len(out) + 1} steps, mask {BIAS['shape']}", flush=True)
            assert BIAS["fired"] == len(_ATTN) * (len(out) + 1), (BIAS["fired"], len(_ATTN), len(out) + 1)
        BIAS["on"] = False
    return tok.decode(out, skip_special_tokens=True)


@torch.inference_mode()
def generate_guided(rec_key, delta, channel, g):
    """two branches of the SAME intact recipient: plain, and with the displacement applied
    (tail KV rows, plus the recurrent states on kvrec) at alpha 1. Every step both produce
    log-probs; the guided distribution lp0 + g*(lpD - lp0), restricted to tokens within
    a factor tau of branch D's own maximum, picks the token, which is fed to BOTH caches.
    Returns (text, margin_log): margin_log = per-step (lpD - lp0) of the chosen token
    for the first 8 steps, to see whether the displacement moved the preference."""
    assert a.ablate_desc == "none", "guidance is a words-intact instrument"
    ids = enc_of(padded[rec_key])
    L = ids.shape[1]
    c0 = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    cD = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    S0, S1 = W0, W1
    hi = min(S1, L - 1)
    if channel in ("kv", "kvrec"):
        for i in KV_L:
            kv_patch(cD, i, delta, a.alpha, S0, hi, L - 1, L)
    if channel in ("rec", "kvrec"):
        patch_rec(cD, delta)
    s0 = model(input_ids=ids[:, L - 1:], past_key_values=c0, use_cache=True)
    sD = model(input_ids=ids[:, L - 1:], past_key_values=cD, use_cache=True)
    if channel in ("kv", "kvrec") and S1 >= L - 1:
        for i in KV_L:                      # launch slot on the displaced branch, once
            kv_patch(cD, i, delta, a.alpha, L - 1, L, L, L)
    c0, cD = s0.past_key_values, sD.past_key_values
    l0, lD = s0.logits[:, -1], sD.logits[:, -1]
    out, margins = [], []
    logtau = float(torch.log(torch.tensor(a.guidance_tau)))
    for _ in range(a.max_new_tokens):
        lp0 = torch.log_softmax(l0.float(), -1)
        lpD = torch.log_softmax(lD.float(), -1)
        guided = lp0 + g * (lpD - lp0)
        guided[lpD < lpD.max() + logtau] = float("-inf")     # plausibility under branch D
        nxt = int(guided.argmax(-1))
        if len(margins) < 8:
            margins.append(round(float(lpD[0, nxt] - lp0[0, nxt]), 3))
        if nxt in stop:
            break
        out.append(nxt)
        txt = tok.decode(out[-24:], skip_special_tokens=True)
        if stop_strings and any(s_ in txt for s_ in stop_strings):
            break
        t = torch.tensor([[nxt]], device=dev)
        s0 = model(input_ids=t, past_key_values=c0, use_cache=True)
        sD = model(input_ids=t, past_key_values=cD, use_cache=True)
        c0, cD = s0.past_key_values, sD.past_key_values
        l0, lD = s0.logits[:, -1], sD.logits[:, -1]
    return tok.decode(out, skip_special_tokens=True), margins


def score(txt, lam):
    """execute under the REQUESTED language's runner, never by detection"""
    if a.strip_think and "</think>" in txt:
        txt = txt.split("</think>")[-1]
    code, _ = extract_code(txt, "auto")
    entry = find_entry(lam, code) if code else None
    st, got = run_forms(lam, code, entry)     # memoised calling-form sweep
    return {"code": code, "entry": entry, "status": st,
            "impl": which_function(got), "built": st == "ok"}


outdir = Path(a.outdir or ROOT / "results" / f"tci_{a.model}")
outdir.mkdir(parents=True, exist_ok=True)
CHANNELS = a.channels or (["kv"] + (["kvrec", "rec"] if HYBRID else []))
NOCH_ARMS = ("none", "prompted", "prompted_sev")   # channel-free: written ONCE,
                                     # tagged kv, whatever --channels says
DEF = Deferred(workers=2)            # compile/run off the GPU's critical path
LANGS_T = [l for l in a.langs if l in LANGS]
n_done = 0

SRC = "python"        # source language for the language-transport recipient

for op in a.ops:
    if op in ("install", "compose"):
        cells = [(A, lam) for A in a.funcs for lam in LANGS_T]
    else:
        cells = [(p, lam) for p in a.pairs for lam in LANGS_T]

    for cell, lam in cells:
        # ---- language-factor (CFS_L) variant of this cell ----------------
        AL = cell.split(":")[0]
        othf = [f for f in FKEYS if f != AL]
        if lam != SRC and op != "compose" and not a.only_base and not a.no_lfactor:
            if op == "install":
                Lrec, Ldelta = ("nolang", AL, "-"), mean_delta(
                    [(("full", f, lam), ("nolang", f, "-")) for f in othf])
            else:
                Lrec, Ldelta = ("full", AL, SRC), mean_delta(
                    [(("full", f, lam), ("full", f, SRC)) for f in othf])
            Lexact = (mean_delta([(("full", AL, lam), ("full", AL, SRC))])
                      if op == "transport" else
                      mean_delta([(("full", AL, lam), ("nolang", AL, "-"))]))
            # --- specificity controls for the LANGUAGE factor --------------
            # exact mirrors of wrongf/meanf. wrongl is the LOO estimate for a
            # DIFFERENT target language: if the estimate carries the language
            # and not merely "some language", the recipient should come back
            # in lam2, so the text is scored under lam2 as well as under the
            # requested lam. meanl is the uniform mean over the target
            # languages -- language-shaped state with no identity.
            # (The strategy factor gets no analogue: it is a binary choice,
            # so "a different value" is just the recipient's own baseline and
            # "the mean over values" sits halfway between two options.
            # Neither isolates identity the way it does for f and lambda.)
            _rot = [x for x in LANGS_T if x != SRC]
            lam2 = (_rot[(_rot.index(lam) + 1) % len(_rot)]
                    if lam in _rot and len(_rot) > 1 else None)
            if lam2 is None:
                wrongl = meanl = None
            elif op == "install":
                wrongl = mean_delta([(("full", f, lam2), ("nolang", f, "-"))
                                     for f in othf])
                meanl = mean_delta([(("full", f, x), ("nolang", f, "-"))
                                    for x in _rot for f in othf])
            else:
                wrongl = mean_delta([(("full", f, lam2), ("full", f, SRC))
                                     for f in othf])
                meanl = mean_delta([(("full", f, x), ("full", f, SRC))
                                    for x in _rot for f in othf])
            armsL = [("none", None), ("delta", Ldelta), ("exact", Lexact),
                     ("rand", rand_matched(Ldelta, 1))]
            if wrongl is not None:
                armsL += [("wrongl", wrongl), ("meanl", meanl)]
            for ch in CHANNELS:
                for arm, dd in armsL:
                    if arm in NOCH_ARMS and ch != CHANNELS[0]:
                        continue
                    cht = "kv" if arm in NOCH_ARMS else ch
                    tagL = (f"L_{op}_{cell.replace(':', '-')}_{lam}_{cht}"
                            f"_{arm}")
                    fpL = outdir / f"c_{tagL}.json"
                    if fpL.exists():
                        continue
                    txt = generate(Lrec, dd, ch)

                    def _scoreL(txt=txt, lam=lam, arm=arm, lam2=lam2):
                        sc = score(txt, lam)
                        # for the wrong-language arm, also run the SAME text
                        # under the language that was injected: "did not build
                        # in lam" and "built in lam2 instead" are different
                        # findings, and only the second shows the estimate
                        # carries identity
                        extra = {}
                        if arm == "wrongl":
                            sc2 = score(txt, lam2)
                            extra = {"injected_lang": lam2,
                                     "built_injected": sc2["built"],
                                     "impl_injected": sc2["impl"]}
                        return sc, extra

                    def _emitL(res, fpL=fpL, tagL=tagL, op=op, cell=cell,
                               lam=lam, cht=cht, arm=arm, AL=AL, Lrec=Lrec,
                               txt=txt, lam2=lam2):
                        global n_done
                        sc, extra = res
                        fpL.write_text(json.dumps(
                            {"op": op, "factor": "L", "cell": cell, "lang": lam,
                             "channel": cht, "arm": arm, "want": AL,
                             "recipient": list(Lrec), "alpha": a.alpha,
                             "ablate_desc": a.ablate_desc,
                             "hit": bool(sc["built"]), "hit_fn": sc["impl"] == AL,
                             **extra, **sc, "text": txt}, indent=1))
                        n_done += 1
                        print(f"[{tagL}] built={sc['built']} impl={sc['impl']}"
                              + (f" built_in_{lam2}={extra['built_injected']}"
                                 if extra else ""), flush=True)
                    DEF.submit(_scoreL, (), _emitL)
            DEF.flush()

        if a.only_lfactor:
            continue
        if op == "transport":
            A, B = cell.split(":")
            rec_key = ("full", B, lam)
            oth = POOL(lam)
            delta = mean_delta([(("full", A, l), ("full", B, l)) for l in oth])
            want = A
        elif op == "install":
            A, B = cell, None
            rec_key = ("nofn", "-", lam)
            oth = POOL(lam)
            delta = mean_delta([(("full", A, l), ("nofn", "-", l))
                                for l in oth])
            want = A
        else:
            A, B = cell, None
            rec_key = ("carrier", "-", "-")
            oth = POOL(lam)
            othf_c = [f for f in FKEYS if f != A]
            othf = [f for f in FKEYS if f != A]
            f_A = mean_delta([(("full", A, l), ("nofn", "-", l)) for l in oth])
            l_lam = mean_delta([(("full", f, lam), ("nolang", f, "-"))
                                for f in othf_c])
            delta = add_deltas(f_A, l_lam)      # compose = both factors
            want = A

        # --- specificity controls, built from the SAME LOO estimates ------
        # wrongf: the LOO estimate for a different value of this factor. If
        #   the estimate carries the value and not merely "some factor
        #   content", this should fail on the target and install the donor
        #   value instead -- impl is logged either way.
        # meanf: the uniform mean of the per-value LOO estimates. Generic,
        #   on-manifold, carries no value. Tests whether patching the severed
        #   hole with factor-shaped state is enough on its own.
        if op == "transport":
            wrong_src = next(pp for pp in a.pairs
                             if pp.split(":")[0] != A) .split(":")
            wrongf = mean_delta([(("full", wrong_src[0], l),
                                  ("full", wrong_src[1], l)) for l in oth])
            meanf = mean_delta([(("full", x, l), ("full", y, l))
                                for pp in a.pairs
                                for x, y in [pp.split(":")]
                                for l in oth])
        elif op == "install":
            wrong_v = next(f for f in FKEYS if f != A)
            wrongf = mean_delta([(("full", wrong_v, l), ("nofn", "-", l))
                                 for l in oth])
            meanf = mean_delta([(("full", f, l), ("nofn", "-", l))
                                for f in FKEYS for l in oth])
        else:
            # compose has no wrong-value arm because it would not be a new
            # intervention: the recipient is the bare carrier, identical for
            # every cell, so injecting other values IS the delta arm of the
            # cell those values name. The 4x4 grid is already the full
            # cross-classification.
            # The mean arm is new. Neither summand carries identity, and the
            # compose score is a conjunction -- right function AND right
            # language -- so a coincidental hit needs both to coincide.
            # wrong value: BOTH factors taken for values other than the
            # target's. Redundant with the delta arm of the cell those values
            # name -- the carrier is the same for every cell -- but run and
            # plotted so the control row has no gap.
            _wA = next(f for f in FKEYS if f != A)
            _wl = next(x for x in LANGS_T if x != lam)
            wrongf = add_deltas(
                mean_delta([(("full", _wA, l), ("nofn", "-", l))
                            for l in oth]),
                mean_delta([(("full", f, _wl), ("nolang", f, "-"))
                            for f in othf_c]))
            meanf = add_deltas(
                mean_delta([(("full", f, l), ("nofn", "-", l))
                            for f in FKEYS for l in oth]),
                mean_delta([(("full", f, x), ("nolang", f, "-"))
                            for f in othf_c for x in LANGS_T]))

        # exact: the same-cell contrast, nothing held out (F factor)
        if op == "transport":
            exact = mean_delta([(("full", A, lam), ("full", B, lam))])
        elif op == "install":
            exact = mean_delta([(("full", A, lam), ("nofn", "-", lam))])
        else:
            exact = add_deltas(
                mean_delta([(("full", A, lam), ("nofn", "-", lam))]),
                mean_delta([(("full", A, lam), ("nolang", A, "-"))]))

        if a.match_norm and op == "compose":
            delta = rescale_to(delta, win_norm(exact), exact)
            # the mean arm is a sum of factors too, so it inflates the same
            # way -- dose it exactly as the delta arm is dosed, or the
            # comparison is between an intervention and an overdose
            meanf = rescale_to(meanf, win_norm(exact), exact)
            wrongf = rescale_to(wrongf, win_norm(exact), exact)
        for ch in CHANNELS:
            fixed_arms = False
            if a.rec_contrast_arms and ch == "kvrec" and op == "transport":
                # the recurrent contrast three ways on the SAME recipient (see --rec-contrast-arms)
                pp_ = [(("full", A, l), ("full", B, l)) for l in oth]
                pe_ = [(("full", A, lam), ("full", B, lam))]
                d_nat, e_nat = mean_delta(pp_, "_rec_native"), mean_delta(pe_, "_rec_native")
                d_suf, e_suf = mean_delta(pp_, "_rec_suffix"), mean_delta(pe_, "_rec_suffix")
                arms_ = [("delta_native", d_nat), ("exact_native", e_nat),
                         ("delta_suffix", d_suf), ("exact_suffix", e_suf),
                         ("delta_suffix_nm", rescale_rec_to(d_suf, d_nat)),
                         ("exact_suffix_nm", rescale_rec_to(e_suf, e_nat)),
                         ("rand_native", rand_matched(d_nat, 0))]
                fixed_arms = True
            else:
                arms_ = [("none", None), ("delta", delta), ("exact", exact),
                         ("rand", rand_matched(delta, 0))]
                if wrongf is not None:
                    arms_ += [("wrongf", wrongf)]
                if meanf is not None:
                    arms_ += [("meanf", meanf)]
            for arm, dd in arms_:
                if a.arms and arm not in a.arms and not fixed_arms:
                    continue
                if arm in NOCH_ARMS and ch != CHANNELS[0]:
                    continue
                cht = "kv" if arm in NOCH_ARMS else ch
                if a.skip_base and arm in ("none", "delta", "rand"):
                    continue
                if a.only_base and arm not in ("none", "delta", "rand"):
                    continue
                tag = f"{op}_{cell.replace(':', '-')}_{lam}_{cht}_{arm}"
                fp = outdir / f"c_{tag}.json"
                if fp.exists():
                    continue
                txt = generate(rec_key, dd, ch,
                               check=("full", A, lam) if arm == "exact_native" else None)
                injected = None
                if arm == "wrongf":
                    injected = (wrong_src[0] if op == "transport" else
                                wrong_v if op == "install" else
                                f"{_wA}:{_wl}")

                def _emitF(sc, fp=fp, tag=tag, op=op, cell=cell, lam=lam,
                           injected=injected, cht=cht, arm=arm, want=want,
                           rec_key=rec_key, txt=txt):
                    global n_done
                    rec = {"op": op, "factor": "F", "cell": cell, "lang": lam,
                           "injected_value": injected,
                           "donor_pool": a.donor_pool, "pool_langs": oth,
                           "channel": cht, "arm": arm, "want": want,
                           "recipient": list(rec_key),
                           "alpha": a.alpha, "ablate_desc": a.ablate_desc,
                           "hit": sc["impl"] == want, **sc, "text": txt}
                    fp.write_text(json.dumps(rec, indent=1))
                    n_done += 1
                    print(f"[{tag}] impl={sc['impl']} built={sc['built']} "
                          f"hit={sc['impl'] == want}", flush=True)
                DEF.submit(score, (txt, lam), _emitF)
        if a.readout_offset and op == "transport" and RC_L:
            pp_ = [(("full", A, l), ("full", B, l)) for l in oth]
            pe_ = [(("full", A, lam), ("full", B, lam))]
            d_nat, e_nat = mean_delta(pp_, "_rec_native"), mean_delta(pe_, "_rec_native")
            for ro_a in a.readout_offset:
                atag = f"a{ro_a:g}".replace(".", "p")
                for arm, dd in (("ro_delta", d_nat), ("ro_exact", e_nat), ("ro_rand", rand_matched(d_nat, 0))):
                    tag = f"{op}_{cell.replace(':', '-')}_{lam}_kv_{arm}_{atag}"
                    fp = outdir / f"c_{tag}.json"
                    if fp.exists():
                        continue
                    txt = generate(rec_key, dd, "kv", readout=dd, ro_alpha=ro_a)   # tail KV as usual + readout offset

                    def _emitR(sc, fp=fp, tag=tag, op=op, cell=cell, lam=lam, arm=arm, ro_a=ro_a,
                               want=want, rec_key=rec_key, txt=txt, atag=atag):
                        global n_done
                        fp.write_text(json.dumps(
                            {"op": op, "factor": "F", "cell": cell, "lang": lam, "channel": "kv+readout",
                             "arm": f"{arm}_{atag}", "base_arm": arm, "readout_alpha": ro_a,
                             "want": want, "recipient": list(rec_key), "alpha": a.alpha,
                             "ablate_desc": a.ablate_desc, "donor_pool": a.donor_pool,
                             "hit": sc["impl"] == want, **sc, "text": txt}, indent=1))
                        n_done += 1
                        print(f"[{tag}] impl={sc['impl']} built={sc['built']} hit={sc['impl'] == want}", flush=True)
                    DEF.submit(score, (txt, lam), _emitR)
            DEF.flush()

        if a.guidance and op == "transport":
            gch = CHANNELS[-1]                  # kv on attention models, kvrec on hybrids
            for g in a.guidance:
                gtag = f"g{g:g}".replace(".", "p")
                for arm, dd in (("delta", delta), ("exact", exact), ("rand", rand_matched(delta, 0))):
                    tag = f"{op}_{cell.replace(':', '-')}_{lam}_{gch}_{arm}_{gtag}"
                    fp = outdir / f"c_{tag}.json"
                    if fp.exists():
                        continue
                    txt, margins = generate_guided(rec_key, dd, gch, g)

                    def _emitG(sc, fp=fp, tag=tag, op=op, cell=cell, lam=lam, gch=gch, arm=arm,
                               g=g, want=want, rec_key=rec_key, txt=txt, margins=margins):
                        global n_done
                        fp.write_text(json.dumps(
                            {"op": op, "factor": "F", "cell": cell, "lang": lam, "channel": gch,
                             "arm": f"{arm}_{gtag}", "base_arm": arm, "guidance": g,
                             "guidance_tau": a.guidance_tau, "margins_first8": margins,
                             "want": want, "recipient": list(rec_key), "alpha": a.alpha,
                             "ablate_desc": a.ablate_desc, "donor_pool": a.donor_pool,
                             "hit": sc["impl"] == want, **sc, "text": txt}, indent=1))
                        n_done += 1
                        print(f"[{tag}] impl={sc['impl']} built={sc['built']} hit={sc['impl'] == want}", flush=True)
                    DEF.submit(score, (txt, lam), _emitG)
            DEF.flush()

        # prompted: just ask. two versions -- words intact, and words
        # severed exactly as in the intervention arms (= how much of the
        # ask survives in the compiled store alone)
        # prompted: the fallback reference -- just ask, words intact.
        # (a severed version is meaningless here: ablating the whole spec
        # leaves the empty carrier, which specifies nothing.)
        # prompted: ask, words intact (the ceiling). prompted_sev: ask, then
        # ablate the target prompt's own specification exactly as the
        # intervention arms ablate the recipient's -- what the compiled store
        # carries unsteered, the ceiling an injected factor can be held to
        for parm, pabl in (() if a.only_base
                           else (("prompted", "none"), ("prompted_sev", a.ablate_desc))):
         if a.arms and parm not in a.arms:
             continue
         fp = outdir / (f"c_{op}_{cell.replace(':', '-')}_{lam}_kv_"
                        f"{parm}.json")
         if not fp.exists():
            _abl, a.ablate_desc = a.ablate_desc, pabl
            txt = generate(("full", want, lam), None, "kv")
            a.ablate_desc = _abl

            def _emitP(sc, fp=fp, op=op, cell=cell, lam=lam, parm=parm,
                       want=want, pabl=pabl, txt=txt):
                global n_done
                fp.write_text(json.dumps(
                    {"op": op, "cell": cell, "lang": lam, "channel": "kv",
                     "arm": parm, "want": want,
                     "recipient": ["full", want, lam], "alpha": 0.0,
                     "ablate_desc": pabl, "ablate_span": a.ablate_span, "desc_bias": a.desc_bias,
                     "hit": sc["impl"] == want,
                     **sc, "text": txt}, indent=1))
                n_done += 1
                print(f"[{op}_{cell}_{lam}_{parm}] impl={sc['impl']} "
                      f"hit={sc['impl'] == want}", flush=True)
            DEF.submit(score, (txt, lam), _emitP)
            DEF.flush()

DEF.flush()
(outdir / "meta.json").write_text(json.dumps(
    {"model": a.model, "ops": a.ops, "langs": LANGS_T, "funcs": a.funcs,
     "pairs": a.pairs, "alpha": a.alpha, "ablate_desc": a.ablate_desc,
     "site": [W0, W1], "align": "pad", "K": K, "donor_para": DPARA, "recipient_para": a.recipient_para, "online": (ONLINE.info() if ONLINE is not None else None), "channels": CHANNELS, "hybrid": HYBRID,
     "rec_sever": a.rec_sever, "rec_delta": a.rec_delta, "arms": a.arms,
     "desc_band": DESC_BAND, "filler": a.filler, "passage_rows": a.passage_rows,
     "ablate_span": a.ablate_span, "site_name": a.site, "sever_tail": a.sever_tail,
     "desc_bias": a.desc_bias, "rec_contrast_arms": a.rec_contrast_arms,
     "guidance": a.guidance, "guidance_tau": a.guidance_tau, "readout_offset": a.readout_offset,
     "passage_rows_span": [P0, T0] if FILLER is not None else None,
     "donor_pool": a.donor_pool, "families": FAMS, "pool_always": a.pool_always,
     "kv_layers": len(KV_L), "rec_layers": len(RC_L)}, indent=1))
print(f"DONE ({n_done} new cells)", flush=True)

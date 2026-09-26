"""Transport, installation and composition (EXPERIMENT_GUIDE sections 5-7) for the program-call process.

Draft: transport   Z(b, c*) + mean_{c != c*}[Z(a, c) - Z(b, c)]        -> (a, c*)
       installation Z(0_phi, c*) + mean_{c != c*}[Z(a, c) - Z(0_phi, c)] -> (a, c*)
       composition  Z(0_FL) + F_hat_a + L_hat_l                         -> (a, l)

State Z = the K/V written at the context positions over all layers (cache edit: the context's own
computation is frozen, generated tokens read the edited memory). Everything is estimated from
realization-matched renderings of the same latent PATH: a path fixes the content/filler mode sequence
and the shared/language coin at every content position, and carries one event stream per function.
  language pairs   the same events rendered in language a vs b (position-identical, tokens differ only
                   where the coin says "language")
  function pairs   the same path with function a's vs b's event stream (the same positions are content)
  0_L carrier      the same events rendered all-shared (function evidence, no language evidence)
  0_F carrier      symbols replaced by blanks, returns kept and rendered in the language (language
                   evidence, no function evidence: returns are function-neutral)
  0_FL carrier     the 0_F carrier with the returns shared
Three estimators of the displacement, applied at the recipient's context positions:
  position  mean difference at every position, added position-wise (the guide's estimator)
  token     mean difference per token TYPE (symbol 0/1/2, return, blank), added by the recipient's own
            token types; cancels the token-identity component of the state
  pooled    mean difference over all content positions, one vector added at every content position
Controls per cell: exact (recipient's own context), norm-matched random, wrong value, mean over values,
the unedited recipient, and the prompted ceiling (the recipient's path rendered with the target value).

Readouts, all on outputs: after the edit, the target function's designed unit test (return a b -> a+b;
return s -> not s; return s s s -> sticky; suffix rendered shared) and, for the language, the model's
language-specific mass on the next token; with --generate, sampled continuations scored by the exact
observer's per-function / per-language log-likelihood (argmax = exhibited) and by signature statistics.
The observer's own value on the ceiling context is reported next to every model number.

  RUN_ID=<id> uv run python programming_minimal_model/toy/program_transport.py --step last --generate 2

Paper-faithful mode (the draft's figure "Factors are Mostly Independent"): append token-identical blanks
(--suffix-blanks 16), edit ONLY those blanks for every condition and control (--edit-positions suffix,
position-wise estimator), and zero the cached K/V at every content position before the readout (--sever),
so the readout can only use what the prefill compiled into the blanks. The prompted ceiling stays intact
(the paper's "prompted, no intervention" line); a "ceiling_severed" job (target context, severed, no edit) is
added, which is the compiler-like prompt-state test inside the same cells. Controls then include "shuffled" =
the paper's random labelling (donor labels permuted per context before the mean). The joint readout
fl_exhibits = function test passes AND the language readout is the target language is the analogue of the
paper's score (the program runs correctly AND exhibits the injected factor); --n-probes k draws k designed
tests per (cell, recipient) and f_exhibits_all / fl_exhibits_all require all k to pass (the paper's "all
unit tests pass").

  uv run python programming_minimal_model/toy/program_transport.py --run-id <id> --suffix-blanks 16 --edit-positions suffix \
      --sever --estimators position --alphas 1,2 --n-probes 3 --out results/programming_toy_v2_paper
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_REPO / "training"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import program_process as pp
import torch
from behavioral_checks import (
    batched,
    cache_hooks,
    cache_kv,
    last_probs,
    list_checkpoints,
    load_run,
    sample_continuations,
)

ESTIMATORS = ("position", "token", "pooled")
CONDITIONS = ("loo", "exact", "random", "wrong", "mean", "shuffled")
N_TYPES = 5                      # symbol 0, 1, 2, return, blank
TYPE_BLANK = 4


# --- paths and renderings ----------------------------------------------------------------------------

def neutral_events(n_c, observer, params, rng):
    """A content-event stream that keeps the observer's FUNCTION posterior as flat as possible: at each step
    the symbol (shared-rendered, language 0) minimising KL(posterior over functions || uniform) is chosen; a
    return is emitted whenever the watcher completes the pattern (as the process does 90% of the time).
    In-support for every function; the language is set by the rendering later."""
    lay = params.layout
    nxt = _KMP = pp._kmp_table(params.pattern, pp.N_SYM)
    length = len(params.pattern)
    w = observer.w0[None, :].copy()
    b = [observer.init[c][None, :].copy() for c in range(observer.n_comp)]
    m, out = 0, []
    n_langs = params.n_langs
    for _ in range(n_c):
        if m == length:
            k = pp.RET
        else:
            best, k = None, None
            for cand in rng.permutation(pp.N_SYM):
                w2, _ = observer._step(w, b, np.array([lay.shared(int(cand))]))
                post = w2.reshape(n_langs, pp.N_FUNC).sum(0)
                post = post / post.sum()
                score = float(np.sum(post * np.log(post * pp.N_FUNC + 1e-30)))
                if best is None or score < best:
                    best, k = score, int(cand)
        w, b = observer._step(w, b, np.array([lay.shared(k)]))
        m = 0 if k == pp.RET else nxt[m][k]
        out.append(k)
    return out


def make_paths(params, rng, n, T, observer=None, suffix_blanks=0):
    """Realization-matched material: mode path, coin, one event stream per function, and (if an observer is
    given) a function-neutral event stream for the symbol-bearing installation carrier. `suffix_blanks`
    appends that many filler positions to every path (the paper's token-identical suffix)."""
    paths = []
    for _ in range(n):
        modes = pp.language_mode_path(T, rng, params)
        if suffix_blanks:
            modes = np.concatenate([modes, np.zeros(suffix_blanks, dtype=bool)])
        n_c = int(modes.sum())
        coin = rng.random(n_c) < (1.0 - params.q_shared)          # True = language-rendered
        if n_c:
            coin[0] = True                                          # the first content token always shows the language
        events = {f: pp.simulate_function_machine(f, n_c, rng, params, fire="sample") for f in range(pp.N_FUNC)}
        if observer is not None:
            events["neutral"] = neutral_events(n_c, observer, params, rng)
        paths.append({"modes": modes, "coin": coin, "events": events})
    return paths


def render_path(path, f, lang, lay, how="full"):
    """how = full (coin rendering), shared (0_L), carrier (0_F: returns kept, symbols blanked),
    carrier_shared (0_FL), neutral (function-ambiguous symbols, coin rendering), neutral_shared."""
    out, j = [], 0
    for is_content in path["modes"]:
        if not is_content:
            out.append(lay.blank)
            continue
        k = path["events"]["neutral" if how.startswith("neutral") else f][j]
        rendering = "lang" if (path["coin"][j] and how in ("full", "carrier", "neutral")) else "shared"
        if how in ("carrier", "carrier_shared"):
            out.append(lay.render(pp.RET, lang, rendering) if (k == pp.RET or j == 0) else lay.blank)
        else:
            out.append(lay.render(k, lang, rendering))
        j += 1
    return out


def token_types(tokens, lay):
    out = []
    for t in tokens:
        k = lay.token_symbol(t)
        out.append(TYPE_BLANK if k is None else k)
    return np.array(out)


# --- donor statistics --------------------------------------------------------------------------------

class Stats:
    """Per rendering key and hook name: position means [P, H, D], type means [5, H, D], content mean [H, D]."""

    def __init__(self):
        self.M, self.TM, self.PM = {}, {}, {}


def rendering_keys(n_langs):
    keys = [("neutral_shared", 0, None)]
    for lang in range(n_langs):
        keys.append(("neutral", 0, lang))
    for f in range(pp.N_FUNC):
        keys.append(("shared", f, None))
        keys.append(("carrier_shared", f, None))
        for lang in range(n_langs):
            keys.append(("full", f, lang))
            keys.append(("carrier", f, lang))
    return keys


def donor_stats(model, paths, params, bos, device, batch) -> Stats:
    lay = params.layout
    st = Stats()
    n = len(paths)
    for how, f, lang in rendering_keys(params.n_langs):
        pos_sum, type_sum, type_cnt, con_sum, con_cnt = None, None, np.zeros(N_TYPES), None, 0
        for idx in batched(list(range(n)), batch):
            toks = [render_path(paths[i], f, lang if lang is not None else 0, lay, how) for i in idx]
            types = np.array([token_types(t, lay) for t in toks])                       # [B, T]
            cache = cache_kv(model, torch.as_tensor([[bos, *t] for t in toks], dtype=torch.long, device=device))
            masks = torch.zeros((len(idx), 1 + types.shape[1], N_TYPES), dtype=torch.float32, device=device)
            for ty in range(N_TYPES):
                masks[:, 1:, ty] = torch.as_tensor(types == ty, dtype=torch.float32, device=device)
            content = masks[:, :, :TYPE_BLANK].sum(-1)                                     # [B, P]
            type_cnt += masks.sum((0, 1)).cpu().numpy()
            con_cnt += float(content.sum())
            if pos_sum is None:
                pos_sum = {k: v.sum(0) for k, v in cache.items()}
                type_sum = {k: torch.einsum("bphd,bpt->thd", v, masks) for k, v in cache.items()}
                con_sum = {k: torch.einsum("bphd,bp->hd", v, content) for k, v in cache.items()}
            else:
                for k, v in cache.items():
                    pos_sum[k] += v.sum(0)
                    type_sum[k] += torch.einsum("bphd,bpt->thd", v, masks)
                    con_sum[k] += torch.einsum("bphd,bp->hd", v, content)
        key = (how, f, lang)
        st.M[key] = {k: (v / n).cpu().numpy() for k, v in pos_sum.items()}
        tc = torch.as_tensor(np.maximum(type_cnt, 1), dtype=torch.float32)[:, None, None]
        st.TM[key] = {k: (v.cpu() / tc).numpy() for k, v in type_sum.items()}
        st.PM[key] = {k: (v / max(con_cnt, 1)).cpu().numpy() for k, v in con_sum.items()}
    return st


# --- edit-vector algebra ---------------------------------------------------------------------------------

def dsub(A, B):
    return {k: A[k] - B[k] for k in A}


def dadd(A, B):
    return {k: A[k] + B[k] for k in A}


def dmean(ds):
    return {k: np.mean([d[k] for d in ds], axis=0) for k in ds[0]}


def dnorm_matched_noise(d, rng):
    """Gaussian noise with the same norm as `d` at every leading index (position or type; whole vector for pooled)."""
    out = {}
    for k, v in d.items():
        lead = v.shape[0] if v.ndim == 3 else 1
        noise = rng.standard_normal(v.shape).astype(v.dtype)
        fn, fv = noise.reshape(lead, -1), v.reshape(lead, -1)
        scale = np.linalg.norm(fv, axis=1) / np.maximum(np.linalg.norm(fn, axis=1), 1e-12)
        out[k] = (fn * scale[:, None]).reshape(v.shape)
    return out


def dcos(A, B):
    a = np.concatenate([A[k].ravel() for k in A])
    b = np.concatenate([B[k].ravel() for k in A])
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def diff(st: Stats, key_a, key_b, est):
    store = {"position": st.M, "token": st.TM, "pooled": st.PM}[est]
    return dsub(store[key_a], store[key_b])


def random_pair(values, rng):
    """An ordered pair of two distinct values (the paper's random-labelling control: a contrast whose labels
    were drawn at random for that context)."""
    x, y = rng.choice(values, size=2, replace=False)
    return int(x), int(y)


def edit_vectors(st: Stats, cell, est, n_langs, rec_langs, rng):
    """All conditions' displacement for one cell under one estimator."""
    a, b, ctx, tier, factor = cell["a"], cell["b"], cell["ctx"], cell["tier"], cell["factor"]
    funcs = list(range(pp.N_FUNC))
    if tier == "transport" and factor == "F":
        third = next(f for f in funcs if f not in (a, b))
        d = lambda x, y, lang: diff(st, ("full", x, lang), ("full", y, lang), est)
        others = [lang for lang in range(n_langs) if lang != ctx]
        loo = dmean([d(a, b, lang) for lang in others])
        return {"loo": loo, "exact": d(a, b, ctx), "random": dnorm_matched_noise(loo, rng),
                "wrong": dmean([d(third, b, lang) for lang in others]),
                "mean": dmean([dmean([d(x, b, lang) for lang in others]) for x in (a, third)]),
                "shuffled": dmean([d(*random_pair(funcs, rng), lang) for lang in others])}
    if tier == "transport" and factor == "L":
        wrong_lang = next(lang for lang in range(n_langs) if lang not in (a, b))
        d = lambda x, y, f: diff(st, ("full", f, x), ("full", f, y), est)
        others = [f for f in funcs if f != ctx]
        loo = dmean([d(a, b, f) for f in others])
        return {"loo": loo, "exact": d(a, b, ctx), "random": dnorm_matched_noise(loo, rng),
                "wrong": dmean([d(wrong_lang, b, f) for f in others]),
                "mean": dmean([dmean([d(x, b, f) for f in others]) for x in range(n_langs) if x != b]),
                "shuffled": dmean([d(*random_pair(list(range(n_langs)), rng), f) for f in others])}
    if tier == "install" and factor == "F":
        third = next(f for f in funcs if f != a)
        inst = lambda x, lang: diff(st, ("full", x, lang), ("carrier", x, lang), est)
        others = [lang for lang in range(n_langs) if lang != ctx]
        loo = dmean([inst(a, lang) for lang in others])
        return {"loo": loo, "exact": inst(a, ctx), "random": dnorm_matched_noise(loo, rng),
                "wrong": dmean([inst(third, lang) for lang in others]),
                "mean": dmean([dmean([inst(x, lang) for lang in others]) for x in funcs]),
                "shuffled": dmean([inst(int(rng.choice(funcs)), lang) for lang in others])}
    if tier == "install" and factor == "L":
        wrong_lang = next(lang for lang in range(n_langs) if lang != a)
        inst = lambda x, f: diff(st, ("full", f, x), ("shared", f, None), est)
        others = [f for f in funcs if f != ctx]
        loo = dmean([inst(a, f) for f in others])
        return {"loo": loo, "exact": inst(a, ctx), "random": dnorm_matched_noise(loo, rng),
                "wrong": dmean([inst(wrong_lang, f) for f in others]),
                "mean": dmean([dmean([inst(x, f) for f in others]) for x in range(n_langs)]),
                "shuffled": dmean([inst(int(rng.integers(n_langs)), f) for f in others])}
    if tier == "install_neutral":
        third = next(f for f in funcs if f != a)
        inst = lambda x, lang: diff(st, ("full", x, lang), ("neutral", 0, lang), est)
        others = [lang for lang in range(n_langs) if lang != ctx]
        loo = dmean([inst(a, lang) for lang in others])
        return {"loo": loo, "exact": inst(a, ctx), "random": dnorm_matched_noise(loo, rng),
                "wrong": dmean([inst(third, lang) for lang in others]),
                "mean": dmean([dmean([inst(x, lang) for lang in others]) for x in funcs]),
                "shuffled": dmean([inst(int(rng.choice(funcs)), lang) for lang in others])}
    if tier == "compose_neutral":
        inst_f = dmean([diff(st, ("full", a, lang), ("neutral", 0, lang), est) for lang in range(n_langs) if lang != ctx])
        inst_l = dmean([diff(st, ("full", f, ctx), ("shared", f, None), est) for f in funcs if f != a])
        both = dadd(inst_f, inst_l)
        sh_f = dmean([diff(st, ("full", int(rng.choice(funcs)), lang), ("neutral", 0, lang), est) for lang in range(n_langs) if lang != ctx])
        sh_l = dmean([diff(st, ("full", f, int(rng.integers(n_langs))), ("shared", f, None), est) for f in funcs if f != a])
        return {"loo": both, "f_only": inst_f, "l_only": inst_l, "random": dnorm_matched_noise(both, rng), "shuffled": dadd(sh_f, sh_l),
                "exact": dadd(diff(st, ("full", a, ctx), ("neutral", 0, ctx), est), diff(st, ("full", a, ctx), ("shared", a, None), est))}
    # composition into the neutral carrier: F installed from other languages, L from other functions
    inst_f = dmean([diff(st, ("full", a, lang), ("carrier", a, lang), est) for lang in range(n_langs) if lang != ctx])
    inst_l = dmean([diff(st, ("full", f, ctx), ("shared", f, None), est) for f in funcs if f != a])
    both = dadd(inst_f, inst_l)
    sh_f = dmean([diff(st, ("full", int(rng.choice(funcs)), lang), ("carrier", int(rng.choice(funcs)), lang), est) for lang in range(n_langs) if lang != ctx])
    sh_l = dmean([diff(st, ("full", f, int(rng.integers(n_langs))), ("shared", f, None), est) for f in funcs if f != a])
    return {"loo": both, "f_only": inst_f, "l_only": inst_l, "random": dnorm_matched_noise(both, rng), "shuffled": dadd(sh_f, sh_l),
            "exact": dadd(diff(st, ("full", a, ctx), ("carrier", a, ctx), est), diff(st, ("full", a, ctx), ("shared", a, None), est))}


def apply_edit(edited, row, delta, est, types, alpha, layer_names=None, keep=None):
    """Add alpha * delta to the cached K/V of one batch row at the context positions (BOS untouched).
    `keep`: optional boolean mask over the T context positions; only those positions are edited."""
    T = len(types)
    keep = np.ones(T, dtype=bool) if keep is None else keep
    for name, v in edited.items():
        if layer_names is not None and name not in layer_names:
            continue
        d = torch.as_tensor(delta[name], dtype=v.dtype, device=v.device)
        if est == "position":
            pos = np.where(keep)[0] + 1
            v[row, torch.as_tensor(pos, device=v.device)] += alpha * d[torch.as_tensor(pos, device=v.device)]
        elif est == "token":
            pos = np.where(keep)[0] + 1
            v[row, torch.as_tensor(pos, device=v.device)] += alpha * d[torch.as_tensor(types[pos - 1], device=v.device)]
        else:
            content = np.where((types != TYPE_BLANK) & keep)[0] + 1
            v[row, torch.as_tensor(content, device=v.device)] += alpha * d


def subset_mask(types, spec, rng, suffix_k=0):
    """Which context positions to edit: 'f0.25' = a random quarter of the content positions, 'first' / 'last' =
    the first / last half of the content positions (blanks are never edited under these specs), 'suffix' =
    only the appended blank suffix (the paper's interface: edit the token-identical suffix, not the content)."""
    content = np.where(types != TYPE_BLANK)[0]
    keep = np.zeros(len(types), dtype=bool)
    if spec == "suffix":
        keep[len(types) - suffix_k:] = True
    elif spec == "first":
        keep[content[: len(content) // 2]] = True
    elif spec == "last":
        keep[content[len(content) // 2:]] = True
    else:
        k = round(float(spec[1:]) * len(content))
        keep[rng.choice(content, size=k, replace=False)] = True
    return keep


# --- cells ---------------------------------------------------------------------------------------------------

def make_cells(rec_langs):
    cells = []
    for lang in rec_langs:
        for a in range(pp.N_FUNC):
            for b in range(pp.N_FUNC):
                if a != b:
                    cells.append({"tier": "transport", "factor": "F", "a": a, "b": b, "ctx": lang})
    for f in range(pp.N_FUNC):
        for a in rec_langs:
            for b in rec_langs:
                if a != b:
                    cells.append({"tier": "transport", "factor": "L", "a": a, "b": b, "ctx": f})
    for lang in rec_langs:
        for a in range(pp.N_FUNC):
            cells.append({"tier": "install", "factor": "F", "a": a, "b": None, "ctx": lang})
    for f in range(pp.N_FUNC):
        for a in rec_langs:
            cells.append({"tier": "install", "factor": "L", "a": a, "b": None, "ctx": f})
    for lang in rec_langs:
        for a in range(pp.N_FUNC):
            cells.append({"tier": "compose", "factor": "FL", "a": a, "b": None, "ctx": lang})
    for lang in rec_langs:
        for a in range(pp.N_FUNC):
            cells.append({"tier": "install_neutral", "factor": "F", "a": a, "b": None, "ctx": lang})
            cells.append({"tier": "compose_neutral", "factor": "FL", "a": a, "b": None, "ctx": lang})
    return cells


# --- belief readout ----------------------------------------------------------------------------------
# The designed probes above read the model's next-symbol distribution AFTER a probe whose tokens are themselves
# evidence about the function (three repeats favour Mess3, two differing arguments disfavour no-repeat), so a
# model that carried no function can update on the probe and then look informed. The belief readout instead
# scores the model's probability of every three-symbol continuation after a return, teacher-forced, and
# expresses that distribution as a mixture of the three functions' exact continuation distributions. A hedger
# pays for its uncertainty on the evidence-bearing tokens, so the mixture weights recover the belief the model
# held BEFORE the probe. The return resets the function machine, so the exact table is context-independent.

AB_PAIRS = [(a, b) for a in range(pp.N_SYM) for b in range(pp.N_SYM)]


def probe_family(lay):
    """The nine prefixes [return, a, b] rendered in the shared vocabulary, in AB_PAIRS order."""
    return [pp.render([pp.RET, a, b], 0, "shared", lay=lay) for a, b in AB_PAIRS]


def exact_continuation_table(observer, lay):
    """L[f, a, b, c]: probability that function f, freshly reset by the return, emits a, b, c (each conditional
    renormalised over the three symbols, as the model's are)."""
    fam = probe_family(lay)
    comp, local, _ = observer(np.array(fam))                 # index t = state after t tokens
    L = np.zeros((pp.N_FUNC, pp.N_SYM, pp.N_SYM, pp.N_SYM))
    for i, (a, b) in enumerate(AB_PAIRS):
        qs = []
        for t in (1, 2, 3):                                  # after R (predicts a); after R,a (b); after R,a,b (c)
            q, _ = pp.function_predictives(comp[i, t], local[i, t])
            qs.append(q / np.maximum(q.sum(1, keepdims=True), 1e-30))
        L[:, a, b, :] = (qs[0][:, a] * qs[1][:, b])[:, None] * qs[2]
    return L


@torch.no_grad()
def probs_last_k(model, tokens, k, fwd_hooks=None):
    """Softmax over the vocabulary at the last k positions: [B, k, d_vocab]."""
    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks) if fwd_hooks else model(tokens)
    return torch.softmax(logits[:, -k:, :].float(), dim=-1).detach().cpu().numpy()


def fit_belief(m, L, iters=1000, tol=1e-12):
    """m [27]: the model's distribution over continuations; L [F, 27]: the functions' exact distributions.
    Mixture weights w minimising KL(m || w @ L) by expectation-maximisation on the simplex; also returns that
    divergence (how far the model is from any mixture) and KL(m || L_f) for each f (distance to each certain
    model)."""
    F = L.shape[0]
    w = np.full(F, 1.0 / F)
    for _ in range(iters):
        mix = w @ L
        resp = (w[:, None] * L) / np.maximum(mix[None, :], 1e-300)
        w_new = (m[None, :] * resp).sum(1)
        w_new /= max(w_new.sum(), 1e-300)
        done = np.abs(w_new - w).max() < tol
        w = w_new
        if done:
            break
    return w, float(pp.kl(m, w @ L)), [float(pp.kl(m, L[f])) for f in range(F)]


def recipient_tokens(cell, path, lay, ceiling=False):
    tier, factor, a, b, ctx = cell["tier"], cell["factor"], cell["a"], cell["b"], cell["ctx"]
    if tier == "transport" and factor == "F":
        return render_path(path, a if ceiling else b, ctx, lay)
    if tier == "transport" and factor == "L":
        return render_path(path, ctx, a if ceiling else b, lay)
    if tier == "install" and factor == "F":
        return render_path(path, a, ctx, lay, "full" if ceiling else "carrier")
    if tier == "install" and factor == "L":
        return render_path(path, ctx, a, lay, "full" if ceiling else "shared")
    if tier == "install_neutral":
        return render_path(path, a, ctx, lay, "full" if ceiling else "neutral")
    if tier == "compose_neutral":
        return render_path(path, a, ctx, lay, "full" if ceiling else "neutral_shared")
    return render_path(path, a, ctx, lay, "full" if ceiling else "carrier_shared")


def target_function(cell):
    if cell["factor"] == "F" or cell["factor"] == "FL":
        return cell["a"]
    return cell["ctx"]


def target_language(cell):
    if cell["factor"] == "L":
        return cell["a"]
    return cell["ctx"]


# --- main ---------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))
    ap.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow store holding the trained run (run_toy.sh trains into ./mlruns)")
    ap.add_argument("--registry-uri", default=None)
    ap.add_argument("--step", default="last")
    ap.add_argument("--n-donor-paths", type=int, default=256)
    ap.add_argument("--n-recipients", type=int, default=6)
    ap.add_argument("--rec-langs", type=int, default=4, help="recipient languages 0..k-1 (donors use all)")
    ap.add_argument("--T", type=int, default=36)
    ap.add_argument("--alphas", default="0.5,1,1.5,2", help="dose sweep for loo (controls at 1)")
    ap.add_argument("--estimators", default="position,token,pooled")
    ap.add_argument("--per-layer", action="store_true", help="also edit one layer at a time (loo, alpha 1)")
    ap.add_argument("--subsets", default="", help="also edit only a subset of positions (loo, alpha 1), e.g. f0.25,f0.5,f0.75,first,last")
    ap.add_argument("--tiers", default="", help="restrict to these tiers, e.g. transport,install_neutral")
    ap.add_argument("--suffix-blanks", type=int, default=0, help="append K blanks to every context (use with --subsets suffix)")
    ap.add_argument("--edit-positions", default="all", choices=("all", "suffix"),
                    help="suffix: every condition and control edits only the appended blanks (the paper's interface)")
    ap.add_argument("--sever-mode", default="zero", choices=("zero", "mask"),
                    help="zero: set the content positions' cached K/V to 0 (the paper's ablation; a zero key still takes "
                         "softmax weight). mask: exclude the content positions from attention (score -1e9 before the softmax), "
                         "for every query; a check on how much of the severed score is attention dilution")
    ap.add_argument("--sever-scope", default="all", choices=("all", "paper"),
                    help="all: ablate every content position in every tier (the toy's uniform ablation). paper: follow the "
                         "language-model protocol; ablate only in transport and only the positions carrying the intervened "
                         "factor (function transport: every content position, since the function lives in the correlations "
                         "between all of them; language transport: the private-vocabulary tokens only, the analogue of the "
                         "word naming the language), and nothing in installation or composition, where the carrier holds "
                         "no conflicting value")
    ap.add_argument("--sever", action="store_true",
                    help="zero the cached K/V at all content positions before the readout (paper: ablate the description after prefill); the prompted ceiling stays intact")
    ap.add_argument("--n-probes", type=int, default=1, help="designed tests per (cell, recipient); *_all fields require all to pass")
    ap.add_argument("--belief-probes", type=int, default=1,
                    help="1: also read the model's distribution over every three-symbol continuation after a return and fit the "
                         "function belief (fields f_belief*, f_believes*, fl_believes*); 0 to skip")
    ap.add_argument("--belief-threshold", type=float, default=0.9, help="f_believes: belief in the target at least this")
    ap.add_argument("--belief-threshold-strict", type=float, default=0.95, help="f_believes_strict: belief in the target at least this")
    ap.add_argument("--min-function-certainty", type=float, default=0.0,
                    help="keep only recipient paths whose content leaves the exact observer at least this sure of the true function (for every function); 0 = no filter")
    ap.add_argument("--recipient-pool", type=int, default=60, help="candidate recipient paths to draw when filtering")
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--gen-len", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="results/programming_toy_v2_transport")
    args = ap.parse_args()
    if not args.run_id:
        ap.error("pass --run-id or set RUN_ID")
    alphas = [float(v) for v in args.alphas.split(",")]
    estimators = args.estimators.split(",")
    torch.set_num_threads(args.threads)
    t0 = time.time()
    if args.edit_positions == "suffix" and not args.suffix_blanks:
        ap.error("--edit-positions suffix needs --suffix-blanks > 0")

    cfg, run, persister, process, make_model = load_run(args.run_id, args.tracking_uri, args.registry_uri, args.device)
    params = pp.ProgramParams.from_run_config(cfg)
    lay = params.layout
    bos = int(cfg.generative_process.bos_token)
    n_ctx = int(cfg.predictive_model.instance.cfg.n_ctx)
    steps = list_checkpoints(persister)
    step = steps[-1] if args.step == "last" else int(args.step)
    model = make_model(step)
    run_name = run.info.run_name or run.info.run_id
    T = args.T + args.suffix_blanks
    P = 1 + T
    assert P + 4 + (args.gen_len if args.generate else 0) <= n_ctx, "prompt too long for n_ctx; lower --T or --gen-len"
    names_by_layer = {layer: [f"blocks.{layer}.attn.hook_k", f"blocks.{layer}.attn.hook_v"] for layer in range(model.cfg.n_layers)}
    device = args.device
    observer = pp.NumpyObserver(process)
    rec_langs = list(range(min(args.rec_langs, params.n_langs)))
    out_dir = Path(args.out) / run_name / f"step_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run_name} step {step}; {params.n_langs} languages; T={T}; estimators {estimators}; recipient languages {rec_langs}")

    rng = np.random.default_rng(args.seed)
    donors = make_paths(params, rng, args.n_donor_paths, args.T, observer, args.suffix_blanks)
    ts = time.time()
    st = donor_stats(model, donors, params, bos, device, args.batch)
    print(f"donor stats from {args.n_donor_paths} paths x {len(rendering_keys(params.n_langs))} renderings in {time.time() - ts:.0f}s")
    recipients = make_paths(params, np.random.default_rng(args.seed + 1), args.n_recipients, args.T, observer, args.suffix_blanks)
    recipient_info = []
    if args.min_function_certainty > 0:
        # the same generator, so the pool's first paths ARE the unfiltered recipients; keep the paths whose content
        # (before the appended blanks) leaves the observer >= threshold sure of the true function under every function
        pool = make_paths(params, np.random.default_rng(args.seed + 1), args.recipient_pool, args.T, observer, args.suffix_blanks)
        kept = []
        for i, path in enumerate(pool):
            toks = np.array([render_path(path, f, 0, lay) for f in range(pp.N_FUNC)])
            comp, _, _ = observer(toks)
            pf = pp.function_posterior(comp[:, args.T - 1])
            cert = min(float(pf[f, f]) for f in range(pp.N_FUNC))
            n_content = int(path["modes"][:args.T].sum())
            ok = cert >= args.min_function_certainty
            print(f"  candidate recipient {i:2d}: {n_content:2d} content tokens; observer P(true function) per function "
                  f"{[round(float(pf[f, f]), 3) for f in range(pp.N_FUNC)]} -> {'kept' if ok else 'dropped'}")
            if ok:
                kept.append(path)
                recipient_info.append({"pool_index": i, "n_content": n_content, "certainty": [float(pf[f, f]) for f in range(pp.N_FUNC)]})
        if len(kept) < args.n_recipients:
            raise SystemExit(f"only {len(kept)} of {args.recipient_pool} candidate paths pass the certainty filter; raise --recipient-pool")
        recipients = kept[: args.n_recipients]
        recipient_info = recipient_info[: args.n_recipients]
        print(f"recipients: {len(recipients)} paths with observer certainty >= {args.min_function_certainty} (of {args.recipient_pool} candidates)")

    # sanity: what the observer makes of the carriers
    for how in ("full", "shared", "carrier", "carrier_shared", "neutral", "neutral_shared"):
        toks = np.array([render_path(r, 2, 1, lay, how) for r in recipients])
        comp, _, _ = observer(toks)
        pf, pl = pp.function_posterior(comp[:, -1]), pp.language_posterior(comp[:, -1])
        print(f"  observer on {how:15s}: P(function) {np.round(pf.mean(0), 2).tolist()} max {pf.max(1).mean():.2f}  P(language=1) {pl[:, 1].mean():.2f}")

    cells = make_cells(rec_langs)
    if args.tiers:
        cells = [c for c in cells if c["tier"] in args.tiers.split(",")]
    subsets = [x for x in args.subsets.split(",") if x]
    edit_rng = np.random.default_rng(args.seed + 2)
    for c in cells:
        c["edits"] = {est: edit_vectors(st, c, est, params.n_langs, rec_langs, edit_rng) for est in estimators}
        c["cosine"] = {est: dcos(c["edits"][est]["loo"], c["edits"][est]["exact"]) for est in estimators}
    for tier, factor in (("transport", "F"), ("transport", "L"), ("install", "F"), ("install", "L"), ("compose", "FL"),
                         ("install_neutral", "F"), ("compose_neutral", "FL")):
        cs = [c for c in cells if c["tier"] == tier and c["factor"] == factor]
        print(f"  {tier:9s} {factor:2s}: {len(cs)} cells; LOO-vs-exact cosine " +
              " ".join(f"{est} {np.mean([c['cosine'][est] for c in cs]):.3f}" for est in estimators))

    # jobs: (cell index, recipient index, estimator, condition, alpha, layer or None)
    jobs = []
    for ci, c in enumerate(cells):
        for r in range(len(recipients)):
            jobs.append((ci, r, "none", "none", 0.0, None))
            jobs.append((ci, r, "none", "ceiling", 0.0, None))
            if args.sever:
                jobs.append((ci, r, "none", "ceiling_severed", 0.0, None))
            for est in estimators:
                conds = ("loo", "exact", "f_only", "l_only", "random", "shuffled") if c["tier"].startswith("compose") else CONDITIONS
                for cond in conds:
                    for alpha in (alphas if cond == "loo" else [1.0]):
                        jobs.append((ci, r, est, cond, alpha, None))
                for spec in subsets:
                    jobs.append((ci, r, est, "loo", 1.0, f"sub:{spec}"))
                if args.per_layer:
                    for layer in names_by_layer:
                        jobs.append((ci, r, est, "loo", 1.0, layer))
                        if c["factor"] == "L" and c["tier"] == "transport":
                            jobs.append((ci, r, est, "loo", 1.0, f"{layer}K"))
                            jobs.append((ci, r, est, "loo", 1.0, f"{layer}V"))
    print(f"{len(jobs)} jobs; edit positions {args.edit_positions}; sever {args.sever}; probes {args.n_probes}")

    fam = probe_family(lay) if args.belief_probes else None
    L_table = exact_continuation_table(observer, lay) if args.belief_probes else None
    if args.belief_probes:
        L_flat = L_table.reshape(pp.N_FUNC, -1)
        w_chk = np.array([0.2, 0.3, 0.5])
        w_hat, fit_kl_chk, _ = fit_belief(w_chk @ L_flat, L_flat)
        assert np.abs(w_hat - w_chk).max() < 1e-6 and fit_kl_chk < 1e-9, (w_hat, fit_kl_chk)
        s = pp.N_SYM
        print("belief readout: exact continuation probabilities, e.g. P(0,0,0) per function "
              + ", ".join(f"f{f}={L_table[f, 0, 0, 0]:.4f}" for f in range(pp.N_FUNC))
              + f"; P(0,1,1 = 0+1) per function " + ", ".join(f"f{f}={L_table[f, 0, 1, 1]:.4f}" for f in range(pp.N_FUNC))
              + f"; fit self-check recovers {w_chk.tolist()} -> {np.round(w_hat, 6).tolist()}")

    records = []
    # One designed test prompt per (cell, recipient), drawn BEFORE the job loop so that every estimator,
    # condition and dose of that cell is read on the same prompt, and so that the prompts do not depend on
    # which doses / estimators / layers were requested (they used to be drawn per job, which made runs with
    # different --alphas lists replicates over prompts rather than exact re-runs).
    suffix_rng = np.random.default_rng(args.seed + 3)
    prompt_by_cell = {}
    for ci, c in enumerate(cells):
        test = next(t for t, f in pp.UNIT_TEST_FUNCTION.items() if f == target_function(c))
        for r in range(len(recipients)):
            prompt_by_cell[ci, r] = [(test, *pp.designed_suffix(test, suffix_rng)) for _ in range(args.n_probes)]
    subset_rng = np.random.default_rng(args.seed + 5)
    gen = torch.Generator().manual_seed(args.seed + 4)
    for chunk in batched(jobs, args.batch):
        prefixes, ceilings, probes, types_list = [], [], [], []
        for ci, r, est, cond, alpha, layer in chunk:
            c = cells[ci]
            toks = recipient_tokens(c, recipients[r], lay, ceiling=cond.startswith("ceiling"))
            ceil = recipient_tokens(c, recipients[r], lay, ceiling=True)
            prefixes.append([bos, *toks])
            ceilings.append(ceil)
            probes.append([(test, pp.render(suffix, target_language(c), "shared", lay=lay), target)
                           for test, suffix, target in prompt_by_cell[ci, r]])
            types_list.append(token_types(toks, lay))
        prefix = torch.as_tensor(prefixes, dtype=torch.long, device=device)
        cache = cache_kv(model, prefix)
        edited = {k: v.clone() for k, v in cache.items()}
        for row, (ci, r, est, cond, alpha, layer) in enumerate(chunk):
            if cond in ("none", "ceiling", "ceiling_severed") or alpha == 0:
                continue
            delta = cells[ci]["edits"][est][cond]
            keep = subset_mask(types_list[row], "suffix", subset_rng, args.suffix_blanks) if args.edit_positions == "suffix" else None
            if layer is None:
                names = None
            elif isinstance(layer, str) and layer.startswith("sub:"):   # a subset of the positions, all layers
                names = None
                keep = subset_mask(types_list[row], layer[4:], subset_rng, args.suffix_blanks)
            elif isinstance(layer, str):                       # "2K" / "2V": one layer, keys only or values only
                names = [n for n in names_by_layer[int(layer[:-1])] if n.endswith("hook_k" if layer[-1] == "K" else "hook_v")]
            else:
                names = names_by_layer[layer]
            apply_edit(edited, row, delta, est, types_list[row], alpha, names, keep)
        if args.sever:
            # the description ablation, applied after prefill to every row except the intact prompted ceiling.
            # sev_mask[row, pos] marks the prefix positions whose cached K/V are removed: with --sever-scope all,
            # the content positions 1..T in every tier; with --sever-scope paper, only in transport and only the
            # positions carrying the intervened factor (BOS and the appended blanks always keep their prefill state).
            sev_mask = torch.zeros(prefix.shape, dtype=torch.bool, device=device)
            for row, (ci, r, est, cond, alpha, layer) in enumerate(chunk):
                if cond == "ceiling":
                    continue
                c = cells[ci]
                if args.sever_scope == "all" or (c["tier"] == "transport" and c["factor"] == "F"):
                    sev_mask[row, 1:1 + args.T] = True
                elif c["tier"] == "transport" and c["factor"] == "L":
                    private = [lay.token_language(t) is not None for t in prefix[row, 1:1 + args.T].tolist()]
                    sev_mask[row, 1:1 + args.T] = torch.as_tensor(private, device=device)
            if args.sever_mode == "zero" and bool(sev_mask.any()):
                for v in edited.values():
                    v[sev_mask] = 0.0

            def score_mask_hook(mask):
                # exclude the marked key positions from every query's softmax; keys beyond the prefix (the appended
                # probe tokens) are never masked
                def _mask_scores(scores, hook, m=mask):
                    full = torch.zeros(scores.shape[0], scores.shape[-1], dtype=torch.bool, device=scores.device)
                    full[:, :m.shape[1]] = m
                    scores.masked_fill_(full[:, None, None, :], -1e9)
                    return scores
                return [(f"blocks.{layer}.attn.hook_attn_scores", _mask_scores) for layer in range(model.cfg.n_layers)]
        hooks = cache_hooks(edited, P)
        if args.sever and args.sever_mode == "mask" and bool(sev_mask.any()):
            hooks = hooks + score_mask_hook(sev_mask)

        # observer on the ceiling contexts = the target state; language readout at the end of the (edited) context
        comp_c, _local_c, glob_c = observer(np.array(ceilings))
        p_ctx = last_probs(model, prefix, fwd_hooks=hooks)
        # function readout after each designed probe, grouped by probe length (the three tests differ in length)
        n_pr = args.n_probes
        p_suf = [[None] * n_pr for _ in chunk]
        comp_s, local_s, glob_s = [[None] * n_pr for _ in chunk], [[None] * n_pr for _ in chunk], [[None] * n_pr for _ in chunk]
        for k in range(n_pr):
            for n_suf in sorted({len(pr[k][1]) for pr in probes}):
                rows = [i for i, pr in enumerate(probes) if len(pr[k][1]) == n_suf]
                sub_hooks = cache_hooks({kk: v[rows] for kk, v in edited.items()}, P)
                if args.sever and args.sever_mode == "mask" and bool(sev_mask[rows].any()):
                    sub_hooks = sub_hooks + score_mask_hook(sev_mask[rows])
                full = torch.cat([prefix[rows], torch.as_tensor([probes[i][k][1] for i in rows], dtype=torch.long, device=device)], dim=1)
                ps = last_probs(model, full, fwd_hooks=sub_hooks)
                cs, ls, gs = observer(np.array([[*ceilings[i], *probes[i][k][1]] for i in rows]))
                for j, i in enumerate(rows):
                    p_suf[i][k], comp_s[i][k], local_s[i][k], glob_s[i][k] = ps[j], cs[j, -1], ls[j, -1], gs[j, -1]

        beliefs = None
        if args.belief_probes:
            n_rows = len(chunk)
            pa, pb, pc = np.zeros((n_rows, 3)), np.zeros((n_rows, 3, 3)), np.zeros((n_rows, 3, 3, 3))
            for i, (a, b) in enumerate(AB_PAIRS):
                full = torch.cat([prefix, torch.as_tensor([fam[i]] * n_rows, dtype=torch.long, device=device)], dim=1)
                ps = probs_last_k(model, full, 3, fwd_hooks=hooks)           # predicting a, b, c
                for row in range(n_rows):
                    if i == 0:
                        pa[row] = pp.symbol_distribution(ps[row, 0], lay)
                    if b == 0:
                        pb[row, a] = pp.symbol_distribution(ps[row, 1], lay)
                    pc[row, a, b] = pp.symbol_distribution(ps[row, 2], lay)
            beliefs = []
            for row in range(n_rows):
                m = (pa[row][:, None, None] * pb[row][:, :, None] * pc[row]).reshape(-1)
                beliefs.append(fit_belief(m / max(m.sum(), 1e-300), L_flat))

        for row, (ci, r, est, cond, alpha, layer) in enumerate(chunk):
            c = cells[ci]
            urs, frs = [], []
            for k in range(n_pr):
                test, _, target = probes[row][k]
                fr = pp.function_readout(p_suf[row][k], comp_s[row][k], local_s[row][k], lay)
                fr_b = pp.function_readout(glob_s[row][k], comp_s[row][k], local_s[row][k], lay)
                urs.append(pp.unit_readout(p_suf[row][k], glob_s[row][k], test, target, fr["argmin"], fr_b["argmin"], lay))
                frs.append(fr)
            test, _, target = probes[row][0]
            ur, fr = urs[0], frs[0]
            lr = pp.language_readout(p_ctx[row], comp_c[row, -1], lay)
            lr_b = pp.language_readout(glob_c[row, -1], comp_c[row, -1], lay)
            tl = target_language(c)
            l_ex = int(lr["argmax"] == tl and lr["other_mass"] < 0.05)
            l_ex_b = int(lr_b["argmax"] == tl and lr_b["other_mass"] < 0.05)
            f_all = int(all(u["exhibits"] for u in urs))
            f_all_b = int(all(u["bayes_exhibits"] for u in urs))
            # nearest-exact-predictive rule on every probe: the model's symbol distribution is closer (KL) to the
            # target function's exact predictive than to either other function's (the rule the Mess3 test already uses)
            f_near_all = int(all(fr_k["argmin"] == target_function(c) for fr_k in frs))
            rec = {"kind": "unit", "tier": c["tier"], "factor": c["factor"], "a": c["a"], "b": c["b"], "ctx": c["ctx"],
                   "recipient": r, "estimator": est, "alpha": alpha,
                   "condition": cond if layer is None else (f"loo@{layer[4:]}" if str(layer).startswith("sub:") else f"loo@L{layer}"),
                   "test": test, "f_exhibits": ur["exhibits"], "f_exhibits_bayes": ur["bayes_exhibits"],
                   "f_exhibits_all": f_all, "f_exhibits_all_bayes": f_all_b, "f_exhibits_mean": float(np.mean([u["exhibits"] for u in urs])),
                   "fl_exhibits": ur["exhibits"] * l_ex, "fl_exhibits_bayes": ur["bayes_exhibits"] * l_ex_b,
                   "fl_exhibits_all": f_all * l_ex, "fl_exhibits_all_bayes": f_all_b * l_ex_b,
                   "p_target": ur["p_target"], "bayes_target": ur["bayes_target"],
                   "f_argmin_hit": int(fr["argmin"] == target_function(c)), "f_kl_target": float(fr["kl"][target_function(c)]),
                   "f_nearest_all": f_near_all, "fl_nearest_all": f_near_all * l_ex,
                   "l_exhibits": l_ex, "l_exhibits_bayes": l_ex_b,
                   "lang_mass_target": float(lr["lang_mass"][tl]), "lang_mass_target_bayes": float(lr_b["lang_mass"][tl]),
                   "l_other": lr["other_mass"], "p_bos": float(p_ctx[row][lay.bos]),
                   "kl_ctx": pp.kl(p_ctx[row][:lay.n_tokens], glob_c[row, -1]),
                   "kl_suffix": pp.kl(p_suf[row][0][:lay.n_tokens], glob_s[row][0])}
            if beliefs is not None:
                w, fit_kl, certain = beliefs[row]
                tf = target_function(c)
                w_t = float(w[tf])
                _, post_f = pp.function_predictives(comp_c[row, -1], _local_c[row, -1])
                believes = int(w_t >= args.belief_threshold)
                believes_strict = int(w_t >= args.belief_threshold_strict)
                l_belief = float(lr["lang_mass"][tl] / max(float(np.sum(lr["lang_mass"])), 1e-30))   # share of private-token mass on the target language
                rec.update({"f_belief": [float(x) for x in w], "f_belief_target": w_t, "l_belief_target": l_belief, "fl_belief": w_t * l_belief,
                            "f_belief_max_other": float(max(w[f] for f in range(pp.N_FUNC) if f != tf)),
                            "f_belief_fit_kl": fit_kl, "f_kl_certain_target": certain[tf],
                            "f_believes": believes, "f_believes_strict": believes_strict,
                            "fl_believes": believes * l_ex, "fl_believes_strict": believes_strict * l_ex,
                            "bayes_post_target": float(post_f[tf] / max(post_f.sum(), 1e-30))})
            records.append(rec)

        if args.generate:
            wanted = [i for i, (ci, r, est, cond, alpha, layer) in enumerate(chunk)
                      if layer is None and cond in ("none", "ceiling", "loo", "exact", "random", "f_only", "l_only") and alpha in (0.0, 1.0, 2.0)]
            if wanted:
                sub_hooks = cache_hooks({k: v[wanted] for k, v in edited.items()}, P)
                for _ in range(args.generate):
                    cont = sample_continuations(model, prefix[wanted], args.gen_len, gen, fwd_hooks=sub_hooks)
                    fulls = np.array([[*prefixes[i][1:], *row] for i, row in zip(wanted, cont, strict=True)])
                    comp_g, local_g, _ = observer(np.where(fulls >= lay.n_tokens, 0, fulls))   # BOS in a continuation -> harmless placeholder for filtering
                    for j, i in enumerate(wanted):
                        ci, r, est, cond, alpha, layer = chunk[i]
                        c = cells[ci]
                        s = pp.continuation_loglik(comp_g[j], local_g[j], fulls[j], start=T, lay=lay)
                        stats = pp.continuation_stats(prefixes[i][1:], cont[j], target_language(c), params)
                        # language of the continuation read off its tokens (the observer's state after a
                        # language-b prefix gives language a zero likelihood, so the per-language LL cannot score transport)
                        langs_seen = {lay.token_language(t) for t in cont[j] if t != lay.bos} - {None}
                        records.append({"kind": "gen", "tier": c["tier"], "factor": c["factor"], "a": c["a"], "b": c["b"],
                                        "ctx": c["ctx"], "recipient": r, "estimator": est, "condition": cond, "alpha": alpha,
                                        "gen_f_hit": int(s["argmax_f"] == target_function(c)),
                                        "gen_l_hit": int(langs_seen == {target_language(c)}),
                                        "gen_runs": int(stats["other_lang"] == 0 and stats["n_bos"] == 0),
                                        "gen_repeat_rate": stats["repeat_rate"], "gen_sum_correct": stats["sum_correct"],
                                        "gen_lang_share": stats["lang_share"], "gen_blank_frac": stats["blank_frac"]})

    with open(out_dir / "records.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)

    fields = ("f_exhibits", "f_exhibits_bayes", "f_exhibits_all", "f_exhibits_all_bayes", "f_exhibits_mean",
              "fl_exhibits", "fl_exhibits_bayes", "fl_exhibits_all", "fl_exhibits_all_bayes",
              "p_target", "bayes_target", "f_argmin_hit", "l_exhibits", "l_exhibits_bayes",
              "lang_mass_target", "lang_mass_target_bayes", "l_other", "p_bos", "kl_ctx", "kl_suffix",
              "gen_f_hit", "gen_l_hit", "gen_runs", "gen_repeat_rate", "gen_sum_correct", "gen_lang_share", "gen_blank_frac",
              "f_belief_target", "f_belief_max_other", "f_belief_fit_kl", "f_kl_certain_target",
              "f_believes", "f_believes_strict", "fl_believes", "fl_believes_strict", "bayes_post_target", "l_belief_target", "fl_belief",
              "f_nearest_all", "fl_nearest_all")

    def summarize(recs, keys):
        groups = defaultdict(list)
        for r in recs:
            groups[tuple(r[k] for k in keys)].append(r)
        out = {}
        for g, rs in groups.items():
            agg = {"n": len(rs)}
            for fld in fields:
                vals = np.array([r[fld] for r in rs if fld in r and r[fld] is not None], dtype=float)
                if vals.size and not np.all(np.isnan(vals)):
                    agg[fld] = float(np.nanmean(vals))
            out[g] = agg
        return out

    order = {c: i for i, c in enumerate(("ceiling", "ceiling_severed", "none", "loo", "exact", "f_only", "l_only", "random", "wrong", "mean", "shuffled"))}

    def show(title, rows, keys, cols):
        print(f"\n{title}")
        hdr = " | ".join(f"{k:>9s}" for k in keys) + " | " + f"{'n':>5s} | " + " | ".join(f"{c:>9s}" for c in cols)
        print(hdr)
        print("-" * len(hdr))
        for g in sorted(rows, key=lambda g: tuple(order.get(x.split("@")[0], 99) if isinstance(x, str) else x for x in g)):
            a = rows[g]
            print(" | ".join(f"{x!s:>9s}" for x in g) + " | " + f"{a['n']:>5d} | " + " | ".join(f"{a[c]:>9.3f}" if c in a else " " * 9 for c in cols))

    units = [r for r in records if r["kind"] == "unit" and "@" not in r["condition"]]
    layers = [r for r in records if r["kind"] == "unit" and "@" in r["condition"]]
    gens = [r for r in records if r["kind"] == "gen"]
    f_cols = ["f_exhibits", "f_exhibits_bayes", "fl_exhibits", "fl_exhibits_all", "p_target", "bayes_target", "f_argmin_hit", "kl_suffix", "l_other", "p_bos",
              "f_nearest_all", "fl_nearest_all", "f_belief_target", "l_belief_target", "fl_belief", "f_believes", "fl_believes", "f_belief_fit_kl"]
    l_cols = ["l_exhibits", "l_exhibits_bayes", "fl_exhibits", "fl_exhibits_all", "lang_mass_target", "lang_mass_target_bayes", "l_other", "kl_ctx", "p_bos"]
    tables = {}
    for tier, factor, cols in (("transport", "F", f_cols), ("install", "F", f_cols), ("install_neutral", "F", f_cols),
                               ("transport", "L", l_cols), ("install", "L", l_cols),
                               ("compose", "FL", f_cols + ["l_exhibits", "l_exhibits_bayes"]),
                               ("compose_neutral", "FL", f_cols + ["l_exhibits", "l_exhibits_bayes"])):
        rows = summarize([r for r in units if r["tier"] == tier and r["factor"] == factor], ["estimator", "condition", "alpha"])
        tables[f"{tier}_{factor}"] = rows
        show(f"{tier.upper()} {factor}: unit readout after the edit (target function's designed test / language mass; bayes = observer on the ceiling context)",
             rows, ["estimator", "condition", "alpha"], cols)
    if layers:
        rows = summarize(layers, ["tier", "factor", "estimator", "condition"])
        tables["per_layer"] = rows
        show("ONE LAYER or a SUBSET of positions edited (loo, alpha 1)", rows, ["tier", "factor", "estimator", "condition"], ["f_exhibits", "l_exhibits", "p_target", "lang_mass_target", "f_argmin_hit"])
    if gens:
        rows = summarize(gens, ["tier", "factor", "estimator", "condition", "alpha"])
        tables["generation"] = rows
        show("GENERATION: continuation's exhibited function (observer log-lik) / language (its tokens), runs, signatures", rows,
             ["tier", "factor", "estimator", "condition", "alpha"], ["gen_f_hit", "gen_l_hit", "gen_runs", "gen_sum_correct", "gen_repeat_rate", "gen_lang_share"])

    def jsonable(d):
        return {"|".join(str(x) for x in k): v for k, v in d.items()}
    summary = {"run_id": args.run_id, "run_name": run_name, "step": step, "T": T, "n_donor_paths": args.n_donor_paths,
               "n_recipients": args.n_recipients, "rec_langs": rec_langs, "alphas": alphas, "estimators": estimators,
               "suffix_blanks": args.suffix_blanks, "edit_positions": args.edit_positions, "sever": args.sever, "sever_mode": args.sever_mode, "sever_scope": args.sever_scope, "n_probes": args.n_probes,
               "min_function_certainty": args.min_function_certainty, "recipients": recipient_info,
               "belief_probes": args.belief_probes, "belief_threshold": args.belief_threshold, "belief_threshold_strict": args.belief_threshold_strict,
               "exact_continuation_table": L_table.tolist() if L_table is not None else None,
               "cosine_loo_exact": {f"{c['tier']}_{c['factor']}": c["cosine"] for c in cells[:1]},
               "tables": {k: jsonable(v) for k, v in tables.items()}, "seconds": time.time() - t0}
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\ndone in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()

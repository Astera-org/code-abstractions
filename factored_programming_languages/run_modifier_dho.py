"""Double-held-out modifier transfer with frozen weights (authors, 2026-09-13, paper v3 priority experiment).

Let M[f] be a specified transformation of a computation f. The modifier is estimated from a DONOR function in a
DONOR language,     Delta_M = Z(M[f_a], lam_a) - Z(f_a, lam_a),
and applied to a DIFFERENT base in a DIFFERENT language, f_b != f_a, lam_b != lam_a. M[f_b] never contributes to the
modifier estimate in any language, the recipient language never contributes, and the donor wording differs from the
recipient's wording. Target: executable code implementing M[f_b] in lam_b.

Two ways of supplying the base (--mode):
  sever    Z~ = Z(f_b, lam_b) + alpha * Delta_M on the recipient's own prompt, its description rows ablated after
           prefill so the words cannot re-assert f_b (the paper's persistent-state + description-ablation setup).
  install  Z~ = Z(nofn, lam_b) + beta * F(f_b) + alpha * Delta_M on a function-free carrier "Write a function in
           <lam_b>." with F(f_b) the base factor pooled over languages != lam_b (authors, 2026-09-14: removes the
           severed-state confound; the base is an installed factor, nothing in the words names a function).

Z(f, lam) = the KV rows of the shared suffix window (TAIL + chat suffix) of the padded prompt.

Grid: bases {sum, prod, max, min, altsum} x modifiers {post_sq, pre_sq, post_inc, post_dbl} x languages {python,
javascript, java, go, rust, haskell} x 2 wordings (modifier_dho_grid.py). post_sq and pre_sq coincide on prod / max /
min and differ on sum / altsum.

Phases: dev (python / javascript recipients, wording 0) selects ONE (band, alpha) [band fixed to 'all' in install
mode]; test (java / go / rust / haskell, both wordings) runs the arms at that setting.
Arms, sever mode:   none | single | pooled | rand | wrong | ceiling
Arms, install mode: base (beta F alone; the premise) | single | pooled | rand | ceil_install (beta F(M[f_b]) alone)
                    | ceiling (explicit unsevered prompt of M[f_b])
Grading: modifier_dho_grid.grade (first fenced block; whole-output kept as whole_*).

    uv run factored_programming_languages/run_modifier_dho.py --phase dev  --shard 0/6 --device cuda:0 [--mode install]
    uv run factored_programming_languages/run_modifier_dho.py --phase test --shard 0/6 --device cuda:0 [--mode install]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from elicit import render_prompt                                           # noqa: E402
from recipes import RECIPES                                                # noqa: E402
from tasks_function import TAIL                                            # noqa: E402
from languages import DISPLAY                                              # noqa: E402
import execute as S                                                        # noqa: E402
import prompt_memory as pm                                                 # noqa: E402
import loader                                                              # noqa: E402
from modifier_dho_grid import (BASES, MODS, LANGS, DEV_LANGS, TEST_LANGS, BANDS, ALPHAS, BATTERY,   # noqa: E402
                               FUNCS, WANTV, key, desc, grade)

torch.set_num_threads(8)          # the delta arithmetic is CPU-side; six shards at default threads overloaded the box

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-31B-it")
ap.add_argument("--device", default=None)
ap.add_argument("--phase", choices=["dev", "test"], required=True)
ap.add_argument("--mode", choices=["sever", "install"], default="sever")
ap.add_argument("--shard", default="0/1", help="i/n over the cell list")
ap.add_argument("--band", default=None, help="test: override the selected band")
ap.add_argument("--alpha", type=float, default=None, help="test: override the selected dose")
ap.add_argument("--beta", type=float, default=1.0, help="install: base-factor dose")
ap.add_argument("--alphas", default=None, help="dev: comma list of doses (default grid: sever 0.5,1,1.5,2; install 1,1.5,2)")
ap.add_argument("--bands", default=None, help="dev: comma list of bands (default: sever all four; install 'all')")
ap.add_argument("--arms", default=None, help="test: comma list of arms to run (default: all for the mode)")
ap.add_argument("--ablate", default="zero", choices=["zero", "rand"], help="sever: description-row ablation")
ap.add_argument("--channels", default="kv", choices=["kv", "kvrec"],
                help="kv: attention rows only (correct for attention-only models). kvrec: also patch a hybrid's "
                     "recurrent state -- on qwen3.5/3.6 the KV-only intervention is near-inert (11/36 vs 28/36), "
                     "so a hybrid measured kv-only reads artificially weak. The recurrent contrast is harvested "
                     "from prompt[:-1] so it aligns with the state at the patch point (a full-prompt harvest sits "
                     "one token ahead; recurrent-contrast-alignment, 2026-09-08).")
ap.add_argument("--max-new-tokens", type=int, default=320)
ap.add_argument("--outdir", default=None)
ap.add_argument("--dry-run", action="store_true", help="build prompts, padding and spans; no model")
a = ap.parse_args()
S.BATTERY = BATTERY
outdir = Path(a.outdir or ROOT / "results" / (f"modifier_dho_{a.model}" + ("_install" if a.mode == "install" else "")
                                              + ("_kvrec" if a.channels == "kvrec" else "")))
(outdir / a.phase).mkdir(parents=True, exist_ok=True)
DEV_ALPHAS = [float(x) for x in a.alphas.split(",")] if a.alphas else (ALPHAS if a.mode == "sever" else [1.0, 1.5, 2.0])
DEV_BANDS = a.bands.split(",") if a.bands else (list(BANDS) if a.mode == "sever" else ["all"])

recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]
from transformers import AutoTokenizer                                     # noqa: E402
tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]),
                                    **recipe.get("tokenizer_kwargs", {}))
ids_of = lambda t: tok(render_prompt(tok, recipe, t), add_special_tokens=False)["input_ids"]   # noqa: E731

# ---------------------------------------------------------------- prompts, padded to one length ------------------
raw = {}
for b, m in FUNCS:
    for lam in LANGS:
        for w in (0, 1):
            raw[(key(b, m), lam, w)] = f"Write a function in {DISPLAY[lam]} that {desc(b, m, w)}. {TAIL}"
for lam in LANGS:
    raw[("nofn", lam, 0)] = f"Write a function in {DISPLAY[lam]}. {TAIL}"
target = max(len(ids_of(u)) for u in raw.values())
FILL = ["\n", " ", ".", "-", "#", "*", ". ", "- ", "; "]


def _pad_one(u):
    base = len(ids_of(u))
    if base == target:
        return u
    for c1 in FILL:
        if len(ids_of(c1 + u)) == base and len(ids_of(c1 * 3 + u)) == base:
            continue
        for n in range(1, 400):
            m_ = len(ids_of(c1 * n + u))
            if m_ == target:
                return c1 * n + u
            if abs(m_ - target) <= 2:
                for c2 in FILL:
                    if c2 != c1 and len(ids_of(c1 * n + c2 + u)) == target:
                        return c1 * n + c2 + u
            if m_ > target + 8:
                break
    return None


padded = {k: _pad_one(u) for k, u in raw.items()}
bad = [k for k, v in padded.items() if v is None]
assert not bad, f"padding failed for {bad[:5]}"
seqs = [ids_of(u) for u in padded.values()]
K = 0
while K < target and len({s[-1 - K] for s in seqs}) == 1:
    K += 1
LFULL = target
W0, W1 = LFULL - K, LFULL
print(f"[prompts] {len(padded)} prompts padded to {LFULL} tokens; shared suffix window {W0}..{W1} (K={K})", flush=True)


def desc_rows(k):
    """token rows of the description inside the padded, rendered prompt (the rows the ablation removes)"""
    b_m, lam, w = k
    b, m = (b_m.split(":")[1], b_m.split(":")[0]) if ":" in b_m else (b_m, None)
    spec = desc(b, m, w)
    r = render_prompt(tok, recipe, padded[k])
    off = tok(r, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    c = r.find(spec)
    assert c >= 0, k
    rows = [i for i, (s_, e) in enumerate(off) if s_ < c + len(spec) and e > c and e > s_]
    return min(rows), max(rows) + 1


# ---------------------------------------------------------------- cells --------------------------------------------
def other_wording(w):
    return 1 - w


def single_donor(fb, lam_b, wb):
    fa = BASES[(BASES.index(fb) + 1) % len(BASES)]
    lam_a = "python" if lam_b != "python" else "javascript"
    return fa, lam_a, other_wording(wb)


def pooled_donors(fb, lam_b, wb):
    return [(fa, lam_a, other_wording(wb)) for fa in BASES if fa != fb for lam_a in LANGS if lam_a != lam_b]


def wrong_modifier(fb, m):
    """the next modifier in the list whose target on THIS base differs extensionally from M[f_b] (post_sq and pre_sq
    coincide on prod / max / min, so there the 'wrong' modifier for one of them must skip the other)"""
    for k in range(1, len(MODS)):
        mm = MODS[(MODS.index(m) + k) % len(MODS)]
        if WANTV[key(fb, mm)] != WANTV[key(fb, m)]:
            return mm
    raise AssertionError((fb, m))


def arm_names(fb, m, lam_b, wb):
    if a.phase == "dev":
        return ["none" if a.mode == "sever" else "base"] + [f"pooled_{band}_a{al}" for band in DEV_BANDS for al in DEV_ALPHAS]
    default = (["none", "single", "pooled", "rand", "wrong", "ceiling"] if a.mode == "sever"
               else ["base", "single", "pooled", "rand", "wrong", "ceil_install", "ceiling"])
    return a.arms.split(",") if a.arms else default


cells = []
if a.phase == "dev":
    for lam_b in DEV_LANGS:
        for fb in BASES:
            for m in MODS:
                cells.append((fb, m, lam_b, 0))
else:
    for lam_b in TEST_LANGS:
        for fb in BASES:
            for m in MODS:
                for wb in (0, 1):
                    cells.append((fb, m, lam_b, wb))
si, sn = map(int, a.shard.split("/"))
cells = cells[si::sn]


def tag_of(fb, m, lam_b, wb):
    return f"{m}_{fb}_{lam_b}_w{wb}"


todo = [c for c in cells if not all((outdir / a.phase / f"{tag_of(*c)}__{n}.json").exists() for n in arm_names(*c))]
print(f"[cells] phase {a.phase} mode {a.mode}: {len(cells)} cells in shard {si}/{sn}, {len(todo)} with work left", flush=True)

if a.dry_run:
    k0 = (key("altsum", "post_sq"), "haskell", 1)
    print("[dry-run] example prompt:\n" + padded[k0])
    print("[dry-run] carrier:\n" + padded[("nofn", "haskell", 0)])
    print("[dry-run] description rows", desc_rows(k0), "of", LFULL, "; suffix window", (W0, W1))
    print("[dry-run] arms:", arm_names(*cells[0]) if cells else None)
    sys.exit(0)
if not todo:
    print("[done] nothing to do", flush=True)
    sys.exit(0)

# ---------------------------------------------------------------- model, states -----------------------------------
_L = loader.load(a.model, device=a.device, attn="sdpa")
model, dev, n_layers = _L.model, _L.device, _L.n_layers
gc = model.generation_config
stop = set()
for e in ([gc.eos_token_id] if isinstance(gc.eos_token_id, int) else (gc.eos_token_id or [])):
    stop.add(int(e))
if tok.eos_token_id is not None:
    stop.add(int(tok.eos_token_id))
stop_strings = recipe.get("stop_strings") or []
enc_of = lambda t: tok(render_prompt(tok, recipe, t), return_tensors="pt",     # noqa: E731
                       add_special_tokens=False).to(dev)["input_ids"]

with torch.inference_mode():
    _probe = model(input_ids=enc_of(padded[(key("sum"), "python", 0)]), use_cache=True)
KINDS = pm.profile(_probe.past_key_values, n_layers)
del _probe
PATCHED = ([i for i in pm.select(KINDS, "both") if KINDS[i] == pm.KV] if a.channels == "kv"
           else list(pm.select(KINDS, "both")))
REC = [i for i in PATCHED if KINDS[i] != pm.KV]
print(f"[channels] {a.channels}: {len(PATCHED)} layers patched, {len(REC)} of them recurrent", flush=True)


def band_layers(name):
    lo, hi = BANDS[name]
    return [i for i in PATCHED if lo * n_layers <= i < hi * n_layers]


def harvest(k):
    """suffix-window KV rows (the last K positions) of prompt k. Recurrent slots are harvested from prompt[:-1]:
    generate() patches after prompt[:-1] and then replays the final token, so a recurrent contrast taken from the
    full prompt would sit one step ahead of the state it is added to."""
    ids = enc_of(padded[k])
    with torch.inference_mode():
        c = model(input_ids=ids, use_cache=True).past_key_values
    out = {i: pm.harvest(c, i, KINDS[i], K) for i in PATCHED if KINDS[i] == pm.KV}
    if REC:
        with torch.inference_mode():
            c2 = model(input_ids=ids[:, :-1], use_cache=True).past_key_values
        for i in REC:
            out[i] = pm.harvest(c2, i, KINDS[i], K)
    return out


NEEDED = set()
for fb, m, lam_b, wb in todo:
    for fa, lam_a, wa in pooled_donors(fb, lam_b, wb):
        for mm in MODS:
            NEEDED.add((key(fa), lam_a, wa)); NEEDED.add((key(fa, mm), lam_a, wa))
    if a.mode == "install":
        for lam in LANGS:
            NEEDED.add(("nofn", lam, 0))
            if lam != lam_b:
                NEEDED.add((key(fb), lam, wb)); NEEDED.add((key(fb, m), lam, wb))
STORE = {}
for n_, k in enumerate(sorted(NEEDED)):
    STORE[k] = harvest(k)
    if n_ % 50 == 0:
        print(f"[harvest] {n_}/{len(NEEDED)}", flush=True)
print(f"[harvest] {len(STORE)} states", flush=True)


def delta_of(fa, lam_a, wa, m):
    A, B = STORE[(key(fa, m), lam_a, wa)], STORE[(key(fa), lam_a, wa)]
    return {i: {n: A[i][n].float() - B[i][n].float() for n in A[i]} for i in PATCHED}


def mean_delta(ds):
    return {i: {n: torch.stack([d[i][n] for d in ds]).mean(0) for n in ds[0][i]} for i in PATCHED}


def factor(fkey, lam_b, w):
    """install mode: the function's absolute factor pooled over languages != lam_b (state minus the carrier state)"""
    return mean_delta([{i: {n: STORE[(fkey, lam, w)][i][n].float() - STORE[("nofn", lam, 0)][i][n].float()
                            for n in STORE[(fkey, lam, w)][i]} for i in PATCHED} for lam in LANGS if lam != lam_b])


def combo(F, beta, D, alpha):
    if F is None:
        return None if D is None else {i: {n: alpha * D[i][n] for n in D[i]} for i in PATCHED}
    if D is None:
        return {i: {n: beta * F[i][n] for n in F[i]} for i in PATCHED}
    return {i: {n: beta * F[i][n] + alpha * D[i][n] for n in F[i]} for i in PATCHED}


def rand_matched(delta, seed):
    g = torch.Generator().manual_seed(seed)
    out = {}
    for i in delta:
        out[i] = {}
        for n, v in delta[i].items():
            r = torch.randn(v.shape, generator=g)
            out[i][n] = r * (v.norm() / (r.norm() + 1e-9))
    return out


# ---------------------------------------------------------------- intervention + decode --------------------------
_gabl = torch.Generator().manual_seed(7)


def sever(cache, d0, d1):
    # KV layers only. A hybrid's recurrent state has no per-token rows to remove, and the main qwen experiments
    # (9F transport, "bnative") deliberately leave it INTACT while steering both channels.
    for i in [j for j in PATCHED if KINDS[j] == pm.KV]:
        L = cache.layers[i]
        for attr in ("keys", "values"):
            t = getattr(L, attr)
            if t.shape[2] < d1:
                continue
            w = t[:, :, d0:d1, :]
            if a.ablate == "zero":
                t[:, :, d0:d1, :] = 0
            else:
                r = torch.randn(w.shape, generator=_gabl, dtype=torch.float32).to(w.device, w.dtype)
                t[:, :, d0:d1, :] = r * (w.float().norm(dim=-1, keepdim=True).to(w.dtype)
                                         / (r.float().norm(dim=-1, keepdim=True).to(w.dtype) + 1e-6))


@torch.inference_mode()
def generate(prompt_key, delta, layers, do_sever):
    """delta is already scaled (combo); patched with unit weight on the suffix rows, once on the final row"""
    ids = enc_of(padded[prompt_key])
    L = ids.shape[1]
    assert L == LFULL
    cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    if do_sever:
        d0, d1 = desc_rows(prompt_key)
        sever(cache, d0, d1)
    hi = min(W1, L - 1)
    if delta is not None:
        for i in layers:
            if KINDS[i] == pm.KV:
                pm.patch(cache, i, pm.KV, delta[i], 1.0, lo=W0, hi=hi, s0=0, s1=hi - W0)
            else:
                pm.patch(cache, i, KINDS[i], delta[i], 1.0)        # the whole recurrent state, once
    step = model(input_ids=ids[:, L - 1:], past_key_values=cache, use_cache=True)
    if delta is not None and W1 >= L:
        for i in layers:
            if KINDS[i] == pm.KV:
                pm.patch(cache, i, pm.KV, delta[i], 1.0, lo=L - 1, hi=L, s0=K - 1, s1=K)
    logits = step.logits[:, -1]
    out = []
    for _ in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in stop:
            break
        out.append(nxt)
        if stop_strings and any(t_ in tok.decode(out[-24:], skip_special_tokens=True) for t_ in stop_strings):
            break
        o = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True)
        logits, cache = o.logits[:, -1], o.past_key_values
    txt = tok.decode(out, skip_special_tokens=True)
    for t_ in stop_strings:
        txt = txt.split(t_)[0]
    return txt


# ---------------------------------------------------------------- run ----------------------------------------------
sel = None
if a.phase == "test":
    sp = outdir / "selection.json"
    if a.band is None or a.alpha is None:
        assert sp.exists(), "test phase needs selection.json (summarize_modifier_dho.py --select) or --band/--alpha"
        sel = json.loads(sp.read_text())
    BAND_T = a.band or sel["band"]; ALPHA_T = a.alpha if a.alpha is not None else sel["alpha"]
    print(f"[test] mode {a.mode} band {BAND_T} alpha {ALPHA_T} beta {a.beta}", flush=True)

for fb, m, lam_b, wb in todo:
    want, base_key = key(fb, m), key(fb)
    tag = tag_of(fb, m, lam_b, wb)
    names = arm_names(fb, m, lam_b, wb)
    if all((outdir / a.phase / f"{tag}__{n}.json").exists() for n in names):
        continue
    pooled = mean_delta([delta_of(fa, lam_a, wa, m) for fa, lam_a, wa in pooled_donors(fb, lam_b, wb)])
    Fb = factor(base_key, lam_b, wb) if a.mode == "install" else None
    carrier = ("nofn", lam_b, 0) if a.mode == "install" else (base_key, lam_b, wb)
    sev = a.mode == "sever"
    arms = []                                                   # (name, prompt_key, delta, layers, do_sever, want_key)
    if a.phase == "dev":
        arms.append(("none" if sev else "base", carrier, combo(Fb, a.beta, None, 0.0), band_layers("all"), sev, base_key))
        for band in DEV_BANDS:
            for al in DEV_ALPHAS:
                arms.append((f"pooled_{band}_a{al}", carrier, combo(Fb, a.beta, pooled, al), band_layers(band), sev, want))
    else:
        fa, lam_a, wa = single_donor(fb, lam_b, wb)
        seed = abs(hash(tag)) % (2 ** 31)
        cand = {
            "none": (carrier, None, sev, base_key),
            "base": (carrier, combo(Fb, a.beta, None, 0.0), sev, base_key),
            "single": (carrier, combo(Fb, a.beta, delta_of(fa, lam_a, wa, m), ALPHA_T), sev, want),
            "pooled": (carrier, combo(Fb, a.beta, pooled, ALPHA_T), sev, want),
            "rand": (carrier, combo(Fb, a.beta, rand_matched(pooled, seed), ALPHA_T), sev, want),
            "ceiling": ((want, lam_b, wb), None, False, want),
        }
        if sev:
            wrong_m = wrong_modifier(fb, m)
            cand["wrong"] = (carrier, combo(None, 0.0, mean_delta([delta_of(fa_, la_, wa_, wrong_m)
                                                                   for fa_, la_, wa_ in pooled_donors(fb, lam_b, wb)]), ALPHA_T), sev, want)
        else:
            cand["ceil_install"] = (carrier, combo(factor(want, lam_b, wb), a.beta, None, 0.0), False, want)
            # the specificity control for install mode (added 2026-09-16): the base factor plus ANOTHER modifier's
            # displacement. random says magnitude alone does nothing; the ceiling says the interface works; only
            # this says the displacement carries THIS transformation rather than "a transformation".
            wrong_m = wrong_modifier(fb, m)
            cand["wrong"] = (carrier, combo(Fb, a.beta,
                                            mean_delta([delta_of(fa_, la_, wa_, wrong_m)
                                                        for fa_, la_, wa_ in pooled_donors(fb, lam_b, wb)]),
                                            ALPHA_T), sev, want)
        for n in names:
            pk, d, ds, wk = cand[n]
            arms.append((n, pk, d, band_layers(BAND_T), ds, wk))
    for name, pk, delta, layers, do_sever, want_key in arms:
        fp = outdir / a.phase / f"{tag}__{name}.json"
        if fp.exists():
            continue
        txt = generate(pk, delta, layers, do_sever)
        g = grade(txt, lam_b, want_key, base_key)
        g["hit_target"] = want in g["impls"] and g["lang_ok"]        # M[f_b], regardless of the arm's own want
        rec = {"model": a.model, "phase": a.phase, "mode": a.mode, "arm": name, "base": fb, "mod": m, "lang": lam_b,
               "wording": wb, "want": want, "arm_want": want_key, "beta": a.beta if a.mode == "install" else None,
               "alpha": (float(name.split("_a")[-1]) if name.startswith("pooled_") else
                         (ALPHA_T if (a.phase == "test" and name in ("single", "pooled", "rand", "wrong")) else 0.0)),
               "band": (name.split("_")[1] if name.startswith("pooled_") else (BAND_T if a.phase == "test" else "all")),
               "severed": do_sever, "ablate": a.ablate if sev else None,
               "single_donor": list(single_donor(fb, lam_b, wb)) if name == "single" else None,
               "wrong_mod": (wrong_modifier(fb, m) if name == "wrong" else None), **g}
        fp.write_text(json.dumps(rec, indent=1))
        print(f"  {tag:34s} {name:18s} impl={str(g['impl']):18s} lang={g['lang_detect']:10s} "
              f"{g['status']:10s}" + ("   HIT" if g["hit"] else ""), flush=True)
print(f"[done] shard {si}/{sn} -> {outdir / a.phase}", flush=True)

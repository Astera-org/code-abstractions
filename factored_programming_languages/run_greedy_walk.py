"""Local computational neighbourhoods around digit_sum (Fig. 9b, c).

Question: do nearby validity-preserving activation states implement NEARBY
computations, or arbitrary unrelated ones?

Start from the state that implements `digit_sum`, perturb it inside a semantic
proposal subspace built from genuine causal function differences, and ask what
the reachable valid programs actually COMPUTE.

    delta = sum_k c_k * dF_k,   c ~ N(0, I),   ||delta|| = r * scale
    dF_k  = LOO cross-language mean of [ h(f_b) - h(f_a) ]

Nothing about the parameter is written in the prompt: the recipient prompt is
FIXED (digit_sum, one language) and only the activation state moves. That is the
prompt-surface control from the spec, and it is what the top-k / threshold
families could not provide -- there the "tangent" turned out to be numeral-token
substitution (experiments/result_writeups/threshold_continuity.md).

BEHAVIOUR, NOT SOURCE. Every generation is characterised by its output vector on
a fixed integer evaluation set. Two programs are the same computation iff their
output vectors agree everywhere, so a syntax-only mutation is not a new state.

DIGIT-FAMILY, PREREGISTERED AND BEHAVIOURAL. A program counts as digit-family iff
its output depends only on the MULTISET OF DECIMAL DIGITS of the input -- tested
by giving the evaluation set groups of digit-permutations and requiring equal
outputs within every group -- and it is not constant. That admits digit_sum,
digit_prod, max_digit, count_even_digits and friends while excluding `n mod 7`,
`n + 1`, `n * n`. It is decided from behaviour alone, so a NOVEL function can be
classified without ever having been named.

Arms per radius:
  semantic   delta drawn from the span of the function-difference bank
  random     matched norm, drawn isotropically in the full patched space
Plus the unperturbed recipient, which fixes the starting behaviour.

    uv run factored_programming_languages/run_greedy_walk.py \
        --model gemma-4-31B-it --device cuda:0
"""
import os
# a 104-core box with several of these in parallel: torch's default thread
# count per process drove the load average past 400 and the work stopped
# making progress. Cap before torch is imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import json
import sys
from pathlib import Path

import torch
torch.set_num_threads(8)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from elicit import extract_code, render_prompt                        # noqa: E402
from recipes import RECIPES                                           # noqa: E402
from tasks_function import LANGS, TAIL                                 # noqa: E402
from languages import DISPLAY                                          # noqa: E402
import execute as S                                                    # noqa: E402
import prompt_memory as pm                                             # noqa: E402
import loader                                                          # noqa: E402
from entry import find_entry, detect_language                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--start", default="digit_sum")
ap.add_argument("--lang", default="python", help="recipient language (fixed)")
ap.add_argument("--radii", nargs="+", type=float,
                default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
ap.add_argument("--per-radius", type=int, default=80,
                help="candidates per radius PER ARM")
ap.add_argument("--basis", type=int, default=24,
                help="function-difference directions in the proposal bank")
ap.add_argument("--donor-langs", nargs="+",
                default=["python", "javascript", "java", "go", "rust"])
ap.add_argument("--max-new-tokens", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rays", action="store_true",
                help="one-hot RAY SWEEP instead of the Gaussian-mixture walk: "
                     "step along each single basis direction, both signs")
ap.add_argument("--ray-radii", nargs="+", type=float,
                default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.25, 1.5])
ap.add_argument("--ray-randoms", type=int, default=8,
                help="norm-matched random rays, as the control")
ap.add_argument("--score-rays", action="store_true",
                help="teacher-force two FIXED programs -- the unperturbed "
                     "generation and the ray's own target program -- at every "
                     "magnitude. Scoring the same tokens throughout makes the "
                     "readout continuous in the perturbation, which the "
                     "margin over a generated sequence cannot be: there the "
                     "sequence itself changes when the behaviour flips.")
ap.add_argument("--score-texts", default=None,
                help="JSON {name: program}. Score THESE fixed programs along "
                     "each ray instead of the ray's own target -- for asking "
                     "whether equivalent spellings of the SAME function have "
                     "their own direction in state space.")
ap.add_argument("--score-only", nargs="+", default=None,
                help="restrict --score-rays to these ray names, so the sweep "
                     "can be sharded across cards (each shard writes its own "
                     "rayscores.json)")
ap.add_argument("--score-radii", nargs="+", type=float, default=None,
                help="magnitudes for --score-rays (default: 0 to 1.5 by 0.05)")
ap.add_argument("--margin-cells", nargs="+", default=None,
                help="ray:radius specs to trace, e.g. to_even_digits:-0.4")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

# ---- the function bank: prompt states used to BUILD the basis ----------------
# Reference implementations are used only to NAME a generation after the fact
# (basis-leakage control: known endpoint vs novel function). They never enter the
# digit-family classifier, which is purely behavioural.
FUNCS = {
    "digit_sum":       ("returns the sum of the decimal digits of a number",
                        lambda n: sum(int(c) for c in str(abs(n)))),
    "digit_prod":      ("returns the product of the decimal digits of a number",
                        lambda n: __import__("math").prod(int(c) for c in str(abs(n)))),
    "sq_digit_sum":    ("returns the sum of the squares of the decimal digits of a number",
                        lambda n: sum(int(c) ** 2 for c in str(abs(n)))),
    "even_digits":     ("returns how many decimal digits of a number are even",
                        lambda n: sum(1 for c in str(abs(n)) if int(c) % 2 == 0)),
    "odd_digits":      ("returns how many decimal digits of a number are odd",
                        lambda n: sum(1 for c in str(abs(n)) if int(c) % 2 == 1)),
    "count_digits":    ("returns how many decimal digits a number has",
                        lambda n: len(str(abs(n)))),
    "max_digit":       ("returns the largest decimal digit of a number",
                        lambda n: max(int(c) for c in str(abs(n)))),
    "min_digit":       ("returns the smallest decimal digit of a number",
                        lambda n: min(int(c) for c in str(abs(n)))),
    "distinct_digits": ("returns how many distinct decimal digits a number has",
                        lambda n: len(set(str(abs(n))))),
    "digit_range":     ("returns the largest decimal digit of a number minus the smallest",
                        lambda n: max(int(c) for c in str(abs(n)))
                        - min(int(c) for c in str(abs(n)))),
    "sum_even_digits": ("returns the sum of the even decimal digits of a number",
                        lambda n: sum(int(c) for c in str(abs(n)) if int(c) % 2 == 0)),
    "zero_digits":     ("returns how many decimal digits of a number are zero",
                        lambda n: sum(1 for c in str(abs(n)) if c == "0")),
    # deliberately NOT digit-family: these give the classifier something to exclude
    "square":          ("returns the square of a number", lambda n: n * n),
    "plus_one":        ("returns one more than a number", lambda n: n + 1),
    "mod_seven":       ("returns the remainder of a number divided by seven",
                        lambda n: n % 7),
    "double":          ("returns twice a number", lambda n: 2 * n),
}
FK = list(FUNCS)
assert a.start in FUNCS

# ---- evaluation set: digit-permutation GROUPS make the classifier decidable ---
PERM_GROUPS = [
    [123, 132, 213, 231, 312, 321],
    [406, 460, 604, 640],
    [2468, 2486, 2648, 4268, 6842, 8642],
    [1234, 4321, 2143, 3412],
    [905, 950, 590, 509],
]
SINGLES = [7, 5, 90, 1000, 999, 10, 100, 88, 1, 60]
EVAL = [n for g in PERM_GROUPS for n in g] + SINGLES
GROUP_IDX = []
i = 0
for g in PERM_GROUPS:
    GROUP_IDX.append(list(range(i, i + len(g))))
    i += len(g)
S.BATTERY = EVAL                      # the runners read this at call time
SIG = {k: [f(n) for n in EVAL] for k, (_, f) in FUNCS.items()}


def is_digit_family(outs):
    """behavioural, preregistered: output depends only on the digit multiset"""
    if outs is None or len(outs) != len(EVAL):
        return False
    if len(set(outs)) <= 1:           # constant functions are not computations
        return False
    return all(len({outs[j] for j in idx}) == 1 for idx in GROUP_IDX)


def name_of(outs):
    """exact behavioural match to a bank function, else None (a novel state)"""
    if outs is None or len(outs) != len(EVAL):
        return None
    for k, sig in SIG.items():
        if list(outs) == sig:
            return k
    return None


def text_for(k, lang):
    return (f"Write a function in {DISPLAY[lang]} that {FUNCS[k][0]}. {TAIL}")


recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]
from transformers import AutoTokenizer                                # noqa: E402
tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]),
                                    **recipe.get("tokenizer_kwargs", {}))
ids_of = lambda t: tok(render_prompt(tok, recipe, t),                 # noqa: E731
                       add_special_tokens=False)["input_ids"]

USE_LANGS = sorted(set(a.donor_langs) | {a.lang})
raw = {(k, l): text_for(k, l) for k in FK for l in USE_LANGS}
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


padded = {}
for kk, u in raw.items():
    t2 = _pad_one(u)
    if t2 is None:
        raise SystemExit(f"could not equalise prompt lengths for {kk}")
    padded[kk] = t2
LFULL = len(ids_of(next(iter(padded.values()))))
print(f"[prompt] {len(padded)} prompts ({len(FK)} functions x {len(USE_LANGS)} "
      f"langs) padded to {LFULL}", flush=True)

_L = loader.load(a.model, device=a.device, max_memory=a.max_memory, attn="sdpa")
model, dev, n_layers = _L.model, _L.device, _L.n_layers
gc = model.generation_config
stop = {int(e) for e in ([gc.eos_token_id] if isinstance(gc.eos_token_id, int)
                         else (gc.eos_token_id or []))}
if tok.eos_token_id is not None:
    stop.add(int(tok.eos_token_id))
stop_strings = recipe.get("stop_strings") or []
enc_of = lambda t: tok(render_prompt(tok, recipe, t), return_tensors="pt",  # noqa: E731
                       add_special_tokens=False).to(dev)["input_ids"]

_probe = model(input_ids=enc_of(padded[(FK[0], USE_LANGS[0])]), use_cache=True)
KINDS = pm.profile(_probe.past_key_values, n_layers)
del _probe
PATCHED = pm.select(KINDS, "both")
W0, W1 = 0, LFULL
print(f"[site] span patch, positions {W0}..{W1}", flush=True)


def harvest(key):
    with torch.inference_mode():
        c = model(input_ids=enc_of(padded[key]), use_cache=True).past_key_values
    return {i: pm.harvest(c, i, KINDS[i], LFULL) for i in PATCHED}


STORE = {kk: harvest(kk) for kk in padded}
print(f"[harvest] {len(STORE)} states cached", flush=True)


def loo_diff(fa, fb):
    """h(fb) - h(fa), averaged over the donor languages, recipient's excluded"""
    langs = [l for l in a.donor_langs if l != a.lang]
    out = {}
    for lay in PATCHED:
        ref = STORE[(fa, langs[0])][lay]
        out[lay] = {n: torch.stack([STORE[(fb, l)][lay][n].float()
                                    - STORE[(fa, l)][lay][n].float()
                                    for l in langs]).mean(0) for n in ref}
    return out


# ---- proposal basis ---------------------------------------------------------
g = torch.Generator().manual_seed(a.seed)
PAIRS = [(x, y) for x in FK for y in FK if x != y]
sel = torch.randperm(len(PAIRS), generator=g)[:a.basis].tolist()
BASIS = [loo_diff(*PAIRS[i]) for i in sel]
BASIS_PAIRS = [PAIRS[i] for i in sel]


def _flat(d):
    return torch.cat([d[l][n].reshape(-1) for l in PATCHED for n in sorted(d[l])])


SCALE = float(torch.stack([_flat(b).norm() for b in BASIS]).mean())
print(f"[basis] {len(BASIS)} function differences, mean ||dF|| = {SCALE:.1f}",
      flush=True)


def combine(coeffs):
    out = {}
    for lay in PATCHED:
        out[lay] = {}
        for n in sorted(BASIS[0][lay]):
            acc = None
            for c, b in zip(coeffs, BASIS):
                t = b[lay][n] * c
                acc = t if acc is None else acc + t
            out[lay][n] = acc
    return out


def rescale(d, target_norm):
    nrm = float(_flat(d).norm()) + 1e-9
    return {l: {n: v * (target_norm / nrm) for n, v in d[l].items()} for l in d}


def rand_dir(seed):
    gg = torch.Generator().manual_seed(seed)
    out = {}
    for lay in PATCHED:
        out[lay] = {}
        for n in sorted(BASIS[0][lay]):
            v = BASIS[0][lay][n]
            out[lay][n] = torch.randn(v.shape, generator=gg).to(v.dtype)
    return out


@torch.inference_mode()
def generate(delta):
    ids = enc_of(padded[(a.start, a.lang)])
    L = ids.shape[1]
    cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    hi = min(W1, L - 1)
    if delta is not None:
        for i in PATCHED:
            if KINDS[i] != pm.KV:
                continue
            layer = cache.layers[i]
            for name, attr in (("keys", "keys"), ("values", "values")):
                t_ = getattr(layer, attr)
                add = delta[i][name][:, W0:hi, :].float().to(t_.device)
                t_[:, :, W0:hi, :] = (t_[:, :, W0:hi, :].float() + add).to(t_.dtype)
    step = model(input_ids=ids[:, L - 1:], past_key_values=cache, use_cache=True)
    cache = step.past_key_values
    out, nxt = [], int(step.logits[:, -1].argmax(-1))
    for _ in range(a.max_new_tokens):
        if nxt in stop:
            break
        out.append(nxt)
        if stop_strings and any(s in tok.decode(out[-24:], skip_special_tokens=True)
                                for s in stop_strings):
            break
        step = model(input_ids=torch.tensor([[nxt]], device=dev),
                     past_key_values=cache, use_cache=True)
        cache = step.past_key_values
        nxt = int(step.logits[:, -1].argmax(-1))
    return tok.decode(out, skip_special_tokens=True)


@torch.inference_mode()
def generate_traced(delta):
    """Identical decode path to generate(), but records the top-2 logprob margin
    at every step. That is what distinguishes 'the perturbation moved a confident
    decision' from 'the token was a near-tie and any nudge flips it'."""
    ids = enc_of(padded[(a.start, a.lang)])
    L = ids.shape[1]
    cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    hi = min(W1, L - 1)
    if delta is not None:
        for i in PATCHED:
            if KINDS[i] != pm.KV:
                continue
            layer = cache.layers[i]
            for name, attr in (("keys", "keys"), ("values", "values")):
                t_ = getattr(layer, attr)
                add = delta[i][name][:, W0:hi, :].float().to(t_.device)
                t_[:, :, W0:hi, :] = (t_[:, :, W0:hi, :].float() + add).to(t_.dtype)
    step = model(input_ids=ids[:, L - 1:], past_key_values=cache, use_cache=True)
    cache = step.past_key_values
    out, trace = [], []

    def record(logits):
        lp = torch.log_softmax(logits[0, -1].float(), -1)
        top = torch.topk(lp, 2)
        return (int(top.indices[0]), float(top.values[0]),
                int(top.indices[1]), float(top.values[1]),
                float(top.values[0] - top.values[1]))

    t1, l1, t2, l2, m = record(step.logits)
    nxt = t1
    for _ in range(a.max_new_tokens):
        if nxt in stop:
            break
        out.append(nxt)
        trace.append({"pos": len(out) - 1, "tok": tok.decode([nxt]),
                      "top1": t1, "lp1": l1, "top2": t2,
                      "tok2": tok.decode([t2]), "lp2": l2, "margin": m})
        if stop_strings and any(s_ in tok.decode(out[-24:], skip_special_tokens=True)
                                for s_ in stop_strings):
            break
        step = model(input_ids=torch.tensor([[nxt]], device=dev),
                     past_key_values=cache, use_cache=True)
        cache = step.past_key_values
        t1, l1, t2, l2, m = record(step.logits)
        nxt = t1
    return out, trace


def characterise(txt):
    code, _ = extract_code(txt, "auto")
    det = detect_language(code)
    entry = find_entry(det, code) if det != "unknown" else None
    if not (code and entry and det in S.RUNNERS):
        return {"valid": False, "status": "no_entry", "lang_detect": det,
                "outs": None, "name": None, "digit_family": False,
                "code": (code or txt)[:2000]}
    st, got = None, None
    for form in S.FORMS:
        st, got = S.RUNNERS[det](code, entry, form)
        if st == "ok" and got is not None and len(got) == len(EVAL):
            break
    ok = st == "ok" and got is not None and len(got) == len(EVAL)
    return {"valid": bool(ok), "status": st, "lang_detect": det,
            "code": code[:2000],
            "outs": list(got) if ok else None,
            "name": name_of(got) if ok else None,
            "digit_family": is_digit_family(got) if ok else False}


outdir = Path(a.outdir or ROOT / "results" / f"greedy_{a.model}")
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "meta.json").write_text(json.dumps(
    {"model": a.model, "start": a.start, "lang": a.lang, "eval": EVAL,
     "perm_groups": PERM_GROUPS, "radii": a.radii, "basis_pairs": BASIS_PAIRS,
     "scale": SCALE, "signatures": SIG}, indent=1))

# baseline: the unperturbed recipient fixes the starting behaviour
fp = outdir / "gw_base.json"
if not fp.exists():
    r = {"arm": "none", "radius": 0.0, **characterise(generate(None))}
    fp.write_text(json.dumps(r, indent=1))
    print(f"[base] name={r['name']} valid={r['valid']} digit={r['digit_family']}",
          flush=True)
BASE = json.loads((outdir / "gw_base.json").read_text())["outs"]

# ---- FIXED-SEQUENCE SCORES -------------------------------------------------
# One forward per (ray, magnitude, reference): feed the patched prompt cache
# followed by the reference tokens and read the log-prob the model assigns to
# each of them. Both references are held fixed across the whole sweep.
if a.score_rays:
    import re as _re
    rays_dir = ROOT / "results" / f"rays_{a.model}"
    refs = {}
    for f in sorted(rays_dir.glob("ray_to_*.json")):
        r = json.loads(f.read_text())
        if r.get("hit_target") and r.get("code") and r["ray"] not in refs:
            refs[r["ray"]] = r["code"]
    if a.score_only:
        refs = {k: v for k, v in refs.items() if k in set(a.score_only)}
    print(f"[score] {len(refs)} rays have a target program to score",
          flush=True)

    @torch.inference_mode()
    def logprob_of(delta, seq):
        ids = enc_of(padded[(a.start, a.lang)])
        L = ids.shape[1]
        cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
        hi = min(W1, L - 1)
        if delta is not None:
            for i2 in PATCHED:
                if KINDS[i2] != pm.KV:
                    continue
                layer = cache.layers[i2]
                for name, attr in (("keys", "keys"), ("values", "values")):
                    t_ = getattr(layer, attr)
                    add = delta[i2][name][:, W0:hi, :].float().to(t_.device)
                    t_[:, :, W0:hi, :] = (t_[:, :, W0:hi, :].float()
                                          + add).to(t_.dtype)
        inp = torch.cat([ids[:, L - 1:], seq[:, :-1]], dim=1)
        lg = model(input_ids=inp, past_key_values=cache).logits[0]
        lp = torch.log_softmax(lg.float(), -1)
        tot = float(sum(lp[k, int(seq[0, k])] for k in range(seq.shape[1])))
        return tot, tot / seq.shape[1]

    base_ids, _ = generate_traced(None)
    src = torch.tensor([base_ids], device=dev)
    texts = (json.loads(Path(a.score_texts).read_text())
             if a.score_texts else None)
    if texts:
        seqs = {k: tok(v, add_special_tokens=False,
                       return_tensors="pt")["input_ids"].to(dev)
                for k, v in texts.items()}
        print(f"[score] {len(seqs)} fixed programs x {len(refs)} rays",
              flush=True)
    grid = a.score_radii or [round(0.05 * k, 3) for k in range(0, 31)]
    rows = []
    for ray, code in refs.items():
        tgt = tok(code, add_special_tokens=False,
                  return_tensors="pt")["input_ids"].to(dev)
        v = loo_diff(a.start, ray[3:])
        for m in grid:
            d = rescale(v, m * SCALE) if m else None
            if texts:
                row = {"ray": ray, "mag": m}
                for nm, sq in seqs.items():
                    tot, mean = logprob_of(d, sq)
                    row[f"{nm}_logprob"] = tot
                    row[f"{nm}_mean"] = mean
                    row[f"n_{nm}"] = int(sq.shape[1])
                rows.append(row)
                continue
            s_tot, s_mean = logprob_of(d, src)
            t_tot, t_mean = logprob_of(d, tgt)
            rows.append({"ray": ray, "mag": m,
                         "source_logprob": s_tot, "source_mean": s_mean,
                         "target_logprob": t_tot, "target_mean": t_mean,
                         "n_source": int(src.shape[1]),
                         "n_target": int(tgt.shape[1])})
        print(f"  {ray:>22}: scored {len(grid)} magnitudes", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "rayscores.json").write_text(json.dumps(
        {"model": a.model, "start": a.start, "lang": a.lang,
         "grid": grid, "rows": rows}, indent=1))
    print(f"saved -> {outdir / 'rayscores.json'}", flush=True)
    raise SystemExit(0)

# ---- MARGIN TRACE ----------------------------------------------------------
# Are the implementation variants "the perturbation moved a CONFIDENT decision"
# or "that token was a near-tie and any nudge flips it"? Record the unperturbed
# top-2 margin at every position, then re-run the exact cells that produced a
# variant and compare at the divergence point.
if a.margin_cells:
    outm = {}
    b_ids, b_tr = generate_traced(None)
    outm["baseline"] = {"ids": b_ids, "text": tok.decode(b_ids, skip_special_tokens=True),
                        "trace": b_tr}
    print(f"[margin] baseline {len(b_ids)} tokens, median margin "
          f"{sorted(x['margin'] for x in b_tr)[len(b_tr)//2]:.3f}", flush=True)
    outm["cells"] = []
    for spec in a.margin_cells:
        ray, mag = spec.rsplit(":", 1)
        r = float(mag)
        v = loo_diff(a.start, ray[3:]) if ray.startswith("to_") else rand_dir(0)
        d = rescale(v, abs(r) * SCALE)
        if r < 0:
            d = {l: {n: -w for n, w in d[l].items()} for l in d}
        ids, tr = generate_traced(d)
        # first position where the token sequence departs from baseline
        div = next((i for i in range(min(len(ids), len(b_ids)))
                    if ids[i] != b_ids[i]), None)
        outm["cells"].append({
            "ray": ray, "r": r, "ids": ids, "trace": tr,
            "text": tok.decode(ids, skip_special_tokens=True),
            "divergence_pos": div,
            "baseline_margin_at_div": (b_tr[div]["margin"] if div is not None
                                       and div < len(b_tr) else None),
            "baseline_tok_at_div": (b_tr[div]["tok"] if div is not None
                                    and div < len(b_tr) else None),
            "baseline_runnerup_at_div": (b_tr[div]["tok2"] if div is not None
                                         and div < len(b_tr) else None),
            "perturbed_margin_at_div": (tr[div]["margin"] if div is not None
                                        and div < len(tr) else None),
            "perturbed_tok_at_div": (tr[div]["tok"] if div is not None
                                     and div < len(tr) else None)})
        print(f"  {ray:20s} r={r:+6g} divergence@{div} "
              f"baseline_margin={outm['cells'][-1]['baseline_margin_at_div']}",
              flush=True)
    (outdir / "margins.json").write_text(json.dumps(outm, indent=1))
    print(f"saved -> {outdir / 'margins.json'}", flush=True)
    raise SystemExit(0)

# ---- RAY SWEEP -------------------------------------------------------------
# One-hot instead of a Gaussian mixture. H1 from the walk is a statement about
# the SUBSPACE: a random mixture of 24 factors is a generic vector in the span
# and says nothing about whether any single genuine dF is itself
# validity-preserving. Here each direction is stepped alone, both signs, to find
# the radius at which behaviour first leaves the starting function -- a boundary
# profile of the basin rather than a cloud sample.
#
# Two ray families:
#   outward   v = dF(start -> f) for every other bank function. Interpretable:
#             does the ray toward f actually arrive at f?
#   random    norm-matched isotropic directions, the control.
if a.rays:
    rays = [(f"to_{f}", f, loo_diff(a.start, f)) for f in FK if f != a.start]
    for j in range(a.ray_randoms):
        rays.append((f"rand{j}", None,
                     rand_dir(abs(hash(("ray", a.seed, j))) % (2 ** 31))))
    mags = sorted(set(a.ray_radii))
    print(f"[rays] {len(rays)} directions x {len(mags)} magnitudes x 2 signs "
          f"= {len(rays) * len(mags) * 2} generations", flush=True)
    for name, target, v in rays:
        for mag in mags:
            for sgn in (+1, -1):
                r = sgn * mag
                fp = outdir / f"ray_{name}_{'p' if sgn > 0 else 'm'}{mag:g}.json"
                if fp.exists():
                    continue
                d = rescale(v, mag * SCALE)
                if sgn < 0:
                    d = {l: {n: -w for n, w in d[l].items()} for l in d}
                rec = {"kind": "ray", "ray": name, "target": target, "r": r,
                       "mag": mag, "sign": sgn, **characterise(generate(d))}
                rec["same_as_start"] = (rec["outs"] == BASE) if rec["outs"] else False
                rec["hit_target"] = (target is not None and rec["name"] == target)
                fp.write_text(json.dumps(rec, indent=1))
                print(f"  {name:18s} r={r:+6g} valid={int(rec['valid'])} "
                      f"name={str(rec['name']):15s} same={int(rec['same_as_start'])} "
                      f"hit={int(rec['hit_target'])}", flush=True)
    print(f"saved -> {outdir}", flush=True)
    raise SystemExit(0)

n_tot = len(a.radii) * a.per_radius * 2
print(f"[run] {n_tot} candidates ({len(a.radii)} radii x {a.per_radius} x 2 arms)",
      flush=True)
for r in a.radii:
    for arm in ("semantic", "random"):
        for m in range(a.per_radius):
            fp = outdir / f"gw_r{r}_{arm}_{m:03d}.json"
            if fp.exists():
                continue
            seed = abs(hash((a.seed, arm, r, m))) % (2 ** 31)
            if arm == "semantic":
                gg = torch.Generator().manual_seed(seed)
                c = torch.randn(len(BASIS), generator=gg).tolist()
                d = combine(c)
            else:
                c = None
                d = rand_dir(seed)
            d = rescale(d, r * SCALE)
            rec = {"arm": arm, "radius": r, "idx": m, "seed": seed,
                   **characterise(generate(d))}
            rec["same_as_start"] = (rec["outs"] == BASE) if rec["outs"] else False
            fp.write_text(json.dumps(rec, indent=1))
            print(f"  r={r:<5g} {arm:8s} {m:03d} valid={int(rec['valid'])} "
                  f"name={str(rec['name']):15s} digit={int(rec['digit_family'])} "
                  f"same={int(rec['same_as_start'])}", flush=True)
print(f"saved -> {outdir}", flush=True)

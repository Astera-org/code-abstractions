"""The instruct severing experiment, run on a PRETRAINED checkpoint.

Same instrument as run_ablate_tiers.py's `abl_rand`, which is what defines the
compiler-like split: prefill the prompt, replace the description's KV rows in
every KV layer with random vectors rescaled to each row's original norm, then
generate. The suffix rows -- computed while the description was still there --
are what the model is left with.

The only thing that differs from the instruct arm is the prompt format, because
a pretrained checkpoint has no chat template. The request is posed as a
completion instead (completion.py). Two layouts, --layout:

    godoc    (default) a file header, the request as a comment, a generic doc
             comment where Godoc / Doxygen put it, and the signature opened;
             suffix sized to the chat prompt's 31 post-description rows.
             Nothing in context but the description says what to write.
    minex    the same with a zero-argument string-returning primer function in
             place of the header: demonstrates "comment, signature, body"
             without anything that fits an int -> int body. Not enough to stop
             the Go stub ("// TODO: Implement this function") on 26b/31B.
    primer   a completed exercise with a real loop body (longest string in a
             list) in place of the header: the completed-exercise pattern the
             example layout had, without a body that fits ours.
    file     the doc comment INSIDE the body. Failed its own conditioner on
             C/Go (the base model continues the comment or stubs the body);
             results/cabl_<model>_file.
    example  the first attempt: a one-shot worked example instead of the
             header, and a generic comment line before the signature. Kept so
             results/cabl_<model> stays reproducible. Under abl_rand the
             example is the most coherent thing left in context and the base
             model copies it verbatim or lets its predicate bleed into the
             task -- about 14 of 24 mask-cut misses on the gemma-4 pretrained
             twins were the example, not the model. The chat prompt has no
             example, so that distractor is an artifact of this layout.

Conditions:

    none         uncut. A cell only counts if this passes.
    abl_mask     the description's COLUMNS cut from the attention mask during generation, prefill
                 left intact. Same instrument as the main figures' abl_mask; nothing is substituted
                 for the rows, so they take no softmax share.
    abl_rand     the description's KV rows replaced as above. THE intervention.
    abl_rand_ex  (example layout only, off by default) the description AND the
                 one-shot example both replaced. Randomises ~75 of ~99 rows
                 where the chat arm randomises 34 of 53; the outputs degrade
                 to garbage, so it is not the comparable arm.
    abl_rec      HYBRID models only (qwen3.5: 10 kv + 30 recurrent layers).
                 The recurrent and conv states of every linear-attention layer
                 replaced by their mean over five UNRELATED distractor
                 descriptions in the same layout (run_ablate_tiers.py's
                 abl_full recipe; the state is position-free, so no padding).
                 KV rows untouched: how much the recurrent channel alone
                 carries.
    abl_rand_rec abl_rand + abl_rec: the description severed in BOTH channels.
                 abl_rand alone leaves the 30 recurrent layers reading the
                 description, so on a hybrid it is not a severing. But abl_rec
                 also erases the SUFFIX's contribution to the recurrent state,
                 which is where the compiled pointer would live -- it over-cuts.
    abl_replay   HYBRID only. The recurrent analogue of abl_rand proper: the
                 per-position inputs (q, k, v, gate, beta) of every linear
                 layer are recorded at prefill; the recurrent state is then
                 RECOMPUTED by replaying the gated delta rule over the same
                 positions with k and v RANDOMISED (at their original norms)
                 on the description positions only, gates kept. The
                 description's contribution to the state is destroyed, the
                 suffix's contribution is exactly what prefill computed with
                 the description present -- like the KV rows. KV untouched.
    abl_rand_replay  abl_rand + abl_replay: the severing on one definition.
                 Positive control at start-up: replaying the recorded inputs
                 unmodified must reproduce the cached state (printed; the run
                 aborts if the relative error exceeds 2%).
                 RESULT (qwen3.5-pt, primer): token salad in 24/24 cells, KV
                 cut or not. A delta-rule memory has no softmax to ignore a
                 random key, and random keys are never erased by later writes
                 the way real ones are, so the state is swamped. Hence:
    abl_replay0 / abl_rand_replay0   the description positions are SKIPPED in
                 the replay (k = v = 0: no write, no erase, only the gate's
                 decay) -- the recurrent analogue of abl_0 on KV rows.
    abl_replayv / abl_rand_replayv   keys kept, VALUES randomised at their
                 norms on the description positions: addresses stay, content
                 goes -- the analogue of the v-only KV ablation.
                 Start-up prints ||S'|| / ||S|| for each variant so the
                 in-distribution one can be told from the swamping one.
    abl_0        KV rows of the description ZEROED (keys and values), as in the
                 instruct battery's abl_0: removal rather than noise on the KV
                 side too.
    abl_0_replay0  abl_0 + abl_replay0: the description removed from both
                 channels with the same operation.

The same runner is the format control for the instruction-tuned twins: run
them with their model key and no chat template is applied.

    uv run factored_programming_languages/run_completion_ablate.py \
        --model gemma-4-31B --device cuda:0
    uv run factored_programming_languages/run_completion_ablate.py \
        --model gemma-4-31B-it --device cuda:0        # format control
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from completion import build, program, finished, LAYOUTS                  # noqa: E402
from tasks_function import BATTERY, WANT, FDESC, which_function           # noqa: E402
import execute as S                                                       # noqa: E402
import loader                                                             # noqa: E402
import prompt_memory as pm                                                # noqa: E402
from entry import find_entry, detect_language                          # noqa: E402

TASKS = ["digit_sum", "digit_prod", "even_digits", "sq_digit_sum"]
# the recurrent-state mean is taken over UNRELATED descriptions (copied from
# run_ablate_tiers.py: related ones leak -- the tier tasks collapsed into each
# other when they were used as the mean)
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
REC_MODES = ("abl_rec", "abl_rand_rec")
# replay modes -> how the description positions enter the replayed recurrence
REPLAY_HOW = {"abl_replay": "rand", "abl_rand_replay": "rand",
              "abl_replay0": "zero", "abl_rand_replay0": "zero",
              "abl_0_replay0": "zero",
              "abl_replayv": "randv", "abl_rand_replayv": "randv"}
REPLAY_MODES = tuple(REPLAY_HOW)
KV_RAND = ("abl_rand", "abl_rand_ex", "abl_rand_rec",
           "abl_rand_replay", "abl_rand_replay0", "abl_rand_replayv")
KV_ZERO = ("abl_0", "abl_0_replay0")
# abl_mask (authors, 2026-09-18): cut the ATTENTION PATH to the description's columns instead of
# editing their content, so the base-vs-instruct comparison uses the same instrument as the main
# figures. Prefill is unmasked -- the tail rows are computed with the description visible and only
# the decode steps cannot reach it, which is the "cannot re-read" condition abl_rand also tests.
KV_MASK = ("abl_mask",)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--langs", nargs="+", default=["c", "go", "haskell"])
ap.add_argument("--tasks", nargs="+", default=TASKS)
ap.add_argument("--modes", nargs="+", default=["none", "abl_rand"])
ap.add_argument("--layout", default="godoc", choices=LAYOUTS)
ap.add_argument("--width", default="wide", choices=["wide", "narrow", "xwide"],
                help="narrow = signature only after the description (control); "
                     "xwide = ~55-row in-body docstring (file layout only)")
ap.add_argument("--max-new-tokens", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()
S.BATTERY = BATTERY                     # standing rule: rebind before running

_L = loader.load(a.model, device=a.device, max_memory=a.max_memory, attn="eager")
model, dev, n_layers = _L.model, _L.device, _L.n_layers
import yaml                                                               # noqa: E402
from transformers import AutoTokenizer                                    # noqa: E402
cfg = yaml.safe_load(open(ROOT.parent / "models.yaml"))[a.model]
tok = AutoTokenizer.from_pretrained(cfg["hf_name"])

# which cache layers hold KV at all (a hybrid model's recurrent layers do not)
with torch.inference_mode():
    _probe = model(input_ids=torch.tensor([[tok.bos_token_id or 0]], device=dev),
                   use_cache=True)
KINDS = pm.profile(_probe.past_key_values, n_layers)
del _probe
KV_L = [i for i in range(n_layers) if KINDS[i] == pm.KV]
RC_L = [i for i in range(n_layers) if KINDS[i] == pm.RECURRENT]
print(f"[cache] {pm.describe(KINDS)}", flush=True)

# ---- recurrent replay: record each linear layer's per-position rule inputs ----
# the module calls self.chunk_gated_delta_rule(query, key, value, g=, beta=, ...)
# at prefill; wrapping that attribute records the inputs without touching the
# forward. Only the pure-torch kernels are installed here (no fla), so the
# per-step torch rule is the exact same arithmetic for the replay.
REC = {"on": False, "inputs": {}}
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


for i, m in LIN.items():
    _record(i, m)
if RC_L:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (              # noqa: E402
        torch_recurrent_gated_delta_rule as _rec_rule)


def _randlike(t, gen):
    r = torch.randn(t.shape, generator=gen).to(t.device)
    return (r * t.float().norm(dim=-1, keepdim=True)
            / (r.norm(dim=-1, keepdim=True) + 1e-8)).to(t.dtype)


@torch.inference_mode()
def replay_state(inp, w0=None, w1=None, gen=None, how="rand"):
    """the final recurrent state from replaying the recorded inputs, with the
    description positions w0:w1 altered per `how`: rand = k and v random at
    their norms (k is L2-normalised in the kernel, so its norm is moot);
    zero = k and v zero, so those steps neither write nor erase; randv = keys
    kept, values random at their norms"""
    q, k, v, g, b = (inp[x] for x in ("query", "key", "value", "g", "beta"))
    if w0 is not None and w1 > w0:
        k, v = k.clone(), v.clone()
        if how == "zero":
            k[:, w0:w1] = 0
            v[:, w0:w1] = 0
        else:
            if how == "rand":
                k[:, w0:w1] = _randlike(k[:, w0:w1], gen)
            v[:, w0:w1] = _randlike(v[:, w0:w1], gen)
    _, S = _rec_rule(q, k, v, g, b, initial_state=None, output_final_state=True,
                     use_qk_l2norm_in_kernel=True)
    return S


if RC_L and any(m in a.modes for m in REPLAY_MODES):
    # positive control: an unmodified replay must reproduce the cached state
    _text, _ = build(a.langs[0], FDESC[a.tasks[0]], wide=a.width, layout=a.layout)
    _ids = tok(_text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
    REC["on"], REC["inputs"] = True, {}
    with torch.inference_mode():
        _c = model(input_ids=_ids[:, :-1], use_cache=True).past_key_values
    REC["on"] = False
    worst = 0.0
    for i in RC_L:
        S0 = _c.layers[i].recurrent_states.float()
        S1 = replay_state(REC["inputs"][i]).to(S0.device, S0.dtype)
        worst = max(worst, float((S1 - S0).norm() / (S0.norm() + 1e-8)))
    print(f"[replay check] {len(RC_L)} layers, worst relative error "
          f"{worst:.2e} (cached dtype {_c.layers[RC_L[0]].recurrent_states.dtype})",
          flush=True)
    assert worst < 2e-2, "replay does not reproduce the cached recurrent state"
    # how far each variant moves the state: ||S'|| / ||S|| and ||S' - S|| / ||S||
    _text, (_d0c, _d1c) = build(a.langs[0], FDESC[a.tasks[0]], wide=a.width, layout=a.layout)
    _offs = tok(_text, return_offsets_mapping=True, add_special_tokens=True)["offset_mapping"]
    _didx = [i for i, (s_, e_) in enumerate(_offs) if s_ < _d1c and e_ > _d0c and e_ > s_]
    _w0, _w1 = min(_didx), min(max(_didx) + 1, _ids.shape[1] - 1)
    for how in ("rand", "zero", "randv"):
        ratio, move = [], []
        for i in RC_L:
            S0 = _c.layers[i].recurrent_states.float()
            S1 = replay_state(REC["inputs"][i], _w0, _w1,
                              torch.Generator(device="cpu").manual_seed(0), how).to(S0)
            ratio.append(float(S1.norm() / (S0.norm() + 1e-8)))
            move.append(float((S1 - S0).norm() / (S0.norm() + 1e-8)))
        ratio, move = sorted(ratio), sorted(move)
        print(f"[replay {how:5s}] ||S'||/||S|| median {ratio[len(ratio)//2]:.2f} "
              f"max {ratio[-1]:.2f}; ||S'-S||/||S|| median {move[len(move)//2]:.2f} "
              f"max {move[-1]:.2f}", flush=True)
    del _c


WIDE = a.width                          # width name, passed through to build()


def spans(lang, task):
    """(ids, description token span) -- located by character offsets, so the
    span cannot silently go stale if the layout changes"""
    text, (d0c, d1c) = build(lang, FDESC[task], wide=WIDE, layout=a.layout)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    offs = enc["offset_mapping"]
    idx = [i for i, (s, e) in enumerate(offs) if s < d1c and e > d0c and e > s]
    # the one-shot example (or the file header) is everything before the
    # description
    eidx = [i for i, (s, e) in enumerate(offs) if e <= d0c and e > s]
    return enc["input_ids"], (min(idx), max(idx) + 1), \
        ((min(eidx), max(eidx) + 1) if eidx else None)


_REC_MEAN = {}


@torch.inference_mode()
def rec_mean(lang):
    """per recurrent layer, the mean conv / recurrent state over the five
    distractor prompts, each prefilled exactly as the target is (all but the
    last token) so the states are one step short in the same way"""
    if lang not in _REC_MEAN:
        stores = []
        for desc in DISTRACT.values():
            text, _ = build(lang, desc, wide=WIDE, layout=a.layout)
            ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(dev)
            c = model(input_ids=ids[:, :-1], use_cache=True).past_key_values
            stores.append({i: pm.harvest(c, i, pm.RECURRENT, 0) for i in RC_L})
        _REC_MEAN[lang] = {
            i: {k: torch.stack([st[i][k].float() for st in stores]).mean(0)
                for k in stores[0][i]}
            for i in RC_L}
    return _REC_MEAN[lang]


@torch.inference_mode()
def generate(lang, task, mode):
    ids_l, (d0, d1), espan = spans(lang, task)
    ids = torch.tensor([ids_l], device=dev)
    L = ids.shape[1]
    REC["on"], REC["inputs"] = mode in REPLAY_MODES, {}
    cache = model(input_ids=ids[:, :L - 1], use_cache=True).past_key_values
    REC["on"] = False
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    if mode in REPLAY_MODES:
        for i in RC_L:
            S = replay_state(REC["inputs"][i], d0, min(d1, L - 1), gen, REPLAY_HOW[mode])
            tgt = cache.layers[i].recurrent_states
            tgt.copy_(S.to(tgt.device, tgt.dtype))
    if mode in REC_MODES:
        mu = rec_mean(lang)
        for i in RC_L:
            layer = cache.layers[i]
            for nm, attr in (("conv", "conv_states"), ("recurrent", "recurrent_states")):
                st = getattr(layer, attr, None)
                if st is None:
                    continue
                if isinstance(st, dict):
                    for k in st:
                        dk = f"{nm}.{k}"
                        if st[k] is None or dk not in mu[i]:
                            continue
                        st[k].copy_(mu[i][dk].to(st[k].device, st[k].dtype))
                elif nm in mu[i]:
                    st.copy_(mu[i][nm].to(st.device, st.dtype))
    amask = None
    if mode in KV_MASK:
        amask = torch.ones(1, L, dtype=torch.long, device=dev)
        amask[:, d0:min(d1, L - 1)] = 0
    wins = []
    if mode in KV_RAND or mode in KV_ZERO:
        wins.append((d0, min(d1, L - 1)))
    if mode == "abl_rand_ex" and espan:
        wins.append((espan[0], min(espan[1], L - 1)))
    if wins:
        for i in KV_L:
            layer = cache.layers[i]
            for attr in ("keys", "values"):
                t = getattr(layer, attr)
                for w0, w1 in wins:
                    if w1 <= w0:
                        continue
                    if mode in KV_ZERO:
                        t[:, :, w0:w1, :] = 0
                        continue
                    orig = t[:, :, w0:w1, :].float().cpu()
                    r = torch.randn(orig.shape, generator=gen)
                    r = r * (orig.norm(dim=-1, keepdim=True)
                             / (r.norm(dim=-1, keepdim=True) + 1e-8))
                    t[:, :, w0:w1, :] = r.to(t.device, t.dtype)
    step = model(input_ids=ids[:, L - 1:], past_key_values=cache, use_cache=True,
                 attention_mask=amask)
    cache, logits, out = step.past_key_values, step.logits[:, -1], []
    # any special token ends the answer: gemma-4-12b emits <image|> runs where
    # a newline or <eos> belongs, and they would otherwise fill the budget
    stop = set(tok.all_special_ids) | {tok.eos_token_id}
    for _ in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in stop:
            break
        out.append(nxt)
        if finished(lang, tok.decode(out)):
            break
        if amask is not None:
            amask = torch.cat([amask, torch.ones(1, 1, dtype=amask.dtype, device=dev)], 1)
        o = model(input_ids=torch.tensor([[nxt]], device=dev),
                  past_key_values=cache, use_cache=True, attention_mask=amask)
        logits, cache = o.logits[:, -1], o.past_key_values
    return tok.decode(out), d1 - d0


REFV = {t: WANT[t] for t in TASKS}


def grade(code, lang, task):
    """execute in the language ASKED FOR, never the detector's guess"""
    entry = find_entry(lang, code)
    got, st = None, "no_entry"
    if code and entry and lang in S.RUNNERS:
        for form in S.FORMS:
            st, got = S.RUNNERS[lang](code, entry, form)
            if st == "ok" and got is not None:
                break
    impl = next((k for k, v in REFV.items()
                 if got is not None and list(got) == v), None)
    if impl is None and got is not None:
        impl = which_function(got)
    return {"code": code, "lang_detect": detect_language(code), "status": st,
            "outputs": got, "impl": impl, "intact": impl == task}


# results/cabl_<model> is the example layout at wide width (the first runs);
# every other setting gets its own directory so nothing is mixed
tag = ("" if (a.layout, a.width) == ("example", "wide")
       else f"_{a.layout}" + ("" if a.width == "wide" else f"_{a.width}"))
outdir = Path(a.outdir or ROOT / "results" / f"cabl_{a.model}{tag}")
outdir.mkdir(parents=True, exist_ok=True)
for task in a.tasks:
    for lang in a.langs:
        for mode in a.modes:
            fp = outdir / f"ca_{task}_{lang}_{mode}.json"
            if fp.exists():
                continue
            if mode in REC_MODES + REPLAY_MODES and not RC_L:
                print(f"  {task:13s} {lang:8s} {mode:12s} skipped: no recurrent layers",
                      flush=True)
                continue
            gen, n_desc = generate(lang, task, mode)
            g = grade(program(lang, FDESC[task], gen, wide=WIDE, layout=a.layout),
                      lang, task)
            fp.write_text(json.dumps(
                {"model": a.model, "task": task, "lang": lang,
                 "cell": f"{task}:{lang}", "mode": mode, "layout": a.layout,
                 "width": a.width, "n_desc_rows": n_desc, "gen": gen, **g},
                indent=1))
            print(f"  {task:13s} {lang:8s} {mode:12s} desc={n_desc:>3} "
                  f"impl={str(g['impl']):13s} {g['status']:12s}"
                  + ("   INTACT" if g["intact"] else ""), flush=True)
print(f"saved -> {outdir}", flush=True)

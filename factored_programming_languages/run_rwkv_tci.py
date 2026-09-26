"""RWKV-7 as a fifth model in the TCI ladder (authors, 2026-09-14).

Emits cells in the format `figures/fig_tci_ladder4.py` reads (results/tci<SET>_<model>/c_*.json + meta.json), so
the attention-free model can be drawn beside gemma-4-31B, gemma-4-26B-A4B, qwen3.6-27B and the pythia control.

**Why there is no severed condition, and why that is fine** (authors: "the transport will just be unsevered but
again, after prefill. It is essentially equivalent since the model cannot reach back to the description").
Severing exists in the transformer runs to stop the recipient re-reading its own description during decode. RWKV-7
has no KV cache: after prefill the recurrent state is the only thing carried forward, so no token can be re-read,
by construction. The intervention arms therefore run on the recipient's natural state with `ablate_desc: "none"`,
which is the WORDS-INTACT regime (gemma's 9U set), and every rung's ceiling is the plain prompted ask.
Note what is NOT equivalent: gemma's severed state has lost the description's *content* (that is why its twists
regress to canon), whereas RWKV's state retains it.

Ops, factors and arms mirror run_tci_severed.py exactly:

  transport  recipient = full(A, lam), want B          F: delta = mean_{l != lam}[Z(B,l) - Z(A,l)]
  install    recipient = nofn(lam) / nolang(A), want A  F: delta = mean_{l != lam}[Z(A,l) - Z(nofn,l)]
  compose    recipient = carrier, want A in lam         F: the sum of the install F and L deltas
  arms       none | delta (held out) | exact (same-cell contrast) | rand (norm-matched) | wrongf/wrongl |
             meanf/meanl | prompted (ceiling)
  L factor   delta = mean_{f != cell}[Z(f, lam) - Z(f, SRC)]  (transport) or ... - Z(nolang, f)  (install)

    uv run --with "transformers==5.15.1" factored_programming_languages/run_rwkv_tci.py --device cuda:0 --langs java
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import execute as S                                                        # noqa: E402
from code_style import detect_entry                                        # noqa: E402
from elicit import extract_code                                            # noqa: E402
from tasks_function import BATTERY, FKEYS, WANT, prompt, which_function     # noqa: E402
from tasks_compose import CARRIER, nofn, nolang                            # noqa: E402
from tasks_strategy import STRATEGIES, detect_strategy, nostrat, strat      # noqa: E402
from entry import detect_language                                       # noqa: E402

MODEL_ID = "RWKV/RWKV7-G1j-13.3B-20260831"
MODEL_KEY = "rwkv7-g1j-13.3b"
SRC = "python"                                          # the donor language of the L factor, as in the 9F runs
# the gemma 9F recipient set, unchanged, so the columns are comparable. (An earlier version dropped haskell on the
# theory that its failures were missing imports; they are NOT -- run_haskell already injects the stdlib names a
# generation references, and the failures are the model's own type errors, e.g. `solve :: Integer -> Integer` with
# `digitToInt :: Char -> Int`. A hard recipient is a result, not an instrument fault; --sev conditioning handles it.)
# NOTE: this list is the donor pool too, so changing it invalidates every pooled delta -- re-run the whole set.
LANGS_T = ["c", "cpp", "go", "haskell", "java", "javascript", "ocaml", "ruby", "rust"]
PAIRS = [("digit_sum", "digit_prod"), ("digit_prod", "digit_sum"),
         ("even_digits", "sq_digit_sum"), ("sq_digit_sum", "even_digits")]   # (recipient fn A, target fn B)

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--impl", default="eager", choices=["eager", "chunked"])
ap.add_argument("--langs", nargs="+", default=LANGS_T, help="recipient languages (shard one per GPU)")
ap.add_argument("--ops", nargs="+", default=["transport", "install", "compose"])
ap.add_argument("--factors", nargs="+", default=["F", "L", "S"], help="which factor rungs to run")
ap.add_argument("--l-anchor", default="nolang", choices=["nolang", "python"],
                help="install-L base state: the language-free prompt (as in the transformer runs) or the SRC-language "
                     "prompt. authors, 2026-09-14: the language-free recipient BEHAVES as python, so the displacement "
                     "that should move it is python->lam, not nolang->lam.")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--set", default="9R", help="result-set tag: results/tci<SET>_<model>")
ap.add_argument("--patch-point", default="full", choices=["full", "prelast"],
                help="full: harvest/patch the state after the WHOLE prompt and decode from the recipient's own "
                     "last-token logits (the first generated token is then chosen BEFORE the patch -- and in RWKV "
                     "the language is usually decided by that token). prelast: harvest/patch after prompt[:-1], "
                     "then run the final token through the patched state so the first sampled token already sees "
                     "the intervention. Every prompt ends with the same TAIL, so the held-out token is identical "
                     "across prompts. (authors, 2026-09-14: 'it makes no sense that [language] would be unaddressable "
                     "-- everything is generated based on that hidden state'.)")
ap.add_argument("--max-new-tokens", type=int, default=320)
ap.add_argument("--single-donor-lang", default=None,
                help="S factor only: the donor grid. One held-out donor from THIS language (the first other function, as "
                     "run_tci_strategy.py --single-donor-lang does), arm `single`, cells written to "
                     "results/donorgrid[I]_<model>_<lang>/ in the format fig_donor_recipient.py reads (authors, 2026-09-24).")
ap.add_argument("--donor-para", default="0", help="wordings the donor prompts are pooled over (0 canonical, 1-3 = tasks_atlas_paraphrases.PARA); every contrast becomes the mean over these wordings")
ap.add_argument("--recipient-para", default="0", choices=["0", "1", "2", "3", "rotate"])
a = ap.parse_args()
S.BATTERY = BATTERY
from tasks_atlas_paraphrases import PARA                                  # noqa: E402
from languages import DISPLAY as _DISP                                    # noqa: E402
from tasks_function import FDESC as _FD, FKEYS as _FK, LANGS as _LANGS, TAIL as _TAIL   # noqa: E402
DPARA = [int(x) for x in a.donor_para.split(",")]


def RP(kind, f, lam):
    if kind not in ("full", "nolang"):
        return 0
    if a.recipient_para != "rotate":
        return int(a.recipient_para)
    return 1 + ((_LANGS.index(lam) if lam in _LANGS else 0) + _FK.index(f)) % 3
outdir = ROOT / "results" / f"tci{a.set}_{MODEL_KEY}"
outdir.mkdir(parents=True, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer                # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
                                             device_map=a.device)
model.eval()
model.config.wkv_implementation = a.impl
dev = next(model.parameters()).device
NL = model.config.num_hidden_layers
EOS = {0} | ({int(tok.eos_token_id)} if tok.eos_token_id is not None else set())
print(f"[load] {MODEL_ID} in {time.time() - t0:.0f}s; {NL} layers", flush=True)


def text_of(kind, f, lam, p=0):
    """the prompt shapes of the ladder; strat/nostrat carry the strategy slot (f is (fkey, s) there)"""
    if p and kind in ("full", "nolang"):          # a paraphrased wording (--donor-para / --recipient-para)
        d = PARA[f][p - 1]
        return (f"Write a function in {_DISP[lam]} that {d}. {_TAIL}" if kind == "full" else f"Write a function that {d}. {_TAIL}")
    if kind == "strat":
        return strat(lam, f[0], f[1])
    if kind == "nostrat":
        return nostrat(lam, f)
    return {"full": lambda: prompt(lam, f), "nofn": lambda: nofn(lam),
            "nolang": lambda: nolang(f), "carrier": lambda: CARRIER}[kind]()


def ids_of(text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True, thinking=False)
    return tok(s, return_tensors="pt", add_special_tokens=False).to(dev)["input_ids"]


@torch.no_grad()            # NOT inference_mode: harvested states are copied back into a cache
def prefill(text, upto_last=False):
    """(logits, state). upto_last: consume prompt[:-1] only, so the final token can be replayed through a
    PATCHED state -- the logits returned are then the ones before that token and must not be used to decode."""
    ids = ids_of(text)
    if upto_last:
        ids = ids[:, :-1]
    out = model(input_ids=ids, use_cache=True)
    return out.logits[:, -1], out.state


@torch.no_grad()
def replay_last(text, state):
    """run the prompt's final token through `state`; returns the logits the patched state produces"""
    last = ids_of(text)[:, -1:]
    o = model(input_ids=last, state=state, use_cache=True)
    return o.logits[:, -1], o.state


@torch.no_grad()
def decode(logits, state):
    toks = []
    for _ in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in EOS:
            break
        toks.append(nxt)
        o = model(input_ids=torch.tensor([[nxt]], device=dev), state=state, use_cache=True)
        logits, state = o.logits[:, -1], o.state
    return tok.decode(toks, skip_special_tokens=True)


def harvest(state):
    # kept on the host: with four wordings per prompt the per-prompt states no longer fit beside other jobs on the card
    return [tuple(t.detach().to('cpu', copy=True) for t in state.read(i)) for i in range(NL)]


def install(state, snap):
    for i, (att, ffn, wkv) in enumerate(snap):
        state.write(i, att.to(dev).clone(), ffn.to(dev).clone(), wkv.to(dev).clone())
    return state


ST, LG = {}, {}


def state_of(kind, f, lam, p=0):
    k = (kind, f, lam, p)
    if k not in ST:
        lg, st = prefill(text_of(kind, f, lam, p), upto_last=(a.patch_point == "prelast"))
        ST[k], LG[k] = harvest(st), lg
    return ST[k]


def sub(A, B):
    return [tuple(A[i][s] .float() - B[i][s].float() for s in range(3)) for i in range(NL)]


def mean_delta(pairs):
    """mean over (to, from) state-key pairs of [Z(to) - Z(from)], exactly as the transformer runner does"""
    ds = [sub(state_of(*to, p), state_of(*fr, p)) for to, fr in pairs for p in DPARA]   # --donor-para
    return [tuple(torch.stack([d[i][s] for d in ds]).mean(0) for s in range(3)) for i in range(NL)]


def add_deltas(D1, D2):
    return [tuple(D1[i][s] + D2[i][s] for s in range(3)) for i in range(NL)]


def rand_matched(D, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(NL):
        row = []
        for s in range(3):
            v = D[i][s]
            r = torch.randn(v.shape, generator=g).to(v.device)
            row.append(r * (v.norm() / (r.norm() + 1e-9)))
        out.append(tuple(row))
    return out


def apply_to(base, D, alpha):
    if D is None:
        return base
    return [tuple((base[i][s].float() + alpha * D[i][s]).to(base[i][s].dtype) for s in range(3)) for i in range(NL)]


def run(tag, rec_key, D, meta):
    fp = outdir / f"c_{tag}.json"
    if fp.exists():
        return
    rp = RP(*rec_key)                              # --recipient-para
    snap = apply_to(state_of(*rec_key, rp), D, a.alpha)
    _, st = prefill(text_of(*rec_key, rp), upto_last=(a.patch_point == "prelast"))
    install(st, snap)
    if a.patch_point == "prelast":            # the first sampled token now sees the intervention
        logits, st = replay_last(text_of(*rec_key, rp), st)
    else:
        logits = LG[tuple(rec_key) + (rp,)]
    txt = decode(logits, st)
    code, _ = extract_code(txt, "auto")
    lam = meta["lang"]
    det = detect_language(code) if code else "unknown"
    entry = detect_entry(lam, code, None) if code else None
    stt, got = "no_entry", None
    if code and entry and lam in S.RUNNERS:
        for form in S.FORMS:
            stt, got = S.RUNNERS[lam](code, entry, form)
            if stt == "ok" and got:
                break
    impl = which_function(list(got)) if got else None
    strat_got = detect_strategy(code, lam) if code else "unknown"
    if meta["factor"] == "S":                     # the S claim: the requested strategy, function preserved
        hit_fn = (strat_got == meta["want_strategy"]) and impl == meta["want"]
    else:
        hit_fn = impl == meta["want"]
    built = stt == "ok"                       # it ran under the requested language's own runner
    rec = {"model": MODEL_KEY, **meta, "channel": "kv", "alpha": a.alpha, "ablate_desc": "none",
           "recipient": [rec_key[0], rec_key[1] if rec_key[1] else "-", rec_key[2] if rec_key[2] else "-"],
           "hit": hit_fn, "hit_fn": hit_fn, "built": built, "entry": entry, "status": stt, "strategy": strat_got,
           "impl": impl, "lang_detect": det, "code": code, "text": txt,
           "hit_strat": strat_got == meta.get("want_strategy"), "fn_ok": impl == meta["want"]}   # separate reads (grid figure)
    fp.write_text(json.dumps(rec, indent=1))
    print(f"  {meta['op']:9s} {meta['factor']:2s} {meta['cell']:24s} {lam:10s} {meta['arm']:9s} "
          f"{stt:12s} impl={str(impl):13s}" + ("  HIT" if hit_fn else ""), flush=True)


for lam in a.langs:
    oth = [l for l in LANGS_T + [SRC] if l != lam]                 # donor languages for the F factor
    _rot = [x for x in LANGS_T if x != SRC]
    lam2 = _rot[(_rot.index(lam) + 1) % len(_rot)] if lam in _rot and len(_rot) > 1 else None
    for op in a.ops:
        pairs = PAIRS if op == "transport" else [(f, f) for f in FKEYS]
        for A, B in pairs:
            cell = f"{B}:{A}" if op == "transport" else B
            othf = [f for f in FKEYS if f != B]
            # ---------------- F factor ----------------
            if op == "transport":
                rec_key = ("full", A, lam)
                delta = mean_delta([(("full", B, l), ("full", A, l)) for l in oth])
                exact = mean_delta([(("full", B, lam), ("full", A, lam))])
                wrongf = mean_delta([(("full", next(f for f in FKEYS if f not in (A, B)), l), ("full", A, l))
                                     for l in oth])
                meanf = mean_delta([(("full", f, l), ("full", A, l)) for f in FKEYS for l in oth])
            elif op == "install":
                rec_key = ("nofn", None, lam)
                delta = mean_delta([(("full", B, l), ("nofn", None, l)) for l in oth])
                exact = mean_delta([(("full", B, lam), ("nofn", None, lam))])
                wrongf = mean_delta([(("full", next(f for f in FKEYS if f != B), l), ("nofn", None, l)) for l in oth])
                meanf = mean_delta([(("full", f, l), ("nofn", None, l)) for f in FKEYS for l in oth])
            else:
                rec_key = ("carrier", None, None)
                delta = add_deltas(mean_delta([(("full", B, l), ("nofn", None, l)) for l in oth]),
                                   mean_delta([(("full", f, lam), ("nolang", f, None)) for f in othf]))
                exact = add_deltas(mean_delta([(("full", B, lam), ("nofn", None, lam))]),
                                   mean_delta([(("full", B, lam), ("nolang", B, None))]))
                _wA = next(f for f in FKEYS if f != B)
                _wl = next(x for x in LANGS_T if x != lam)
                wrongf = add_deltas(mean_delta([(("full", _wA, l), ("nofn", None, l)) for l in oth]),
                                    mean_delta([(("full", f, _wl), ("nolang", f, None)) for f in othf]))
                meanf = add_deltas(mean_delta([(("full", f, l), ("nofn", None, l)) for f in FKEYS for l in oth]),
                                   mean_delta([(("full", f, x), ("nolang", f, None)) for f in othf for x in LANGS_T]))
            armsF = [("none", None), ("delta", delta), ("exact", exact), ("rand", rand_matched(delta, 0)),
                     ("wrongf", wrongf), ("meanf", meanf)] if "F" in a.factors else []
            for arm, D in armsF:
                run(f"F_{op}_{cell.replace(':', '-')}_{lam}_kv_{arm}", rec_key, D,
                    {"op": op, "factor": "F", "cell": cell, "lang": lam, "arm": arm, "want": B})
            # ---------------- L factor (not defined for compose: the F line already carries both) ----
            if op != "compose" and "L" in a.factors:
                AL = A if op == "transport" else B
                if op == "transport":
                    rec_keyL = ("full", AL, SRC)
                    Ldelta = mean_delta([(("full", f, lam), ("full", f, SRC)) for f in othf])
                    Lexact = mean_delta([(("full", AL, lam), ("full", AL, SRC))])
                    wrongl = (mean_delta([(("full", f, lam2), ("full", f, SRC)) for f in othf]) if lam2 else None)
                    meanl = (mean_delta([(("full", f, x), ("full", f, SRC)) for x in _rot for f in othf])
                             if lam2 else None)
                else:
                    rec_keyL = ("nolang", AL, None)
                    base_of = ((lambda f: ("nolang", f, None)) if a.l_anchor == "nolang"
                               else (lambda f: ("full", f, SRC)))
                    Ldelta = mean_delta([(("full", f, lam), base_of(f)) for f in othf])
                    Lexact = mean_delta([(("full", AL, lam), base_of(AL))])
                    wrongl = (mean_delta([(("full", f, lam2), base_of(f)) for f in othf]) if lam2 else None)
                    meanl = (mean_delta([(("full", f, x), base_of(f)) for x in _rot for f in othf])
                             if lam2 else None)
                armsL = [("none", None), ("delta", Ldelta), ("exact", Lexact), ("rand", rand_matched(Ldelta, 1))]
                if wrongl is not None:
                    armsL += [("wrongl", wrongl), ("meanl", meanl)]
                for arm, D in armsL:
                    run(f"L_{op}_{cell.replace(':', '-')}_{lam}_kv_{arm}", rec_keyL, D,
                        {"op": op, "factor": "L", "cell": cell, "lang": lam, "arm": arm, "want": AL})
            # ---------------- S factor: recursion vs iteration ----------------
            if "S" in a.factors and op != "compose" and a.single_donor_lang:
                dl = a.single_donor_lang
                if dl != lam and dl in LANGS_T:                              # a recipient's own language is never its donor
                    outdir = ROOT / "results" / f"{'donorgrid' if op == 'transport' else 'donorgridI'}_{MODEL_KEY}_{dl}"
                    outdir.mkdir(parents=True, exist_ok=True); d0 = othf[0]
                    for want_s in STRATEGIES:
                        other_s = next(x for x in STRATEGIES if x != want_s)
                        if op == "install":
                            rec_keyS = ("nostrat", B, lam)
                            Ssingle = mean_delta([(("strat", (d0, want_s), dl), ("nostrat", d0, dl))])
                        else:
                            rec_keyS = ("strat", (B, other_s), lam)
                            Ssingle = mean_delta([(("strat", (d0, want_s), dl), ("strat", (d0, other_s), dl))])
                        run(f"S_{op}_{cell.replace(':', '-')}-{want_s}_{lam}_kv_single_{dl}", rec_keyS, Ssingle,
                            {"op": op, "factor": "S", "cell": B, "lang": lam, "arm": "single", "want": B,
                             "want_strategy": want_s, "from_strategy": other_s if op == "transport" else None,
                             "donor": [d0, dl, None], "n_donors": 1, "donor_pool": "single"})
                    outdir = ROOT / "results" / f"tci{a.set}_{MODEL_KEY}"
            elif "S" in a.factors and op != "compose":
                for want_s in STRATEGIES:
                    other_s = next(x for x in STRATEGIES if x != want_s)
                    if op == "install":
                        rec_keyS = ("nostrat", B, lam)
                        Sdelta = mean_delta([(("strat", (f, want_s), l), ("nostrat", f, l))
                                             for f in othf for l in oth])
                        Sexact = mean_delta([(("strat", (B, want_s), lam), ("nostrat", B, lam))])
                        Swrong = mean_delta([(("strat", (f, other_s), l), ("nostrat", f, l))
                                             for f in othf for l in oth])
                    else:
                        rec_keyS = ("strat", (B, other_s), lam)
                        Sdelta = mean_delta([(("strat", (f, want_s), l), ("strat", (f, other_s), l))
                                             for f in othf for l in oth])
                        Sexact = mean_delta([(("strat", (B, want_s), lam), ("strat", (B, other_s), lam))])
                        Swrong = mean_delta([(("strat", (f, other_s), l), ("strat", (f, want_s), l))
                                             for f in othf for l in oth])
                    armsS = [("none", None), ("delta", Sdelta), ("exact", Sexact),
                             ("rand", rand_matched(Sdelta, 2)), ("wrongs", Swrong)]
                    for arm, D in armsS:
                        run(f"S_{op}_{cell.replace(':', '-')}-{want_s}_{lam}_kv_{arm}", rec_keyS, D,
                            {"op": op, "factor": "S", "cell": f"{cell}-{want_s}", "lang": lam, "arm": arm,
                             "want": B, "want_strategy": want_s})
                    tagS = f"S_{op}_{cell.replace(':', '-')}-{want_s}_{lam}_kv_prompted"
                    if not (outdir / f"c_{tagS}.json").exists():
                        run(tagS, ("strat", (B, want_s), lam), None,
                            {"op": op, "factor": "S", "cell": f"{cell}-{want_s}", "lang": lam, "arm": "prompted",
                             "want": B, "want_strategy": want_s})

            # ---------------- the ceiling: the target asked for directly, words intact ----------------
            tag = f"F_{op}_{cell.replace(':', '-')}_{lam}_kv_prompted"
            fp = outdir / f"c_{tag}.json"
            if "F" in a.factors and not fp.exists():
                run(tag, ("full", B, lam), None,
                    {"op": op, "factor": "F", "cell": cell, "lang": lam, "arm": "prompted", "want": B})

(outdir / "meta.json").write_text(json.dumps({
    "set": a.set, "model": MODEL_KEY, "hf_name": MODEL_ID,
    "ceiling": {"transport": "prompted", "install": "prompted", "compose": "prompted"},
    "ablate_desc": "none",
    "note": ("RWKV-7 has no KV cache: after prefill nothing can be re-read, so the intervention arms need no "
             "severing and the ceiling is the plain prompted ask (the words-intact regime, gemma's 9U). The "
             "state does retain the description's content -- see the parameter tier."),
    "langs": a.langs, "src_lang": SRC, "pairs": PAIRS, "alpha": a.alpha,
}, indent=1))
print(f"[done] -> {outdir}", flush=True)

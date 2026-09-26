"""RWKV-7 tier ladder: does the attention-free state carry a non-canonical PARAMETER and a MODIFIER, or only
task identity? (authors, 2026-09-14, after the clean function-factor result: "let's try that too".)

run_rwkv_factor.py showed the paper's function factor installs 20/20 with controls at 0 on RWKV7-G1j-13.3B.
That is tier 1 (known algorithm = a binding into prior knowledge), the tier gemma also does well. The question is
whether the recurrent bottleneck is cleaner at the tiers where gemma FAILS:

  tier 2a PARAMETER (run_param_install.py's minimal pair; gemma: p4 donor installs canonical p89 6/6, parameter
          survives 0/12): install F(p4) and F(p89) into the carrier and read the stop-set the code implements.
          Binding-only -> both install p89. Content-bearing -> the p4 donor installs p4.
  tier 2b MODIFIER (modifier_transfer_dho.py install mode; gemma: pooled 46/160, squaring placement regresses):
          Z~ = Z(nofn, lam_b) + beta*F(f_b) + alpha*Delta_M with Delta_M = mean[Z(M[f_a],lam_a) - Z(f_a,lam_a)]
          over f_a != f_b, lam_a != lam_b, other wording. Same double hold-out as the gemma run.

Arms. Parameter: carrier | install_loo_p4 | install_loo_p89 | install_exact_p4 | install_exact_p89 | rand |
gate_p4 | gate_p89 (the explicit prompts). Modifier: base (beta F alone) | pooled | single | rand | ceil_install
(F(M[f_b]) alone) | ceiling (explicit prompt).

    uv run --with "transformers==5.15.1" factored_programming_languages/run_rwkv_tiers.py --device cuda:0
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
from languages import DISPLAY                                              # noqa: E402
from tasks_compose import TAIL, nofn                                       # noqa: E402
from modifier_dho_grid import BASES, MODS, WANTV, key, desc                # noqa: E402

MODEL_ID = "RWKV/RWKV7-G1j-13.3B-20260831"
LANGS = ["python", "javascript", "java", "go", "rust", "haskell"]   # the gemma modifier grid's six languages
DONOR_LANGS = list(LANGS)

# ---- tier 2a: the parameter minimal pair (identical to run_param_install.py) ---------------------------------
PDESC = {"p4": ("returns how many times the sum-of-squared-digits map must be applied to a positive integer "
                "before reaching 1 or 4"),
         "p89": ("returns how many times the sum-of-squared-digits map must be applied to a positive integer "
                 "before reaching 1 or 89")}
PBAT = [1, 2, 7, 4, 19, 89, 12345]


def _steps(n, stopset):
    c = 0
    while n not in stopset:
        n = sum(int(d) ** 2 for d in str(n))
        c += 1
    return c


PWANT = {"p4": [_steps(x, {1, 4}) for x in PBAT], "p89": [_steps(x, {1, 89}) for x in PBAT]}
assert sum(x != y for x, y in zip(PWANT["p4"], PWANT["p89"])) >= 3

# ---- tier 2b: the modifier grid (same objects as the gemma double-held-out run) -------------------------------
MBAT = [7, 1234, 5, 90, 406, 2468, 3917, 8052]
ALL_BASES = list(BASES)      # sum, prod, max, min, altsum -- ALWAYS the donor pool for Delta_M and the harvest set
MOD_BASES_DEFAULT = ALL_BASES                                   # --bases selects RECIPIENT bases (sharding only)

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--impl", default="eager", choices=["eager", "chunked"])
ap.add_argument("--tiers", nargs="+", default=["param", "modifier"])
ap.add_argument("--langs", nargs="+", default=LANGS, help="recipient languages; shard across GPUs with one process per language")
ap.add_argument("--mode", default="install", choices=["install", "transport"],
                help="install: Z(nofn,lam_b) + beta*F(f_b) + alpha*Delta_M (build the base from a carrier). "
                     "transport: Z(f_b, lam_b) + alpha*Delta_M -- the recipient's OWN prompt state, the analogue "
                     "of the transformers' severed transport. No severing is needed or possible here: the "
                     "recurrent state is the only store, so overriding it overrides everything the description "
                     "could have left behind (authors, 2026-09-16).")
ap.add_argument("--wordings", nargs="+", type=int, default=[0, 1],
                help="recipient wordings (the gemma double-held-out run varied both; donors always take the other)")
ap.add_argument("--bases", nargs="+", default=None, help="RECIPIENT bases to run (donors are always all five)")
ap.add_argument("--beta", type=float, default=1.0)
ap.add_argument("--alphas", nargs="+", type=float, default=[1.0, 1.5])
ap.add_argument("--max-new-tokens", type=int, default=320)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()
MOD_BASES = a.bases or MOD_BASES_DEFAULT
outdir = Path(a.outdir or ROOT / "results" / "rwkv_tiers")
outdir.mkdir(parents=True, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer                # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
                                             device_map=a.device)
model.eval()
model.config.wkv_implementation = a.impl
dev = next(model.parameters()).device
n_layers = model.config.num_hidden_layers
EOS = {0} | ({int(tok.eos_token_id)} if tok.eos_token_id is not None else set())
print(f"[load] {MODEL_ID} in {time.time() - t0:.0f}s; {n_layers} layers", flush=True)


def ids_of(text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True, thinking=False)
    return tok(s, return_tensors="pt", add_special_tokens=False).to(dev)["input_ids"]


@torch.no_grad()          # NOT inference_mode: harvested states are copied back into a cache
def prefill(text):
    out = model(input_ids=ids_of(text), use_cache=True)
    return out.logits[:, -1], out.state


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
    return [tuple(t.detach().clone() for t in state.read(i)) for i in range(n_layers)]


def install(state, snap):
    for i, (att_shift, ffn_shift, wkv) in enumerate(snap):
        state.write(i, att_shift.clone(), ffn_shift.clone(), wkv.clone())
    return state


def sub(A, B):
    return [tuple(A[i][s_].float() - B[i][s_].float() for s_ in range(3)) for i in range(n_layers)]


def mean(ds):
    return [tuple(torch.stack([d[i][s_] for d in ds]).mean(0) for s_ in range(3)) for i in range(n_layers)]


def plus(base, *terms):
    """base + sum(coef * delta); terms are (delta, coef)"""
    out = []
    for i in range(n_layers):
        row = []
        for s_ in range(3):
            v = base[i][s_].float()
            for d, c in terms:
                v = v + c * d[i][s_]
            row.append(v.to(base[i][s_].dtype))
        out.append(tuple(row))
    return out


def rand_like(D, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n_layers):
        row = []
        for s_ in range(3):
            v = D[i][s_]
            r = torch.randn(v.shape, generator=g).to(v.device)
            row.append(r * (v.norm() / (r.norm() + 1e-9)))
        out.append(tuple(row))
    return out


def run_cell(fp, prompt_key_state, logits, snap, meta, battery, wants, want_key):
    """decode with `snap` installed (or None) and grade by exact output-vector match against `wants`"""
    if fp.exists():
        return json.loads(fp.read_text())
    S.BATTERY = battery
    _, st = prefill(prompt_key_state)
    if snap is not None:
        install(st, snap)
    txt = decode(logits, st)
    code, _ = extract_code(txt, "auto")
    lam = meta["lang"]
    entry = detect_entry(lam, code, None) if code else None
    stt, got = "no_entry", None
    if code and entry and lam in S.RUNNERS:
        for form in S.FORMS:
            stt, got = S.RUNNERS[lam](code, entry, form)
            if stt == "ok" and got:
                break
    impls = [k for k, v in wants.items() if got is not None and list(got) == list(v)]
    rec = {"model": MODEL_ID, **meta, "status": stt, "outputs": got, "impls": impls,
           "impl": "|".join(impls) if impls else None, "hit": want_key in impls, "code": code, "text": txt}
    fp.write_text(json.dumps(rec, indent=1))
    print(f"  {meta['tier']:5s} {meta.get('fn', ''):14s} {lam:11s} {meta['arm']:16s} {stt:12s} "
          f"impl={str(rec['impl']):14s}" + ("  HIT" if rec["hit"] else ""), flush=True)
    return rec


# ================================================================= tier 2a: parameter =========================
if "param" in a.tiers:
    print("\n[tier 2a] parameter retention (happy_steps stop-at-4 vs stop-at-89)", flush=True)
    full_p = lambda v, lam: f"Write a function in {DISPLAY[lam]} that {PDESC[v]}. {TAIL}"    # noqa: E731
    CARR, LOG, PST = {}, {}, {}
    for lam in DONOR_LANGS:
        lg, st = prefill(nofn(lam)); CARR[lam], LOG[lam] = harvest(st), lg
        for v in ("p4", "p89"):
            PST[(v, lam)] = harvest(prefill(full_p(v, lam))[1])
    for lam in a.langs:
        others = [l for l in DONOR_LANGS if l != lam]
        for v in ("p4", "p89"):
            loo = mean([sub(PST[(v, l)], CARR[l]) for l in others])
            exact = sub(PST[(v, lam)], CARR[lam])            # same language, not held out: the decisive arm
            arms = [(f"install_loo_{v}", plus(CARR[lam], (loo, a.beta))),
                    (f"install_exact_{v}", plus(CARR[lam], (exact, a.beta)))]
            if v == "p4":
                arms.append(("rand", plus(CARR[lam], (rand_like(loo, abs(hash(("p", lam))) % 2**31), a.beta))))
                arms.append(("carrier", CARR[lam]))
            for name, snap in arms:
                run_cell(outdir / f"param_{lam}__{name}.json", nofn(lam), LOG[lam], snap,
                         {"tier": "param", "lang": lam, "arm": name, "fn": v}, PBAT, PWANT, v)
        for v in ("p4", "p89"):                              # gates: the explicit prompts
            fp = outdir / f"param_{lam}__gate_{v}.json"
            if not fp.exists():
                lg, _ = prefill(full_p(v, lam))
                run_cell(fp, full_p(v, lam), lg, None, {"tier": "param", "lang": lam, "arm": f"gate_{v}", "fn": v},
                         PBAT, PWANT, v)

# ================================================================= tier 2b: modifier ==========================
if "modifier" in a.tiers:
    print("\n[tier 2b] modifier transfer (double held out: f_b and lam_b out of Delta_M)", flush=True)
    ask = lambda k, lam, w: f"Write a function in {DISPLAY[lam]} that {desc(k[0], k[1], w)}. {TAIL}"  # noqa: E731
    MC, MLOG, MST = {}, {}, {}
    for lam in DONOR_LANGS:
        lg, st = prefill(nofn(lam)); MC[lam], MLOG[lam] = harvest(st), lg
    for b in ALL_BASES:                                            # harvest the whole grid: donors come from it
        for lam in DONOR_LANGS:
            for w in (0, 1):
                MST[(b, None, lam, w)] = harvest(prefill(ask((b, None), lam, w))[1])
                for m in MODS:
                    MST[(b, m, lam, w)] = harvest(prefill(ask((b, m), lam, w))[1])
    print(f"[harvest] {len(MST)} modifier-grid states", flush=True)
    MWANT = {k: v for k, v in WANTV.items()}
    for lam_b in a.langs:
        others = [l for l in DONOR_LANGS if l != lam_b]
        for b in MOD_BASES:
          for wb in a.wordings:                                   # recipient wording; donors take the other one
            wd = 1 - wb
            F = mean([sub(MST[(b, None, l, wb)], MC[l]) for l in others])         # base factor, recipient held out
            for m in MODS:
                want = key(b, m)
                D = mean([sub(MST[(fa, m, la, wd)], MST[(fa, None, la, wd)])
                          for fa in ALL_BASES if fa != b for la in others])
                fa1 = ALL_BASES[(ALL_BASES.index(b) + 1) % len(ALL_BASES)]
                single = sub(MST[(fa1, m, others[0], wd)], MST[(fa1, None, others[0], wd)])
                # wrong modifier: the next one whose target on THIS base differs extensionally (post_sq and pre_sq
                # coincide on prod / max / min, so there the control must skip its twin) -- modifier_dho_grid.py
                wm = next(mm for k_ in range(1, len(MODS))
                          for mm in [MODS[(MODS.index(m) + k_) % len(MODS)]]
                          if WANTV[key(b, mm)] != WANTV[key(b, m)])
                W = mean([sub(MST[(fa, wm, la, wd)], MST[(fa, None, la, wd)])
                          for fa in ALL_BASES if fa != b for la in others])
                meta = {"tier": "mod", "lang": lam_b, "fn": want, "base": b, "mod": m, "wording": wb,
                        "wrong_mod": wm, "mode": a.mode}
                if a.mode == "transport":
                    # recipient = the model's own prompt for f_b; nothing is severed and nothing is installed
                    rec_txt, rec_lg = ask((b, None), lam_b, wb), None
                    _lg, _st = prefill(rec_txt); rec_lg = _lg
                    RB = harvest(_st)
                    run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__none.json", rec_txt, rec_lg, RB,
                             {**meta, "arm": "none"}, MBAT, MWANT, key(b))
                    for al in a.alphas:
                        run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__pooled_a{al}.json", rec_txt, rec_lg,
                                 plus(RB, (D, al)), {**meta, "arm": f"pooled_a{al}", "alpha": al}, MBAT, MWANT, want)
                    run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__single.json", rec_txt, rec_lg,
                             plus(RB, (single, a.alphas[0])), {**meta, "arm": "single"}, MBAT, MWANT, want)
                    run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__rand.json", rec_txt, rec_lg,
                             plus(RB, (rand_like(D, abs(hash((m, b, lam_b))) % 2**31), a.alphas[0])),
                             {**meta, "arm": "rand"}, MBAT, MWANT, want)
                    run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__wrong.json", rec_txt, rec_lg,
                             plus(RB, (W, a.alphas[0])), {**meta, "arm": "wrong"}, MBAT, MWANT, want)
                    fpc = outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__ceiling.json"
                    if not fpc.exists():
                        clg, _ = prefill(ask((b, m), lam_b, wb))
                        run_cell(fpc, ask((b, m), lam_b, wb), clg, None, {**meta, "arm": "ceiling"},
                                 MBAT, MWANT, want)
                    continue
                run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__base.json", nofn(lam_b), MLOG[lam_b],
                         plus(MC[lam_b], (F, a.beta)), {**meta, "arm": "base"}, MBAT, MWANT, key(b))
                for al in a.alphas:
                    run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__pooled_a{al}.json", nofn(lam_b), MLOG[lam_b],
                             plus(MC[lam_b], (F, a.beta), (D, al)), {**meta, "arm": f"pooled_a{al}", "alpha": al},
                             MBAT, MWANT, want)
                run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__single.json", nofn(lam_b), MLOG[lam_b],
                         plus(MC[lam_b], (F, a.beta), (single, a.alphas[0])), {**meta, "arm": "single"},
                         MBAT, MWANT, want)
                run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__rand.json", nofn(lam_b), MLOG[lam_b],
                         plus(MC[lam_b], (F, a.beta), (rand_like(D, abs(hash((m, b, lam_b))) % 2**31), a.alphas[0])),
                         {**meta, "arm": "rand"}, MBAT, MWANT, want)
                run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__wrong.json", nofn(lam_b), MLOG[lam_b],
                         plus(MC[lam_b], (F, a.beta), (W, a.alphas[0])), {**meta, "arm": "wrong"}, MBAT, MWANT, want)
                Fm = mean([sub(MST[(b, m, l, wb)], MC[l]) for l in others])
                run_cell(outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__ceil_install.json", nofn(lam_b), MLOG[lam_b],
                         plus(MC[lam_b], (Fm, a.beta)), {**meta, "arm": "ceil_install"}, MBAT, MWANT, want)
                fp = outdir / f"mod_{m}_{b}_{lam_b}_w{wb}__ceiling.json"
                if not fp.exists():
                    lg, _ = prefill(ask((b, m), lam_b, wb))
                    run_cell(fp, ask((b, m), lam_b, wb), lg, None, {**meta, "arm": "ceiling"}, MBAT, MWANT, want)
print(f"[done] -> {outdir}", flush=True)

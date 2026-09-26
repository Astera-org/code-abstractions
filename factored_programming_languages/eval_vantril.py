"""Grade a model (base or vantril-teacher) on the novel computation, and run
the eligibility gates that fine_tuning.md section 4 requires before any
cross-language failure counts as evidence about reuse.

    vantril(n) = number of DISTINCT decimal digits of n

BATTERY: the campaign's standard inputs cannot grade this -- none of
[7, 1234, 5, 90, 406, 2468] has a repeated digit, so "count distinct digits"
and "count digits" agree on every one. VANTRIL_BATTERY below is chosen so the
two differ on most inputs (777 -> 1 not 3; 1002 -> 3 not 4; 112358 -> 5 not 6),
which also rules out the commonest near-miss implementations.

Two gates, both reported per (model, language):
  learned    does it produce a CORRECT vantril in Python?  (teacher only)
  renders    can it write a KNOWN function (digit_sum) in this language at all?
A model that fails `renders` for a language is not eligible to be scored on
vantril there -- its failure would be about the language, not about reuse.

    uv run factored_programming_languages/eval_vantril.py --model pythia-6.9b-tulu \
        --ckpt results/allenai--.../sft_vantril/full_vantril_portable --device cuda:0
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from elicit import extract_code, render_prompt                            # noqa: E402
from recipes import RECIPES                                               # noqa: E402
from tasks_function import LANGS, FDESC                                   # noqa: E402
from languages import DISPLAY                                             # noqa: E402
import execute as S                                                       # noqa: E402
import loader                                                             # noqa: E402
from entry import find_entry, detect_language                          # noqa: E402

VANTRIL_BATTERY = [7, 777, 1002, 112358, 90, 2468, 100, 9999]
VANTRIL_WANT = [len(set(str(n))) for n in VANTRIL_BATTERY]   # 1,1,3,5,2,4,2,1
DIGIT_SUM_WANT = [sum(int(c) for c in str(n)) for n in VANTRIL_BATTERY]

# Held-out Python acquisition prompts for the section 7.1 selection rule
# (choose the teacher by ACQUISITION, S_python >= 0.90, not by update count).
# The three training paraphrases are "Give me a Python implementation of...",
# "Implement ... as a Python function", "Write a Python function that...", each
# ending "no explanation, no examples, no tests". These deliberately differ in
# framing and in the trailing instruction so the score is not a memorisation
# read-out of a seen prompt.
ACQ_PROMPTS = [
    "Define vantril(n) in Python for a non-negative integer n. Code only.",
    "I need a Python routine named vantril taking one integer argument. "
    "Output nothing but the definition.",
    "In Python: a single function, vantril, of one whole-number parameter. "
    "Source only, please.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ckpt", default=None, help="teacher checkpoint; omit for base")
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--langs", nargs="+", default=list(LANGS))
ap.add_argument("--max-new-tokens", type=int, default=320)
ap.add_argument("--tag", default=None)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()
recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]

from transformers import (AutoTokenizer, AutoModelForCausalLM,            # noqa: E402
                          AutoModelForImageTextToText)
import yaml                                                               # noqa: E402

if a.ckpt:
    src = str(Path(a.ckpt).resolve())
    _trc = ({"trust_remote_code": True} if yaml.safe_load(open(Path(__file__).resolve().parent.parent / "models.yaml"))
            .get(a.model, {}).get("trust_remote_code") else {})                   # custom-code models (RWKV-7)
    tok = AutoTokenizer.from_pretrained(src, **_trc)
    # Same class dispatch as run_sft_vantril.py / loader.py -- the multimodal
    # checkpoints (mistral-3.2, qwen3.6, gemma-4) cannot be reloaded through
    # AutoModelForCausalLM, so a teacher gate would fail at load time.
    _cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "models.yaml"))
    _mc = _cfg.get(a.model, {}).get("model_class")
    _cls = (AutoModelForImageTextToText if _mc == "image_text_to_text"
            else AutoModelForCausalLM)
    model = _cls.from_pretrained(
        src, dtype=torch.bfloat16, device_map=a.device or "auto",
        attn_implementation="eager", **_trc)
    model.eval()
    dev = next(model.parameters()).device
else:
    tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]),
                                        **recipe.get("tokenizer_kwargs", {}))
    _L = loader.load(a.model, device=a.device, max_memory=a.max_memory, attn="sdpa")
    model, dev = _L.model, _L.device

gc = model.generation_config
stop = set()
for e in ([gc.eos_token_id] if isinstance(gc.eos_token_id, int) else (gc.eos_token_id or [])):
    stop.add(int(e))
if tok.eos_token_id is not None:
    stop.add(int(tok.eos_token_id))
stop_strings = recipe.get("stop_strings") or []


@torch.inference_mode()
def gen(user):
    ids = tok(render_prompt(tok, recipe, user), return_tensors="pt",
              add_special_tokens=False).to(dev)["input_ids"]
    out = model(input_ids=ids, use_cache=True)
    # prompt memory: a KV cache on transformers, the recurrent `state` on RWKV-7 (same greedy loop either way)
    recurrent = getattr(out, "past_key_values", None) is None and getattr(out, "state", None) is not None
    cache, logits, toks = (out.state if recurrent else out.past_key_values), out.logits[:, -1], []
    for _ in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in stop:
            break
        toks.append(nxt)
        if stop_strings and any(t_ in tok.decode(toks[-24:], skip_special_tokens=True)
                                for t_ in stop_strings):
            break
        o = model(input_ids=torch.tensor([[nxt]], device=dev), use_cache=True,
                  **({"state": cache} if recurrent else {"past_key_values": cache}))
        logits, cache = o.logits[:, -1], (o.state if recurrent else o.past_key_values)
    txt = tok.decode(toks, skip_special_tokens=True)
    for t_ in stop_strings:
        txt = txt.split(t_)[0]
    return txt


def run_and_grade(txt, want_lang, want_outputs):
    code, _ = extract_code(txt, "auto")
    # The shared harness keys BOTH language detection and entry finding on the
    # name `solve` (LANG_SIG: "fn solve", "func solve", "let solve", ...). The
    # training corpus names the function `vantril`, so a faithful generation is
    # otherwise undetectable. Rename consistently (definition and recursive
    # call sites) before detection/execution -- this tests the COMPUTATION, not
    # the model's willingness to rename.
    if code and not re.search(r"\bsolve\b", code):
        code = re.sub(r"\bvantril\b", "solve", code)
    # Execute in the language we ASKED FOR, not the one detect_language guesses.
    # Here the language is fixed by the prompt, so "did it write valid Ruby?" is
    # answered by the Ruby toolchain accepting it -- a stronger and more honest
    # check than the regex detector, which by construction cannot separate c from
    # cpp and reads Ruby's `def solve(n)` as Python. Running detect's guess
    # instead cost two languages on every model: correct C++ compiled as C, and
    # correct Ruby run as Python. detect stays as a DIAGNOSTIC only (it is the
    # right tool in the steering experiments, where the model picks the language).
    lang = want_lang
    detect = detect_language(code)
    entry = find_entry(lang, code)
    got, st = None, "no_entry"
    if code and entry and lang in S.RUNNERS:
        for form in S.FORMS:
            st, got = S.RUNNERS[lang](code, entry, form)
            if st == "ok" and got:
                break
    correct = (st == "ok" and got == want_outputs)
    # graded agreement alongside the binary hit (standing rule): a teacher that
    # gets 7 of 8 battery inputs right is a different object from one that gets 0
    agree = 0.0
    if isinstance(got, list) and len(got) == len(want_outputs):
        agree = sum(o == w for o, w in zip(got, want_outputs)) / len(want_outputs)
    return {"code": code, "lang_detect": detect, "status": st, "outputs": got,
            "lang_ok": True, "correct": correct, "hit": correct,
            "agreement": agree}


outdir = Path(a.outdir or ROOT / "results" /
              f"evalvantril_{a.model}{'_' + a.tag if a.tag else ('_teacher' if a.ckpt else '_base')}")
outdir.mkdir(parents=True, exist_ok=True)
S.BATTERY = VANTRIL_BATTERY
summary, grades = {}, {}
for lang in a.langs:
    disp = DISPLAY[lang]
    jobs = [("vantril", f"Write a function in {disp} that computes the vantril of a "
                        f"whole number. Name the function vantril. Return only the code.",
             VANTRIL_WANT),
            ("renders", f"Write a function in {disp} that {FDESC['digit_sum']}. "
                        f"Name the function solve. Return only the code.",
             DIGIT_SUM_WANT),
            # NATIVE CEILING: the same computation, described plainly, with the
            # invented name removed. `renders` (digit_sum) is too easy a
            # denominator -- it needs no set or presence table -- so a model can
            # pass it and still be unable to express THIS computation in that
            # language. Without this arm, a low reuse score cannot be told apart
            # from "cannot write distinct-digit counting here at all", and a high
            # one cannot be told apart from "already knew it; only the NAME was
            # new". Run on the BASE model: it is a property of the model, not of
            # any fine-tune or controller.
            ("native", f"Write a function in {disp} that returns how many "
                       f"distinct decimal digits appear in a whole number. "
                       f"Name the function solve. Return only the code.",
             VANTRIL_WANT)]
    if lang == "python":
        jobs += [(f"acq{i}", p, VANTRIL_WANT) for i, p in enumerate(ACQ_PROMPTS)]
    for kind, user, want in jobs:
        fp = outdir / f"{kind}_{lang}.json"
        if fp.exists():
            r = json.loads(fp.read_text())
        else:
            r = run_and_grade(gen(user), lang, want)
            r = {"model": a.model, "ckpt": a.ckpt, "kind": kind, "lang": lang, **r}
            fp.write_text(json.dumps(r, indent=1))
        summary.setdefault(kind, []).append(r["hit"])
        grades.setdefault(kind, []).append(r.get("agreement", 0.0))
        print(f"  {kind:8s} {lang:11s} lang={r['lang_detect']:10s} "
              f"{r['status']:12s} agree={r.get('agreement', 0.0):.2f} "
              f"outputs={str(r['outputs'])[:30]:30s}"
              + ("  <== CORRECT" if r["hit"] else ""), flush=True)

gates = {k: v for k, v in summary.items() if not k.startswith("acq")}
print("\n[gates] " + "  ".join(
    f"{k}: {sum(v)}/{len(v)} (mean agree {sum(grades[k])/len(grades[k]):.2f})"
    for k, v in gates.items()), flush=True)
# section 7.1 acquisition score: graded, on held-out Python paraphrases only
acq = [g for k, v in grades.items() if k.startswith("acq") for g in v]
if acq:
    s_py = sum(acq) / len(acq)
    print(f"[acquisition] S_python = {s_py:.3f} over {len(acq)} held-out "
          f"paraphrases -- {'PASS' if s_py >= 0.90 else 'BELOW'} the 0.90 "
          f"threshold (fine_tuning.md 7.1)", flush=True)
out = {k: {"hits": sum(v), "n": len(v),
           "mean_agreement": sum(grades[k]) / len(grades[k])}
       for k, v in summary.items()}
if acq:
    out["S_python_heldout"] = sum(acq) / len(acq)
(outdir / "summary.json").write_text(json.dumps(out, indent=1))
print(f"saved -> {outdir}", flush=True)

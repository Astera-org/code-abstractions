"""Grade a model (base or teacher) on ONE of the five invented computations.

This is eval_vantril.py with the algorithm lifted out into `algos.py`, so that
the reuse result can be read across five computations rather than one.
eval_vantril.py is left in place untouched -- it produced the published
vantril numbers, and `--algo vantril` here reproduces its prompts exactly
(checked by --selftest).

Three arms per (model, language), all reported:
  <algo>     does it produce a CORRECT <algo> in this language?
  renders    can it write a KNOWN function (digit_sum) in this language at all?
  native     can it write THIS COMPUTATION described in plain words, with the
             invented name removed? Run on the BASE model: a low reuse score
             is only about reuse if the base could have written it here.
Plus, in Python only, three held-out acquisition paraphrases (`acq0..2`) --
the section 7.1 selection rule picks the teacher by acquisition, not by dose.

    uv run factored_programming_languages/eval_algo.py --algo thulep \
        --model qwen3.6-27b --ckpt <path> --device cuda:0
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from algos import ALGOS, BATTERY, ACQ, want                               # noqa: E402
from elicit import extract_code, render_prompt                            # noqa: E402
from recipes import RECIPES                                               # noqa: E402
from tasks_function import LANGS, FDESC                                   # noqa: E402
from languages import DISPLAY                                             # noqa: E402
import execute as S                                                       # noqa: E402
import loader                                                             # noqa: E402
from entry import find_entry, detect_language                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--algo", required=True, choices=list(ALGOS))
ap.add_argument("--model", required=True)
ap.add_argument("--ckpt", default=None, help="teacher checkpoint; omit for base")
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--langs", nargs="+", default=list(LANGS))
ap.add_argument("--max-new-tokens", type=int, default=320)
ap.add_argument("--tag", default=None)
ap.add_argument("--outdir", default=None)
ap.add_argument("--selftest", action="store_true",
                help="print the prompts and exit, loading no model")
a = ap.parse_args()

ALGO = a.algo
SPEC = ALGOS[ALGO]
WANT = want(ALGO)
DIGIT_SUM_WANT = [sum(int(c) for c in str(n)) for n in BATTERY]
ACQ_PROMPTS = [p.format(name=ALGO) for p in ACQ]


def jobs_for(lang):
    disp = DISPLAY[lang]
    out = [(ALGO, f"Write a function in {disp} that {SPEC['desc']}. "
                  f"Name the function {ALGO}. Return only the code.", WANT),
           ("renders", f"Write a function in {disp} that {FDESC['digit_sum']}. "
                       f"Name the function solve. Return only the code.",
            DIGIT_SUM_WANT),
           ("native", f"Write a function in {disp} that {SPEC['plain']}. "
                      f"Name the function solve. Return only the code.", WANT)]
    if lang == "python":
        out += [(f"acq{i}", p, WANT) for i, p in enumerate(ACQ_PROMPTS)]
    return out


if a.selftest:
    print(f"battery {BATTERY}\nwant    {WANT}\n")
    for kind, user, w in jobs_for("python") + jobs_for("rust")[:1]:
        print(f"[{kind}] {user}\n   want {w}")
    sys.exit(0)

recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]

from transformers import (AutoTokenizer, AutoModelForCausalLM,            # noqa: E402
                          AutoModelForImageTextToText)
import yaml                                                               # noqa: E402

if a.ckpt:
    src = str(Path(a.ckpt).resolve())
    _trc = ({"trust_remote_code": True} if yaml.safe_load(open(ROOT.parent / "models.yaml"))
            .get(a.model, {}).get("trust_remote_code") else {})                   # custom-code models (RWKV-7)
    tok = AutoTokenizer.from_pretrained(src, **_trc)
    # Same class dispatch as the SFT runner and loader.py -- the multimodal
    # checkpoints (mistral-3.2, qwen3.6, gemma-4) cannot be reloaded through
    # AutoModelForCausalLM, so a teacher gate would fail at load time.
    _cfg = yaml.safe_load(open(ROOT.parent / "models.yaml"))
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
    # training corpus names the function after the algorithm, so a faithful
    # generation is otherwise undetectable. Rename consistently (definition and
    # recursive call sites) before detection/execution -- this tests the
    # COMPUTATION, not the model's willingness to rename.
    if code and not re.search(r"\bsolve\b", code):
        code = re.sub(rf"\b{ALGO}\b", "solve", code)
    # Execute in the language we ASKED FOR, not the one detect_language guesses:
    # the detector cannot separate c from cpp and reads Ruby's `def solve(n)` as
    # Python, which cost two languages on every model. detect stays a DIAGNOSTIC.
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
              f"eval{ALGO}_{a.model}"
              f"{'_' + a.tag if a.tag else ('_teacher' if a.ckpt else '_base')}")
outdir.mkdir(parents=True, exist_ok=True)
S.BATTERY = BATTERY                       # standing rule: rebind before running
summary, grades = {}, {}
for lang in a.langs:
    for kind, user, w in jobs_for(lang):
        fp = outdir / f"{kind}_{lang}.json"
        if fp.exists():
            r = json.loads(fp.read_text())
        else:
            r = run_and_grade(gen(user), lang, w)
            r = {"model": a.model, "algo": ALGO, "ckpt": a.ckpt, "kind": kind,
                 "lang": lang, **r}
            fp.write_text(json.dumps(r, indent=1))
        summary.setdefault(kind, []).append(r["hit"])
        grades.setdefault(kind, []).append(r.get("agreement", 0.0))
        print(f"  {kind:8s} {lang:11s} lang={r['lang_detect']:10s} "
              f"{r['status']:12s} agree={r.get('agreement', 0.0):.2f} "
              f"outputs={str(r['outputs'])[:30]:30s}"
              + ("  <== CORRECT" if r["hit"] else ""), flush=True)

gates = {k: v for k, v in summary.items() if not k.startswith("acq")}
print(f"\n[{ALGO}] " + "  ".join(
    f"{k}: {sum(v)}/{len(v)} (mean agree {sum(grades[k])/len(grades[k]):.2f})"
    for k, v in gates.items()), flush=True)
acq = [g for k, v in grades.items() if k.startswith("acq") for g in v]
if acq:
    s_py = sum(acq) / len(acq)
    print(f"[acquisition] S_python = {s_py:.3f} over {len(acq)} held-out "
          f"paraphrases -- {'PASS' if s_py >= 0.90 else 'BELOW'} the 0.90 "
          f"threshold (fine_tuning.md 7.1)", flush=True)
out = {k: {"hits": sum(v), "n": len(v),
           "mean_agreement": sum(grades[k]) / len(grades[k])}
       for k, v in summary.items()}
out["algo"] = ALGO
if acq:
    out["S_python_heldout"] = sum(acq) / len(acq)
(outdir / "summary.json").write_text(json.dumps(out, indent=1))
print(f"saved -> {outdir}", flush=True)

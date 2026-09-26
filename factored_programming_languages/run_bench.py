"""Run the in-house benchmark (tasks_bench.BENCH): 20 functions x 10 languages,
greedy, reasoning off, execution-scored. pass = all six battery outputs exact.

    uv run factored_programming_languages/run_bench.py --model gemma-4-31B-it --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from elicit import extract_code, render_prompt                            # noqa: E402
from recipes import RECIPES                                               # noqa: E402
from tasks_function import TAIL                                           # noqa: E402
from tasks_bench import BENCH                                             # noqa: E402
from languages import LANGS, DISPLAY                                      # noqa: E402
import execute as S                                                       # noqa: E402
import loader                                                             # noqa: E402
from entry import find_entry, detect_language                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--max-new-tokens", type=int, default=640)
ap.add_argument("--shard", default=None,
                help="i/n -- run only cells with index %% n == i, so N processes "
                     "can share the benchmark on N cards with DISJOINT work. "
                     "The per-cell resume guard makes concurrent copies safe but "
                     "wasteful; this makes them non-overlapping.")
ap.add_argument("--outdir", default=None)
a = ap.parse_args()
recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]

from transformers import AutoTokenizer                                    # noqa: E402
tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]), **recipe.get("tokenizer_kwargs", {}))
_L = loader.load(a.model, device=a.device, max_memory=a.max_memory, attn="sdpa")
model, dev = _L.model, _L.device
gc = model.generation_config
stop = set()
for e in ([gc.eos_token_id] if isinstance(gc.eos_token_id, int) else (gc.eos_token_id or [])):
    stop.add(int(e))
if tok.eos_token_id is not None:
    stop.add(int(tok.eos_token_id))
stop_strings = recipe.get("stop_strings") or []


def generate(text):
    ids = tok(render_prompt(tok, recipe, text), return_tensors="pt",
              add_special_tokens=False).to(dev)["input_ids"]
    with torch.inference_mode():
        step = model(input_ids=ids, use_cache=True)
        cache, logits, out = step.past_key_values, step.logits[:, -1], []
        for _ in range(a.max_new_tokens):
            nxt = int(logits.argmax(-1))
            if nxt in stop:
                break
            out.append(nxt)
            if stop_strings and any(t_ in tok.decode(out[-24:], skip_special_tokens=True)
                                    for t_ in stop_strings):
                break
            o = model(input_ids=torch.tensor([[nxt]], device=dev),
                      past_key_values=cache, use_cache=True)
            logits, cache = o.logits[:, -1], o.past_key_values
    txt = tok.decode(out, skip_special_tokens=True)
    for t_ in stop_strings:
        txt = txt.split(t_)[0]
    return txt


outdir = Path(a.outdir or ROOT / "results" / f"bench_{a.model}")
outdir.mkdir(parents=True, exist_ok=True)
passes = 0
total = 0
SHARD_I, SHARD_N = (None, None)
if a.shard:
    SHARD_I, SHARD_N = (int(x) for x in a.shard.split("/"))
    print(f"[shard] {SHARD_I}/{SHARD_N}", flush=True)
_cell = 0

for name, desc, ref, battery in BENCH:
    want = [ref(x) for x in battery]
    S.BATTERY = battery
    for lang in LANGS:
        _cell += 1
        if SHARD_N is not None and (_cell - 1) % SHARD_N != SHARD_I:
            continue
        fp = outdir / f"b_{name}_{lang}.json"
        if fp.exists():
            r = json.loads(fp.read_text())
        else:
            txt = generate(f"Write a function in {DISPLAY[lang]} that {desc}. {TAIL}")
            code, _ = extract_code(txt, "auto")
            entry = find_entry(lang, code)
            got, st = None, "no_entry"
            if code and entry:
                for form in S.FORMS:
                    st, got = S.RUNNERS[lang](code, entry, form)
                    if st == "ok" and got == want:
                        break
            r = {"model": a.model, "task": name, "lang": lang, "status": st,
                 "outputs": got, "want": want, "pass": got == want and st == "ok",
                 "n_match": (sum(g == w for g, w in zip(got, want))
                             if isinstance(got, list) and len(got) == len(want) else 0),
                 "code": code}
            fp.write_text(json.dumps(r, indent=1))
        passes += r["pass"]
        total += 1
    done = sum(1 for _ in outdir.glob(f"b_{name}_*.json"))
    print(f"  [{name}] cumulative pass {passes}/{total}", flush=True)
print(f"BENCH {a.model}: {passes}/{total} = {passes/total:.3f}", flush=True)

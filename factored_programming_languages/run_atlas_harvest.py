"""Harvest suffix-window KV states for the 42-function atlas, one language per
invocation (shard by language across GPUs). Prefill-only.

Saves results/atlas_<model>/states_<lang>.pt:
  {"meta": ..., "states": {fkey_or_"nofn": {layer: {"keys","values"} fp16}}}
Padding and the aligned-suffix K are computed over the full grid x all shard
languages, so every language's states share one positional frame.

    uv run factored_programming_languages/run_atlas_harvest.py --lang python
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from elicit import render_prompt                                           # noqa: E402
from recipes import RECIPES                                                # noqa: E402
from tasks_atlas import DESC, FKEYS                                        # noqa: E402
from tasks_compose import TAIL, nofn                                       # noqa: E402
from languages import DISPLAY                                              # noqa: E402
import prompt_memory as pm                                                 # noqa: E402
import loader                                                              # noqa: E402

ALL_LANGS = ["python", "javascript", "java", "go", "rust", "haskell"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-31B-it")
ap.add_argument("--lang", required=True, choices=ALL_LANGS)
ap.add_argument("--device", default=None)
ap.add_argument("--max-memory", default=None)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()

recipe = RECIPES[a.model.lower()] if a.model.lower() in RECIPES else RECIPES[a.model]
from transformers import AutoTokenizer                                     # noqa: E402
tok = AutoTokenizer.from_pretrained(recipe.get("tokenizer_name", recipe["hf_name"]),
                                    **recipe.get("tokenizer_kwargs", {}))
ids_of = lambda t: tok(render_prompt(tok, recipe, t),                      # noqa: E731
                       add_special_tokens=False)["input_ids"]

raw = {}
for l in ALL_LANGS:
    raw[("nofn", l)] = nofn(l)
    for f in FKEYS:
        raw[(f, l)] = f"Write a function in {DISPLAY[l]} that {DESC[f]}. {TAIL}"
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


mine = {k: v for k, v in raw.items() if k[1] == a.lang}
padded = {k: _pad_one(u) for k, u in raw.items()}
assert all(padded.values()), \
    f"padding failed: {[k for k, v in padded.items() if not v][:5]}"
seqs = [ids_of(u) for u in padded.values()]
K = 0
while K < target and len({s[-1 - K] for s in seqs}) == 1:
    K += 1
print(f"[grid] {len(raw)} prompts, target={target}, K={K}", flush=True)
assert K >= 10

_L = loader.load(a.model, device=a.device, max_memory=a.max_memory, attn="sdpa")
model, dev, n_layers = _L.model, _L.device, _L.n_layers
enc_of = lambda t: tok(render_prompt(tok, recipe, t), return_tensors="pt",  # noqa: E731
                       add_special_tokens=False).to(dev)["input_ids"]

with torch.inference_mode():
    _probe = model(input_ids=enc_of(padded[("nofn", a.lang)]), use_cache=True)
KINDS = pm.profile(_probe.past_key_values, n_layers)
del _probe
PATCHED = pm.select(KINDS, "kv")

states = {}
keys = sorted(mine, key=str)
for n, key in enumerate(keys):
    with torch.inference_mode():
        c = model(input_ids=enc_of(padded[key]), use_cache=True).past_key_values
    states[key[0]] = {i: pm.harvest(c, i, pm.KV, K) for i in PATCHED}
    if (n + 1) % 10 == 0 or n + 1 == len(keys):
        print(f"  [{n + 1}/{len(keys)}]", flush=True)

outdir = Path(a.outdir or ROOT / "results" / f"atlas_{a.model}")
outdir.mkdir(parents=True, exist_ok=True)
torch.save({"meta": {"model": a.model, "lang": a.lang, "K": K,
                     "target": target, "layers": PATCHED, "fkeys": FKEYS},
            "states": states},
           outdir / f"states_{a.lang}.pt")
(outdir / f"states_{a.lang}.done").touch()
print(f"saved -> {outdir}/states_{a.lang}.pt", flush=True)

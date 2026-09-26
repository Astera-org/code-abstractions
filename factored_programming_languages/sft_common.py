"""Chat-format SFT plumbing for the Python-only fine-tuning experiments (Fig. 4b, Fig. 18),
self-contained in this directory:
  * renders through THIS directory's recipes.py / elicit.py, so one prompt renderer serves
    training and every eval here;
  * the corpus is named explicitly (--corpus): the 9-example PORTABLE realization
    (3 paraphrases x 3 implementations, no Python-only idioms);
  * the model table covers the paper's cast.

Training-example construction (loss on the reply only):
    prefix = chat template of [user], add_generation_prompt=True
    target = lead + assistant_text + eot
    labels[:len(tok(prefix))] = -100
with the token-prefix identity tok(prefix + target)[:len(tok(prefix))] ==
tok(prefix) asserted per example -- never truncate labels.
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from recipes import RECIPES                                              # noqa: E402
from elicit import render_prompt                                         # noqa: E402

# models.yaml key -> SFT config. `ranks`/`per_device` give a global batch >= 9
# so one optimizer step == one epoch over the 9-example corpus.
SFT_MODELS = {
    "pythia-6.9b-tulu": dict(recipe="pythia-6.9b-tulu", style="bare", lead="",
                             eot=None, ranks=2, per_device=6),
    # smoke-test models (not in the paper): one rank, the nine examples in one batch
    "pythia-70m": dict(recipe="pythia-70m", style="bare", lead="", eot=None, ranks=1, per_device=9),
    "pythia-160m": dict(recipe="pythia-160m", style="bare", lead="", eot=None, ranks=1, per_device=9),
    "pythia-410m": dict(recipe="pythia-410m", style="bare", lead="", eot=None, ranks=1, per_device=9),
    "mistral-small-3.2-24b": dict(recipe="mistral-small-3.2-24b", style="fence",
                                  lead="", eot="</s>", ranks=6, per_device=2),
    "qwen3-32b": dict(recipe="qwen3-32b", style="fence", lead="",
                      eot="<|im_end|>", ranks=8, per_device=2),
    "qwen3.6-27b": dict(recipe="qwen3.6-27b", style="fence", lead="",
                        eot="<|im_end|>", ranks=8, per_device=2),
    "gemma-4-31B-it": dict(recipe="gemma-4-31b-it", style="fence", lead="",
                           eot="<turn|>", ranks=8, per_device=2),
    # same gemma-4 chat template and end-of-turn token as the 31B, verified
    # by rendering both; MoE, so it trains through the same zero3 path
    "gemma-4-26b-a4b": dict(recipe="gemma-4-26b-a4b", style="fence", lead="",
                            eot="<turn|>", ranks=8, per_device=2),
    # attention-free RWKV-7: its chat template ends every turn with a blank line, so that is the end-of-turn the
    # model must learn to emit (stop string "\n\nUser:" at eval). 13.3B dense -> 4 ranks x 3 = 12 >= 9 (full batch).
    "rwkv7-g1j-13.3b": dict(recipe="rwkv7-g1j-13.3b", style="fence", lead="",
                            eot="\n\n", ranks=4, per_device=3),
}


def corpus_file(name, style):
    """data/<name>_<style>.json -- copied from the archive's chatv2 set"""
    return SCRIPT_DIR / "data" / f"{name}_{style}.json"


def encode_chat(tok, model_key, user, assistant, max_len=448):
    spec = SFT_MODELS[model_key]
    recipe = RECIPES[spec["recipe"]]
    prefix = render_prompt(tok, recipe, user)
    eot = spec["eot"] if spec["eot"] is not None else tok.eos_token
    full = prefix + spec["lead"] + assistant + eot
    pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if ids[:len(pre_ids)] != pre_ids:
        raise ValueError(f"token-prefix mismatch for {model_key}")
    if len(ids) > max_len:
        raise ValueError(f"example exceeds max_len ({len(ids)} > {max_len}) "
                         f"for {model_key} -- raise --max-len, never truncate")
    labels = [-100] * len(pre_ids) + ids[len(pre_ids):]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def load_examples(tok, model_key, corpus, repeats=1, seed=0, max_len=448):
    style = SFT_MODELS[model_key]["style"]
    path = corpus_file(corpus, style)
    docs = json.loads(path.read_text())
    examples = [(d["user"], d["assistant"]) for d in docs] * repeats
    import random
    random.Random(seed).shuffle(examples)
    return ([encode_chat(tok, model_key, u, a, max_len=max_len) for u, a in examples],
            path.name)

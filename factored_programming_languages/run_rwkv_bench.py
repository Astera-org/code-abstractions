"""The in-house benchmark (tasks_bench.BENCH, 31 functions x 10 languages) on RWKV-7, for the coding-ability axis
of paper_figures/fig5_scatter.py (authors, 2026-09-14: "can we put this RWKV in our coding performance vs. severed
performance scatterplot?").

Identical prompts, battery and grading to run_bench.py -- the only differences are the load path (remote code, a
transformers 5.15 overlay) and the decode loop (a recurrent state instead of a KV cache). Writes the same
`results/bench_<model>/b_<task>_<lang>.json` records the figure reads.

    uv run --with "transformers==5.15.1" factored_programming_languages/run_rwkv_bench.py --device cuda:0 --langs java
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
from elicit import extract_code                                            # noqa: E402
from languages import DISPLAY, LANGS                                       # noqa: E402
from tasks_bench import BENCH                                              # noqa: E402
from tasks_function import TAIL                                            # noqa: E402
from entry import find_entry                                            # noqa: E402

MODEL_ID = "RWKV/RWKV7-G1j-13.3B-20260831"
MODEL_KEY = "rwkv7-g1j-13.3b"

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--langs", nargs="+", default=list(LANGS))
ap.add_argument("--max-new-tokens", type=int, default=320)
ap.add_argument("--outdir", default=None)
a = ap.parse_args()
outdir = Path(a.outdir or ROOT / "results" / f"bench_{MODEL_KEY}")
outdir.mkdir(parents=True, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer                # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
                                             device_map=a.device)
model.eval()
model.config.wkv_implementation = "eager"
dev = next(model.parameters()).device
EOS = {0} | ({int(tok.eos_token_id)} if tok.eos_token_id is not None else set())
print(f"[load] {MODEL_ID} in {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def generate(text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True, thinking=False)
    ids = tok(s, return_tensors="pt", add_special_tokens=False).to(dev)["input_ids"]
    out = model(input_ids=ids, use_cache=True)
    logits, state, toks = out.logits[:, -1], out.state, []
    for _ in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in EOS:
            break
        toks.append(nxt)
        o = model(input_ids=torch.tensor([[nxt]], device=dev), state=state, use_cache=True)
        logits, state = o.logits[:, -1], o.state
    return tok.decode(toks, skip_special_tokens=True)


passes = total = 0
for name, desc, ref, battery in BENCH:
    want = [ref(x) for x in battery]
    S.BATTERY = battery
    for lang in a.langs:
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
            r = {"model": MODEL_KEY, "task": name, "lang": lang, "status": st, "outputs": got, "want": want,
                 "pass": got == want and st == "ok",
                 "n_match": (sum(g == w for g, w in zip(got, want))
                             if isinstance(got, list) and len(got) == len(want) else 0),
                 "code": code}
            fp.write_text(json.dumps(r, indent=1))
        passes += r["pass"]
        total += 1
    print(f"  [{name}] cumulative pass {passes}/{total}", flush=True)
print(f"BENCH {MODEL_KEY} ({','.join(a.langs)}): {passes}/{total} = {passes / max(total, 1):.3f}", flush=True)

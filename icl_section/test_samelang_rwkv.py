"""Same-language tests on RWKV-7 (attention-free): (a) example VISIBLE, ask for the same language -> copy? (b) the recurrent
state after the L-example prompt installed into the neutral prompt, ask for L -> the donor text?  4 functions x 10 languages.
    uv run --with "transformers==5.15.1" icl_section/test_samelang_rwkv.py --device cuda:4"""
import argparse, difflib, json, re, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "factored_programming_languages"))   # shared modules (algos, recipes, elicit, code_style, languages, tasks_function, entry)
import execute as S
from entry import find_entry
from stage_common import ASK
from tasks_function import LANGS
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:4"); ap.add_argument("--funcs", nargs="+", default=["thulep", "vantril", "lufemun", "vupovuth"]); ap.add_argument("--max-new-tokens", type=int, default=320)
a = ap.parse_args(); DON = json.loads((HERE / "donors.json").read_text()); MODEL_ID = "RWKV/RWKV7-G1j-13.3B-20260831"
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map=a.device).eval()
model.config.wkv_implementation = "eager"; dev = next(model.parameters()).device; n_layers = model.config.num_hidden_layers
EOS = {0} | ({int(tok.eos_token_id)} if tok.eos_token_id is not None else set())
def ids_of(text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, thinking=False)
    return tok(s, return_tensors="pt", add_special_tokens=False).to(dev)["input_ids"]
def prompt(L, code): return (f"Here is a function:\n```{L}\n{code}\n```\n" if code is not None else "") + ASK
@torch.no_grad()
def prefill(text): return model(input_ids=ids_of(text), use_cache=True).state
def harvest(st): return [tuple(t.detach().clone() for t in st.read(i)) for i in range(n_layers)]
def install(st, snap):
    for i, parts in enumerate(snap): st.write(i, *[p.clone() for p in parts])
    return st
@torch.no_grad()
def decode_from(st, L):
    fence = tok(f"```{L}\n", return_tensors="pt", add_special_tokens=False).to(dev)["input_ids"]
    o = model(input_ids=fence, state=st, use_cache=True); logits, st, toks = o.logits[:, -1], o.state, []
    for _ in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in EOS: break
        toks.append(nxt); o = model(input_ids=torch.tensor([[nxt]], device=dev), state=st, use_cache=True); logits, st = o.logits[:, -1], o.state
    return f"```{L}\n" + tok.decode(toks, skip_special_tokens=True)
from grading import first_code
def grade(txt, f, L):
    S.BATTERY = DON[f]["battery"]; code, _ = first_code(txt); entry = find_entry(L, code) if code else None; st, got = "no_entry", None
    if code and entry and L in S.RUNNERS:
        for form in S.FORMS:
            st, got = S.RUNNERS[L](code, entry, form)
            if st == "ok" and got: break
    return code or "", st == "ok" and got == DON[f]["want"]
norm = lambda c: re.sub(r"\s+", " ", (c or "").strip()); OUT = HERE / "results" / "samelang_rwkv7.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {}   # merge: new functions join the old cells
for f in a.funcs:
    for L in LANGS:
        donor = DON[f]["impls"][L][0]
        vis = decode_from(prefill(prompt(L, donor)), L)                                            # example visible
        ins = decode_from(install(prefill(prompt(None, None)), harvest(prefill(prompt(L, donor)))), L)   # state installed into the neutral prompt
        for arm, txt in (("visible", vis), ("install", ins)):
            code, hit = grade(txt, f, L); res[f"{f}|{L}|{arm}"] = {"hit": hit, "identical": norm(code) == norm(donor), "similarity": difflib.SequenceMatcher(None, norm(code), norm(donor)).ratio(), "code": code[:500]}
        r1, r2 = res[f"{f}|{L}|visible"], res[f"{f}|{L}|install"]
        print(f"  {f:<9}{L:<11} visible: hit={r1['hit']!s:<5} sim={r1['similarity']:.2f} | install: hit={r2['hit']!s:<5} sim={r2['similarity']:.2f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
for arm in ("visible", "install"):
    rr = [v for k, v in res.items() if k.endswith("|" + arm)]; print(f"{arm}: identical {sum(v['identical'] for v in rr)}/{len(rr)}, mean similarity {sum(v['similarity'] for v in rr)/len(rr):.2f}, hits {sum(v['hit'] for v in rr)}/{len(rr)}")

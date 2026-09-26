"""Stages A and B on RWKV7-G1j-13.3B (attention-free; needs `uv run --with "transformers==5.15.1"`).
A: for each (function, donor, target): demo + the name-free language-free ask, then the answer prefilled with the
   target's fence. There is no severing arm: the recurrent state after the ask IS the compiled suffix, so `full` is
   the compiled-state condition. Floor: ask alone.
B: the recurrent state after the donor prompt (demo + ask), averaged over admissible donors (Stage A hits at that
   target), installed as the state of the neutral prompt (ask alone); then the fence and decode. Arms: single donors,
   pooled_all, pooled_py, wrong, none. (No norm-matched noise: the state is a tuple of matrices; add later if needed.)
    uv run --with "transformers==5.15.1" icl_section/run_stage_rwkv.py --stage A --device cuda:0 --shard 0/4 --out results/stageA_rwkv.shard0.json
    uv run --with "transformers==5.15.1" icl_section/run_stage_rwkv.py --stage B --funcs f1 f2 f3 f4 --device cuda:0 --out results/stageB_rwkv.json
"""
import argparse, json, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "factored_programming_languages"))   # shared modules (algos, recipes, elicit, code_style, languages, tasks_function, entry)
from stage_common import ASK, DON, competitors, admissible               # noqa: E402  (no model loaded by the import)
from grading import first_code                                          # noqa: E402
from entry import find_entry                                             # noqa: E402
import execute as S                                                      # noqa: E402
from tasks_function import LANGS                                         # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer            # noqa: E402

MODEL_ID = "RWKV/RWKV7-G1j-13.3B-20260831"
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["A", "B"], required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--funcs", nargs="+", default=None); ap.add_argument("--langs", nargs="+", default=LANGS)
ap.add_argument("--stageA", default=str(HERE / "results" / "stageA_rwkv.json"))
ap.add_argument("--shard", default=None); ap.add_argument("--max-new-tokens", type=int, default=320); ap.add_argument("--out", required=True)
a = ap.parse_args(); FUNCS = a.funcs or list(DON)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map=a.device).eval()
model.config.wkv_implementation = "eager"; dev = next(model.parameters()).device; n_layers = model.config.num_hidden_layers
EOS = {0} | ({int(tok.eos_token_id)} if tok.eos_token_id is not None else set())


def ids_of(text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, thinking=False)
    return tok(s, return_tensors="pt", add_special_tokens=False).to(dev)["input_ids"]


def prompt(demo_lang, code):
    return (f"Here is a function:\n```{demo_lang}\n{code}\n```\n" if code is not None else "") + ASK


@torch.no_grad()
def prefill(text):
    out = model(input_ids=ids_of(text), use_cache=True); return out.state


def harvest(state):
    return [tuple(t.detach().clone() for t in state.read(i)) for i in range(n_layers)]


def install(state, snap):
    for i, parts in enumerate(snap): state.write(i, *[p.clone() for p in parts])
    return state


@torch.no_grad()
def decode_from(state, L):
    fence = tok(f"```{L}\n", return_tensors="pt", add_special_tokens=False).to(dev)["input_ids"]
    o = model(input_ids=fence, state=state, use_cache=True); logits, state, toks = o.logits[:, -1], o.state, []
    for _ in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in EOS: break
        toks.append(nxt)
        o = model(input_ids=torch.tensor([[nxt]], device=dev), state=state, use_cache=True); logits, state = o.logits[:, -1], o.state
    return f"```{L}\n" + tok.decode(toks, skip_special_tokens=True)


def grade(txt, f, L):
    battery = DON[f]["battery"]; S.BATTERY = battery; comp = competitors(battery)
    code, multi = first_code(txt); entry = find_entry(L, code) if code else None; st, got = "no_entry", None
    if code and entry and L in S.RUNNERS:
        for form in S.FORMS:
            st, got = S.RUNNERS[L](code, entry, form)
            if st == "ok" and got: break
    which = next((g for g, w in comp.items() if st == "ok" and got == w), None)
    return {"status": st, "built": st == "ok", "hit": st == "ok" and got == DON[f]["want"], "which": which, "outputs": got, "code": code, "multi_block": multi, "text": txt[:800]}


def donor_list(f):
    """10-demo-language design: one program per language (donors.json)"""
    return [(x, 0) for x in a.langs if x in DON[f]["impls"]]


out_path = Path(a.out); res = json.loads(out_path.read_text()) if out_path.exists() else {}
SK, SN = (int(x) for x in a.shard.split("/")) if a.shard else (0, 1); t0 = time.time(); n = 0; idx = -1
if a.stage == "A":
    for f in FUNCS:
        for L in a.langs:
            idx += 1
            if idx % SN != SK: continue
            k = f"{f}|none|{L}|floor"
            if k not in res:
                r = grade(decode_from(prefill(prompt(None, None)), L), f, L); res[k] = r; n += 1
                print(f"  {f:<9}{'floor':<10}{L:<11}hit={r['hit']!s:<5} which={r['which']}", flush=True)
            for x, v in donor_list(f):
                if x == L: continue                                     # same-language prompts are left out
                k = f"{f}|{x}|{L}|full"
                if k in res: continue
                r = grade(decode_from(prefill(prompt(x, DON[f]["impls"][x][v])), L), f, L); res[k] = r; n += 1
                print(f"  {f:<9}{x:<11}{L:<11}full   hit={r['hit']!s:<5} which={str(r['which']):<10} ({(time.time() - t0) / n:.1f}s/cell)", flush=True)
            out_path.write_text(json.dumps(res, indent=1))
else:
    A = json.loads(Path(a.stageA).read_text())
    def adm(f, L): return [(d.rstrip("0123456789"), int(d[len(d.rstrip("0123456789")):] or 0)) for k, v in A.items() for d in [k.split("|")[1]]
                           if k.split("|")[0] == f and k.split("|")[2] == L and k.split("|")[3] == "full" and v["hit"]]
    SNAP = {}
    def snap(f, x, v):
        if (f, x, v) not in SNAP: SNAP[(f, x, v)] = harvest(prefill(prompt(x, DON[f]["impls"][x][v])))
        return SNAP[(f, x, v)]
    def mean_snap(keys):
        snaps = [snap(*k) for k in keys]
        return [tuple(torch.stack([s[i][j] for s in snaps]).mean(0) for j in range(len(snaps[0][i]))) for i in range(n_layers)]
    for f in FUNCS:
        for L in a.langs:
            idx += 1
            if idx % SN != SK: continue
            ad = adm(f, L)
            if not ad: res[f"{f}|not_addressable|{L}"] = {}; print(f"  {f:<9}{L:<11}not addressable", flush=True); continue
            arms = {"none": None, "pooled": mean_snap([(f, x, v) for x, v in ad])}
            neu = harvest(prefill(prompt(None, None))); gen = torch.Generator().manual_seed(0)      # norm-matched noise control
            arms["rand"] = [tuple(n_ + (z := torch.randn(n_.shape, generator=gen).to(n_.device, n_.dtype)) * ((p_ - n_).norm() / (z.norm() + 1e-9))
                                  for n_, p_ in zip(neu[i], arms["pooled"][i])) for i in range(n_layers)]
            for x, v in ad: arms[x] = snap(f, x, v)
            others = [g for g in FUNCS if g != f and adm(g, L)]
            if others:
                g = others[0]; arms["wrong"] = mean_snap([(g, x, v) for x, v in adm(g, L)]); wrong_g = g
            for arm, sn in arms.items():
                k = f"{f}|{arm}|{L}"
                if k in res: continue
                st = prefill(prompt(None, None))
                if sn is not None: st = install(st, sn)
                r = grade(decode_from(st, L), f, L); r["admissible"] = [x for x, _ in ad]
                if arm == "wrong": r["expect"] = wrong_g
                res[k] = r; n += 1
                print(f"  {f:<9}{arm:<13}{L:<11}hit={r['hit']!s:<5} which={str(r['which']):<10} ({(time.time() - t0) / n:.1f}s/cell)", flush=True)
            out_path.write_text(json.dumps(res, indent=1))
out_path.write_text(json.dumps(res, indent=1)); print(f"\n[saved] {out_path} ({len(res)} cells)")

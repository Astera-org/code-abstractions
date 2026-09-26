"""Stage A (10-demo-language design: every prompt = demo language x, target L, x != L; one verified program per language) — compile map. For every (function, donor implementation, target language): does the compiled suffix render
the function in situ once the demo is hidden?

Prompt: "Here is a function:" + demo (donor language x, variant v) + the language-free, name-free ask
    "Write the function above. It takes a whole number and returns an integer. Use the same algorithm.
     Name the function solve. Return only the code."
then the answer is prefilled with ```<target>. Arms: full (nothing masked) / carry (fence + generated rows blind to
the demo). Floor: the ask alone (no demo), once per (function, target). Grading: execution on the function's battery,
competitor detection (invented + nonce + bank functions).
    uv run icl_section/run_stageA.py --device cuda:0 --shard 0/8 --out icl_section/results/stageA.shard0.json
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch, yaml
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "factored_programming_languages"))   # shared modules (algos, recipes, elicit, code_style, languages, tasks_function, entry)
from algos import ALGOS                                                  # noqa: E402
from elicit import render_prompt                                         # noqa: E402
from grading import first_code                                          # noqa: E402
from recipes import RECIPES                                              # noqa: E402
from tasks_function import LANGS                                         # noqa: E402
from entry import find_entry                                             # noqa: E402
import execute as S                                                      # noqa: E402
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM, DynamicCache   # noqa: E402
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask  # noqa: E402

ASK = ("Write the function above. It takes a whole number and returns an integer. Use the same algorithm. "
       "Name the function solve. Return only the code.")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-31B-it"); ap.add_argument("--device", default=None)
ap.add_argument("--donors", default=str(HERE / "donors.json"))
ap.add_argument("--funcs", nargs="+", default=None); ap.add_argument("--langs", nargs="+", default=LANGS)
ap.add_argument("--max-variants", type=int, default=2, help="java/go variants per function (python: all)")
ap.add_argument("--shard", default=None); ap.add_argument("--max-new-tokens", type=int, default=320)
ap.add_argument("--skip-from", default=None, help="results file whose keys count as done")
ap.add_argument("--out", default=str(HERE / "results" / "stageA.json"))
a = ap.parse_args()
DON = json.loads(Path(a.donors).read_text()); FUNCS = a.funcs or list(DON)
NON = json.loads((HERE / "nonce_functions.json").read_text())
BANK = {"digit_sum": lambda n: sum(int(c) for c in str(n)), "digit_prod": lambda n: math.prod(int(c) for c in str(n)),
        "even_digits": lambda n: sum(1 for c in str(n) if int(c) % 2 == 0), "sq_digit_sum": lambda n: sum(int(c) ** 2 for c in str(n))}


def competitors(battery):
    c = {g: [ALGOS[g]["ref"](n) for n in battery] for g in ALGOS}
    c.update({k: [g(n) for n in battery] for k, g in BANK.items()})
    for fn in NON["functions"]:
        if list(NON["battery"]) == list(battery): c[fn["name"]] = fn["want"]
    return c


cfg = yaml.safe_load(open(HERE.parent / "models.yaml"))[a.model]; hf = cfg["hf_name"]; recipe = RECIPES[a.model.lower()]
tok = AutoTokenizer.from_pretrained(hf)
CLS = AutoModelForImageTextToText if cfg.get("model_class") == "image_text_to_text" else AutoModelForCausalLM
model = CLS.from_pretrained(hf, dtype=torch.bfloat16, device_map=({"": a.device} if a.device else "auto"), attn_implementation="eager").eval()
dev = next(model.parameters()).device; tcfg = model.config.get_text_config()
gc = model.generation_config
stop = {int(e) for e in ([gc.eos_token_id] if isinstance(gc.eos_token_id, int) else (gc.eos_token_id or []))}
if tok.eos_token_id is not None: stop.add(int(tok.eos_token_id))
SENT = "QQSENTINELQQ"; PREFIX, SUFFIX = render_prompt(tok, recipe, SENT).split(SENT)
ids_of = lambda s: tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()   # noqa: E731
print(f"[model] {hf}; {len(FUNCS)} functions; ask {len(ids_of(ASK))} tokens", flush=True)


def build(demo_lang, code, L):
    pieces = [(0, PREFIX)]
    if code is not None: pieces.append((1, f"Here is a function:\n```{demo_lang}\n{code}\n```\n"))
    pieces += [(3, ASK + SUFFIX), (4, f"```{L}\n")]
    ids, bid = [], []
    for b, t in pieces:
        x = ids_of(t); ids += x; bid += [b] * len(x)
    return torch.tensor([ids], device=dev), torch.tensor(bid, device=dev)


HYBRID = "linear_attention" in tcfg.layer_types                 # Qwen3.5/3.6: recurrent layers ignore attention masks


@torch.inference_mode()
def gen(ids, bid, masked):
    Lp = ids.shape[1]
    if masked and HYBRID:                         # two passes: prefill up to the ask unmasked, then the fence rows under a 2D mask
        nf = int((bid == 4).sum()); pre = ids[:, :Lp - nf]  # that hides the demo from the attention layers (the recurrent
        cache = model(input_ids=pre, use_cache=True).past_key_values      # channel cannot be blinded: it is the compiled state)
        amask = torch.ones(1, Lp, dtype=torch.long, device=dev); amask[0, bid == 1] = 0
        for j in range(Lp - nf, Lp):                                   # fence tokens ONE AT A TIME: the DeltaNet layers only continue
            cp = torch.arange(j, j + 1, device=dev)                      # from their cached state for single-token steps
            out = model(input_ids=ids[:, j:j + 1], past_key_values=cache, use_cache=True, attention_mask=amask[:, :j + 1], cache_position=cp, position_ids=cp.unsqueeze(0))
            cache = out.past_key_values
    elif masked:                                  # fence rows blind to the demo at prefill; generated rows blind at decode
        emb = model.get_input_embeddings()(ids); pos = torch.arange(Lp, device=dev).unsqueeze(0); cache = DynamicCache(config=tcfg)
        def allowed(bi, hi, q, kv): return ~((bid[q] == 4) & (bid[kv] == 1))
        kw = dict(config=tcfg, inputs_embeds=emb, attention_mask=None, past_key_values=cache, position_ids=pos, and_mask_function=allowed)
        m = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
        out = model(input_ids=ids, attention_mask=m, past_key_values=cache, use_cache=True)
        amask = torch.ones(1, Lp, dtype=torch.long, device=dev); amask[0, bid == 1] = 0
    else:
        out = model(input_ids=ids, use_cache=True); amask = None
    logits, cache, toks = out.logits[:, -1], out.past_key_values, []; Lc = Lp
    for _ in range(a.max_new_tokens):
        nxt = int(logits.argmax(-1))
        if nxt in stop: break
        toks.append(nxt)
        if amask is not None: amask = torch.cat([amask, torch.ones(1, 1, dtype=amask.dtype, device=dev)], 1)
        pk = ({"cache_position": (cp := torch.arange(Lc, Lc + 1, device=dev)), "position_ids": cp.unsqueeze(0)} if HYBRID else {})
        step = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True, attention_mask=amask, **pk)
        logits, cache = step.logits[:, -1], step.past_key_values; Lc += 1
    return tok.decode(toks, skip_special_tokens=True)


def grade(txt, f, L):
    battery = DON[f]["battery"]; S.BATTERY = battery; comp = competitors(battery)
    code, multi = first_code(txt); entry = find_entry(L, code) if code else None; st, got = "no_entry", None
    if code and entry and L in S.RUNNERS:
        for form in S.FORMS:
            st, got = S.RUNNERS[L](code, entry, form)
            if st == "ok" and got: break
    which = next((g for g in comp if st == "ok" and got == w) for w in [None]) if False else next((g for g, w in comp.items() if st == "ok" and got == w), None)
    return {"status": st, "built": st == "ok", "hit": st == "ok" and got == DON[f]["want"], "which": which, "outputs": got,
            "code": code, "multi_block": multi, "text": txt[:800]}


def donor_list(f):
    """10-demo-language design (2026-09-23): one program per language (donors.json, built by build_donors10.py)"""
    return [(x, 0) for x in a.langs if x in DON[f]["impls"]]


out_path = Path(a.out); out_path.parent.mkdir(parents=True, exist_ok=True)
res = json.loads(out_path.read_text()) if out_path.exists() else {}
DONE = set(json.loads(Path(a.skip_from).read_text())) if a.skip_from else set()
SK, SN = (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)
t0 = time.time(); n = 0; idx = -1
for f in FUNCS:
    for L in a.langs:
        idx += 1
        if idx % SN != SK: continue
        key = f"{f}|none|{L}|floor"
        if key not in res and key not in DONE:
            ids, bid = build(None, None, L); r = grade(f"```{L}\n" + gen(ids, bid, False), f, L); res[key] = r; n += 1
            print(f"  {f:<9}{'floor':<10}{L:<11}hit={r['hit']!s:<5} which={r['which']}", flush=True)
        for x, v in donor_list(f):
            if x == L: continue                                         # same-language prompts are left out
            for arm in ("full", "carry"):
                key = f"{f}|{x}|{L}|{arm}"
                if key in res or key in DONE: continue
                ids, bid = build(x, DON[f]["impls"][x][v], L)
                r = grade(f"```{L}\n" + gen(ids, bid, arm == "carry"), f, L); res[key] = r; n += 1
                print(f"  {f:<9}{x:<11}{L:<11}{arm:<6} hit={r['hit']!s:<5} which={str(r['which']):<10} ({(time.time() - t0) / n:.1f}s/cell)", flush=True)
        out_path.write_text(json.dumps(res, indent=1))
out_path.write_text(json.dumps(res, indent=1)); print(f"\n[saved] {out_path} ({len(res)} cells)")

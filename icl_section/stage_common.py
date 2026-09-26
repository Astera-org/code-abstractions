"""Shared machinery for Stages B/C: model, the name-free language-free ask, harvest of the compiled suffix (last K rows),
re-rotation to the recipient's positions, fence-driven generation, grading, and the Stage-A gate."""
import json, math, sys
from pathlib import Path
import torch, yaml
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "factored_programming_languages"))   # shared modules (algos, recipes, elicit, code_style, languages, tasks_function, entry)
from algos import ALGOS                                                  # noqa: E402
from elicit import render_prompt                                         # noqa: E402
from grading import first_code                                          # noqa: E402
from recipes import RECIPES                                              # noqa: E402
from entry import find_entry                                             # noqa: E402
from rerotate import rerotate                                            # noqa: E402
import execute as S                                                      # noqa: E402
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM, DynamicCache   # noqa: E402
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask  # noqa: E402

ASK = ("Write the function above. It takes a whole number and returns an integer. Use the same algorithm. "
       "Name the function solve. Return only the code.")
DON = json.loads((HERE / "donors.json").read_text())
NON = json.loads((HERE / "nonce_functions.json").read_text())
BANK = {"digit_sum": lambda n: sum(int(c) for c in str(n)), "digit_prod": lambda n: math.prod(int(c) for c in str(n)),
        "even_digits": lambda n: sum(1 for c in str(n) if int(c) % 2 == 0), "sq_digit_sum": lambda n: sum(int(c) ** 2 for c in str(n))}


def competitors(battery):
    c = {g: [ALGOS[g]["ref"](n) for n in battery] for g in ALGOS}
    c.update({k: [g(n) for n in battery] for k, g in BANK.items()})
    for fn in NON["functions"]:
        if list(NON["battery"]) == list(battery): c[fn["name"]] = fn["want"]
    return c


class Rig:
    def __init__(self, model_key, device, max_new=320):
        cfg = yaml.safe_load(open(HERE.parent / "models.yaml"))[model_key]; self.hf = cfg["hf_name"]; self.recipe = RECIPES[model_key.lower()]
        self.tok = AutoTokenizer.from_pretrained(self.hf)
        CLS = AutoModelForImageTextToText if cfg.get("model_class") == "image_text_to_text" else AutoModelForCausalLM
        self.model = CLS.from_pretrained(self.hf, dtype=torch.bfloat16, device_map=({"": device} if device else "auto"), attn_implementation="eager").eval()
        self.dev = next(self.model.parameters()).device; self.tcfg = self.model.config.get_text_config(); self.LT = self.tcfg.layer_types
        self.n_layers = len(self.LT); self.max_new = max_new
        # hybrid (Qwen3.5/3.6): linear-attention layers keep a recurrent state + conv buffer instead of K/V rows.
        # The compiled state then has two channels; both are harvested/installed unless `channels` restricts them.
        self.hybrid = "linear_attention" in self.LT
        self.names = {i: (("conv_states", "recurrent_states") if self.LT[i] == "linear_attention" else ("keys", "values")) for i in range(self.n_layers)}
        gc = self.model.generation_config
        self.stop = {int(e) for e in ([gc.eos_token_id] if isinstance(gc.eos_token_id, int) else (gc.eos_token_id or []))}
        if self.tok.eos_token_id is not None: self.stop.add(int(self.tok.eos_token_id))
        SENT = "QQSENTINELQQ"; self.PREFIX, self.SUFFIX = render_prompt(self.tok, self.recipe, SENT).split(SENT)
        self.K = len(self.ids_of(ASK + self.SUFFIX))            # the compiled suffix: identical tokens in every prompt
        self._store = {}

    def ids_of(self, s):
        return self.tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

    def demo_block(self, lang, code):
        return f"Here is a function:\n```{lang}\n{code}\n```\n"

    def donor_ids(self, f, x, v):
        return self.ids_of(self.PREFIX) + self.ids_of(self.demo_block(x, DON[f]["impls"][x][v])) + self.ids_of(ASK + self.SUFFIX)

    def neutral_ids(self):
        return self.ids_of(self.PREFIX) + self.ids_of(ASK + self.SUFFIX)

    def rot_keys(self, K, delta, i):
        """keys written at positions p -> as if written at p + delta (RoPE is a rotation; compose by delta)"""
        if delta == 0: return K
        if not self.hybrid: return rerotate(self.model, K, delta, self.LT[i])
        rot = self.model.model.language_model.rotary_emb                  # Qwen3.5: partial rotary, interleaved M-RoPE (text: 3 equal axes)
        cos, sin = rot(K, torch.tensor([[delta]], device=K.device)); cos, sin = cos[0, 0].float(), sin[0, 0].float(); d = cos.shape[-1]
        kr = K[..., :d]; half = d // 2; rh = torch.cat((-kr[..., half:], kr[..., :half]), -1)
        return torch.cat((kr * cos + rh * sin, K[..., d:]), -1)

    def harvest(self, c, n_rows):
        """compiled state at the end of a prompt: last n_rows K/V of attention layers; recurrent state + conv buffer of linear layers"""
        out = {}
        for i in range(self.n_layers):
            if self.LT[i] == "linear_attention":
                out[i] = {"conv_states": c.layers[i].conv_states[0].float().clone(), "recurrent_states": c.layers[i].recurrent_states[0].float().clone()}
            else:
                out[i] = {"keys": c.layers[i].keys[0, :, -n_rows:].float().clone(), "values": c.layers[i].values[0, :, -n_rows:].float().clone()}
        return out

    def write(self, c, Lr, suffix, channels=("kv", "rec")):
        """write a compiled state into cache c (prompt length Lr): rows for attention layers, state for linear layers"""
        for i in range(self.n_layers):
            if self.LT[i] == "linear_attention":
                if "rec" in channels:
                    for n in ("conv_states", "recurrent_states"): getattr(c.layers[i], n)[0].copy_(suffix[i][n].to(getattr(c.layers[i], n).dtype))
            elif "kv" in channels:
                for n in ("keys", "values"):
                    t = getattr(c.layers[i], n); t[:, :, Lr - self.K:, :] = suffix[i][n].to(t.dtype)

    @torch.inference_mode()
    def suffix_kv(self, f, x, v, L_rec):
        """compiled suffix of donor (f, x, v): last K rows' K/V (keys re-rotated to a recipient of length L_rec); on a
        hybrid also the recurrent state + conv buffer at the end of the ask (no positions to correct)"""
        key = (f, x, v)
        if key not in self._store:
            ids = self.donor_ids(f, x, v)
            c = self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=True).past_key_values
            self._store[key] = (self.harvest(c, self.K), len(ids))
        kv, Ld = self._store[key]
        return {i: ({"keys": self.rot_keys(kv[i]["keys"], L_rec - Ld, i), "values": kv[i]["values"]} if "keys" in kv[i] else kv[i]) for i in range(self.n_layers)}

    def seq_len(self, c):
        """tokens in cache c, read from an ATTENTION layer (a hybrid cache's layer 0 may be a linear layer with no rows:
        DynamicCache.get_seq_length then returns 0 and every decode step lands at position 0 -> garbage; found 23:40)"""
        return next(c.layers[i].keys.shape[-2] for i in range(self.n_layers) if self.LT[i] != "linear_attention")

    def pos_kw(self, Lp, n):
        """explicit cache_position / position_ids for the next n tokens after Lp; only needed (and only used) on hybrids"""
        if not self.hybrid: return {}
        cp = torch.arange(Lp, Lp + n, device=self.dev); return {"cache_position": cp, "position_ids": cp.unsqueeze(0)}

    def feed(self, c, toks, Lp, amask=None):
        """continue cache c with tokens `toks` (positions Lp..); returns (last logits, cache). On a hybrid the tokens go in
        ONE AT A TIME: Qwen3.5's GatedDeltaNet only continues from its cached recurrent state when seq_len == 1 — a
        multi-token cached step restarts the recurrence from nothing and overwrites the state (found 2026-09-23 23:50)."""
        chunks = [[t] for t in toks] if self.hybrid else [list(toks)]
        for ch in chunks:
            kw = {"attention_mask": amask[:, :Lp + len(ch)]} if amask is not None else {}
            o = self.model(input_ids=torch.tensor([ch], device=self.dev), past_key_values=c, use_cache=True, **kw, **self.pos_kw(Lp, len(ch)))
            c = o.past_key_values; Lp += len(ch)
        return o.logits[:, -1], c

    @torch.inference_mode()
    def generate_from_cache(self, c, L):
        fence = self.ids_of(f"```{L}\n"); Lp = self.seq_len(c)
        logits, c = self.feed(c, fence, Lp); toks = []; Lp += len(fence)
        for _ in range(self.max_new):
            nxt = int(logits.argmax(-1))
            if nxt in self.stop: break
            toks.append(nxt)
            o = self.model(input_ids=torch.tensor([[nxt]], device=self.dev), past_key_values=c, use_cache=True, **self.pos_kw(Lp, 1))
            logits, c = o.logits[:, -1], o.past_key_values; Lp += 1
        return f"```{L}\n" + self.tok.decode(toks, skip_special_tokens=True)

    @torch.inference_mode()
    def install_and_generate(self, suffix, L, channels=("kv", "rec")):
        """neutral prompt (no demo); its compiled state <- suffix (both channels unless restricted); then the fence and decode"""
        ids = self.neutral_ids(); Lr = len(ids)
        c = self.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=True).past_key_values
        if suffix is not None: self.write(c, Lr, suffix, channels)
        return self.generate_from_cache(c, L)

    @torch.inference_mode()
    def swap_and_generate(self, f_A, xA, vA, suffix_B, L, channels=("kv", "rec")):
        """recipient read f_A's demo; its compiled state <- suffix_B (or kept); fence + generation blind to the demo
        (attention channel; a hybrid's recurrent channel cannot be blinded and is replaced instead)"""
        pre = self.ids_of(self.PREFIX); demo = self.ids_of(self.demo_block(xA, DON[f_A]["impls"][xA][vA])); ask = self.ids_of(ASK + self.SUFFIX)
        ids = torch.tensor([pre + demo + ask], device=self.dev); Lr = ids.shape[1]
        bid = torch.tensor([0] * len(pre) + [1] * len(demo) + [3] * len(ask), device=self.dev)
        c = self.model(input_ids=ids, use_cache=True).past_key_values
        if suffix_B is not None: self.write(c, Lr, suffix_B, channels)
        amask = torch.ones(1, Lr, dtype=torch.long, device=self.dev); amask[0, bid == 1] = 0
        fence = self.ids_of(f"```{L}\n"); Lp = Lr
        amask = torch.cat([amask, torch.ones(1, len(fence), dtype=torch.long, device=self.dev)], 1)
        logits, c = self.feed(c, fence, Lp, amask); toks = []; Lp += len(fence)
        for _ in range(self.max_new):
            nxt = int(logits.argmax(-1))
            if nxt in self.stop: break
            toks.append(nxt)
            amask = torch.cat([amask, torch.ones(1, 1, dtype=torch.long, device=self.dev)], 1)
            o = self.model(input_ids=torch.tensor([[nxt]], device=self.dev), past_key_values=c, use_cache=True, attention_mask=amask, **self.pos_kw(Lp, 1))
            logits, c = o.logits[:, -1], o.past_key_values; Lp += 1
        return f"```{L}\n" + self.tok.decode(toks, skip_special_tokens=True)

    def grade(self, txt, f, L):
        battery = DON[f]["battery"]; S.BATTERY = battery; comp = competitors(battery)
        code, multi = first_code(txt); entry = find_entry(L, code) if code else None; st, got = "no_entry", None
        if code and entry and L in S.RUNNERS:
            for form in S.FORMS:
                st, got = S.RUNNERS[L](code, entry, form)
                if st == "ok" and got: break
        which = next((g for g, w in comp.items() if st == "ok" and got == w), None)
        return {"status": st, "built": st == "ok", "hit": st == "ok" and got == DON[f]["want"], "which": which, "outputs": got,
                "code": code, "multi_block": multi, "text": txt[:800]}


def admissible(stageA, f, L):
    """donors (x, v) whose compiled suffix rendered f in situ at target L (Stage A carry hit). Donor tags are a
    language ("java": one program per language, the 10-demo-language design) or language+index ("java1", the
    3-demo-language table); x == L never occurs in the 10-language design (same-language prompts are left out)."""
    out = []
    for k, v in stageA.items():
        ff, d, LL, arm = k.split("|")
        if ff == f and LL == L and arm == "carry" and v["hit"]:
            x = d.rstrip("0123456789"); out.append((x, int(d[len(x):] or 0)))
    return sorted(out)

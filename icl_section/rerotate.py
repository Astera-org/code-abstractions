"""Position-free KV patching: re-rotate a donor's keys from their absolute positions to the recipient's.

RoPE rotates each (x_i, x_{i+d/2}) pair of a key by an angle proportional to its position, so a key written at
position p can be moved to position p' by one further rotation by (p' - p); values carry no position. With this,
donor prompts need no left-padding to share the suffix's absolute rows: harvest each at its natural length, align
the suffix rows from the end, re-rotate, then average and patch.

Gemma-4 has two rotary parameterisations (full_attention: theta 1e6, proportional/partial; sliding_attention:
theta 1e4, default), so the rotation is taken from the model's own rotary module per layer type, unscaled.

    uv run icl_section/rerotate.py --selftest
"""
import argparse, sys
from pathlib import Path
import torch, yaml
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "factored_programming_languages"))   # shared modules
from transformers import AutoTokenizer, AutoModelForImageTextToText                      # noqa: E402
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb                 # noqa: E402


def rotation(model, delta, layer_type, device):
    """unscaled (cos, sin) of the model's rotary embedding at position `delta`, shape [d_rot], on `device`"""
    rot = model.model.language_model.rotary_emb
    inv_freq = getattr(rot, f"{layer_type}_inv_freq").float().to(device)
    freqs = inv_freq * float(delta)
    emb = torch.cat((freqs, freqs), dim=-1)                                 # the model's own layout: cat(freqs, freqs)
    return emb.cos(), emb.sin()


def rerotate(model, K, delta, layer_type):
    """K: [heads, rows, head_dim] keys written at positions p; returns the same keys as if written at p + delta.
    Only the first d_rot dims are rotated (partial rotary on full-attention layers); the rest pass through."""
    if delta == 0:
        return K
    cos, sin = rotation(model, delta, layer_type, K.device)
    d = cos.shape[-1]; Kf = K.float(); x = Kf[..., :d]; rest = Kf[..., d:]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot_half = torch.cat((-x2, x1), dim=-1)
    out = torch.cat((x * cos + rot_half * sin, rest), dim=-1)
    return out.to(K.dtype)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--model", default="gemma-4-31B-it")
    a = ap.parse_args()
    from recipes import RECIPES                                              # noqa: E402
    from elicit import render_prompt                                         # noqa: E402
    cfg = yaml.safe_load(open(HERE.parent / "models.yaml"))[a.model]; hf = cfg["hf_name"]; recipe = RECIPES[a.model.lower()]
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForImageTextToText.from_pretrained(hf, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager").eval()
    dev = next(model.parameters()).device; tcfg = model.config.get_text_config(); LT = tcfg.layer_types
    text = "Here is a function called vantril in Python:\n```python\ndef vantril(n):\n    return len(set(str(n)))\n```\nWrite vantril in Java. Name the function solve. Return only the code."
    ids = tok(render_prompt(tok, recipe, text), return_tensors="pt", add_special_tokens=False)["input_ids"].to(dev)
    L = ids.shape[1]; SHIFT = 37
    with torch.inference_mode():
        c0 = model(input_ids=ids, use_cache=True).past_key_values
        c1 = model(input_ids=ids, position_ids=torch.arange(SHIFT, SHIFT + L, device=dev).unsqueeze(0), use_cache=True).past_key_values
        print(f"[selftest] {L} tokens; same prompt encoded at positions 0.. and {SHIFT}..")
        worst = {"values": 0.0, "keys_raw": 0.0, "keys_rerotated": 0.0}
        per_type = {}
        for i in range(len(LT)):
            K0, V0 = c0.layers[i].keys[0].float(), c0.layers[i].values[0].float()
            K1, V1 = c1.layers[i].keys[0].float(), c1.layers[i].values[0].float()
            dv = (V1 - V0).abs().max().item(); dk = (K1 - K0).abs().max().item()
            Kr = rerotate(model, c1.layers[i].keys[0], -SHIFT, LT[i]).float()
            dr = (Kr - K0).abs().max().item()
            worst["values"] = max(worst["values"], dv); worst["keys_raw"] = max(worst["keys_raw"], dk); worst["keys_rerotated"] = max(worst["keys_rerotated"], dr)
            per_type.setdefault(LT[i], []).append(dr)
        scale = max(c0.layers[i].keys.abs().max().item() for i in range(len(LT)))
        print(f"[selftest] max |V(shifted) - V|          = {worst['values']:.4f}   (must be ~0: values carry no position)")
        print(f"[selftest] max |K(shifted) - K| raw      = {worst['keys_raw']:.4f}   (large: keys are rotated)")
        print(f"[selftest] max |rerotate(K(shifted), -{SHIFT}) - K| = {worst['keys_rerotated']:.4f}   (must be ~bf16 ulp; |K| max {scale:.1f}, ulp ~{scale / 256:.3f})")
        for lt, v in per_type.items():
            print(f"[selftest]   {lt:<18} worst layer {max(v):.4f}")
        # layer 0 has no accumulated bf16 noise: its residual is the pure rotation-math check; deeper layers show
        # how absolute-position rounding noise in q/k propagates into hidden states (values) and hence keys
        print("[selftest] per layer: |dV| max, |V| max, |K rerotated - K| max, |K| max")
        for i in [0, 1, 2, 5, 6, 12, 30, 59]:
            K0, V0 = c0.layers[i].keys[0].float(), c0.layers[i].values[0].float()
            K1, V1 = c1.layers[i].keys[0].float(), c1.layers[i].values[0].float()
            Kr = rerotate(model, c1.layers[i].keys[0], -SHIFT, LT[i]).float()
            print(f"[selftest]   L{i:<3}{LT[i][:7]:<9} dV {(V1 - V0).abs().max().item():8.4f}  |V| {V0.abs().max().item():7.2f}   "
                  f"dK_rerot {(Kr - K0).abs().max().item():8.4f}  |K| {K0.abs().max().item():6.2f}   raw dK {(K1 - K0).abs().max().item():7.3f}")

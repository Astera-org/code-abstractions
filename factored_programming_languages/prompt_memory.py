"""Prompt memory: whatever a layer carries forward from the prompt into generation.

The transplant intervention is architecture-general. What differs between models is only
*what object* the prompt leaves behind for generation to attend to:

  kv         a full-attention layer leaves per-position keys and values.  The memory is
             POSITION-INDEXED, so a contrast averaged across prompts is only meaningful
             if those prompts put the harvested positions at the same absolute indices
             (keys are cached post-RoPE).  Hence the equal-length prompt padding.
             -- gemma-4 (shapes differ between layers), Mistral-4, pythia, and Qwen's
                16 full-attention layers.

  recurrent  a linear-attention layer leaves a fixed-size recurrent state plus a short
             causal-conv window.  The recurrent state is a summary of the WHOLE prompt
             and carries no position index at all, so neither the padding nor the RoPE
             phase constraint applies to it.
             -- Qwen3.6's 48 linear layers.

Both are just tensors the model reads at every generated token, so the same contrast
arithmetic applies to both:

    delta = E_lambda [ M(A, lambda) - M(B, lambda) ]      M' = M(B, t) + alpha * delta

`select` is what makes the Qwen dissociation possible: run the identical contrast into
the KV layers only, the recurrent layers only, or both, and ask which memory carries the
function factor.
"""
import torch

KV, RECURRENT = "kv", "recurrent"


def layer_kind(layer):
    """what this cache layer carries; None if it carries nothing yet"""
    if getattr(layer, "keys", None) is not None and getattr(layer, "keys").numel():
        return KV
    if (getattr(layer, "recurrent_states", None) is not None
            or getattr(layer, "conv_states", None) is not None):
        return RECURRENT
    return None


def profile(cache, n_layers):
    """[kind per layer] -- the model's memory architecture, read off a real cache"""
    return [layer_kind(cache.layers[i]) for i in range(n_layers)]


def describe(kinds):
    n_kv = sum(k == KV for k in kinds)
    n_rc = sum(k == RECURRENT for k in kinds)
    idx = [i for i, k in enumerate(kinds) if k == KV]
    s = f"{len(kinds)} layers: {n_kv} kv, {n_rc} recurrent"
    if n_rc and n_kv:
        s += f"  (kv at {idx[:4]}{'...' if len(idx) > 4 else ''})"
    return s


def harvest(cache, i, kind, k):
    """the memory layer i carries. For kv, only the last k positions."""
    L = cache.layers[i]
    if kind == KV:
        return {"keys": L.keys[0, :, -k:, :].detach().half().cpu(),
                "values": L.values[0, :, -k:, :].detach().half().cpu()}
    out = {}
    for name, attr in (("conv", "conv_states"), ("recurrent", "recurrent_states")):
        t = getattr(L, attr, None)
        if t is None:
            continue
        # transformers 5.15.x stores linear-attention state as a dict of
        # tensors on some models (qwen3.5 hybrid); flatten with dotted names
        if isinstance(t, dict):
            for sk, sv in t.items():
                if sv is not None:
                    out[f"{name}.{sk}"] = sv.detach().half().cpu()
        else:
            out[name] = t.detach().half().cpu()
    return out


def contrast(store_a, store_b, langs_a, langs_b):
    """E_lambda[ M(A, a_lambda) - M(B, b_lambda) ], elementwise, per tensor.

    langs_a and langs_b are zipped, so passing the same list gives the matched
    cross-language average and passing a permutation gives a mismatched one. (A permuted
    control is vacuous for a difference of means -- it equals the matched contrast -- so
    do not use one; see FINDINGS_FUNCTION_TRANSPLANT.md.)
    """
    keys = store_a[langs_a[0]].keys()
    return {t: torch.stack([store_a[p][t].float() - store_b[q][t].float()
                            for p, q in zip(langs_a, langs_b)]).mean(0)
            for t in keys}


def patch(cache, i, kind, delta, alpha, lo=None, hi=None, s0=None, s1=None,
          part="kv"):
    """in-place  M += alpha * delta.  lo:hi are cache positions (kv only).

    `part` restricts a KV patch to the keys or the values alone ("k" / "v" /
    "kv"). Keys and values are separate tensors, so the split is exact: a
    k-only patch changes WHERE attention looks, a v-only patch changes WHAT it
    retrieves from unchanged addresses. Ignored for recurrent layers, which
    have no such decomposition.
    """
    L = cache.layers[i]
    if kind == KV:
        want = {"k": ("keys",), "v": ("values",), "kv": ("keys", "values")}[part]
        for name, attr in (("keys", "keys"), ("values", "values")):
            if attr not in want:
                continue
            t = getattr(L, attr)
            d = delta[name].to(t.device)
            t[:, :, lo:hi, :] = (t[:, :, lo:hi, :].float()
                                 + alpha * d[:, s0:s1, :]).to(t.dtype)
        return
    for name, attr in (("conv", "conv_states"), ("recurrent", "recurrent_states")):
        t = getattr(L, attr, None)
        if t is None:
            continue
        if isinstance(t, dict):
            for sk, sv in t.items():
                dk = f"{name}.{sk}"
                if sv is None or dk not in delta:
                    continue
                d = delta[dk].to(sv.device)
                sv.copy_((sv.float() + alpha * d).to(sv.dtype))
        elif name in delta:
            d = delta[name].to(t.device)
            t.copy_((t.float() + alpha * d).to(t.dtype))


def select(kinds, which):
    """which layers receive the transplant: 'kv', 'recurrent', or 'both'"""
    if which == "both":
        return [i for i, k in enumerate(kinds) if k is not None]
    return [i for i, k in enumerate(kinds) if k == which]

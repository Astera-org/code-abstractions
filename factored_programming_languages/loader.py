"""One model loader for the whole directory, and the one place architecture quirks live.

Every run script in the archive repeated the same fifteen lines of `from_pretrained` +
`layer_path` walk, and each copy drifted: some honoured `model_class`, some did not, one
never used `layer_path` at all. This is that block, once, plus the per-family fixes the
cast needs.

Known quirks handled here:

  mistral-small-4-119b  transformers 5.5.0 cannot load this fp8 MoE checkpoint as
                        shipped. integrations/finegrained_fp8.py:599 does
                            getattr(config, "num_local_experts", config.num_experts)
                        and Python evaluates that default EAGERLY, so the
                        `config.num_experts` fallback raises AttributeError even though
                        Mistral4Config's attribute_map already aliases
                        num_local_experts -> n_routed_experts. Setting the alias on the
                        text config before loading sidesteps it without patching
                        site-packages.

  fp8 MoE + static activation scheme: transformers' fast grouped_mm experts dispatch
                        raises NotImplementedError for activation_scheme="static". The
                        two ways out are not equivalent -- switching the scheme to
                        "dynamic" would abandon the calibrated activation scales the
                        checkpoint shipped with, changing numerics -- so we force the
                        EAGER experts dispatch instead and keep the static scales.

  fp8 checkpoints need the `kernels` package to run in fp8 (transformers looks up
                        `w8a8_fp8_matmul` from the hub kernels). It is not installed here
                        and adding it would disturb the pinned venv, so when it is absent
                        we DEQUANTIZE on load instead: the weights materialise as bf16 and
                        no fp8 kernel is ever called. Costs memory (119B -> ~238 GB, so
                        shard it) and buys not touching the environment.
"""
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import models as cast                                                    # noqa: E402


class Loaded:
    def __init__(self, key, model, tok, layers, cfg):
        self.key, self.model, self.tok, self.layers, self.cfg = key, model, tok, layers, cfg
        self.n_layers = len(layers)
        self.d = model.get_input_embeddings().weight.shape[1]
        self.device = next(model.parameters()).device
        self.spec = cast.get(key)

    def __repr__(self):
        return (f"Loaded({self.key}, {self.n_layers} layers, d={self.d}, "
                f"{self.device})")


def _have_fp8_kernels():
    try:
        import kernels  # noqa: F401
        return True
    except ImportError:
        return False


def _fix_config(key, config):
    """per-family config repairs, applied before the weights are touched"""
    tc = getattr(config, "text_config", config)
    # see the module docstring: eager getattr default in the fp8 MoE integration
    if not hasattr(tc, "num_experts") and hasattr(tc, "n_routed_experts"):
        tc.num_experts = tc.n_routed_experts
    return config


def _maybe_dequantize(config, dequantize):
    """returns True if the load should materialise bf16 instead of running fp8"""
    q = getattr(config, "quantization_config", None)
    if q is None:
        return False
    if dequantize is None:                       # decide for ourselves
        dequantize = not _have_fp8_kernels()
    if not dequantize:
        return False
    if isinstance(q, dict):
        q["dequantize"] = True
    else:
        q.dequantize = True
    config.quantization_config = q
    return True


def load(key, device=None, max_memory=None, dtype=None, attn="sdpa",
         dequantize=None):
    cfg = yaml.safe_load(open(ROOT.parent / "models.yaml"))[key]
    trc = {"trust_remote_code": True} if cfg.get("trust_remote_code") else {}   # custom-code models (RWKV-7)
    from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                              AutoModelForImageTextToText)
    config = _fix_config(key, AutoConfig.from_pretrained(cfg["hf_name"], **trc))
    deq = _maybe_dequantize(config, dequantize)
    if dtype is None:
        dtype = "bfloat16" if (deq or not getattr(config, "quantization_config", None)) \
            else "auto"
    attn = cfg.get("attn", attn)                    # models.yaml may pin one (RWKV-7: eager only)
    kw = {"config": config, "attn_implementation": attn,
          "dtype": "auto" if dtype == "auto" else getattr(torch, dtype)}
    if max_memory:
        kw["device_map"] = "auto"
        kw["max_memory"] = ({int(k): v for k, v in
                             (p.split(":") for p in max_memory.split(","))}
                            if isinstance(max_memory, str) else max_memory)
    else:
        kw["device_map"] = {"": torch.device(device or "cuda:0")}

    q = getattr(config, "quantization_config", None)
    scheme = (q.get("activation_scheme") if isinstance(q, dict)
              else getattr(q, "activation_scheme", None)) if q else None
    if scheme == "static" and not deq:
        # keep the checkpoint's calibrated static scales; the fast grouped_mm path
        # refuses them, so take the slower eager experts dispatch
        kw["experts_implementation"] = "eager"

    cls_ = (AutoModelForImageTextToText
            if cfg.get("model_class") == "image_text_to_text" else AutoModelForCausalLM)
    tok = AutoTokenizer.from_pretrained(cfg["hf_name"], **trc)
    model = cls_.from_pretrained(cfg["hf_name"], **kw, **trc).eval()

    layers = model
    for part in cfg["layer_path"].split("."):
        layers = getattr(layers, part)
    return Loaded(key, model, tok, layers, cfg)

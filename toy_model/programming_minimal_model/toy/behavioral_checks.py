"""Preliminary behavioral checks (EXPERIMENT_GUIDE section 4) for a trained programming-toy model.

Everything is read off model outputs; no probing of internal beliefs.

  4.1  Has it learned the task (= the prompted ceiling). Held-out sequences, cut after >= 2
       firings at a point where the pattern memory is empty, then force a firing by appending the
       designated pattern, rendered in the sequence's own language or as shared tokens. Read the
       next-token distribution. Exhibits: argmax over response symbols is the true function's.
       Runs: negligible mass on other languages' tokens, no collapse onto the blank. Also the KL
       from the exact Bayes next-token distribution to the model's, and a no-firing control whose
       Bayes answer is the uniform response distribution.

  4.2  Compiler-like state (the draft's first test). Same cut, then a run of K blanks. Prefill
       the context + blanks and cache K/V at every layer; ablate the cached K/V at every pre-blank
       position except BOS (zeroed, and separately replaced by norm-matched random vectors, and as a
       third condition masked: queries after the blanks may not attend to the context at all); then
       force a firing from the blanks and read the response. If the function (and, for a
       shared-rendered pattern, the language) is still produced, the model carried the compiled
       state into the blanks. Reported against context length, per condition, next to the intact
       prompt and a no-context floor.

  Optional (--generate N): the draft-style behavioral readout. Sample continuations after the
       prefix, decode, and score them (tokens stay in the language's set; responses at firings
       give the target function under the exact Bayes posterior).

Run from the repo root, on CPU by default (the model is tiny; JAX is pinned to CPU here):

  RUN_ID=<mlflow run id> uv run python programming_minimal_model/toy/behavioral_checks.py \
      --tracking-uri file:./mlruns --steps last --n-seqs 512 --out results/programming_toy

Outputs (gitignored): <out>/<run_name>/step_<step>/{cells_4_1.jsonl, cells_4_2.jsonl,
summary.json, figure.png} and a printed summary table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _p in (str(_REPO / "training"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import matrices  # noqa: F401 -- registers wrapped_mess3
import numpy as np
import torch
import toy_process as tp
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from utils import expand_state_by_batch_size

OmegaConf.register_new_resolver("mult", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("div", lambda x, y: x // y, replace=True)


# --- loading a run ---------------------------------------------------------------------------

def _force_device(node, device: str) -> None:
    """Set every `device` key under `node` (the saved config has them resolved to literals)."""
    if isinstance(node, DictConfig):
        for k in list(node.keys()):
            if k == "device":
                node[k] = device
            else:
                _force_device(node[k], device)
    elif isinstance(node, ListConfig):
        for item in node:
            _force_device(item, device)


def load_run(run_id: str, tracking_uri: str, registry_uri: str | None, torch_device: str):
    """Return (cfg, run, persister, process, make_model); make_model(step) loads that checkpoint."""
    import hydra
    import mlflow
    from genproc.persistence.mlflow_persister import MLFlowPersister
    from genproc.utils.config_utils import typed_instantiate

    registry_uri = registry_uri or tracking_uri
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    persister = MLFlowPersister(experiment_id=run.info.experiment_id, run_id=run_id,
                                tracking_uri=tracking_uri, registry_uri=registry_uri)
    with tempfile.TemporaryDirectory() as td:
        path = persister.client.download_artifacts(run_id, "config.yaml", dst_path=td)
        cfg = OmegaConf.load(path)

    # The process always runs on CPU JAX; the model goes where asked.
    with open_dict(cfg):
        _force_device(cfg.generative_process.instance, "cpu")
        cfg.device = torch_device
        OmegaConf.update(cfg, "predictive_model.instance.cfg.device", torch_device, force_add=True)
        OmegaConf.update(cfg, "predictive_model.instance.cfg.n_devices", 1, force_add=True)
    process = hydra.utils.instantiate(cfg.generative_process.instance)

    def make_model(step: int):
        instance = OmegaConf.select(cfg, "predictive_model.instance", throw_on_missing=True)
        model = typed_instantiate(instance, str(instance._target_))
        model = persister.load_weights(model, step)
        model.eval()
        return model

    return cfg, run, persister, process, make_model


def list_checkpoints(persister) -> list[int]:
    steps = []
    for a in persister.client.list_artifacts(persister.run_id, persister.model_dir):
        name = a.path.split("/")[-1]
        if name.isdigit():
            steps.append(int(name))
    return sorted(steps)


# --- data ---------------------------------------------------------------------------------------

def sample_sequences(process, n: int, seq_len: int, seed: int):
    """Held-out sequences with their true (language, function)."""
    init = expand_state_by_batch_size(process.initial_state, n)
    keys = jax.random.split(jax.random.key(seed), n)
    final, toks = process.generate(init, keys, seq_len, False)
    comp = np.argmax(np.asarray(final.component_beliefs), axis=1)
    return np.asarray(toks, dtype=np.int64), comp // tp.N_FUNC, comp % tp.N_FUNC


def make_bayes_next(process):
    """Exact Bayes next-token distribution after a batch of equal-length prefixes: [B, L] -> [B, 16]."""

    def one(tokens):
        def step(state, x):
            return process.transition_states(state, x), None
        final, _ = jax.lax.scan(step, process.initial_state, tokens)
        return process.observation_probability_distribution(final)

    fn = jax.jit(jax.vmap(one))

    def call(tokens_np: np.ndarray) -> np.ndarray:
        return np.asarray(fn(jnp.asarray(tokens_np, dtype=jnp.int32)))

    return call


# --- model calls --------------------------------------------------------------------------------

def kv_hook_names(model) -> list[str]:
    names = []
    for layer in range(model.cfg.n_layers):
        names += [f"blocks.{layer}.attn.hook_k", f"blocks.{layer}.attn.hook_v"]
    return names


@torch.no_grad()
def last_probs(model, tokens: torch.Tensor, fwd_hooks=None) -> np.ndarray:
    """Softmax over the vocabulary at the final position: [B, d_vocab]."""
    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks) if fwd_hooks else model(tokens)
    return torch.softmax(logits[:, -1, :].float(), dim=-1).cpu().numpy()


@torch.no_grad()
def cache_kv(model, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
    names = set(kv_hook_names(model))
    _, cache = model.run_with_cache(prefix, names_filter=lambda n: n in names)
    return {n: cache[n] for n in names}


def ablate_cache(cache: dict[str, torch.Tensor], positions: np.ndarray, mode: str, gen: torch.Generator):
    """Copy of the cache with the given positions zeroed, replaced by norm-matched noise, or kept."""
    out = {}
    pos = torch.as_tensor(positions, dtype=torch.long)
    for name, val in cache.items():
        mod = val.clone()
        if mode == "zero":
            mod[:, pos] = 0.0
        elif mode == "random":
            sub = mod[:, pos]
            noise = torch.randn(sub.shape, generator=gen).to(sub.device, sub.dtype)
            noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12) * sub.norm(dim=-1, keepdim=True)
            mod[:, pos] = noise
        elif mode != "intact":
            raise ValueError(mode)
        out[name] = mod
    return out


def mask_hooks(model, prefix_len: int, positions: np.ndarray):
    """Hooks that forbid queries at positions >= prefix_len from attending to `positions`.

    Applied to the pre-softmax scores after the causal mask, so the query can only read BOS, the
    blanks, and the appended pattern: the cleanest form of "is the state in the blanks?"."""
    pos = torch.as_tensor(positions, dtype=torch.long)

    def hook(scores, hook, P=prefix_len, pos=pos):
        scores = scores.clone()
        scores[:, :, P:, pos] = -1e9
        return scores

    return [(f"blocks.{layer}.attn.hook_attn_scores", hook) for layer in range(model.cfg.n_layers)]


def cache_hooks(cache: dict[str, torch.Tensor], prefix_len: int):
    """Hooks that overwrite K/V at positions < prefix_len with the (possibly ablated) cache.

    This reproduces KV-cache semantics in a stateless HookedTransformer: everything the prompt
    wrote during prefill stays as it was, except where we ablate it, and later positions read it.
    """
    hooks = []
    for name, mod in cache.items():
        def hook(act, hook, mod=mod, prefix_len=prefix_len):
            act = act.clone()
            act[:, :prefix_len] = mod.to(act.device)
            return act
        hooks.append((name, hook))
    return hooks


@torch.no_grad()
def sample_continuations(model, prefix: torch.Tensor, n_new: int, gen: torch.Generator,
                         fwd_hooks=None) -> np.ndarray:
    """Autoregressive sampling (temperature 1) of n_new tokens after each prefix row: [B, n_new]."""
    toks = prefix.clone()
    for _ in range(n_new):
        logits = model.run_with_hooks(toks, fwd_hooks=fwd_hooks) if fwd_hooks else model(toks)
        probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
        nxt = torch.multinomial(probs.cpu(), 1, generator=gen).to(toks.device)
        toks = torch.cat([toks, nxt], dim=1)
    return toks[:, prefix.shape[1]:].cpu().numpy()


# --- cells -------------------------------------------------------------------------------------

def choose_cut(tokens_row, params: tp.ToyParams, target_T: int, slack: int, min_firings: int):
    """Largest T' in [target_T - slack, target_T] with empty memory and >= min_firings firings
    before it, or None."""
    fires = np.cumsum(tp.firing_mask(tokens_row, params))
    for T in range(target_T, max(target_T - slack, 1) - 1, -1):
        mem = tp.memory_trajectory(tokens_row[:T], params)[-1]
        if mem == 0 and fires[T - 1] >= min_firings:
            return T
    return None


def build_cells(tokens, langs, funcs, params, cuts, slack, min_firings):
    cells = []
    for i in range(tokens.shape[0]):
        for target in cuts:
            T = choose_cut(tokens[i], params, target, slack, min_firings)
            if T is None:
                continue
            n_fire = int(tp.firing_mask(tokens[i, :T], params).sum())
            cells.append({"seq": i, "target_T": int(target), "T": int(T), "lang": int(langs[i]),
                          "f": int(funcs[i]), "n_firings": n_fire})
    return cells


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def group_by_T(cells):
    by_T = defaultdict(list)
    for c in cells:
        by_T[c["T"]].append(c)
    return sorted(by_T.items())


SUMMARY_FIELDS = ("score", "exhibits", "runs", "target_mass", "bayes_target_mass", "other_lang_mass",
                  "blank_mass", "kl_bayes_model", "n_firings", "floor_exhibits",
                  "gen_score", "gen_runs", "gen_exhibits", "gen_fired")


def summarize(records, keys):
    """Mean of numeric fields grouped by `keys`."""
    groups = defaultdict(list)
    for r in records:
        groups[tuple(r[k] for k in keys)].append(r)
    out = {}
    for g, rs in groups.items():
        agg = {"n": len(rs)}
        for field in SUMMARY_FIELDS:
            vals = [r[field] for r in rs if r.get(field) is not None]
            if vals:
                agg[field] = float(np.mean(vals))
        out[g] = agg
    return out


def unit_record(cell, p_model, p_bayes, **extra):
    r = dict(cell)
    r.update(tp.read_unit_test(p_model, cell["f"], cell["lang"]))
    r["bayes_target_mass"] = float(tp.symbol_mass(p_bayes)[cell["f"]])
    r["bayes_symbol_mass"] = tp.symbol_mass(p_bayes).tolist()
    r["kl_bayes_model"] = tp.kl(p_bayes, p_model[:tp.N_TOKENS])
    r["model_probs"] = [round(float(v), 6) for v in p_model]
    r.update(extra)
    return r


def generation_record(cell, row, params, **extra):
    s = tp.score_generation(row, cell["lang"], cell["f"], params, m0=0)
    r = dict(cell)
    r.update({"gen_score": s["score"], "gen_runs": s["runs"], "gen_exhibits": s["exhibits"],
              "gen_fired": s["fired"], "gen_n_firings": s["n_firings"], "gen_tokens": [int(t) for t in row]})
    r.update(extra)
    return r


# --- the two checks -----------------------------------------------------------------------------

def run_4_1(model, tokens, langs, funcs, params, args, bos, device, bayes_next, gen):
    """Has it learned / prompted ceiling, plus the no-firing control."""
    records = []
    pattern = list(params.pattern)
    n_ctx = model.cfg.n_ctx
    cells = build_cells(tokens, langs, funcs, params, args.cuts, args.slack, min_firings=2)
    controls = [c for c in build_cells(tokens, langs, funcs, params, args.cuts, args.slack, min_firings=0)
                if c["n_firings"] == 0]
    print(f"  4.1: {len(cells)} cells with >=2 firings, {len(controls)} no-firing controls")

    for kind, cs in (("ceiling", cells), ("no_firing_control", controls)):
        for T, group in group_by_T(cs):
            for sub in batched(group, args.batch):
                prefixes = [[bos] + tokens[c["seq"], :T].tolist() for c in sub]
                for render in ("lang", "shared"):
                    prompts = [pre + tp.render(pattern, c["lang"], render) for pre, c in zip(prefixes, sub)]
                    p_model = last_probs(model, torch.as_tensor(prompts, dtype=torch.long, device=device))
                    p_bayes = bayes_next(np.asarray([p[1:] for p in prompts]))     # Bayes sees no BOS
                    records += [unit_record(c, pm, pb, kind=kind, render=render)
                                for c, pm, pb in zip(sub, p_model, p_bayes)]
                if args.generate and kind == "ceiling" and T + 1 + args.gen_len <= n_ctx:
                    prefix = torch.as_tensor(prefixes, dtype=torch.long, device=device)
                    for _ in range(args.generate):
                        cont = sample_continuations(model, prefix, args.gen_len, gen)
                        records += [generation_record(c, row, params, kind="ceiling_generation", render="none")
                                    for c, row in zip(sub, cont)]
    return records


def run_4_2(model, tokens, langs, funcs, params, args, bos, device, bayes_next, gen):
    """Compiler-like state: ablate pre-blank K/V after prefill, force a firing from the blanks."""
    records = []
    pattern = list(params.pattern)
    n_ctx = model.cfg.n_ctx
    conditions = ("intact", "zero", "random", "mask")
    cells = build_cells(tokens, langs, funcs, params, args.cuts, args.slack, min_firings=2)
    print(f"  4.2: {len(cells)} cells x {len(args.n_blanks)} blank runs x {len(conditions)} conditions x 2 renders")

    for K in args.n_blanks:
        # no-context floor: BOS + blanks + pattern and nothing else; one distribution per (render, language)
        floor = {}
        for render in ("lang", "shared"):
            x = torch.as_tensor([[bos] + [tp.BLANK] * K + tp.render(pattern, lang, render) for lang in range(tp.N_LANG)],
                                dtype=torch.long, device=device)
            floor[render] = last_probs(model, x)
        for T, group in group_by_T(cells):
            P = 1 + T + K
            if P + len(pattern) > n_ctx:
                continue
            ablate_pos = np.arange(0 if args.ablate_bos else 1, 1 + T)   # everything before the blanks
            for sub in batched(group, args.batch):
                prefix = torch.as_tensor([[bos] + tokens[c["seq"], :T].tolist() + [tp.BLANK] * K for c in sub],
                                         dtype=torch.long, device=device)
                cache = cache_kv(model, prefix)
                hooks = {cond: cache_hooks(ablate_cache(cache, ablate_pos, "intact" if cond == "mask" else cond, gen), P)
                         for cond in conditions}
                hooks["mask"] = hooks["mask"] + mask_hooks(model, P, ablate_pos)
                for render in ("lang", "shared"):
                    tail = torch.as_tensor([tp.render(pattern, c["lang"], render) for c in sub],
                                           dtype=torch.long, device=device)
                    full = torch.cat([prefix, tail], dim=1)
                    p_bayes = bayes_next(full[:, 1:].cpu().numpy())
                    for cond in conditions:
                        p_model = last_probs(model, full, fwd_hooks=hooks[cond])
                        if cond == "intact":
                            # replaying the intact cache must reproduce the plain forward pass
                            plain = last_probs(model, full)
                            assert np.abs(plain - p_model).max() < 1e-4, "cache replay != forward pass"
                        records += [unit_record(
                            c, pm, pb, kind="compiler", condition=cond, render=render, K=int(K),
                            floor_exhibits=bool(int(np.argmax(tp.symbol_mass(floor[render][c["lang"]]))) == c["f"]),
                        ) for c, pm, pb in zip(sub, p_model, p_bayes)]
                if args.generate and P + args.gen_len <= n_ctx:
                    for cond in conditions:
                        for _ in range(args.generate):
                            cont = sample_continuations(model, prefix, args.gen_len, gen, fwd_hooks=hooks[cond])
                            records += [generation_record(c, row, params, kind="compiler_generation",
                                                          condition=cond, render="none", K=int(K))
                                        for c, row in zip(sub, cont)]
    return records


# --- reporting ----------------------------------------------------------------------------------

def _fmt(agg, key, width=8):
    v = agg.get(key)
    if isinstance(v, float):
        return f"{v:>{width}.3f}"
    return f"{'' if v is None else str(v):>{width}s}"


def print_table(title, summary, keys):
    print(f"\n{title}")
    cols = ("n", "score", "exhibits", "runs", "tgt_mass", "bayes_tm", "othr_lng", "blank", "KL")
    header = " | ".join(f"{k:>10s}" for k in keys) + " | " + " | ".join(f"{k:>8s}" for k in cols)
    print(header)
    print("-" * len(header))
    for g in sorted(summary):
        a = summary[g]
        cells = [f"{a['n']:>8d}"] + [_fmt(a, k) for k in ("score", "exhibits", "runs", "target_mass",
                                                            "bayes_target_mass", "other_lang_mass",
                                                            "blank_mass", "kl_bayes_model")]
        print(" | ".join(f"{x!s:>10s}" for x in g) + " | " + " | ".join(cells))


def print_generation(title, summary):
    if not summary:
        return
    print(f"\n{title} (score / runs / exhibits / fired)")
    for g, a in sorted(summary.items()):
        print("  " + " ".join(str(x) for x in g) + f"  n={a['n']:>4d} score={a.get('gen_score', float('nan')):.3f} "
              f"runs={a.get('gen_runs', float('nan')):.3f} exhibits={a.get('gen_exhibits', float('nan')):.3f} "
              f"fired={a.get('gen_fired', float('nan')):.3f}")


def make_figure(rec41, rec42, out_png: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    s = summarize([r for r in rec41 if r["kind"] == "ceiling"], ["render", "target_T"])
    for render, marker in (("lang", "o"), ("shared", "s")):
        xs = sorted(t for (rd, t) in s if rd == render)
        if not xs:
            continue
        ax.plot(xs, [s[(render, t)]["score"] for t in xs], marker=marker, label=f"score, pattern in {render}")
        ax.plot(xs, [s[(render, t)]["target_mass"] for t in xs], marker=marker, ls="--", alpha=0.6,
                label=f"target mass, {render}")
        ax.plot(xs, [s[(render, t)]["bayes_target_mass"] for t in xs], color="k", ls=":", alpha=0.5,
                label="Bayes target mass" if render == "lang" else None)
    ctrl = summarize([r for r in rec41 if r["kind"] == "no_firing_control"], ["render"])
    for (render,), a in ctrl.items():
        ax.axhline(a["target_mass"], color="grey", ls="-.", alpha=0.5, label=f"no-firing control target mass ({render})")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("context length T")
    ax.set_ylabel("fraction of cells")
    ax.set_title("4.1 forced firing after >=2 firings")
    ax.legend(fontsize=7)

    styles = ["-", "--", ":", "-."]
    for ax, render in zip(axes[1:], ("shared", "lang")):
        recs = [r for r in rec42 if r["kind"] == "compiler" and r["render"] == render]
        s = summarize(recs, ["condition", "K", "target_T"])
        Ks = sorted({k for (_, k, _) in s})
        for cond, color in (("intact", "C0"), ("zero", "C3"), ("random", "C1"), ("mask", "C2")):
            for K, ls in zip(Ks, styles):
                xs = sorted(t for (c, k, t) in s if c == cond and k == K)
                if xs:
                    ax.plot(xs, [s[(cond, K, t)]["score"] for t in xs], color=color, ls=ls, marker="o", ms=3,
                            label=f"{cond}, K={K}")
        fl = [r["floor_exhibits"] for r in recs]
        if fl:
            ax.axhline(float(np.mean(fl)), color="grey", ls="-.", alpha=0.6, label="no-context floor (exhibits)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("context length T (ablated positions 1..T)")
        ax.set_title(f"4.2 K/V ablation, pattern rendered {render}")
        ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# --- main ---------------------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))
    ap.add_argument("--tracking-uri", default="file:./mlruns")
    ap.add_argument("--registry-uri", default=None)
    ap.add_argument("--steps", default="last", help="'last', 'all', or comma-separated checkpoint steps")
    ap.add_argument("--n-seqs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260904, help="held-out data seed (not the run's)")
    ap.add_argument("--cuts", default="16,24,32,40,48", help="target context lengths")
    ap.add_argument("--slack", type=int, default=4, help="how far below a target cut to look for an empty memory")
    ap.add_argument("--n-blanks", default="4,8")
    ap.add_argument("--ablate-bos", action="store_true", help="also ablate the BOS position in 4.2")
    ap.add_argument("--generate", type=int, default=0, help="samples per cell for the generation readout (0 = off)")
    ap.add_argument("--gen-len", type=int, default=24)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default="results/programming_toy")
    args = ap.parse_args()
    if not args.run_id:
        ap.error("pass --run-id or set RUN_ID")
    args.cuts = [int(v) for v in args.cuts.split(",")]
    args.n_blanks = [int(v) for v in args.n_blanks.split(",")]
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    cfg, run, persister, process, make_model = load_run(args.run_id, args.tracking_uri, args.registry_uri, args.device)
    params = tp.ToyParams.from_run_config(cfg)
    bos = int(cfg.generative_process.bos_token)
    n_ctx = int(cfg.predictive_model.instance.cfg.n_ctx)
    seq_len = n_ctx - 1
    run_name = run.info.run_name or run.info.run_id
    print(f"run {run_name} ({args.run_id}); {params}; n_ctx {n_ctx}; bos {bos}")

    steps_all = list_checkpoints(persister)
    if args.steps == "last":
        steps = steps_all[-1:]
    elif args.steps == "all":
        steps = steps_all
    else:
        steps = [int(s) for s in args.steps.split(",")]
    print(f"checkpoints available: {steps_all[:5]}...{steps_all[-3:]}; evaluating {steps}")

    tokens, langs, funcs = sample_sequences(process, args.n_seqs, seq_len, args.seed)
    n_fire = np.array([tp.firing_mask(row, params).sum() for row in tokens])
    print(f"sampled {args.n_seqs} held-out sequences of {seq_len} tokens; firings/seq mean {n_fire.mean():.2f}, "
          f"no-firing fraction {np.mean(n_fire == 0):.3f}; languages {np.bincount(langs, minlength=4).tolist()}, "
          f"functions {np.bincount(funcs, minlength=3).tolist()}")
    bayes_next = make_bayes_next(process)

    out_root = Path(args.out) / run_name
    all_summaries = {}
    for step in steps:
        ts = time.time()
        model = make_model(step)
        gen = torch.Generator().manual_seed(args.seed + step)
        out_dir = out_root / f"step_{step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== step {step} ===")
        rec41 = run_4_1(model, tokens, langs, funcs, params, args, bos, args.device, bayes_next, gen)
        rec42 = run_4_2(model, tokens, langs, funcs, params, args, bos, args.device, bayes_next, gen)
        for name, recs in (("cells_4_1.jsonl", rec41), ("cells_4_2.jsonl", rec42)):
            with open(out_dir / name, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs)

        s41 = summarize([r for r in rec41 if r["kind"] == "ceiling"], ["render", "target_T"])
        s41c = summarize([r for r in rec41 if r["kind"] == "no_firing_control"], ["render"])
        s41g = summarize([r for r in rec41 if r["kind"] == "ceiling_generation"], ["target_T"])
        s42 = summarize([r for r in rec42 if r["kind"] == "compiler"], ["render", "condition", "K", "target_T"])
        s42g = summarize([r for r in rec42 if r["kind"] == "compiler_generation"], ["condition", "K", "target_T"])
        print_table("4.1 prompted ceiling (forced firing after >=2 firings)", s41, ["render", "target_T"])
        print_table("4.1 no-firing control (Bayes: uniform responses, target mass 1/3)", s41c, ["render"])
        print_generation("4.1 generation readout", s41g)
        print_table("4.2 compiler-like state (forced firing from the blanks)", s42, ["render", "condition", "K", "target_T"])
        print_generation("4.2 generation readout", s42g)

        def jsonable(d):
            return {"|".join(str(x) for x in k): v for k, v in d.items()}
        summary = {
            "run_id": args.run_id, "run_name": run_name, "step": step,
            "params": dataclasses.asdict(params) | {"pattern": list(params.pattern)},
            "n_seqs": args.n_seqs, "seed": args.seed, "cuts": args.cuts, "n_blanks": args.n_blanks,
            "ceiling": jsonable(s41), "no_firing_control": jsonable(s41c), "ceiling_generation": jsonable(s41g),
            "compiler": jsonable(s42), "compiler_generation": jsonable(s42g), "seconds": time.time() - ts,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=1)
        try:
            make_figure(rec41, rec42, out_dir / "figure.png", f"{run_name} step {step}")
        except Exception as e:  # noqa: BLE001 -- the figure is a convenience; never lose the numbers over it
            print(f"  (figure skipped: {e!r})")
        all_summaries[step] = summary
        print(f"  step {step} done in {time.time() - ts:.0f}s -> {out_dir}")

    with open(out_root / "summaries.json", "w") as f:
        json.dump({str(k): v for k, v in all_summaries.items()}, f, indent=1)
    print(f"\nall done in {time.time() - t0:.0f}s; results under {out_root}")


if __name__ == "__main__":
    main()

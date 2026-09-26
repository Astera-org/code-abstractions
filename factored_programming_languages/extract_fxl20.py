"""
Harvest code-semantics activations from a single model across all layers.

For each program of the fxl20 corpus (20 functions x {loop, rec} x {fwd, rev} x 10 languages), runs a forward
pass and mean-pools the residual stream over the snippet's content tokens
(excluding BOS/specials, position 0, and pure-whitespace tokens). Variant
representations are averaged into one mean per (task, language) group; PCA is
run per layer over the group means (both raw and per-language-centered).

Layer index 0 is the embedding layer output (lexical baseline); transformer
layers are stored at indices 1..n_layers.

Also caches all group means and variant-level representations to
results/<model_dir>/group_means.npz for the alignment analysis and steering.

Usage (from the package root):
    uv run factored_programming_languages/extract_fxl20.py --model gemma-4-31B-it --device cuda:0 --suffix fxl20
"""

import argparse
import json
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loader                                                            # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
with open(REPO_ROOT / "models.yaml") as f:
    MODEL_CONFIGS = yaml.safe_load(f)


def get_layers_module(model, layer_path):
    module = model
    for attr in layer_path.split("."):
        module = getattr(module, attr)
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest code-semantics activations")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--data-file",
                        default=str(SCRIPT_DIR / "data" / "programs_fxl20_full.json"))
    parser.add_argument("--suffix", default=None,
                        help="Condition suffix for results dir / manifest key (e.g. 'scrambled')")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--max-memory", default=None,
                        help='shard a large model, e.g. "0:74GiB,3:74GiB,4:74GiB"')
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    hf_name = config["hf_name"]
    layer_path = config["layer_path"]
    use_bos = config["use_bos"]

    model_key = args.model + (f"+{args.suffix}" if args.suffix else "")
    dir_name = hf_name.replace("/", "--") + (f"--{args.suffix}" if args.suffix else "")
    output_dir = Path(args.output_dir)
    model_dir = output_dir / dir_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # -- Load programs -------------------------------------------------------
    print(f"Loading programs from {args.data_file}...")
    with open(args.data_file) as f:
        data = json.load(f)
    programs = data["programs"]
    languages = data["languages"]
    task_ids = [t["id"] for t in data["tasks"]]
    n_programs = len(programs)
    print(f"  {n_programs} programs, {len(task_ids)} tasks, {len(languages)} languages, "
          f"condition={data['condition']}")

    # Group index per (task, language), ordered task-major
    group_keys = [(t, l) for t in task_ids for l in languages]
    group_index = {k: i for i, k in enumerate(group_keys)}
    n_groups = len(group_keys)
    program_group = np.array([group_index[(p["task"], p["language"])] for p in programs])

    input_texts = [p["code"] for p in programs]

    # -- Load model and tokenizer --------------------------------------------
    # via loader.py, NOT AutoModelForCausalLM directly: the archive's version ignored
    # models.yaml `model_class`, which happens to survive gemma-4 through a text_config
    # fallback but misloads the Qwen3.6 / Mistral-4 wrappers, and it knows nothing of the
    # fp8 fixes Mistral needs.
    print(f"Loading model: {hf_name}...")
    _L = loader.load(args.model, device=args.device,
                     max_memory=args.max_memory, attn="sdpa")
    model, tokenizer = _L.model, _L.tok
    if use_bos and tokenizer("x", add_special_tokens=True)["input_ids"][0] != tokenizer.bos_token_id:
        # some repos (e.g. gemma-4-31B-it) ship a no-BOS post-processor even though
        # the family needs BOS; models.yaml use_bos is authoritative
        tokenizer.add_bos_token = True
        assert tokenizer("x", add_special_tokens=True)["input_ids"][0] == tokenizer.bos_token_id
    if tokenizer.pad_token is None:
        # A tekken conversion can expose NO special tokens at all -- pad, eos,
        # bos and unk all None, all_special_tokens empty (Mistral-Small-3.2-24B
        # under the transformers 5.15.1 pin it needs for a correct vocab). The
        # usual pad = eos fallback then leaves pad None and `padding=True`
        # raises. Take the pad/eos token straight out of the vocab instead:
        # padding is masked out of both the forward pass and the pooling below,
        # so this only has to be a real in-range id.
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            for cand in ("<pad>", "</s>", "<unk>"):
                i = tokenizer.convert_tokens_to_ids(cand)
                if isinstance(i, int) and i >= 0:
                    tokenizer.pad_token = cand
                    print(f"  no special tokens on this tokenizer; "
                          f"padding with {cand!r} (id {i})", flush=True)
                    break
            else:
                raise SystemExit("tokenizer has no pad/eos and no <pad> in vocab")
    tokenizer.padding_side = 'right'
    special_ids = set(tokenizer.all_special_ids)

    layers_module = _L.layers
    n_layers = _L.n_layers
    d_model = _L.d
    n_store = n_layers + 1  # index 0 = embedding output
    print(f"  {n_layers} layers (+1 embedding row), d_model={d_model}")

    # -- Tokenize everything up front, sort by length for batching ------------
    header_ntok = [0] * n_programs
    all_enc = tokenizer(input_texts, add_special_tokens=use_bos,
                        truncation=True, max_length=args.max_tokens)
    all_lengths = [len(ids) for ids in all_enc["input_ids"]]
    print(f"  token lengths: min={min(all_lengths)}, max={max(all_lengths)}, "
          f"mean={sum(all_lengths)/len(all_lengths):.1f}")
    order = sorted(range(n_programs), key=lambda i: all_lengths[i])

    # token-id -> is-pure-whitespace cache
    ws_cache = {}

    def is_whitespace_token(tid):
        if tid not in ws_cache:
            ws_cache[tid] = tokenizer.decode([tid]).strip() == ""
        return ws_cache[tid]

    # -- Hooks: embedding + every layer ---------------------------------------
    layer_activations = {}

    def make_hook(store_idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_activations[store_idx] = hidden.detach().float().cpu()
        return hook_fn

    hooks = [model.get_input_embeddings().register_forward_hook(make_hook(0))]
    for i, layer in enumerate(layers_module):
        hooks.append(layer.register_forward_hook(make_hook(i + 1)))

    # -- Storage ---------------------------------------------------------------
    # variant-level reps for the primary pooling; running sums for group means
    variant_mean = np.zeros((n_store, n_programs, d_model), dtype=np.float16)
    pool_sums = {p: np.zeros((n_store, n_groups, d_model), dtype=np.float64)
                 for p in ["mean_pool", "mean_all", "last_tok"]}
    group_counts = np.zeros(n_groups, dtype=np.int64)
    group_token_sums = np.zeros(n_groups, dtype=np.int64)
    print(f"  variant storage: {variant_mean.nbytes / 1e9:.2f} GB")

    # -- Forward passes ----------------------------------------------------------
    print(f"Processing {n_programs} forward passes (batch_size={args.batch_size})...")
    is_first_batch = True
    for batch_start in tqdm(range(0, n_programs, args.batch_size)):
        batch_idx = order[batch_start:batch_start + args.batch_size]
        batch_codes = [input_texts[i] for i in batch_idx]

        encoded = tokenizer(batch_codes, padding=True, return_tensors='pt',
                            add_special_tokens=use_bos, truncation=True,
                            max_length=args.max_tokens)
        input_ids = encoded['input_ids'].to(args.device)
        attention_mask = encoded['attention_mask'].to(args.device)

        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)

        ids_cpu = encoded['input_ids'].numpy()
        mask_cpu = encoded['attention_mask'].numpy()

        # per-sequence pooling masks
        batch_masks = []
        for j in range(len(batch_idx)):
            seq_len = int(mask_cpu[j].sum())
            content = np.zeros(ids_cpu.shape[1], dtype=bool)   # non-special, real, pos>0
            nonws = np.zeros(ids_cpu.shape[1], dtype=bool)     # content AND not whitespace
            code_start = header_ntok[batch_idx[j]] + (1 if use_bos else 0)
            for pos in range(seq_len):
                tid = int(ids_cpu[j, pos])
                if pos == 0 or pos < code_start or tid in special_ids:
                    continue
                content[pos] = True
                if not is_whitespace_token(tid):
                    nonws[pos] = True
            if not nonws.any():
                nonws = content.copy()
            last_pos = int(np.nonzero(nonws)[0][-1])
            batch_masks.append((content, nonws, last_pos))

        if is_first_batch:
            is_first_batch = False
            print(f"\nFirst batch sanity check:")
            j = 0
            p = programs[batch_idx[j]]
            content, nonws, last_pos = batch_masks[j]
            toks = [tokenizer.decode([t]) for t in ids_cpu[j][:int(mask_cpu[j].sum())]]
            print(f"  program: task={p['task']} lang={p['language']} variant={p['variant']}")
            print(f"  n_tokens={len(toks)}, content={int(content.sum())}, "
                  f"non-ws={int(nonws.sum())}, last_pos={last_pos} ('{toks[last_pos]}')")
            print(f"  first 12 tokens: {toks[:12]}")
            print(f"  code starts: {p['code'][:60]!r}")

        for store_idx in range(n_store):
            acts = layer_activations[store_idx].numpy()  # (B, T, d) float32
            for j, prog_i in enumerate(batch_idx):
                content, nonws, last_pos = batch_masks[j]
                g = program_group[prog_i]
                mp = acts[j][nonws].mean(axis=0)
                variant_mean[store_idx, prog_i] = mp.astype(np.float16)
                pool_sums["mean_pool"][store_idx, g] += mp
                pool_sums["mean_all"][store_idx, g] += acts[j][content].mean(axis=0)
                pool_sums["last_tok"][store_idx, g] += acts[j][last_pos]
        for j, prog_i in enumerate(batch_idx):
            g = program_group[prog_i]
            group_counts[g] += 1
            group_token_sums[g] += int(batch_masks[j][1].sum())

        layer_activations.clear()

    for h in hooks:
        h.remove()

    assert group_counts.min() > 0, "empty group"

    # -- Group means + cache -------------------------------------------------------
    print("Computing group means and saving cache...")
    group_means = {p: (pool_sums[p] / group_counts[None, :, None]).astype(np.float32)
                   for p in pool_sums}
    np.savez_compressed(
        model_dir / "group_means.npz",
        mean_pool=group_means["mean_pool"],
        mean_all=group_means["mean_all"],
        last_tok=group_means["last_tok"],
        variant_mean_pool=variant_mean,
        group_tasks=np.array([k[0] for k in group_keys]),
        group_languages=np.array([k[1] for k in group_keys]),
        program_tasks=np.array([p["task"] for p in programs]),
        program_languages=np.array([p["language"] for p in programs]),
        program_variants=np.array([p["variant"] for p in programs]),
        n_tokens_per_group=(group_token_sums / group_counts).astype(np.float32),
    )
    print(f"  saved {model_dir / 'group_means.npz'}")

    # -- PCA per layer (raw + per-language-centered) -------------------------------
    print("Computing PCA per layer...")
    group_tasks = [k[0] for k in group_keys]
    group_langs = [k[1] for k in group_keys]
    lang_arr = np.array(group_langs)
    for store_idx in tqdm(range(n_store)):
        acts = group_means["mean_pool"][store_idx]  # (270, d)

        n_comp = min(args.n_components, n_groups, d_model)
        pca = PCA(n_components=n_comp)
        projections = pca.fit_transform(acts)

        centered = acts.copy()
        for lang in languages:
            m = lang_arr == lang
            centered[m] -= centered[m].mean(axis=0, keepdims=True)
        pca_c = PCA(n_components=n_comp)
        projections_c = pca_c.fit_transform(centered)

        layer_data = {
            "tasks": group_tasks,
            "languages": group_langs,
            "n_tokens": (group_token_sums / group_counts).tolist(),
            "pca_projections": projections.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca_projections_centered": projections_c.tolist(),
            "explained_variance_ratio_centered": pca_c.explained_variance_ratio_.tolist(),
        }
        with open(model_dir / f"layer_{store_idx:02d}.json", "w") as f:
            json.dump(layer_data, f)

    # -- Metadata --------------------------------------------------------------------
    metadata = {
        "model_key": model_key,
        "hf_name": hf_name,
        "family": config["family"],
        "n_layers": n_store,
        "includes_embedding_row": True,
        "d_model": d_model,
        "condition": data["condition"],
        "n_programs": n_programs,
        "n_tasks": len(task_ids),
        "languages": languages,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "n_pca_components": args.n_components,
        "data_file": args.data_file,
        "pooling": "mean over content tokens (no BOS/specials, no position 0, no pure-whitespace)",
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # -- Update manifest ----------------------------------------------------------------
    manifest_file = output_dir / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
    else:
        manifest = {"models": []}

    manifest["models"] = [m for m in manifest["models"] if m["model_key"] != model_key]
    manifest["models"].append({
        "model_key": model_key,
        "hf_name": hf_name,
        "family": config["family"],
        "n_layers": n_store,
        "d_model": d_model,
        "dir_name": dir_name,
    })
    model_order = {k: i for i, k in enumerate(MODEL_CONFIGS)}
    manifest["models"].sort(key=lambda m: (model_order.get(m["model_key"].split("+")[0],
                                                           len(model_order)),
                                           m["model_key"]))
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    # -- Register experiment ---------------------------------------------------------------
    experiments_file = REPO_ROOT / "experiments.json"
    experiment_entry = {
        "dir": SCRIPT_DIR.name,
        "name": "Code Semantics Manifold",
        "description": "Functionally equivalent programs in 7 programming languages plus compiled x86-64 assembly and English descriptions (30 tasks x 9 languages, identifiers randomized). Mean-pooled activations per (task, language) group, testing whether code converges to a shared semantic representation across surface forms.",
    }
    if experiments_file.exists():
        with open(experiments_file) as f:
            experiments = json.load(f)
    else:
        experiments = []
    if not any(e["dir"] == experiment_entry["dir"] for e in experiments):
        experiments.append(experiment_entry)
        experiments.sort(key=lambda e: e["dir"])
        with open(experiments_file, "w") as f:
            json.dump(experiments, f, indent=2)

    print(f"Done! Results saved to {model_dir}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

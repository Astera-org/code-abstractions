"""FULL-WEIGHT Python-only teacher fine-tune on the novel computation
`vantril` (Fig. 4b; Tables 6, 7).

Nine Python examples (3 paraphrases x 3 demonstrations), all using the
PORTABLE 10-slot presence table -- never a Python-only idiom like
len(set(str(n))) -- so that any later cross-language reuse cannot be
explained by memorising a Python shortcut. No target-language implementation
of vantril appears in training, controller fitting, or model selection.

Optimizer recipe carried over from the archive's validated wave: lr 5e-7,
36 full-batch epochs, adamw + cosine with 3% warmup, deepspeed zero3,
gradient checkpointing. Model is loaded BEFORE TrainingArguments so deepspeed
partitions a fully materialised model -- zero.Init (like FSDP's meta-device
path) silently corrupts gemma-4.

    CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --nproc_per_node=2 \
        factored_programming_languages/run_sft_vantril.py \
        --model pythia-6.9b-tulu --batch-size 6
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForImageTextToText, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from sft_common import SFT_MODELS, load_examples                         # noqa: E402
from recipes import RECIPES                                              # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=f"one of {list(SFT_MODELS)}")
    ap.add_argument("--corpus", default="vantril_portable")
    ap.add_argument("--epochs", type=float, default=36)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--batch-size", type=int, required=True,
                    help="per-device; batch x accum x ranks must be >= 9 "
                         "(1 optimizer step == 1 epoch over the 9 examples)")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=448)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-suffix", default="")
    # the results subdirectory. Defaults to the original name so every existing
    # vantril checkpoint path is unchanged; the four later algorithms pass
    # sft_algo, which would otherwise sit under a directory named for a
    # different computation.
    ap.add_argument("--out-group", default="sft_vantril")
    ap.add_argument("--deepspeed-config", default=str(SCRIPT_DIR / "ds_zero3.json"))
    ap.add_argument("--text-only", action="store_true",
                    help="gemma-4: train the language model as a plain Gemma4ForCausalLM (the multimodal "
                         "wrapper's forward is corrupted under ZeRO-3, diag 2026-09-11); the checkpoint is "
                         "then a text-only model (model_type gemma4_text)")
    ap.add_argument("--init-ckpt", default=None,
                    help="start from this saved checkpoint instead of the base "
                         "weights (the sequential arms of the new-function x "
                         "new-language experiment: lesson B on top of lesson A "
                         "or the reverse). Tokenizer still comes from the recipe.")
    ap.add_argument("--eval-sets", default=None,
                    help="json {name: [{user, assistant}]}: each set is scored as loss "
                         "at every epoch (eval_<name>_loss in train_meta.json). The "
                         "composition experiment passes data/compose_eval_sets.json so the "
                         "never-trained vantril-in-Quorl targets are measured while training.")
    ap.add_argument("--save-every-epochs", type=int, default=0,
                    help="also write an intermediate checkpoint every N epochs (one "
                         "optimizer step is one epoch under the full-batch recipe) to "
                         "<out>/_trainer/checkpoint-<step>, tokenizer included, for "
                         "offline grading while training continues (grade_checkpoints.sh)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(REPO_ROOT / "models.yaml"))[args.model]
    hf_name = cfg["hf_name"]
    trc = {"trust_remote_code": True} if cfg.get("trust_remote_code") else {}   # custom-code models (RWKV-7)
    is_main = os.environ.get("RANK", "0") == "0"
    world = int(os.environ.get("WORLD_SIZE", "1"))

    # Load the tokenizer through the RECIPE, exactly as every eval in this
    # directory does. Loading from hf_name instead is not just a crash for
    # Mistral-3.2 (whose chat template lives in external_tokenizers/ and is
    # supplied via tokenizer_kwargs, so the bare repo tokenizer has none) -- it
    # would silently train on a different prompt rendering than we evaluate.
    # The recipe tokenizer is what gets saved with the checkpoint, so the gate
    # reloads the same template. No-op for pythia (same name, no kwargs).
    recipe = RECIPES[SFT_MODELS[args.model]["recipe"]]
    tok = AutoTokenizer.from_pretrained(
        recipe.get("tokenizer_name", recipe["hf_name"]),
        **recipe.get("tokenizer_kwargs", {}))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    exs, corpus_name = load_examples(tok, args.model, args.corpus,
                                     repeats=args.repeats, seed=args.seed,
                                     max_len=args.max_len)
    ds = Dataset.from_list(exs)
    eval_ds = None
    if args.eval_sets:
        from sft_common import encode_chat
        raw = json.loads(Path(args.eval_sets).read_text())
        eval_ds = {name: Dataset.from_list([encode_chat(tok, args.model, d["user"], d["assistant"],
                                                        max_len=args.max_len) for d in docs])
                   for name, docs in raw.items()}
        if is_main:
            print("[eval-sets] " + ", ".join(f"{k}: {len(v)}" for k, v in eval_ds.items()), flush=True)
    if is_main:
        gb = args.batch_size * args.grad_accum * world
        print(f"[corpus] {corpus_name}: {len(ds)} examples; global batch {gb} "
              f"({'FULL-BATCH' if gb >= len(ds) else 'WARNING: not full-batch'})",
              flush=True)

    out_dir = (SCRIPT_DIR / "results" / hf_name.replace("/", "--")
               / args.out_group / (f"full_{args.corpus}" + args.out_suffix))

    # Three of the five cast models (mistral-3.2, qwen3.6, gemma-4) ship as
    # multimodal checkpoints; AutoModelForCausalLM rejects their configs
    # outright (Mistral3Config -> ValueError). Dispatch on models.yaml's
    # model_class exactly as loader.py does, so training loads the same object
    # every other script in this directory does. Text-only batches leave the
    # vision tower untouched.
    # a --init-ckpt saved by a --text-only run is already a Gemma4ForCausalLM (stage-wise training,
    # 2026-09-13): load it with the causal-LM class and skip the re-housing below
    init_arch = (json.loads((Path(args.init_ckpt) / "config.json").read_text()).get("architectures") or []) if args.init_ckpt else []
    init_text_only = any(x.endswith("ForCausalLM") for x in init_arch)
    cls_ = (AutoModelForCausalLM if init_text_only
            else AutoModelForImageTextToText
            if cfg.get("model_class") == "image_text_to_text"
            else AutoModelForCausalLM)
    model = cls_.from_pretrained(
        args.init_ckpt or hf_name, dtype=torch.bfloat16, attn_implementation="eager", **trc)
    if args.text_only and init_text_only:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_pretrained(hf_name)   # base eos [1, 106, 50]
        if is_main:
            print(f"[text-only] init checkpoint is already text-only ({init_arch}); base generation_config restored", flush=True)
    elif args.text_only:
        # re-house the language model + lm_head in the text-only class; the vision/audio towers are dropped
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        tcfg = model.config.get_text_config()
        with torch.device("meta"):
            text = Gemma4ForCausalLM(tcfg)
        text.model = model.model.language_model
        text.lm_head = model.lm_head
        text.config = tcfg
        text.config.tie_word_embeddings = model.config.tie_word_embeddings
        text.generation_config = model.generation_config          # keep eos [1, 106, 50]; a fresh config has [1] only
        del model
        model = text
        if is_main:
            print(f"[text-only] Gemma4ForCausalLM around the language model; tied={model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()}", flush=True)
    model.config.use_cache = False
    if is_main and args.init_ckpt:
        print(f"[init] weights from {args.init_ckpt}", flush=True)

    targs = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=True,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps" if args.save_every_epochs else "no",
        save_steps=args.save_every_epochs or 500,
        save_only_model=True,
        eval_strategy="epoch" if eval_ds else "no",
        per_device_eval_batch_size=args.batch_size,
        report_to=[],
        dataloader_num_workers=2,
        deepspeed=args.deepspeed_config,
        gradient_checkpointing=True,
        seed=args.seed,
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100,
                                      return_tensors="pt")
    if cfg.get("family") == "gemma4" and not args.text_only:
        base_collator = collator

        def collator(feats, _b=base_collator):  # noqa: F811
            batch = _b(feats)
            batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
            return batch
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      eval_dataset=eval_ds, data_collator=collator,
                      processing_class=tok)      # so checkpoints carry the tokenizer
    train_out = trainer.train()
    trainer.save_model(str(out_dir))
    if is_main:
        tok.save_pretrained(str(out_dir))
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        eval_curves = {}
        for h in trainer.state.log_history:
            for k, v in h.items():
                if k.startswith("eval_") and k.endswith("_loss"):
                    eval_curves.setdefault(k, []).append(v)
        json.dump({"corpus": corpus_name, "full_ft": True, "lr": args.lr,
                   "epochs": args.epochs, "n_examples": len(ds),
                   "world_size": world, "per_device": args.batch_size,
                   "grad_accum": args.grad_accum,
                   "mean_loss": train_out.training_loss,
                   "final_step_loss": losses[-1] if losses else None,
                   "loss_curve": losses, "eval_curves": eval_curves},
                  open(out_dir / "train_meta.json", "w"), indent=1)
        print(f"[save] teacher -> {out_dir}; final-step loss "
              f"{losses[-1] if losses else float('nan'):.4f}", flush=True)


if __name__ == "__main__":
    main()

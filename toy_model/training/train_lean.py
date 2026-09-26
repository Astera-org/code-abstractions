"""Lean trainer for the programming toy: same data, model, optimiser and learning-rate schedule as train.py,
but no activation analysis, no visualisations, no Monte-Carlo process optimum and no model-registry push.

Kept from train.py so that checkpoints are interchangeable with the full trainer's:
  * the fixed evaluation batch (seeded by `eval_seed`, independent of `seed`) and its exact-observer optimum,
    logged as the run parameter `optimal_loss/eval_set_specific`;
  * the manual warmup + cosine learning-rate schedule (`training.lr_schedule: warmup_cosine`);
  * the checkpoint schedule, saved through the MLflow persister (loadable by behavioral_checks.load_run);
  * the `config.yaml` artifact written by genproc's managed run.

Dropped: activation tracker, visualisation, `compute_position_optimal_loss` (the process-expected optimum for
the 24-component program_call process is 1.3854 +- 0.0006 nats; see programming_minimal_model/toy/JOURNAL.md),
registry upload. An independent re-implementation trained the same models 8-10x faster than train.py this way.

Run from the repository root (see programming_minimal_model/toy/launch_train_lean.sh):
  uv run python training/train_lean.py mlflow=local generative_process=nonergodic_24x_program_call \
      predictive_model=transformer_d64_ctx64_8L_ln_w8_rope training=schedule_50K_warmup_cosine \
      seed=1 weight_init_seed=1 experiment_name=programming_toy_lean run_name=<name>
"""
import math
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matrices  # noqa: F401 — registers the custom processes (program_call among them)
import mlflow
import genproc
import torch
from omegaconf import OmegaConf
from optimal_loss import compute_eval_set_specific_optimal_loss
from genproc.generative_processes.torch_generator import (
    generate_data_batch,
    generate_data_batch_with_full_history,
)
from tqdm import tqdm
from utils import expand_state_by_batch_size

CONFIG_DIR = str(Path(__file__).parent / "configs")
OmegaConf.register_new_resolver("mult", lambda x, y: x * y, replace=True)
OmegaConf.register_new_resolver("div", lambda x, y: x // y, replace=True)

# Sequences are generated synchronously, POOL_BATCHES x batch at a time: one JAX call for 8192 sequences costs
# about 130 ms against 70 ms for 512, so generation adds ~10 ms per 28 ms training step. A background
# prefetch thread (train.py's DataPrefetcher) is much WORSE here: it contends with the launch-bound torch
# loop and quadruples the step time.
POOL_BATCHES = 16


def should_run_at_step(step, schedule):
    if not schedule:
        return False
    for start, end, interval in schedule:
        end = end if end is not None else float("inf")
        if start <= step < end:
            return (step - start) % interval == 0
    return False


@hydra.main(config_path=CONFIG_DIR, config_name="config.yaml", version_base="1.3")
@genproc.managed_run(strict=False, verbose=True)
def main(cfg, components: "genproc.Components") -> None:
    assert mlflow.active_run() is not None
    logger = components.get_logger()
    gen = components.get_generative_process()
    persister = components.get_persister()
    model = components.get_predictive_model()
    # Continuation / fine-tuning: initialise the weights from a checkpoint of ANOTHER run, e.g.
    #   +init_from.run_id=<id> +init_from.tracking_uri=file:.../mlruns +init_from.step=50000
    # Same mechanism as train.py: the step counter starts at 0 and the source is recorded as run parameters.
    init_from = cfg.get("init_from")
    if init_from:
        from genproc.persistence.mlflow_persister import MLFlowPersister

        _tracking = str(init_from.tracking_uri)
        _src_run = mlflow.MlflowClient(tracking_uri=_tracking).get_run(str(init_from.run_id))
        _src = MLFlowPersister(experiment_id=_src_run.info.experiment_id, run_id=str(init_from.run_id),
                               tracking_uri=_tracking, registry_uri=str(init_from.get("registry_uri", _tracking)))
        _dev = next(model.parameters()).device
        model = _src.load_weights(model, int(init_from.step)).to(_dev)
        print(f"[train_lean] init_from: loaded weights of run {init_from.run_id} step {init_from.step}", flush=True)
        for _k, _v in (("init_from/run_id", str(init_from.run_id)), ("init_from/step", int(init_from.step)),
                       ("init_from/tracking_uri", _tracking), ("init_from/run_name", _src_run.info.run_name or "")):
            mlflow.log_param(_k, _v)
    optimizer = components.get_optimizer()
    lr_scheduler = components.get_learning_rate_scheduler()

    device = next(model.parameters()).device
    n_ctx = int(cfg.predictive_model.instance.cfg.n_ctx)
    bos = cfg.generative_process.bos_token
    eos = cfg.generative_process.get("eos_token", None)
    seqlen = n_ctx - int(bos is not None) - int(eos is not None)
    batch = int(cfg.training.batch_size)
    total = int(cfg.training.num_steps)
    sched = cfg.training.checkpoint_schedule
    log_cheap = int(cfg.training.log_cheap_every)
    eval_every = int(cfg.training.evaluate_every)

    # Learning-rate schedule: identical to train.py's manual warmup + cosine.
    lr_for_step = None
    if getattr(cfg.training, "lr_schedule", None) == "warmup_cosine":
        lr_scheduler = None
        lr_base = optimizer.param_groups[0]["lr"]
        lr_warmup = int(cfg.training.lr_warmup_steps)
        lr_min = float(cfg.training.lr_min)

        def lr_for_step(step: int) -> float:
            if lr_warmup > 0 and step < lr_warmup:
                return lr_base * (step + 1) / lr_warmup
            prog = min(1.0, max(0.0, (step - lr_warmup) / max(1, total - lr_warmup)))
            return lr_min + 0.5 * (lr_base - lr_min) * (1 + math.cos(math.pi * prog))

    init = expand_state_by_batch_size(gen.initial_state, batch)
    loss_fn = torch.nn.CrossEntropyLoss()

    def get_loss(out, lab):
        return loss_fn(out.reshape(-1, out.shape[-1]), lab.reshape(-1).long().to(out.device))

    pool_init = expand_state_by_batch_size(gen.initial_state, batch * POOL_BATCHES)

    def generate_pool(k):
        # Pool k holds the training batches for steps k*POOL_BATCHES+1 .. (k+1)*POOL_BATCHES.
        key = jax.random.fold_in(jax.random.key(int(cfg.seed)), k)
        _, inp, lab = generate_data_batch(pool_init, gen, batch * POOL_BATCHES, seqlen, key, bos_token=bos, eos_token=eos)
        return inp.to(device), lab.to(device)

    # Fixed evaluation batch, built exactly as train.py builds it, with its exact-observer optimum.
    eval_key = jax.random.key(int(cfg.get("eval_seed", 10**9)))
    eval_outs = generate_data_batch_with_full_history(init, gen, batch, seqlen, eval_key, bos_token=bos, eos_token=eos)
    eval_inp = eval_outs["inputs"].to(device)
    eval_lab = eval_outs["labels"].to(device)
    eval_opt = compute_eval_set_specific_optimal_loss(
        gen, eval_outs["belief_states"], jnp.asarray(eval_outs["labels"].detach().cpu().numpy()))["average_loss"]
    logger.log_params({"optimal_loss/eval_set_specific": eval_opt, "trainer": "train_lean"})
    print(f"[train_lean] eval-set-specific optimal loss {eval_opt:.6f} nats", flush=True)

    def evaluate():
        model.eval()
        with torch.no_grad():
            return float(get_loss(model(eval_inp), eval_lab).item())

    n_ckpt = sum(1 for s in range(1, total + 1) if should_run_at_step(s, sched))
    print(f"[train_lean] steps={total} batch={batch} seqlen={seqlen} checkpoints={n_ckpt} device={device}", flush=True)

    t0 = time.time()
    pool = None
    for step in tqdm(range(total + 1), mininterval=30):
        if step == 0:
            logger.log_metrics(0, {"eval/loss": evaluate()})
            persister.save_weights(model, 0)
            continue
        model.train()
        if lr_for_step is not None:
            lr = lr_for_step(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        optimizer.zero_grad()
        j = (step - 1) % POOL_BATCHES
        if j == 0:
            pool = generate_pool((step - 1) // POOL_BATCHES)
        inp, lab = pool[0][j * batch:(j + 1) * batch], pool[1][j * batch:(j + 1) * batch]
        loss = get_loss(model(inp), lab)
        loss.backward()
        optimizer.step()
        ml = float(loss.detach().item())
        if lr_scheduler is not None:
            lr_scheduler.step(ml)
        if step % log_cheap == 0:
            logger.log_metrics(step, {"train/loss": ml, "train/learning_rate": optimizer.param_groups[0]["lr"],
                                      "train/steps_per_second": step / (time.time() - t0)})
        if step % eval_every == 0:
            ev = evaluate()
            logger.log_metrics(step, {"eval/loss": ev, "eval/excess_over_optimal": ev - eval_opt})
        if should_run_at_step(step, sched):
            persister.save_weights(model, step)
    ev = evaluate()
    logger.log_metrics(total, {"eval/loss": ev, "eval/excess_over_optimal": ev - eval_opt})
    print(f"[train_lean] done in {time.time() - t0:.0f} s; final eval loss {ev:.6f} "
          f"(optimum on this batch {eval_opt:.6f}, excess {ev - eval_opt:+.6f})", flush=True)


if __name__ == "__main__":
    main()

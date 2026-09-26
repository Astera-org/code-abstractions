"""Train and grade the four new algorithms across the fig5 roster.

One unit of work is a (algorithm, model, dose) cell: full-weight SFT on the
nine Python examples, then eval_algo.py on ten languages, then DELETE the
checkpoint -- each is about 62 GB and 48 of them do not fit.

The dose ladder per model is copied from the vantril runs rather than
re-chosen, so the new algorithms are comparable to the published column.
Note pythia's two upper doses ran 80 epochs, not 36; that asymmetry is in the
original ladder and is reproduced here on purpose.

The gate is not set here. It is DERIVED afterwards, per (algorithm, model), as
the lowest dose whose three held-out Python acquisition probes all pass -- so
all three doses are run even when the vantril gate is already known, because
the learning rate a computation needs is a property of the computation.

Each cell writes a marker under results/_sweep/ when its eval is complete, so
an interrupted sweep resumes without retraining. Markers, not process greps.

    uv run factored_programming_languages/run_algo_sweep.py --lane pythia
    uv run factored_programming_languages/run_algo_sweep.py --lane big
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from algos import ALGOS                                                  # noqa: E402
from sft_common import SFT_MODELS                                        # noqa: E402

REPO = ROOT.parent
MARK = ROOT / "results" / "_sweep"
LOGS = ROOT / "results" / "_sweep" / "logs"

# (tag, lr, epochs). Verified against the train_meta.json of every vantril run.
LADDER = {
    "pythia-6.9b-tulu": [("teacher", 5e-7, 36), ("teacher_hi", 2e-6, 80),
                         ("teacher_vhi", 8e-6, 80)],
}
DEFAULT_LADDER = [("teacher", 5e-7, 36), ("teacher_hi", 2e-6, 36),
                  ("teacher_vhi", 8e-6, 36)]
SUFFIX = {"teacher": "", "teacher_hi": "_hi", "teacher_vhi": "_vhi"}

BIG = ["gemma-4-31B-it", "gemma-4-26b-a4b", "qwen3.6-27b"]
SMALL = ["pythia-6.9b-tulu"]
RWKV = ["rwkv7-g1j-13.3b"]          # attention-free; 4 ranks; transformers==5.15.1 overlay (custom code)
NEW_ALGOS = [a for a in ALGOS if a != "vantril"]

ap = argparse.ArgumentParser()
ap.add_argument("--lane", required=True, choices=["pythia", "big", "rwkv"])
ap.add_argument("--algos", nargs="+", default=NEW_ALGOS)
ap.add_argument("--models", nargs="+", default=None)
ap.add_argument("--doses", nargs="+", default=None,
                help="subset of teacher/teacher_hi/teacher_vhi")
ap.add_argument("--cards", default=None,
                help="comma-separated CUDA_VISIBLE_DEVICES for this lane")
ap.add_argument("--base-evals", action="store_true",
                help="only run the untouched-base evals (the native ceiling)")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

MODELS = a.models or {"pythia": SMALL, "big": BIG, "rwkv": RWKV}[a.lane]
CARDS = a.cards or {"pythia": "0,1", "big": "0,1,2,3,4,5,6,7", "rwkv": "4,5,6,7"}[a.lane]
NRANK = len(CARDS.split(","))
MARK.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
CFG = yaml.safe_load(open(REPO / "models.yaml"))


def uvrun(model):
    """custom-code models (RWKV-7) run under the transformers==5.15.1 overlay"""
    return os.environ.get("UVRUN", "uv run").split() + (["--with", "transformers==5.15.1"] if CFG[model].get("trust_remote_code") else [])


def ensure_custom_code(model, ck):
    """a custom-code checkpoint must carry its own modeling files for --ckpt loading"""
    if not CFG[model].get("trust_remote_code"):
        return
    import glob
    hub = os.environ.get("HF_HUB_CACHE") or str(Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    snaps = sorted(glob.glob(str(Path(hub) / ("models--" + CFG[model]["hf_name"].replace("/", "--")) / "snapshots" / "*")))
    for f in ("modeling_rwkv7.py", "configuration_rwkv7.py", "tokenization_rwkv7.py"):
        if snaps and not (ck / f).exists() and (Path(snaps[-1]) / f).exists():
            shutil.copy(Path(snaps[-1]) / f, ck / f)


def sh(cmd, log, env=None):
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = CARDS
    e.update(env or {})
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                               cwd=str(REPO), env=e)


def base_eval(algo, model):
    mk = MARK / f"base_{algo}_{model}.done"
    if mk.exists():
        return
    log = LOGS / f"base_{algo}_{model}.log"
    print(f"[base ] {algo} {model} -> {log.name}", flush=True)
    if a.dry_run:
        return
    rc = sh(uvrun(model) + ["factored_programming_languages/eval_algo.py",
             "--algo", algo, "--model", model, "--tag", "base"], log)
    if rc:
        print(f"  FAILED rc={rc}, see {log}", flush=True)
        return
    mk.write_text(json.dumps({"algo": algo, "model": model, "arm": "base"}))


def cell(algo, model, tag, lr, epochs):
    mk = MARK / f"{algo}_{model}_{tag}.done"
    if mk.exists():
        print(f"[skip ] {algo} {model} {tag} (marker present)", flush=True)
        return
    hf = CFG[model]["hf_name"].replace("/", "--")
    ck = ROOT / "results" / hf / "sft_algo" / f"full_{algo}_portable{SUFFIX[tag]}"
    log = LOGS / f"{algo}_{model}_{tag}.log"
    spec = SFT_MODELS[model]
    t0 = time.time()
    print(f"[train] {algo} {model} {tag} lr={lr} ep={epochs} "
          f"ranks={NRANK} cards={CARDS} -> {log.name}", flush=True)
    if a.dry_run:
        return
    rc = sh(uvrun(model) + ["torchrun", f"--nproc_per_node={NRANK}",
             # deterministic, not hash(): PYTHONHASHSEED is randomised per
             # process, so parallel lanes could collide on a port
             f"--master_port={29500 + zlib.crc32(f'{algo}{model}{tag}'.encode()) % 900}",
             "factored_programming_languages/run_sft_vantril.py",
             "--model", model, "--corpus", f"{algo}_portable",
             "--batch-size", str(spec["per_device"]),
             "--lr", str(lr), "--epochs", str(epochs),
             "--out-suffix", SUFFIX[tag], "--out-group", "sft_algo"], log)
    if rc or not (ck / "config.json").exists():
        print(f"  TRAIN FAILED rc={rc}, see {log}", flush=True)
        return
    ensure_custom_code(model, ck)
    t1 = time.time()
    print(f"[eval ] {algo} {model} {tag}  (train {t1 - t0:.0f}s)", flush=True)
    rc = sh(uvrun(model) + ["factored_programming_languages/eval_algo.py",
             "--algo", algo, "--model", model, "--ckpt", str(ck),
             "--tag", tag], log)
    if rc:
        print(f"  EVAL FAILED rc={rc}, see {log} -- keeping checkpoint",
              flush=True)
        return
    evd = ROOT / "results" / f"eval{algo}_{model}_{tag}"
    summ = json.loads((evd / "summary.json").read_text())
    # train_meta.json holds the lr and the loss curve and lives INSIDE the
    # checkpoint, so it has to be copied out before the delete or the only
    # record of what was trained goes with the weights.
    tm = ck / "train_meta.json"
    meta = json.loads(tm.read_text()) if tm.exists() else None
    if meta:
        (evd / "train_meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(ck, ignore_errors=True)
    mk.write_text(json.dumps({"algo": algo, "model": model, "tag": tag,
                              "lr": lr, "epochs": epochs,
                              "final_step_loss": (meta or {}).get("final_step_loss"),
                              "mean_loss": (meta or {}).get("mean_loss"),
                              "train_s": round(t1 - t0),
                              "eval_s": round(time.time() - t1),
                              "summary": summ}, indent=1))
    print(f"  done in {time.time() - t0:.0f}s  "
          f"S_python={summ.get('S_python_heldout')}  "
          f"{algo}={summ.get(algo, {}).get('hits')}/"
          f"{summ.get(algo, {}).get('n')}  (checkpoint deleted)", flush=True)


for algo in a.algos:
    for model in MODELS:
        if a.base_evals:
            base_eval(algo, model)
            continue
        for tag, lr, ep in LADDER.get(model, DEFAULT_LADDER):
            if a.doses and tag not in a.doses:
                continue
            cell(algo, model, tag, lr, ep)
print("[lane complete]", flush=True)

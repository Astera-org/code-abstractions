"""Stage B — install the compiled suffix into a demo-free prompt (10-demo-language design: one program per language
per function; every donor's demo language x differs from the target L, so the pooled state never saw a demo in the
language it is asked to render). Cells are gated per (f, target) on Stage A: only donor languages whose suffix
rendered f in situ at that target (carry hit) are used; a cell with no admissible donor is recorded as not addressable.
Arms: none (neutral prompt) / pooled (mean over admissible donor languages) / each admissible donor language alone /
wrong (another chosen function's pooled admissible suffix) / rand (norm-matched noise around the neutral suffix).
    uv run icl_section/run_stageB.py --funcs f1 f2 f3 f4 --device cuda:0 --shard 0/8 --out results/stageB.shard0.json
"""
import argparse, json, sys, time
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "factored_programming_languages"))   # shared modules (algos, recipes, elicit, code_style, languages, tasks_function, entry)
from stage_common import Rig, DON, admissible                            # noqa: E402
from tasks_function import LANGS                                         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gemma-4-31B-it"); ap.add_argument("--device", default=None)
ap.add_argument("--funcs", nargs="+", required=True); ap.add_argument("--langs", nargs="+", default=LANGS)
ap.add_argument("--stageA", default=str(HERE / "results" / "stageA.json"))
ap.add_argument("--shard", default=None); ap.add_argument("--out", default=str(HERE / "results" / "stageB.json"))
a = ap.parse_args()
A = json.loads(Path(a.stageA).read_text()); rig = Rig(a.model, a.device); Lr = len(rig.neutral_ids())
print(f"[rig] K={rig.K} suffix rows; neutral prompt {Lr} tokens", flush=True)
out_path = Path(a.out); res = json.loads(out_path.read_text()) if out_path.exists() else {}
SK, SN = (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)


def mean(sufs):
    return {i: {nm: torch.stack([s[i][nm] for s in sufs]).mean(0) for nm in rig.names[i]} for i in range(rig.n_layers)}


t0 = time.time(); n = 0; idx = -1
for f in a.funcs:
    for L in a.langs:
        idx += 1
        if idx % SN != SK: continue
        adm = admissible(A, f, L)                                       # [(x, 0)] with x != L
        if not adm:
            res[f"{f}|not_addressable|{L}"] = {"admissible": []}; print(f"  {f:<9}{L:<11}not addressable (no donor compiles here)", flush=True); continue
        suf = {x: rig.suffix_kv(f, x, v, Lr) for x, v in adm}
        arms = {"none": None, "pooled": mean(list(suf.values()))}
        for x in suf: arms[x] = suf[x]
        others = [g for g in a.funcs if g != f and admissible(A, g, L)]
        if others:
            g = others[0]; arms["wrong"] = mean([rig.suffix_kv(g, x, v, Lr) for x, v in admissible(A, g, L)]); wrong_g = g
        with torch.inference_mode():                                    # norm-matched noise around the neutral compiled state
            ids = rig.neutral_ids(); c = rig.model(input_ids=torch.tensor([ids], device=rig.dev), use_cache=True).past_key_values
            neutral = rig.harvest(c, rig.K)
        gen = torch.Generator().manual_seed(0)
        arms["rand"] = {i: {nm: neutral[i][nm] + (r := torch.randn(neutral[i][nm].shape, generator=gen).to(rig.dev)) * ((arms["pooled"][i][nm] - neutral[i][nm]).norm() / (r.norm() + 1e-9))
                            for nm in rig.names[i]} for i in range(rig.n_layers)}
        chan = {arm: ("kv", "rec") for arm in arms}
        if rig.hybrid:                                                  # channel decomposition: attention rows only / recurrent state only
            arms["pooled_kv"] = arms["pooled"]; chan["pooled_kv"] = ("kv",)
            arms["pooled_rec"] = arms["pooled"]; chan["pooled_rec"] = ("rec",)
        for arm, kv in arms.items():
            key = f"{f}|{arm}|{L}"
            if key in res: continue
            r = rig.grade(rig.install_and_generate(kv, L, chan[arm]), f, L); r["admissible"] = [x for x, _ in adm]
            if arm == "wrong": r["expect"] = wrong_g
            res[key] = r; n += 1
            print(f"  {f:<9}{arm:<11}{L:<11}hit={r['hit']!s:<5} which={str(r['which']):<10} ({(time.time() - t0) / n:.1f}s/cell)", flush=True)
        out_path.write_text(json.dumps(res, indent=1))
out_path.write_text(json.dumps(res, indent=1)); print(f"\n[saved] {out_path} ({len(res)} cells)")

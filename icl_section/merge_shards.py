"""Merge results/<out>.shard*.json into results/<out>.json (keys are disjoint by construction).
    merge_shards.py results/stageA [--drop-old-donors]
--drop-old-donors removes keys whose donor tag carries a variant index (python0, java1, ...): cells of the archived
3-demo-language design that a runner re-loaded from a same-named shard file (2026-09-23)."""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
args = [x for x in sys.argv[1:] if not x.startswith("--")]; drop_old = "--drop-old-donors" in sys.argv
base = Path(args[0]); base = base if base.is_absolute() else HERE / base
out = base.parent / (base.name + ".json")                   # a plain concatenation: "stageA_qwen3.6-27b" carries a fake ".6-27b" extension
merged = json.loads(out.read_text()) if out.exists() else {}
n = 0
for sh in sorted(base.parent.glob(base.name + ".shard*.json")):
    part = json.loads(sh.read_text()); n += len(part); merged.update(part)
if drop_old:
    before = len(merged); merged = {k: v for k, v in merged.items() if not re.search(r"\d$", k.split("|")[1])}
    print(f"[merge] dropped {before - len(merged)} old-design keys")
out.write_text(json.dumps(merged, indent=1))
print(f"[merge] {n} cells from shards -> {out.name} ({len(merged)} total)")

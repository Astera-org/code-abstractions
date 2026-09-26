"""Assemble the FINAL ladder set 9F for one model by hard-linking cells (authors, 2026-09-09):
transport rungs from the factor-specific severed runs, install and compose rungs from the
unsevered alpha-1 runs (set 9U1); relabel controls symlinked from 9U2. meta.json records the
ceiling arm per rung so fig_tci_ladder4.py normalises transport by the severed ask and
install/compose by the plain prompted ask.

    uv run factored_programming_languages/assemble_9F.py <model> [--transport-suffix bzero|bnative]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--transport-suffix", default="bzero", help="tcispan9_<f|l|s>_<suffix>_<model>: bzero (severed, "
                "gemma; both channels on hybrids) or bnative (hybrids: recurrent state intact, native contrast)")
ap.add_argument("--transport-prefix", default="tcispan9",
                help="tcispan9 = the factor-specific cut (only the steered factor's words go); tcispan9w = the "
                     "whole-specification cut (authors, 2026-09-17, tier 2)")
ap.add_argument("--install-from", default="9U1", choices=["9U1", "masked", "masked_native"],
                help="9U1 = install rungs unsevered, the historical regime; masked = install severed by the same "
                     "attention mask as transport (results/tciins9m_*, tcisins9m_*), which makes install's ceiling "
                     "prompted_sev like transport's instead of the plain prompted ask")
ap.add_argument("--force", action="store_true")
ap.add_argument("--set", default="9F", help="destination set name, tci<set>_<model>. Use a fresh name when the "
                "transport rungs come from a different severing instrument so the existing set is not overwritten "
                "-- e.g. --transport-suffix bmask --set 9M (authors, 2026-09-17: transport under the attention "
                "mask). Install and compose rungs are the same 9U1 cells either way.")
a = ap.parse_args()
M, T, SET, TP = a.model, a.transport_suffix, a.set, a.transport_prefix


def link_all(src, dst, keep):
    """hard-link the cells of src that `keep` accepts into dst"""
    n = 0
    for f in src.glob("c_*.json"):
        if keep(f.name):
            tgt = dst / f.name
            if not tgt.exists():
                os.link(f, tgt); n += 1
    return n


# install rungs: unsevered (9U1) or severed by the same mask as transport. Compose is never in this
# choice -- its recipient is the carrier, which has no specification between the frame and the tail,
# so there is nothing a mask could remove and the 9U1 compose cells ARE the masked ones.
# masked_native: the masked install run in the NATIVE recurrent regime, so a hybrid's install
# rungs use the same donor contrast as its transport rungs (authors, 2026-09-18). On an
# all-attention model the two masked variants are the same run.
_INS = {"9U1": (f"tci9U1_{M}", f"tcis9U1_{M}"),
        "masked": (f"tciins9m_{M}", f"tcisins9m_{M}"),
        "masked_native": (f"tciins9mn_{M}", f"tcisins9mn_{M}")}[a.install_from]
if a.install_from == "masked_native" and not (R / _INS[0]).is_dir():
    _INS = (f"tciins9m_{M}", f"tcisins9m_{M}")     # no native run for this model: masked is it
INS_FL, INS_S = R / _INS[0], R / _INS[1]
INS_CEIL = "prompted" if a.install_from == "9U1" else "prompted_sev"

plan = [  # (destination, [(source, filter)])
    (R / f"tci{SET}_{M}", [(INS_FL, lambda n: n.startswith(("c_install_", "c_L_install_"))),
                           (R / f"tci9U1_{M}", lambda n: n.startswith("c_compose_")),
                        (R / f"{TP}_f_{T}_{M}", lambda n: n.startswith("c_transport_")),
                        (R / f"{TP}_l_{T}_{M}", lambda n: n.startswith("c_L_transport_"))]),
    (R / f"tcim{SET}_{M}", [(R / f"tcim9U1_{M}", lambda n: True)]),
    (R / f"tcis{SET}_{M}", [(INS_S, lambda n: n.startswith("c_install_")),
                         (R / f"{TP}_s_{T}_{M}", lambda n: n.startswith("c_transport_"))]),
    (R / f"tcic3m{SET}_{M}", [(R / f"tcic3m9U1_{M}", lambda n: True)]),
]
missing = [str(src.name) for _, srcs in plan for src, _ in srcs if not src.is_dir()]
if missing:
    print("missing sources:", missing); sys.exit(1)
for dst, srcs in plan:
    if dst.exists() and a.force:
        shutil.rmtree(dst)
    dst.mkdir(exist_ok=True)
    counts = [link_all(src, dst, keep) for src, keep in srcs]
    print(f"{dst.name}: " + ", ".join(f"{src.name} -> {c}" for (src, _), c in zip(srcs, counts)))
    # meta: the ceiling per rung, plus provenance
    metas = {src.name: json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {} for src, _ in srcs}
    meta = {"set": SET, "model": M, "sources": list(metas),
            "ceiling": {"transport": "prompted_sev", "install": INS_CEIL, "compose": "prompted"},
            "transport_regime": metas.get(f"{TP}_f_{T}_{M}", metas.get(f"{TP}_s_{T}_{M}", {})),
            "install_from": a.install_from, "transport_prefix": TP,
            "install_compose_regime": metas.get(INS_FL.name, metas.get(INS_S.name, {}))}
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
# relabel controls: the 9U2 seeds, symlinked under the 9F name
n = 0
for pref in ("tcirels", "tcirelsi"):
    for src in sorted(R.glob(f"{pref}9U2_{M}_s*")):
        dst = R / src.name.replace("9U2", SET)
        if not dst.exists():
            os.symlink(src.name, dst); n += 1
print(f"relabel controls linked: {n}")

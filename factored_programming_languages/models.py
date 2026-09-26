"""The four-model cast, and the ONE place a band is looked up.

Narrowed 2026-08-20. Everything in this directory runs on exactly these four; a fifth
model does not enter without a decision recorded in README.md.

There is one table here and one loader.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
BAND_FILE = ROOT / "results" / "bands_fxl20.json"


class Model:
    def __init__(self, key, recipe, band_title, dirname, memory, note=""):
        self.key = key                  # models.yaml key (and --model value)
        self.recipe = recipe            # recipes.py key
        self.band_title = band_title    # key inside bands_fxl20.json
        self.dirname = dirname          # results/<dirname>--fxl20
        self.memory = memory            # prompt-memory backend: "kv" or "hybrid"
        self.note = note

    def __repr__(self):
        return f"Model({self.key})"


CAST = [
    Model("gemma-4-31B-it", "gemma-4-31b-it", "gemma-4-31B-it",
          "google--gemma-4-31B-it", "kv",
          "primary; 60 layers, d=5376; KV shapes differ BETWEEN layers"),
    Model("gemma-4-31B", "gemma-4-31b-it", "gemma-4-31B (base)",
          "google--gemma-4-31B", "kv",
          "PRETRAINED base of the primary; same architecture; completion "
          "prompts only -- no chat template, recipe key borrowed for loader "
          "plumbing and unused"),
    # pretrained twins of the compiler-like models (added 2026-09-03). Keys
    # are -pt because "base" already means "untouched pre-SFT instruct model"
    # in the fine-tuning work. Completion prompts only -- no chat template; the
    # recipe key is borrowed for loader plumbing and never used to render.
    Model("gemma-4-12b-pt", "gemma-4-12b-it", "gemma-4-12b (pretrained)",
          "google--gemma-4-12b", "kv",
          "PRETRAINED twin of gemma-4-12b-it. NB model_type is gemma4_unified, "
          "not gemma4 like the 31B/26b -- needs a transformers that knows it"),
    Model("gemma-4-26b-a4b-pt", "gemma-4-26b-a4b", "gemma-4-26B-A4B (pretrained)",
          "google--gemma-4-26b-a4b", "kv",
          "PRETRAINED twin of gemma-4-26b-a4b; models.yaml maps the un-suffixed "
          "key to the -it weights, hence the -pt key here"),
    Model("qwen3.5-35b-a3b-pt", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B (pretrained)",
          "Qwen--Qwen3.5-35B-A3B-Base", "kv",
          "PRETRAINED twin of qwen3.5-35b-a3b; the only Qwen Base at this size "
          "(3.6 and 3.8 ship no Base)"),
    Model("glm-5.3-flash", "glm-5.3-flash", "GLM-5.3-Flash",
          "zai-org--GLM-5.3-Flash", "hybrid",
          "GLM flagship-Flash (authors, 2026-08-28): 328GB fp8, 288-expert MoE, "
          "45 layers d=4096, hybrid linear attention. Interface spectrum + "
          "second non-qwen hybrid for the recurrent-store question. "
          "transformers==5.16.1 overlay; prefill-closed thought channel"),
    Model("qwen3.8-27b", "qwen3.8-27b", "Qwen3.8-27B",
          "Qwen--Qwen3.8-27B", "hybrid",
          "successor of qwen3.6-27b, same re-unified hybrid class: 64 layers, "
          "d=5120, 16 full-attention / 48 linear; dense. Onboarded 2026-08-28 "
          "for the interface spectrum (mean-ablation instrument)"),
    Model("qwen3.6-27b", "qwen3.6-27b", "Qwen3.6-27B",
          "Qwen--Qwen3.6-27B", "hybrid",
          "64 layers, d=5120; only 16 are full-attention (3,7,...,63), 48 linear"),
    Model("mistral-small-4-119b", "mistral-small-4-119b", "Mistral-Small-4-119B",
          "mistralai--Mistral-Small-4-119B-2603", "kv",
          "119B MoE (A6.5B), fp8; 36 layers, d=4096; MLA but HF decompresses to "
          "a uniform [1,32,S,128] cache"),
    Model("qwen3.6-35b-a3b", "qwen3.6-35b-a3b", "Qwen3.6-35B-A3B",
          "Qwen--Qwen3.6-35B-A3B", "hybrid",
          "MoE sibling of qwen3.6-27b: 256 experts top-8, ~3B active; 40 layers, "
          "d=2048 (lowest state budget in the cast); full attention every 4th layer. "
          "Onboarded for the family-vs-architecture contrast"),
    Model("exaone-4-32b", "exaone-4-32b", "EXAONE-4.0-32B",
          "LGAI-EXAONE--EXAONE-4.0-32B", "kv", "LCB ladder 66.7; dense, LG lab"),
    Model("gemma-4-12b-it", "gemma-4-12b-it", "gemma-4-12b-it",
          "google--gemma-4-12b-it", "kv", "LCB ladder ~72; small compiling-lineage"),
    Model("qwen3.5-397b-a17b", "qwen3.5-397b-a17b", "Qwen3.5-397B-A17B",
          "Qwen--Qwen3.5-397B-A17B-FP8", "hybrid",
          "qwen3.5 size-sweep top point, fp8 8-card. Not a scatter point"),
    Model("qwen3.5-0.8b", "qwen3.5-0.8b", "Qwen3.5-0.8B",
          "Qwen--Qwen3.5-0.8B", "hybrid",
          "qwen3.5 size-sweep checkpoint. Not a scatter point"),
    Model("qwen3.5-2b", "qwen3.5-2b", "Qwen3.5-2B",
          "Qwen--Qwen3.5-2B", "hybrid",
          "qwen3.5 size-sweep checkpoint. Not a scatter point"),
    Model("qwen3.5-4b", "qwen3.5-4b", "Qwen3.5-4B",
          "Qwen--Qwen3.5-4B", "hybrid",
          "qwen3.5 size-sweep checkpoint. Not a scatter point"),
    Model("qwen3.5-9b", "qwen3.5-9b", "Qwen3.5-9B",
          "Qwen--Qwen3.5-9B", "hybrid",
          "qwen3.5 size-sweep checkpoint. Not a scatter point"),
    Model("qwen3.5-27b", "qwen3.5-27b", "Qwen3.5-27B",
          "Qwen--Qwen3.5-27B", "hybrid",
          "qwen3.5 size-sweep checkpoint. Not a scatter point"),
    Model("qwen3.5-122b-a10b", "qwen3.5-122b-a10b", "Qwen3.5-122B-A10B",
          "Qwen--Qwen3.5-122B-A10B-FP8", "hybrid",
          "qwen3.5 size-sweep checkpoint. Not a scatter point"),
    Model("qwen3.5-35b-a3b", "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B",
          "Qwen--Qwen3.5-35B-A3B", "hybrid", "LCB ladder ~74.6; prior qwen gen MoE"),
    Model("gemma-4-26b-a4b", "gemma-4-26b-a4b", "gemma-4-26B-A4B",
          "google--gemma-4-26b-a4b-it", "kv", "LCB ladder ~77.1; google MoE"),
    Model("gemma-2-9b-it", "gemma-2-9b-it", "gemma-2-9b-it",
          "google--gemma-2-9b-it", "kv",
          "red-team: weak coder (2024, old-LCB teens), generalist, compiling lineage"),
    Model("qwen2.5-14b", "qwen2.5-14b", "Qwen2.5-14B",
          "Qwen--Qwen2.5-14B-Instruct", "kv",
          "red-team: weak coder (2024, LCB ~20s), generalist"),
    Model("qwen3-coder-30b", "qwen3-coder-30b", "Qwen3-Coder-30B",
          "Qwen--Qwen3-Coder-30B-A3B-Instruct", "kv",
          "code specialist from a compiling lab: qwen3_moe 128 experts top-8, d=2048 "
          "x 48, no MLA. Separates specialization from lab lineage. No certified band"),
    Model("dscoder-v2-lite", "dscoder-v2-lite", "DeepSeek-Coder-V2-Lite",
          "deepseek-ai--DeepSeek-Coder-V2-Lite-Instruct", "kv",
          "confound discriminator: MLA (kv_lora_rank 512) + code-heavy + MoE (64 top-6); "
          "d=2048 x 27, lowest state budget. No certified band yet"),
    Model("devstral-2-123b", "devstral-2-123b", "Devstral-2-123B",
          "mistralai--Devstral-2-123B-Instruct-2512", "kv",
          "confound discriminator (authors): Mistral-lab code-heavy, DENSE ministral3, "
          "no MLA, d=12288 x 88, highest state budget, fp8 static. No certified band yet"),
    Model("ministral-3-8b", "ministral-3-8b", "Ministral-3-8B",
          "mistralai--Ministral-3-8B-Instruct-2512", "kv",
          "scatter fill-in (authors): current-gen regular Mistral, Devstral's "
          "generalist sibling (2512, ministral3, fp8). No certified band"),
    Model("mistral-small-3.2-24b", "mistral-small-3.2-24b", "Mistral-Small-3.2-24B",
          "mistralai--Mistral-Small-3.2-24B-Instruct-2506", "kv",
          "scatter fill-in: prior-gen regular Mistral (2506, dense bf16, 40 layers "
          "d=5120). Lineage-trajectory point. No certified band"),
    Model("mistral-7b-v0.3", "mistral-7b-v0.3", "Mistral-7B-v0.3",
          "mistralai--Mistral-7B-Instruct-v0.3", "kv",
          "scatter fill-in: 2023-era Mistral anchor, era peer of the red-team "
          "witnesses. No certified band"),
    Model("gemma-3-27b-it", "gemma-3-27b-it", "gemma-3-27b-it",
          "google--gemma-3-27b-it", "kv",
          "scatter fill-in: gemma era gap between gemma-2 and gemma-4 (2025, dense "
          "62 layers d=5376, hybrid sliding-window). No certified band"),
    Model("glm-4-32b-instruct", "glm-4-32b-instruct", "GLM-4-32B",
          "zai-org--GLM-4-32B-0414", "kv",
          "interface spectrum ONLY (bench + flow-cut; no CFS/compose/dose). A "
          "lineage outside gemma/qwen/mistral, which is the test: the "
          "compiler/re-reader split has so far tracked family, so an "
          "independent draw says whether that is lineage or coincidence. "
          "No certified band"),
    Model("qwen3-32b", "qwen3-32b", "Qwen3-32B",
          "Qwen--Qwen3-32B", "kv",
          "fill-in: qwen 3.0 dense, brackets the 2.5->3.5 override emergence. "
          "No certified band"),
    Model("mistral-small-24b-2501", "mistral-small-24b-2501", "Mistral-Small-24B-2501",
          "mistralai--Mistral-Small-24B-Instruct-2501", "kv",
          "fill-in: Mistral lineage point 2501 (Small 3.0). No certified band"),
    Model("gemma-3-12b-it", "gemma-3-12b-it", "gemma-3-12b-it",
          "google--gemma-3-12b-it", "kv",
          "fill-in round 3: second gemma-3 point. No certified band"),
    Model("qwen3-30b-a3b", "qwen3-30b-a3b", "Qwen3-30B-A3B",
          "Qwen--Qwen3-30B-A3B", "kv",
          "fill-in: gen-3.0 MoE, disambiguates the qwen3-32b dip (generation vs "
          "architecture). No certified band"),
    Model("mistral-nemo-12b", "mistral-nemo-12b", "Mistral-Nemo-12B",
          "mistralai--Mistral-Nemo-Instruct-2407", "kv",
          "fill-in: mid-2024 Mistral, bisects the factorization emergence "
          "interval. No certified band"),
    Model("qwen3-32b-think", "qwen3-32b-think", "Qwen3-32B-think",
          "Qwen--Qwen3-32B--think", "kv",
          "H1 discriminator: qwen3-32b with thinking ON, same weights. Not a "
          "scatter point"),
    Model("qwen3-32b-thinkstub", "qwen3-32b-thinkstub", "Qwen3-32B-thinkstub",
          "Qwen--Qwen3-32B--thinkstub", "kv",
          "H1 control: inert prefilled scratchpad. Not a scatter point"),
    Model("qwen3-32b-thinkpad", "qwen3-32b-thinkpad", "Qwen3-32B-thinkpad",
          "Qwen--Qwen3-32B--thinkpad", "kv",
          "H1 arm: length-matched inert scratchpad. Not a scatter point"),
    Model("qwen3-32b-padnothink", "qwen3-32b-padnothink", "Qwen3-32B-padnothink",
          "Qwen--Qwen3-32B--padnothink", "kv",
          "H1 confound control: thinkpad prose as plain assistant text outside "
          "think tags, token-count-matched (462=462). Not a scatter point"),
    Model("qwen3-30b-2507", "qwen3-30b-2507", "Qwen3-30B-2507",
          "Qwen--Qwen3-30B-A3B-Instruct-2507", "kv",
          "recipe-surgery witness: split-era Instruct twin of the dip MoE. "
          "No certified band"),
    # attention-free replication model (run_rwkv_tci.py, icl_section). No K/V rows and no hybrid split: its
    # prompt memory is the recurrent state alone, so neither transformer rig applies. Onboarded to the cast
    # 2026-09-24 for the vantril teacher ladder (fig5_reuse_bars); custom code, transformers==5.15.1 overlay.
    Model("rwkv7-g1j-13.3b", "rwkv7-g1j-13.3b", "RWKV7-G1j-13.3B",
          "RWKV--RWKV7-G1j-13.3B-20260831", "recurrent",
          "RWKV-7 G1j 13.3B, 61 blocks, d=4096; attention-free"),
    Model("pythia-6.9b-tulu", "pythia-6.9b-tulu", "pythia-6.9b-tulu",
          "allenai--open-instruct-pythia-6.9b-tulu", "kv",
          "control, weak code prior; tulu template, INVERTED elicitation ladder "
          "(bare V1 wins) and turn-boundary stop strings are mandatory"),
    # smoke-test models (not in the paper); see README.md "Smoke test"
    Model("pythia-70m", "pythia-70m", "pythia-70m", "EleutherAI--pythia-70m", "kv", "smoke test only"),
    Model("pythia-160m", "pythia-160m", "pythia-160m", "EleutherAI--pythia-160m", "kv", "smoke test only"),
    Model("pythia-410m", "pythia-410m", "pythia-410m", "EleutherAI--pythia-410m", "kv", "smoke test only"),
    Model("qwen3.5-0.8b", "qwen3.5-0.8b", "qwen3.5-0.8b", "Qwen--Qwen3.5-0.8B", "hybrid", "smoke test only (hybrid code paths)"),
]

BY_KEY = {m.key: m for m in CAST}


def get(key):
    if key not in BY_KEY:
        raise KeyError(f"{key!r} is not in the cast: {sorted(BY_KEY)}")
    return BY_KEY[key]


def band(key):
    """(lo, hi) certified band rows for a model, 1-indexed layer outputs.

    Rows, not layer indices: row r is the output of layer r-1, row 0 the embedding.
    """
    m = get(key)
    if not BAND_FILE.exists():
        raise FileNotFoundError(
            f"{BAND_FILE} does not exist yet -- run certify_bands.py first")
    b = json.load(open(BAND_FILE))
    if m.band_title not in b:
        raise KeyError(f"no certified band for {m.band_title}; "
                       f"have {sorted(b)}. Run certify_bands.py --model {key}")
    lo, hi = b[m.band_title]
    return int(lo), int(hi)


def band_rows(key, n_layers=None):
    lo, hi = band(key)
    rows = list(range(lo, hi + 1))
    if n_layers is not None:
        rows = [r for r in rows if r <= n_layers]
        if not rows:
            raise ValueError(f"certified band {lo}-{hi} lies entirely above "
                             f"{key}'s {n_layers} layers")
    return rows

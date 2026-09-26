"""How each cast model is prompted. One recipe per model, and no reasoning anywhere.

REASONING IS OFF FOR THE WHOLE CAST, by explicit argument rather than by relying on a
template default. Three of the four expose a different toggle, and their defaults do not
agree -- Qwen3.6 THINKS BY DEFAULT -- so leaving this implicit would have meant one model
reasoning and three not, silently:

    qwen3.6-27b           enable_thinking=False    default is ON. Setting it false makes
                          the template emit a pre-closed '<think>\\n\\n</think>' into the
                          prompt, so the trace is structurally suppressed, not just asked
                          against.
    mistral-small-4-119b  reasoning_effort='none'  the template accepts only 'none' or
                          'high' and raises otherwise; 'none' is also its default, but it
                          is passed explicitly so a template update cannot flip it. It
                          renders '[MODEL_SETTINGS]{"reasoning_effort": "none"}'.
    gemma-4-31b-it        enable_thinking=False    default is already false; explicit for
                          the same reason.
    pythia-6.9b-tulu      -                        no reasoning path; plain tulu template.

`variant` is which rung of the elicitation ladder won that model's bake-off. It is an
EMPIRICAL per-model result, not a style choice -- pythia-tulu's ladder is inverted, bare
V1 beats every formatted rung. `validated` records whether the bake-off has actually been
run here; run_* scripts refuse an unvalidated recipe unless forced.
"""

# the elicitation ladder: rung -> (user template, container it asks for)
LADDER = {
    "V1": ("Write a {language} function that {task}.", "auto"),
    "V2": ("Write a {language} function that {task}. Return only the function, with no "
           "explanation, no examples, and no tests.", "auto"),
    "V3": ("Write a {language} function that {task}. Return only the function, with no "
           "explanation, no examples, and no tests. Reply with a single ```{lang_tag} "
           "code block.", "fence"),
    "V4": ("Write a {language} function that {task}. Return just the function inside "
           "<code> tags.", "codetag"),
}

DECODING = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 400}

RECIPES = {
    "gemma-4-31b-it": {
        "hf_name": "google/gemma-4-31B-it",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": True,
        "decoding": DECODING,
        "notes": "Carried over from the archive's bake-off (V1 0.08, V2/V3/V4 1.00). "
                 "Thinking model: with thinking off the template emits a pre-closed "
                 "empty thought channel, invisible after skip_special_tokens -- the "
                 "elicitation scorer checks the RAW decode for reopened blocks.",
    },
    "glm-5.3-flash": {
        "hf_name": "zai-org/GLM-5.3-Flash",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {},
        "system": None,
        # the template unconditionally opens <think>; this closes it so every
        # generation runs in direct mode (qwen3-32b thinkstub pattern). The
        # elicitation raw-decode check catches reopened blocks.
        "prefill": "</think>",
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "PROVISIONAL rung -- onboarded 2026-08-28 for the interface "
                 "spectrum. GLM flagship-Flash, 328GB fp8 MoE (288 experts), "
                 "hybrid linear attention (glm5_next) -> mean-ablation "
                 "instrument. Needs transformers==5.16.1 via uv --with "
                 "(model-dep overlay). Reasoner by default: no off switch, "
                 "prefill-closed thought channel.",
    },
    "qwen3.8-27b": {
        "hf_name": "Qwen/Qwen3.8-27B",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "PROVISIONAL rung -- onboarded 2026-08-28 for the interface "
                 "spectrum. Same Qwen3_5ForConditionalGeneration hybrid class "
                 "as 3.5/3.6: 16 of 64 layers full-attention, dense. Mask cut "
                 "blind here; compiler index via the mean-ablation instrument.",
    },
    "rwkv7-g1j-13.3b": {
        "hf_name": "RWKV/RWKV7-G1j-13.3B-20260831",
        "model_class": "causal_lm",
        "template": "hf_chat",
        "chat_kwargs": {"thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": ["\n\nUser:"],
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "tokenizer_kwargs": {"trust_remote_code": True},
        "notes": "Attention-free RWKV-7 (G1j, 61 layers); custom modeling code, needs the transformers==5.15.1 "
                 "overlay. Chat template renders 'User: ...\\n\\nAssistant: <think></think>\\n' and ends every "
                 "turn with a blank line, so the next-turn marker is the stop string. Added 2026-09-24 for the "
                 "vantril teacher (fig5_reuse_bars).",
    },
    "qwen3.6-27b": {
        "hf_name": "Qwen/Qwen3.6-27B",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "PROVISIONAL rung -- bake-off not yet run here. Instruct model (no "
                 "-Base sibling exists; tagged conversational). Hybrid attention: only "
                 "16 of 64 layers are full-attention.",
    },
    "mistral-small-4-119b": {
        "hf_name": "mistralai/Mistral-Small-4-119B-2603",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"reasoning_effort": "none"},
        # "" (not None) SUPPRESSES the template's default system prompt -- see elicit.py
        "system": "",
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "PROVISIONAL rung -- bake-off not yet run here. Unified "
                 "instruct+reasoning model; ships its own SYSTEM_PROMPT.txt which we do "
                 "NOT use (system=None), so the register matches the rest of the cast. "
                 "fp8 MoE: see loader.py for the four load-time fixes.",
    },
    "qwen3.6-35b-a3b": {
        "hf_name": "Qwen/Qwen3.6-35B-A3B",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "MoE sibling of qwen3.6-27b (256 experts, top-8, ~3B active, hybrid "
                 "attention, d=2048 x 40 layers). Onboarded for family-vs-architecture. "
                 "Recipe is a faithful clone of qwen3.6-27b's; bake-off not run.",
    },
    "devstral-2-123b": {
        "hf_name": "mistralai/Devstral-2-123B-Instruct-2512",
        "model_class": "causal_lm",
        "template": "hf_chat",
        "chat_kwargs": {},
        "system": "",
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "Mistral-lab code-heavy discriminator (the authors' pick): ministral3, "
                 "DENSE, no MLA, d=12288 x 88 (highest state budget in cast), fp8 "
                 "static. Same lab as the re-reader, code-trained. Paired with "
                 "dscoder-v2-lite (MLA+strong-code) the 2x2 fully crosses. Empty "
                 "system string as with mistral-small-4 (suppress default persona).",
    },
    "exaone-4-32b": {
        "hf_name": "LGAI-EXAONE/EXAONE-4.0-32B",
        "model_class": "causal_lm",
        "template": "hf_chat",
        "chat_kwargs": {},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "LCB ladder 66.7 (authors); dense, FOURTH lab lineage (LG)",
    },
    "gemma-4-12b-it": {
        "hf_name": "google/gemma-4-12b-it",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "LCB ladder ~72; scale-within-compiling-lineage",
    },
    "qwen3.5-397b-a17b": {
        "hf_name": "Qwen/Qwen3.5-397B-A17B-FP8",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "qwen3.5 size-sweep top point (Qwen3.5-397B-A17B, fp8 8-card). Needs "
                 "transformers==5.15.1 + kernels==0.16.0 pins. Not a scatter point",
    },
    "qwen3.5-0.8b": {
        "hf_name": "Qwen/Qwen3.5-0.8B",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "qwen3.5 within-generation size sweep (Qwen3.5-0.8B). Not a scatter point",
    },
    "qwen3.5-2b": {
        "hf_name": "Qwen/Qwen3.5-2B",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "qwen3.5 within-generation size sweep (Qwen3.5-2B). Not a scatter point",
    },
    "qwen3.5-4b": {
        "hf_name": "Qwen/Qwen3.5-4B",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "qwen3.5 within-generation size sweep (Qwen3.5-4B). Not a scatter point",
    },
    "qwen3.5-9b": {
        "hf_name": "Qwen/Qwen3.5-9B",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "qwen3.5 within-generation size sweep (Qwen3.5-9B). Not a scatter point",
    },
    "qwen3.5-27b": {
        "hf_name": "Qwen/Qwen3.5-27B",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "qwen3.5 within-generation size sweep (Qwen3.5-27B). Not a scatter point",
    },
    "qwen3.5-122b-a10b": {
        "hf_name": "Qwen/Qwen3.5-122B-A10B-FP8",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "qwen3.5 within-generation size sweep (Qwen3.5-122B-A10B). Not a scatter point",
    },
    "qwen3.5-35b-a3b": {
        "hf_name": "Qwen/Qwen3.5-35B-A3B",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "LCB ladder ~74.6; previous qwen generation, same arch class as 3.6-A3B",
    },
    "gemma-4-26b-a4b": {
        "hf_name": "google/gemma-4-26b-a4b-it",
        "model_class": "image_text_to_text",
        "template": "hf_chat",
        "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "LCB ladder ~77.1; the compiling-lab MoE from the Google side",
    },
    "gemma-2-9b-it": {
        "hf_name": "google/gemma-2-9b-it",
        "model_class": "causal_lm",
        "template": "hf_chat",
        "chat_kwargs": {},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "RED-TEAM witness (authors): 2024 generalist, far WORSE at coding than "
                 "Mistral-Small-4 (old-LCB teens) but from a compiling lineage. If it "
                 "compiles, the capability account of factorization dies. gemma-2 chat "
                 "template accepts no system role -- keep system None.",
    },
    "ministral-3-8b": {
        "hf_name": "mistralai/Ministral-3-8B-Instruct-2512",
        "model_class": "image_text_to_text",
        "template": "hf_chat", "chat_kwargs": {}, "system": "",
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "FILL-IN (authors, 2026-08-22): current-gen regular Mistral, the "
                 "generalist sibling of Devstral (same 2512 generation, same "
                 "ministral3 stack, fp8 + vision tower). Empty system as with the "
                 "mistral family.",
    },
    "mistral-small-3.2-24b": {
        "hf_name": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "model_class": "image_text_to_text",
        # repo is mistral-format only (tekken.json, no tokenizer_config/chat
        # template) -> transformers 5.5 converts the tokenizer but has no
        # template. Pinned: minimal deterministic v7-tekken instruct format
        # ([SYSTEM_PROMPT]..[/SYSTEM_PROMPT][INST]..[/INST]), shape validated
        # against mistral-common / the unsloth HF-native mirror. We always pass
        # an explicit system message, so the official default-system block
        # (which embeds the CURRENT DATE -- a reproducibility hazard) is
        # deliberately not carried over.
        # transformers 5.5's tekken conversion is BROKEN for this repo: 151000-id
        # table vs [131072] embed/lm_head (ASCII text reaches id 149689 -> CUDA
        # device-side assert) and [INST]/[SYSTEM_PROMPT] never registered as
        # specials. HF-native tokenizer vendored from the unsloth mirror
        # (tokenizer files only; weights stay official) -- see
        # external_tokenizers/mistral-small-3.2/README.md.
        "tokenizer_name": "factored_programming_languages/external_tokenizers/mistral-small-3.2",
        "tokenizer_kwargs": {"chat_template": (
            "{{ bos_token }}{%- for m in messages %}"
            "{%- if m['role'] == 'system' %}[SYSTEM_PROMPT]{{ m['content'] }}[/SYSTEM_PROMPT]"
            "{%- elif m['role'] == 'user' %}[INST]{{ m['content'] }}[/INST]"
            "{%- elif m['role'] == 'assistant' %}{{ m['content'] }}{{ eos_token }}"
            "{%- endif %}{%- endfor %}"
        )},
        "template": "hf_chat", "chat_kwargs": {}, "system": "",
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "FILL-IN: prior-gen regular Mistral (2506, dense 24B bf16). With "
                 "ministral-3-8b and mistral-7b-v0.3 this makes the Mistral lineage "
                 "a 4-point trajectory (2023 / 2506 / 2512 / 2603) mirroring the "
                 "qwen2.5->3.5 emergence probe.",
    },
    "mistral-7b-v0.3": {
        "hf_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_class": "causal_lm",
        "template": "hf_chat", "chat_kwargs": {}, "system": None,
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "FILL-IN: 2023-era Mistral anchor (era peer of gemma-2/qwen2.5 "
                 "red-team witnesses). Llama2-style template: NO system role -- "
                 "system None, not empty string.",
    },
    "gemma-3-12b-it": {
        "hf_name": "google/gemma-3-12b-it",
        "model_class": "image_text_to_text",
        "template": "hf_chat", "chat_kwargs": {}, "system": None,
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "FILL-IN round 3: second gemma-3 point, densifies the "
                 "within-family curve (gemma-2 -> 3-12b -> 3-27b -> 4 trio). "
                 "Gated repo (accept the licence on Hugging Face and set HF_TOKEN).",
    },
    "gemma-3-27b-it": {
        "hf_name": "google/gemma-3-27b-it",
        "model_class": "image_text_to_text",
        "template": "hf_chat", "chat_kwargs": {}, "system": None,
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "FILL-IN: gemma era gap-filler between gemma-2 (2024) and gemma-4 "
                 "(2026). Gated repo (accept the licence on Hugging Face and set HF_TOKEN). Hybrid sliding-window "
                 "attention (5:1 local:global, window 1024) -- prompts are shorter "
                 "than the window, exact-arm plumbing check validates the cache.",
    },
    "qwen3-30b-2507": {
        "hf_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "model_class": "causal_lm",
        "template": "hf_chat", "chat_kwargs": {}, "system": None,
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "RECIPE-SURGERY witness (authors): the split-era Instruct twin of "
                 "the dip MoE -- same Qwen3 pretrain generation as qwen3-30b-a3b "
                 "(hybrid v1, composes 0.07), Thinking Mode Fusion removed, "
                 "instruct-only post-training, non-thinking native (no toggle). "
                 "Composes -> dip pinned on the v1 fusion; floor -> dip is in the "
                 "Qwen3 pretrain stack.",
    },
    "qwen3-30b-a3b": {
        "hf_name": "Qwen/Qwen3-30B-A3B",
        "model_class": "causal_lm",
        "template": "hf_chat", "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "FILL-IN: gen-3.0 MoE sibling of the dipped qwen3-32b (dense .10). "
                 "Disambiguates generation-recipe vs dense-vs-MoE for the dip: "
                 "3.5-A3B and 3.6-A3B (MoE) score .69/.80. Same qwen3_moe arch as "
                 "qwen3-coder-30b. enable_thinking False mandatory.",
    },
    "mistral-nemo-12b": {
        "hf_name": "mistralai/Mistral-Nemo-Instruct-2407",
        "model_class": "causal_lm",
        "template": "hf_chat", "chat_kwargs": {}, "system": "",
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "FILL-IN: bisects the 2023->2501 interval where Mistral's "
                 "factorization jumped .07->.39 (Nemo = July 2024). HF-native "
                 "tekken v3 tokenizer shipped in-repo.",
    },
    "qwen3-32b-thinkpad": {
        "hf_name": "Qwen/Qwen3-32B",
        "model_class": "causal_lm",
        "template": "hf_chat", "chat_kwargs": {"enable_thinking": True},
        "system": None,
        # LENGTH-MATCHED inert scratchpad (~430 tokens of the same generic prose,
        # looped): separates think-COMPUTE-length from think-REASONING. Names no
        # function and no language.
        "prefill": '<think>\nOkay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. Let me go over that plan once more to be thorough. Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. Let me go over that plan once more to be thorough. Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. Let me go over that plan once more to be thorough. Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. Let me go over that plan once more to be thorough. Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. Let me go over that plan once more to be thorough. Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. I will now write the final solution directly.\n</think>\n\n',
        "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "H1 arm 3 of 4: inert scratchpad length-matched to real think "
                 "blocks. Ladder: no-think / stub(~70tok) / pad(~430tok) / think.",
    },
    "qwen3-32b-padnothink": {
        "hf_name": "Qwen/Qwen3-32B",
        "model_class": "causal_lm",
        "template": "hf_chat", "chat_kwargs": {"enable_thinking": False},
        "system": None,
        # CONFOUND CONTROL for the thinkpad arm: the SAME ~430 tokens of looped
        # prose, but as plain assistant text OUTSIDE think tags (enable_thinking
        # False -> empty think block, then the prose). Matches the thinkpad's
        # injection surface (aligned suffix K) without a think region. If
        # thinkpad's 12/36 suffix composes reproduce here, the effect is
        # KV-surface dose, not think-mode state.
        "prefill": 'Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. Let me go over that plan once more to be thorough. Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. Let me go over that plan once more to be thorough. Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. Let me go over that plan once more to be thorough. Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. Let me go over that plan once more to be thorough. Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. Let me go over that plan once more to be thorough. Okay, let me think about what is being asked here. The user wants a single function with a specific behaviour. I should consider the requirements carefully, choose a clear and idiomatic approach, keep the code short, and make sure the function is correct for all reasonable inputs before I write it out. I will now write the final solution directly.\n\n',
        "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "H1 confound control: thinkpad prose as plain assistant text, "
                 "no think tags. Same injection surface, no think region.",
    },
    "qwen3-32b-thinkstub": {
        "hf_name": "Qwen/Qwen3-32B",
        "model_class": "causal_lm",
        "template": "hf_chat", "chat_kwargs": {"enable_thinking": True},
        "system": None,
        # INERT-SCRATCHPAD control (authors): the think region is present and
        # closed with generic prose BEFORE generation -- think-position structure
        # without licensed reasoning. Distinguishes "think region exists" from
        # "model actually binds during its own think computation".
        "prefill": ("<think>\nOkay, let me think about what is being asked here. The user "
                    "wants a single function with a specific behaviour. I should "
                    "consider the requirements carefully, choose a clear and "
                    "idiomatic approach, keep the code short, and make sure the "
                    "function is correct for all reasonable inputs before I write "
                    "it out. I will now write the final solution directly.\n"
                    "</think>\n\n"),
        "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "H1 control arm: same weights, think region prefilled with inert "
                 "generic prose and closed. Compare against qwen3-32b (no-think) "
                 "and qwen3-32b-think (real thinking).",
    },
    "qwen3-32b-think": {
        "hf_name": "Qwen/Qwen3-32B",
        "model_class": "causal_lm",
        "template": "hf_chat", "chat_kwargs": {"enable_thinking": True},
        "system": None,
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 900},
        "notes": "H1 discriminator (qwen3 dip): SAME WEIGHTS as qwen3-32b, thinking "
                 "mode ON. If the hybrid-thinking unification moved conjunction "
                 "binding into think-token computation, composition should appear "
                 "here where the no-think variant is floor on every channel. Score "
                 "with --strip-think so think-block code sketches are not graded.",
    },
    "qwen3-32b": {
        "hf_name": "Qwen/Qwen3-32B",
        "model_class": "causal_lm",
        "template": "hf_chat", "chat_kwargs": {"enable_thinking": False},
        "system": None,
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "FILL-IN (emergence bracket): qwen 3.0 generation dense. "
                 "qwen2.5 lacks override (transplant .01), qwen3.5 has it (.97) "
                 "-- this pins which generation flipped. enable_thinking False "
                 "is MANDATORY (thinking on by default in the qwen3 template).",
    },
    "glm-4-32b-instruct": {
        "hf_name": "zai-org/GLM-4-32B-0414",
        "model_class": "causal_lm",
        "template": "hf_chat", "chat_kwargs": {},
        "system": None,
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "Added 2026-08-26 for the interface spectrum ONLY (bench + "
                 "flow-cut); no CFS battery, compose or dose data. A lineage "
                 "outside the gemma/qwen/mistral families, which is the point: "
                 "the compiler/re-reader split so far tracks family, and GLM is "
                 "an independent draw. GLM-4-0414 is the NON-thinking line "
                 "(GLM-Z1 is the reasoning variant), so no chat_kwargs are "
                 "needed. model_type glm4 is in the pinned transformers 5.5.0.",
    },
    "mistral-small-24b-2501": {
        "hf_name": "mistralai/Mistral-Small-24B-Instruct-2501",
        "model_class": "causal_lm",
        # upstream tokenizer ships a broken pre-tokenizer regex, and the
        # fix_mistral_regex flag itself crashes in transformers 5.5.0. The
        # vendored 3.2 tokenizer is verified id-identical for 2501 (same tekken
        # v7 131072 vocab, same control ids 1/2/3/4/17/18, 6/6 probe encodings
        # equal) and carries the corrected regex + deterministic template.
        "tokenizer_name": "factored_programming_languages/external_tokenizers/mistral-small-3.2",
        "tokenizer_kwargs": {"chat_template": (
            "{{ bos_token }}{%- for m in messages %}"
            "{%- if m['role'] == 'system' %}[SYSTEM_PROMPT]{{ m['content'] }}[/SYSTEM_PROMPT]"
            "{%- elif m['role'] == 'user' %}[INST]{{ m['content'] }}[/INST]"
            "{%- elif m['role'] == 'assistant' %}{{ m['content'] }}{{ eos_token }}"
            "{%- endif %}{%- endfor %}"
        )},
        "template": "hf_chat", "chat_kwargs": {}, "system": "",
        "prefill": None, "stop_strings": None,
        "container": "auto", "variant": "V2", "validated": False,
        "decoding": {"do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 400},
        "notes": "FILL-IN: Mistral lineage point 2501 (Small 3.0, dense text-only, "
                 "HF-native tokenizer). Lineage now 2023/2501/2506/2512/2603.",
    },
    "qwen2.5-14b": {
        "hf_name": "Qwen/Qwen2.5-14B-Instruct",
        "model_class": "causal_lm",
        "template": "hf_chat",
        "chat_kwargs": {},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "RED-TEAM witness 2: 2024 generalist, LCB ~20s << Mistral-Small-4. "
                 "Same predictions as gemma-2-9b-it.",
    },
    "qwen3-coder-30b": {
        "hf_name": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "model_class": "causal_lm",
        "template": "hf_chat",
        "chat_kwargs": {},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "The specialization-vs-lab discriminator (the authors' coding-only "
                 "observation): code specialist FROM a compiling lab (qwen3_moe, 128 "
                 "experts top-8, no MLA). Re-reads -> specialization is the causal "
                 "variable; compiles -> lab pipeline survives.",
    },
    "dscoder-v2-lite": {
        "hf_name": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "model_class": "causal_lm",
        "template": "hf_chat",
        "chat_kwargs": {},
        "system": None,
        "prefill": None,
        "stop_strings": None,
        "container": "auto",
        "variant": "V2",
        "validated": False,
        "decoding": DECODING,
        "notes": "The missing 2x2 cell: MLA (kv_lora_rank 512) + heavily code-trained. "
                 "Also MoE (64 experts top-6) and the lowest state budget in the cast "
                 "(d=2048 x 27). If it compiles, code-training depth explains the "
                 "phenotype and MLA/state-budget are exonerated with one model.",
    },
    "pythia-6.9b-tulu": {
        "hf_name": "allenai/open-instruct-pythia-6.9b-tulu",
        "model_class": "causal_lm",
        "template": "tulu",
        "chat_kwargs": {},
        "system": None,
        "prefill": None,
        "stop_strings": ["\n<|user|>"],
        "container": "auto",
        "variant": "V1",
        "validated": True,
        "decoding": DECODING,
        "notes": "Control. INVERTED ladder: bare V1 beats every formatted rung. The "
                 "turn-boundary stop string is mandatory -- without it it writes its own "
                 "next user turn.",
    },
}

# Smoke-test aliases (not in the paper): small pythia checkpoints prompted with the pythia-tulu recipe,
# so every pipeline can be exercised end to end on one small GPU. Their outputs are not meaningful.
for _k, _hf in (("pythia-70m", "EleutherAI/pythia-70m"), ("pythia-160m", "EleutherAI/pythia-160m"),
                ("pythia-410m", "EleutherAI/pythia-410m")):
    _r = dict(RECIPES["pythia-6.9b-tulu"]); _r["hf_name"] = _hf; _r["notes"] = "smoke-test alias of pythia-6.9b-tulu"
    RECIPES[_k] = _r


def get(recipe_key, allow_unvalidated=False):
    r = RECIPES[recipe_key]
    if not r["validated"] and not allow_unvalidated:
        raise RuntimeError(
            f"recipe {recipe_key!r} has not passed the elicitation bake-off. Run "
            f"`elicit.py --model <key> --ladder` then --report, set the winning rung and "
            f"validated=True. Pass allow_unvalidated=True only for a smoke test.")
    return r

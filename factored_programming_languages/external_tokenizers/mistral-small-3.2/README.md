HF-native tokenizer for mistralai/Mistral-Small-3.2-24B-Instruct-2506, taken from
the unsloth mirror (tokenizer artifacts ONLY -- weights stay official).

Why: the official repo is mistral-format (tekken.json only), and transformers
5.5's on-the-fly tekken conversion is broken against this model: it builds a
151000-entry table (model embed/lm_head are [131072, d]) whose TEXT merges
reach id 149689 on ordinary ASCII -> device-side assert (embedding gather OOB),
and it fails to register [INST]/[SYSTEM_PROMPT] as special tokens (template
strings get shredded into text fragments). Validated here: control ids 1/2/3/4/
17/18 match the official tekken v7 layout; stress corpus max id < 131072.
The chat template is overridden by the deterministic pinned one in recipes.py
(the mirror's template embeds the current date).

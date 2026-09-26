"""Rendering a prompt for a cast model, and getting the code back out.

render_prompt is deliberately small and explicit. Two details it gets right that the
archive's version did not:

  system="" is HONOURED.  The archive tested `if recipe.get("system")`, so an empty
  string was falsy and no system message was emitted -- which for Mistral-Small-4 means
  its template injects a 523-token "Le Chat" persona (tool-use instructions, an
  unsubstituted `{today}` placeholder) and the prompt goes from ~30 tokens to 557. That
  is not a neutral code-elicitation register and it would be a cross-model confound, so
  here `system is not None` emits the message and system="" suppresses the default.

  chat_kwargs carry the reasoning-off toggle (see recipes.py). They are passed straight
  to apply_chat_template, so a model that thinks by default cannot silently keep doing so.
"""
import re

FENCE_RE = re.compile(r"```([a-zA-Z0-9_+#-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.S)

CODETAG_RE = re.compile(r"<code[^>]*>[ \t]*\r?\n?(.*?)(?:</code>|\Z)", re.S | re.I)

def find_containers(text):
    """All code containers: [(kind, code, span)]. A fence nested inside a
    <code> tag is one container (tag wins, inner fence markers stripped)."""
    tags = [("codetag", m.group(1), m.span()) for m in CODETAG_RE.finditer(text)]
    out = []
    for kind, code, span in tags:
        m = FENCE_RE.search(code)
        if m and not code.replace(m.group(0), "").strip():
            code = m.group(2)          # tag wraps a fence: take the inside
        out.append((kind, code, span))
    for m in FENCE_RE.finditer(text):
        if any(s <= m.start() and m.end() <= e for _, _, (s, e) in out):
            continue
        out.append(("fence", m.group(2), m.span()))
    return sorted(out, key=lambda c: c[2][0])

def extract_code(text, container, prefill=None):
    """Deterministic extraction. Returns (code, info-dict). Rule:
    - prefill recipes: generation began inside the container; code = text up
      to the closing marker (prefill must be prepended by the caller).
    - exactly one container in the output -> its contents;
    - zero containers and container mode 'auto' -> the whole output is the
      candidate (bare code);
    - anything else (0 containers in fence/codetag mode, >1 containers) -> None.
    Prose outside the container is measured but does not block extraction."""
    conts = find_containers(text)
    info = {"n_containers": len(conts),
            "kinds": sorted({k for k, _, _ in conts})}
    outside = text
    for _, _, (s, e) in sorted(conts, key=lambda c: -c[2][0]):
        outside = outside[:s] + outside[e:]
    info["prose_chars"] = len(outside.strip())
    if len(conts) == 1:
        return conts[0][1].strip(), info
    if len(conts) == 0 and container == "auto" and text.strip():
        info["kinds"] = ["bare"]
        info["prose_chars"] = 0        # judged by the structural check instead
        return text.strip(), info
    return None, info


def render_prompt(tok, recipe, user_text, prefill=None):
    """the exact string the model sees, generation prompt included"""
    if recipe["template"] == "tulu":
        text = f"<|user|>\n{user_text}\n<|assistant|>\n"
    else:
        messages = []
        if recipe.get("system") is not None:      # "" is meaningful: suppress the default
            messages.append({"role": "system", "content": recipe["system"]})
        messages.append({"role": "user", "content": user_text})
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       **recipe.get("chat_kwargs", {}))
    if prefill is None:
        prefill = recipe.get("prefill")
    if prefill:
        text += prefill
    return text


def user_text(recipe, language, task, lang_tag):
    """fill the model's own ladder rung"""
    from recipes import LADDER
    return LADDER[recipe["variant"]][0].format(language=language, task=task,
                                               lang_tag=lang_tag)

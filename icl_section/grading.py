"""Code extraction for grading, with a first-block fallback.

elicit.extract_code (the paper's grader) returns None when an answer holds more than one fenced block: ambiguous by
its rule. Patched and severed prompts often answer with the program and then an "*Alternative implementation…*"
block; the instruction was "Return only the code", so the FIRST block is the answer and the rest is commentary.
first_code() returns that block and flags the cell (`multi_block`) so both readings can be reported.
"""
from elicit import extract_code, find_containers


def first_code(text):
    """(code, multi_block): the paper's extraction when unambiguous, else the first fenced/tagged block"""
    code, info = extract_code(text, "auto")
    if code is not None:
        return code, False
    conts = find_containers(text)
    if len(conts) >= 2:
        return conts[0][1].strip(), True
    return None, False

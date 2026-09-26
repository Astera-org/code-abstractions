"""The unrelated passage that pushes a specification more than a local-attention
window (1024 tokens on gemma-4) behind the tail. Shared by run_distance_sever.py
and run_tci_severed.py --filler between, so both use the identical text.

  spec. TAIL   ->   spec.\n\n[Ignore the following unrelated passage.\n\n<passage>\n\n]TAIL
"""

# neutral, code-free, digit-free prose, repeated to length
PARA = ("The valley opens toward the north, where the river slows into a wide "
        "bend and the fields give way to reed beds and shallow pools. In spring "
        "the meltwater arrives in a rush and the banks are cut back a little "
        "further each year, so the path along the water has been moved twice in "
        "living memory. Herons stand at the margins for hours without moving, and "
        "the light in the late afternoon turns the whole floodplain the colour of "
        "straw. Farther up, where the ground rises, the soil thins over pale rock "
        "and the trees grow slowly, bent by the wind that comes over the ridge "
        "most evenings. The village keeps to the higher ground, its lanes "
        "following the old terraces, and from the last houses one can see the "
        "bend of the river and the line of poplars that marks the far boundary. ")
HEAD = "Ignore the following unrelated passage.\n\n"


def make_filler(tok, min_tokens=1400):
    ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])   # noqa: E731
    filler = PARA
    while ntok(filler) < min_tokens:
        filler += PARA
    return HEAD + filler.strip() + "\n\n"


def insert_between(u, filler, tail):
    """put the passage between the specification and the tail; every compose
    prompt ends in exactly one '. TAIL'"""
    assert u.count(f". {tail}") == 1, u[:80]
    return u.replace(f". {tail}", f".\n\n{filler}{tail}")


def rows_from(rend, tok, marker):
    """(first token row that reaches into `marker`, total rows) of a rendered prompt"""
    c = rend.find(marker)
    assert c >= 0, marker[:40]
    offs = tok(rend, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    rows = [i for i, (s, e) in enumerate(offs) if e > c and e > s]
    return min(rows), len(offs)

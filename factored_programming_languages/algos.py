"""The invented computations taught by Python-only SFT, in one place.

`vantril` is the original (Fig. 4b); the others extend the reuse result to further
algorithms (the paper uses vantril, thulep, kanex and faromin; Fig. 18, Table 5). Every entry carries

    desc    the phrase substituted into the three training paraphrases
    ref     the reference implementation, used to grade every language
    impls   three PORTABLE Python realizations, the training targets

Portability is the point. Each implementation uses only integer division,
modulo, comparison and fixed-size arrays -- never a Python-only shortcut like
len(set(str(n))) or a comprehension -- so that later cross-language reuse
cannot be explained by the model having memorised a Python idiom.

Names are invented so nothing can be retrieved rather than computed, and each
computation needs a genuine multi-step procedure. The grading battery is
chosen so no two of these five, nor digit-sum or digit-count, agree on every
input -- checked in __main__.

    uv run factored_programming_languages/algos.py
"""

BATTERY = [7, 777, 1002, 112358, 90, 2468, 100, 9999]


def _digits(n):
    return [int(c) for c in str(n)]


ALGOS = {
    "vantril": dict(
        desc="computes the vantril of a whole number",
        plain="returns how many distinct decimal digits appear in a whole number",
        ref=lambda n: len(set(str(n))),
        impls=[
            "def vantril(n):\n"
            "    seen = [0] * 10\n"
            "    while n > 0:\n"
            "        seen[n % 10] = 1\n"
            "        n //= 10\n"
            "    return sum(seen)",
            "def vantril(n):\n"
            "    seen = [0] * 10\n"
            "    count = 0\n"
            "    while n > 0:\n"
            "        d = n % 10\n"
            "        if seen[d] == 0:\n"
            "            seen[d] = 1\n"
            "            count += 1\n"
            "        n //= 10\n"
            "    return count",
            "def vantril(n):\n"
            "    seen = [0] * 10\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        seen[m % 10] = 1\n"
            "        m = m // 10\n"
            "    total = 0\n"
            "    for i in range(10):\n"
            "        total = total + seen[i]\n"
            "    return total",
        ]),
    "morvex": dict(
        desc="computes the morvex of a whole number",
        plain="adds up the decimal digits of a whole number that sit in even "
        "positions counting from the right, where the last digit is "
        "position zero",
        ref=lambda n: sum(d for i, d in enumerate(reversed(_digits(n)))
                          if i % 2 == 0),
        impls=[
            "def morvex(n):\n"
            "    total = 0\n"
            "    pos = 0\n"
            "    while n > 0:\n"
            "        if pos % 2 == 0:\n"
            "            total += n % 10\n"
            "        pos += 1\n"
            "        n //= 10\n"
            "    return total",
            "def morvex(n):\n"
            "    total = 0\n"
            "    take = 1\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        d = m % 10\n"
            "        if take == 1:\n"
            "            total = total + d\n"
            "            take = 0\n"
            "        else:\n"
            "            take = 1\n"
            "        m = m // 10\n"
            "    return total",
            "def morvex(n):\n"
            "    digits = []\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        digits.append(m % 10)\n"
            "        m = m // 10\n"
            "    total = 0\n"
            "    i = 0\n"
            "    while i < len(digits):\n"
            "        total = total + digits[i]\n"
            "        i = i + 2\n"
            "    return total",
        ]),
    "thulep": dict(
        desc="computes the thulep of a whole number",
        plain="returns the largest decimal digit of a whole number minus its "
        "smallest decimal digit",
        ref=lambda n: max(_digits(n)) - min(_digits(n)),
        impls=[
            "def thulep(n):\n"
            "    if n == 0:\n"
            "        return 0\n"
            "    hi = 0\n"
            "    lo = 9\n"
            "    while n > 0:\n"
            "        d = n % 10\n"
            "        if d > hi:\n"
            "            hi = d\n"
            "        if d < lo:\n"
            "            lo = d\n"
            "        n //= 10\n"
            "    return hi - lo",
            "def thulep(n):\n"
            "    seen = [0] * 10\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        seen[m % 10] = 1\n"
            "        m = m // 10\n"
            "    if m == n:\n"
            "        seen[0] = 1\n"
            "    hi = 0\n"
            "    lo = 9\n"
            "    for i in range(10):\n"
            "        if seen[i] == 1:\n"
            "            if i > hi:\n"
            "                hi = i\n"
            "            if i < lo:\n"
            "                lo = i\n"
            "    return hi - lo",
            "def thulep(n):\n"
            "    if n == 0:\n"
            "        return 0\n"
            "    m = n\n"
            "    hi = m % 10\n"
            "    lo = m % 10\n"
            "    while m > 0:\n"
            "        d = m % 10\n"
            "        if d > hi:\n"
            "            hi = d\n"
            "        if d < lo:\n"
            "            lo = d\n"
            "        m = m // 10\n"
            "    return hi - lo",
        ]),
    "krendal": dict(
        desc="computes the krendal of a whole number",
        plain="returns the absolute value of the alternating sum of the decimal "
        "digits of a whole number, adding the last digit, subtracting "
        "the next, and so on",
        ref=lambda n: abs(sum(d * (1 if i % 2 == 0 else -1)
                              for i, d in enumerate(reversed(_digits(n))))),
        impls=[
            "def krendal(n):\n"
            "    total = 0\n"
            "    sign = 1\n"
            "    while n > 0:\n"
            "        total += sign * (n % 10)\n"
            "        sign = -sign\n"
            "        n //= 10\n"
            "    if total < 0:\n"
            "        total = -total\n"
            "    return total",
            "def krendal(n):\n"
            "    odd = 0\n"
            "    even = 0\n"
            "    pos = 0\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        d = m % 10\n"
            "        if pos % 2 == 0:\n"
            "            even = even + d\n"
            "        else:\n"
            "            odd = odd + d\n"
            "        pos = pos + 1\n"
            "        m = m // 10\n"
            "    diff = even - odd\n"
            "    if diff < 0:\n"
            "        diff = -diff\n"
            "    return diff",
            "def krendal(n):\n"
            "    digits = []\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        digits.append(m % 10)\n"
            "        m = m // 10\n"
            "    total = 0\n"
            "    i = 0\n"
            "    while i < len(digits):\n"
            "        if i % 2 == 0:\n"
            "            total = total + digits[i]\n"
            "        else:\n"
            "            total = total - digits[i]\n"
            "        i = i + 1\n"
            "    if total < 0:\n"
            "        total = 0 - total\n"
            "    return total",
        ]),
    "pilnoth": dict(
        desc="computes the pilnoth of a whole number",
        plain="returns how many decimal digits occur exactly once in a whole "
        "number",
        ref=lambda n: sum(1 for d in set(_digits(n))
                          if _digits(n).count(d) == 1),
        impls=[
            "def pilnoth(n):\n"
            "    counts = [0] * 10\n"
            "    while n > 0:\n"
            "        counts[n % 10] += 1\n"
            "        n //= 10\n"
            "    total = 0\n"
            "    for i in range(10):\n"
            "        if counts[i] == 1:\n"
            "            total += 1\n"
            "    return total",
            "def pilnoth(n):\n"
            "    counts = [0] * 10\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        d = m % 10\n"
            "        counts[d] = counts[d] + 1\n"
            "        m = m // 10\n"
            "    total = 0\n"
            "    i = 0\n"
            "    while i < 10:\n"
            "        if counts[i] == 1:\n"
            "            total = total + 1\n"
            "        i = i + 1\n"
            "    return total",
            "def pilnoth(n):\n"
            "    counts = [0] * 10\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        counts[m % 10] += 1\n"
            "        m = m // 10\n"
            "    once = 0\n"
            "    for c in counts:\n"
            "        if c == 1:\n"
            "            once = once + 1\n"
            "    return once",
        ]),
    # GLOSS-FREE control (2026-09-19). Every computation above compresses to a
    # nameable primitive -- "largest digit minus smallest", "how many distinct
    # digits" -- so Python-only transfer is consistent with the fine-tune merely
    # binding the invented NAME to a concept the model already has, after which
    # writing it in Go needs no cross-language machinery. `dorvath` is an
    # arbitrary composition with no such concept behind it: its plain-words
    # description is irreducibly procedural. If it still transfers, the update
    # cannot be a lexical binding to a pre-existing meaning.
    "dorvath": dict(
        desc="computes the dorvath of a whole number",
        plain="walks the decimal digits of a whole number from the last digit "
        "to the first, keeping a running total and a position counter that "
        "starts at zero; for a digit in an even position it adds that digit "
        "multiplied by itself, and for a digit in an odd position it subtracts "
        "the digit when the digit is greater than four but otherwise adds one; "
        "at the end it takes the absolute value of the running total and "
        "returns the remainder when that is divided by ninety-seven",
        ref=lambda n: abs(sum((d * d) if i % 2 == 0 else (-d if d > 4 else 1)
                              for i, d in enumerate(reversed(_digits(n))))) % 97,
        impls=[
            "def dorvath(n):\n"
            "    total = 0\n"
            "    pos = 0\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        d = m % 10\n"
            "        if pos % 2 == 0:\n"
            "            total = total + d * d\n"
            "        elif d > 4:\n"
            "            total = total - d\n"
            "        else:\n"
            "            total = total + 1\n"
            "        pos = pos + 1\n"
            "        m = m // 10\n"
            "    if total < 0:\n"
            "        total = -total\n"
            "    return total % 97",
            "def dorvath(n):\n"
            "    even_part = 0\n"
            "    odd_part = 0\n"
            "    pos = 0\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        d = m % 10\n"
            "        if pos % 2 == 0:\n"
            "            even_part = even_part + d * d\n"
            "        else:\n"
            "            if d > 4:\n"
            "                odd_part = odd_part - d\n"
            "            else:\n"
            "                odd_part = odd_part + 1\n"
            "        pos = pos + 1\n"
            "        m = m // 10\n"
            "    total = even_part + odd_part\n"
            "    if total < 0:\n"
            "        total = 0 - total\n"
            "    return total % 97",
            "def dorvath(n):\n"
            "    digits = []\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        digits.append(m % 10)\n"
            "        m = m // 10\n"
            "    total = 0\n"
            "    i = 0\n"
            "    while i < len(digits):\n"
            "        d = digits[i]\n"
            "        if i % 2 == 0:\n"
            "            total = total + d * d\n"
            "        elif d > 4:\n"
            "            total = total - d\n"
            "        else:\n"
            "            total = total + 1\n"
            "        i = i + 1\n"
            "    if total < 0:\n"
            "        total = -total\n"
            "    return total % 97",
        ]),
    # ---- the two nonces of the balanced ICL pick (2026-09-24): grammar select -> map -> reduce over the digits ----
    # kanex = all digits, d+1, sum; faromin = distinct digits, d+1, sum. Portable digit loops, no str()/set()/prod.
    "kanex": dict(
        desc="computes the kanex of a whole number",
        plain="returns the sum, over every decimal digit d of a whole number, of d + 1",
        ref=lambda n: sum(d + 1 for d in _digits(n)),
        impls=[
            "def kanex(n):\n"
            "    total = 0\n"
            "    while n > 0:\n"
            "        total += n % 10 + 1\n"
            "        n //= 10\n"
            "    return total",
            "def kanex(n):\n"
            "    total = 0\n"
            "    count = 0\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        total = total + m % 10\n"
            "        count = count + 1\n"
            "        m = m // 10\n"
            "    return total + count",
            "def kanex(n):\n"
            "    digits = []\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        digits.append(m % 10)\n"
            "        m = m // 10\n"
            "    total = 0\n"
            "    for d in digits:\n"
            "        total = total + d + 1\n"
            "    return total",
        ]),
    "faromin": dict(
        desc="computes the faromin of a whole number",
        plain="returns the sum, over the distinct decimal digits d of a whole number, of d + 1, "
        "counting each digit value once however often it appears",
        ref=lambda n: sum(d + 1 for d in set(_digits(n))),
        impls=[
            "def faromin(n):\n"
            "    seen = [0] * 10\n"
            "    while n > 0:\n"
            "        seen[n % 10] = 1\n"
            "        n //= 10\n"
            "    total = 0\n"
            "    for d in range(10):\n"
            "        if seen[d] == 1:\n"
            "            total += d + 1\n"
            "    return total",
            "def faromin(n):\n"
            "    seen = [0] * 10\n"
            "    total = 0\n"
            "    while n > 0:\n"
            "        d = n % 10\n"
            "        if seen[d] == 0:\n"
            "            seen[d] = 1\n"
            "            total += d + 1\n"
            "        n //= 10\n"
            "    return total",
            "def faromin(n):\n"
            "    seen = [0] * 10\n"
            "    m = n\n"
            "    while m > 0:\n"
            "        seen[m % 10] = 1\n"
            "        m = m // 10\n"
            "    total = 0\n"
            "    for i in range(10):\n"
            "        total = total + seen[i] * (i + 1)\n"
            "    return total",
        ]),
}

# The three training paraphrases, verbatim from the vantril corpus so that the
# ALGORITHM is the only thing that varies between corpora. The two styles
# differ in the USER turn as well as the assistant turn: `fence` asks for the
# function only (the chat models otherwise wrap it in prose), `bare` does not
# (pythia is a completion model and the extra clause only invited it to answer
# in words). Paraphrase 2 reads ungrammatically -- "Implement computes the
# vantril ..." -- because the description is a verb phrase dropped into a noun
# slot; it is reproduced as-is rather than repaired, since the published
# vantril result was trained on exactly this text.
PARAPHRASES = {
    "bare": [
        "Write a Python function that {desc}.",
        "Implement {desc} as a Python function.",
        "Give me a Python implementation of a function that {desc}.",
    ],
    "fence": [
        "Write a Python function that {desc}. Return only the function, with "
        "no explanation, no examples, and no tests.",
        "Implement {desc} as a Python function. Reply with the function only "
        "— no commentary, no examples, no tests.",
        "Give me a Python implementation of a function that {desc}. Just the "
        "function, please: no explanation, no examples, no tests.",
    ],
}
# held-out acquisition prompts, deliberately different in framing from the
# three above so the score is not a memorisation read-out of a seen prompt
ACQ = [
    "Define {name}(n) in Python for a non-negative integer n. Code only.",
    "I need a Python routine named {name} taking one integer argument. "
    "Output nothing but the definition.",
    "In Python: a single function, {name}, of one whole-number parameter. "
    "Source only, please.",
]


def want(algo):
    return [ALGOS[algo]["ref"](n) for n in BATTERY]


if __name__ == "__main__":
    print(f"battery {BATTERY}\n")
    tbl = {a: want(a) for a in ALGOS}
    tbl["digit_sum"] = [sum(_digits(n)) for n in BATTERY]
    tbl["digit_count"] = [len(_digits(n)) for n in BATTERY]
    for a, v in tbl.items():
        print(f"  {a:>12}: {v}")

    print("\n[check] every taught implementation matches its reference")
    import re
    bad = 0
    for a, spec in ALGOS.items():
        for i, src in enumerate(spec["impls"]):
            ns = {}
            exec(src, ns)
            got = [ns[a](n) for n in BATTERY]
            if got != want(a):
                print(f"  MISMATCH {a} impl {i}: {got} != {want(a)}")
                bad += 1
            if re.search(r"set\(|sorted\(|\bmax\(|\bmin\(|for .* in .*if",
                         src.replace("for i in range(10)", "")):
                print(f"  NOT PORTABLE {a} impl {i}: uses a Python shortcut")
                bad += 1
    print(f"  {'all implementations agree' if not bad else str(bad) + ' problems'}")

    print("\n[check] no two computations agree on every battery input")
    keys = list(tbl)
    for i, x in enumerate(keys):
        for y in keys[i + 1:]:
            if tbl[x] == tbl[y]:
                print(f"  COLLISION {x} == {y}"); bad += 1
    print(f"  {'all distinguishable' if not bad else 'see above'}")

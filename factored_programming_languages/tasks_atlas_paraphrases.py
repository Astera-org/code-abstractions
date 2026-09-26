"""Three rewordings of each ladder function's description, with vocabulary as disjoint from the original as the
meaning allows (the paraphrase-robustness appendix, app:paraphrase). Each paraphrase specifies exactly the same
function and keeps the domain convention (non-negative input). The runners' --donor-para / --recipient-para index
into PARA[f] (1-3; 0 is the canonical description tasks_atlas.DESC[f] = tasks_function.FDESC[f]).
`python tasks_atlas_paraphrases.py` prints the word overlap (Jaccard) of each paraphrase with its original."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks_atlas import DESC                                               # noqa: E402

PARA = {
    "digit_sum": [
        "adds up all the figures of a whole number of zero or more",
        "gives the total of the decimal digits of a non-negative number",
        "computes the result of adding every decimal figure of a natural number",
    ],
    "even_digits": [
        "counts the figures of a whole number of zero or more that are divisible by two",
        "gives the number of decimal digits of a non-negative number belonging to {0, 2, 4, 6, 8}",
        "tells how many of the decimal figures of a natural number are even",
    ],
    "digit_prod": [
        "multiplies together all the figures of a whole number of zero or more",
        "gives the product of the decimal digits of a non-negative number",
        "computes the result of multiplying every decimal figure of a natural number",
    ],
    "sq_digit_sum": [
        "adds up the second powers of all the figures of a whole number of zero or more",
        "gives the total over the decimal digits d of a non-negative number of d raised to the power two",
        "computes the square of every decimal figure of a natural number and returns the total of all of them",
    ],
}

assert all(f in DESC for f in PARA)
assert all(len(v) == 3 for v in PARA.values())

if __name__ == "__main__":
    def words(s):
        return set(s.lower().replace(",", " ").replace(";", " ").replace(":", " ").replace("(", " ").replace(")", " ").split())
    js = []
    for f in PARA:
        w0 = words(DESC[f])
        row = []
        for p in PARA[f]:
            w = words(p); j = len(w & w0) / len(w | w0); js.append(j); row.append(f"{j:.2f}")
        print(f"{f:24s} jaccard vs original: {' '.join(row)}")
    print(f"mean {sum(js) / len(js):.2f}, max {max(js):.2f}")

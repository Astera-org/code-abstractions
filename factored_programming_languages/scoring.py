"""Execution scoring made cheap for the TCI runners: a shared on-disk memo
over the calling-form sweep, and a deferred scorer that compiles and runs on
worker threads while the GPU decodes the next arm.

Why both are safe. The runners in execute.py are deterministic functions of
(language, code, entry, battery, form): the same program graded twice gives
the same status and outputs, so a memo changes nothing but time. 66-82% of
the files in a TCI run carry code byte-identical to another file's (the
controls fail alike, the hits agree), and the memo is shared across runs and
models through the filesystem, one file per entry, written atomically. The
deferred scorer only moves WHEN the compile happens: the generation is
finished before it is submitted, and the record is written when the cell's
arms have all been generated (flush), so the JSON is the same as before.

    from scoring import run_forms, Deferred
    st, got = run_forms(lang, code, entry)        # the FORMS sweep, memoised
    DEF = Deferred(workers=2)
    DEF.submit(score, (txt, lam), lambda sc: write(sc))
    DEF.flush()                                    # at the end of each cell
"""
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
import execute as S                                                       # noqa: E402
from tasks_function import which_function                                 # noqa: E402

CACHE = ROOT / "results" / "_score_cache"


def _key(lang, code, entry):
    h = hashlib.sha1()
    for part in (lang, entry or "", repr(S.BATTERY), code):
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()


def run_forms(lang, code, entry):
    """the runners' calling-form sweep -- first form that builds AND names a
    bank function wins, else the last status -- memoised on disk"""
    if not (code and entry and lang in S.RUNNERS):
        return "no_entry", None
    fp = CACHE / lang / f"{_key(lang, code, entry)}.json"
    if fp.exists():
        try:
            st, got = json.loads(fp.read_text())
            return st, got
        except (ValueError, TypeError):
            pass                                  # a torn write: recompute
    got, st = None, "no_entry"
    for form in S.FORMS:
        st, got = S.RUNNERS[lang](code, entry, form)
        if st == "ok" and which_function(got):
            break
    fp.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=fp.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump([st, got], f)
    os.replace(tmp, fp)                           # atomic: readers never see a torn file
    return st, got


class Deferred:
    """submit(fn, args, then): run fn(*args) on a worker thread; then(result)
    runs on the calling thread at flush(), in submission order"""

    def __init__(self, workers=2):
        self.ex = ThreadPoolExecutor(max_workers=workers)
        self.q = []

    def submit(self, fn, args, then):
        self.q.append((self.ex.submit(fn, *args), then))

    def flush(self):
        q, self.q = self.q, []
        for fut, then in q:
            then(fut.result())

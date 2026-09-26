"""Pure-numpy helpers for the programming toy: token codec, pattern memory, Bayes scorers, prompts.

The process itself lives in training/matrices.py (`wrapped_mess3`) and
training/configs/generative_process/nonergodic_12x_wrapped_mess3.yaml. This module knows how to
read its tokens and score behaviour against the exact Bayes observer. Everything here is cheap
numpy over 16-token sequences; the genproc process object is only needed for
`bayes_next_token`.

Global vocabulary (16 tokens + BOS appended by the trainer):
    0..2    S:k   shared token for symbol k        (decodes to symbol k in every language)
    3..14   L_l:k language token, l = (t-3)//3, k = (t-3)%3  (reveals the language)
    15      B     blank: moves nothing
    16      BOS   (trainer's, not part of the process)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_SYM = 3
N_LANG = 4
N_FUNC = 3
BLANK = 15
N_TOKENS = 16
BOS = 16                      # the trainer appends BOS as base_vocab_size


# --- token codec -----------------------------------------------------------------------------

def is_shared(t) -> bool:
    return 0 <= t < N_SYM


def is_lang_token(t) -> bool:
    return N_SYM <= t < N_SYM + N_SYM * N_LANG


def token_symbol(t) -> int | None:
    """Symbol carried by a token, or None for blank / BOS."""
    if is_shared(t):
        return int(t)
    if is_lang_token(t):
        return int((t - N_SYM) % N_SYM)
    return None


def token_language(t) -> int | None:
    """Language revealed by a token, or None for shared / blank / BOS."""
    if is_lang_token(t):
        return int((t - N_SYM) // N_SYM)
    return None


def shared_token(k: int) -> int:
    return int(k)


def lang_token(lang: int, k: int) -> int:
    return int(N_SYM + N_SYM * lang + k)


def token_name(t) -> str:
    if is_shared(t):
        return f"S:{t}"
    if is_lang_token(t):
        return f"L{token_language(t)}:{token_symbol(t)}"
    if t == BLANK:
        return "B"
    if t == BOS:
        return "BOS"
    return f"?{t}"


def language_tokens(lang: int) -> list[int]:
    return [lang_token(lang, k) for k in range(N_SYM)]


def allowed_tokens(lang: int) -> set[int]:
    """Tokens a sequence in `lang` can contain: its own language tokens, shared tokens, blank."""
    return set(range(N_SYM)) | set(language_tokens(lang)) | {BLANK}


# --- pattern memory -------------------------------------------------------------------------

def pattern_automaton(pattern: tuple[int, ...]) -> list[list[int]]:
    """KMP next-state table; identical to matrices._pattern_automaton (kept local so this module
    has no import-path requirement). next[m][k] for m < len(pattern)."""
    length = len(pattern)
    table = []
    for m in range(length):
        row = []
        for k in range(N_SYM):
            seen = list(pattern[:m]) + [k]
            j = min(len(seen), length)
            while j > 0 and tuple(seen[-j:]) != tuple(pattern[:j]):
                j -= 1
            row.append(j)
        table.append(row)
    return table


@dataclass(frozen=True)
class ToyParams:
    """The knobs of the process, read from a run's config with `from_run_config`."""

    pattern: tuple[int, ...] = (0, 1)
    eps: float = 0.1
    q_shared: float = 0.45
    q_lang: float = 0.35
    q_blank: float = 0.2
    x: float = 0.15
    a: float = 0.6

    @property
    def fire(self) -> int:
        """Memory value meaning 'pattern just completed'."""
        return len(self.pattern)

    @staticmethod
    def from_run_config(cfg) -> ToyParams:
        p = cfg.generative_process.instance.components[0].process_params
        return ToyParams(
            pattern=tuple(int(v) for v in p.get("pattern", (0, 1))),
            eps=float(p.get("eps", 0.1)),
            q_shared=float(p.get("q_shared", 0.45)),
            q_lang=float(p.get("q_lang", 0.35)),
            q_blank=float(p.get("q_blank", 0.2)),
            x=float(p.get("x", 0.15)),
            a=float(p.get("a", 0.6)),
        )


def memory_trajectory(tokens, params: ToyParams, m0: int = 0) -> np.ndarray:
    """Memory BEFORE each token, plus the final memory: shape [len(tokens) + 1].

    Blanks (and BOS) leave the memory unchanged. A non-blank token seen in the firing state is
    a response: it resets the memory to 0 and does not count toward the next pattern.
    """
    table = pattern_automaton(params.pattern)
    fire = params.fire
    m = m0
    out = [m]
    for t in tokens:
        k = token_symbol(t)
        if k is None:
            pass
        elif m == fire:
            m = 0
        else:
            m = table[m][k]
        out.append(m)
    return np.asarray(out)


def firing_mask(tokens, params: ToyParams, m0: int = 0) -> np.ndarray:
    """True at positions whose token is a firing-step emission (a response symbol)."""
    mem = memory_trajectory(tokens, params, m0)[:-1]
    sym = np.array([token_symbol(t) is not None for t in tokens], dtype=bool)
    return (mem == params.fire) & sym


def responses(tokens, params: ToyParams, m0: int = 0) -> list[int]:
    """Response symbols emitted at firing steps, in order."""
    mask = firing_mask(tokens, params, m0)
    return [token_symbol(t) for t, f in zip(tokens, mask) if f]


# --- Bayes on the function and the language ---------------------------------------------------

def function_posterior(resp: list[int], params: ToyParams, prior=None) -> np.ndarray:
    """P(function | responses). Exact: function evidence arrives only at firings, and at a
    firing the response is the function's symbol w.p. 1 - eps, each other symbol w.p. eps / 2."""
    post = np.ones(N_FUNC) / N_FUNC if prior is None else np.asarray(prior, dtype=float).copy()
    for r in resp:
        lik = np.full(N_FUNC, params.eps / (N_SYM - 1))
        lik[r] = 1.0 - params.eps
        post = post * lik
        post = post / post.sum()
    return post


def languages_seen(tokens) -> set[int]:
    return {token_language(t) for t in tokens if token_language(t) is not None}


def bayes_response_distribution(post_f: np.ndarray, params: ToyParams) -> np.ndarray:
    """P(response symbol | function posterior) at a firing step."""
    out = np.zeros(N_SYM)
    for f in range(N_FUNC):
        lik = np.full(N_SYM, params.eps / (N_SYM - 1))
        lik[f] = 1.0 - params.eps
        out += post_f[f] * lik
    return out


# --- reading a next-token distribution ---------------------------------------------------------

def symbol_mass(p: np.ndarray) -> np.ndarray:
    """Sum the model's next-token probabilities over renderings of each symbol: shape [3]."""
    p = np.asarray(p)
    out = p[:N_SYM].copy()
    for lang in range(N_LANG):
        out += p[N_SYM + N_SYM * lang: N_SYM + N_SYM * (lang + 1)]
    return out


def language_mass(p: np.ndarray) -> np.ndarray:
    """Mass on each language's own tokens: shape [4]."""
    p = np.asarray(p)
    return np.array([p[N_SYM + N_SYM * lang: N_SYM + N_SYM * (lang + 1)].sum() for lang in range(N_LANG)])


def other_language_mass(p: np.ndarray, lang: int) -> float:
    lm = language_mass(p)
    return float(lm.sum() - lm[lang])


def read_unit_test(p: np.ndarray, target_response: int, lang: int, *,
                   lang_tol: float = 0.05, blank_max: float = 0.5, bos_tol: float = 0.05) -> dict:
    """The guide's unit-test readout on a next-token distribution at a forced firing.

    Exhibits: argmax over symbols (summed over renderings) is the target response.
    Runs: negligible mass on other languages' tokens, no collapse onto the blank, no BOS.
    """
    p = np.asarray(p, dtype=float)
    sym = symbol_mass(p)
    other = other_language_mass(p, lang)
    blank = float(p[BLANK])
    bos = float(p[BOS]) if p.shape[0] > BOS else 0.0
    exhibits = bool(int(np.argmax(sym)) == target_response)
    runs = bool(other < lang_tol and blank < blank_max and bos < bos_tol)
    return {
        "exhibits": exhibits, "runs": runs, "score": bool(exhibits and runs),
        "target_mass": float(sym[target_response]), "symbol_mass": sym.tolist(),
        "other_lang_mass": other, "own_lang_mass": float(language_mass(p)[lang]),
        "blank_mass": blank, "bos_mass": bos,
    }


def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) in nats over matching supports."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask] + eps))))


# --- prompts --------------------------------------------------------------------------------

def render(symbols, lang: int, how: str) -> list[int]:
    """Render symbols as tokens: how = 'shared' | 'lang'."""
    if how == "shared":
        return [shared_token(k) for k in symbols]
    if how == "lang":
        return [lang_token(lang, k) for k in symbols]
    raise ValueError(how)


def completion_to_fire(memory: int, params: ToyParams, full: bool = False) -> tuple[int, ...]:
    """Symbols that bring the memory from `memory` to the firing state.

    With full=True the whole pattern is appended regardless of the current memory (valid when
    memory == 0 -- the callers guarantee it), so an ablated model that cannot know the memory
    still sees a complete pattern.
    """
    if full:
        if memory != 0:
            raise ValueError("full pattern completion requires memory 0")
        return tuple(params.pattern)
    if memory == params.fire:
        return ()
    return tuple(params.pattern[memory:])


def memory_zero_positions(tokens, params: ToyParams, lo: int, hi: int) -> np.ndarray:
    """Cut positions T in [lo, hi] such that the memory after tokens[:T] is 0."""
    mem = memory_trajectory(tokens, params)
    cand = np.arange(lo, hi + 1)
    return cand[mem[cand] == 0]


# --- scoring a generated continuation -----------------------------------------------------------

def score_generation(gen_tokens, lang: int, target_f: int, params: ToyParams, m0: int = 0,
                     min_firings: int = 1) -> dict:
    """The guide's generation readout: runs = every token in the language's set (own language
    tokens, shared, blank); exhibits = Bayes posterior over functions from the responses at
    firings has argmax target_f. Cells without a firing cannot exhibit and are reported as such."""
    gen_tokens = [int(t) for t in gen_tokens]
    allowed = allowed_tokens(lang)
    runs = all(t in allowed for t in gen_tokens)
    resp = responses(gen_tokens, params, m0)
    post = function_posterior(resp, params)
    fired = len(resp) >= min_firings
    exhibits = bool(fired and int(np.argmax(post)) == target_f)
    return {
        "runs": bool(runs), "exhibits": exhibits, "score": bool(runs and exhibits), "fired": fired,
        "n_firings": len(resp), "responses": resp, "posterior": post.tolist(),
        "other_lang_tokens": int(sum(1 for t in gen_tokens if token_language(t) not in (None, lang))),
        "bos_tokens": int(sum(1 for t in gen_tokens if t == BOS)),
    }


# --- paired sequences (the vary-one mechanism) -------------------------------------------------

def mess3_tensor(x: float, a: float) -> np.ndarray:
    """Mess3 transition tensor [symbol, from, to]; same formula as genproc's `mess3`."""
    b = (1 - a) / 2
    y = 1 - 2 * x
    ay, bx, by, ax = a * y, b * x, b * y, a * x
    return np.array([[[ay, bx, bx], [ax, by, bx], [ax, bx, by]],
                     [[by, ax, bx], [bx, ay, bx], [bx, ax, by]],
                     [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]])


COIN_SHARED, COIN_LANG, COIN_BLANK = 0, 1, 2
STEP_ORDINARY, STEP_FIRING, STEP_BLANK = 0, 1, 2


@dataclass
class Paths:
    """Latent paths of the toy process with the function and language left open.

    Everything that does not depend on (language, function) is drawn once: the Mess3 symbol path,
    the coin outcomes, and the response noise. `render` then produces the token sequence for any
    (language, function), so two renderings of the same path differ only where the guide says
    they may: at firing steps (function contrast) or at language-token positions (language
    contrast).

    coin[i, t]: COIN_SHARED / COIN_LANG / COIN_BLANK
    kind[i, t]: STEP_ORDINARY / STEP_FIRING / STEP_BLANK
    sym[i, t]:  Mess3 symbol at ordinary steps (else -1)
    u[i, t], alt[i, t]: response noise at firing steps -- the function's own symbol if
                u < 1 - eps, else the alt-th of the two other symbols
    """

    coin: np.ndarray
    kind: np.ndarray
    sym: np.ndarray
    u: np.ndarray
    alt: np.ndarray

    def __len__(self) -> int:
        return self.coin.shape[0]

    @property
    def length(self) -> int:
        return self.coin.shape[1]


def sample_paths(rng: np.random.Generator, n: int, length: int, params: ToyParams) -> Paths:
    T = mess3_tensor(params.x, params.a)
    table = pattern_automaton(params.pattern)
    fire = params.fire
    coin = rng.choice(3, size=(n, length), p=[params.q_shared, params.q_lang, params.q_blank])
    u = rng.random((n, length))
    alt = rng.integers(0, N_SYM - 1, size=(n, length))
    kind = np.full((n, length), STEP_ORDINARY, dtype=np.int8)
    sym = np.full((n, length), -1, dtype=np.int8)
    flat = T.reshape(N_SYM, N_SYM, N_SYM)
    for i in range(n):
        s = int(rng.integers(N_SYM))          # Mess3 stationary distribution is uniform
        m = 0
        for t in range(length):
            if coin[i, t] == COIN_BLANK:
                kind[i, t] = STEP_BLANK
                continue
            if m == fire:
                kind[i, t] = STEP_FIRING       # response chosen at render time; Mess3 does not move
                m = 0
                continue
            joint = flat[:, s, :].ravel()
            j = int(rng.choice(joint.size, p=joint / joint.sum()))
            k, s = divmod(j, N_SYM)
            sym[i, t] = k
            m = table[m][k]
    return Paths(coin=coin, kind=kind, sym=sym, u=u, alt=alt)


def response_symbol(f: int, u: float, alt: int, params: ToyParams) -> int:
    if u < 1.0 - params.eps:
        return int(f)
    others = [k for k in range(N_SYM) if k != f]
    return others[int(alt)]


def render_path(paths: Paths, i: int, lang: int, f: int, params: ToyParams, upto: int | None = None) -> list[int]:
    """Tokens of path i under (language, function), optionally only the first `upto` steps."""
    length = paths.length if upto is None else upto
    out = []
    for t in range(length):
        c = paths.coin[i, t]
        if c == COIN_BLANK:
            out.append(BLANK)
            continue
        k = int(paths.sym[i, t]) if paths.kind[i, t] == STEP_ORDINARY else response_symbol(
            f, paths.u[i, t], paths.alt[i, t], params)
        out.append(shared_token(k) if c == COIN_SHARED else lang_token(lang, k))
    return out


def path_firing_count(paths: Paths, i: int, upto: int) -> int:
    return int(np.sum(paths.kind[i, :upto] == STEP_FIRING))


def path_memory_after(paths: Paths, i: int, upto: int, params: ToyParams) -> int:
    """Pattern memory after the first `upto` steps (function- and language-independent)."""
    return int(memory_trajectory(render_path(paths, i, 0, 0, params, upto=upto), params)[-1])


# --- readouts for the language factor ------------------------------------------------------------

def read_unit_test_language(p: np.ndarray, target_lang: int, f_star: int, *, lang_tol: float = 0.05,
                            min_target_lang_mass: float = 0.1, blank_max: float = 0.5, bos_tol: float = 0.05) -> dict:
    """Unit-test readout when the transported factor is the language.

    Exhibits: the language tokens the model would emit are the target language's (dominant among
    languages, clearly present, negligible mass on the other languages). Runs: the response symbol
    is still f_star's, no blank collapse, no BOS.
    """
    p = np.asarray(p, dtype=float)
    lm = language_mass(p)
    sym = symbol_mass(p)
    other = float(lm.sum() - lm[target_lang])
    blank = float(p[BLANK])
    bos = float(p[BOS]) if p.shape[0] > BOS else 0.0
    exhibits = bool(int(np.argmax(lm)) == target_lang and lm[target_lang] >= min_target_lang_mass and other < lang_tol)
    runs = bool(int(np.argmax(sym)) == f_star and blank < blank_max and bos < bos_tol)
    return {
        "exhibits": exhibits, "runs": runs, "score": bool(exhibits and runs),
        "target_mass": float(lm[target_lang]), "language_mass": lm.tolist(), "symbol_mass": sym.tolist(),
        "other_lang_mass": other, "f_star_mass": float(sym[f_star]), "blank_mass": blank, "bos_mass": bos,
    }


def score_generation_language(gen_tokens, target_lang: int, f_star: int, params: ToyParams, m0: int = 0) -> dict:
    """Generation readout when the transported factor is the language: exhibits = the target
    language's tokens appear and no other language's do; runs = the firings give f_star."""
    gen_tokens = [int(t) for t in gen_tokens]
    langs = [token_language(t) for t in gen_tokens]
    n_target = sum(1 for lg in langs if lg == target_lang)
    n_other = sum(1 for lg in langs if lg is not None and lg != target_lang)
    exhibits = bool(n_target > 0 and n_other == 0)
    resp = responses(gen_tokens, params, m0)
    post = function_posterior(resp, params)
    fired = len(resp) >= 1
    runs = bool(fired and int(np.argmax(post)) == f_star)
    return {"runs": runs, "exhibits": exhibits, "score": bool(runs and exhibits), "fired": fired,
            "n_firings": len(resp), "n_target_lang_tokens": n_target, "n_other_lang_tokens": n_other,
            "posterior": post.tolist(), "bos_tokens": int(sum(1 for t in gen_tokens if t == BOS))}

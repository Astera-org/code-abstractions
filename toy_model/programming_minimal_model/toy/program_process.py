"""Exact-observer helpers for the program toy's second process (`program_call`).

Everything here is behavioral: a token codec, constructed probe contexts, and readouts that
compare a model's next-token distribution with the exact Bayes observer -- per function
(which generator's statistics the model exhibits), per language (which rendering), and the
return (does the model fire after the shared pattern, also across blank runs).

Global vocabulary for L languages (see gen_program_config.py):
  0..2    shared symbols S:k          3        shared return S:F
  4+4l+k  language-l symbols L_l:k    7+4l     language-l return L_l:F
  4+4L    blank B                     4+4L+1   BOS (appended by the trainer)
Component c = 3 * language + function; functions 0, 1, 2 = mess3, no_repeat, sum_mod3.
Component-local vocabulary (9): 0..2 S:k, 3 S:F, 4..6 L:k, 7 L:F, 8 B.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from matrices import _kmp_table, _program_generator
except ImportError:  # pragma: no cover - the toy scripts put training/ on sys.path themselves
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))
    from matrices import _kmp_table, _program_generator

FUNCTIONS = ("mess3", "no_repeat", "sum_mod3")
N_SYM, N_FUNC, RET = 3, 3, 3                 # RET: the "symbol" slot of the return in a rendering block
LOCAL_S, LOCAL_SF, LOCAL_L, LOCAL_LF, LOCAL_B = 0, 3, 4, 7, 8


# --- codec ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    n_langs: int

    @property
    def blank(self) -> int:
        return 4 + 4 * self.n_langs

    @property
    def bos(self) -> int:
        return self.blank + 1

    @property
    def n_tokens(self) -> int:
        """Tokens the process emits (BOS comes on top)."""
        return self.blank + 1

    def shared(self, k: int) -> int:
        return int(k)

    def lang(self, lang: int, k: int) -> int:
        return 4 + 4 * int(lang) + int(k)

    def render(self, k: int, lang: int, how: str) -> int:
        """k in 0..2 (symbol) or RET; how = 'shared' | 'lang'."""
        return self.shared(k) if how == "shared" else self.lang(lang, k)

    def is_blank(self, t) -> bool:
        return int(t) == self.blank

    def is_content(self, t) -> bool:
        return 0 <= int(t) < self.blank

    def token_symbol(self, t) -> int | None:
        """0..2 for a symbol, RET for a return, None for blank / BOS."""
        t = int(t)
        if not self.is_content(t):
            return None
        return t if t < 4 else (t - 4) % 4

    def is_return(self, t) -> bool:
        return self.token_symbol(t) == RET

    def token_language(self, t) -> int | None:
        t = int(t)
        if t < 4 or t >= self.blank:
            return None
        return (t - 4) // 4

    def language_tokens(self, lang: int) -> list[int]:
        return [self.lang(lang, k) for k in range(4)]

    def token_name(self, t) -> str:
        t = int(t)
        if t == self.bos:
            return "BOS"
        if t == self.blank:
            return "B"
        k = self.token_symbol(t)
        sym = "F" if k == RET else str(k)
        lang = self.token_language(t)
        return f"S:{sym}" if lang is None else f"L{lang}:{sym}"

    def local_to_global(self, lang: int) -> list[int]:
        return [0, 1, 2, 3, *self.language_tokens(lang), self.blank]


@dataclass(frozen=True)
class ProgramParams:
    n_langs: int
    pattern: tuple[int, ...]
    p_fire: float
    p_floor: float
    q_shared: float
    p_content_to_filler: float
    p_filler_to_content: float
    generators: tuple[dict, ...]          # per function: {"generator": name, <its params>}

    @property
    def n_funcs(self) -> int:
        return len(self.generators)

    @classmethod
    def from_run_config(cls, cfg) -> ProgramParams:
        """Components are ordered language-major (function inner); the number of functions is the
        number of leading components that share the first vocab map (3 in the original process, 4 in
        the extended one used by the fine-tuning transfer experiment)."""
        comps = cfg.generative_process.instance.components
        vms = [list(vm) for vm in cfg.generative_process.instance.vocab_maps]
        n_funcs = 0
        while n_funcs < len(vms) and vms[n_funcs] == vms[0]:
            n_funcs += 1
        first = comps[0].process_params
        gens = []
        for f in range(n_funcs):
            pp = comps[f].process_params
            gens.append({k: (str(pp[k]) if k == "generator" else float(pp[k]))
                         for k in ("generator", "x", "a", "rho", "delta") if k in pp})
        return cls(
            n_langs=len(comps) // n_funcs,
            pattern=tuple(int(p) for p in first.pattern),
            p_fire=float(first.p_fire), p_floor=float(first.p_floor), q_shared=float(first.q_shared),
            p_content_to_filler=float(first.p_content_to_filler),
            p_filler_to_content=float(first.p_filler_to_content),
            generators=tuple(gens),
        )

    @property
    def layout(self) -> Layout:
        return Layout(self.n_langs)

    def generator(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        """(T [3, S, S], start [S]) of function f's generator."""
        g = dict(self.generators[f])
        name = g.pop("generator")
        return _program_generator(name, x=g.get("x", 0.15), a=g.get("a", 0.6), rho=g.get("rho", 0.05),
                                  delta=g.get("delta", 0.1))


# --- the shared watcher on a token stream -------------------------------------------------

def watcher_trajectory(tokens, params: ProgramParams, m0: int = 0) -> np.ndarray:
    """Watcher progress after each token (blanks keep it, returns reset it, symbols advance it).
    Value len(pattern) = the pattern has just completed, so the next content step fires w.p. p_fire."""
    lay = params.layout
    nxt = _kmp_table(params.pattern, N_SYM)
    out = np.zeros(len(tokens), dtype=int)
    m = m0
    for i, t in enumerate(tokens):
        k = lay.token_symbol(t)
        if k is None:
            pass
        elif k == RET:
            m = 0
        else:
            m = nxt[m][k]
        out[i] = m
    return out


def expected_fire_positions(tokens, params: ProgramParams) -> np.ndarray:
    """Indices i such that the watcher is complete after token i (a return is due at i + 1)."""
    m = watcher_trajectory(tokens, params)
    return np.where(m == len(params.pattern))[0]


def kl(p, q, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


# --- the exact observer ---------------------------------------------------------------------

def make_observer(process):
    """Exact Bayesian filtering of the mixture along equal-length token batches (no BOS).

    Returns call(tokens [B, T]) -> (comp [B, T+1, C], local_next [B, T+1, C, 9], glob_next [B, T+1, V]):
    component posterior, per-component local next-token distribution and the mixture's global
    next-token distribution *before* each position t (index T is after the last token).
    """
    import jax
    import jax.numpy as jnp

    components = process.components

    def outputs(state):
        local = jnp.stack([c.observation_probability_distribution(s)
                           for c, s in zip(components, state.component_states, strict=True)])
        return state.component_beliefs, local, process.observation_probability_distribution(state)

    def one(tokens):
        def step(state, x):
            return process.transition_states(state, x), outputs(state)
        final, (comp, local, glob) = jax.lax.scan(step, process.initial_state, tokens)
        c_f, l_f, g_f = outputs(final)
        return (jnp.concatenate([comp, c_f[None]]), jnp.concatenate([local, l_f[None]]),
                jnp.concatenate([glob, g_f[None]]))

    fn = jax.jit(jax.vmap(one))

    def call(tokens_np: np.ndarray):
        comp, local, glob = fn(jnp.asarray(tokens_np, dtype=jnp.int32))
        return np.asarray(comp), np.asarray(local), np.asarray(glob)

    return call


def function_posterior(comp: np.ndarray, n_funcs: int = N_FUNC) -> np.ndarray:
    """[..., C] -> [..., n_funcs]: marginal over languages."""
    return comp.reshape(*comp.shape[:-1], -1, n_funcs).sum(-2)


def language_posterior(comp: np.ndarray, n_funcs: int = N_FUNC) -> np.ndarray:
    return comp.reshape(*comp.shape[:-1], -1, n_funcs).sum(-1)


def local_symbol_distribution(local: np.ndarray) -> np.ndarray:
    """[..., 9] local next distribution -> [..., 3] over symbols (both renderings, renormalized)."""
    p = local[..., LOCAL_S:LOCAL_S + 3] + local[..., LOCAL_L:LOCAL_L + 3]
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-30)


def function_predictives(comp: np.ndarray, local: np.ndarray, n_funcs: int = N_FUNC) -> tuple[np.ndarray, np.ndarray]:
    """Per-function next-symbol predictive (languages marginalized within the function).

    comp [C], local [C, 9] -> (q [n_funcs, 3] rows = functions, post_f [n_funcs])."""
    n_langs = comp.shape[0] // n_funcs
    w = comp.reshape(n_langs, n_funcs)
    sym = local_symbol_distribution(local).reshape(n_langs, n_funcs, N_SYM)
    post_f = w.sum(0)
    w_in_f = w / np.maximum(post_f[None, :], 1e-30)
    q = np.einsum("lf,lfk->fk", w_in_f, sym)
    return q, post_f


# --- model-side readouts ------------------------------------------------------------------------

def symbol_distribution(p_global: np.ndarray, lay: Layout) -> np.ndarray:
    """Model next-token distribution [V] -> [3] over symbols, all renderings summed, renormalized."""
    out = np.zeros(N_SYM)
    for k in range(N_SYM):
        out[k] = p_global[lay.shared(k)] + sum(p_global[lay.lang(lang, k)] for lang in range(lay.n_langs))
    return out / max(out.sum(), 1e-30)


def return_mass(p_global: np.ndarray, lay: Layout) -> float:
    return float(p_global[lay.shared(RET)] + sum(p_global[lay.lang(lang, RET)] for lang in range(lay.n_langs)))


def blank_mass(p_global: np.ndarray, lay: Layout) -> float:
    return float(p_global[lay.blank])


def language_masses(p_global: np.ndarray, lay: Layout) -> np.ndarray:
    """Mass on each language's tokens (symbols and return)."""
    return np.array([sum(p_global[t] for t in lay.language_tokens(lang)) for lang in range(lay.n_langs)])


def shared_mass(p_global: np.ndarray, lay: Layout) -> float:
    return float(sum(p_global[lay.shared(k)] for k in range(4)))


def function_readout(p_global: np.ndarray, comp: np.ndarray, local: np.ndarray, lay: Layout,
                     n_funcs: int = N_FUNC) -> dict:
    """Which function's next-symbol statistics does the model exhibit at this position?

    p: the model's symbol distribution; q[f]: the exact predictive of function f given the same
    context; kl[f] = KL(p || q[f]); `argmin` the exhibited function; `margin` the gap to the
    runner-up (small margin = the functions do not disagree here, so the readout is weak);
    `kl_bayes` = KL(p || Bayes mixture); `post_f` the observer's function posterior.
    """
    p = symbol_distribution(p_global, lay)
    q, post_f = function_predictives(comp, local, n_funcs)
    kls = np.array([kl(p, q[f]) for f in range(n_funcs)])
    order = np.argsort(kls)
    return {
        "p": p, "q": q, "post_f": post_f, "kl": kls, "argmin": int(order[0]),
        "margin": float(kls[order[1]] - kls[order[0]]),
        "disagreement": float(max(kl(q[i], q[j]) for i in range(n_funcs) for j in range(n_funcs))),
        "kl_bayes": kl(p, post_f @ q),
    }


def language_readout(p_global: np.ndarray, comp: np.ndarray, lay: Layout, n_funcs: int = N_FUNC) -> dict:
    """Language-specific mass by language vs the observer's language posterior."""
    masses = language_masses(p_global, lay)
    post_l = language_posterior(comp, n_funcs)
    return {
        "lang_mass": masses, "shared_mass": shared_mass(p_global, lay), "blank_mass": blank_mass(p_global, lay),
        "argmax": int(masses.argmax()), "post_l": post_l, "bayes_argmax": int(post_l.argmax()),
        "other_mass": float(masses.sum() - masses.max()),
    }


def return_readout(p_global: np.ndarray, glob_bayes: np.ndarray, lay: Layout) -> dict:
    return {"p_return": return_mass(p_global, lay), "bayes_return": return_mass(glob_bayes, lay),
            "p_blank": blank_mass(p_global, lay), "bayes_blank": blank_mass(glob_bayes, lay)}


# --- constructed probes --------------------------------------------------------------------------

def simulate_function_machine(f: int, n_content: int, rng: np.random.Generator, params: ProgramParams,
                              fire: str = "sample", m0: int = 0) -> list[int]:
    """Content events (0..2 symbols, RET returns) of function f's machine: generator + shared watcher.

    fire = 'sample' (hazard p_fire / p_floor), 'always' (return exactly after each completion, never
    otherwise), 'never' (no returns at all; the generator just runs). Returns reset the generator.
    """
    T, start = params.generator(f)
    nxt = _kmp_table(params.pattern, N_SYM)
    length = len(params.pattern)
    g = rng.choice(T.shape[1], p=start)
    m = m0
    out = []
    for _ in range(n_content):
        hazard = params.p_fire if m == length else params.p_floor
        if fire == "always":
            hazard = 1.0 if m == length else 0.0
        elif fire == "never":
            hazard = 0.0
        if rng.random() < hazard:
            out.append(RET)
            g = rng.choice(T.shape[1], p=start)
            m = 0
            continue
        probs = T[:, g, :].sum(1)
        k = int(rng.choice(N_SYM, p=probs / probs.sum()))
        nxt_g = T[k, g, :]
        g = rng.choice(T.shape[1], p=nxt_g / nxt_g.sum())
        m = nxt[m][k]
        out.append(k)
    return out


def render(events, lang: int, how: str, rng: np.random.Generator | None = None, q_shared: float = 0.8,
           lay: Layout | None = None) -> list[int]:
    """Render content events in language `lang`: how = 'shared' | 'lang' | 'coin' (q_shared shared)."""
    lay = lay or Layout(4)
    out = []
    for k in events:
        if how == "coin":
            assert rng is not None
            h = "shared" if rng.random() < q_shared else "lang"
        else:
            h = how
        out.append(lay.render(k, lang, h))
    return out


def insert_blank_run(tokens, at: int, length: int, lay: Layout) -> list[int]:
    tokens = list(tokens)
    return tokens[:at] + [lay.blank] * length + tokens[at:]


def language_mode_path(n: int, rng: np.random.Generator, params: ProgramParams, start_content: bool = True) -> np.ndarray:
    """Content (True) / filler (False) mode per position from the two-state language machine."""
    out = np.zeros(n, dtype=bool)
    content = start_content
    for i in range(n):
        out[i] = content
        if content:
            content = rng.random() >= params.p_content_to_filler
        else:
            content = rng.random() < params.p_filler_to_content
    return out


def sample_component_sequence(lang: int, f: int, n: int, rng: np.random.Generator, params: ProgramParams) -> list[int]:
    """One sequence from component (lang, f) by simulating the two machines directly (no JAX)."""
    modes = language_mode_path(n, rng, params)
    events = simulate_function_machine(f, int(modes.sum()), rng, params)
    rendered = render(events, lang, "coin", rng, params.q_shared, params.layout)
    out, j = [], 0
    for is_content in modes:
        if is_content:
            out.append(rendered[j])
            j += 1
        else:
            out.append(params.layout.blank)
    return out


# --- generation scoring ---------------------------------------------------------------------------

def continuation_loglik(comp_traj: np.ndarray, local_traj: np.ndarray, tokens, start: int, lay: Layout, n_funcs: int = N_FUNC) -> dict:
    """Per-function and per-language log-likelihood of tokens[start:] under the exact observer.

    comp_traj [T+1, C], local_traj [T+1, C, 9] from make_observer on the whole sequence. For each
    generated position t, function f explains token x_t with q_f(x_t) = sum_l w(l | f, x_<t) P(x_t | c=(l, f)),
    the component's local distribution mapped through its vocab map; likewise per language.
    Returns argmax functions / languages and the log-likelihood vectors."""
    n_langs = lay.n_langs
    ll_f = np.zeros(n_funcs)
    ll_l = np.zeros(n_langs)
    for t in range(start, len(tokens)):
        x = int(tokens[t])
        w = comp_traj[t].reshape(n_langs, n_funcs)
        px = np.zeros((n_langs, n_funcs))
        for lang in range(n_langs):
            l2g = lay.local_to_global(lang)
            loc = l2g.index(x) if x in l2g else None
            if loc is not None:
                px[lang] = local_traj[t, lang * n_funcs:(lang + 1) * n_funcs, loc]
        post_f = w.sum(0)
        post_l = w.sum(1)
        qf = np.einsum("lf,lf->f", w / np.maximum(post_f[None], 1e-30), px)
        ql = np.einsum("lf,lf->l", w / np.maximum(post_l[:, None], 1e-30), px)
        ll_f += np.log(np.maximum(qf, 1e-30))
        ll_l += np.log(np.maximum(ql, 1e-30))
    return {"ll_f": ll_f, "argmax_f": int(ll_f.argmax()), "ll_l": ll_l, "argmax_l": int(ll_l.argmax())}


# --- the exact observer in numpy (no JIT; handy for many distinct prefix lengths) ------------------

class NumpyObserver:
    """Exact Bayesian filter of the nonergodic mixture in numpy, same outputs as `make_observer`.

    call(tokens [B, T]) -> (comp [B, T+1, C], local [B, T+1, C, 9], glob [B, T+1, V]); `filter_state`
    returns the component weights and per-component beliefs after a batch of prefixes, and
    `sample_continuations` draws from the mixture's posterior predictive from such a state."""

    def __init__(self, process):
        self.n_comp = len(process.components)
        self.T = [np.asarray(c.transition_matrices, dtype=np.float64) for c in process.components]
        self.init = [np.asarray(c.initial_state, dtype=np.float64) for c in process.components]
        self.w0 = np.asarray(process.component_weights, dtype=np.float64)
        self.vocab_size = int(process.vocab_size)
        self.vmaps = [np.asarray(vm, dtype=int) for vm in process.vocab_maps]
        self.inv = []
        for vm in self.vmaps:
            inv = -np.ones(self.vocab_size, dtype=int)
            inv[vm] = np.arange(len(vm))
            self.inv.append(inv)
        self.emit = [T.sum(2) for T in self.T]                     # [9, S]: P(local token | state)

    def _outputs(self, w, b):
        B = w.shape[0]
        local = np.zeros((B, self.n_comp, 9))
        glob = np.zeros((B, self.vocab_size))
        for c in range(self.n_comp):
            local[:, c] = b[c] @ self.emit[c].T
            glob[:, self.vmaps[c]] += w[:, c:c + 1] * local[:, c]
        return local, glob

    def _step(self, w, b, x):
        B = x.shape[0]
        lik = np.zeros((B, self.n_comp))
        new_b = []
        in_vocab = (x >= 0) & (x < self.vocab_size)          # a sampled BOS is outside the process's vocabulary
        x_safe = np.where(in_vocab, x, 0)
        for c in range(self.n_comp):
            loc = np.where(in_vocab, self.inv[c][x_safe], -1)
            valid = loc >= 0
            Tsel = self.T[c][np.where(valid, loc, 0)]              # [B, S, S]
            nb = np.einsum("bs,bst->bt", b[c], Tsel)
            lk = nb.sum(1) * valid
            lik[:, c] = lk
            nb = np.where(valid[:, None], nb / np.maximum(lk[:, None], 1e-300), b[c])
            new_b.append(nb)
        w = w * lik
        norm = w.sum(1, keepdims=True)
        w = np.where(norm > 0, w / np.maximum(norm, 1e-300), w)
        return w, new_b

    def filter_state(self, tokens):
        tokens = np.asarray(tokens, dtype=int)
        B, T = tokens.shape
        w = np.tile(self.w0, (B, 1))
        b = [np.tile(self.init[c], (B, 1)) for c in range(self.n_comp)]
        for t in range(T):
            w, b = self._step(w, b, tokens[:, t])
        return w, b

    def __call__(self, tokens):
        tokens = np.asarray(tokens, dtype=int)
        B, T = tokens.shape
        w = np.tile(self.w0, (B, 1))
        b = [np.tile(self.init[c], (B, 1)) for c in range(self.n_comp)]
        comp = np.zeros((B, T + 1, self.n_comp))
        local = np.zeros((B, T + 1, self.n_comp, 9))
        glob = np.zeros((B, T + 1, self.vocab_size))
        for t in range(T + 1):
            comp[:, t] = w
            local[:, t], glob[:, t] = self._outputs(w, b)
            if t < T:
                w, b = self._step(w, b, tokens[:, t])
        return comp, local, glob

    def sample_continuations(self, w, b, n_new: int, rng: np.random.Generator) -> np.ndarray:
        """Posterior-predictive samples [B, n_new]: pick a component from w, a state from its belief,
        then run that component forward."""
        B = w.shape[0]
        out = np.zeros((B, n_new), dtype=int)
        for i in range(B):
            c = rng.choice(self.n_comp, p=w[i] / w[i].sum())
            s = rng.choice(b[c].shape[1], p=b[c][i] / b[c][i].sum())
            T = self.T[c]
            for t in range(n_new):
                probs = T[:, s, :].sum(1)
                loc = rng.choice(9, p=probs / probs.sum())
                nxt = T[loc, s, :]
                s = rng.choice(nxt.shape[0], p=nxt / nxt.sum())
                out[i, t] = self.vmaps[c][loc]
        return out


# --- designed unit tests ---------------------------------------------------------------------------

UNIT_TESTS = ("sum", "no_repeat", "mess3_run")      # one per function, in function order 2, 1, 0
UNIT_TEST_FUNCTION = {"sum": 2, "no_repeat": 1, "mess3_run": 0, "diff": 3}


def designed_suffix(test: str, rng: np.random.Generator) -> tuple[list[int], int]:
    """Content events appended to a context to set up a function's unit test, and the target symbol.

      sum        -- a return (resets the phase), then two arguments a, b: expects a + b mod 3 next.
      no_repeat  -- a return, then one symbol s: expects anything BUT s next (no-repeat 0.05, Mess3
                    ~0.39 after a reset and one s, sum 1/3 at phase 1). Without the return the test
                    would also pass under sum at phase 2 whenever a != 0, since the answer a + b differs
                    from b.
      mess3_run  -- a return, then s s s: Mess3 is sticky, so expects an elevated P(s); the two other
                    functions predict 1/3 (sum, phase 0) and 0.05 (no-repeat) for s.
    A return is in-support at any content step (probability >= p_floor), so every prompt is
    in-distribution for every function."""
    if test == "sum":
        a, b = int(rng.integers(3)), int(rng.integers(3))
        return [RET, a, b], (a + b) % 3
    if test == "diff":
        # the held-out fourth function (a - b) mod 3; b != 0 so that sum and difference disagree
        a, b = int(rng.integers(3)), int(rng.integers(1, 3))
        return [RET, a, b], (a - b) % 3
    if test == "no_repeat":
        s = int(rng.integers(3))
        return [RET, s], s
    if test == "mess3_run":
        s = int(rng.integers(3))
        return [RET, s, s, s], s
    raise ValueError(test)


def unit_exhibits(test: str, p_target: float, argmin_f: int) -> int:
    """Pass rule per unit test (the observer's own value under the right function: 0.9, 0.05, ~0.45)."""
    if test in ("sum", "diff"):
        return int(p_target > 0.5)
    if test == "no_repeat":
        return int(p_target < 0.15)
    return int(argmin_f == 0)


def unit_readout(p_model: np.ndarray, glob: np.ndarray, test: str, target: int, argmin_f: int, argmin_bayes: int,
                 lay: Layout) -> dict:
    """Model and observer on one designed prompt: P(target symbol | next is a symbol) and the pass rule."""
    p = symbol_distribution(p_model, lay)
    q = symbol_distribution(glob, lay)
    return {"p_target": float(p[target]), "bayes_target": float(q[target]),
            "exhibits": unit_exhibits(test, float(p[target]), argmin_f),
            "bayes_exhibits": unit_exhibits(test, float(q[target]), argmin_bayes)}


# --- interpretable statistics of a continuation -----------------------------------------------------

def continuation_stats(prefix, cont, lang: int, params: ProgramParams) -> dict:
    """Signature statistics of the tokens in `cont` (given `prefix` for the watcher / phase state).

    repeat_rate    fraction of consecutive content-symbol pairs (no return in between) that repeat
                   (no-repeat 0.05, sum ~1/3, Mess3 ~0.39)
    sum_correct    fraction of answer positions (third symbol since the last return) equal to a+b mod 3
                   (sum 0.9, others ~1/3)
    ret_precision  returns that follow a completed pattern / returns
    ret_recall     completed patterns followed by a return (blanks skipped) / completed patterns (0.9)
    blank_frac, mean_blank_run, lang_share (language-specific among content tokens, 0.2),
    other_lang (content tokens of another language), n_bos
    """
    lay = params.layout
    nxt = _kmp_table(params.pattern, N_SYM)
    length = len(params.pattern)
    full = list(prefix) + list(cont)
    start = len(prefix)
    m, phase, last_sym, args = 0, 0, None, []
    pairs = repeats = answers = correct = 0
    n_ret = ret_ok = completions = completions_ret = 0
    pending_completion = False
    blank_runs, run = [], 0
    n_content = n_lang = other = n_bos = 0
    for i, t in enumerate(full):
        t = int(t)
        in_cont = i >= start
        if t == lay.bos:
            if in_cont:
                n_bos += 1
            continue
        if lay.is_blank(t):
            if in_cont:
                run += 1
            continue
        if in_cont and run:
            blank_runs.append(run)
        run = 0
        k = lay.token_symbol(t)
        tl = lay.token_language(t)
        if in_cont:
            n_content += 1
            n_lang += int(tl is not None)
            other += int(tl is not None and tl != lang)
        if pending_completion:
            if in_cont:
                completions += 1
                completions_ret += int(k == RET)
            pending_completion = False
        if k == RET:
            if in_cont:
                n_ret += 1
                ret_ok += int(m == length)
            m, phase, last_sym, args = 0, 0, None, []
            continue
        if in_cont and last_sym is not None:
            pairs += 1
            repeats += int(k == last_sym)
        if phase == 2:
            if in_cont:
                answers += 1
                correct += int(k == (args[0] + args[1]) % 3)
            phase, args = 0, []
        else:
            args.append(k)
            phase += 1
        last_sym = k
        m = nxt[m][k]
        pending_completion = m == length
    if run:
        blank_runs.append(run)
    n_cont = len(cont)
    return {
        "repeat_rate": repeats / pairs if pairs else float("nan"), "n_pairs": pairs,
        "sum_correct": correct / answers if answers else float("nan"), "n_answers": answers,
        "ret_precision": ret_ok / n_ret if n_ret else float("nan"), "n_returns": n_ret,
        "ret_recall": completions_ret / completions if completions else float("nan"), "n_completions": completions,
        "blank_frac": sum(1 for t in cont if lay.is_blank(t)) / n_cont if n_cont else float("nan"),
        "mean_blank_run": float(np.mean(blank_runs)) if blank_runs else float("nan"),
        "lang_share": n_lang / n_content if n_content else float("nan"),
        "other_lang": other, "n_bos": n_bos,
    }


# --- shortcut discriminators: compute vs lookup, symbol vs token ---------------------------------------

def simulate_sum_context(rng: np.random.Generator, params: ProgramParams, n_content: int, pair: tuple[int, int],
                         condition: str) -> list[int]:
    """Content events of the sum machine (with the shared watcher and soft returns) under a control on the
    argument pair `pair` = (a*, b*):

      unseen      the pair never occurs as (a, b) of a triple (b is resampled away from b* when a == a*)
      seen        every occurrence of the pair gets the correct answer; at least one occurrence is forced
                  (the first triple uses the pair)
      seen_wrong  the first triple uses the pair and every occurrence gets a WRONG answer (in-support: the
                  machine answers wrongly with probability delta)
    Everything else follows the machine: returns fire at 0.9 after the pattern and 0.01 otherwise and reset
    the phase."""
    nxt = _kmp_table(params.pattern, N_SYM)
    length = len(params.pattern)
    m = 0
    phase, args = 0, []
    forced = condition != "unseen"
    out = []
    while len(out) < n_content:
        hazard = params.p_fire if m == length else params.p_floor
        if rng.random() < hazard:
            out.append(RET)
            m, phase, args = 0, 0, []
            continue
        if phase == 0:
            k = pair[0] if forced else int(rng.integers(3))
        elif phase == 1:
            k = pair[1] if forced else int(rng.integers(3))
            if condition == "unseen" and args[0] == pair[0]:
                while k == pair[1]:
                    k = int(rng.integers(3))
        else:
            correct = (args[0] + args[1]) % 3
            if tuple(args) == tuple(pair) and condition == "seen_wrong":
                k = int(rng.choice([s for s in range(3) if s != correct]))
            elif tuple(args) == tuple(pair) and condition == "seen":
                k = correct
            else:
                k = correct if rng.random() >= params.generators[2].get("delta", 0.1) else \
                    int(rng.choice([s for s in range(3) if s != correct]))
        if phase == 2:
            forced = False          # only the first triple is forced; later ones follow the machine
            phase, args = 0, []
        else:
            args.append(k)
            phase += 1
        out.append(k)
        m = nxt[m][k]
    return out


def sum_pairs_seen(events) -> dict[tuple[int, int], list[int]]:
    """Argument pairs of the completed triples in a content-event stream -> list of the answers shown."""
    seen: dict[tuple[int, int], list[int]] = {}
    phase, args = 0, []
    for k in events:
        if k == RET:
            phase, args = 0, []
            continue
        if phase == 2:
            seen.setdefault((args[0], args[1]), []).append(k)
            phase, args = 0, []
        else:
            args.append(k)
            phase += 1
    return seen


RENDERINGS = (("shared", "shared"), ("shared", "lang"), ("lang", "shared"), ("lang", "lang"))


def render_with_suffix_renderings(events, suffix, lang: int, rng: np.random.Generator, params: ProgramParams,
                                  renderings=RENDERINGS) -> list[list[int]]:
    """The context coin-rendered once, then the suffix's *last two* events in each rendering combination
    (earlier suffix events shared). Returns one token list per combination."""
    lay = params.layout
    ctx = render(events, lang, "coin", rng, params.q_shared, lay)
    head = render(suffix[:-2], lang, "shared", lay=lay)
    out = []
    for how_a, how_b in renderings:
        out.append(ctx + head + [lay.render(suffix[-2], lang, how_a), lay.render(suffix[-1], lang, how_b)])
    return out

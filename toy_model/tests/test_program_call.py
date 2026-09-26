"""Invariants for the program toy's second process: `program_call` and its mixtures.

Pins (1) agreement with an independent Kronecker construction of the same machine, (2) the
rank-one return, the frozen function machine under blanks and full support of the content
tokens, (3) that the exact observer identifies language and function from samples, and (4)
that both Hydra configs build a NonErgodicGenerativeProcess whose components start in content
mode at the generator's start state.
"""

from pathlib import Path

import matrices
import numpy as np
import pytest
from omegaconf import OmegaConf
from genproc.generative_processes.transition_matrices import HMM_MATRIX_FUNCTIONS

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "training" / "configs" / "generative_process"
CONFIGS = {4: CONFIG_DIR / "nonergodic_12x_program_call.yaml", 8: CONFIG_DIR / "nonergodic_24x_program_call.yaml"}

P_FIRE, P_FLOOR, Q_SHARED, P_CF, P_FC = 0.9, 0.01, 0.8, 1.0 / 24.0, 1.0 / 8.0
GENS = {"mess3": {"x": 0.15, "a": 0.6}, "no_repeat": {"rho": 0.05}, "sum_mod3": {"delta": 0.1}}
SIZES = {"mess3": 18, "no_repeat": 24, "sum_mod3": 42}
S_SYM, S_FIRE, L_SYM, L_FIRE, BLANK = 0, 3, 4, 7, 8
CONTENT, FILLER = 0, 1


def _registered(gen, **kw):
    return np.asarray(HMM_MATRIX_FUNCTIONS["program_call"](
        generator=gen, **GENS[gen], pattern=(0, 1), p_fire=P_FIRE, p_floor=P_FLOOR, q_shared=Q_SHARED,
        p_content_to_filler=P_CF, p_filler_to_content=P_FC, **kw,
    ))


# --- independent reference: Kronecker construction of the same machine -------------------

def _ref_mess3(x, a):
    b = (1 - a) / 2
    y = 1 - 2 * x
    ay, bx, by, ax = a * y, b * x, b * y, a * x
    return np.array([[[ay, bx, bx], [ax, by, bx], [ax, bx, by]],
                     [[by, ax, bx], [bx, ay, bx], [bx, ax, by]],
                     [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]])


def _ref_generator(gen):
    if gen == "mess3":
        return _ref_mess3(0.15, 0.6), np.full(3, 1 / 3)
    if gen == "no_repeat":
        rho = 0.05
        T = np.zeros((3, 4, 4))
        for v in range(3):
            T[v, 3, v] = 1 / 3
            for i in range(3):
                T[v, i, v] = rho if v == i else (1 - rho) / 2
        start = np.zeros(4)
        start[3] = 1
        return T, start
    delta = 0.1
    T = np.zeros((3, 7, 7))
    for v in range(3):
        T[v, 0, 1 + v] = 1 / 3
        for arg in range(3):
            T[v, 1 + arg, 4 + (arg + v) % 3] = 1 / 3
        for c in range(3):
            T[v, 4 + c, 0] = 1 - delta if v == c else delta / 2
    start = np.zeros(7)
    start[0] = 1
    return T, start


def _kmp(pattern, m, k):
    seen = list(pattern[:m]) + [k]
    j = min(len(seen), len(pattern))
    while j > 0 and tuple(seen[-j:]) != tuple(pattern[:j]):
        j -= 1
    return j


def _ref_program_call(gen, pattern=(0, 1)):
    T, start = _ref_generator(gen)
    S, L = T.shape[1], len(pattern)
    n_mem = L + 1
    W = np.zeros((3, n_mem, n_mem))
    for m in range(n_mem):
        for k in range(3):
            W[k, m, _kmp(pattern, m, k)] = 1
    hazard = np.kron(np.ones(S), np.array([P_FLOOR] * L + [P_FIRE]))
    e0 = np.zeros(n_mem)
    e0[0] = 1
    fire = np.outer(hazard, np.kron(start, e0))
    sym = [np.diag(1 - hazard) @ np.kron(T[k], W[k]) for k in range(3)]
    n = S * n_mem
    Z, eye = np.zeros((n, n)), np.eye(n)

    def content_block(X):
        return np.block([[(1 - P_CF) * X, P_CF * X], [Z, Z]])

    out = np.zeros((9, 2 * n, 2 * n))
    for k in range(3):
        out[S_SYM + k] = Q_SHARED * content_block(sym[k])
        out[L_SYM + k] = (1 - Q_SHARED) * content_block(sym[k])
    out[S_FIRE] = Q_SHARED * content_block(fire)
    out[L_FIRE] = (1 - Q_SHARED) * content_block(fire)
    out[BLANK] = np.block([[Z, Z], [P_FC * eye, (1 - P_FC) * eye]])
    return out, start


# --- the component ------------------------------------------------------------------------

@pytest.mark.parametrize("gen", sorted(GENS))
def test_matches_kronecker_reference(gen):
    got = _registered(gen)
    want, _ = _ref_program_call(gen)
    assert got.shape == (9, SIZES[gen], SIZES[gen])
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_defaults_are_the_chosen_defaults():
    got = np.asarray(HMM_MATRIX_FUNCTIONS["program_call"]())
    np.testing.assert_allclose(got, _registered("mess3"), atol=1e-6)


@pytest.mark.parametrize("gen", sorted(GENS))
def test_row_stochastic_with_full_support_on_content(gen):
    T = _registered(gen)
    n = T.shape[1] // 2
    assert T.min() >= 0
    np.testing.assert_allclose(T.sum(axis=(0, 2)), 1.0, atol=1e-6)
    tok = T.sum(2)                                   # P(token | state)
    assert (tok[:BLANK, :n] > 0).all(), "every content token possible at every content step"
    np.testing.assert_allclose(tok[BLANK, :n], 0.0)
    np.testing.assert_allclose(tok[BLANK, n:], 1.0)
    np.testing.assert_allclose(tok[:BLANK, n:], 0.0)
    # the language coin: 0.8 of content mass is shared, whatever the state
    np.testing.assert_allclose(tok[[0, 1, 2, 3], :n].sum(0), Q_SHARED, atol=1e-6)


@pytest.mark.parametrize("gen", sorted(GENS))
def test_blank_freezes_the_function_machine(gen):
    T = _registered(gen)
    n = T.shape[1] // 2
    blank = T[BLANK, n:, :]                          # from filler states
    np.testing.assert_allclose(blank[:, :n], P_FC * np.eye(n), atol=1e-6)
    np.testing.assert_allclose(blank[:, n:], (1 - P_FC) * np.eye(n), atol=1e-6)


@pytest.mark.parametrize("gen", sorted(GENS))
def test_return_is_a_rank_one_reset(gen):
    T = _registered(gen)
    _, start = _ref_generator(gen)
    n = T.shape[1] // 2
    n_mem = 3
    fire = (T[S_FIRE] + T[L_FIRE])[:n]               # content sources, both renderings
    fire = fire[:, :n] + fire[:, n:]                 # both next modes
    target = np.kron(start, np.eye(1, n_mem, 0).ravel())
    for i in range(n):
        m = i % n_mem
        want = (P_FIRE if m == 2 else P_FLOOR) * target
        np.testing.assert_allclose(fire[i], want, atol=1e-6)
    assert np.linalg.matrix_rank(fire, tol=1e-5) == 1


def test_kmp_table_has_the_completed_row():
    table = matrices._kmp_table((0, 1), 3)
    assert table[:2] == matrices._pattern_automaton((0, 1), 3)
    assert table[2] == [1, 0, 0]                     # a miss after '0 1': a new 0 re-arms
    table = matrices._kmp_table((0, 0), 3)
    assert table[2] == [2, 0, 0]                     # '0 0' then 0 is again complete


@pytest.mark.parametrize("gen", sorted(GENS))
def test_initial_state_helper(gen):
    init = np.asarray(matrices.program_call_initial_state(gen, **GENS[gen]))
    _, start = _ref_generator(gen)
    want = np.zeros(SIZES[gen])
    want[: SIZES[gen] // 2 : 3] = start
    np.testing.assert_allclose(init, want, atol=1e-6)


@pytest.mark.parametrize("bad", [{"generator": "foo"}, {"p_fire": 1.5}, {"pattern": ()}, {"q_shared": -0.1}])
def test_rejects_bad_parameters(bad):
    with pytest.raises(ValueError):
        HMM_MATRIX_FUNCTIONS["program_call"](**bad)


# --- the exact observer on samples ------------------------------------------------------

def _vocab_map(lang, n_langs):
    return [0, 1, 2, 3, *[4 + 4 * lang + k for k in range(4)], 4 + 4 * n_langs]


def _mixture(n_langs):
    """Block-diagonal HMM over the global tokens, plus block offsets and initial belief."""
    gens = list(GENS)
    blocks = [(_registered(g), g) for g in gens]
    sizes = [SIZES[g] for g in gens] * n_langs
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    n_tok = 4 + 4 * n_langs + 1
    G = np.zeros((n_tok, offsets[-1], offsets[-1]))
    b0 = np.zeros(offsets[-1])
    for lang in range(n_langs):
        for f, (T, g) in enumerate(blocks):
            c = 3 * lang + f
            sl = slice(offsets[c], offsets[c + 1])
            for loc, glob in enumerate(_vocab_map(lang, n_langs)):
                G[glob, sl, sl] += T[loc]
            b0[sl] = np.asarray(matrices.program_call_initial_state(g, **GENS[g])) / (3 * n_langs)
    return G, offsets, b0


def test_observer_identifies_language_and_function():
    n_langs = 4
    G, offsets, b0 = _mixture(n_langs)
    rng = np.random.default_rng(1)
    n_comp = 3 * n_langs
    lang_ok, func_ok, blanks, fires, n_tok = 0, 0, 0, 0, 0
    for trial in range(60):
        c = trial % n_comp
        i = rng.choice(G.shape[1], p=b0 * n_comp) if False else None
        block = b0[offsets[c]:offsets[c + 1]] * n_comp
        i = offsets[c] + rng.choice(block.size, p=block)
        toks = []
        for _ in range(63):
            row = G[:, i, :].ravel()
            j = rng.choice(row.size, p=row / row.sum())
            x, i = divmod(j, G.shape[1])
            toks.append(x)
        b = b0.copy()
        for x in toks:
            b = b @ G[x]
            b /= b.sum()
        comp = np.array([b[offsets[k]:offsets[k + 1]].sum() for k in range(n_comp)]).reshape(n_langs, 3)
        lang_ok += int(comp.sum(1).argmax() == c // 3)
        func_ok += int(comp.sum(0).argmax() == c % 3)
        toks = np.array(toks)
        blanks += int((toks == 4 + 4 * n_langs).sum())
        fires += int(np.isin(toks, [3] + [7 + 4 * lang for lang in range(n_langs)]).sum())
        n_tok += toks.size
    assert lang_ok >= 57, lang_ok
    assert func_ok >= 51, func_ok
    assert 0.18 < blanks / n_tok < 0.32, blanks / n_tok
    assert 3.0 < 63 * fires / n_tok < 8.0, 63 * fires / n_tok


# --- the Hydra configs ---------------------------------------------------------------------

@pytest.fixture(scope="module", params=[4, 8])
def built(request):
    import hydra
    cfg = OmegaConf.load(CONFIGS[request.param])
    root = OmegaConf.create({"device": "cpu", "generative_process": cfg})
    return request.param, hydra.utils.instantiate(root.generative_process.instance)


def test_config_builds_nonergodic_process(built):
    from genproc.generative_processes.nonergodic_generative_process import NonErgodicGenerativeProcess
    n_langs, process = built
    assert isinstance(process, NonErgodicGenerativeProcess)
    assert process.vocab_size == 4 + 4 * n_langs + 1
    assert len(process.components) == 3 * n_langs
    sizes = [c.num_states for c in process.components]
    assert sizes == [18, 24, 42] * n_langs
    for lang in range(n_langs):
        for f in range(3):
            np.testing.assert_array_equal(np.asarray(process.vocab_maps[3 * lang + f]), _vocab_map(lang, n_langs))


def test_config_components_start_in_content_at_generator_start(built):
    n_langs, process = built
    for c, comp in enumerate(process.components):
        g = list(GENS)[c % 3]
        want = np.asarray(matrices.program_call_initial_state(g, **GENS[g]))
        np.testing.assert_allclose(np.asarray(comp.initial_state), want, atol=1e-7)
    init = process.initial_state
    n_comp = 3 * n_langs
    np.testing.assert_allclose(np.asarray(init.component_beliefs), np.ones(n_comp) / n_comp, atol=1e-7)
    dist = np.asarray(process.observation_probability_distribution(init))
    assert dist.sum() == pytest.approx(1.0, abs=1e-6)
    # first step: content mode for sure, so no blank; 0.8 shared, 0.2 spread over the languages
    assert dist[4 + 4 * n_langs] == pytest.approx(0.0, abs=1e-7)
    assert dist[:4].sum() == pytest.approx(Q_SHARED, abs=1e-6)
    for lang in range(n_langs):
        assert dist[4 + 4 * lang: 8 + 4 * lang].sum() == pytest.approx((1 - Q_SHARED) / n_langs, abs=1e-6)

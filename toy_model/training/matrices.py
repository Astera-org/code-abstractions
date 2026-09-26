"""Custom HMM/GHMM transition matrices, registered into genproc's registries.

Importing this module is what makes `process_name: <name>` resolvable in a
generative-process config -- each function below assigns itself into
`HMM_MATRIX_FUNCTIONS` or `GHMM_MATRIX_FUNCTIONS` at import time. To add your own
process, write a function returning a `[symbol, from, to]` transition tensor and
register it the same way; no other wiring is needed.

`mess3` and `tom_quantum` are not here -- they ship with genproc.
"""
import jax.numpy as jnp
from jax import Array

from genproc.generative_processes.transition_matrices import GHMM_MATRIX_FUNCTIONS, HMM_MATRIX_FUNCTIONS


def left_right_mix(a: float = 0.0, b: float = 0.0) -> Array:
    """
    Parameters:
    a (float): L/R Asymmetry parameter.
    b (float): Leak parameter for Symbol 1 (A->C).
    """
    # Symbol 0: Left Cycle (A->B->C->A)
    val0 = 0.5 + a
    T0 = jnp.array([
        [0.0, val0, 0.0],  # A -> B
        [0.0, 0.0, val0],  # B -> C
        [val0, 0.0, 0.0]   # C -> A
    ])

    # Symbol 1: Right Cycle (A->C->B->A)
    val1 = 0.44 - a
    val1_leak = 0.44 - a - b

    T1 = jnp.array([
        [0.0, 0.0, val1_leak],  # A -> C
        [val1, 0.0, 0.0],       # B -> A
        [0.0, val1, 0.0]        # C -> B
    ])

    # Symbol 2: uniform noise, with compensation for leak
    val2_bc = 0.02
    val2_a = 0.02 + (b / 3.0)

    T2 = jnp.array([
        [val2_a, val2_a, val2_a],
        [val2_bc, val2_bc, val2_bc],
        [val2_bc, val2_bc, val2_bc]
    ])

    return jnp.array([T0, T1, T2])


# Register with genproc
HMM_MATRIX_FUNCTIONS["left_right_mix"] = left_right_mix


def mess3_reset(x: float, a: float, r: float = 0.05) -> Array:
    """mess3 with a 4th reset token that resets belief to uniform over 3 states.

    Parameters:
        x: mess3 entropy parameter
        a: mess3 asymmetry parameter
        r: emission probability of the reset token from each state (keep small)
    """
    from genproc.generative_processes.transition_matrices import mess3 as _mess3
    T_mess3 = _mess3(x, a)                               # (3, 3, 3)
    T_scaled = (1 - r) * T_mess3                         # scale to make room for reset
    T_reset = jnp.full((1, 3, 3), r / 3.0)              # (1, 3, 3), uniform rows
    return jnp.concatenate([T_scaled, T_reset], axis=0)  # (4, 3, 3)


HMM_MATRIX_FUNCTIONS["mess3_reset"] = mess3_reset


def mess3_xyz(x1: float = 0.15, x2: float = 0.15, x3: float = 0.15, a: float = 0.6) -> Array:
    """Per-state generalization of mess3: each state has its own focus weight.

    Standard mess3 uses a single x for all states (x1=x2=x3=x). Here each state's
    off-column mass is controlled independently, allowing asymmetric retention.
    Token k concentrates weight ``a`` on column k and ``b=(1-a)/2`` on others;
    within each column j the off-diagonal entries are x_j and the diagonal is
    y_i = 1 - sum_{j != i} x_j. Sum over (token, next-state) equals 1 per state.

    Parameters:
        x1: off-diagonal column weight for state 0
        x2: off-diagonal column weight for state 1
        x3: off-diagonal column weight for state 2
        a:  token-matched column weight (vs. (1-a)/2 on the other two columns)
    """
    b = (1 - a) / 2
    y0 = 1 - x2 - x3
    y1 = 1 - x1 - x3
    y2 = 1 - x1 - x2

    T0 = jnp.array([
        [a*y0, b*x2, b*x3],
        [a*x1, b*y1, b*x3],
        [a*x1, b*x2, b*y2],
    ])
    T1 = jnp.array([
        [b*y0, a*x2, b*x3],
        [b*x1, a*y1, b*x3],
        [b*x1, a*x2, b*y2],
    ])
    T2 = jnp.array([
        [b*y0, b*x2, a*x3],
        [b*x1, b*y1, a*x3],
        [b*x1, b*x2, a*y2],
    ])
    return jnp.stack([T0, T1, T2], axis=0)


HMM_MATRIX_FUNCTIONS["mess3_xyz"] = mess3_xyz


def _quantum_rrxor_kraus(phi: float = 1.0, theta: float = 2.0, epsilon: float = 0.0) -> Array:
    """Original Kraus-operator implementation of quantum_rrxor (not registered).

    Kept for reference / cross-validation.  Suffers from float32 precision loss
    in the complex→real conversion, causing the net transition matrix eigenvalue
    to drift from 1 (~0.9995 instead of 1.0).
    """
    ct = jnp.cos(theta / 2)
    st = jnp.sin(theta / 2)
    cp = jnp.cos(phi / 2)
    sp = jnp.sin(phi / 2)

    K0 = jnp.array([
        [ct * cp,        -ct * sp],
        [-1j * st * sp,  -1j * st * cp],
    ])
    K1 = jnp.array([
        [-1j * st * cp,  1j * st * sp],
        [ct * sp,        ct * cp],
    ])

    B = jnp.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ], dtype=complex) / 2

    Ts = []
    for K in [K0, K1]:
        C = K @ B @ K.conj().T           # (4, 2, 2)
        G = 2 * jnp.einsum('mij,nji->mn', B, C).real
        Ts.append(G.T)

    T = jnp.stack(Ts, axis=0)            # (2, 4, 4)

    if epsilon > 0:
        reset = jnp.zeros((4, 4)).at[0, 0].set(1.0)
        T = (1 - epsilon) * T + (epsilon / 2) * reset

    return T


def quantum_rrxor(phi: float = 1.0, theta: float = 2.0, epsilon: float = 0.0) -> Array:
    """Quantum RRXOR: 2-vocab 4-state GHMM from a quantum circuit with persistent memory qubit.

    Derived from a repeating circuit: R_y(φ) on memory, CNOT, R_x(θ) on ancilla,
    measure ancilla.  The belief geometry is a 3D fractal in the Bloch ball.
    For φ/π irrational, the process has no finite classical HMM.

    The Pauli transfer matrix (PTM) entries are computed analytically from
    R_mn = ½ tr(σ_m K_x σ_n K_x†) in the {I, σ_x, σ_y, σ_z} basis, avoiding
    complex intermediate arithmetic and the float32 precision loss of the
    Kraus-operator version (see ``_quantum_rrxor_kraus``).

    Parameters:
        phi: memory rotation angle (R_y gate on memory qubit)
        theta: readout angle (R_x gate on ancilla before measurement);
               θ→π is classical limit, θ→0 is uninformative
        epsilon: leakage toward stationary state (0 = none)

    Returns:
        (2, 4, 4) GHMM transition matrices in the extended Bloch basis
        {I/2, σ_x/2, σ_y/2, σ_z/2}.  Stationary vector ⟨⟨π| = (1,0,0,0).
    """
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    cf = jnp.cos(phi)
    sf = jnp.sin(phi)

    # T(x) = T_sym ± T_anti  (+ for x=0, - for x=1)
    # Entries derived from the PTM of each Kraus operator — all real arithmetic.
    T_sym = jnp.array([
        [0.5,  0.0,    0.0,    0.0    ],
        [0.0,  0.0,    0.0,   -sf / 2.],
        [0.0,  0.0,    0.0,    0.0    ],
        [0.0,  0.0,    0.0,    cf / 2.],
    ])

    T_anti = jnp.array([
        [0.0,          0.0,           0.0,       ct / 2.     ],
        [-ct * sf / 2., 0.0,  -st * cf / 2.,     0.0         ],
        [0.0,        st / 2.,          0.0,       0.0         ],
        [ct * cf / 2.,  0.0,  -st * sf / 2.,     0.0         ],
    ])

    T = jnp.stack([T_sym + T_anti, T_sym - T_anti], axis=0)  # (2, 4, 4)

    # Leakage: T_ε(x) = (1-ε)T(x) + (ε/|X|)|1⟩⟩⟨⟨π|
    if epsilon > 0:
        reset = jnp.zeros((4, 4)).at[0, 0].set(1.0)
        T = (1 - epsilon) * T + (epsilon / 2) * reset

    return T


HMM_MATRIX_FUNCTIONS["quantum_rrxor"] = quantum_rrxor
GHMM_MATRIX_FUNCTIONS["quantum_rrxor"] = quantum_rrxor


def linewrap(w_max: int = 8, token_chars=(1, 2, 3)) -> Array:
    """Line-wrap process: each char-token represents a fixed number of characters.

    `token_chars[k]` = number of characters token k represents. States S_0..S_{w_max}
    encode "i chars into the current line". From S_i the intended emission k occurs with
    prob 1/n_tok; if i + token_chars[k] <= w_max the chain emits k and moves to
    S_{i+token_chars[k]}, else it emits the newline token and parks in buffer state B_k.
    From B_k the chain deterministically emits k and advances to S_{token_chars[k]}, so the
    wrapped token's char-count is preserved.

    Vocab = n_tok + 1 (char tokens 0..n_tok-1 plus newline = token n_tok). Total states =
    w_max + 1 + n_tok (S_0..S_{w_max} plus one buffer per char token). S_0 is unreachable
    under the stationary/buffer start; kept for index clarity.

    Defaults (1,2,3) reproduce the original 1/2/3-char, vocab-4 process.

    Args:
        w_max: line width. Must be >= max(token_chars) so the largest token fits a fresh line.
        token_chars: per-token character counts (uniform emission, prob 1/len each).
    """
    token_chars = tuple(int(c) for c in token_chars)
    n_tok = len(token_chars)
    if w_max < max(token_chars):
        raise ValueError(f"linewrap requires w_max >= max(token_chars)={max(token_chars)}, got {w_max}")

    n_mid = w_max + 1              # S_0..S_{w_max}
    n_states = n_mid + n_tok       # + one buffer B_k per char token
    NL = n_tok                     # newline token id
    p = 1.0 / n_tok

    T = jnp.zeros((n_tok + 1, n_states, n_states))

    # Mid-line: from S_i, intended token k -> advance by token_chars[k] or wrap.
    for i in range(n_mid):
        for k in range(n_tok):
            dest = i + token_chars[k]
            if dest <= w_max:
                T = T.at[k, i, dest].set(p)
            else:
                T = T.at[NL, i, n_mid + k].set(p)

    # Buffer states: deterministically emit the deferred token, land token_chars[k] in.
    for k in range(n_tok):
        T = T.at[k, n_mid + k, token_chars[k]].set(1.0)

    return T


HMM_MATRIX_FUNCTIONS["linewrap"] = linewrap


# ---------------------------------------------------------------------------
# Programming toy (programming_minimal_model/): "wrapped" Mess3
# ---------------------------------------------------------------------------

def _pattern_automaton(pattern: tuple[int, ...], n_symbols: int) -> list[list[int]]:
    """KMP-style memory update for a designated symbol pattern.

    Memory m in 0..len(pattern)-1 is "the last m emitted symbols match the first m
    symbols of the pattern". next[m][k] is the memory after additionally seeing symbol
    k: the longest prefix of `pattern` that is a suffix of pattern[:m] + (k,). The
    value len(pattern) means the pattern just completed (the firing state).
    """
    length = len(pattern)
    table: list[list[int]] = []
    for m in range(length):
        row = []
        for k in range(n_symbols):
            seen = list(pattern[:m]) + [k]
            j = min(len(seen), length)
            while j > 0 and tuple(seen[-j:]) != tuple(pattern[:j]):
                j -= 1
            row.append(j)
        table.append(row)
    return table


def wrapped_mess3(
    x: float = 0.15,
    a: float = 0.6,
    response: int = 0,
    eps: float = 0.1,
    q_shared: float = 0.45,
    q_lang: float = 0.35,
    q_blank: float = 0.2,
    pattern=(0, 1),
) -> Array:
    """Mess3 wrapped by a pattern memory and a "language coin" -- one function of the toy.

    States are (Mess3 state s, memory m), indexed i = s * n_mem + m with
    n_mem = len(pattern) + 1. Memory m < len(pattern) tracks progress toward the
    designated `pattern` of emitted symbols (KMP automaton, so overlapping patterns are
    handled). Memory m == len(pattern) is the FIRING state: the pattern has just
    completed.

      ordinary step (m < len(pattern)):  Mess3 emits k and moves s -> s'; memory advances.
      firing step   (m == len(pattern)): the function emits its `response` symbol with
                                          prob 1 - eps, else one of the other symbols with
                                          prob eps / 2 each; Mess3's state does NOT move; the
                                          memory resets to 0 (a response never re-arms the
                                          pattern).

    The three functions of the toy are `response` = 0, 1, 2 and are identical except at
    firing steps. `eps = 0` makes the wrong response a forbidden word; `eps > 0` keeps full
    support.

    The language coin then splits every emitted symbol k into a shared token S:k
    (q_shared), a language token L:k (q_lang), or replaces the whole step by a blank B
    (q_blank) on which nothing moves. Local vocab: 0..2 = S:k, 3..5 = L:k, 6 = B. Which
    language the L-tokens belong to is decided outside this function, by the mixture's
    `vocab_maps` (see nonergodic_12x_wrapped_mess3.yaml).

    Initial state: this chain's stationary distribution puts mass on the armed and firing
    memory states, which makes the first step ambiguous and breaks the exact factorization
    of the observer's belief. Start it at Mess3-stationary (uniform) with memory 0, i.e.
    mass 1/3 on each i = s * n_mem. The process config sets that explicitly.

    Reference construction and belief-factorization checks:
    programming_minimal_model/Programs_Are_Factored_And_We_Can_Do_Many_Things/notes/wrapped_mess3_hmm.py
    """
    import numpy as np
    from genproc.generative_processes.transition_matrices import mess3

    n_sym = 3
    pattern = tuple(int(p) for p in pattern)
    if len(pattern) == 0 or any(p < 0 or p >= n_sym for p in pattern):
        raise ValueError(f"pattern must be a non-empty tuple of symbols in 0..{n_sym - 1}, got {pattern}")
    if not (0 <= response < n_sym):
        raise ValueError(f"response must be in 0..{n_sym - 1}, got {response}")
    if not (0.0 <= eps <= 1.0):
        raise ValueError(f"eps must be in [0, 1], got {eps}")
    if abs(q_shared + q_lang + q_blank - 1.0) > 1e-9 or min(q_shared, q_lang, q_blank) < 0:
        raise ValueError(f"q_shared + q_lang + q_blank must be 1 with all >= 0, got {(q_shared, q_lang, q_blank)}")

    T = np.asarray(mess3(x, a))                    # [symbol, from, to]
    length = len(pattern)
    n_mem = length + 1
    n_states = n_sym * n_mem
    nxt = _pattern_automaton(pattern, n_sym)

    core = np.zeros((n_sym, n_states, n_states))
    for s in range(n_sym):
        for m in range(n_mem):
            i = s * n_mem + m
            if m < length:
                for k in range(n_sym):
                    for s2 in range(n_sym):
                        core[k, i, s2 * n_mem + nxt[m][k]] += T[k, s, s2]
            else:
                for k in range(n_sym):
                    p = 1.0 - eps if k == response else eps / (n_sym - 1)
                    core[k, i, s * n_mem + 0] += p

    out = np.zeros((2 * n_sym + 1, n_states, n_states))
    out[0:n_sym] = q_shared * core
    out[n_sym:2 * n_sym] = q_lang * core
    out[2 * n_sym] = q_blank * np.eye(n_states)
    return jnp.asarray(out)


HMM_MATRIX_FUNCTIONS["wrapped_mess3"] = wrapped_mess3


# ---------------------------------------------------------------------------
# Program toy, second process: a function *generator* called under a language machine.
# ---------------------------------------------------------------------------

def _kmp_table(pattern: tuple[int, ...], n_symbols: int) -> list[list[int]]:
    """Watcher progress after seeing symbol k from progress m, for m = 0..len(pattern).

    Like `_pattern_automaton` but with one more row: row len(pattern) is the completed state
    continuing *without* a fire (a miss), which keeps matching in the ordinary KMP way.
    """
    length = len(pattern)
    table: list[list[int]] = []
    for m in range(length + 1):
        row = []
        for k in range(n_symbols):
            seen = list(pattern[:m]) + [k]
            j = min(len(seen), length)
            while j > 0 and tuple(seen[-j:]) != tuple(pattern[:j]):
                j -= 1
            row.append(j)
        table.append(row)
    return table


PROGRAM_GENERATORS = ("mess3", "no_repeat", "sum_mod3", "diff_mod3")


def _program_generator(generator: str, x: float, a: float, rho: float, delta: float):
    """One of the toy's three function generators over the symbols 0, 1, 2.

    Returns (T, start) with T[k, g, g'] = P(emit k, go to g' | g) and `start` the state
    distribution a call begins from (also the reset target of a fire).

      mess3     -- Mess3(x, a); start = its stationary (uniform) distribution.
      no_repeat -- order-1 Markov chain on the last symbol: repeat it w.p. `rho`, otherwise
                   uniform over the other two. State 3 is the start state, which emits
                   uniformly. rho = 0 makes a repeat a forbidden word.
      sum_mod3  -- period-3 arithmetic: emit a uniform, then b uniform, then (a + b) mod 3 w.p.
                   1 - delta (else uniform over the other two). States: 0 = expecting a,
                   1 + a = holding a, 4 + c = holding the answer c. Start = state 0.
      diff_mod3 -- as sum_mod3 with the answer (a - b) mod 3 (the held-out "new algorithm").
    """
    import numpy as np

    if generator == "mess3":
        from genproc.generative_processes.transition_matrices import mess3

        T = np.asarray(mess3(x, a), dtype=np.float64)
        return T, np.full(3, 1.0 / 3.0)
    if generator == "no_repeat":
        T = np.zeros((3, 4, 4))
        for v in range(3):
            T[v, 3, v] = 1.0 / 3.0
            for i in range(3):
                T[v, i, v] = rho if v == i else (1.0 - rho) / 2.0
        start = np.zeros(4)
        start[3] = 1.0
        return T, start
    if generator == "sum_mod3":
        T = np.zeros((3, 7, 7))
        for v in range(3):
            T[v, 0, 1 + v] = 1.0 / 3.0
            for arg in range(3):
                T[v, 1 + arg, 4 + (arg + v) % 3] = 1.0 / 3.0
            for c in range(3):
                T[v, 4 + c, 0] = (1.0 - delta) if v == c else delta / 2.0
        start = np.zeros(7)
        start[0] = 1.0
        return T, start
    if generator == "diff_mod3":
        # Same seven-state shape as sum_mod3 with the answer (a - b) mod 3: the "new algorithm" of the
        # fine-tuning transfer experiment (a twist on a known computation, like the paper's vantril).
        T = np.zeros((3, 7, 7))
        for v in range(3):
            T[v, 0, 1 + v] = 1.0 / 3.0
            for arg in range(3):
                T[v, 1 + arg, 4 + (arg - v) % 3] = 1.0 / 3.0
            for c in range(3):
                T[v, 4 + c, 0] = (1.0 - delta) if v == c else delta / 2.0
        start = np.zeros(7)
        start[0] = 1.0
        return T, start
    raise ValueError(f"generator must be one of {PROGRAM_GENERATORS}, got {generator!r}")


def program_call(
    generator: str = "mess3",
    x: float = 0.15,
    a: float = 0.6,
    rho: float = 0.05,
    delta: float = 0.1,
    pattern=(0, 1),
    p_fire: float = 0.9,
    p_floor: float = 0.01,
    q_shared: float = 0.8,
    p_content_to_filler: float = 1.0 / 24.0,
    p_filler_to_content: float = 1.0 / 8.0,
) -> Array:
    """A function generator called under a two-mode language machine -- one component of the
    program toy (second design; the first was `wrapped_mess3`).

    The *language machine* is the parent: a two-state chain, content / filler.
      content: the function machine takes one step (below) and the emitted symbol is rendered
               as a shared token S:k (prob q_shared) or a language token L:k (1 - q_shared).
               Then content -> filler w.p. p_content_to_filler.
      filler:  emits the blank token B; the function machine is frozen (identity).
               Then filler -> content w.p. p_filler_to_content.
    Which language the L-tokens belong to is decided outside, by the mixture's vocab_maps.

    The *function machine* is the child: a generator (see `_program_generator`) plus a shared
    watcher for `pattern` (KMP progress m, so overlapping patterns are handled). On a content
    step it either FIRES -- emits the return token (rendered S:F or L:F like a symbol) and
    resets to its start distribution with the watcher at 0, a rank-one transition -- with
    probability p_fire if the watcher has just completed the pattern and p_floor otherwise;
    or (1 - that) emits a symbol from the generator and advances the watcher.

    Every content token has positive probability at every content step (full support),
    so any constructed content prefix is in-distribution for every function; the functions
    differ only in the statistics of the symbols between returns.

    Local vocab (9): 0..2 = S:k, 3 = S:F, 4..6 = L:k, 7 = L:F, 8 = B.
    State index: i = (mode * S + g) * n_mem + m, mode 0 = content, 1 = filler, g the
    generator state (S of them), m the watcher progress (n_mem = len(pattern) + 1).
    Start: content mode, generator start distribution, watcher 0 -- see
    `program_call_initial_state`; the config sets it explicitly.
    """
    import numpy as np

    n_sym = 3
    pattern = tuple(int(p) for p in pattern)
    if len(pattern) == 0 or any(p < 0 or p >= n_sym for p in pattern):
        raise ValueError(f"pattern must be a non-empty tuple of symbols in 0..{n_sym - 1}, got {pattern}")
    for name, val in (("p_fire", p_fire), ("p_floor", p_floor), ("q_shared", q_shared), ("rho", rho),
                      ("delta", delta), ("p_content_to_filler", p_content_to_filler),
                      ("p_filler_to_content", p_filler_to_content)):
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {val}")

    T, start = _program_generator(generator, x, a, rho, delta)
    n_gen = T.shape[1]
    length = len(pattern)
    n_mem = length + 1
    nxt = _kmp_table(pattern, n_sym)
    n_states = 2 * n_gen * n_mem
    s_sym, s_fire, l_sym, l_fire, blank = 0, 3, 4, 7, 8
    content, filler = 0, 1

    def idx(mode: int, g: int, m: int) -> int:
        return (mode * n_gen + g) * n_mem + m

    out = np.zeros((9, n_states, n_states))
    p_cf, p_fc = p_content_to_filler, p_filler_to_content
    for g in range(n_gen):
        for m in range(n_mem):
            i = idx(content, g, m)
            hazard = p_fire if m == length else p_floor
            for mode2, pm in ((content, 1.0 - p_cf), (filler, p_cf)):
                for g2 in range(n_gen):
                    out[s_fire, i, idx(mode2, g2, 0)] += hazard * q_shared * start[g2] * pm
                    out[l_fire, i, idx(mode2, g2, 0)] += hazard * (1.0 - q_shared) * start[g2] * pm
                    for k in range(n_sym):
                        j = idx(mode2, g2, nxt[m][k])
                        out[s_sym + k, i, j] += (1.0 - hazard) * q_shared * T[k, g, g2] * pm
                        out[l_sym + k, i, j] += (1.0 - hazard) * (1.0 - q_shared) * T[k, g, g2] * pm
            i = idx(filler, g, m)
            out[blank, i, idx(filler, g, m)] += 1.0 - p_fc
            out[blank, i, idx(content, g, m)] += p_fc
    return jnp.asarray(out)


def program_call_initial_state(
    generator: str = "mess3", x: float = 0.15, a: float = 0.6, rho: float = 0.05, delta: float = 0.1, pattern=(0, 1)
) -> list[float]:
    """Start of a `program_call` component: content mode, generator start, watcher 0."""
    import numpy as np

    _, start = _program_generator(generator, x, a, rho, delta)
    n_gen = start.shape[0]
    n_mem = len(tuple(pattern)) + 1
    init = np.zeros(2 * n_gen * n_mem)
    for g in range(n_gen):
        init[g * n_mem] = start[g]
    return [float(v) for v in init]


HMM_MATRIX_FUNCTIONS["program_call"] = program_call

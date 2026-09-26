"""Compute position-specific Bayes-optimal loss for generative processes.

This module computes H(X_t | X_{0:t-1}) for each position t — the conditional
entropy of the next token given the prefix, which is the theoretical minimum
achievable loss. See docs/optimal_loss_derivation.md for the full derivation.
"""

from typing import Any

import jax
import jax.numpy as jnp

from genproc.generative_processes.generative_process import GenerativeProcess
from genproc.generative_processes.nonergodic_generative_process import NonErgodicState

from utils import expand_state_by_batch_size


def _extract_state_at_time(belief_states: Any, t: int) -> Any:
    """Extract the belief state at timestep t from a trajectory.

    Handles NonErgodicState (NamedTuple), plain tuples, and arrays.
    """
    if isinstance(belief_states, NonErgodicState):
        # NonErgodicState: slice both component_beliefs and component_states
        comp_beliefs_t = belief_states.component_beliefs[:, t, ...]
        comp_states_t = tuple(
            _extract_state_at_time(cs, t) for cs in belief_states.component_states
        )
        return NonErgodicState(
            component_beliefs=comp_beliefs_t,
            component_states=comp_states_t,
        )
    elif isinstance(belief_states, tuple):
        return tuple(b[:, t, ...] for b in belief_states)
    else:
        return belief_states[:, t, ...]


def compute_eval_set_specific_optimal_loss(
    generative_process: GenerativeProcess,
    belief_states: Any,
    labels_jax: jax.Array,
) -> dict[str, Any]:
    """Eval-set-specific Bayes-optimal loss on a fixed batch of (state, token) pairs.

    For each position t, computes -log P_true(labels[:, t] | belief_states[:, t]),
    averaged over the batch, then averaged across positions. Unlike
    `compute_position_optimal_loss` — which estimates the process-expected entropy
    H(X_t | state_t) on fresh MC samples — this quantity is the *actual* lower
    bound a Bayes-optimal observer can achieve on the specific tokens that were
    drawn. On any finite eval batch the two values differ by sampling noise.

    Args:
        generative_process: genproc generative process used to produce the batch.
        belief_states: belief-state trajectory from
            `generate_data_batch_with_full_history`. Convention:
            `belief_states[:, t]` emitted `labels[:, t]`.
        labels_jax: `(batch, n_positions)` jax.Array of emitted tokens.

    Returns:
        Dict with `per_position_loss` (list[float]) and `average_loss` (float).
    """
    n_positions = int(labels_jax.shape[1])
    batch_size = int(labels_jax.shape[0])
    per_position_losses: list[float] = []
    for t in range(n_positions):
        state_t = _extract_state_at_time(belief_states, t)
        obs_probs = jax.vmap(generative_process.observation_probability_distribution)(state_t)
        p_actual = obs_probs[jnp.arange(batch_size), labels_jax[:, t]]
        per_position_losses.append(float(jnp.mean(-jnp.log(p_actual + 1e-30))))
    average_loss = float(sum(per_position_losses) / max(1, n_positions))
    return {
        "per_position_loss": per_position_losses,
        "average_loss": average_loss,
    }


def _compute_per_position_losses(
    generative_process: GenerativeProcess,
    num_samples: int,
    sequence_len: int,
    key: jax.Array,
) -> tuple[list[float], jax.Array]:
    """Run a single MC batch and return per-position mean losses and tokens.

    Uses the entropy of the predictive distribution (Rao-Blackwellized estimator)
    rather than scoring individual tokens, for lower variance.
    """
    initial_states = expand_state_by_batch_size(
        generative_process.initial_state, num_samples
    )
    batch_keys = jax.random.split(key, num_samples)
    belief_states, tokens = generative_process.generate(
        initial_states, batch_keys, sequence_len, True
    )

    per_position_losses = []
    for t in range(sequence_len):
        state_t = _extract_state_at_time(belief_states, t)
        obs_probs = jax.vmap(generative_process.observation_probability_distribution)(state_t)
        entropies = -jnp.sum(jax.scipy.special.xlogy(obs_probs, obs_probs), axis=-1)
        per_position_losses.append(float(jnp.mean(entropies)))

    return per_position_losses, tokens


def compute_position_optimal_loss(
    generative_process: GenerativeProcess,
    sequence_len: int,
    seed: int,
    *,
    min_samples: int = 4096,
    max_samples: int = 2**20,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compute position-specific Bayes-optimal loss with adaptive sample count.

    Doubles the batch size each iteration, accumulating a cumulative mean across
    all batches. Stops when the cumulative mean changes by less than tolerance
    between doublings, or max_samples total is reached.

    Args:
        generative_process: The generative process to analyze
        sequence_len: Length of sequences to generate
        seed: Random seed
        min_samples: Starting batch size (doubled each iteration)
        max_samples: Upper bound on total accumulated samples
        tolerance: Stop when cumulative mean changes by less than this (nats)

    Returns:
        Dictionary with per_position_loss, average_loss, asymptotic_estimate,
        position_1_loss, num_positions, marginal_loss, uniform_loss, num_samples.
    """
    batch_size = min_samples
    key = jax.random.PRNGKey(seed)
    total_samples = 0
    cumulative_position_sums = [0.0] * sequence_len
    prev_avg = None
    last_tokens = None

    while True:
        key, subkey = jax.random.split(key)
        per_position_losses, tokens = _compute_per_position_losses(
            generative_process, batch_size, sequence_len, subkey,
        )
        last_tokens = tokens

        # Update cumulative weighted sums
        for t in range(sequence_len):
            cumulative_position_sums[t] = (
                cumulative_position_sums[t] * total_samples
                + per_position_losses[t] * batch_size
            ) / (total_samples + batch_size)
        total_samples += batch_size

        average_loss = sum(cumulative_position_sums) / sequence_len

        if prev_avg is not None:
            delta = abs(average_loss - prev_avg)
            print(f"  n={total_samples:>7d}: avg_loss={average_loss:.6f} (delta={delta:.6f})")
            if delta < tolerance:
                break
        else:
            print(f"  n={total_samples:>7d}: avg_loss={average_loss:.6f}")

        prev_avg = average_loss

        next_batch = batch_size * 2
        if total_samples + next_batch > max_samples:
            print(f"  Reached max_samples={max_samples}, stopping")
            break
        batch_size = next_batch

    # Compute baseline losses from the final (largest) batch
    vocab_size = generative_process.vocab_size
    flat_tokens = last_tokens.flatten()
    counts = jnp.zeros(vocab_size).at[flat_tokens].add(1.0)
    marginal_probs = counts / counts.sum()
    marginal_loss = float(-jnp.mean(jnp.log(marginal_probs[flat_tokens] + 1e-10)))
    uniform_loss = float(jnp.log(vocab_size))

    return {
        "per_position_loss": cumulative_position_sums,
        "average_loss": average_loss,
        "asymptotic_estimate": cumulative_position_sums[-1],
        "position_1_loss": cumulative_position_sums[0],
        "num_positions": sequence_len,
        "marginal_loss": marginal_loss,
        "uniform_loss": uniform_loss,
        "num_samples": total_samples,
    }

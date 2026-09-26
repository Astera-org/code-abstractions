"""Utility functions for factored nonergodic training."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from genproc.generative_processes.nonergodic_generative_process import NonErgodicState


def expand_state_by_batch_size(
    state: jax.Array | tuple[jax.Array, ...] | NonErgodicState,
    batch_size: int,
) -> jax.Array | tuple[jax.Array, ...] | NonErgodicState:
    """Expand a single state to a batch of identical states.

    Handles flat arrays, tuples of arrays, and NonErgodicState.
    """
    if isinstance(state, NonErgodicState):
        return NonErgodicState(
            component_beliefs=jnp.repeat(state.component_beliefs[None, :], batch_size, axis=0),
            component_states=tuple(
                expand_state_by_batch_size(cs, batch_size) for cs in state.component_states
            ),
        )
    elif isinstance(state, tuple):
        return tuple(jnp.repeat(s[None, :], batch_size, axis=0) for s in state)
    else:
        return jnp.repeat(state[None, :], batch_size, axis=0)


def flatten_nonergodic_beliefs(
    belief_states: NonErgodicState,
) -> tuple[np.ndarray, ...]:
    """Flatten NonErgodicState into weighted per-factor beliefs for the activation tracker.

    Multiplies each component's mixing weight into its per-factor beliefs,
    producing unnormalized beliefs that encode both component identity and
    within-component state in a single representation.

    For a 3-state factor in component k with weight w_k:
        weighted_belief = [w_k * p0, w_k * p1, w_k * p2]  (sums to w_k)

    Returns:
        Tuple of numpy arrays:
        (w0*C0_F0, w0*C0_F1, ..., w1*C1_F0, w1*C1_F1, ...)
    """
    component_beliefs = np.asarray(belief_states.component_beliefs)
    result = []

    for comp_idx, comp_states in enumerate(belief_states.component_states):
        # w_k has shape [..., 1] for broadcasting against factor dims
        w_k = component_beliefs[..., comp_idx:comp_idx + 1]

        if isinstance(comp_states, tuple):
            for factor_state in comp_states:
                result.append(np.asarray(factor_state) * w_k)
        else:
            result.append(np.asarray(comp_states) * w_k)

    return tuple(result)


def slice_nonergodic_belief_states(
    belief_states: NonErgodicState,
    seq_slice: slice,
) -> NonErgodicState:
    """Properly slice both component_beliefs AND component_states along seq dim.

    The library's _slice_belief_states only slices component_beliefs for
    NonErgodicState, but when return_all_states=True, component_states
    also have a sequence dimension from jax.lax.scan.
    """

    def _slice_component_state(
        state: Any, seq_slice: slice
    ) -> Any:
        if isinstance(state, tuple):
            return tuple(s[:, seq_slice, ...] for s in state)
        else:
            return state[:, seq_slice, ...]

    return NonErgodicState(
        component_beliefs=belief_states.component_beliefs[:, seq_slice, ...],
        component_states=tuple(
            _slice_component_state(cs, seq_slice) for cs in belief_states.component_states
        ),
    )


def get_belief_factor_info(
    belief_states: NonErgodicState,
) -> tuple[int, list[int]]:
    """Get the number of belief factors and their dimensions.

    Returns:
        (num_factors, factor_dims) — one entry per component per factor.
        Matches the output of flatten_nonergodic_beliefs (weighted beliefs).
    """
    factor_dims = []
    for comp_states in belief_states.component_states:
        if isinstance(comp_states, tuple):
            for factor_state in comp_states:
                factor_dims.append(factor_state.shape[-1])
        else:
            factor_dims.append(comp_states.shape[-1])

    return len(factor_dims), factor_dims


def get_component_factor_metadata(
    belief_states: NonErgodicState,
) -> tuple[list[str], list[tuple[str, list[int]]]]:
    """Derive factor names and component groupings from NonErgodicState structure.

    Returns:
        factor_names: Human-readable names for each factor, e.g.
            ["C0.F0", "C0.F1", "C1.F0", "C1.F1"]
        component_groups: List of (group_label, [factor_indices]) for visual grouping, e.g.
            [("C0", [0, 1]), ("C1", [2, 3])]
    """
    factor_names: list[str] = []
    component_groups: list[tuple[str, list[int]]] = []

    factor_idx = 0
    for comp_idx, comp_states in enumerate(belief_states.component_states):
        if isinstance(comp_states, tuple):
            group_indices = []
            for sub_idx in range(len(comp_states)):
                factor_names.append(f"C{comp_idx}.F{sub_idx}")
                group_indices.append(factor_idx)
                factor_idx += 1
            component_groups.append((f"C{comp_idx}", group_indices))
        else:
            factor_names.append(f"C{comp_idx}")
            component_groups.append((f"C{comp_idx}", [factor_idx]))
            factor_idx += 1

    return factor_names, component_groups


def project_to_simplex_plane(beliefs: np.ndarray) -> np.ndarray:
    """Project beliefs to (d-1)-dim by centering and dropping the last coordinate.

    For normalized beliefs (sum=1), this removes the constraint direction.
    For weighted beliefs (sum=w_k), the projected coordinates scale with w_k,
    preserving mixing weight information in the magnitude.

    Args:
        beliefs: Array of shape [..., d].

    Returns:
        Array of shape [..., d-1].
    """
    centered = beliefs - beliefs.mean(axis=-1, keepdims=True)
    return centered[..., :-1]


def get_factor_eigvecs(generative_process) -> list[np.ndarray] | None:
    """Return per-factor eigvec for each factor in flatten_nonergodic_beliefs order.

    - HMM (non-generalized, non-factored): eigvec = np.ones(dim)
    - GHMM: eigvec = component.normalizing_eigenvector
    - Factored: eigvec = component.normalizing_eigenvectors[fi][0]
    """
    from genproc.generative_processes.nonergodic_generative_process import NonErgodicGenerativeProcess
    from genproc.generative_processes.factored_generative_process import FactoredGenerativeProcess
    if not isinstance(generative_process, NonErgodicGenerativeProcess):
        return None
    eigvecs = []
    init_state = generative_process.initial_state
    for component, comp_states in zip(generative_process.components, init_state.component_states):
        if isinstance(component, FactoredGenerativeProcess) and isinstance(comp_states, tuple):
            for fi in range(len(comp_states)):
                ev = np.asarray(component.normalizing_eigenvectors[fi][0]).flatten()
                eigvecs.append(ev)
        else:
            sv = np.asarray(comp_states).flatten()
            if "Generalized" in type(component).__name__:
                eigvecs.append(np.asarray(component.normalizing_eigenvector).flatten())
            else:
                eigvecs.append(np.ones(len(sv)))
    return eigvecs


def project_nonergodic_beliefs(
    beliefs_tuple: tuple[np.ndarray, ...],
    factor_dims: list[int],
) -> tuple[tuple[np.ndarray, ...], list[int]]:
    """Project weighted per-factor beliefs to simplex plane.

    Applies simplex-plane projection to all factors. For weighted beliefs
    (sum = w_k), centering preserves mixing weight info in the magnitude
    of the projected coordinates.

    Args:
        beliefs_tuple: Weighted beliefs from flatten_nonergodic_beliefs().
        factor_dims: Dimensions of each factor (from get_belief_factor_info()).

    Returns:
        (projected_tuple, projected_dims) where:
            projected_tuple: tuple of projected arrays (one per factor)
            projected_dims: list of projected dimensions (each is original dim - 1)
    """
    projected = []
    projected_dims = []
    for i in range(len(beliefs_tuple)):
        proj = project_to_simplex_plane(np.asarray(beliefs_tuple[i]))
        projected.append(proj)
        projected_dims.append(factor_dims[i] - 1)
    return tuple(projected), projected_dims

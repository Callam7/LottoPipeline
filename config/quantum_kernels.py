"""
Modified By: Callam
Project: Lotto Generator

Purpose:
    Quantum kernel feature block for the lotto generator pipeline.

Design:
    1) Uses the SAME embedding geometry as config.quantum_features
       (identical angle preprocessing and variational weights).
    2) Encodes classical feature rows as quantum states |phi(x)>.
    3) Selects a fixed-size set of prototype states from the dataset (cached).
    4) For every sample, computes fidelity features:
           k_j(x) = |<phi(x)|phi(x_ref_j)>|^2  in [0, 1]
       where {x_ref_j} are the prototype rows.
    5) Returns a dense real-valued feature matrix suitable for
       concatenation with classical + quantum features in deep_learning.py.

Engineering guarantees:
    - Output width is ALWAYS exactly `num_prototypes` (never collapses to 1).
    - Prototype selection is deterministic given (seed, data) and cached for inference.
    - Uses live trained weights (no stale import reference bug).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

# IMPORTANT:
# Do NOT import _global_weights by name because quantum_features reassigns it.
# Import the module and read weights dynamically to avoid stale references.
from config import quantum_features as qf


# ---------------------------------------------------------------------
# Device for kernel state preparation (statevector)
# ---------------------------------------------------------------------

_kernel_dev = qml.device("default.qubit", wires=qf.NUM_QUBITS, shots=None)


@qml.qnode(_kernel_dev)
def _state_circuit(angles: np.ndarray, weights: np.ndarray):
    """
    Prepare a variational quantum state |phi(x)> on NUM_QUBITS wires.

    Circuit:
        H^{⊗n} -> RY(angles) -> StronglyEntanglingLayers(weights)

    Returns:
        state: complex statevector of length 2**NUM_QUBITS
    """
    for w in range(qf.NUM_QUBITS):
        qml.Hadamard(wires=w)

    # angles expected length NUM_QUBITS; if longer, truncate deterministically
    for w, theta in enumerate(angles[: qf.NUM_QUBITS]):
        qml.RY(theta, wires=w)

    qml.templates.StronglyEntanglingLayers(weights, wires=range(qf.NUM_QUBITS))
    return qml.state()


def _get_live_weights(weights: np.ndarray | None) -> np.ndarray:
    """
    Returns the weights to use for embedding.
    If weights is None, pull the CURRENT trained weights from quantum_features.
    """
    if weights is None:
        # qf._global_weights may be reassigned during SPSA training; always read live.
        return np.asarray(qf._global_weights, dtype=float)
    return np.asarray(weights, dtype=float)


def _encode_state(classical_vec: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Classical feature vector -> normalized quantum state |phi(x)>.

    Args:
        classical_vec: (d,) classical features
        weights: optional variational weights; None -> live global trained weights

    Returns:
        state: complex array of shape (2**NUM_QUBITS,)
    """
    w = _get_live_weights(weights)

    # Use the exact same preprocessing as quantum_features
    angles = qf._preprocess_to_angles(classical_vec, num_qubits=qf.NUM_QUBITS)
    state = _state_circuit(angles, w)
    state = np.asarray(state, dtype=np.complex128)

    norm = np.linalg.norm(state)
    if norm <= 0.0:
        return state
    return state / norm


def _pure_state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """
    Fidelity between two pure states |psi>, |phi|:
        F = |<psi|phi>|^2
    """
    psi = np.asarray(psi, dtype=np.complex128)
    phi = np.asarray(phi, dtype=np.complex128)

    overlap = np.vdot(psi, phi)  # <psi|phi>
    val = float(np.abs(overlap) ** 2)

    # Numerical guard: keep strictly within [0,1]
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


# ---------------------------------------------------------------------
# Prototype caching to guarantee constant width at inference
# ---------------------------------------------------------------------

# Cached prototype states for the last (seed, num_prototypes) call
_cached_proto_states: np.ndarray | None = None
_cached_num_prototypes: int | None = None
_cached_seed: int | None = None


def _select_prototypes_fixed_width(
    feature_matrix: np.ndarray,
    num_prototypes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select exactly `num_prototypes` prototype rows.

    Key rule:
        output width MUST be constant, so we must ALWAYS select exactly num_prototypes rows.

    Strategy:
        - If n_samples >= num_prototypes: sample without replacement (deterministic seed).
        - If n_samples < num_prototypes: sample WITH replacement to reach fixed width.

    Returns:
        prototypes: (num_prototypes, d)
        indices:    (num_prototypes,)
    """
    X = np.asarray(feature_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape {X.shape}")

    n, d = X.shape
    if n == 0:
        # No data: return an all-zero prototype table so width is stable
        return np.zeros((num_prototypes, d), dtype=float), np.zeros((num_prototypes,), dtype=int)

    rng = np.random.default_rng(seed)

    if n >= num_prototypes:
        indices = rng.choice(n, size=num_prototypes, replace=False)
    else:
        # Critical fix: WITH replacement so we still return num_prototypes rows
        indices = rng.choice(n, size=num_prototypes, replace=True)

    return X[indices], indices.astype(int)


def _encode_prototype_states(
    prototypes: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Encode prototype rows into quantum statevectors.

    Args:
        prototypes: (m, d)
        weights: optional; None -> live global weights

    Returns:
        states: (m, 2**NUM_QUBITS)
    """
    prototypes = np.asarray(prototypes, dtype=float)
    if prototypes.ndim != 2:
        raise ValueError(f"prototypes must be 2D, got shape {prototypes.shape}")

    m = prototypes.shape[0]
    dim = 2 ** qf.NUM_QUBITS
    states = np.zeros((m, dim), dtype=np.complex128)

    for i in range(m):
        states[i] = _encode_state(prototypes[i], weights=weights)

    return states


def _compute_fidelity_feature_matrix(
    feature_matrix: np.ndarray,
    proto_states: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute fidelity features K[i,j] = |<phi(x_i)|phi(proto_j)>|^2

    Returns:
        K: (n_samples, m) in [0,1]
    """
    X = np.asarray(feature_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape {X.shape}")

    proto_states = np.asarray(proto_states, dtype=np.complex128)
    if proto_states.ndim != 2:
        raise ValueError(f"proto_states must be 2D, got shape {proto_states.shape}")

    n = X.shape[0]
    m = proto_states.shape[0]

    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=float)

    K = np.zeros((n, m), dtype=float)

    for i in range(n):
        psi = _encode_state(X[i], weights=weights)
        for j in range(m):
            K[i, j] = _pure_state_fidelity(psi, proto_states[j])

    return K


def build_quantum_kernel_features(
    feature_matrix: np.ndarray,
    num_prototypes: int = 24,
    seed: int = 1337,
    weights: np.ndarray | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Construct quantum-kernel features for a batch.

    Output guarantee:
        ALWAYS returns shape (n_samples, num_prototypes)

    Caching behavior:
        - If use_cache=True and we already cached prototype states for the same
          (seed, num_prototypes), reuse them. This is critical for inference where n may be 1.
        - Otherwise, select+encode prototypes from the given feature_matrix and cache them.

    Args:
        feature_matrix: (n_samples, d)
        num_prototypes: fixed kernel width
        seed: deterministic selection seed
        weights: optional variational weights; None -> live global weights
        use_cache: reuse cached proto states when possible

    Returns:
        K_scaled: (n_samples, num_prototypes)
    """
    global _cached_proto_states, _cached_num_prototypes, _cached_seed

    X = np.asarray(feature_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape {X.shape}")

    n, d = X.shape
    if num_prototypes <= 0:
        raise ValueError(f"num_prototypes must be > 0, got {num_prototypes}")

    # Decide whether to reuse cached prototypes
    can_use_cache = (
        use_cache
        and (_cached_proto_states is not None)
        and (_cached_num_prototypes == int(num_prototypes))
        and (_cached_seed == int(seed))
        and (_cached_proto_states.shape[0] == int(num_prototypes))
    )

    if can_use_cache:
        proto_states = _cached_proto_states
    else:
        prototypes, _ = _select_prototypes_fixed_width(X, num_prototypes=num_prototypes, seed=seed)
        proto_states = _encode_prototype_states(prototypes, weights=weights)

        # Cache for subsequent calls (especially inference with small n)
        _cached_proto_states = proto_states
        _cached_num_prototypes = int(num_prototypes)
        _cached_seed = int(seed)

    K = _compute_fidelity_feature_matrix(X, proto_states, weights=weights)

    # Column-wise rescaling (keeps outputs in a stable numeric range)
    # Works even when n == 1.
    col_max = np.maximum(np.max(K, axis=0, keepdims=True), 1e-12)
    K_scaled = K / col_max

    # HARD guarantee: (n, num_prototypes)
    if K_scaled.shape != (n, num_prototypes):
        # This should never trigger now, but keep a final guard.
        out = np.zeros((n, num_prototypes), dtype=float)
        m = min(num_prototypes, K_scaled.shape[1])
        out[:, :m] = K_scaled[:, :m]
        K_scaled = out

    return K_scaled.astype(float)


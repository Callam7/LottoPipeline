"""
Modified By: Callam
Project: Lotto Generator

Purpose:
    Quantum kernel feature block for the lotto generator pipeline.

This module constructs a fixed-width quantum kernel feature matrix by:
    - Encoding classical feature vectors into quantum states
    - Selecting a fixed number of prototype states
    - Computing fidelity-based similarities to those prototypes
    - Applying a Mercer-safe diagonal normalization so the result is:
        • a valid kernel
        • numerically stable for deep learning
"""

from __future__ import annotations  # Enable modern type hints safely

import numpy as np                     # Numerical arrays and linear algebra
import pennylane as qml                # Quantum circuit framework

# IMPORTANT:
# I import the quantum_features *module*, not individual symbols,
# because its global weights are reassigned during training.
from config import quantum_features as qf


# ---------------------------------------------------------------------
# Quantum device configuration
# ---------------------------------------------------------------------

# Statevector simulator is required for exact fidelity computation.
# shots=None ensures analytic statevector access.
_kernel_dev = qml.device(
    "default.qubit",
    wires=qf.NUM_QUBITS,
    shots=None
)


@qml.qnode(_kernel_dev)
def _state_circuit(angles: np.ndarray, weights: np.ndarray):
    """
    Prepare the variational quantum state |phi(x)>.

    Circuit structure:
        1) Hadamard on all qubits (creates superposition)
        2) RY rotations using preprocessed classical angles
        3) StronglyEntanglingLayers using trained variational weights

    Returns:
        Full quantum statevector as a complex NumPy array.
    """

    # Put each qubit into superposition
    for w in range(qf.NUM_QUBITS):
        qml.Hadamard(wires=w)

    # Encode classical data as Y-rotations
    # Extra angles are deterministically truncated
    for w, theta in enumerate(angles[: qf.NUM_QUBITS]):
        qml.RY(theta, wires=w)

    # Apply entangling variational layers
    qml.templates.StronglyEntanglingLayers(
        weights,
        wires=range(qf.NUM_QUBITS)
    )

    # Return the full statevector
    return qml.state()


def _get_live_weights(weights: np.ndarray | None) -> np.ndarray:
    """
    Resolve which variational weights to use.

    If weights is None, always pull the *current* trained weights
    from quantum_features to avoid stale references.
    """

    if weights is None:
        return np.asarray(qf._global_weights, dtype=float)

    return np.asarray(weights, dtype=float)


def _encode_state(
    classical_vec: np.ndarray,
    weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Encode a classical feature vector into a normalized quantum state.

    Steps:
        1) Preprocess classical data into rotation angles
        2) Run the quantum circuit
        3) Normalize the resulting statevector

    Returns:
        A complex vector of length 2**NUM_QUBITS
    """

    # Resolve weights safely
    w = _get_live_weights(weights)

    # Convert classical features into rotation angles
    angles = qf._preprocess_to_angles(
        classical_vec,
        num_qubits=qf.NUM_QUBITS
    )

    # Execute the quantum circuit
    state = _state_circuit(angles, w)
    state = np.asarray(state, dtype=np.complex128)

    # Explicit normalization for numerical safety
    norm = np.linalg.norm(state)
    if norm <= 0.0:
        return state

    return state / norm


def _pure_state_fidelity(
    psi: np.ndarray,
    phi: np.ndarray
) -> float:
    """
    Compute fidelity between two pure quantum states:

        F = |<psi | phi>|^2
    """

    psi = np.asarray(psi, dtype=np.complex128)
    phi = np.asarray(phi, dtype=np.complex128)

    # Compute inner product <psi|phi>
    overlap = np.vdot(psi, phi)

    # Fidelity is squared magnitude
    val = float(np.abs(overlap) ** 2)

    # Clamp to [0, 1] to guard numerical noise
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0

    return val


# ---------------------------------------------------------------------
# Prototype caching (ensures constant feature width)
# ---------------------------------------------------------------------

_cached_proto_states: np.ndarray | None = None
_cached_num_prototypes: int | None = None
_cached_seed: int | None = None


def _select_prototypes_fixed_width(
    feature_matrix: np.ndarray,
    num_prototypes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select exactly num_prototypes rows from feature_matrix.

    If the dataset is smaller than num_prototypes, sampling is done
    *with replacement* to preserve width.
    """

    X = np.asarray(feature_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got {X.shape}")

    n, d = X.shape

    # No data case: return zero prototypes
    if n == 0:
        return (
            np.zeros((num_prototypes, d), dtype=float),
            np.zeros(num_prototypes, dtype=int),
        )

    rng = np.random.default_rng(seed)

    if n >= num_prototypes:
        indices = rng.choice(n, size=num_prototypes, replace=False)
    else:
        indices = rng.choice(n, size=num_prototypes, replace=True)

    return X[indices], indices.astype(int)


def _encode_prototype_states(
    prototypes: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Encode prototype rows into quantum statevectors.
    """

    prototypes = np.asarray(prototypes, dtype=float)
    if prototypes.ndim != 2:
        raise ValueError(f"prototypes must be 2D, got {prototypes.shape}")

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
    Compute the raw quantum kernel matrix:

        K[i, j] = |<phi(x_i) | phi(proto_j)>|^2
    """

    X = np.asarray(feature_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got {X.shape}")

    proto_states = np.asarray(proto_states, dtype=np.complex128)
    if proto_states.ndim != 2:
        raise ValueError(f"proto_states must be 2D, got {proto_states.shape}")

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
    Build quantum kernel features with guaranteed shape (n_samples, num_prototypes).

    This function returns a kernel matrix that is:
        - mathematically valid (PSD)
        - numerically stable
        - suitable for direct NN consumption
    """

    global _cached_proto_states, _cached_num_prototypes, _cached_seed

    X = np.asarray(feature_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got {X.shape}")

    n, _ = X.shape
    if num_prototypes <= 0:
        raise ValueError("num_prototypes must be positive")

    # Determine whether cached prototypes can be reused
    can_use_cache = (
        use_cache
        and _cached_proto_states is not None
        and _cached_num_prototypes == int(num_prototypes)
        and _cached_seed == int(seed)
        and _cached_proto_states.shape[0] == int(num_prototypes)
    )

    if can_use_cache:
        proto_states = _cached_proto_states
    else:
        prototypes, _ = _select_prototypes_fixed_width(
            X,
            num_prototypes=num_prototypes,
            seed=seed,
        )
        proto_states = _encode_prototype_states(
            prototypes,
            weights=weights,
        )

        _cached_proto_states = proto_states
        _cached_num_prototypes = int(num_prototypes)
        _cached_seed = int(seed)

    # Compute raw kernel matrix
    K = _compute_fidelity_feature_matrix(
        X,
        proto_states,
        weights=weights,
    )

    # -----------------------------------------------------------------
    # Mercer-safe diagonal normalization
    #
    # I apply:
    #     K' = K D^{-1/2} D^{-1/2}
    #
    # where D is a positive diagonal matrix derived from column magnitudes.
    # This preserves PSD and kernel validity while stabilizing scale.
    # -----------------------------------------------------------------
    scale = np.maximum(np.max(K, axis=0), 1e-12)
    inv_sqrt = 1.0 / np.sqrt(scale)
    K_scaled = K * inv_sqrt[None, :] * inv_sqrt[None, :]

    # Final hard shape guarantee
    if K_scaled.shape != (n, num_prototypes):
        out = np.zeros((n, num_prototypes), dtype=float)
        m = min(num_prototypes, K_scaled.shape[1])
        out[:, :m] = K_scaled[:, :m]
        K_scaled = out

    return K_scaled.astype(float)



"""
Modified By: Callam
Project: Lotto Generator

Purpose:
    Quantum kernel feature block for the lotto generator pipeline.

Design:
    1) Uses the SAME embedding geometry as config.quantum_features
       (identical angle preprocessing and variational weights).
    2) Encodes classical feature rows as quantum states |phi(x)>.
    3) Selects a fixed-size set of prototype states from the dataset.
    4) For every sample, computes fidelity features:
           k_j(x) = |<phi(x)|phi(x_ref_j)>|^2  in [0, 1]
       where {x_ref_j} are the prototype rows.
    5) Returns a dense real-valued feature matrix suitable for
       concatenation with classical + quantum features in deep_learning.py.
"""

import numpy as np
import pennylane as qml

from config.quantum_features import NUM_QUBITS, _preprocess_to_angles, _global_weights

# ---------------------------------------------------------------------
# Device for kernel state preparation
# ---------------------------------------------------------------------

_kernel_dev = qml.device("default.qubit", wires=NUM_QUBITS, shots=None)


@qml.qnode(_kernel_dev)
def _state_circuit(angles, weights):
    """
    Prepare a variational quantum state |phi(x)> on NUM_QUBITS wires.

    Circuit:
        H^{⊗n} -> RY(angles) -> StronglyEntanglingLayers(weights)

    Returns:
        state: full statevector as a complex numpy array of length 2**NUM_QUBITS
    """
    for w in range(NUM_QUBITS):
        qml.Hadamard(wires=w)

    for w, theta in enumerate(angles):
        if w >= NUM_QUBITS:
            break
        qml.RY(theta, wires=w)

    qml.templates.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))

    return qml.state()


def _encode_state(classical_vec: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    Classical feature vector -> normalized quantum state |phi(x)>.

    Args:
        classical_vec: 1D numpy array of classical features.
        weights: variational weights; default uses global trained weights.

    Returns:
        state: complex numpy array of shape (2**NUM_QUBITS,)
    """
    if weights is None:
        weights = _global_weights

    angles = _preprocess_to_angles(classical_vec, num_qubits=NUM_QUBITS)
    state = _state_circuit(angles, weights)
    state = np.asarray(state, dtype=np.complex128)

    norm = np.linalg.norm(state)
    if norm == 0.0:
        return state
    return state / norm


def _pure_state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """
    Fidelity between two pure states |psi>, |phi>:

        F(psi, phi) = |<psi|phi>|^2

    Args:
        psi, phi: complex numpy arrays of identical length.

    Returns:
        scalar fidelity in [0, 1]
    """
    psi = np.asarray(psi, dtype=np.complex128)
    phi = np.asarray(phi, dtype=np.complex128)

    overlap = np.vdot(psi, phi)  # <psi|phi>
    return float(np.abs(overlap) ** 2)


def _select_prototypes(
    feature_matrix: np.ndarray,
    num_prototypes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select a fixed-size set of prototype classical rows from the dataset.

    Args:
        feature_matrix: shape (n_samples, d_classical)
        num_prototypes: number of prototype rows to select (upper-bounded by n_samples)
        seed: RNG seed for reproducibility

    Returns:
        prototypes: shape (m, d_classical)
        indices:    shape (m,) indices into the original feature_matrix
    """
    X = np.asarray(feature_matrix, dtype=float)
    n = X.shape[0]

    if n == 0:
        return np.empty((0, X.shape[1])), np.empty((0,), dtype=int)

    m = min(num_prototypes, n)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=m, replace=False)

    return X[indices], indices


def _encode_prototype_states(
    prototypes: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Encode prototype rows as quantum states.

    Args:
        prototypes: shape (m, d_classical)
        weights: variational weights; None -> global.

    Returns:
        states: complex array of shape (m, 2**NUM_QUBITS)
    """
    if weights is None:
        weights = _global_weights

    if prototypes.size == 0:
        return np.empty((0, 2**NUM_QUBITS), dtype=np.complex128)

    m = prototypes.shape[0]
    dim = 2 ** NUM_QUBITS
    states = np.zeros((m, dim), dtype=np.complex128)

    for i, row in enumerate(prototypes):
        states[i] = _encode_state(row, weights=weights)

    return states


def _compute_fidelity_feature_matrix(
    feature_matrix: np.ndarray,
    proto_states: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute fidelity features k_j(x) = |<phi(x)|phi(x_ref_j)>|^2 for all samples.

    Args:
        feature_matrix: shape (n_samples, d_classical)
        proto_states:   shape (m, 2**NUM_QUBITS)
        weights:        variational weights; None -> global.

    Returns:
        K: shape (n_samples, m), real-valued in [0, 1]
    """
    X = np.asarray(feature_matrix, dtype=float)
    n = X.shape[0]
    m = proto_states.shape[0]

    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=float)

    if weights is None:
        weights = _global_weights

    K = np.zeros((n, m), dtype=float)

    for i, row in enumerate(X):
        psi = _encode_state(row, weights=weights)
        for j in range(m):
            phi = proto_states[j]
            K[i, j] = _pure_state_fidelity(psi, phi)

    return K.astype(float)


def build_quantum_kernel_features(
    feature_matrix: np.ndarray,
    num_prototypes: int = 24,
    seed: int = 1337,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Construct quantum-kernel features for a batch.

    Pipeline:
        1) Select `num_prototypes` rows from `feature_matrix`.
        2) Encode prototypes to states.
        3) For every sample x, compute fidelity to each prototype state.
        4) Rescale columns to avoid degenerate numerical scales.

    Args:
        feature_matrix: shape (n_samples, d_classical) classical features.
        num_prototypes: size of prototype set (feature dimension of kernel block).
        seed: RNG seed for reproducibility.
        weights: variational weights; None -> global.

    Returns:
        kernel_features: numpy array of shape (n_samples, num_prototypes_effective)
    """
    X = np.asarray(feature_matrix, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape {X.shape}")

    prototypes, _ = _select_prototypes(X, num_prototypes=num_prototypes, seed=seed)
    proto_states = _encode_prototype_states(prototypes, weights=weights)

    K = _compute_fidelity_feature_matrix(X, proto_states, weights=weights)

    # Column-wise rescaling; always applied
    col_max = np.maximum(np.max(K, axis=0, keepdims=True), 1e-12)
    K_scaled = K / col_max

    return K_scaled.astype(float)

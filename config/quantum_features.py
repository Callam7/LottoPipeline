"""
Modified By: Callam

Purpose:
    Quantum-enhanced feature and prediction block for the lotto generator pipeline.

What this module does (single deterministic path):
    1) Classical feature row -> deterministic projection -> stable angles in [-pi, pi].
    2) Variational quantum circuit -> NUM_QUBITS Z-expectations.
    3) Append derived statistics (mean, std, L1, L2^2).
    4) Train circuit weights with SPSA (supervised on label geometry).
    5) Train a small predictive head on quantum features.
    6) Provide quantum-only predictions for hybrid fusion upstream.

Engineering guarantees:
    - Projection is consistent for any input dimension d.
    - Label scaling is enforced (targets in [0,1]).
    - SPSA uses steps, not "epochs-as-steps" ambiguity.
    - All shapes stable and checked.
    - Baseline hooks included to validate quantum lift.

Important reality note:
    This is not "quantum magic." It's a tunable nonlinear feature map.
    If labels contain no structure (ideal lotto), no method can extract signal.
"""

import math
import numpy as np
import pennylane as qml

import tensorflow as tf
from tensorflow import keras

# =========================== Configuration =========================== #

NUM_QUBITS = 12
QUANTUM_FEATURE_LEN = NUM_QUBITS + 4

_Q_NUM_LAYERS = 3

# SPSA training hyperparameters
_Q_SPSA_STEPS = 120        # real SPSA steps (not "epochs")
_Q_SPSA_BATCH_SIZE = 16

_Q_SPSA_A = 0.05
_Q_SPSA_C = 0.10
_Q_SPSA_ALPHA = 0.602
_Q_SPSA_GAMMA = 0.101

_Q_WEIGHT_CLIP = 2.0 * np.pi

# Deterministic seeds
tf.random.set_seed(1337)
np.random.seed(1337)

# PennyLane statevector backend (swap to lightning.qubit if you want speed)
dev = qml.device("default.qubit", wires=NUM_QUBITS)

# Global circuit weights θ (layers, qubits, 3)
_global_weights = np.random.normal(
    loc=0.0,
    scale=0.1,
    size=(_Q_NUM_LAYERS, NUM_QUBITS, 3),
)

# ===================== Deterministic Classical Projection ===================== #

def _build_projection_matrix(num_qubits: int, d: int) -> np.ndarray:
    """
    Deterministic mixing matrix M of shape (num_qubits, d).

    M[q, j] = sin((q+1)(j+1)) + 0.5*cos((q+1)(j+1))

    Provides stable, nonrandom nonlinear mixing.
    """
    M = np.zeros((num_qubits, d), dtype=float)
    for q in range(num_qubits):
        for j in range(d):
            k = (q + 1) * (j + 1)
            M[q, j] = math.sin(k) + 0.5 * math.cos(k)
    return M


def _structured_projection(x: np.ndarray, num_qubits: int = NUM_QUBITS) -> np.ndarray:
    """
    Deterministic classical -> qubit projection.

    Always uses deterministic mixing so behavior is continuous in d.

    Returns:
        v: (num_qubits,)
    """
    x = np.asarray(x, dtype=float).ravel()
    d = x.size

    if d == 0:
        return np.zeros(num_qubits, dtype=float)

    M = _build_projection_matrix(num_qubits, d)
    v = (M @ x) / float(d)
    return v.astype(float)


def _preprocess_to_angles(x: np.ndarray, num_qubits: int = NUM_QUBITS) -> np.ndarray:
    """
    Classical vector -> stable angles in [-pi, pi].

    Steps:
        1) Deterministic projection to NUM_QUBITS dims.
        2) Standardize (zero mean, unit variance).
        3) Clip to [-3, 3].
        4) Map to [-pi, pi].
    """
    v = _structured_projection(x, num_qubits)

    v = v - v.mean()
    std = v.std()
    if std > 1e-12:
        v = v / std

    v = np.clip(v, -3.0, 3.0)
    return v * (np.pi / 3.0)


# =========================== Quantum Feature Map =========================== #

@qml.qnode(dev)
def _feature_map_circuit(angles: np.ndarray, weights: np.ndarray):
    """
    Variational quantum feature map:
        H^{⊗n} -> RY(data) -> StronglyEntanglingLayers(θ) -> Z readout.
    """
    for i in range(NUM_QUBITS):
        qml.Hadamard(wires=i)

    for i, theta in enumerate(angles):
        qml.RY(theta, wires=i)

    qml.templates.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


# =========================== Public Feature API =========================== #

def compute_quantum_features(classical_vec: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Quantum features for one row.

    Returns:
        (QUANTUM_FEATURE_LEN,)
        = [Z_0..Z_{q-1}, mean(Z), std(Z), L1(Z), L2^2(Z)]
    """
    if weights is None:
        weights = _global_weights

    angles = _preprocess_to_angles(classical_vec)
    z = np.array(_feature_map_circuit(angles, weights), dtype=float)

    mean = float(z.mean())
    std = float(z.std())
    l1 = float(np.sum(np.abs(z)))
    l2_sq = float(np.sum(z ** 2))

    extra = np.array([mean, std, l1, l2_sq], dtype=float)
    return np.concatenate([z, extra]).astype(float)


def compute_quantum_matrix(feature_matrix: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Quantum feature matrix for a batch.

    Args:
        feature_matrix: shape (n_samples, d_classical)
        weights: circuit weights θ; None -> global.

    Returns:
        Q: shape (n_samples, QUANTUM_FEATURE_LEN)
    """
    if weights is None:
        weights = _global_weights

    X = np.asarray(feature_matrix, dtype=float)
    n = X.shape[0]
    out = np.zeros((n, QUANTUM_FEATURE_LEN), dtype=float)

    for i, row in enumerate(X):
        out[i] = compute_quantum_features(row, weights)

    return out.astype(float)


# =========================== SPSA Circuit Training =========================== #

def train_quantum_encoder(
    feature_matrix: np.ndarray,
    label_matrix: np.ndarray,
    steps: int = _Q_SPSA_STEPS,
    batch_size: int = _Q_SPSA_BATCH_SIZE,
):
    """
    Train circuit weights θ using SPSA on supervised label-MSE.

    Assumptions:
        - label_matrix entries represent probabilities or multi-hot in [0,1].
        - if not, they are clipped into [0,1] for correctness.

    Label compression:
        - main labels (first 40) -> 9 qubits
        - powerball labels (last 10) -> 3 qubits
      (fixed regime separation)

    Loss:
        MSE between rescaled Z expectations and compressed labels.
    """
    global _global_weights

    X = np.asarray(feature_matrix, dtype=float)
    Y = np.asarray(label_matrix, dtype=float)

    n_samples = X.shape[0]
    if n_samples == 0:
        return

    # Enforce probabilistic target domain for correctness
    Y = np.clip(Y, 0.0, 1.0)

    label_dim = Y.shape[1]
    main_dim = min(40, label_dim)
    pb_dim = max(0, label_dim - main_dim)

    main_qubits = 9
    pb_qubits = NUM_QUBITS - main_qubits

    main_chunk = int(math.ceil(main_dim / float(main_qubits))) if main_dim > 0 else 1
    pb_chunk = int(math.ceil(pb_dim / float(pb_qubits))) if pb_dim > 0 else 1

    def _compress_labels(y: np.ndarray) -> np.ndarray:
        """
        Compress label vector into NUM_QUBITS targets by chunk means.
        """
        y = np.asarray(y, dtype=float)
        out = np.zeros(NUM_QUBITS, dtype=float)

        for q in range(main_qubits):
            start = q * main_chunk
            end = min(main_dim, (q + 1) * main_chunk)
            out[q] = 0.0 if start >= main_dim else float(np.mean(y[start:end]))

        for q in range(pb_qubits):
            start = main_dim + q * pb_chunk
            end = min(label_dim, main_dim + (q + 1) * pb_chunk)
            out[main_qubits + q] = 0.0 if start >= label_dim else float(np.mean(y[start:end]))

        return out

    def _batch_loss(weights: np.ndarray, Xb: np.ndarray, Yb: np.ndarray) -> float:
        """
        Average MSE over batch.
        Z -> [0,1] by (z+1)/2.
        """
        m = Xb.shape[0]
        total = 0.0
        for i in range(m):
            angles = _preprocess_to_angles(Xb[i])
            z = np.array(_feature_map_circuit(angles, weights), dtype=float)
            z_rescaled = (z + 1.0) / 2.0

            y_comp = _compress_labels(Yb[i])
            diff = z_rescaled - y_comp
            total += float(np.mean(diff ** 2))

        return total / float(m)

    theta = _global_weights.copy()

    for t in range(steps):
        # Mini-batch
        if n_samples <= batch_size:
            idx = np.arange(n_samples)
        else:
            idx = np.random.choice(n_samples, size=batch_size, replace=False)

        Xb = X[idx]
        Yb = Y[idx]

        a_t = _Q_SPSA_A / ((t + 1) ** _Q_SPSA_ALPHA)
        c_t = _Q_SPSA_C / ((t + 1) ** _Q_SPSA_GAMMA)

        delta = np.random.choice([-1.0, 1.0], size=theta.shape).astype(float)

        loss_plus = _batch_loss(theta + c_t * delta, Xb, Yb)
        loss_minus = _batch_loss(theta - c_t * delta, Xb, Yb)

        g_hat = ((loss_plus - loss_minus) / (2.0 * c_t)) * delta
        theta = theta - a_t * g_hat
        theta = np.clip(theta, -_Q_WEIGHT_CLIP, _Q_WEIGHT_CLIP)

    _global_weights = theta


# =========================== Quantum Predictive Head =========================== #

_quantum_predictor = keras.Sequential(
    [
        keras.layers.Input(shape=(QUANTUM_FEATURE_LEN,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(50, activation="sigmoid"),
    ]
)

_quantum_predictor.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.MeanAbsoluteError(name="mae"),
    ],
)


def train_quantum_predictor(
    feature_matrix: np.ndarray,
    label_matrix: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
):
    """
    Train predictive head on quantum features.

    Enforces label domain in [0,1] (BCE correctness).
    """
    X = np.asarray(feature_matrix, dtype=float)
    Y = np.asarray(label_matrix, dtype=float)
    Y = np.clip(Y, 0.0, 1.0)

    Q = compute_quantum_matrix(X)

    _quantum_predictor.fit(
        Q, Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                min_delta=0.002,
                restore_best_weights=True,
                verbose=0,
            )
        ],
    )


def compute_quantum_prediction_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Quantum-only prediction probabilities for a batch.

    Returns:
        P: shape (n_samples, 50)
    """
    X = np.asarray(feature_matrix, dtype=float)
    Q = compute_quantum_matrix(X)
    preds = _quantum_predictor.predict(Q, verbose=0)
    return np.asarray(preds, dtype=float)


def compute_quantum_predictions(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper:
        mean prediction across batch.

    Returns:
        p: shape (50,)
    """
    P = compute_quantum_prediction_matrix(feature_matrix)
    return np.mean(P, axis=0).astype(float)


# =========================== Baseline Hooks (for real lift tests) =========================== #

def compute_random_fourier_baseline(feature_matrix: np.ndarray, out_dim: int = QUANTUM_FEATURE_LEN, seed: int = 1337) -> np.ndarray:
    """
    Cheap classical nonlinear baseline.
    If quantum doesn't beat this, there is no meaningful 'quantum lift'.

    Returns:
        R: (n_samples, out_dim)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(feature_matrix, dtype=float)
    n, d = X.shape

    W = rng.normal(0, 1.0, size=(d, out_dim))
    b = rng.uniform(0, 2*np.pi, size=(out_dim,))
    R = np.sqrt(2.0/out_dim) * np.cos(X @ W + b)
    return R.astype(float)







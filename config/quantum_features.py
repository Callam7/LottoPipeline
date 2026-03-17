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

Important correctness notes:
    - Z expectations are in [-1, 1]. We rescale to [0, 1] for MSE vs label targets.
    - Predictor head is trained with multi-label BCE (50 independent logits via sigmoid).
    - AUC metric MUST be multi-label for correctness in 50-label space.
"""

import math  # Math utilities (sin/cos/ceil/pi)
import numpy as np  # Numerical arrays and operations
import pennylane as qml  # Quantum circuit framework (PennyLane)

import tensorflow as tf  # TensorFlow runtime + seeding
from tensorflow import keras  # Keras API for neural head

# =========================== Configuration =========================== #

NUM_QUBITS = 12  # Number of qubits (and Z expectations)
QUANTUM_FEATURE_LEN = NUM_QUBITS + 4  # Total feature length (Zs + 4 stats)

_Q_NUM_LAYERS = 3  # Number of entangling layers in StronglyEntanglingLayers

# SPSA training hyperparameters (steps are true SPSA iterations)
_Q_SPSA_STEPS = 120  # Default SPSA iterations
_Q_SPSA_BATCH_SIZE = 16  # Default SPSA mini-batch size

# SPSA schedules: a_t = A/(t+1)^alpha, c_t = C/(t+1)^gamma
_Q_SPSA_A = 0.05  # SPSA step-size coefficient A
_Q_SPSA_C = 0.10  # SPSA perturbation coefficient C
_Q_SPSA_ALPHA = 0.602  # SPSA decay exponent alpha
_Q_SPSA_GAMMA = 0.101  # SPSA decay exponent gamma

_Q_WEIGHT_CLIP = 2.0 * np.pi  # Max absolute weight value for safety

# Deterministic seeds (NOTE: full determinism still depends on backend/threading)
tf.random.set_seed(1337)  # Seed TensorFlow RNG
np.random.seed(1337)  # Seed NumPy RNG

# PennyLane device
dev = qml.device("default.qubit", wires=NUM_QUBITS)  # Statevector simulator backend

# Global circuit weights θ with correct StronglyEntanglingLayers shape: (layers, wires, 3)
_global_weights = np.random.normal(loc=0.0, scale=0.1, size=(_Q_NUM_LAYERS, NUM_QUBITS, 3))  # Init weights tensor

# ===================== Deterministic Classical Projection ===================== #
# We cache projection matrices by (num_qubits, d) to avoid rebuilding them in loops.
_PROJ_CACHE = {}  # Cache dict for projection matrices


def _build_projection_matrix(num_qubits: int, d: int) -> np.ndarray:  # Build deterministic mixing matrix
    """
    Deterministic mixing matrix M of shape (num_qubits, d).

    M[q, j] = sin((q+1)(j+1)) + 0.5*cos((q+1)(j+1))

    This gives a stable, non-random feature mixing that works for any input dim d.
    """
    key = (num_qubits, d)  # Cache key for this (qubits, dim) pair
    if key in _PROJ_CACHE:  # If matrix already computed
        return _PROJ_CACHE[key]  # Return cached matrix

    M = np.zeros((num_qubits, d), dtype=float)  # Allocate projection matrix
    for q in range(num_qubits):  # Iterate rows (qubits)
        for j in range(d):  # Iterate columns (input dims)
            k = (q + 1) * (j + 1)  # Deterministic index product (1-based)
            M[q, j] = math.sin(k) + 0.5 * math.cos(k)  # Fill entry by deterministic formula

    _PROJ_CACHE[key] = M  # Store matrix in cache
    return M  # Return built matrix


def _structured_projection(x: np.ndarray, num_qubits: int = NUM_QUBITS) -> np.ndarray:  # Project R^d -> R^q deterministically
    """
    Deterministic projection from R^d -> R^{num_qubits}.

    Returns:
        v: shape (num_qubits,)
    """
    x = np.asarray(x, dtype=float).ravel()  # Convert to float array and flatten
    d = x.size  # Determine input dimensionality
    if d == 0:  # Guard: empty input
        return np.zeros(num_qubits, dtype=float)  # Return zeros vector if no features

    M = _build_projection_matrix(num_qubits, d)  # Get projection matrix for this dimension
    v = (M @ x) / float(d)  # Apply linear mix and normalize by d
    return v.astype(float)  # Return float vector


def _preprocess_to_angles(x: np.ndarray, num_qubits: int = NUM_QUBITS) -> np.ndarray:  # Convert features -> stable angles
    """
    Classical vector -> stable angles in [-pi, pi].

    Steps:
        1) deterministic projection to num_qubits dims
        2) standardize (mean 0, variance 1) to stabilize scaling
        3) clip to [-3,3] to avoid extreme rotations
        4) map linearly to [-pi, pi]
    """
    v = _structured_projection(x, num_qubits)  # Deterministically project to num_qubits dims

    v = v - float(v.mean())  # Center vector to mean 0
    std = float(v.std())  # Compute standard deviation
    if std > 1e-12:  # Avoid divide-by-near-zero
        v = v / std  # Standardize to unit variance

    v = np.clip(v, -3.0, 3.0)  # Clip to limit rotation magnitude
    return v * (np.pi / 3.0)  # Map [-3,3] -> [-pi,pi]


# =========================== Quantum Feature Map =========================== #

@qml.qnode(dev)  # Bind the circuit function to the device
def _feature_map_circuit(angles: np.ndarray, weights: np.ndarray):  # Quantum circuit: angles + weights -> Z expectations
    """
    Variational quantum feature map producing Z expectations per qubit.

    Circuit:
        - Hadamard on each qubit (create superposition)
        - RY(angle_i) on each qubit (data encoding)
        - StronglyEntanglingLayers(weights) (trainable nonlinear map)
        - Measure <Z_i> for each qubit
    """
    for i in range(NUM_QUBITS):  # Loop across qubits
        qml.Hadamard(wires=i)  # Put each qubit into superposition

    for i, theta in enumerate(angles):  # Iterate angle per qubit
        qml.RY(theta, wires=i)  # Encode data into Y-rotations

    qml.templates.StronglyEntanglingLayers(weights, wires=range(NUM_QUBITS))  # Apply parameterized entangling layers
    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]  # Measure Z expectation for each qubit


# =========================== Public Feature API =========================== #

def compute_quantum_features(classical_vec: np.ndarray, weights: np.ndarray = None) -> np.ndarray:  # Compute one feature vector
    """
    Quantum features for one row.

    Returns:
        shape (QUANTUM_FEATURE_LEN,)
        = [Z_0..Z_{q-1}, mean(Z), std(Z), L1(Z), L2^2(Z)]
    """
    if weights is None:  # If no weights passed in
        weights = _global_weights  # Use global circuit weights

    angles = _preprocess_to_angles(classical_vec)  # Convert classical vector to circuit angles
    z = np.array(_feature_map_circuit(angles, weights), dtype=float)  # Run circuit and collect Z expectations

    # Summary statistics (also deterministic functions of z)
    mean = float(z.mean())  # Mean of Z expectations
    std = float(z.std())  # Std dev of Z expectations
    l1 = float(np.sum(np.abs(z)))  # L1 norm of Z vector
    l2_sq = float(np.sum(z ** 2))  # Squared L2 norm of Z vector

    extra = np.array([mean, std, l1, l2_sq], dtype=float)  # Pack summary stats into array
    return np.concatenate([z, extra]).astype(float)  # Concatenate Z vector + stats and return


def compute_quantum_matrix(feature_matrix: np.ndarray, weights: np.ndarray = None) -> np.ndarray:  # Compute batch quantum features
    """
    Quantum feature matrix for a batch.

    Args:
        feature_matrix: shape (n_samples, d_classical)
        weights: optional circuit weights; None => global weights

    Returns:
        Q: shape (n_samples, QUANTUM_FEATURE_LEN)
    """
    if weights is None:  # If no weights passed in
        weights = _global_weights  # Use global weights

    X = np.asarray(feature_matrix, dtype=float)  # Convert input to float array
    if X.ndim != 2:  # Validate rank
        raise ValueError(f"feature_matrix must be 2D, got shape {X.shape}")  # Enforce 2D

    n = X.shape[0]  # Number of samples
    out = np.zeros((n, QUANTUM_FEATURE_LEN), dtype=float)  # Allocate output matrix

    for i in range(n):  # Iterate samples
        out[i] = compute_quantum_features(X[i], weights)  # Compute quantum features per row

    return out.astype(float)  # Return float output matrix


# =========================== SPSA Circuit Training =========================== #

def train_quantum_encoder(  # Train the quantum encoder (circuit weights)
    feature_matrix: np.ndarray,  # Feature matrix X
    label_matrix: np.ndarray,  # Label matrix Y
    steps: int = _Q_SPSA_STEPS,  # Number of SPSA steps
    batch_size: int = _Q_SPSA_BATCH_SIZE,  # Mini-batch size
):
    """
    Train circuit weights θ using SPSA to reduce supervised label-MSE.

    Target construction:
        - labels are in [0,1] (multi-hot / probabilities)
        - compress labels into NUM_QUBITS targets via chunk means
        - compare against rescaled Z: z_rescaled = (z+1)/2 in [0,1]

    Loss:
        MSE between z_rescaled and compressed-label targets.
    """
    global _global_weights  # Declare we will update global weights

    X = np.asarray(feature_matrix, dtype=float)  # Convert features to float array
    Y = np.asarray(label_matrix, dtype=float)  # Convert labels to float array

    if X.ndim != 2:  # Validate X rank
        raise ValueError(f"feature_matrix must be 2D, got {X.shape}")  # Enforce 2D
    if Y.ndim != 2:  # Validate Y rank
        raise ValueError(f"label_matrix must be 2D, got {Y.shape}")  # Enforce 2D
    if X.shape[0] != Y.shape[0]:  # Validate matching rows
        raise ValueError(f"X rows {X.shape[0]} != Y rows {Y.shape[0]}")  # Enforce same sample count

    n_samples = X.shape[0]  # Store number of samples
    if n_samples == 0:  # Guard empty input
        return  # Nothing to train

    # Enforce correct label domain for MSE target
    Y = np.clip(Y, 0.0, 1.0)  # Clip labels to [0,1]

    label_dim = Y.shape[1]  # Total label width
    main_dim = min(40, label_dim)  # Main label width capped at 40
    pb_dim = max(0, label_dim - main_dim)  # Remaining labels are powerball section

    # Fixed regime split: 9 qubits for main, remainder for powerball.
    main_qubits = 9  # Qubits allocated to main numbers
    pb_qubits = NUM_QUBITS - main_qubits  # Qubits allocated to powerball

    # Chunk sizes map label indices into qubit targets by averaging.
    main_chunk = int(math.ceil(main_dim / float(main_qubits))) if main_dim > 0 else 1  # Chunk size for main compression
    pb_chunk = int(math.ceil(pb_dim / float(pb_qubits))) if pb_dim > 0 else 1  # Chunk size for PB compression

    def _compress_labels(y: np.ndarray) -> np.ndarray:  # Compress label vector into NUM_QUBITS targets
        """
        Compress a (50,) label vector into (NUM_QUBITS,) by chunk means.
        """
        y = np.asarray(y, dtype=float)  # Ensure float vector
        out = np.zeros(NUM_QUBITS, dtype=float)  # Allocate compressed vector

        # Main block
        for q in range(main_qubits):  # Iterate main qubits
            start = q * main_chunk  # Start index of chunk
            end = min(main_dim, (q + 1) * main_chunk)  # End index of chunk
            out[q] = 0.0 if start >= main_dim else float(np.mean(y[start:end]))  # Chunk mean or 0

        # Powerball block
        for q in range(pb_qubits):  # Iterate PB qubits
            start = main_dim + q * pb_chunk  # Start index in PB region
            end = min(label_dim, main_dim + (q + 1) * pb_chunk)  # End index in PB region
            out[main_qubits + q] = 0.0 if start >= label_dim else float(np.mean(y[start:end]))  # Mean or 0

        return out  # Return compressed target

    def _batch_loss(weights: np.ndarray, Xb: np.ndarray, Yb: np.ndarray) -> float:  # Compute MSE loss over batch
        """
        Average MSE over batch:
            z_rescaled = (z+1)/2 in [0,1]
            MSE(z_rescaled, compressed_labels)
        """
        m = Xb.shape[0]  # Batch size
        total = 0.0  # Accumulate total loss

        for i in range(m):  # Loop batch samples
            angles = _preprocess_to_angles(Xb[i])  # Convert features -> angles
            z = np.array(_feature_map_circuit(angles, weights), dtype=float)  # Circuit Z expectations
            z_rescaled = (z + 1.0) / 2.0  # Rescale [-1,1] -> [0,1]

            y_comp = _compress_labels(Yb[i])  # Compress labels for sample
            diff = z_rescaled - y_comp  # Compute error vector
            total += float(np.mean(diff ** 2))  # Add sample MSE to total

        return total / float(m)  # Return mean loss per sample

    theta = _global_weights.copy()  # Local copy of weights to update

    for t in range(int(steps)):  # SPSA iteration loop
        # Choose batch indices
        if n_samples <= batch_size:  # If dataset smaller than batch size
            idx = np.arange(n_samples)  # Use all samples
        else:  # Otherwise
            idx = np.random.choice(n_samples, size=batch_size, replace=False)  # Sample without replacement

        Xb = X[idx]  # Select batch features
        Yb = Y[idx]  # Select batch labels

        # SPSA schedules
        a_t = _Q_SPSA_A / ((t + 1) ** _Q_SPSA_ALPHA)  # Step size schedule
        c_t = _Q_SPSA_C / ((t + 1) ** _Q_SPSA_GAMMA)  # Perturbation schedule

        # Rademacher perturbation (+1/-1)
        delta = np.random.choice([-1.0, 1.0], size=theta.shape).astype(float)  # Random +/-1 tensor

        loss_plus = _batch_loss(theta + c_t * delta, Xb, Yb)  # Loss at theta + c*delta
        loss_minus = _batch_loss(theta - c_t * delta, Xb, Yb)  # Loss at theta - c*delta

        # Gradient estimate
        g_hat = ((loss_plus - loss_minus) / (2.0 * c_t)) * delta  # SPSA gradient estimate

        # Update and clip
        theta = theta - a_t * g_hat  # Apply update
        theta = np.clip(theta, -_Q_WEIGHT_CLIP, _Q_WEIGHT_CLIP)  # Clip weights for stability

    _global_weights = theta  # Commit updated weights to global state


# =========================== Quantum Predictive Head =========================== #
# This is a standard multi-label classifier mapping quantum features -> 50 probabilities.

_quantum_predictor = keras.Sequential(  # Define predictor network
    [
        keras.layers.Input(shape=(QUANTUM_FEATURE_LEN,)),  # Input layer for quantum feature vector
        keras.layers.Dense(128, activation="relu"),  # Hidden layer 1
        keras.layers.Dense(64, activation="relu"),  # Hidden layer 2
        keras.layers.Dense(50, activation="sigmoid"),  # Output layer (50 independent probabilities)
    ]
)

_quantum_predictor.compile(  # Compile model with optimizer/loss/metrics
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # Adam optimizer
    loss=keras.losses.BinaryCrossentropy(),  # Multi-label BCE loss
    metrics=[
        keras.metrics.BinaryAccuracy(name="binary_accuracy"),  # Binary accuracy metric
        # CRITICAL FIX: multi-label AUC for 50-label output
        keras.metrics.AUC(name="auc", multi_label=True, num_labels=50),  # Multi-label AUC metric
        keras.metrics.MeanAbsoluteError(name="mae"),  # Mean absolute error metric
    ],
)


def train_quantum_predictor(  # Train Keras head on quantum features
    feature_matrix: np.ndarray,  # Classical feature matrix (n,d)
    label_matrix: np.ndarray,  # Label matrix (n,50)
    epochs: int = 20,  # Training epochs
    batch_size: int = 32,  # Training batch size
):
    """
    Train predictive head on quantum features.

    Enforces label domain in [0,1] for BCE correctness.
    """
    X = np.asarray(feature_matrix, dtype=float)  # Convert features to float array
    Y = np.asarray(label_matrix, dtype=float)  # Convert labels to float array

    if X.ndim != 2 or Y.ndim != 2:  # Validate both inputs are 2D
        raise ValueError(f"X and Y must be 2D, got X={X.shape}, Y={Y.shape}")  # Raise on wrong shape
    if X.shape[0] != Y.shape[0]:  # Validate row counts match
        raise ValueError(f"X rows {X.shape[0]} != Y rows {Y.shape[0]}")  # Raise on mismatch
    if Y.shape[1] != 50:  # Validate label width
        raise ValueError(f"Expected Y width 50, got {Y.shape[1]}")  # Enforce 50 outputs

    Y = np.clip(Y, 0.0, 1.0)  # Clip labels into BCE-valid range

    Q = compute_quantum_matrix(X)  # Compute quantum features for all samples

    _quantum_predictor.fit(  # Fit Keras head
        Q,  # Inputs: quantum features
        Y,  # Targets: labels
        epochs=int(epochs),  # Epoch count
        batch_size=int(batch_size),  # Batch size
        validation_split=0.15,  # Validation split fraction
        shuffle=True,  # Shuffle training data
        verbose=0,  # Silent training
        callbacks=[
            keras.callbacks.EarlyStopping(  # Early stopping callback
                monitor="val_loss",  # Watch validation loss
                patience=4,  # Wait this many epochs without improvement
                min_delta=0.002,  # Minimum improvement threshold
                restore_best_weights=True,  # Revert to best weights
                verbose=0,  # Silent callback
            )
        ],
    )


def compute_quantum_prediction_matrix(feature_matrix: np.ndarray) -> np.ndarray:  # Predict probabilities for batch
    """
    Quantum-only prediction probabilities for a batch.

    Returns:
        P: shape (n_samples, 50)
    """
    X = np.asarray(feature_matrix, dtype=float)  # Convert features to float array
    if X.ndim != 2:  # Validate rank
        raise ValueError(f"feature_matrix must be 2D, got {X.shape}")  # Enforce 2D input

    Q = compute_quantum_matrix(X)  # Compute quantum features for batch
    preds = _quantum_predictor.predict(Q, verbose=0)  # Predict using Keras head
    preds = np.asarray(preds, dtype=float)  # Convert to float array

    # Enforce stable output shape
    if preds.ndim != 2 or preds.shape[1] != 50:  # Validate output shape
        raise ValueError(f"Quantum predictor returned bad shape {preds.shape}, expected (n,50)")  # Raise on mismatch

    return preds  # Return prediction matrix


def compute_quantum_predictions(feature_matrix: np.ndarray) -> np.ndarray:  # Return mean prediction across batch
    """
    Convenience wrapper: mean prediction across batch.

    Returns:
        p: shape (50,)
    """
    P = compute_quantum_prediction_matrix(feature_matrix)  # Compute batch predictions
    return np.mean(P, axis=0).astype(float)  # Average across samples and return vector


# =========================== Baseline Hooks =========================== #

def compute_random_fourier_baseline(feature_matrix: np.ndarray, out_dim: int = QUANTUM_FEATURE_LEN, seed: int = 1337) -> np.ndarray:  # Random Fourier baseline
    """
    Cheap classical nonlinear baseline.
    If quantum doesn't beat this, there is no meaningful 'quantum lift'.

    Returns:
        R: (n_samples, out_dim)
    """
    rng = np.random.default_rng(seed)  # Create RNG for deterministic baseline
    X = np.asarray(feature_matrix, dtype=float)  # Convert feature matrix to float
    if X.ndim != 2:  # Validate input rank
        raise ValueError(f"feature_matrix must be 2D, got {X.shape}")  # Enforce 2D

    n, d = X.shape  # Extract sample count and feature dimension
    W = rng.normal(0, 1.0, size=(d, out_dim))  # Sample random projection weights
    b = rng.uniform(0, 2 * np.pi, size=(out_dim,))  # Sample random phase offsets
    R = np.sqrt(2.0 / out_dim) * np.cos(X @ W + b)  # Compute random Fourier features
    return R.astype(float)  # Return baseline features as float


"""
Modified By: Callam
Project: Lotto Generator
Purpose:
    Deep learning prediction pipeline for lottery probabilities:
        - 40 main numbers
        - 10 Powerball numbers
    Output is always shape (50,), compatible with the ticket generator.
Design:
    This module does NOT assume determinism.
    It assumes that if weak signal exists, it should not be suppressed
    by over-regularisation, premature stopping, or metric noise.
Pipeline stages:
    1) Build classical feature matrix from pipeline signals.
    2) Build strict multi-hot labels from historical draws.
    3) Train quantum encoder (SPSA) to tune circuit weights.
    4) Compute quantum feature matrix from tuned circuit.
    5) Compute quantum kernel features (fidelity-based).
    6) Fuse classical + quantum + kernel features.
    7) Train deep learning model on fused features.
    8) Post-training pipe importance ablation (for Optuna bridge).
"""
import logging # Standard Python logging
from typing import Any, Dict, List, Tuple # Type hints for clarity and static checking

import numpy as np # Core numerical array library used throughout
from adaptor.ablation import compute_post_training_pipe_importance, get_most_impactful_pipe
import tensorflow as tf # TensorFlow backend used for training and tensor ops
from tensorflow import keras # Keras API for model definition/training
from config.logs import EpochLogger # Custom callback to log epoch progress cleanly
from pipeline import get_dynamic_params # Dynamic training params (supports Optuna overrides)
from config.quantum_features import ( # Imports quantum feature utilities/constants
    compute_quantum_matrix, # Builds quantum feature matrix from classical inputs
    train_quantum_encoder, # Trains/tunes the quantum encoder parameters
    QUANTUM_FEATURE_LEN, # Fixed width expected from compute_quantum_matrix output
)
from config import quantum_kernels as qk # Imports module itself (to access cache vars)
from config.quantum_kernels import build_quantum_kernel_features # Builds kernel features

# Optional sklearn import moved to top for cleanliness
try:
    from sklearn.metrics import roc_auc_score # Used only inside pipe importance function
except ImportError:
    roc_auc_score = None # Graceful fallback if sklearn missing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # Set log format/level

# ===================== Constants ===================== #
NUM_MAIN = 40 # Main number count (1..40)
NUM_POWERBALL = 10 # Powerball number count (1..10)
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL # Total output width (50)

EPOCH_SIZE = 60 # Default number of training epochs (upper bound, can be overridden by Optuna)
BATCH_SIZE = 32 # Default mini-batch size
DATA_AUGMENTATION_ROUNDS = 3 # How many noisy copies of train data to add
NOISE_STDDEV = 0.01 # Noise scale applied to features (not labels)

MIN_CLASS_WEIGHT = 1.0 # Minimum positive-class weight
MAX_CLASS_WEIGHT = 4.0 # Maximum positive-class weight
MIN_PROB = 1e-7 # Used for clipping probabilities and avoiding divide-by-zero / log(0)

KERNEL_PROTOTYPES = 24 # Number of kernel prototypes (feature width for K_*)

class_weights = None # Numpy weights computed from training label imbalance
class_weights_tf = None # TensorFlow constant version used inside loss

# ===================== CLASSICAL FEATURE BLOCK DEFINITION (Future-Proof) ===================== #
# Single source of truth for feature block order.
# When future pipes are added/removed/reordered, only update this list.
CLASSICAL_FEATURE_BLOCKS: List[str] = [
    "bayesian_fusion_norm", # Bayesian fused probabilities
    "monte_carlo", # Monte Carlo simulation probabilities
    "redundancy", # Redundancy / coverage features
    "markov_features", # Markov chain transition features
    "entropy_features", # Shannon entropy features
    "centroids", # Cluster centroid vectors
    "clusters", # Cluster assignment vectors
]

def _get_classical_feature_arrays(
    mc: np.ndarray, # Monte Carlo probability vector
    rd: np.ndarray, # Redundancy probability vector
    mk: np.ndarray, # Markov feature vector
    en: np.ndarray, # Entropy feature vector
    fn: np.ndarray, # Bayesian fusion norm vector
    centroids: np.ndarray, # Centroid vector
    clusters: np.ndarray, # Cluster vector
) -> List[Tuple[str, np.ndarray]]:
    """Returns the classical feature blocks in the exact order defined by CLASSICAL_FEATURE_BLOCKS."""
    return [
        ("bayesian_fusion_norm", fn), # Must match order in CLASSICAL_FEATURE_BLOCKS
        ("monte_carlo", mc),
        ("redundancy", rd),
        ("markov_features", mk),
        ("entropy_features", en),
        ("centroids", centroids),
        ("clusters", clusters),
    ]

# ===================== Quantum kernel cache reset ===================== #
def _reset_quantum_kernel_cache():
    """
    Hard reset of quantum kernel prototype cache.
    Prototype states are quantum statevectors that depend on
    variational circuit weights. After encoder training, those
    weights may change.
    Reusing cached prototype states after a weight update would
    silently corrupt kernel features.
    This reset guarantees semantic consistency.
    """
    qk._cached_proto_states = None # Drops cached prototype statevectors
    qk._cached_num_prototypes = None # Drops cached prototype count
    qk._cached_seed = None # Drops cached seed (prototypes are seed-dependent)

# ===================== Weighted BCE ===================== #
def weighted_bce(y_true, y_pred):
    """
    Stable weighted binary cross-entropy.
    Key properties:
        - NO label smoothing (preserves ranking signal)
        - Positive-class weighting only
        - y_pred clipped for numerical safety
    """
    y_true = tf.cast(y_true, tf.float32) # Loss math assumes float tensors
    y_pred = tf.cast(y_pred, tf.float32) # Ensure predictions match dtype too
    y_pred = tf.clip_by_value(y_pred, MIN_PROB, 1.0 - MIN_PROB) # Clamp to safe open interval (avoid log(0))
    bce = keras.backend.binary_crossentropy(y_true, y_pred) # Elementwise BCE per label -> (batch, 50)
    w = y_true * class_weights_tf + (1.0 - y_true) # Apply class weight only when y_true == 1, else weight = 1.0
    return tf.reduce_mean(bce * w, axis=-1) # Mean across the 50 labels, keep batch dimension

# ===================== Shape utilities ===================== #
def _ensure_2d(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=float) # Converts to float NumPy array
    if X.ndim == 1: # If vector, treat as single row
        X = X.reshape(1, -1) # Shape becomes (1, features)
    if X.ndim != 2: # Rejects higher-rank inputs early
        raise ValueError(f"{name} must be 2D, got shape {X.shape}") # Fail fast with clear message
    return X # Returns guaranteed 2D matrix

def _force_width(M: np.ndarray, width: int, name: str) -> np.ndarray:
    M = _ensure_2d(M, name) # Ensures we can safely read .shape
    n, d = M.shape # n = rows, d = current columns
    if d == width: # Already correct width → nothing to do
        return M.astype(float) # Ensure float dtype
    out = np.zeros((n, width), dtype=float) # Allocate output matrix of target width
    m = min(d, width) # Size of overlapping region
    out[:, :m] = M[:, :m] # Copy the overlapping columns
    logging.warning(f"{name} width {d} != {width}; padded/trimmed to {width}.") # Notify that shape correction happened
    return out # Return width-corrected matrix

def _prob_norm_vec(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel() # Flatten to 1D float array
    if x.size != NUM_TOTAL: # Enforce expected output width (50)
        logging.warning(f"{name} expected len {NUM_TOTAL}, got {x.size}. Padding/truncating.")
        if x.size < NUM_TOTAL:
            x = np.pad(x, (0, NUM_TOTAL - x.size), constant_values=0.0) # Pad with zeros if too short
        x = x[:NUM_TOTAL] # Truncate if too long
    x = np.clip(x, 0.0, None) # Probabilities must not be negative
    s = float(x.sum()) # Total mass
    if s <= 0.0: # If vector is all zeros (or invalid), fallback to uniform
        return np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL # Uniform distribution across 50 bins
    return x / s # Normalise so sum == 1.0

def _build_feature_matrix(
    F: np.ndarray, # Prefix frequency matrix (n_draws, 50)
    feature_blocks: List[Tuple[str, np.ndarray]], # List of (name, vector) pairs in correct order
) -> np.ndarray:
    """Centralized feature matrix builder to keep training and inference identical."""
    X_parts = [] # Will hold each feature block before horizontal concatenation
    for name, arr in feature_blocks: # Iterate in the order defined by CLASSICAL_FEATURE_BLOCKS
        arr = arr.reshape(1, -1) # Ensure row vector shape (1, 50)
        if name in ["centroids", "clusters"]: # These are static per-number vectors (not multiplied by F)
            X_parts.append(np.tile(arr, (F.shape[0], 1))) # Repeat the vector for every historical timestep
        else:
            X_parts.append(F * arr) # Multiply prefix frequencies by the probability vector
    return np.column_stack(X_parts).astype(float) # Concatenate all blocks horizontally → final classical feature matrix


# ===================== Main entry ===================== #
def deep_learning_prediction(pipeline: Any) -> None:
    global class_weights, class_weights_tf # Allows loss function to access run-specific weights

    # Resolve dynamic training parameters (supports Optuna overrides)
    _, dynamic_epochs = get_dynamic_params( # Get dynamic epoch count based on data size
        len(pipeline.get_data("historical_data") or [])
    )
    optuna_params = pipeline.get_data("optuna_best_params") or {} # Pull best params from Optuna if present
    epochs = optuna_params.get("epochs", dynamic_epochs) # Use Optuna value or fallback to dynamic
    batch_size = optuna_params.get("batch_size", BATCH_SIZE) # Use Optuna value or default
    learning_rate = optuna_params.get("learning_rate", 8e-4) # Use Optuna value or default
    dropout_rate = optuna_params.get("dropout_rate", 0.25) # Use Optuna value or default

    logging.info( # Log the final resolved training hyperparameters
        f"DL params -> epochs={epochs}, "
        f"batch_size={batch_size}, "
        f"lr={learning_rate}, "
        f"dropout={dropout_rate}"
    )

    # ---------- Step 1: Load pipeline inputs ---------- #
    historical_data = pipeline.get_data("historical_data") # Historical draw records used for labels and prefix stats
    if not historical_data: # If missing/empty history, cannot train anything meaningful
        pipeline.add_data(
            "deep_learning_predictions", # Store result into pipeline
            np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL # Uniform fallback (no information)
        )
        return # Exit early

    monte_carlo = pipeline.get_data("monte_carlo") # Probabilities from Monte Carlo stage
    redundancy = pipeline.get_data("redundancy") # Probabilities from redundancy/coverage stage
    markov = pipeline.get_data("markov_features") # Probabilities from Markov stage
    entropy = pipeline.get_data("entropy_features") # Probabilities from entropy stage
    fusion_norm = pipeline.get_data("bayesian_fusion_norm") # Bayesian fused and normalised probabilities
    clusters = pipeline.get_data("clusters") # Cluster assignment vector (per number)
    centroids = pipeline.get_data("centroids") # Centroid-related vector (per number)

    required = [monte_carlo, redundancy, markov, entropy, fusion_norm, clusters, centroids] # All required inputs
    if any(v is None for v in required): # Abort if any upstream feature missing
        logging.error("Deep learning aborted: missing required pipeline features.") # Emit diagnostics
        pipeline.add_data(
            "deep_learning_predictions", # Store fallback into pipeline
            np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL # Uniform fallback
        )
        return # Exit early

    # ---------- Step 2: Build strict multi-hot labels ---------- #
    labels = [] # Will hold one 50-dim multi-hot vector per historical draw
    for draw in historical_data: # Iterates over draw dicts
        y = np.zeros(NUM_TOTAL, dtype=float) # Allocates empty label vector
        for n in draw.get("numbers", []): # Pulls main number list; default empty
            if isinstance(n, int) and 1 <= n <= NUM_MAIN: # Validate as integer and in range
                y[n - 1] = 1.0 # Converts 1-based lotto number to 0-based index
        pb = draw.get("powerball") # Reads powerball field
        if isinstance(pb, int) and 1 <= pb <= NUM_POWERBALL: # Single powerball integer
            y[NUM_MAIN + pb - 1] = 1.0 # Map to indices 40..49 (0-based)
        elif isinstance(pb, (list, tuple)): # Some sources may store multiple PBs
            for p in pb: # Iterates PB list/tuple
                if isinstance(p, int) and 1 <= p <= NUM_POWERBALL: # Validate range
                    y[NUM_MAIN + p - 1] = 1.0 # Set PB index as active
        labels.append(y) # Stores label vector for this draw

    Y = np.asarray(labels, dtype=float) # Stack labels into shape (n_draws, 50)
    n_draws = Y.shape[0] # Number of historical examples available
    if n_draws < 10: # Too little data to reasonably train
        pipeline.add_data(
            "deep_learning_predictions", # Store fallback
            np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL # Uniform distribution
        )
        return # Exit early

    # ---------- Step 3: Normalise pipeline vectors ---------- #
    mc = _prob_norm_vec(monte_carlo, "monte_carlo") # Ensure valid probability vector
    rd = _prob_norm_vec(redundancy, "redundancy") # Ensure valid probability vector
    mk = _prob_norm_vec(markov, "markov_features") # Ensure valid probability vector
    en = _prob_norm_vec(entropy, "entropy_features") # Ensure valid probability vector
    fn = _prob_norm_vec(fusion_norm, "bayesian_fusion_norm") # Ensure valid probability vector
    clusters_arr = np.asarray(clusters, dtype=float).ravel() # Force to flat float vector
    centroids_arr = np.asarray(centroids, dtype=float).ravel() # Force to flat float vector

    if clusters_arr.size != NUM_TOTAL: # If wrong size, discard rather than misalign features
        clusters_arr = np.zeros(NUM_TOTAL, dtype=float) # Replace with zeros to preserve dimensions
    if centroids_arr.size != NUM_TOTAL: # If wrong size, discard rather than misalign features
        centroids_arr = np.zeros(NUM_TOTAL, dtype=float) # Replace with zeros to preserve dimensions

    # ---------- Step 4: Build causal prefix frequencies ---------- #
    F = np.zeros((n_draws, NUM_TOTAL), dtype=float) # One frequency vector per timestep (before seeing that draw)
    counts = np.zeros(NUM_TOTAL, dtype=float) # Running occurrence counts up to time t-1
    for t in range(n_draws): # Walk forward through history
        s = counts.sum() # Total counts so far
        F[t] = counts / s if s > 0 else np.ones(NUM_TOTAL) / NUM_TOTAL # Convert counts to frequencies or uniform
        for n in historical_data[t].get("numbers", []): # Adds numbers from the current draw into counts
            if isinstance(n, int) and 1 <= n <= NUM_MAIN: # Validate main number
                counts[n - 1] += 1.0 # Increments main number count
        pb = historical_data[t].get("powerball") # Reads powerball for this draw
        if isinstance(pb, int) and 1 <= pb <= NUM_POWERBALL: # Single PB integer
            counts[NUM_MAIN + pb - 1] += 1.0 # Increments PB count
        elif isinstance(pb, (list, tuple)): # Multiple PBs
            for p in pb: # Iterates PB list/tuple
                if isinstance(p, int) and 1 <= p <= NUM_POWERBALL: # Validate PB
                    counts[NUM_MAIN + p - 1] += 1.0 # Increments PB count

    # ---------- Step 5: Classical feature matrix (Future-Proof) ---------- #
    feature_blocks = _get_classical_feature_arrays( # Get blocks in canonical order
        mc, rd, mk, en, fn, centroids_arr, clusters_arr
    )
    X = _build_feature_matrix(F, feature_blocks) # Build matrix using central helper (keeps train/inference identical)

    # ---------- Step 6: Time-aware train/validation split ---------- #
    n_val = max(1, int(0.15 * n_draws)) # Validation is the last ~15% of history (at least 1 example)
    X_train, X_val = X[:-n_val], X[-n_val:] # Train on early history, validate on most recent history
    Y_train, Y_val = Y[:-n_val], Y[-n_val:] # Same split for labels

    # ---------- Step 7: Compute global class weights ---------- #
    pos = Y_train.sum(axis=0) # Positive counts per class across training set
    neg = Y_train.shape[0] - pos # Negative counts per class
    cw = neg / (pos + MIN_PROB) # Ratio-based positive class weight (avoid div-by-zero)
    cw = np.clip(cw, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT).astype(np.float32) # Clamp to keep gradients sane
    class_weights = cw # Store NumPy version globally for loss
    class_weights_tf = tf.constant(class_weights, dtype=tf.float32) # Store TF constant version for loss

    # ---------- Step 8: Train quantum encoder + reset kernel cache ---------- #
    try:
        train_quantum_encoder(X_train, Y_train) # Fit/tune the quantum encoder using training data only
        logging.info("Quantum encoder training complete.") # Confirm completion
    except Exception as e:
        logging.warning(f"Quantum encoder training failed: {e}") # Continue even if quantum training fails
    _reset_quantum_kernel_cache() # Ensure kernel prototypes are rebuilt under latest encoder weights

    # ---------- Step 9: Compute quantum and kernel features ---------- #
    try:
        Q_train = _force_width( # Ensure quantum feature matrix has fixed width
            compute_quantum_matrix(X_train), # Quantum feature extraction on training matrix
            QUANTUM_FEATURE_LEN, # Expected width from quantum feature extractor
            "Q_train" # Name used for error messages
        )
        Q_val = _force_width( # Ensure quantum feature matrix has fixed width
            compute_quantum_matrix(X_val), # Quantum feature extraction on validation matrix
            QUANTUM_FEATURE_LEN, # Expected width from quantum feature extractor
            "Q_val" # Name used for error messages
        )
    except Exception as e:
        logging.error(f"Quantum feature computation failed: {e}") # Report quantum feature failure
        Q_train = np.zeros((X_train.shape[0], QUANTUM_FEATURE_LEN)) # Fallback to zeros with correct shape
        Q_val = np.zeros((X_val.shape[0], QUANTUM_FEATURE_LEN)) # Fallback to zeros with correct shape

    try:
        K_train_raw = build_quantum_kernel_features( # Compute kernel features for training set
            X_train, # Use training examples (defines prototype cache)
            num_prototypes=KERNEL_PROTOTYPES, # Requested number of prototypes / output width
            seed=1337 # Deterministic prototype selection/reproducibility
        )
        K_val_raw = build_quantum_kernel_features( # Compute kernel features for validation set
            X_val, # Validation examples mapped against same cached prototypes
            num_prototypes=KERNEL_PROTOTYPES, # Output width
            seed=1337 # Seed matches to ensure cache compatibility
        )
        K_train = _force_width(K_train_raw, KERNEL_PROTOTYPES, "K_train") # Enforce fixed width on training kernel feats
        K_val = _force_width(K_val_raw, KERNEL_PROTOTYPES, "K_val") # Enforce fixed width on validation kernel feats
    except Exception as e:
        logging.error(f"Kernel feature computation failed: {e}") # Report kernel feature failure
        K_train = np.zeros((X_train.shape[0], KERNEL_PROTOTYPES)) # Fallback zero kernel features (train)
        K_val = np.zeros((X_val.shape[0], KERNEL_PROTOTYPES)) # Fallback zero kernel features (val)

    Xf_train = np.column_stack((X_train, Q_train, K_train)).astype(float) # Fuse classical + quantum + kernel (train)
    Xf_val = np.column_stack((X_val, Q_val, K_val)).astype(float) # Fuse classical + quantum + kernel (val)
    input_dim = Xf_train.shape[1] # Store final fused feature width for model input validation

    # ---------- Step 10: Train-only data augmentation ---------- #
    Xa = [Xf_train] # List of training matrices to stack (original + noisy versions)
    Ya = [Y_train] # Labels duplicated for each augmented copy
    for _ in range(DATA_AUGMENTATION_ROUNDS): # Generate multiple noisy copies of training data
        Xa.append(Xf_train + np.random.normal(0.0, NOISE_STDDEV, Xf_train.shape)) # Add Gaussian noise in feature space
        Ya.append(Y_train) # Keep labels unchanged (noise is only on features)
    Xa = np.vstack(Xa).astype(float) # Stack augmented training matrices vertically
    Ya = np.vstack(Ya).astype(float) # Stack labels to match augmented rows

    # ---------- Step 11: Model definition ---------- #
    model = keras.Sequential( # Simple feedforward network for tabular fused features
        [
            keras.layers.Input(shape=(input_dim,)), # Define input layer shape explicitly
            keras.layers.Dense(128, activation="relu"), # First dense layer (128 units)
            keras.layers.BatchNormalization(), # BatchNorm to stabilise hidden activations
            keras.layers.Dropout(0.25), # Prevent memorising historical draws
            keras.layers.Dense(64, activation="relu"), # Second dense layer (64 units)
            keras.layers.BatchNormalization(), # BatchNorm again
            keras.layers.Dropout(dropout_rate), # Regularisation (from Optuna or default)
            keras.layers.Dense(32, activation="relu"), # Third dense layer (32 units)
            keras.layers.Dense(NUM_TOTAL, activation="sigmoid"), # Output layer: independent probs per class (multi-label)

            ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0), # Adam with resolved LR + gradient clipping
        loss=weighted_bce, # Custom weighted BCE loss defined above
        metrics=[
            keras.metrics.AUC(multi_label=True, num_labels=NUM_TOTAL, name="auc"), # Multi-label AUC
            keras.metrics.BinaryAccuracy(name="bin_acc"), # Thresholded accuracy
            keras.metrics.MeanAbsoluteError(name="mae"), # MAE across probabilities
        ],
    )

    # ---------- Step 12: Training ---------- #
    model.fit(
        Xa, # Augmented training feature matrix (original + noisy copies)
        Ya, # Augmented training labels (duplicated to match Xa)
        epochs=epochs, # Resolved number of epochs (Optuna or dynamic)
        batch_size=batch_size, # Resolved batch size (Optuna or default)
        validation_data=(Xf_val, Y_val), # Validation uses clean (non-augmented) fused features
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", # Watch validation loss
                mode="min", # less is better
                factor=0.6, # Multiply LR by this factor when plateau detected
                patience=6, # Epochs to wait before reducing LR
                min_lr=5e-6, # Lower bound on learning rate
                verbose=1, # Prints when LR is reduced
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", # Uses val_auc instead of val_loss
                mode="min", # lower AUC is better
                patience=12, # Epochs to wait before stopping
                min_delta=0.0005, # Minimum improvement required to reset patience
                restore_best_weights=True, # Restores best weights by val_auc
                verbose=1, # Print stop reason
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=20,
                min_delta=0.001,
                restore_best_weights=False,
                verbose=0,
                ),

            EpochLogger(), # Custom callback for epoch logging
        ],
        verbose=1, # Prints training progress per epoch
    )
    
    # ===================== Post-training Pipe Importance (moved to ablation.py) ===================== #
    try:
        pipe_importance = compute_post_training_pipe_importance(model, Xf_val, Y_val)
        pipeline.add_data("pipe_importance", pipe_importance)

        weakest_pipe = get_most_impactful_pipe(pipe_importance)
        if weakest_pipe:
            pipeline.add_data("weakest_pipe", weakest_pipe)
            logging.info(f"Most impactful/weakest pipe identified: {weakest_pipe}")
    except Exception as e:
        logging.error(f"Pipe importance analysis failed: {e}")
        pipeline.add_data("pipe_importance", {})

    # ---------- Step 13: Inference ---------- #
    s = counts.sum() # Total counts after processing all historical draws
    f_now = ( # Construct current frequency vector (post-history)
        counts / s if s > 0 else np.ones(NUM_TOTAL) / NUM_TOTAL # Normalised counts or uniform fallback
    ).reshape(1, -1).astype(float) # Convert to row vector for feature construction

    feature_blocks_now = _get_classical_feature_arrays( # Get blocks in canonical order for inference
        mc, rd, mk, en, fn, centroids_arr, clusters_arr
    )
    x_now = _build_feature_matrix(f_now, feature_blocks_now) # Build current classical feature row using same helper

    try:
        q_now = _force_width( # Ensure quantum feature width matches training expectation
            compute_quantum_matrix(x_now), # Compute quantum features for current row
            QUANTUM_FEATURE_LEN, # Expected width
            "q_now" # Name for diagnostics
        )
    except Exception:
        q_now = np.zeros((1, QUANTUM_FEATURE_LEN)) # Fallback to zeros if quantum feature step fails

    try:
        k_now_raw = build_quantum_kernel_features( # Compute kernel features for current row
            x_now, # Current classical features
            num_prototypes=KERNEL_PROTOTYPES, # Output width
            seed=1337 # Seed must match cache semantics used earlier
        )
        k_now = _force_width(k_now_raw, KERNEL_PROTOTYPES, "k_now") # Enforce fixed kernel feature width
    except Exception:
        k_now = np.zeros((1, KERNEL_PROTOTYPES)) # Fallback to zeros if kernel step fails

    xf_now = np.column_stack((x_now, q_now, k_now)).astype(float) # Fuse classical + quantum + kernel for inference

    if xf_now.shape[1] != input_dim: # Hard check: inference feature width must match model input width
        logging.error(
            f"Inference width mismatch: got {xf_now.shape[1]}, expected {input_dim}" # Log exact mismatch
        )
        pipeline.add_data(
            "deep_learning_predictions", np.ones(NUM_TOTAL) / NUM_TOTAL # Uniform fallback
        )
        return # Exit early to avoid invalid model input

    try:
        dl_pred = model.predict(xf_now, verbose=0).reshape(-1).astype(float) # Run prediction and flatten to (50,)
    except Exception as e:
        logging.error(f"DL inference failed: {e}") # Report inference failure
        pipeline.add_data(
            "deep_learning_predictions", np.ones(NUM_TOTAL) / NUM_TOTAL # Uniform fallback
        )
        return # Exit early

    pipeline.add_data(
        "deep_learning_predictions", # Store output in pipeline
        _prob_norm_vec(np.clip(dl_pred, 0.0, 1.0), "deep_learning_predictions") # Ensure final probabilities are within [0, 1] and sum to 1
    )
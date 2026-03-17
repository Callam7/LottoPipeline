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
"""

import numpy as np  # Core numerical array library used throughout
import tensorflow as tf  # TensorFlow backend used for training and tensor ops
from tensorflow import keras  # Keras API for model definition/training
from config.logs import EpochLogger  # Custom callback to log epoch progress cleanly
import logging  # Standard Python logging

from config.quantum_features import (  # Imports quantum feature utilities/constants
    compute_quantum_matrix,  # Builds quantum feature matrix from classical inputs
    train_quantum_encoder,  # Trains/tunes the quantum encoder parameters
    QUANTUM_FEATURE_LEN,  # Fixed width expected from compute_quantum_matrix output
)

from config import quantum_kernels as qk  # Imports module itself (to access cache vars)
from config.quantum_kernels import build_quantum_kernel_features  # Builds kernel features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")  # Set log format/level

# ===================== Constants ===================== #

# Lottery structure
NUM_MAIN = 40  # Main number count (1..40)
NUM_POWERBALL = 10  # Powerball number count (1..10)
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL  # Total output width (50)

# Training configuration
EPOCH_SIZE = 60  # Number of training epochs for the NN
BATCH_SIZE = 32  # Mini-batch size

# Data augmentation (kept mild to preserve weak structure)
DATA_AUGMENTATION_ROUNDS = 3  # How many noisy copies of train data to add
NOISE_STDDEV = 0.01  # Noise scale applied to features (not labels)

# Class-weight bounds (prevents gradient saturation)
MIN_CLASS_WEIGHT = 1.0  # Minimum positive-class weight
MAX_CLASS_WEIGHT = 4.0  # Maximum positive-class weight

# Numeric safety
MIN_PROB = 1e-7  # Used for clipping probabilities and avoiding divide-by-zero

# Quantum kernel configuration
KERNEL_PROTOTYPES = 24  # Number of kernel prototypes (feature width for K_*)

# ===================== GLOBAL LOSS STATE ===================== #

# Computed once per training run
class_weights = None  # Numpy weights computed from training label imbalance
class_weights_tf = None  # TensorFlow constant version used inside loss

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
    qk._cached_proto_states = None  # Drops cached prototype statevectors
    qk._cached_num_prototypes = None  # Drops cached prototype count
    qk._cached_seed = None  # Drops cached seed (prototypes are seed-dependent)

# ===================== Weighted BCE (stable, no neutrality trap) ===================== #

def weighted_bce(y_true, y_pred):
    """
    Stable weighted binary cross-entropy.

    Key properties:
        - NO label smoothing (preserves ranking signal)
        - Positive-class weighting only
        - y_pred clipped for numerical safety

    Shapes:
        y_true: (batch, 50)
        y_pred: (batch, 50)
    """
    # Ensures float tensors
    y_true = tf.cast(y_true, tf.float32)  # Loss math assumes float tensors
    y_pred = tf.cast(y_pred, tf.float32)  # Ensure predictions match dtype too

    # Prevents log(0) and log(1)
    y_pred = tf.clip_by_value(y_pred, MIN_PROB, 1.0 - MIN_PROB)  # Clamp to safe open interval

    # Per-label BCE -> (batch, 50)
    bce = keras.backend.binary_crossentropy(y_true, y_pred)  # Elementwise BCE per label

    # Applys class weighting to positives only
    w = y_true * class_weights_tf + (1.0 - y_true)  # If y_true==1 apply weight else apply 1.0

    # Reduces to per-sample loss
    return tf.reduce_mean(bce * w, axis=-1)  # Mean across labels, keep batch dimension

# ===================== Shape utilities ===================== #

def _ensure_2d(X, name):
    """
    Ensure input is a 2D NumPy array.
    """
    X = np.asarray(X, dtype=float)  # Converts to float NumPy array
    if X.ndim == 1:  # If vector, treat as single row
        X = X.reshape(1, -1)  # Shape (1, features)
    if X.ndim != 2:  # Rejects higher-rank inputs early
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")  # Fail fast with clear message
    return X  # Returns 2D matrix

def _force_width(M, width, name):
    """
    Enforce fixed feature width via deterministic pad/trim.
    """
    M = _ensure_2d(M, name)  # Ensures I can safely index shape
    n, d = M.shape  # n rows, d columns

    if d == width:  # If already correct width, do nothing
        return M.astype(float)  # Ensures dtype is float

    out = np.zeros((n, width), dtype=float)  # Allocates padded/truncated output
    m = min(d, width)  # Overlaps region length
    out[:, :m] = M[:, :m]  # Copys overlap columns

    logging.warning(f"{name} width {d} != {width}; padded/trimmed to {width}.")  # Notifys shape correction
    return out  # Returns width-fixed matrix

def _prob_norm_vec(x, name):
    """
    Sum-normalise a probability-like vector.

    Guarantees:
        - length == 50
        - non-negative
        - sums to 1 (or uniform fallback)
    """
    x = np.asarray(x, dtype=float).ravel()  # Flattens to 1D float array

    if x.size != NUM_TOTAL:  # Enforces expected output width (50)
        raise ValueError(f"{name} expected len {NUM_TOTAL}, got {x.size}")  # Fail loud if wrong size

    x = np.clip(x, 0.0, None)  # Probabilities must not be negative
    s = float(x.sum())  # Total mass

    if s <= 0.0:  # If vector is all zeros (or invalid), fallback to uniform
        return np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL  # Uniform distribution across 50 bins

    return x / s  # Normalise to sum 1

# ===================== Main entry ===================== #

def deep_learning_prediction(pipeline):
    """
    End-to-end deep learning prediction stage.

    Produces a (50,) probability vector and stores it in the pipeline
    under key 'deep_learning_predictions'.
    """
    global class_weights, class_weights_tf  # Ensures loss can access run-specific weights

    # ---------- Step 1: Load pipeline inputs ---------- #

    historical_data = pipeline.get_data("historical_data")  # Historical draw records used for labels and prefix stats
    if not historical_data:  # If missing/empty history, cannot train anything meaningful
        pipeline.add_data(
            "deep_learning_predictions",  # Stores result into pipeline
            np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL  # Uniform fallback (no information)
        )
        return  # Exit early

    monte_carlo = pipeline.get_data("monte_carlo")  # Probabilities from Monte Carlo stage
    redundancy  = pipeline.get_data("redundancy")  # Probabilities from redundancy/coverage stage
    markov      = pipeline.get_data("markov_features")  # Probabilities from Markov stage
    entropy     = pipeline.get_data("entropy_features")  # Probabilities from entropy stage
    fusion_norm = pipeline.get_data("bayesian_fusion_norm")  # Bayesian fused and normalised probabilities
    clusters    = pipeline.get_data("clusters")  # Cluster assignment vector (per number)
    centroids   = pipeline.get_data("centroids")  # Centroid-related vector (per number)

    required = [monte_carlo, redundancy, markov, entropy, fusion_norm, clusters, centroids]  # All required inputs
    if any(v is None for v in required):  # Abort if any upstream feature missing
        logging.error("Deep learning aborted: missing required pipeline features.")  # Emit diagnostics
        pipeline.add_data(
            "deep_learning_predictions",  # Store fallback into pipeline
            np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL  # Uniform fallback
        )
        return  # Exit early

    # ---------- Step 2: Build strict multi-hot labels ---------- #

    labels = []  # Will hold one 50-dim multi-hot vector per historical draw

    for draw in historical_data:  # Iterates over draw dicts
        y = np.zeros(NUM_TOTAL, dtype=float)  # Allocates empty label vector

        # Main numbers (1–40)
        for n in draw.get("numbers", []):  # Pulls main number list; default empty
            if isinstance(n, int) and 1 <= n <= NUM_MAIN:  # Validate as integer and in range
                y[n - 1] = 1.0  # Converts 1-based lotto number to 0-based index

        # Powerball (1–10)
        pb = draw.get("powerball")  # Reads powerball field
        if isinstance(pb, int) and 1 <= pb <= NUM_POWERBALL:  # Single powerball integer
            y[NUM_MAIN + pb - 1] = 1.0  # Map to indices 40..49 (0-based)
        elif isinstance(pb, (list, tuple)):  # Some sources may store multiple PBs
            for p in pb:  # Iterates PB list/tuple
                if isinstance(p, int) and 1 <= p <= NUM_POWERBALL:  # Validate range
                    y[NUM_MAIN + p - 1] = 1.0  # Set PB index as active

        labels.append(y)  # Stores label vector for this draw

    Y = np.asarray(labels, dtype=float)  # Stack labels into shape (n_draws, 50)
    n_draws = Y.shape[0]  # Number of historical examples available

    if n_draws < 10:  # Too little data to reasonably train
        pipeline.add_data(
            "deep_learning_predictions",  # Store fallback
            np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL  # Uniform distribution
        )
        return  # Exit early

    # ---------- Step 3: Normalise pipeline vectors ---------- #
    #Each one snsures valid probability vectors
    mc = _prob_norm_vec(monte_carlo, "monte_carlo")  
    rd = _prob_norm_vec(redundancy,  "redundancy")  
    mk = _prob_norm_vec(markov,      "markov_features")  
    en = _prob_norm_vec(entropy,     "entropy_features")  
    fn = _prob_norm_vec(fusion_norm, "bayesian_fusion_norm")  

    clusters  = np.asarray(clusters, dtype=float).ravel()  # Force to flat float vector
    centroids = np.asarray(centroids, dtype=float).ravel()  # Force to flat float vector

    if clusters.size != NUM_TOTAL:  # If wrong size, discard rather than misalign features
        clusters = np.zeros(NUM_TOTAL, dtype=float)  # Replace with zeros to preserve dimensions
    if centroids.size != NUM_TOTAL:  # If wrong size, discard rather than misalign features
        centroids = np.zeros(NUM_TOTAL, dtype=float)  # Replace with zeros to preserve dimensions

    # ---------- Step 4: Build causal prefix frequencies ---------- #

    F = np.zeros((n_draws, NUM_TOTAL), dtype=float)  # One frequency vector per timestep (before seeing that draw)
    counts = np.zeros(NUM_TOTAL, dtype=float)  # Running occurrence counts up to time t-1

    for t in range(n_draws):  # Walk forward through history
        s = counts.sum()  # Total counts so far
        F[t] = counts / s if s > 0 else np.ones(NUM_TOTAL) / NUM_TOTAL  # Convert counts to frequencies or uniform

        for n in historical_data[t].get("numbers", []):  # Adds numbers from the current draw into counts
            if isinstance(n, int) and 1 <= n <= NUM_MAIN:  # Validate main number
                counts[n - 1] += 1.0  # Increments main number count

        pb = historical_data[t].get("powerball")  # Reads powerball for this draw
        if isinstance(pb, int) and 1 <= pb <= NUM_POWERBALL:  # Single PB integer
            counts[NUM_MAIN + pb - 1] += 1.0  # Increments PB count
        elif isinstance(pb, (list, tuple)):  # Multiple PBs
            for p in pb:  # Iterates PB list/tuple
                if isinstance(p, int) and 1 <= p <= NUM_POWERBALL:  # Validate PB
                    counts[NUM_MAIN + p - 1] += 1.0  # Increments PB count

    # ---------- Step 5: Classical feature matrix ---------- #

    X = np.column_stack((  # Concatenate feature blocks horizontally
        F * mc.reshape(1, -1),  # Prefix frequencies reweighted by Monte Carlo probabilities
        F * rd.reshape(1, -1),  # Prefix frequencies reweighted by redundancy probabilities
        F * mk.reshape(1, -1),  # Prefix frequencies reweighted by Markov probabilities
        F * en.reshape(1, -1),  # Prefix frequencies reweighted by entropy probabilities
        F * fn.reshape(1, -1),  # Prefix frequencies reweighted by Bayesian fused probabilities
        np.tile(centroids.reshape(1, -1), (n_draws, 1)),  # Repeat centroids across all timesteps
        np.tile(clusters.reshape(1, -1),  (n_draws, 1)),  # Repeat clusters across all timesteps
    )).astype(float)  # Ensure float dtype for downstream models

    # ---------- Step 6: Time-aware train/validation split ---------- #

    n_val = max(1, int(0.15 * n_draws))  # Validation is the last ~15% of history (at least 1)
    X_train, X_val = X[:-n_val], X[-n_val:]  # Train on early history, validate on most recent history
    Y_train, Y_val = Y[:-n_val], Y[-n_val:]  # Same split for labels

    # ---------- Step 7: Compute global class weights ---------- #

    pos = Y_train.sum(axis=0)  # Positive counts per class across training set
    neg = Y_train.shape[0] - pos  # Negative counts per class

    cw = neg / (pos + MIN_PROB)  # Ratio-based positive class weight (avoid div-by-zero)
    cw = np.clip(cw, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT).astype(np.float32)  # Clamp to keep gradients sane

    class_weights = cw  # Store NumPy version globally
    class_weights_tf = tf.constant(class_weights, dtype=tf.float32)  # Store TF constant for loss function

    # ---------- Step 8: Train quantum encoder + reset kernel cache ---------- #

    try:
        train_quantum_encoder(X_train, Y_train)  # Fit/tune the quantum encoder using training data only
        logging.info("Quantum encoder training complete.")  # Confirm completion
    except Exception as e:
        logging.warning(f"Quantum encoder training failed: {e}")  # Continue even if quantum training fails

    _reset_quantum_kernel_cache()  # Ensure kernel prototypes are rebuilt under latest encoder weights

    # ---------- Step 9: Compute quantum and kernel features ---------- #

    try:
        Q_train = _force_width(  # Ensure quantum feature matrix has fixed width
            compute_quantum_matrix(X_train),  # Quantum feature extraction on training matrix
            QUANTUM_FEATURE_LEN,  # Expected width from quantum feature extractor
            "Q_train"  # Name used for error messages
        )
        Q_val = _force_width(  # Ensure quantum feature matrix has fixed width
            compute_quantum_matrix(X_val),  # Quantum feature extraction on validation matrix
            QUANTUM_FEATURE_LEN,  # Expected width from quantum feature extractor
            "Q_val"  # Name used for error messages
        )
    except Exception as e:
        logging.error(f"Quantum feature computation failed: {e}")  # Report quantum feature failure
        Q_train = np.zeros((X_train.shape[0], QUANTUM_FEATURE_LEN))  # Fallback to zeros with correct shape
        Q_val   = np.zeros((X_val.shape[0],   QUANTUM_FEATURE_LEN))  # Fallback to zeros with correct shape

    try:
        # Prototypes anchored to TRAIN by cache
        K_train_raw = build_quantum_kernel_features(  # Compute kernel features for training set
            X_train,  # Use training examples (defines prototype cache)
            num_prototypes=KERNEL_PROTOTYPES,  # Requested number of prototypes / output width
            seed=1337  # Deterministic prototype selection/reproducibility
        )
        K_val_raw = build_quantum_kernel_features(  # Compute kernel features for validation set
            X_val,  # Validation examples mapped against same cached prototypes
            num_prototypes=KERNEL_PROTOTYPES,  # Output width
            seed=1337  # Seed matches to ensure cache compatibility
        )

        K_train = _force_width(K_train_raw, KERNEL_PROTOTYPES, "K_train")  # Enforce fixed width on training kernel feats
        K_val   = _force_width(K_val_raw,   KERNEL_PROTOTYPES, "K_val")  # Enforce fixed width on validation kernel feats
    except Exception as e:
        logging.error(f"Kernel feature computation failed: {e}")  # Report kernel feature failure
        K_train = np.zeros((X_train.shape[0], KERNEL_PROTOTYPES))  # Fallback zero kernel features (train)
        K_val   = np.zeros((X_val.shape[0],   KERNEL_PROTOTYPES))  # Fallback zero kernel features (val)

    Xf_train = np.column_stack((X_train, Q_train, K_train)).astype(float)  # Fuse classical + quantum + kernel (train)
    Xf_val   = np.column_stack((X_val,   Q_val,   K_val)).astype(float)  # Fuse classical + quantum + kernel (val)

    input_dim = Xf_train.shape[1]  # Store final fused feature width for model input validation

    # ---------- Step 10: Train-only data augmentation ---------- #

    Xa = [Xf_train]  # List of training matrices to stack (original + noisy versions)
    Ya = [Y_train]  # Labels duplicated for each augmented copy

    for _ in range(DATA_AUGMENTATION_ROUNDS):  # Generate multiple noisy copies of training data
        Xa.append(Xf_train + np.random.normal(0.0, NOISE_STDDEV, Xf_train.shape))  # Add Gaussian noise in feature space
        Ya.append(Y_train)  # Keep labels unchanged (noise is only on features)

    Xa = np.vstack(Xa).astype(float)  # Stack augmented training matrices vertically
    Ya = np.vstack(Ya).astype(float)  # Stack labels to match augmented rows

    # ---------- Step 11: Model definition ---------- #

    model = keras.Sequential([  # Simple feedforward network for tabular fused features
        keras.layers.Input(shape=(input_dim,)),  # Define input layer shape explicitly

        keras.layers.Dense(128, activation="relu"),  # First dense layer
        keras.layers.BatchNormalization(),  # BatchNorm to stabilise hidden activations
        keras.layers.Dropout(0.25),  # Prevent memorising historical draws

        keras.layers.Dense(64, activation="relu"),  # Second dense layer
        keras.layers.BatchNormalization(),  # BatchNorm again
        keras.layers.Dropout(0.25),  # Regularisation

        keras.layers.Dense(32, activation="relu"),  # Third dense layer

        keras.layers.Dense(NUM_TOTAL, activation="sigmoid"),  # Output layer: independent probs per class (multi-label)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=8e-4, clipnorm=1.0),  # Adam optimiser with moderately high LR
        loss=weighted_bce,  # Custom weighted BCE loss defined above
        metrics=[
            keras.metrics.AUC(  # Multi-label AUC as a broad ranking sanity metric
                multi_label=True,  # Treat each output independently
                num_labels=NUM_TOTAL,  # Total labels = 50
                name="auc"  # Metric name in logs
            ),
            keras.metrics.BinaryAccuracy(name="bin_acc"),  # Thresholded accuracy (coarse, but stable)
            keras.metrics.MeanAbsoluteError(name="mae"),  # MAE across probabilities vs labels
        ],
    )

    # ---------- Step 12: Training ---------- #

    model.fit(
        Xa,  # Augmented training feature matrix (original + noisy copies)
        Ya,  # Augmented training labels (duplicated to match Xa)
        epochs=EPOCH_SIZE,  # Maximum number of training epochs (upper bound)
        batch_size=BATCH_SIZE,  # Mini-batch size per gradient update step
        validation_data=(Xf_val, Y_val),  # Validation uses clean (non-augmented) fused features
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                # Reduce learning rate if validation AUC stops improving
                monitor="val_auc",  # Watch validation AUC (ranking quality proxy)
                mode="max",  
                factor=0.7,  # Multiplys LR by this factor when plateau detected
                patience=8,  # Epochs to wait with no improvement before reducing LR
                min_lr=5e-6,  # Lower bound on learning rate
                verbose=1,  # Prints when LR is reduced
            ),
            keras.callbacks.EarlyStopping(
                # Stops training once validation AUC stops improving
                monitor="val_auc",  # Uses val_auc instead of val_loss (AUC better matches ranking objective)
                mode="max",  # Higher AUC is better
                patience=15,  # Epochs to wait with no improvement before stopping training
                min_delta=0.0003,  # Minimum improvement required to reset patience
                restore_best_weights=True,  # Restores model weights from best epoch by val_auc
                verbose=1,  # Print stop reason + restored epoch
            ),
            EpochLogger(),  # Custom callback (your project-specific epoch log formatting)
        ],
        verbose=1,  # Prints training progress per epoch
    )


    # ---------- Step 13: Inference ---------- #

    # Use post-history frequency (counts AFTER last draw)
    s = counts.sum()  # Total counts after processing all historical draws
    f_now = (  # Construct current frequency vector
        counts / s if s > 0 else np.ones(NUM_TOTAL) / NUM_TOTAL  # Normalised counts or uniform fallback
    ).reshape(1, -1).astype(float)  # Convert to row vector for feature construction

    x_now = np.column_stack((  # Build current classical feature row in the same structure as training X
        f_now * mc.reshape(1, -1),  # Frequency weighted by Monte Carlo
        f_now * rd.reshape(1, -1),  # Frequency weighted by redundancy
        f_now * mk.reshape(1, -1),  # Frequency weighted by Markov
        f_now * en.reshape(1, -1),  # Frequency weighted by entropy
        f_now * fn.reshape(1, -1),  # Frequency weighted by Bayesian fusion
        centroids.reshape(1, -1),  # Static centroid features
        clusters.reshape(1, -1),  # Static cluster features
    )).astype(float)  # Ensure float dtype

    try:
        q_now = _force_width(  # Ensure quantum feature width matches training expectation
            compute_quantum_matrix(x_now),  # Compute quantum features for current row
            QUANTUM_FEATURE_LEN,  # Expected width
            "q_now"  # Name for diagnostics
        )
    except Exception:
        q_now = np.zeros((1, QUANTUM_FEATURE_LEN))  # Fallback to zeros if quantum feature step fails

    try:
        k_now_raw = build_quantum_kernel_features(  # Compute kernel features for current row
            x_now,  # Current classical features
            num_prototypes=KERNEL_PROTOTYPES,  # Output width
            seed=1337  # Seed must match cache semantics used earlier
        )
        k_now = _force_width(k_now_raw, KERNEL_PROTOTYPES, "k_now")  # Enforce fixed kernel feature width
    except Exception:
        k_now = np.zeros((1, KERNEL_PROTOTYPES))  # Fallback to zeros if kernel step fails

    xf_now = np.column_stack((x_now, q_now, k_now)).astype(float)  # Fuse classical + quantum + kernel for inference

    if xf_now.shape[1] != input_dim:  # Hard check: inference feature width must match model input width
        logging.error(
            f"Inference width mismatch: got {xf_now.shape[1]}, expected {input_dim}"  # Log exact mismatch
        )
        pipeline.add_data(
            "deep_learning_predictions",  # Store fallback result
            np.ones(NUM_TOTAL) / NUM_TOTAL  # Uniform fallback
        )
        return  # Exit early to avoid invalid model input

    try:
        dl_pred = model.predict(xf_now, verbose=0).reshape(-1).astype(float)  # Run prediction and flatten to (50,)
    except Exception as e:
        logging.error(f"DL inference failed: {e}")  # Report inference failure
        pipeline.add_data(
            "deep_learning_predictions",  # Store fallback
            np.ones(NUM_TOTAL) / NUM_TOTAL  # Uniform fallback
        )
        return  # Exit early

    pipeline.add_data(
        "deep_learning_predictions",  # Store output in pipeline
        _prop_norm_vec(np.clip(dl_pred, 0.0, 1.0), "deep_learning_predictions")  # Ensure final probabilities are within [0, 1]
    )


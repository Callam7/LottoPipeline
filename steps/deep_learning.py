"""
Modified By: Callam
Project: Lotto Generator

Purpose:
    Deep learning prediction pipeline for lottery probabilities:
        - 40 main numbers
        - 10 Powerball numbers
    Output is always shape (50,), compatible with the ticket generator.

Design (single deterministic path):
    1) Build classical feature matrix from pipeline signals.
    2) Build smoothed multi-hot labels from historical draws.
    3) Train quantum encoder (SPSA) to tune circuit weights against labels.
    4) Compute quantum feature matrix from tuned circuit.
    5) Compute quantum kernel features (fidelity-based) from classical features.
    6) Fuse classical + quantum + kernel features into one feature matrix.
    7) Train deep learning model on fused features.
    8) Train a quantum predictive head on quantum features.
    9) Dynamically learn fusion weight from achieved AUCs.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pipeline import get_dynamic_params
from config.logs import EpochLogger
import logging
from datetime import datetime

from config.quantum_features import (
    compute_quantum_matrix,
    train_quantum_encoder,
    train_quantum_predictor,
    compute_quantum_prediction_matrix,
    QUANTUM_FEATURE_LEN,
)

from config.quantum_kernels import build_quantum_kernel_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== Constants ===================== #

NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL

EPOCH_SIZE = 60
BATCH_SIZE = 32

DATA_AUGMENTATION_ROUNDS = 50
NOISE_STDDEV = 0.03

MIN_CLASS_WEIGHT = 1.0
MAX_CLASS_WEIGHT = 8.0
MIN_PROB = 1e-12

LABEL_SMOOTHING = 0.05
KERNEL_PROTOTYPES = 24

# ===================== GLOBAL LOSS STATE ===================== #
class_weights = None
class_weights_tf = None
label_eps_tf = tf.constant(LABEL_SMOOTHING, dtype=tf.float32)

# ===================== Tensor-safe Weighted BCE ===================== #

def weighted_bce(y_true, y_pred):
    """
    Stable, graph-safe weighted BCE.
    y_true/y_pred: (batch, 50)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Manual label smoothing (shape preserving)
    y_true_s = y_true * (1.0 - label_eps_tf) + 0.5 * label_eps_tf

    # Per-label BCE -> (batch, 50)
    bce = keras.backend.binary_crossentropy(y_true_s, y_pred)

    # Per-label weights -> (batch, 50)
    w = y_true * class_weights_tf + (1.0 - y_true)

    return tf.reduce_mean(bce * w, axis=-1)

# ===================== Shape utilities ===================== #

def _ensure_2d(X, name):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")
    return X

def _force_width(M, width, name):
    """
    Ensures M is (n, width) via pad/trim.
    """
    M = _ensure_2d(M, name)
    n, d = M.shape
    if d == width:
        return M.astype(float)

    out = np.zeros((n, width), dtype=float)
    m = min(d, width)
    out[:, :m] = M[:, :m]
    logging.warning(f"{name} width {d} != {width}; padded/trimmed to {width}.")
    return out

def _safe_norm_vec(x, name):
    x = np.asarray(x, dtype=float).ravel()
    if x.size != NUM_TOTAL:
        raise ValueError(f"{name} expected len {NUM_TOTAL}, got {x.size}")
    return x / max(float(np.max(x)), MIN_PROB)

# ===================== Main Entry ===================== #

def deep_learning_prediction(pipeline):
    global class_weights, class_weights_tf

    # ---------------- Step 1: Load pipeline inputs ---------------- #

    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL)
        return

    monte_carlo = pipeline.get_data("monte_carlo")
    redundancy  = pipeline.get_data("redundancy")
    markov      = pipeline.get_data("markov_features")
    entropy     = pipeline.get_data("entropy_features")
    fusion_norm = pipeline.get_data("bayesian_fusion_norm")
    clusters    = pipeline.get_data("clusters")
    centroids   = pipeline.get_data("centroids")

    required = [monte_carlo, redundancy, markov, entropy, fusion_norm, clusters, centroids]
    if any(v is None for v in required):
        logging.error("Deep learning aborted: missing required pipeline features.")
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL)
        return

    # ---------------- Step 2: Build strict binary labels ---------------- #

    labels = []
    for draw in historical_data:
        y = np.zeros(NUM_TOTAL, dtype=float)

        for n in draw.get("numbers", []):
            if isinstance(n, int) and 1 <= n <= NUM_MAIN:
                y[n - 1] = 1.0

        pb = draw.get("powerball")
        if isinstance(pb, int) and 1 <= pb <= NUM_POWERBALL:
            y[NUM_MAIN + pb - 1] = 1.0
        elif isinstance(pb, (list, tuple)):
            for p in pb:
                if isinstance(p, int) and 1 <= p <= NUM_POWERBALL:
                    y[NUM_MAIN + p - 1] = 1.0

        labels.append(y)

    Y = np.asarray(labels, dtype=float)
    n_draws = Y.shape[0]
    if n_draws < 10:
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL)
        return

    # ---------------- Step 3: Normalize pipe vectors ---------------- #

    mc = _safe_norm_vec(monte_carlo, "monte_carlo")
    rd = _safe_norm_vec(redundancy,  "redundancy")
    mk = _safe_norm_vec(markov,      "markov_features")
    en = _safe_norm_vec(entropy,     "entropy_features")
    fn = _safe_norm_vec(fusion_norm, "bayesian_fusion_norm")

    clusters = np.asarray(clusters, dtype=float).ravel()
    centroids = np.asarray(centroids, dtype=float).ravel()

    if clusters.size != NUM_TOTAL:
        clusters = np.zeros(NUM_TOTAL, dtype=float)
    if centroids.size != NUM_TOTAL:
        centroids = np.zeros(NUM_TOTAL, dtype=float)

    # ---------------- Step 4: Build causal prefix frequency F[t] ---------------- #

    F = np.zeros((n_draws, NUM_TOTAL), dtype=float)
    counts = np.zeros(NUM_TOTAL, dtype=float)

    for t in range(n_draws):
        s = counts.sum()
        F[t] = counts / s if s > 0 else np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL

        for n in historical_data[t].get("numbers", []):
            if isinstance(n, int) and 1 <= n <= NUM_MAIN:
                counts[n - 1] += 1.0

        pb = historical_data[t].get("powerball")
        if isinstance(pb, int) and 1 <= pb <= NUM_POWERBALL:
            counts[NUM_MAIN + pb - 1] += 1.0
        elif isinstance(pb, (list, tuple)):
            for p in pb:
                if isinstance(p, int) and 1 <= p <= NUM_POWERBALL:
                    counts[NUM_MAIN + p - 1] += 1.0

    # ---------------- Step 5: Classical features (keeps np.column_stack) ---------------- #

    X = np.column_stack((
        F * mc.reshape(1, -1),
        F * rd.reshape(1, -1),
        F * mk.reshape(1, -1),
        F * en.reshape(1, -1),
        F * fn.reshape(1, -1),
        np.tile(centroids.reshape(1, -1), (n_draws, 1)),
        np.tile(clusters.reshape(1, -1),  (n_draws, 1)),
    )).astype(float)

    # ---------------- Step 6: Time-aware train/val split ---------------- #

    n_val = max(1, int(0.15 * n_draws))
    X_train, X_val = X[:-n_val], X[-n_val:]
    Y_train, Y_val = Y[:-n_val], Y[-n_val:]

    # ---------------- Step 7: GLOBAL class weights ---------------- #

    pos = Y_train.sum(axis=0)
    neg = Y_train.shape[0] - pos
    cw = neg / (pos + MIN_PROB)
    cw = np.clip(cw, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT).astype(np.float32)

    class_weights = cw
    class_weights_tf = tf.constant(class_weights, dtype=tf.float32)

    # ---------------- Step 8: Quantum encoder (train only) ---------------- #

    try:
        train_quantum_encoder(X_train, Y_train)
        logging.info("Quantum encoder training complete.")
    except Exception as e:
        logging.warning(f"Quantum encoder training failed: {e}")

    # ---------------- Step 9: Compute quantum + kernel with HARD width contract ---------------- #

    try:
        Q_train = _force_width(compute_quantum_matrix(X_train), QUANTUM_FEATURE_LEN, "Q_train")
        Q_val   = _force_width(compute_quantum_matrix(X_val),   QUANTUM_FEATURE_LEN, "Q_val")
    except Exception as e:
        logging.error(f"Quantum feature computation failed: {e}")
        Q_train = np.zeros((X_train.shape[0], QUANTUM_FEATURE_LEN), dtype=float)
        Q_val   = np.zeros((X_val.shape[0],   QUANTUM_FEATURE_LEN), dtype=float)

    try:
        # IMPORTANT: call with keywords so signature mismatches can’t silently break widths
        K_train_raw = build_quantum_kernel_features(X_train, num_prototypes=KERNEL_PROTOTYPES, seed=1337)
        K_val_raw   = build_quantum_kernel_features(X_val,   num_prototypes=KERNEL_PROTOTYPES, seed=1337)

        K_train = _force_width(K_train_raw, KERNEL_PROTOTYPES, "K_train")
        K_val   = _force_width(K_val_raw,   KERNEL_PROTOTYPES, "K_val")
    except Exception as e:
        logging.error(f"Kernel feature computation failed: {e}")
        K_train = np.zeros((X_train.shape[0], KERNEL_PROTOTYPES), dtype=float)
        K_val   = np.zeros((X_val.shape[0],   KERNEL_PROTOTYPES), dtype=float)

    Xf_train = np.column_stack((X_train, Q_train, K_train)).astype(float)
    Xf_val   = np.column_stack((X_val,   Q_val,   K_val)).astype(float)

    input_dim = Xf_train.shape[1]

    # ---------------- Step 10: Augment TRAIN ONLY ---------------- #

    Xa = [Xf_train]
    Ya = [Y_train]

    for _ in range(DATA_AUGMENTATION_ROUNDS):
        Xa.append(Xf_train + np.random.normal(0.0, NOISE_STDDEV, Xf_train.shape))
        Ya.append(Y_train)

    Xa = np.vstack(Xa).astype(float)
    Ya = np.vstack(Ya).astype(float)

    # ---------------- Step 11: Model ---------------- #

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(NUM_TOTAL, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=weighted_bce,
        metrics=[
            keras.metrics.AUC(multi_label=True, num_labels=NUM_TOTAL, name="auc"),
            keras.metrics.BinaryAccuracy(name="bin_acc"),
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    # ---------------- Step 12: Train ---------------- #

    model.fit(
        Xa, Ya,
        epochs=EPOCH_SIZE,
        batch_size=BATCH_SIZE,
        validation_data=(Xf_val, Y_val),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                mode="max",
                factor=0.7,
                patience=6,
                min_lr=1e-6,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            EpochLogger(),
        ],
        verbose=1,
    )

    # ---------------- Step 13: Inference (MUST MATCH TRAIN WIDTH) ---------------- #

    f_now = F[-1].reshape(1, -1).astype(float)

    x_now = np.column_stack((
        f_now * mc.reshape(1, -1),
        f_now * rd.reshape(1, -1),
        f_now * mk.reshape(1, -1),
        f_now * en.reshape(1, -1),
        f_now * fn.reshape(1, -1),
        centroids.reshape(1, -1),
        clusters.reshape(1, -1),
    )).astype(float)

    # Enforce same widths as train
    try:
        q_now = _force_width(compute_quantum_matrix(x_now), QUANTUM_FEATURE_LEN, "q_now")
    except Exception:
        q_now = np.zeros((1, QUANTUM_FEATURE_LEN), dtype=float)

    try:
        k_now_raw = build_quantum_kernel_features(x_now, num_prototypes=KERNEL_PROTOTYPES, seed=1337)
        k_now = _force_width(k_now_raw, KERNEL_PROTOTYPES, "k_now")
    except Exception:
        k_now = np.zeros((1, KERNEL_PROTOTYPES), dtype=float)

    xf_now = np.column_stack((x_now, q_now, k_now)).astype(float)

    if xf_now.shape[1] != input_dim:
        logging.error(f"Inference width mismatch: got {xf_now.shape[1]}, expected {input_dim}. Using uniform fallback.")
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL)
        return

    try:
        dl_pred = model.predict(xf_now, verbose=0).reshape(-1).astype(float)
    except Exception as e:
        logging.error(f"DL inference failed: {e}")
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL)
        return

    pipeline.add_data("deep_learning_predictions", np.clip(dl_pred, 0.0, 1.0))









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

import numpy as np
import tensorflow as tf
from tensorflow import keras
from config.logs import EpochLogger
from pipeline import get_dynamic_params
import logging
from config.quantum_features import (
    compute_quantum_matrix,
    train_quantum_encoder,
    QUANTUM_FEATURE_LEN,
)
from config import quantum_kernels as qk
from config.quantum_kernels import build_quantum_kernel_features
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== Constants ===================== #
NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL

EPOCH_SIZE = 60
BATCH_SIZE = 32
DATA_AUGMENTATION_ROUNDS = 3
NOISE_STDDEV = 0.01
MIN_CLASS_WEIGHT = 1.0
MAX_CLASS_WEIGHT = 4.0
MIN_PROB = 1e-7
KERNEL_PROTOTYPES = 24

class_weights = None
class_weights_tf = None

# ===================== CLASSICAL FEATURE BLOCK DEFINITION (Future-Proof) ===================== #
# This list and helper function serve as the single source of truth for the order
# of classical feature blocks. When the future adaptor adds, removes, or reorders
# pipes, only this definition needs to be updated. This prevents desync between
# feature construction (Step 5) and ablation logic.

CLASSICAL_FEATURE_BLOCKS: List[str] = [
    "bayesian_fusion_norm",
    "monte_carlo",
    "redundancy",
    "markov_features",
    "entropy_features",
    "centroids",
    "clusters",
]


def _get_classical_feature_arrays(
    mc: np.ndarray,
    rd: np.ndarray,
    mk: np.ndarray,
    en: np.ndarray,
    fn: np.ndarray,
    centroids: np.ndarray,
    clusters: np.ndarray,
) -> List[Tuple[str, np.ndarray]]:
    """Returns the classical feature blocks in the exact order used in Step 5."""
    return [
        ("bayesian_fusion_norm", fn),
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
    qk._cached_proto_states = None
    qk._cached_num_prototypes = None
    qk._cached_seed = None


# ===================== Weighted BCE ===================== #
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
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, MIN_PROB, 1.0 - MIN_PROB)
    bce = keras.backend.binary_crossentropy(y_true, y_pred)
    w = y_true * class_weights_tf + (1.0 - y_true)
    return tf.reduce_mean(bce * w, axis=-1)


# ===================== Shape utilities ===================== #
def _ensure_2d(X: np.ndarray, name: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")
    return X


def _force_width(M: np.ndarray, width: int, name: str) -> np.ndarray:
    M = _ensure_2d(M, name)
    n, d = M.shape
    if d == width:
        return M.astype(float)
    out = np.zeros((n, width), dtype=float)
    m = min(d, width)
    out[:, :m] = M[:, :m]
    logging.warning(f"{name} width {d} != {width}; padded/trimmed to {width}.")
    return out


def _prob_norm_vec(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size != NUM_TOTAL:
        raise ValueError(f"{name} expected len {NUM_TOTAL}, got {x.size}")
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= 0.0:
        return np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL
    return x / s


def _compute_pipe_importance(
    model: keras.Model, Xf_val: np.ndarray, Y_val: np.ndarray
) -> Dict[str, float]:
    """
    Post-training ablation (future-proof version).
    Uses CLASSICAL_FEATURE_BLOCKS as the single source of truth.
    Zeros each feature block one at a time and measures change in macro AUC.

    delta = modified_auc - baseline_auc
        < 0  → pipe was useful (removing it hurt performance)
        > 0  → pipe was noisy (removing it helped)
        ≈ 0  → pipe contributed almost nothing
    """
    from sklearn.metrics import roc_auc_score   # moved outside the loop for speed

    block_size = NUM_TOTAL
    pipes: Dict[str, Tuple[int, int]] = {}
    for i, name in enumerate(CLASSICAL_FEATURE_BLOCKS):
        start = i * block_size
        end = start + block_size
        pipes[name] = (start, end)

    # Baseline
    baseline_pred = model.predict(Xf_val, verbose=0)
    try:
        baseline_auc = roc_auc_score(Y_val, baseline_pred, average="macro")
    except Exception:
        baseline_auc = 0.5

    importance: Dict[str, float] = {}

    for pipe_name, (start_col, end_col) in pipes.items():
        X_modified = Xf_val.copy()
        X_modified[:, start_col:end_col] = 0.0

        modified_pred = model.predict(X_modified, verbose=0)
        try:
            modified_auc = roc_auc_score(Y_val, modified_pred, average="macro")
        except Exception:
            modified_auc = baseline_auc

        delta = modified_auc - baseline_auc
        importance[pipe_name] = float(delta)

    # Print clean summary
    print("\n" + "=" * 72)
    print("POST-TRAINING PIPE IMPORTANCE ANALYSIS (future-proof)")
    print("=" * 72)
    print(f"Baseline Validation AUC: {baseline_auc:.4f}")
    print("-" * 72)
    print(f"{'Pipe Name':<28} {'Modified AUC':>14} {'Delta (Impact)':>16}")
    print("-" * 72)
    for name in CLASSICAL_FEATURE_BLOCKS:
        d = importance.get(name, 0.0)
        # We don't have per-pipe modified_auc here anymore for brevity, but you can add it if wanted
        print(f"{name:<28} {'N/A':>14} {d:>16.4f}")

    most_impactful = min(importance, key=importance.get)
    print("-" * 72)
    print(f"Most impactful pipe (largest positive contribution): {most_impactful} "
          f"(delta={importance[most_impactful]:.4f})")
    print("=" * 72 + "\n")

    return importance

    # Clean, relevant summary output (no log flooding)
    print("\n" + "=" * 72)
    print("POST-TRAINING PIPE IMPORTANCE ANALYSIS")
    print("=" * 72)
    print(f"Baseline Validation AUC: {baseline_auc:.4f}")
    print("-" * 72)
    print(f"{'Pipe Name':<28} {'Modified AUC':>14} {'Delta (Impact)':>16}")
    print("-" * 72)

    for r in results:
        print(f"{r['pipe']:<28} {r['modified_auc']:>14.4f} {r['delta']:>16.4f}")

    weakest_pipe = min(importance, key=importance.get)
    weakest_delta = importance[weakest_pipe]

    print("-" * 72)
    print(f"Weakest Pipe Identified: {weakest_pipe} (Delta: {weakest_delta:.4f})")
    print("=" * 72 + "\n")

    return importance


# ===================== Main entry ===================== #
def deep_learning_prediction(pipeline: Any) -> None:
    global class_weights, class_weights_tf

    # Resolve dynamic training parameters (supports Optuna overrides)
    _, dynamic_epochs = get_dynamic_params(
        len(pipeline.get_data("historical_data") or [])
    )
    optuna_params = pipeline.get_data("optuna_best_params") or {}
    epochs = optuna_params.get("epochs", dynamic_epochs)
    batch_size = optuna_params.get("batch_size", BATCH_SIZE)
    learning_rate = optuna_params.get("learning_rate", 8e-4)
    dropout_rate = optuna_params.get("dropout_rate", 0.25)

    logging.info(
        f"DL params -> epochs={epochs}, "
        f"batch_size={batch_size}, "
        f"lr={learning_rate}, "
        f"dropout={dropout_rate}"
    )

    # ---------- Step 1: Load pipeline inputs ---------- #
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        pipeline.add_data(
            "deep_learning_predictions",
            np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL
        )
        return

    monte_carlo = pipeline.get_data("monte_carlo")
    redundancy = pipeline.get_data("redundancy")
    markov = pipeline.get_data("markov_features")
    entropy = pipeline.get_data("entropy_features")
    fusion_norm = pipeline.get_data("bayesian_fusion_norm")
    clusters = pipeline.get_data("clusters")
    centroids = pipeline.get_data("centroids")

    required = [monte_carlo, redundancy, markov, entropy, fusion_norm, clusters, centroids]
    if any(v is None for v in required):
        logging.error("Deep learning aborted: missing required pipeline features.")
        pipeline.add_data(
            "deep_learning_predictions",
            np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL
        )
        return

    # ---------- Step 2: Build strict multi-hot labels ---------- #
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
        pipeline.add_data(
            "deep_learning_predictions",
            np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL
        )
        return

    # ---------- Step 3: Normalise pipeline vectors ---------- #
    mc = _prob_norm_vec(monte_carlo, "monte_carlo")
    rd = _prob_norm_vec(redundancy, "redundancy")
    mk = _prob_norm_vec(markov, "markov_features")
    en = _prob_norm_vec(entropy, "entropy_features")
    fn = _prob_norm_vec(fusion_norm, "bayesian_fusion_norm")
    clusters = np.asarray(clusters, dtype=float).ravel()
    centroids = np.asarray(centroids, dtype=float).ravel()

    if clusters.size != NUM_TOTAL:
        clusters = np.zeros(NUM_TOTAL, dtype=float)
    if centroids.size != NUM_TOTAL:
        centroids = np.zeros(NUM_TOTAL, dtype=float)

    # ---------- Step 4: Build causal prefix frequencies ---------- #
    F = np.zeros((n_draws, NUM_TOTAL), dtype=float)
    counts = np.zeros(NUM_TOTAL, dtype=float)
    for t in range(n_draws):
        s = counts.sum()
        F[t] = counts / s if s > 0 else np.ones(NUM_TOTAL) / NUM_TOTAL
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

    # ---------- Step 5: Classical feature matrix (Future-Proof) ---------- #
    # Uses the central definition so the feature order stays consistent
    # with the ablation logic in _compute_pipe_importance.
    feature_blocks = _get_classical_feature_arrays(mc, rd, mk, en, fn, centroids, clusters)

    X_parts = []
    for name, arr in feature_blocks:
        if name in ["centroids", "clusters"]:
            X_parts.append(np.tile(arr.reshape(1, -1), (n_draws, 1)))
        else:
            X_parts.append(F * arr.reshape(1, -1))

    X = np.column_stack(X_parts).astype(float)

    # ---------- Step 6: Time-aware train/validation split ---------- #
    n_val = max(1, int(0.15 * n_draws))
    X_train, X_val = X[:-n_val], X[-n_val:]
    Y_train, Y_val = Y[:-n_val], Y[-n_val:]

    # ---------- Step 7: Compute global class weights ---------- #
    pos = Y_train.sum(axis=0)
    neg = Y_train.shape[0] - pos
    cw = neg / (pos + MIN_PROB)
    cw = np.clip(cw, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT).astype(np.float32)
    class_weights = cw
    class_weights_tf = tf.constant(class_weights, dtype=tf.float32)

    # ---------- Step 8: Train quantum encoder + reset kernel cache ---------- #
    try:
        train_quantum_encoder(X_train, Y_train)
        logging.info("Quantum encoder training complete.")
    except Exception as e:
        logging.warning(f"Quantum encoder training failed: {e}")

    _reset_quantum_kernel_cache()

    # ---------- Step 9: Compute quantum and kernel features ---------- #
    try:
        Q_train = _force_width(
            compute_quantum_matrix(X_train),
            QUANTUM_FEATURE_LEN,
            "Q_train"
        )
        Q_val = _force_width(
            compute_quantum_matrix(X_val),
            QUANTUM_FEATURE_LEN,
            "Q_val"
        )
    except Exception as e:
        logging.error(f"Quantum feature computation failed: {e}")
        Q_train = np.zeros((X_train.shape[0], QUANTUM_FEATURE_LEN))
        Q_val = np.zeros((X_val.shape[0], QUANTUM_FEATURE_LEN))

    try:
        K_train_raw = build_quantum_kernel_features(
            X_train, num_prototypes=KERNEL_PROTOTYPES, seed=1337
        )
        K_val_raw = build_quantum_kernel_features(
            X_val, num_prototypes=KERNEL_PROTOTYPES, seed=1337
        )
        K_train = _force_width(K_train_raw, KERNEL_PROTOTYPES, "K_train")
        K_val = _force_width(K_val_raw, KERNEL_PROTOTYPES, "K_val")
    except Exception as e:
        logging.error(f"Kernel feature computation failed: {e}")
        K_train = np.zeros((X_train.shape[0], KERNEL_PROTOTYPES))
        K_val = np.zeros((X_val.shape[0], KERNEL_PROTOTYPES))

    Xf_train = np.column_stack((X_train, Q_train, K_train)).astype(float)
    Xf_val = np.column_stack((X_val, Q_val, K_val)).astype(float)
    input_dim = Xf_train.shape[1]

    # ---------- Step 10: Train-only data augmentation ---------- #
    Xa = [Xf_train]
    Ya = [Y_train]
    for _ in range(DATA_AUGMENTATION_ROUNDS):
        Xa.append(Xf_train + np.random.normal(0.0, NOISE_STDDEV, Xf_train.shape))
        Ya.append(Y_train)
    Xa = np.vstack(Xa).astype(float)
    Ya = np.vstack(Ya).astype(float)

    # ---------- Step 11: Model definition ---------- #
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(NUM_TOTAL, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=weighted_bce,
        metrics=[
            keras.metrics.AUC(multi_label=True, num_labels=NUM_TOTAL, name="auc"),
            keras.metrics.BinaryAccuracy(name="bin_acc"),
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    # ---------- Step 12: Training ---------- #
    model.fit(
        Xa, Ya,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(Xf_val, Y_val),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", mode="max", factor=0.7,
                patience=8, min_lr=5e-6, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=15,
                min_delta=0.0003, restore_best_weights=True, verbose=1
            ),
            EpochLogger(),
        ],
        verbose=1,
    )

    # ===================== NEW: Post-training Pipe Importance ===================== #
    try:
        pipe_importance = _compute_pipe_importance(model, Xf_val, Y_val)
        pipeline.add_data("pipe_importance", pipe_importance)
        logging.info("Post-training pipe importance analysis completed successfully.")
    except Exception as e:
        logging.error(f"Pipe importance calculation FAILED: {e}")
        logging.error("Storing empty dict so bridge does not crash.")
        pipeline.add_data("pipe_importance", {})

    # ---------- Step 13: Inference ---------- #
    s = counts.sum()
    f_now = (counts / s if s > 0 else np.ones(NUM_TOTAL) / NUM_TOTAL
             ).reshape(1, -1).astype(float)

    x_now = np.column_stack((
        f_now * mc.reshape(1, -1),
        f_now * rd.reshape(1, -1),
        f_now * mk.reshape(1, -1),
        f_now * en.reshape(1, -1),
        f_now * fn.reshape(1, -1),
        centroids.reshape(1, -1),
        clusters.reshape(1, -1),
    )).astype(float)

    try:
        q_now = _force_width(
            compute_quantum_matrix(x_now), QUANTUM_FEATURE_LEN, "q_now"
        )
    except Exception:
        q_now = np.zeros((1, QUANTUM_FEATURE_LEN))

    try:
        k_now_raw = build_quantum_kernel_features(
            x_now, num_prototypes=KERNEL_PROTOTYPES, seed=1337
        )
        k_now = _force_width(k_now_raw, KERNEL_PROTOTYPES, "k_now")
    except Exception:
        k_now = np.zeros((1, KERNEL_PROTOTYPES))

    xf_now = np.column_stack((x_now, q_now, k_now)).astype(float)

    if xf_now.shape[1] != input_dim:
        logging.error(f"Inference width mismatch: got {xf_now.shape[1]}, expected {input_dim}")
        pipeline.add_data(
            "deep_learning_predictions",
            np.ones(NUM_TOTAL) / NUM_TOTAL
        )
        return

    try:
        dl_pred = model.predict(xf_now, verbose=0).reshape(-1).astype(float)
    except Exception as e:
        logging.error(f"DL inference failed: {e}")
        pipeline.add_data(
            "deep_learning_predictions",
            np.ones(NUM_TOTAL) / NUM_TOTAL
        )
        return

    pipeline.add_data(
        "deep_learning_predictions",
        _prob_norm_vec(np.clip(dl_pred, 0.0, 1.0), "deep_learning_predictions")
    )
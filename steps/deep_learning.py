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
    3) Apply recency decay to emphasize newer draws.
    4) Train quantum encoder (SPSA) to tune circuit weights against labels.
    5) Compute quantum feature matrix and fuse with classical features.
    6) Train deep learning model on fused features.
    7) Train a quantum predictive head on quantum features.
    8) Dynamically learn fusion weight from achieved AUCs.
    9) Final prediction = AUC-weighted blend of DL + quantum-head probabilities.

Notes:
    - The quantum circuit is a black-box embedding. No TF backprop through it.
    - SPSA is gradient-free and stable on simulators.
    - Recency decay improves measurable predictive alignment to evolving statistics.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pipeline import get_dynamic_params
from config.logs import EpochLogger
import logging
from datetime import datetime

from config.quantum_features import (
    compute_quantum_features,
    compute_quantum_matrix,
    train_quantum_encoder,
    train_quantum_predictor,
    compute_quantum_prediction_matrix,
    QUANTUM_FEATURE_LEN,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Constants ---------------- #

NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL

LABEL_SMOOTH = 0.95

DATA_AUGMENTATION_ROUNDS = 100
NOISE_STDDEV = 0.05
BATCH_SIZE = 32
EPOCH_SIZE = 40

MIN_CLASS_WEIGHT = 1.0
MAX_CLASS_WEIGHT = 10.0
MIN_PROB = 1e-12

# Recency decay controls
DECAY_RATE = 0.01  # stronger = more emphasis on recent draws


def deep_learning_prediction(pipeline):
    """
    Main entry point used by the pipeline.

    Writes:
        pipeline["deep_learning_predictions"] = np.ndarray shape (50,)
    """

    # ---------------- Step 1: Extract Pipeline Data ---------------- #

    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL) / NUM_TOTAL)
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
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    # ---------------- Step 2: Safe Normalization ---------------- #

    def _safe_norm(x):
        x = np.asarray(x, dtype=float)
        max_val = max(np.max(x), MIN_PROB)
        return x / max_val

    monte_carlo = _safe_norm(monte_carlo)
    redundancy  = _safe_norm(redundancy)
    markov      = _safe_norm(markov)
    entropy     = _safe_norm(entropy)
    fusion_norm = _safe_norm(fusion_norm)

    clusters = np.asarray(clusters, dtype=int)
    centroids = np.asarray(centroids, dtype=float)
    centroids_for_rows = centroids[clusters]

    # ---------------- Step 3: Classical Feature Block ---------------- #

    base_features = np.column_stack(
        (
            monte_carlo,
            redundancy,
            markov,
            entropy,
            fusion_norm,
            centroids_for_rows,
        )
    )  # shape (N, d_classical)

    # ---------------- Step 4: Label Construction ---------------- #

    labels = []
    dates = []

    for draw in historical_data:
        arr = np.zeros(NUM_TOTAL, dtype=float)
        for num in draw.get("numbers", []):
            if 1 <= num <= NUM_TOTAL:
                arr[num - 1] = LABEL_SMOOTH
        labels.append(arr)

        d_str = draw.get("date") or draw.get("draw_date")
        if d_str:
            try:
                dates.append(datetime.strptime(d_str, "%Y-%m-%d"))
            except Exception:
                dates.append(None)
        else:
            dates.append(None)

    labels = np.asarray(labels, dtype=float)

    # ---------------- Step 5: Recency Decay Weights ---------------- #
    # Newer draws receive higher weight during training.

    n = len(labels)

    if all(d is not None for d in dates):
        max_date = max(dates)
        ages = np.array([(max_date - d).days for d in dates], dtype=float)
    else:
        # Fallback: assume list is chronological
        ages = np.arange(n, dtype=float)[::-1]

    sample_weights = np.exp(-DECAY_RATE * ages)
    sample_weights = sample_weights / np.mean(sample_weights)

    # ---------------- Step 6: Train Quantum Encoder ---------------- #

    try:
        train_quantum_encoder(base_features, labels)
        logging.info("Quantum encoder training complete.")
    except Exception as e:
        logging.warning(f"Quantum encoder training failed: {e}")

    # ---------------- Step 7: Compute Quantum Feature Matrix ---------------- #

    try:
        quantum_features = compute_quantum_matrix(base_features)
        features = np.hstack((base_features, quantum_features))
    except Exception as e:
        logging.warning(f"Quantum feature build failed: {e}")
        features = base_features

    # ---------------- Step 8: Class Weights ---------------- #

    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    class_weights = neg / (pos + MIN_PROB)
    class_weights = np.clip(class_weights, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT)

    # ---------------- Step 9: Weighted BCE ---------------- #

    def weighted_bce(weights):
        weights_tf = tf.constant(weights, dtype=tf.float32)

        def _loss(y_true, y_pred):
            bce = keras.backend.binary_crossentropy(y_true, y_pred)
            w = y_true * weights_tf + (1.0 - y_true)
            return keras.backend.mean(w * bce, axis=-1)

        return _loss

    weighted_loss = weighted_bce(class_weights)

    # ---------------- Step 10: Gaussian Augmentation ---------------- #

    augmented_f = []
    augmented_l = []
    augmented_w = []

    for _ in range(DATA_AUGMENTATION_ROUNDS):
        noise = np.random.normal(0.0, NOISE_STDDEV, features.shape)
        augmented_f.append(features + noise)
        augmented_l.append(labels)
        augmented_w.append(sample_weights)

    augmented_f = np.vstack(augmented_f)
    augmented_l = np.vstack(augmented_l)
    augmented_w = np.hstack(augmented_w)

    # ---------------- Step 11: Dynamic Epoch Discovery ---------------- #

    num_draws = len(historical_data)
    _, dynamic_epochs = get_dynamic_params(num_draws)
    dynamic_epochs = min(dynamic_epochs, EPOCH_SIZE)  # <-- cap restored to 68

    input_dim = augmented_f.shape[1]

    # ---------------- Step 12: DL Architecture ---------------- #

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),

            keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(0.001),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),

            keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(0.001),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),

            keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(0.001),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(NUM_TOTAL, activation="sigmoid"),
        ]
    )

    # ---------------- Step 13: Compile ---------------- #

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=weighted_loss,
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    # ---------------- Step 14: Callbacks ---------------- #

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.8,
            patience=6,
            verbose=1,
            min_lr=1e-7,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            min_delta=0.002,
            restore_best_weights=True,
            verbose=1,
        ),
        EpochLogger(),
    ]

    # ---------------- Step 15: Train DL ---------------- #

    model.fit(
        augmented_f,
        augmented_l,
        sample_weight=augmented_w,
        epochs=dynamic_epochs,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        verbose=1,
        callbacks=callbacks,
    )

    # ---------------- Step 16: DL Inference ---------------- #

    dl_pred_matrix = model.predict(features, verbose=0)
    dl_pred = np.mean(dl_pred_matrix, axis=0).astype(float)

    # ---------------- Step 17: Quantum Predictive Head ---------------- #

    try:
        train_quantum_predictor(base_features, labels)
        logging.info("Quantum predictive head training complete.")
    except Exception as e:
        logging.warning(f"Quantum predictor training failed: {e}")

    try:
        q_pred_matrix = compute_quantum_prediction_matrix(base_features)
        q_pred = np.mean(q_pred_matrix, axis=0).astype(float)
    except Exception as e:
        logging.warning(f"Quantum prediction failed: {e}")
        q_pred_matrix = np.zeros_like(labels, dtype=float)  # keep AUC stage safe
        q_pred = np.zeros(NUM_TOTAL, dtype=float)

    # ---------------- Step 18: Learn Fusion Weight from AUC ---------------- #

    def _auc_score(y_true, y_pred):
        """
        Multi-label AUC for (n_samples, 50).

        Hard guarantees:
            - converts to float32 numpy
            - aligns leading dimension to prevent augmented leakage
            - uses multi_label=True so Keras doesn't misinterpret shape
        """
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)

        # Enforce leading-dimension match deterministically
        n0 = y_true.shape[0]
        n1 = y_pred.shape[0]
        n = min(n0, n1)

        if n0 != n1:
            y_true = y_true[:n]
            y_pred = y_pred[:n]

        m = keras.metrics.AUC(multi_label=True, num_labels=y_true.shape[1])
        m.update_state(y_true, y_pred)
        return float(m.result().numpy())

    try:
        dl_auc = _auc_score(labels, dl_pred_matrix)
        q_auc = _auc_score(labels, q_pred_matrix)
        denom = max(dl_auc + q_auc, 1e-9)
        alpha = dl_auc / denom
    except Exception as e:
        logging.warning(f"Fusion AUC evaluation failed: {e}")
        alpha = 0.7

    # ---------------- Step 19: Final Fusion + Output ---------------- #

    final_prediction = alpha * dl_pred + (1.0 - alpha) * q_pred
    final_prediction = np.clip(final_prediction, 0.0, 1.0).astype(float)

    pipeline.add_data("deep_learning_predictions", final_prediction)
    logging.info(f"Deep learning predictions complete. Learned alpha={alpha:.4f}")



## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Deep Learning Prediction for Lottery Numbers (Shape 50)
## Description:
##    Predict probabilities for 40 main numbers + 10 Powerball numbers using historical,
##    Monte Carlo, clustering, redundancy, Markov, entropy, and Bayesian fusion features.
##    Features are concatenated into shape 50 for deep learning input.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pipeline import get_dynamic_params
from config.logs import EpochLogger
import logging

from steps.quantum_features import compute_quantum_features, QUANTUM_FEATURE_LEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants ---
NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL
LABEL_SMOOTH = 0.95
DATA_AUGMENTATION_ROUNDS = 100
NOISE_STDDEV = 0.05
BATCH_SIZE = 32
MIN_CLASS_WEIGHT = 1.0
MAX_CLASS_WEIGHT = 10.0
MIN_PROB = 1e-12


def deep_learning_prediction(pipeline):

    # Step 1: pull historical + feature data from pipeline
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL) / NUM_TOTAL)
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
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    # Step 2: safe normalization
    def _safe_norm(x):
        x = np.asarray(x, dtype=float)
        max_val = max(np.max(x), MIN_PROB)
        return x / max_val

    monte_carlo = _safe_norm(monte_carlo)
    redundancy = _safe_norm(redundancy)
    markov = _safe_norm(markov)
    entropy = _safe_norm(entropy)
    fusion_norm = _safe_norm(fusion_norm)

    clusters = np.asarray(clusters, dtype=int)
    centroids = np.asarray(centroids, dtype=float)
    centroids_for_rows = centroids[clusters]

    # Step 3: classical feature block
    base_features = np.column_stack(
        (
            monte_carlo,
            redundancy,
            markov,
            entropy,
            fusion_norm,
            centroids_for_rows,
        )
    )

    # Step 4: quantum features per row
    quantum_rows = []
    for row in base_features:
        quantum_rows.append(compute_quantum_features(row))

    quantum_rows = np.vstack(quantum_rows)

    # final feature matrix
    features = np.hstack((base_features, quantum_rows))

    # Step 5: smoothed binary labels
    labels = []
    for draw in historical_data:
        arr = np.zeros(NUM_TOTAL, dtype=float)
        for num in draw.get("numbers", []):
            if 1 <= num <= NUM_TOTAL:
                arr[num - 1] = LABEL_SMOOTH
        labels.append(arr)

    labels = np.asarray(labels, dtype=float)

    # Step 6: class weights
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    class_weights = neg_counts / (pos_counts + MIN_PROB)
    class_weights = np.clip(class_weights, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT)

    # Step 7: weighted BCE
    def weighted_binary_crossentropy(weights):
        weights = tf.constant(weights, dtype=tf.float32)

        def loss_fn(y_true, y_pred):
            bce = keras.backend.binary_crossentropy(y_true, y_pred)
            w = y_true * weights + (1.0 - y_true)
            return keras.backend.mean(w * bce, axis=-1)

        return loss_fn

    # Step 8: gaussian augmentation on features
    augmented_f = []
    augmented_l = []

    for _ in range(DATA_AUGMENTATION_ROUNDS):
        noise = np.random.normal(0.0, NOISE_STDDEV, features.shape)
        augmented_f.append(features + noise)
        augmented_l.extend(labels)

    augmented_f = np.vstack(augmented_f)
    augmented_l = np.vstack(augmented_l)

    # Step 9: dynamic epoch system (then capped to avoid quantum overfit)
    num_draws = len(historical_data)
    mc_sims, dynamic_epochs = get_dynamic_params(num_draws)

    # hard cap for quantum-heavy model
    dynamic_epochs = min(dynamic_epochs, 40)

    input_dim = augmented_f.shape[1]

    # Step 10: model arch (only regularization changed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(
                256, activation="relu",
                kernel_regularizer=keras.regularizers.l2(0.001)  # stronger L2
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),  # stronger dropout

            keras.layers.Dense(
                128, activation="relu",
                kernel_regularizer=keras.regularizers.l2(0.001)
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),

            keras.layers.Dense(NUM_TOTAL, activation="sigmoid"),
        ]
    )

    # Step 11: compile
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=weighted_binary_crossentropy(class_weights),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    # Step 12: callbacks (tighter early stop)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.8,
            patience=5,
            verbose=1,
            min_lr=1e-7,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,        # tighter
            min_delta=0.002,   # stricter improvement requirement
            restore_best_weights=True,
            verbose=1,
        ),
        EpochLogger(),
    ]

    # Step 13: train
    model.fit(
        augmented_f,
        augmented_l,
        epochs=dynamic_epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=0.15,
        callbacks=callbacks,
    )

    # Step 14: inference on clean features
    predictions = model.predict(features)
    final_prediction = np.mean(predictions, axis=0)

    # Step 15: push to pipeline
    pipeline.add_data("deep_learning_predictions", final_prediction)
    logging.info("Deep learning predictions (quantum anti-overfit) complete.")










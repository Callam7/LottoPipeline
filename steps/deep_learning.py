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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    """
    Predicts probability distribution over 50 numbers (40 main + 10 Powerball).
    Features are concatenated into shape 50 for input.
    """
    # Step 1: Retrieve features
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        logging.warning("No historical data available for deep learning prediction.")
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    monte_carlo = pipeline.get_data("monte_carlo")
    redundancy = pipeline.get_data("redundancy")
    markov = pipeline.get_data("markov_features")
    entropy = pipeline.get_data("entropy_features")
    fusion_norm = pipeline.get_data("bayesian_fusion_norm")
    clusters = pipeline.get_data("clusters")
    centroids = pipeline.get_data("centroids")

    required_features = [monte_carlo, redundancy, markov, entropy, fusion_norm, clusters, centroids]
    if any(v is None for v in required_features):
        logging.warning("Missing required features. Falling back to uniform distribution.")
        pipeline.add_data("deep_learning_predictions", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    # Step 2: Normalize each feature
    monte_carlo_norm = monte_carlo / max(np.max(monte_carlo), MIN_PROB)
    redundancy_norm = redundancy / max(np.max(redundancy), MIN_PROB)
    markov_norm = markov / max(np.max(markov), MIN_PROB)
    entropy_norm = entropy / max(np.max(entropy), MIN_PROB)
    fusion_norm = fusion_norm / max(np.max(fusion_norm), MIN_PROB)

    # Step 3: Concatenate features for model input
    features = np.column_stack((
        monte_carlo_norm,
        redundancy_norm,
        markov_norm,
        entropy_norm,
        fusion_norm,
        centroids[clusters]  # cluster centroid values
    ))

    # Step 4: Generate binary labels with label smoothing
    labels = []
    for draw in historical_data:
        binary_label = np.zeros(NUM_TOTAL)
        for num in draw.get("numbers", []):
            if 1 <= num <= NUM_TOTAL:
                binary_label[num - 1] = LABEL_SMOOTH
        labels.append(binary_label)
    labels = np.array(labels)

    # Step 5: Class weights
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    class_weights = neg_counts / (pos_counts + MIN_PROB)
    class_weights = np.clip(class_weights, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT)

    # Step 6: Weighted BCE loss
    def weighted_binary_crossentropy(weights):
        def loss_fn(y_true, y_pred):
            bce = keras.backend.binary_crossentropy(y_true, y_pred)
            weight_tensor = y_true * weights + (1.0 - y_true)
            return keras.backend.mean(weight_tensor * bce, axis=-1)
        return loss_fn

    # Step 7: Data augmentation
    augmented_features = []
    augmented_labels = []
    for _ in range(DATA_AUGMENTATION_ROUNDS):
        noise = np.random.normal(0, NOISE_STDDEV, features.shape)
        augmented_features.append(features + noise)
        augmented_labels.extend(labels)
    augmented_features = np.vstack(augmented_features)
    augmented_labels = np.vstack(augmented_labels)

    # Step 8: Dynamic training parameters
    num_draws = len(historical_data)
    mc_sims, dynamic_epochs = get_dynamic_params(num_draws)

    # Step 9: Model architecture
    model = keras.Sequential([
        keras.layers.Input(shape=(features.shape[1],)),
        keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(NUM_TOTAL, activation="sigmoid")
    ])

    # Step 10: Compile
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=weighted_binary_crossentropy(tf.constant(class_weights, dtype=tf.float32)),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.MeanAbsoluteError(name="mae")
        ]
    )

    # Step 11: Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.8, patience=5, verbose=1, min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, min_delta=0.001, restore_best_weights=True, verbose=1
        ),
        EpochLogger()
    ]

    # Step 12: Train
    model.fit(
        augmented_features,
        augmented_labels,
        epochs=dynamic_epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_split=0.15,
        callbacks=callbacks
    )

    # Step 13: Predict
    predictions = model.predict(features)
    final_prediction = np.mean(predictions, axis=0)

    # Step 14: Store result
    pipeline.add_data("deep_learning_predictions", final_prediction)
    logging.info("Deep learning predictions generated successfully.")








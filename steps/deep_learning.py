## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Deep Learning Prediction for Lottery Numbers
## Description:
## This file utilizes a deep learning model to predict probabilities for the 40 main lottery numbers
## as well as the powerball range between 1-10. The model leverages historical data,
## Monte Carlo results, clustering information, redundancy (recency/gap) data, Markov transition
## probabilities, entropy, and Bayesian fusion. The predictions are normalized to produce a probability
## distribution, which is used in ticket generation.

import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from pipeline import get_dynamic_params
from config.logs import EpochLogger

def deep_learning_prediction(pipeline):
    """
    Utilizes a deep learning model to predict the independent probability of each of the 40 lottery numbers
    being drawn, using sigmoid outputs and binary crossentropy loss to handle multi-label classification.

    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.

    Returns:
    - None: Adds "deep_learning_predictions" to the pipeline.
    """

    # Step 1: Retrieve necessary data
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        print("No historical data available for deep learning prediction.")
        pipeline.add_data("deep_learning_predictions", np.ones(40) / 40)
        return

    monte_carlo = pipeline.get_data("monte_carlo")
    clusters = pipeline.get_data("clusters")
    centroids = pipeline.get_data("centroids")
    redundancy = pipeline.get_data("redundancy")      # Recency/gap data
    markov = pipeline.get_data("markov_features")     # Markov transition probabilities
    entropy = pipeline.get_data("entropy_features")   # Entropy features
    fusion_norm = pipeline.get_data("bayesian_fusion_norm")  # Bayesian fusion (normalized for DL)

    if any(v is None for v in [monte_carlo, clusters, centroids,
                               redundancy, markov, entropy, fusion_norm]):
        print("Necessary data missing for deep learning prediction.")
        pipeline.add_data("deep_learning_predictions", np.ones(40) / 40)
        return

    # Step 2: Normalize inputs
    mc_max = max(monte_carlo.max(), 1e-12)
    red_max = max(redundancy.max(), 1e-12)
    mk_max = max(markov.max(), 1e-12)
    ent_max = max(entropy.max(), 1e-12)
    fu_max = max(fusion_norm.max(), 1e-12)

    monte_carlo_norm = monte_carlo / mc_max
    redundancy_norm = redundancy / red_max
    markov_norm = markov / mk_max
    entropy_norm = entropy / ent_max
    fusion_norm = fusion_norm / fu_max   # re-safeguard to [0,1]

    # Step 3: Assemble feature set (decay removed)
    features = np.column_stack((
        monte_carlo_norm,
        redundancy_norm,
        markov_norm,
        entropy_norm,
        fusion_norm,            # Bayesian Fusion
        centroids[clusters]
    ))

    # Step 4: Create binary labels from historical draw data (with label smoothing)
    labels = []
    for draw in historical_data:
        binary_label = np.zeros(40)
        for num in draw["numbers"]:
            if 1 <= num <= 40:
                binary_label[num - 1] = 0.95  # label smoothing
        labels.append(binary_label)
    labels = np.array(labels)

    # Step 5: Compute class weights
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    class_weights = neg_counts / (pos_counts + 1e-6)
    class_weights = np.clip(class_weights, 1.0, 10.0)

    # Step 6: Weighted binary crossentropy
    def weighted_binary_crossentropy(weights):
        def loss_fn(y_true, y_pred):
            bce = keras.backend.binary_crossentropy(y_true, y_pred)
            weight_tensor = y_true * weights + (1.0 - y_true)
            return keras.backend.mean(weight_tensor * bce, axis=-1)
        return loss_fn

    # Step 7: Data augmentation
    augmented_features = []
    augmented_labels = []
    for _ in range(100):
        noise = np.random.normal(0, 0.05, features.shape)
        augmented_features.append(features + noise)
        augmented_labels.extend(labels)

    augmented_features = np.vstack(augmented_features)
    augmented_labels = np.vstack(augmented_labels)

    # Step 8: Dynamic training parameters
    num_draws = len(historical_data)
    mc_sims, dynamic_epochs = get_dynamic_params(num_draws)

    # Step 9: Define model architecture
    model = keras.Sequential([
        keras.layers.Input(shape=(features.shape[1],)),
        keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0005)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(40, activation="sigmoid")
    ])

    # Step 10: Compile model
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
            monitor="val_loss",
            factor=0.8,
            patience=5,
            verbose=1,
            min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        ),
        EpochLogger()
    ]

    # Step 12: Train model
    history = model.fit(
        augmented_features,
        augmented_labels,
        epochs=dynamic_epochs,
        batch_size=32,
        verbose=1,
        validation_split=0.15,
        callbacks=callbacks
    )

    # Step 13: Predict and average
    predictions = model.predict(features)
    final_prediction = np.mean(predictions, axis=0)

    # Step 14: Store results
    pipeline.add_data("deep_learning_predictions", final_prediction)







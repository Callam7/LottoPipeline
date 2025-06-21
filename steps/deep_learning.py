## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: Deep Learning Prediction for Lottery Numbers
## Description:
## This file utilizes a deep learning model to predict probabilities for the 40 main lottery numbers.
## The model leverages historical data, decay factors, Monte Carlo results, and clustering information.
## The predictions are normalized to produce a probability distribution, which is used in ticket generation.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pipeline import get_dynamic_params

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

    decay_factors = pipeline.get_data("decay_factors")
    monte_carlo = pipeline.get_data("monte_carlo")
    clusters = pipeline.get_data("clusters")
    centroids = pipeline.get_data("centroids")

    if any(v is None for v in [decay_factors, monte_carlo, clusters, centroids]):
        print("Necessary data missing for deep learning prediction.")
        pipeline.add_data("deep_learning_predictions", np.ones(40) / 40)
        return

    # Step 2: Normalize decay factors and Monte Carlo results
    df_max = max(decay_factors["numbers"].max(), 1e-12)
    mc_max = max(monte_carlo.max(), 1e-12)
    decay_factors_norm = decay_factors["numbers"] / df_max
    monte_carlo_norm = monte_carlo / mc_max

    # Step 3: Assemble feature set by combining decay, Monte Carlo, and cluster centroids
    features = np.column_stack((decay_factors_norm, monte_carlo_norm, centroids[clusters]))

    # Step 4: Create binary labels from historical draw data (with label smoothing)
    labels = []
    for draw in historical_data:
        binary_label = np.zeros(40)
        for num in draw["numbers"]:
            if 1 <= num <= 40:
                binary_label[num - 1] = 0.95  # label smoothing
        labels.append(binary_label)

    labels = np.array(labels)

    # Step 5: Compute class weights to address imbalance
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    class_weights = neg_counts / (pos_counts + 1e-6)
    class_weights = np.clip(class_weights, 1.0, 10.0)

    # Step 6: Define a custom weighted binary crossentropy loss function
    def weighted_binary_crossentropy(weights):
        def loss_fn(y_true, y_pred):
            bce = keras.backend.binary_crossentropy(y_true, y_pred)
            weight_tensor = y_true * weights + (1.0 - y_true)
            return keras.backend.mean(weight_tensor * bce, axis=-1)
        return loss_fn

    # Step 7: Data augmentation by injecting Gaussian noise into features
    augmentation_rounds = 100
    augmented_features = []
    augmented_labels = []
    for _ in range(augmentation_rounds):
        noise = np.random.normal(0, 0.05, features.shape)
        augmented_features.append(features + noise)
        augmented_labels.extend(labels)

    augmented_features = np.vstack(augmented_features)
    augmented_labels = np.vstack(augmented_labels)

    # Step 8: Retrieve dynamic training parameters
    num_draws = len(historical_data)
    mc_sims, dynamic_epochs = get_dynamic_params(num_draws)

    # Step 9: Define model architecture with dropout and batch normalization
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

    # Step 10: Compile model using weighted binary crossentropy
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=weighted_binary_crossentropy(tf.constant(class_weights, dtype=tf.float32)),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.MeanAbsoluteError(name="mae")
        ]
    )

    # Step 11: Use callbacks for learning rate adjustment and early stopping
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
        )
    ]

    # Step 12: Train the model on the augmented data with validation split
    history = model.fit(
        augmented_features,
        augmented_labels,
        epochs=dynamic_epochs,
        batch_size=32,
        verbose=1,
        validation_split=0.15,
        callbacks=callbacks
    )

    # Step 13: Predict probabilities for each number independently and average predictions
    predictions = model.predict(features)
    final_prediction = np.mean(predictions, axis=0)

    # Step 14: Store prediction result in the pipeline
    pipeline.add_data("deep_learning_predictions", final_prediction)


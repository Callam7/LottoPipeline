## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: Deep Learning Prediction for Lottery Numbers
## Description:
## This file utilizes a deep learning model to predict probabilities for the 40 main lottery numbers.
## The model leverages historical data, decay factors, Monte Carlo results, and clustering information.
## The predictions are normalized to produce a probability distribution, which is used in ticket generation.

import numpy as np  # For numerical operations and data manipulation
import tensorflow as tf  # TensorFlow framework for deep learning
from tensorflow import keras  # Keras API for building neural networks
from pipeline import get_dynamic_params  # For dynamic parameter calculations based on data size

def deep_learning_prediction(pipeline):
    """
    Utilizes a deep learning model to predict probabilities for each of the 40 main lottery numbers
    based on historical draw data. Processes the data, trains the model, and adds the prediction
    probabilities to the pipeline for ticket generation.

    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.

    Returns:
    - None: Adds "deep_learning_predictions" to the pipeline.
    """

    # Step 1: Retrieve historical draw data from the pipeline
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        print("No historical data available for deep learning prediction.")
        # Assign uniform probabilities if no historical data is available
        pipeline.add_data("deep_learning_predictions", np.ones(40) / 40)
        return

    # Step 2: Retrieve necessary supporting data from the pipeline
    decay_factors = pipeline.get_data("decay_factors")
    monte_carlo = pipeline.get_data("monte_carlo")
    clusters = pipeline.get_data("clusters")
    centroids = pipeline.get_data("centroids")

    if any(v is None for v in [decay_factors, monte_carlo, clusters, centroids]):
        print("Necessary data missing for deep learning prediction.")
        # Assign uniform probabilities if required data is missing
        pipeline.add_data("deep_learning_predictions", np.ones(40) / 40)
        return

    # Step 3: Normalize decay factors and Monte Carlo results
    df_max = max(decay_factors["numbers"].max(), 1e-12)  # Prevent division by zero
    mc_max = max(monte_carlo.max(), 1e-12)
    decay_factors_norm = decay_factors["numbers"] / df_max
    monte_carlo_norm = monte_carlo / mc_max

    # Step 4: Combine features for model input
    features = np.column_stack((decay_factors_norm, monte_carlo_norm, centroids[clusters]))

    # Step 5: Construct label distribution based on historical data
    labels = np.zeros(40)  # Initialize label distribution for all 40 numbers
    total_weight = 0  # Track total weight for normalization
    for i, draw in enumerate(historical_data):
        weight = 1 / (1 + i)  # Assign exponentially decreasing weight to older draws
        total_weight += weight
        for num in draw["numbers"]:
            if 1 <= num <= 40:
                labels[num - 1] += weight

    if total_weight > 0:
        labels /= total_weight  # Normalize label distribution
    else:
        labels = np.ones(40) / 40  # Assign uniform distribution if no valid data is available

    # Step 6: Data augmentation for training robustness
    augmentation_rounds = 100  # Number of augmented feature sets (was 500)
    augmented_features = []
    augmented_labels = []
    for _ in range(augmentation_rounds):
        noise = np.random.normal(0, 0.05, features.shape)  # Add Gaussian noise to features
        augmented_features.append(features + noise)
        augmented_labels.extend([labels] * features.shape[0])  # Replicate labels for each augmented set

    augmented_features = np.vstack(augmented_features)  # Stack augmented features into a single array
    augmented_labels = np.vstack(augmented_labels)  # Stack augmented labels

    # Step 7: Get dynamic parameters based on historical data size
    num_draws = len(historical_data)
    mc_sims, dynamic_epochs = get_dynamic_params(num_draws)

    # Step 8: Build the deep learning model
    model = keras.Sequential([
        keras.layers.Input(shape=(features.shape[1],)),  # Input layer matching feature size
        keras.layers.Dense(
            128, activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.0005)  # L2 regularization for weight decay
        ),
        keras.layers.BatchNormalization(),  # Normalize activations to improve convergence
        keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
        keras.layers.Dense(
            64, activation="relu",
            kernel_regularizer=keras.regularizers.l2(0.0005)
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(40, activation="softmax"),  # Output layer with probabilities for 40 numbers
    ])

    # Step 9: Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),  # Track prediction accuracy
            keras.metrics.KLDivergence(name="kl_div")  # Measure divergence from label distribution
        ]
    )

    # Step 10: Define callbacks for early stopping and learning rate reduction
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12, verbose=1
        )
    ]

    # Step 11: Train the model with the augmented dataset
    history = model.fit(
        augmented_features,
        augmented_labels,
        epochs=dynamic_epochs,  # Dynamically adjusted based on data size
        batch_size=16,  # Smaller batch size for less noisy training
        verbose=1,
        validation_split=0.2,  # 20% of data for validation
        callbacks=callbacks
    )

    # Step 12: Generate predictions from the trained model
    predictions = model.predict(features)
    final_prediction = np.mean(predictions, axis=0)  # Average predictions across samples

    # Step 13: Store the final prediction in the pipeline
    pipeline.add_data("deep_learning_predictions", final_prediction)

# End of file

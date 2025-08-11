## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: Perform K-Means Clustering on Number Frequencies
## Description:
## This file applies K-Means clustering to identify patterns or groupings in the frequency data 
## of both main lottery numbers (1–40) and Powerball numbers (1–10). Clustering results, including 
## labels for each number and cluster centroids, are stored in the pipeline for use in 
## subsequent predictive modeling steps.

import numpy as np  # For numerical operations and data formatting
from sklearn.cluster import KMeans  # For performing K-Means clustering
from sklearn.preprocessing import MinMaxScaler  # For normalizing data before clustering
import logging  # For logging warnings and informational messages

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def kmeans_clustering_and_correlation(pipeline, n_clusters=5):
    """
    Performs K-Means clustering on the main number frequency data to identify groupings.
    Adds the cluster labels and centroids to the pipeline for use in downstream steps.

    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.
    - n_clusters (int): The number of clusters to form for main numbers. Default is 5.

    Returns:
    - None: Adds "clusters" and "centroids" to the pipeline.
    """

    # Step 1: Retrieve normalized frequency data for main numbers (1–40)
    frequency_data = pipeline.get_data("number_frequency")
    if frequency_data is None or len(frequency_data) != 40:
        logging.error("Frequency data is missing or not of expected length 40 for main numbers.")
        return

    # Step 2: Reshape frequency data for K-Means input
    frequency_data = np.array(frequency_data).reshape(-1, 1)

    # Step 3: Normalize values to range [0, 1]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(frequency_data)

    # Step 4: Adjust cluster count if variance is too low
    if np.std(data_scaled) < 0.01:
        logging.warning("Low variance in main number frequency data; reducing clusters to 2.")
        n_clusters = min(n_clusters, 2)

    try:
        # Step 5: Fit KMeans model on scaled main number data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(data_scaled)

        # Step 6: Extract clustering results
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_.flatten()

        # Step 7: Store results in pipeline
        pipeline.add_data("clusters", labels)
        pipeline.add_data("centroids", centroids)
        pipeline.add_data("number_to_cluster", labels)

        logging.info("Main number clustering completed successfully.")

    except Exception as e:
        logging.error(f"Error during main number clustering: {e}")


def cluster_powerball_frequency(pipeline, n_clusters=3):
    """
    Performs K-Means clustering on Powerball frequency data to identify patterns.
    Stores cluster labels and centroids for Powerball numbers separately.

    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.
    - n_clusters (int): The number of clusters to form for Powerball numbers. Default is 3.

    Returns:
    - None: Adds "powerball_clusters" and "powerball_centroids" to the pipeline.
    """

    # Step 1: Retrieve normalized frequency data for Powerball (1–10)
    powerball_data = pipeline.get_data("powerball_frequency")
    if powerball_data is None or len(powerball_data) != 10:
        logging.error("Powerball frequency data is missing or not of expected length 10.")
        return

    # Step 2: Reshape Powerball data for clustering
    powerball_data = np.array(powerball_data).reshape(-1, 1)

    # Step 3: Normalize values to [0, 1]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(powerball_data)

    # Step 4: Adjust cluster count if insufficient variation
    if np.std(data_scaled) < 0.01:
        logging.warning("Low variance in Powerball frequency data; reducing clusters to 2.")
        n_clusters = min(n_clusters, 2)

    try:
        # Step 5: Fit KMeans model to Powerball data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(data_scaled)

        # Step 6: Extract clustering results
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_.flatten()

        # Step 7: Store Powerball results in pipeline
        pipeline.add_data("powerball_clusters", labels)
        pipeline.add_data("powerball_centroids", centroids)
        logging.info("Powerball clustering completed successfully.")

    except Exception as e:
        logging.error(f"Error during Powerball clustering: {e}")


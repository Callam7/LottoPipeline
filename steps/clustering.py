## Modified By: Callam  
## Project: Lotto Generator  
## Purpose of File: Perform K-Means Clustering on Number Frequencies with Bayesian Fusion  
## Description:  
## This file applies K-Means clustering to identify patterns or groupings in the frequency data   
## of both main lottery numbers (1–40) and Powerball numbers (1–10), incorporating Bayesian fusion  
## for both datasets. Clustering results, including labels and centroids, are stored in the pipeline  
## for use in subsequent predictive modeling steps.  

import numpy as np  
from sklearn.cluster import KMeans  
from sklearn.preprocessing import MinMaxScaler  
import logging  

# Configure logging  
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

def kmeans_clustering_and_correlation(pipeline, n_clusters=5):
    """
    Performs K-Means clustering on main number frequency data combined with Bayesian fusion.
    Fusion is mandatory. Maintains 40-number array shape.
    """
    # Step 1: Retrieve frequency and fusion data
    frequency_data = pipeline.get_data("number_frequency")
    fusion_data = pipeline.get_data("bayesian_fusion")  # Must be 40-length normalized array

    if frequency_data is None or len(frequency_data) != 40:
        logging.error("Frequency data missing or not length 40 for main numbers.")
        return
    if fusion_data is None or len(fusion_data) != 40:
        logging.error("Bayesian fusion data missing or not length 40.")
        return

    # Step 2: Combine and normalize
    fused_array = np.column_stack((frequency_data, fusion_data))
    fused_array = fused_array / fused_array.max(axis=0)

    # Step 3: MinMax scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(fused_array)

    # Step 4: Low variance adjustment
    if np.std(data_scaled) < 0.01:
        logging.warning("Low variance in main number data; reducing clusters to 2.")
        n_clusters = min(n_clusters, 2)

    try:
        # Step 5: Fit KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(data_scaled)

        # Step 6: Extract results
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_.flatten()

        # Step 7: Store results
        pipeline.add_data("clusters", labels)
        pipeline.add_data("centroids", centroids)
        pipeline.add_data("number_to_cluster", labels)

        logging.info("Main number clustering with fusion completed successfully.")

    except Exception as e:
        logging.error(f"Error during main number clustering: {e}")


def cluster_powerball_frequency(pipeline, n_clusters=3):
    """
    Performs K-Means clustering on Powerball frequency data combined with Bayesian fusion.
    Fusion is mandatory. Maintains 10-number array shape.
    """
    powerball_data = pipeline.get_data("powerball_frequency")
    fusion_data = pipeline.get_data("bayesian_fusion_powerball")  # Must be 10-length normalized array

    if powerball_data is None or len(powerball_data) != 10:
        logging.error("Powerball frequency data missing or not length 10.")
        return
    if fusion_data is None or len(fusion_data) != 10:
        logging.error("Bayesian fusion Powerball data missing or not length 10.")
        return

    # Combine and normalize
    fused_array = np.column_stack((powerball_data, fusion_data))
    fused_array = fused_array / fused_array.max(axis=0)

    # MinMax scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(fused_array)

    # Low variance adjustment
    if np.std(data_scaled) < 0.01:
        logging.warning("Low variance in Powerball data; reducing clusters to 2.")
        n_clusters = min(n_clusters, 2)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(data_scaled)

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_.flatten()

        pipeline.add_data("powerball_clusters", labels)
        pipeline.add_data("powerball_centroids", centroids)
        logging.info("Powerball clustering with fusion completed successfully.")

    except Exception as e:
        logging.error(f"Error during Powerball clustering: {e}")



## Modified By: Callam  
## Project: Lotto Generator  
## Purpose of File: Perform K-Means Clustering on Bayesian Fusion Probabilities  
## Description:  
<<<<<<< HEAD
<<<<<<< HEAD
## Clusters both main (1–40) and Powerball (1–10) Bayesian fusion probabilities.  
## Produces unified cluster labels and centroid influence vectors of shape (50,).  
## Each section is scaled within its domain, then concatenated to preserve probabilistic structure.
=======
<<<<<<< HEAD
## Clusters both main (1–40) and Powerball (1–10) Bayesian fusion probabilities.  
## Produces unified cluster labels and centroid influence vectors of shape (50,).  
## Each section is scaled within its domain, then concatenated to preserve probabilistic structure.
=======
=======

>>>>>>> ae06da4158e26ede5593b4f110fa860ed1acc769
## This file applies K-Means clustering to identify patterns or groupings in the frequency data   
## of both main lottery numbers (1–40) and Powerball numbers (1–10), incorporating Bayesian fusion  
## for both datasets. Clustering results, including labels and centroids, are stored in the pipeline  
## for use in subsequent predictive modeling steps.  
<<<<<<< HEAD
>>>>>>> 2c8488b95f520be890fa7fd7753dabbafbfe5127
>>>>>>> c6b77838891a23ba80277aa624b8cf48fd3b994f
=======

>>>>>>> ae06da4158e26ede5593b4f110fa860ed1acc769

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import logging

# Constants
NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def kmeans_clustering_and_correlation(pipeline, n_clusters_main=5, n_clusters_powerball=3):
    """
    Perform K-Means clustering on Bayesian fusion data (main + Powerball).

    Adds to pipeline:
        - clusters (np.ndarray, shape=(50,), dtype=int)
        - centroids (np.ndarray, shape=(50,), dtype=float)
        - number_to_cluster (np.ndarray, shape=(50,), dtype=int)
    """

    fusion = pipeline.get_data("bayesian_fusion")

    # --- Validate & normalize input ---
    if fusion is None or len(fusion) != NUM_TOTAL:
        logging.error(f"Bayesian fusion data missing or invalid length (expected {NUM_TOTAL}). Using uniform fallback.")
        fusion = np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL
    else:
        fusion = np.array(fusion, dtype=float)

    # Split into main and Powerball sections
    fusion_main = fusion[:NUM_MAIN]
    fusion_power = fusion[NUM_MAIN:]

    # Normalize each section individually
    fusion_main = fusion_main / (fusion_main.sum() or 1.0)
    fusion_power = fusion_power / (fusion_power.sum() or 1.0)

    # =========================================================
    # MAIN NUMBER CLUSTERING
    # =========================================================
    scaler_main = MinMaxScaler()
    data_main_scaled = scaler_main.fit_transform(fusion_main.reshape(-1, 1))

    if np.std(data_main_scaled) < 0.01:
        logging.warning("Low variance in main fusion; reducing clusters to 2.")
        n_clusters_main = min(n_clusters_main, 2)

    try:
        kmeans_main = KMeans(n_clusters=n_clusters_main, random_state=42, n_init=10)
        kmeans_main.fit(data_main_scaled)
        labels_main = kmeans_main.labels_
        centroids_main = kmeans_main.cluster_centers_[labels_main].flatten()

        # Normalize centroids to [0,1]
        centroids_main = (centroids_main - np.min(centroids_main)) / (np.ptp(centroids_main) + 1e-9)
    except Exception as e:
        logging.error(f"Main number clustering failed: {e}")
        labels_main = np.zeros(NUM_MAIN, dtype=int)
        centroids_main = np.ones(NUM_MAIN, dtype=float) / NUM_MAIN

    # =========================================================
    # POWERBALL CLUSTERING
    # =========================================================
    scaler_power = MinMaxScaler()
    data_power_scaled = scaler_power.fit_transform(fusion_power.reshape(-1, 1))

    if np.std(data_power_scaled) < 0.01:
        logging.warning("Low variance in Powerball fusion; reducing clusters to 2.")
        n_clusters_powerball = min(n_clusters_powerball, 2)

    try:
        kmeans_power = KMeans(n_clusters=n_clusters_powerball, random_state=42, n_init=10)
        kmeans_power.fit(data_power_scaled)
        labels_power = kmeans_power.labels_
        centroids_power = kmeans_power.cluster_centers_[labels_power].flatten()

        # Normalize centroids to [0,1] within Powerball domain
        centroids_power = (centroids_power - np.min(centroids_power)) / (np.ptp(centroids_power) + 1e-9)
    except Exception as e:
        logging.error(f"Powerball clustering failed: {e}")
        labels_power = np.zeros(NUM_POWERBALL, dtype=int)
        centroids_power = np.ones(NUM_POWERBALL, dtype=float) / NUM_POWERBALL

    # =========================================================
    # COMBINE (MAIN + POWERBALL)
    # =========================================================
    combined_labels = np.concatenate((labels_main, labels_power)).astype(int)
    combined_centroids = np.concatenate((centroids_main, centroids_power)).astype(float)

    # --- Ensure valid probabilistic scaling ---
    combined_centroids = np.clip(combined_centroids, 0.0, 1.0)
    combined_centroids /= combined_centroids.sum() or 1.0

    # =========================================================
    # STORE UNIFIED OUTPUTS
    # =========================================================
    pipeline.add_data("clusters", combined_labels)
    pipeline.add_data("centroids", combined_centroids)
    pipeline.add_data("number_to_cluster", combined_labels)

    logging.info("K-Means clustering completed.")


## Modified By: Callam  
## Project: Lotto Generator  
## Purpose of File: Perform K-Means Clustering on Bayesian Fusion Probabilities  

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import logging

NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def kmeans_clustering_and_correlation(pipeline, n_clusters_main=5, n_clusters_powerball=3):

    fusion = pipeline.get_data("bayesian_fusion")

    if fusion is None or len(fusion) != NUM_TOTAL:
        logging.error("Fusion missing/invalid — using uniform.")
        fusion = np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL
    else:
        fusion = np.asarray(fusion, dtype=float)

    fusion_main = fusion[:NUM_MAIN]
    fusion_power = fusion[NUM_MAIN:]

    # Normalize internally but do NOT renormalize centroids later
    fusion_main /= fusion_main.sum() or 1.0
    fusion_power /= fusion_power.sum() or 1.0

    # =========================
    # MAIN K-MEANS
    # =========================
    scaler_main = MinMaxScaler()
    data_main = scaler_main.fit_transform(fusion_main.reshape(-1, 1))

    if np.std(data_main) < 0.01:
        n_clusters_main = min(n_clusters_main, 2)

    try:
        kmeans_main = KMeans(n_clusters=n_clusters_main, random_state=42, n_init=10)
        labels_main = kmeans_main.fit_predict(data_main)
        centroids_main = kmeans_main.cluster_centers_[labels_main].flatten()

        # keep centroids in the scaled KMeans geometry
        centroids_main = np.clip(centroids_main, 0.0, 1.0)

    except Exception as e:
        logging.error(f"Main clustering failed: {e}")
        labels_main = np.zeros(NUM_MAIN, int)
        centroids_main = np.ones(NUM_MAIN) * (1.0 / n_clusters_main)

    # =========================
    # POWERBALL K-MEANS
    # =========================
    scaler_power = MinMaxScaler()
    data_power = scaler_power.fit_transform(fusion_power.reshape(-1, 1))

    if np.std(data_power) < 0.01:
        n_clusters_powerball = min(n_clusters_powerball, 2)

    try:
        kmeans_power = KMeans(n_clusters=n_clusters_powerball, random_state=42, n_init=10)
        labels_power = kmeans_power.fit_predict(data_power)
        centroids_power = kmeans_power.cluster_centers_[labels_power].flatten()

        centroids_power = np.clip(centroids_power, 0.0, 1.0)

    except Exception as e:
        logging.error(f"Powerball clustering failed: {e}")
        labels_power = np.zeros(NUM_POWERBALL, int)
        centroids_power = np.ones(NUM_POWERBALL) * (1.0 / n_clusters_powerball)

    # =========================
    # COMBINE
    # =========================
    combined_labels = np.concatenate([labels_main, labels_power]).astype(int)
    combined_centroids = np.concatenate([centroids_main, centroids_power]).astype(float)

    # DO NOT normalize centroid sum — preserve geometry
    combined_centroids = np.clip(combined_centroids, 0.0, 1.0)

    pipeline.add_data("clusters", combined_labels)
    pipeline.add_data("centroids", combined_centroids)
    pipeline.add_data("number_to_cluster", combined_labels)

    logging.info("K-Means clustering completed correctly.")


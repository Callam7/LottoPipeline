## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Perform K-Means Clustering on Bayesian Fusion Probabilities

import numpy as np                     # Numerical array operations
from sklearn.cluster import KMeans     # K-Means clustering algorithm
from sklearn.preprocessing import MinMaxScaler  # Feature scaling utility
import logging                         # Logging system for status/errors

NUM_MAIN = 40                          # Number of main lotto numbers
NUM_POWERBALL = 10                     # Number of Powerball numbers
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL   # Total length of probability vector (50)

# Configure logging output format and level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def kmeans_clustering_and_correlation(pipeline, n_clusters_main=5, n_clusters_powerball=3):
    fusion = pipeline.get_data("bayesian_fusion")  # Retrieve fused probability vector from pipeline

    # Validate fusion vector
    if fusion is None or len(fusion) != NUM_TOTAL:
        logging.error("Fusion missing/invalid — using uniform.")  # Warn if missing
        fusion = np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL      # Fallback to uniform probabilities
    else:
        fusion = np.asarray(fusion, dtype=float)                  # Ensure float NumPy array

    fusion_main = fusion[:NUM_MAIN].copy()     # First 40 entries = main numbers
    fusion_power = fusion[NUM_MAIN:].copy()    # Last 10 entries = Powerball numbers

    # Normalize internally so clustering works on comparable scale
    fusion_main /= fusion_main.sum() or 1.0
    fusion_power /= fusion_power.sum() or 1.0

    # =========================
    # MAIN K-MEANS
    # =========================
    scaler_main = MinMaxScaler()                                  # Scale values to 0–1 range
    data_main = scaler_main.fit_transform(fusion_main.reshape(-1, 1))  # Reshape for sklearn (40,1)

    # If values are almost identical, reduce cluster count to avoid meaningless splits
    if float(np.std(data_main)) < 0.01:
        n_clusters_main = min(int(n_clusters_main), 2)

    try:
        # Perform K-Means clustering on scaled probabilities
        kmeans_main = KMeans(n_clusters=int(n_clusters_main), random_state=42, n_init=10)
        labels_main = kmeans_main.fit_predict(data_main).astype(int)          # Cluster ID for each number (40,)
        centers_main = np.asarray(kmeans_main.cluster_centers_, dtype=float)  # Cluster center values (K_main,1)

        # Map each number to its cluster center value
        centroids_main = centers_main[labels_main].reshape(-1)                # Per-number centroid strength
        centroids_main = np.clip(centroids_main, 0.0, 1.0)                    # Keep safe numeric bounds

    except Exception as e:
        # If clustering fails, fallback to neutral grouping
        logging.error(f"Main clustering failed: {e}")
        labels_main = np.zeros(NUM_MAIN, dtype=int)
        centers_main = np.ones((1, 1), dtype=float)
        centroids_main = np.ones(NUM_MAIN, dtype=float) * 0.5

    # =========================
    # POWERBALL K-MEANS
    # =========================
    scaler_power = MinMaxScaler()                                 # Separate scaler for PB domain
    data_power = scaler_power.fit_transform(fusion_power.reshape(-1, 1))  # Shape (10,1)

    if float(np.std(data_power)) < 0.01:
        n_clusters_powerball = min(int(n_clusters_powerball), 2)

    try:
        kmeans_power = KMeans(n_clusters=int(n_clusters_powerball), random_state=42, n_init=10)
        labels_power = kmeans_power.fit_predict(data_power).astype(int)          # Cluster ID per PB number
        centers_power = np.asarray(kmeans_power.cluster_centers_, dtype=float)   # Cluster center strengths

        centroids_power = centers_power[labels_power].reshape(-1)                # Per-number PB centroid values
        centroids_power = np.clip(centroids_power, 0.0, 1.0)

    except Exception as e:
        logging.error(f"Powerball clustering failed: {e}")
        labels_power = np.zeros(NUM_POWERBALL, dtype=int)
        centers_power = np.ones((1, 1), dtype=float)
        centroids_power = np.ones(NUM_POWERBALL, dtype=float) * 0.5

    # =========================
    # COMBINE
    # =========================

    # Offset PB cluster labels so they don't overlap with main cluster indices
    labels_power_offset = labels_power + int(centers_main.shape[0])

    # Combined cluster labels for all 50 numbers
    combined_labels = np.concatenate([labels_main, labels_power_offset]).astype(int)

    # Combined centroid strengths per number (this is what other modules use)
    combined_centroids = np.concatenate([centroids_main, centroids_power]).astype(float)
    combined_centroids = np.clip(combined_centroids, 0.0, 1.0)

    # Stack true cluster centers (optional reference data)
    combined_centers = np.vstack([centers_main, centers_power]).astype(float)

    # Store outputs in pipeline
    pipeline.add_data("clusters", combined_labels)             # Cluster ID per number
    pipeline.add_data("centroids", combined_centroids)         # Cluster strength per number
    pipeline.add_data("centroid_centers", combined_centers)    # Raw cluster centers
    pipeline.add_data("number_to_cluster", combined_labels)    # Alias for compatibility

    logging.info("K-Means clustering completed.")



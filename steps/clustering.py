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

    fusion_main = fusion[:NUM_MAIN].copy()
    fusion_power = fusion[NUM_MAIN:].copy()

    # Normalize internally
    fusion_main /= fusion_main.sum() or 1.0
    fusion_power /= fusion_power.sum() or 1.0

    # =========================
    # MAIN K-MEANS
    # =========================
    scaler_main = MinMaxScaler()
    data_main = scaler_main.fit_transform(fusion_main.reshape(-1, 1))

    if float(np.std(data_main)) < 0.01:
        n_clusters_main = min(int(n_clusters_main), 2)

    try:
        kmeans_main = KMeans(n_clusters=int(n_clusters_main), random_state=42, n_init=10)
        labels_main = kmeans_main.fit_predict(data_main).astype(int)          # (40,)
        centers_main = np.asarray(kmeans_main.cluster_centers_, dtype=float)  # (K_main, 1)

        # Per-number centroid value lookup (shape 40,) — REQUIRED by your other steps
        centroids_main = centers_main[labels_main].reshape(-1)                # (40,)
        centroids_main = np.clip(centroids_main, 0.0, 1.0)

    except Exception as e:
        logging.error(f"Main clustering failed: {e}")
        labels_main = np.zeros(NUM_MAIN, dtype=int)
        centers_main = np.ones((1, 1), dtype=float)
        centroids_main = np.ones(NUM_MAIN, dtype=float) * 0.5

    # =========================
    # POWERBALL K-MEANS
    # =========================
    scaler_power = MinMaxScaler()
    data_power = scaler_power.fit_transform(fusion_power.reshape(-1, 1))

    if float(np.std(data_power)) < 0.01:
        n_clusters_powerball = min(int(n_clusters_powerball), 2)

    try:
        kmeans_power = KMeans(n_clusters=int(n_clusters_powerball), random_state=42, n_init=10)
        labels_power = kmeans_power.fit_predict(data_power).astype(int)          # (10,)
        centers_power = np.asarray(kmeans_power.cluster_centers_, dtype=float)   # (K_power, 1)

        # Per-number centroid value lookup (shape 10,) — REQUIRED by your other steps
        centroids_power = centers_power[labels_power].reshape(-1)                # (10,)
        centroids_power = np.clip(centroids_power, 0.0, 1.0)

    except Exception as e:
        logging.error(f"Powerball clustering failed: {e}")
        labels_power = np.zeros(NUM_POWERBALL, dtype=int)
        centers_power = np.ones((1, 1), dtype=float)
        centroids_power = np.ones(NUM_POWERBALL, dtype=float) * 0.5

    # =========================
    # COMBINE
    # =========================

    # IMPORTANT: offset powerball labels so they index into stacked centers correctly
    labels_power_offset = labels_power + int(centers_main.shape[0])

    combined_labels = np.concatenate([labels_main, labels_power_offset]).astype(int)  # (50,)

    # What your existing pipeline expects:
    combined_centroids = np.concatenate([centroids_main, centroids_power]).astype(float)  # (50,)
    combined_centroids = np.clip(combined_centroids, 0.0, 1.0)

    # Optional: true centroid prototypes, for consumers that want centers
    combined_centers = np.vstack([centers_main, centers_power]).astype(float)  # (K_total, 1)

    pipeline.add_data("clusters", combined_labels)
    pipeline.add_data("centroids", combined_centroids)          # <-- keep as length-50 vector
    pipeline.add_data("centroid_centers", combined_centers)     # <-- extra (optional)
    pipeline.add_data("number_to_cluster", combined_labels)

    logging.info("K-Means clustering completed.")



## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Compute Shannon Entropy Features for Lotto Prediction (Full Shape 50)
## Description:
## Calculates Shannon entropy across both main (1–40) and Powerball (1–10)
## using unified clustering, fusion, and redundancy weighting.
## Ensures unified (50,) output covering all numbers.

import numpy as np
import logging

# Constants
NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_shannon_entropy(prob_dist: np.ndarray) -> float:
    """Compute Shannon entropy H = -Σ p(x) log2 p(x)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(prob_dist > 0, np.log2(prob_dist), 0.0)
    return -float(np.sum(prob_dist * logp))


def shannon_entropy_features(pipeline):
    """Generate Shannon entropy features for both main + Powerball numbers (shape 50)."""
    # --- Retrieve unified inputs ---
    clusters = pipeline.get_data("number_to_cluster")
    centroids = pipeline.get_data("centroids")
    redundancy = pipeline.get_data("redundancy")
    fusion = pipeline.get_data("bayesian_fusion")
    markov = pipeline.get_data("markov_features")
    historical_data = pipeline.get_data("historical_data")

    # --- Validate inputs ---
    if any(v is None for v in [clusters, centroids, redundancy, fusion, markov, historical_data]):
        logging.warning("Missing required data for entropy computation — using uniform fallback.")
        pipeline.add_data("entropy_features", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    if len(clusters) != NUM_TOTAL or len(centroids) != NUM_TOTAL:
        logging.warning(f"Invalid cluster or centroid length (expected {NUM_TOTAL}). Using fallback.")
        clusters = np.zeros(NUM_TOTAL, dtype=int)
        centroids = np.ones(NUM_TOTAL) / NUM_TOTAL

    if len(redundancy) != NUM_TOTAL:
        redundancy = np.ones(NUM_TOTAL) / NUM_TOTAL

    # --- Split main/powerball sections ---
    clusters_main = clusters[:NUM_MAIN]
    clusters_power = clusters[NUM_MAIN:]
    redundancy_main = redundancy[:NUM_MAIN]
    redundancy_power = redundancy[NUM_MAIN:]

    # ======================================================
    # MAIN ENTROPY
    # ======================================================
    num_clusters_main = int(np.max(clusters_main)) + 1
    cluster_counts_main = np.zeros(num_clusters_main)
    for draw in historical_data:
        nums = draw.get("numbers") or []
        for n in nums:
            if 1 <= n <= NUM_MAIN:
                cid = clusters_main[n - 1]
                cluster_counts_main[cid] += 1
    cluster_probs_main = cluster_counts_main / (cluster_counts_main.sum() or 1)
    entropy_main_values = -cluster_probs_main * np.log2(np.clip(cluster_probs_main, 1e-9, 1))
    main_entropy = np.array([entropy_main_values[clusters_main[i]] for i in range(NUM_MAIN)])
    main_entropy *= (redundancy_main + fusion[:NUM_MAIN] + markov[:NUM_MAIN] + centroids[:NUM_MAIN])
    main_entropy /= main_entropy.sum() or NUM_MAIN

    # ======================================================
    # POWERBALL ENTROPY
    # ======================================================
    num_clusters_power = int(np.max(clusters_power)) + 1
    cluster_counts_power = np.zeros(num_clusters_power)
    for draw in historical_data:
        pb = draw.get("powerball")
        if pb is None:
            continue
        pb_list = [pb] if isinstance(pb, int) else pb
        for p in pb_list:
            if 1 <= p <= NUM_POWERBALL:
                cid = clusters_power[p - 1]
                cluster_counts_power[cid] += 1
    cluster_probs_power = cluster_counts_power / (cluster_counts_power.sum() or 1)
    entropy_power_values = -cluster_probs_power * np.log2(np.clip(cluster_probs_power, 1e-9, 1))
    power_entropy = np.array([entropy_power_values[clusters_power[i]] for i in range(NUM_POWERBALL)])
    power_entropy *= (redundancy_power + fusion[NUM_MAIN:] + markov[NUM_MAIN:] + centroids[NUM_MAIN:])
    power_entropy /= power_entropy.sum() or NUM_POWERBALL

    # ======================================================
    # COMBINE & NORMALIZE
    # ======================================================
    entropy_full = np.concatenate((main_entropy, power_entropy))
    entropy_full = np.clip(entropy_full, 0, None)
    entropy_full /= entropy_full.sum() or 1.0

    pipeline.add_data("entropy_features", entropy_full)
    logging.info("Shannon entropy features generated successfully.")


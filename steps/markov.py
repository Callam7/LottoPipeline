## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Generate Markov Transition Features (Main + Powerball)
## Description:
## - Computes first-order Markov chain transitions for both main and Powerball domains.
## - Uses unified clustering array (shape 50) split into main(1–40) and Powerball(1–10).
## - Combines transition likelihoods with redundancy weighting.
## - Outputs normalized unified (50,) vector.

import numpy as np
import logging

# --- Global constants ---
NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_markov_matrix(cluster_sequence, num_clusters):
    """Build normalized transition matrix [num_clusters x num_clusters]."""
    matrix = np.zeros((num_clusters, num_clusters), dtype=float)
    for i in range(1, len(cluster_sequence)):
        prev = cluster_sequence[i - 1]
        curr = cluster_sequence[i]
        if 0 <= prev < num_clusters and 0 <= curr < num_clusters:
            matrix[prev, curr] += 1

    # Normalize each row to sum 1
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)

    return matrix


def markov_features(pipeline):
    """Generate unified Markov-based feature vector using clusters and redundancy weights."""
    historical_data = pipeline.get_data("historical_data")
    all_clusters = pipeline.get_data("number_to_cluster")
    redundancy = pipeline.get_data("redundancy")

    # --- Input validation ---
    if historical_data is None or redundancy is None:
        logging.warning("Missing required inputs for Markov computation.")
        pipeline.add_data("markov_features", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    if all_clusters is None or len(all_clusters) != NUM_TOTAL:
        logging.warning("Clustering data missing or invalid length. Using uniform fallback.")
        all_clusters = np.zeros(NUM_TOTAL, dtype=int)

    if len(redundancy) != NUM_TOTAL:
        logging.warning(f"Redundancy length mismatch. Expected {NUM_TOTAL}, got {len(redundancy)}.")
        pipeline.add_data("markov_features", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    # --- Split arrays ---
    num_to_cluster_main = all_clusters[:NUM_MAIN]
    num_to_cluster_power = all_clusters[NUM_MAIN:]
    redundancy_main = redundancy[:NUM_MAIN]
    redundancy_power = redundancy[NUM_MAIN:]

    # ======================================================
    # MAIN NUMBER MARKOV
    # ======================================================
    scores_main = np.ones(NUM_MAIN) / NUM_MAIN
    cluster_sequence_main = []

    for draw in historical_data:
        nums = draw.get("numbers") or []
        for n in nums:
            if 1 <= n <= NUM_MAIN:
                cluster_sequence_main.append(num_to_cluster_main[n - 1])

    if len(cluster_sequence_main) >= 2:
        num_clusters_main = int(np.max(num_to_cluster_main)) + 1
        transition_main = generate_markov_matrix(cluster_sequence_main, num_clusters_main)
        last_cluster_main = cluster_sequence_main[-1]
        scores_main = np.zeros(NUM_MAIN)
        for n in range(NUM_MAIN):
            c_id = num_to_cluster_main[n]
            if 0 <= c_id < num_clusters_main:
                scores_main[n] = transition_main[last_cluster_main, c_id]

        scores_main *= redundancy_main
        total = scores_main.sum()
        scores_main = scores_main / total if total > 0 else np.ones(NUM_MAIN) / NUM_MAIN
    else:
        logging.warning("Not enough main cluster data for Markov modeling.")

    # ======================================================
    # POWERBALL MARKOV
    # ======================================================
    scores_power = np.ones(NUM_POWERBALL) / NUM_POWERBALL
    cluster_sequence_power = []

    for draw in historical_data:
        pb = draw.get("powerball")
        if pb is None:
            continue
        pb_list = [pb] if isinstance(pb, int) else pb
        for p in pb_list:
            if 1 <= p <= NUM_POWERBALL:
                cluster_sequence_power.append(num_to_cluster_power[p - 1])

    if len(cluster_sequence_power) >= 2:
        num_clusters_power = int(np.max(num_to_cluster_power)) + 1
        transition_power = generate_markov_matrix(cluster_sequence_power, num_clusters_power)
        last_cluster_power = cluster_sequence_power[-1]
        scores_power = np.zeros(NUM_POWERBALL)
        for p in range(NUM_POWERBALL):
            c_id = num_to_cluster_power[p]
            if 0 <= c_id < num_clusters_power:
                scores_power[p] = transition_power[last_cluster_power, c_id]

        scores_power *= redundancy_power
        total_p = scores_power.sum()
        scores_power = scores_power / total_p if total_p > 0 else np.ones(NUM_POWERBALL) / NUM_POWERBALL
    else:
        logging.warning("Not enough Powerball cluster data for Markov modeling.")

    # ======================================================
    # COMBINE & NORMALIZE (Main + Powerball)
    # ======================================================
    combined_scores = np.concatenate((scores_main, scores_power))
    combined_scores = np.clip(combined_scores, 0.0, 1.0)
    combined_scores /= combined_scores.sum() or 1.0

    pipeline.add_data("markov_features", combined_scores)
    logging.info("Markov features integrated successfully.")







## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: Compute Shannon Entropy Features for Lotto Prediction
## Description:
## This file calculates the Shannon entropy rate of the lottery number process,
## using Markov transition probabilities between number clusters.
## Entropy measures the inherent unpredictability of the process.
## The calculation incorporates:
## - Clustering: To group numbers into behavioral patterns
## - Markov chain transitions: To model sequential cluster-to-cluster changes
## - Redundancy weighting: To bias towards recent, active numbers
## Output is a (40,) normalized entropy score array for deep learning.

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_stationary_distribution(transition_matrix):
    """
    Computes the stationary distribution π for a Markov chain
    by finding the left eigenvector corresponding to eigenvalue 1.
    """
    eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stationary = np.real(eigvecs[:, idx])
    stationary = stationary / np.sum(stationary)
    stationary = np.maximum(stationary, 0)  # Remove negatives due to floating error
    stationary /= np.sum(stationary)        # Renormalize
    return stationary

def shannon_entropy_rate(transition_matrix, stationary_dist):
    """
    Shannon entropy rate for a first-order Markov chain:
    H = -Σ_i π_i Σ_j P_ij log2(P_ij)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        logP = np.where(transition_matrix > 0, np.log2(transition_matrix), 0)
    entropy_terms = -transition_matrix * logP
    entropy_per_state = np.sum(entropy_terms, axis=1)
    entropy_rate = np.sum(stationary_dist * entropy_per_state)
    return entropy_rate, entropy_per_state  # Return total and per-state entropy

def shannon_entropy_features(pipeline):
    """
    Generates a normalized entropy score for each of the 40 main numbers,
    based on Markov chain dynamics, clustering, and redundancy.

    Parameters:
    - pipeline (DataPipeline): Shared pipeline object.

    Returns:
    - None: Adds "entropy_features" to the pipeline.
    """
    # Step 1: Get required data
    clusters = pipeline.get_data("clusters")
    number_to_cluster = pipeline.get_data("number_to_cluster")
    centroids = pipeline.get_data("centroids")
    redundancy = pipeline.get_data("redundancy")
    markov_scores = pipeline.get_data("markov_features")
    historical_data = pipeline.get_data("historical_data")

    if any(v is None for v in [clusters, number_to_cluster, centroids, redundancy, markov_scores, historical_data]):
        logging.warning("Missing data for Shannon entropy calculation. Using uniform distribution.")
        pipeline.add_data("entropy_features", np.ones(40) / 40)
        return

    # Step 2: Build cluster sequence from historical draws
    cluster_sequence = []
    for draw in historical_data:
        nums = draw.get("numbers", [])
        if not nums:
            continue
        cluster_ids = [number_to_cluster[n - 1] for n in nums if 1 <= n <= 40]
        if cluster_ids:
            avg_cluster = int(round(np.mean(cluster_ids)))
            cluster_sequence.append(avg_cluster)

    if len(cluster_sequence) < 2:
        logging.warning("Not enough data to compute entropy. Using uniform.")
        pipeline.add_data("entropy_features", np.ones(40) / 40)
        return

    # Step 3: Create Markov transition matrix for clusters
    num_clusters = int(np.max(number_to_cluster)) + 1
    trans_matrix = np.zeros((num_clusters, num_clusters), dtype=float)

    for i in range(1, len(cluster_sequence)):
        prev = cluster_sequence[i - 1]
        curr = cluster_sequence[i]
        if 0 <= prev < num_clusters and 0 <= curr < num_clusters:
            trans_matrix[prev, curr] += 1

    # Normalize
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_matrix = np.divide(trans_matrix, row_sums, where=row_sums != 0)

    # Step 4: Compute stationary distribution
    stationary_dist = compute_stationary_distribution(trans_matrix)

    # Step 5: Compute entropy rate and per-state entropy
    _, entropy_per_state = shannon_entropy_rate(trans_matrix, stationary_dist)

    # Step 6: Map entropy per cluster to each number
    entropy_scores = np.array([entropy_per_state[number_to_cluster[n]] for n in range(40)], dtype=float)

    # Step 7: Apply redundancy weighting to bias towards recent active numbers
    if redundancy is not None and len(redundancy) == 40:
        entropy_scores *= redundancy

    # Step 8: Normalize scores to sum to 1
    total = entropy_scores.sum()
    if total > 0:
        entropy_scores /= total
    else:
        entropy_scores = np.ones(40) / 40

    # Step 9: Store results
    pipeline.add_data("entropy_features", entropy_scores)
    logging.info("Shannon entropy features generated successfully.")
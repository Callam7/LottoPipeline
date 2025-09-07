## Modified By: Callam Josef Jackson-Sem
## Project: Lotto Predictor
## Purpose of File: Generate Markov Transition Features from Clustered Historical Data
## Description:
## This file calculates first-order Markov chain transition probabilities based on the clustering 
## and recency of past draw results. The transition matrix represents the likelihood of moving 
## from one cluster state to another. These features are flattened and stored in the pipeline 
## to be consumed by the deep learning step.

# -*- coding: utf-8 -*-


import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_markov_matrix(cluster_sequence, num_clusters):
    """
    Build a transition matrix of shape [num_clusters x num_clusters]
    where entry (i, j) represents the probability of moving
    from cluster i to cluster j in one step.
    """
    matrix = np.zeros((num_clusters, num_clusters), dtype=float)
    
    # Count transitions from previous cluster to current cluster
    for i in range(1, len(cluster_sequence)):
        prev = cluster_sequence[i - 1]
        curr = cluster_sequence[i]
        if 0 <= prev < num_clusters and 0 <= curr < num_clusters:
            matrix[prev, curr] += 1

    # Normalize each row to get transition probabilities
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)

    return matrix


def markov_features(pipeline):
    """
    Generate Markov chain-based features for lottery numbers and store in the pipeline.
    """
    historical_data = pipeline.get_data("historical_data")
    number_to_cluster = pipeline.get_data("number_to_cluster")
    redundancy = pipeline.get_data("redundancy")

    # If critical inputs are missing, fallback to uniform scores
    if historical_data is None or number_to_cluster is None:
        logging.warning("Missing required inputs for Markov computation.")
        pipeline.add_data("markov_features", np.ones(40,) / 40)
        return

    # Determine number of clusters used in training
    num_clusters = int(np.max(number_to_cluster)) + 1

    # Build a sequence of cluster IDs across draws without averaging
    cluster_sequence = []
    for draw in historical_data:
        numbers = draw.get("numbers")
        if not numbers:
            continue
        try:
            for n in numbers:
                if 1 <= n <= 40:
                    cluster_sequence.append(number_to_cluster[n - 1])
        except Exception as e:
            logging.warning(f"Skipping malformed draw: {draw} -- {e}")

    # If not enough history to model transitions, use fallback
    if len(cluster_sequence) < 2:
        logging.warning("Not enough cluster data for Markov modeling.")
        pipeline.add_data("markov_features", np.ones(40,) / 40)
        return

    # Generate transition probability matrix from cluster sequence
    transition_matrix = generate_markov_matrix(cluster_sequence, num_clusters)

    # Score each number based on its cluster's transition probabilities
    scores = np.zeros(40)
    for n in range(40):
        cluster_id = number_to_cluster[n]
        if 0 <= cluster_id < num_clusters:
            # Instead of row mean, use the cluster's row distribution strength
            scores[n] = transition_matrix[cluster_id].mean()

    # Optionally apply redundancy weighting if available
    if redundancy is not None and len(redundancy) == 40:
        scores *= redundancy

    # Normalize scores to sum to 1; fallback if total is zero
    total = scores.sum()
    if total > 0:
        scores /= total
    else:
        scores = np.ones(40,) / 40
        logging.warning("All-zero Markov scores. Falling back to uniform distribution.")

    pipeline.add_data("markov_features", scores)
    logging.info("Markov features integrated successfully.")





## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Generate Markov Transition Features from Clustered Historical Data
## Description:
## Calculates first-order Markov chain transition probabilities based on clustering and recency.
## Scores each number according to its cluster's transitions and multiplies by redundancy.
## Stores normalized features in the pipeline for deep learning consumption.

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_markov_matrix(cluster_sequence, num_clusters):
    """
    Build a transition matrix [num_clusters x num_clusters]
    entry (i,j) = probability of moving from cluster i to cluster j in one step.
    """
    matrix = np.zeros((num_clusters, num_clusters), dtype=float)

    # Count transitions from previous to current cluster
    for i in range(1, len(cluster_sequence)):
        prev = cluster_sequence[i - 1]
        curr = cluster_sequence[i]
        if 0 <= prev < num_clusters and 0 <= curr < num_clusters:
            matrix[prev, curr] += 1

    # Normalize rows to get transition probabilities
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)

    return matrix


def markov_features(pipeline):
    """
    Generate Markov chain features combining clusters, redundancy, and historical data.
    """
    historical_data = pipeline.get_data("historical_data")
    number_to_cluster = pipeline.get_data("number_to_cluster")
    redundancy = pipeline.get_data("redundancy")

    if historical_data is None or number_to_cluster is None or redundancy is None:
        logging.warning("Missing required inputs for Markov computation.")
        pipeline.add_data("markov_features", np.ones(40,) / 40)
        return

    num_clusters = int(np.max(number_to_cluster)) + 1

    # Build cluster sequence from historical draws
    cluster_sequence = []
    for draw in historical_data:
        numbers = draw.get("numbers")
        if not numbers:
            continue
        for n in numbers:
            if 1 <= n <= 40:
                cluster_sequence.append(number_to_cluster[n - 1])

    if len(cluster_sequence) < 2:
        logging.warning("Not enough cluster data for Markov modeling.")
        pipeline.add_data("markov_features", np.ones(40,) / 40)
        return

    # Generate transition matrix
    transition_matrix = generate_markov_matrix(cluster_sequence, num_clusters)

    # Score each number using last cluster in history as source
    last_cluster = cluster_sequence[-1]
    scores = np.zeros(40)
    for n in range(40):
        cluster_id = number_to_cluster[n]
        if 0 <= cluster_id < num_clusters:
            scores[n] = transition_matrix[last_cluster, cluster_id]

    # Multiply by redundancy (recency/gap weighting)
    scores *= redundancy

    # Normalize to sum = 1
    total = scores.sum()
    if total > 0:
        scores /= total
    else:
        scores = np.ones(40,) / 40
        logging.warning("All-zero Markov scores. Falling back to uniform distribution.")

    pipeline.add_data("markov_features", scores)
    logging.info("Markov features integrated successfully.")






## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Compute Shannon Entropy Features for Lotto Prediction
## Description:
## This file calculates the Shannon entropy of the lottery number process,
## using frequency distributions of number clusters from historical draws.
## Entropy measures the inherent unpredictability of the process.
## The calculation incorporates:
## - Clustering: To group numbers into behavioral patterns
## - Historical frequency distributions
## - Redundancy weighting: To bias towards recent, active numbers
## Output is a (40,) normalized entropy score array for deep learning.

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_shannon_entropy(prob_dist):
    """
    Compute Shannon entropy given a probability distribution.
    H = -Σ p(x) log2 p(x)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.where(prob_dist > 0, np.log2(prob_dist), 0)
    entropy = -np.sum(prob_dist * logp)
    return entropy

def shannon_entropy_features(pipeline):
    """
    Generates a normalized entropy score for each of the 40 main numbers,
    based on cluster distributions and redundancy.

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
    historical_data = pipeline.get_data("historical_data")

    if any(v is None for v in [clusters, number_to_cluster, centroids, redundancy, historical_data]):
        logging.warning("Missing data for Shannon entropy calculation. Using uniform distribution.")
        pipeline.add_data("entropy_features", np.ones(40) / 40)
        return

    # Step 2: Count cluster frequencies from historical draws
    num_clusters = int(np.max(number_to_cluster)) + 1
    cluster_counts = np.zeros(num_clusters, dtype=float)

    for draw in historical_data:
        nums = draw.get("numbers", [])
        for n in nums:
            if 1 <= n <= 40:
                cluster_id = number_to_cluster[n - 1]
                cluster_counts[cluster_id] += 1

    if cluster_counts.sum() == 0:
        logging.warning("No cluster counts available for entropy. Using uniform.")
        pipeline.add_data("entropy_features", np.ones(40) / 40)
        return

    # Step 3: Convert to probability distribution
    cluster_probs = cluster_counts / cluster_counts.sum()

    # Step 4: Compute entropy per cluster (local contribution)
    with np.errstate(divide='ignore', invalid='ignore'):
        cluster_entropy = -cluster_probs * np.log2(cluster_probs, where=cluster_probs > 0)

    # Step 5: Map cluster entropy back to each number
    entropy_scores = np.array([cluster_entropy[number_to_cluster[n]] for n in range(40)], dtype=float)

    # Step 6: Apply redundancy weighting
    if redundancy is not None and len(redundancy) == 40:
        entropy_scores *= redundancy

    # Step 7: Normalize scores to sum = 1
    total = entropy_scores.sum()
    if total > 0:
        entropy_scores /= total
    else:
        entropy_scores = np.ones(40) / 40

    # Step 8: Store results
    pipeline.add_data("entropy_features", entropy_scores)
    logging.info("Shannon entropy features generated successfully.")

## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Perform Monte Carlo Simulations on Lottery Number Probabilities
## Description:
##    This file runs Monte Carlo simulations to generate a frequency distribution
##    for the 40 main lottery numbers. It adjusts number probabilities based on
##    Bayesian fusion outputs and clustering information, ensuring recent signals
##    and identified patterns have greater influence. The results are stored in the
##    pipeline for use in subsequent steps of the prediction process.

import numpy as np
import logging
from pipeline import get_dynamic_params  # For dynamic parameter calculations based on historical data size

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration parameters
NUM_MAIN_NUMBERS = 6
NUM_TOTAL_NUMBERS = 40
CLUSTER_MULTIPLIER = 1.2
MIN_PROBABILITY = 1e-8
ENABLE_RANDOM_SEED = True
RANDOM_SEED = 42


def adjust_probabilities(fusion_probs, centroids, clusters):
    """
    Adjusts number probabilities based on Bayesian fusion and clustering information.

    Parameters:
    - fusion_probs (np.ndarray): Bayesian fusion probabilities (shape 40, sums to 1).
    - centroids (np.ndarray): Cluster centroids (array).
    - clusters (np.ndarray): Cluster assignment for each ball index (0..39).

    Returns:
    - np.ndarray: Adjusted + normalized probability distribution (shape 40).
    """
    # Step 1: Multiply fusion by cluster influences
    numbers_prob = fusion_probs * (CLUSTER_MULTIPLIER + centroids[clusters])

    # Step 2: Clip + normalize
    numbers_prob = np.clip(numbers_prob, MIN_PROBABILITY, None)
    numbers_prob /= numbers_prob.sum()

    assert np.isclose(numbers_prob.sum(), 1.0), "Monte Carlo probs not normalized."
    return numbers_prob


def run_simulations_vectorized(numbers_prob, mc_sims):
    picks_matrix = np.empty((mc_sims, NUM_MAIN_NUMBERS), dtype=int)

    for i in range(mc_sims):
        chosen_numbers = np.random.choice(
            np.arange(1, NUM_TOTAL_NUMBERS + 1),
            size=NUM_MAIN_NUMBERS,
            replace=False,
            p=numbers_prob
        )
        picks_matrix[i] = chosen_numbers

    return picks_matrix.flatten()


def calculate_distribution(picks_array):
    counts = np.bincount(picks_array - 1, minlength=NUM_TOTAL_NUMBERS)
    return counts / counts.sum()


def monte_carlo_simulation(pipeline):
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        logging.warning("No historical data available for Monte Carlo simulation.")
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL_NUMBERS) / NUM_TOTAL_NUMBERS)
        return

    # Step 1: Dynamic sim count
    num_draws = len(historical_data)
    mc_sims_base, _ = get_dynamic_params(num_draws)
    mc_sims = int(max(mc_sims_base * 1.5, 1000))
    logging.info(f"Dynamic Monte Carlo simulation count set to {mc_sims}")

    # Step 2: Retrieve Bayesian fusion + clustering
    fusion_probs = pipeline.get_data("bayesian_fusion")
    centroids = pipeline.get_data("centroids")
    clusters = pipeline.get_data("clusters")

    if fusion_probs is None or centroids is None or clusters is None:
        logging.warning("Fusion/clustering data missing. Using uniform distribution.")
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL_NUMBERS) / NUM_TOTAL_NUMBERS)
        return

    try:
        numbers_prob = adjust_probabilities(fusion_probs, centroids, clusters)
    except AssertionError as e:
        logging.error(e)
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL_NUMBERS) / NUM_TOTAL_NUMBERS)
        return

    # Step 3: Monte Carlo sims
    picks_array = run_simulations_vectorized(numbers_prob, mc_sims)
    monte_carlo_distribution = calculate_distribution(picks_array)

    # Step 4: Store result
    pipeline.add_data("monte_carlo", monte_carlo_distribution)
    logging.info(f"Monte Carlo simulation completed with {mc_sims} simulations.")


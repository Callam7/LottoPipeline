## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Perform Monte Carlo Simulations on Lottery Number Probabilities
## Description:
##    This file runs Monte Carlo simulations to generate a frequency distribution
##    for the 40 main lottery numbers. It adjusts number probabilities based on
##    Bayesian fusion outputs and clustering information, ensuring recent signals
##    and identified patterns have greater influence. The results are stored in the
##    pipeline for use in subsequent steps of the prediction process.

## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Perform Monte Carlo Simulations on Lottery Number Probabilities
## Description:
##    Runs Monte Carlo simulations for both main (1–40) and Powerball (1–10) numbers,
##    producing a unified probability vector of shape (50,) for downstream modeling.

import numpy as np
import logging
from pipeline import get_dynamic_params  # Provides dynamic simulation parameters

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL
NUM_PICK = 6
CLUSTER_MULTIPLIER = 1.2
MIN_PROBABILITY = 1e-8
ENABLE_RANDOM_SEED = True
RANDOM_SEED = 42


def adjust_probabilities(fusion_probs, centroids, clusters):
    """Adjust probabilities using Bayesian fusion + clustering."""
    numbers_prob = fusion_probs * (CLUSTER_MULTIPLIER + centroids[clusters])
    numbers_prob = np.clip(numbers_prob, MIN_PROBABILITY, None)
    numbers_prob /= numbers_prob.sum()
    assert np.isclose(numbers_prob.sum(), 1.0), "Monte Carlo probs not normalized."
    return numbers_prob


def run_simulations_vectorized(numbers_prob, mc_sims):
    """Vectorized Monte Carlo draws of NUM_PICK numbers per simulation."""
    if ENABLE_RANDOM_SEED:
        np.random.seed(RANDOM_SEED)

    picks_matrix = np.empty((mc_sims, NUM_PICK), dtype=int)
    for i in range(mc_sims):
        picks_matrix[i] = np.random.choice(
            np.arange(1, NUM_MAIN + 1),
            size=NUM_PICK,
            replace=False,
            p=numbers_prob
        )
    return picks_matrix.flatten()


def calculate_distribution(picks_array, num_total):
    """Counts occurrences and returns normalized probability distribution."""
    counts = np.bincount(picks_array - 1, minlength=num_total)
    return counts / counts.sum()


def monte_carlo_simulation(pipeline):
    """
    Runs Monte Carlo simulation using Bayesian fusion and clustering (main + Powerball).
    Produces a unified shape (50,) vector stored as pipeline["monte_carlo"].
    """
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        logging.warning("No historical data available for Monte Carlo simulation.")
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    # Step 1: Dynamic simulation count
    num_draws = len(historical_data)
    mc_sims_base, _ = get_dynamic_params(num_draws)
    mc_sims = int(max(mc_sims_base * 1.5, 1000))
    logging.info(f"Dynamic Monte Carlo simulation count set to {mc_sims}")

    # Step 2: Retrieve Bayesian fusion (shape 50) and clustering
    fusion_50 = pipeline.get_data("bayesian_fusion")
    clusters = pipeline.get_data("clusters")
    centroids = pipeline.get_data("centroids")

    if fusion_50 is None or clusters is None or centroids is None:
        logging.warning("Fusion/clustering data missing. Using uniform distribution.")
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    # Slice arrays for main and Powerball
    fusion_main = np.array(fusion_50[:NUM_MAIN], dtype=float)
    fusion_power = np.array(fusion_50[NUM_MAIN:], dtype=float)
    clusters_main = np.array(clusters[:NUM_MAIN], dtype=int)
    centroids_main = np.array(centroids[:NUM_MAIN], dtype=float)
    centroids_power = np.array(centroids[NUM_MAIN:], dtype=float)

    # Step 3: Adjust probabilities using fusion + clustering
    try:
        numbers_prob_main = adjust_probabilities(fusion_main, centroids_main, clusters_main)
    except AssertionError as e:
        logging.error(e)
        numbers_prob_main = np.ones(NUM_MAIN) / NUM_MAIN

    # Step 4: Run Monte Carlo for main numbers
    picks_array = run_simulations_vectorized(numbers_prob_main, mc_sims)
    monte_carlo_main = calculate_distribution(picks_array, NUM_MAIN)

    # Step 5: Estimate Powerball distribution (simple weighted fusion-based or uniform)
    powerball_prob = np.clip(fusion_power * (1.0 + centroids_power), MIN_PROBABILITY, None)
    powerball_prob /= powerball_prob.sum()

    # Step 6: Combine (shape 50)
    combined = np.concatenate((monte_carlo_main, powerball_prob))
    combined /= combined.sum()

    pipeline.add_data("monte_carlo", combined)
    logging.info(f"Monte Carlo simulation completed with {mc_sims} simulations.")


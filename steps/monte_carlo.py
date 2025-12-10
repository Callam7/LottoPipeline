## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Perform Monte Carlo Simulations on Lottery Number Probabilities
## Description:
##   Runs Monte Carlo simulations to generate a frequency distribution
##   for main (1-40) and Powerball (1-10) numbers.
##   Uses Bayesian fusion and clustering to adjust probabilities,
##   then simulates draws and returns a shape-(50,) probability vector.

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL
NUM_PICK = 6
CLUSTER_MULTIPLIER = 1.2
MIN_PROBABILITY = 1e-8
ENABLE_RANDOM_SEED = True
RANDOM_SEED = 42

# Seed ONCE (optional reproducibility). Do NOT reseed inside simulation loops.
if ENABLE_RANDOM_SEED:
    np.random.seed(RANDOM_SEED)


def compute_mc_sims(num_draws: int) -> int:
    """
    Dynamic Monte Carlo simulation count that scales linearly with the
    amount of historical data.

    Original behavior (before the cap):
        base = num_draws * 50
        mc_sims = base * 1.5 = num_draws * 75

    We preserve that linear growth, enforce only a minimum,
    and remove the artificial upper cap so it keeps growing
    as the dataset grows (e.g., ~+75 sims per extra draw).
    """
    base = num_draws * 50
    mc_sims = int(max(base * 1.5, 1000))  # at least 1000, then 75 * num_draws
    return mc_sims


def adjust_probabilities(fusion_probs, centroids, clusters):
    """
    Adjust probabilities using Bayesian fusion + clustering.
    This is the OG stable logic, applied domain-local (main or powerball).
    """
    fusion_probs = np.asarray(fusion_probs, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    clusters = np.asarray(clusters, dtype=int)

    weights = CLUSTER_MULTIPLIER + centroids[clusters]
    out = fusion_probs * weights

    # Ensure strictly positive support, then normalize.
    out = np.clip(out, MIN_PROBABILITY, None)
    s = out.sum()
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(out) / len(out)
    out /= s
    return out


def run_main_simulations(numbers_prob, mc_sims):
    """Monte Carlo draws of NUM_PICK main numbers per simulation."""
    numbers_prob = np.asarray(numbers_prob, dtype=float)
    numbers = np.arange(1, NUM_MAIN + 1)

    picks_matrix = np.empty((mc_sims, NUM_PICK), dtype=int)
    for i in range(mc_sims):
        picks_matrix[i] = np.random.choice(
            numbers,
            size=NUM_PICK,
            replace=False,
            p=numbers_prob
        )
    return picks_matrix.flatten()


def run_powerball_simulations(power_prob, mc_sims):
    """Monte Carlo draws of 1 Powerball per simulation (with replacement)."""
    power_prob = np.asarray(power_prob, dtype=float)
    numbers = np.arange(1, NUM_POWERBALL + 1)
    picks = np.random.choice(
        numbers,
        size=mc_sims,
        replace=True,
        p=power_prob
    )
    return picks


def calculate_distribution(picks_array, num_total):
    """
    Counts occurrences and returns a normalized probability distribution
    over 1..num_total, with MIN_PROBABILITY floor applied *before* the
    final normalization so the result is a proper probability vector.
    """
    picks_array = np.asarray(picks_array, dtype=int)
    counts = np.bincount(picks_array - 1, minlength=num_total)

    total = counts.sum()
    if total <= 0:
        # Fallback: no counts (should not happen in practice)
        return np.ones(num_total, dtype=float) / num_total

    dist = counts.astype(float)
    dist = np.clip(dist, MIN_PROBABILITY, None)
    dist /= dist.sum()
    return dist


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

    # Dynamic simulation count (fully dynamic with dataset size)
    num_draws = len(historical_data)
    mc_sims = compute_mc_sims(num_draws)

    # Retrieve Bayesian fusion (shape 50) and clustering
    fusion_50 = pipeline.get_data("bayesian_fusion")
    clusters = pipeline.get_data("clusters")
    centroids = pipeline.get_data("centroids")

    if fusion_50 is None or clusters is None or centroids is None:
        logging.warning("Fusion/clustering data missing. Using uniform distribution.")
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    fusion_50 = np.array(fusion_50, dtype=float)
    clusters = np.array(clusters, dtype=int)
    centroids = np.array(centroids, dtype=float)

    # Slice arrays for main and Powerball
    fusion_main = fusion_50[:NUM_MAIN]
    fusion_power = fusion_50[NUM_MAIN:]

    clusters_main = clusters[:NUM_MAIN]
    centroids_main = centroids[:NUM_MAIN]

    clusters_power = clusters[NUM_MAIN:]
    centroids_power = centroids[NUM_MAIN:]

    # Adjust probabilities using fusion + clustering (domain-local)
    prob_main = adjust_probabilities(fusion_main, centroids_main, clusters_main)
    prob_power = adjust_probabilities(fusion_power, centroids_power, clusters_power)

    # Run Monte Carlo for main numbers
    main_picks = run_main_simulations(prob_main, mc_sims)
    monte_carlo_main = calculate_distribution(main_picks, NUM_MAIN)

    # Run Monte Carlo for Powerball
    power_picks = run_powerball_simulations(prob_power, mc_sims)
    monte_carlo_power = calculate_distribution(power_picks, NUM_POWERBALL)

    # Combine (shape 50), normalize once more for safety
    combined = np.concatenate((monte_carlo_main, monte_carlo_power)).astype(float)
    s = combined.sum()
    if s <= 0 or not np.isfinite(s):
        combined = np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL
    else:
        combined /= s

    pipeline.add_data("monte_carlo", combined)
    logging.info(f"Monte Carlo simulation completed with {mc_sims} simulations.")





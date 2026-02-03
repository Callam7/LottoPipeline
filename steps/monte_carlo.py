## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Perform Monte Carlo Simulations on Lottery Number Probabilities
## Description:
##   Runs Monte Carlo simulations to generate a frequency distribution
##   for main (1-40) and Powerball (1-10) numbers.
##   Uses Bayesian fusion and clustering to adjust probabilities,
##   then simulates draws and returns a shape-(50,) probability vector.

import numpy as np  # Numerical operations and random sampling
import logging      # Logging for runtime diagnostics and monitoring

# Configure logging format and default level
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

NUM_MAIN = 40            # Total possible main numbers
NUM_POWERBALL = 10       # Total possible Powerball numbers
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL  # Combined vector length (50)
NUM_PICK = 6             # Number of main numbers drawn per ticket line
CLUSTER_MULTIPLIER = 1.2 # Base weight applied to clustering influence
MIN_PROBABILITY = 1e-8   # Ensures probabilities never become zero (avoids dead categories)
ENABLE_RANDOM_SEED = True
RANDOM_SEED = 42         # Fixed seed for reproducibility if enabled

# Seed ONCE globally for reproducibility (never reseed inside loops)
if ENABLE_RANDOM_SEED:
    np.random.seed(RANDOM_SEED)


def compute_mc_sims(num_draws: int) -> int:
    """
    Dynamic Monte Carlo simulation count that scales linearly with the
    amount of historical data.
    """
    base = num_draws * 50                  # Base scaling proportional to dataset size
    mc_sims = int(max(base * 1.5, 1000))   # Ensure at least 1000 simulations
    return mc_sims                        # Return number of simulations to run


def adjust_probabilities(fusion_probs, centroids, clusters):
    """
    Adjust probabilities using Bayesian fusion + clustering.
    """
    fusion_probs = np.asarray(fusion_probs, dtype=float)  # Ensure float array
    centroids = np.asarray(centroids, dtype=float)        # Cluster centroid strengths
    clusters = np.asarray(clusters, dtype=int)            # Cluster assignment per number

    weights = CLUSTER_MULTIPLIER + centroids[clusters]    # Compute cluster-based weight per number
    out = fusion_probs * weights                          # Apply weight to fusion probabilities

    out = np.clip(out, MIN_PROBABILITY, None)             # Prevent zeros
    s = out.sum()                                         # Sum for normalization
    if s <= 0 or not np.isfinite(s):                      # Safety fallback
        return np.ones_like(out) / len(out)
    out /= s                                              # Normalize to sum 1
    return out                                            # Return adjusted distribution


def run_main_simulations(numbers_prob, mc_sims):
    """Monte Carlo draws of NUM_PICK main numbers per simulation."""
    numbers_prob = np.asarray(numbers_prob, dtype=float)  # Ensure float probabilities
    numbers = np.arange(1, NUM_MAIN + 1)                  # Possible numbers 1..40

    picks_matrix = np.empty((mc_sims, NUM_PICK), dtype=int)  # Storage for all simulations
    for i in range(mc_sims):                                 # Repeat simulation mc_sims times
        picks_matrix[i] = np.random.choice(                   # Random draw
            numbers,
            size=NUM_PICK,
            replace=False,                                    # Lotto rule: no duplicates per line
            p=numbers_prob                                    # Probability-weighted sampling
        )
    return picks_matrix.flatten()                             # Flatten for frequency counting


def run_powerball_simulations(power_prob, mc_sims):
    """Monte Carlo draws of 1 Powerball per simulation."""
    power_prob = np.asarray(power_prob, dtype=float)        # Ensure float probabilities
    numbers = np.arange(1, NUM_POWERBALL + 1)               # Possible PB numbers 1..10
    picks = np.random.choice(
        numbers,
        size=mc_sims,
        replace=True,                                       # PB is independent per line
        p=power_prob
    )
    return picks                                            # Return drawn PB numbers


def calculate_distribution(picks_array, num_total):
    """
    Convert simulated picks into a probability distribution.
    """
    picks_array = np.asarray(picks_array, dtype=int)        # Ensure integer picks
    counts = np.bincount(picks_array - 1, minlength=num_total)  # Count occurrences

    total = counts.sum()
    if total <= 0:                                          # Safety fallback
        return np.ones(num_total, dtype=float) / num_total

    dist = counts.astype(float)
    dist = np.clip(dist, MIN_PROBABILITY, None)             # Ensure no zero bins
    dist /= dist.sum()                                      # Normalize to probability distribution
    return dist


def monte_carlo_simulation(pipeline):
    """
    Main Monte Carlo driver. Produces pipeline["monte_carlo"].
    """

    historical_data = pipeline.get_data("historical_data")  # Retrieve historical draws
    if not historical_data:                                 # If no history
        logging.warning("No historical data available for Monte Carlo simulation.")
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    num_draws = len(historical_data)                        # Count historical draws
    mc_sims = compute_mc_sims(num_draws)                    # Determine simulation count

    fusion_50 = pipeline.get_data("bayesian_fusion")        # Base probabilities from fusion stage
    clusters = pipeline.get_data("clusters")                # Cluster assignments
    centroids = pipeline.get_data("centroids")              # Cluster centroid strengths

    if fusion_50 is None or clusters is None or centroids is None:
        logging.warning("Fusion/clustering data missing. Using uniform distribution.")
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    fusion_50 = np.array(fusion_50, dtype=float)
    clusters = np.array(clusters, dtype=int)
    centroids = np.array(centroids, dtype=float)

    # Split main vs Powerball
    fusion_main = fusion_50[:NUM_MAIN]
    fusion_power = fusion_50[NUM_MAIN:]

    clusters_main = clusters[:NUM_MAIN]
    centroids_main = centroids[:NUM_MAIN]

    clusters_power = clusters[NUM_MAIN:]
    centroids_power = centroids[NUM_MAIN:]

    # Adjust probabilities per domain
    prob_main = adjust_probabilities(fusion_main, centroids_main, clusters_main)
    prob_power = adjust_probabilities(fusion_power, centroids_power, clusters_power)

    # Simulate main draws
    main_picks = run_main_simulations(prob_main, mc_sims)
    monte_carlo_main = calculate_distribution(main_picks, NUM_MAIN)

    # Simulate Powerball draws
    power_picks = run_powerball_simulations(prob_power, mc_sims)
    monte_carlo_power = calculate_distribution(power_picks, NUM_POWERBALL)

    # Combine into one vector of length 50
    combined = np.concatenate((monte_carlo_main, monte_carlo_power)).astype(float)
    s = combined.sum()
    if s <= 0 or not np.isfinite(s):
        combined = np.ones(NUM_TOTAL, dtype=float) / NUM_TOTAL
    else:
        combined /= s

    pipeline.add_data("monte_carlo", combined)  # Store result in pipeline
    logging.info(f"Monte Carlo simulation completed with {mc_sims} simulations.")






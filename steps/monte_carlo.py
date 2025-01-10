## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: Perform Monte Carlo Simulations on Lottery Number Probabilities
## Description:
## This file runs Monte Carlo simulations to generate a frequency distribution for the 40 main lottery numbers.
## It adjusts number probabilities based on decay factors and clustering information, ensuring more recent
## draws and identified patterns have greater influence. The results are stored in the pipeline for use in
## subsequent steps of the prediction process.

import numpy as np  # For numerical operations and data manipulation
import logging  # For logging warnings and informational messages
from pipeline import get_dynamic_params  # For dynamic parameter calculations based on historical data size

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration parameters
NUM_MAIN_NUMBERS = 6  # Total numbers selected per ticket line
NUM_TOTAL_NUMBERS = 40  # Total unique numbers available in the lottery
MIN_PROBABILITY = 1e-10  # Minimum probability to prevent division by zero or invalid values

def adjust_probabilities(decay_factors, centroids, clusters):
    """
    Adjusts number probabilities based on decay factors and clustering information.

    Parameters:
    - decay_factors (dict): Contains 'numbers' and 'powerball' frequency distributions.
    - centroids (np.ndarray): Array of cluster centroids.
    - clusters (np.ndarray): Array indicating cluster assignments for each number.

    Returns:
    - np.ndarray: Adjusted and normalized probability distribution for the main numbers.
    """
    # Step 1: Combine decay factors and cluster centroids to emphasize certain numbers
    numbers_prob = decay_factors["numbers"] * (1.0 + centroids[clusters])

    # Step 2: Clip probabilities to a minimum value and normalize to create a valid distribution
    numbers_prob = np.clip(numbers_prob, MIN_PROBABILITY, None)
    numbers_prob /= numbers_prob.sum()  # Normalize so probabilities sum to 1

    # Step 3: Ensure the probabilities sum to 1 (validation check)
    assert np.isclose(numbers_prob.sum(), 1.0), "numbers_prob in Monte Carlo is not normalized."

    return numbers_prob

def run_simulations(numbers_prob, mc_sims):
    """
    Runs Monte Carlo simulations to generate lottery numbers based on the adjusted probabilities.

    Parameters:
    - numbers_prob (np.ndarray): Probability distribution for the main numbers.
    - mc_sims (int): Number of Monte Carlo simulations to run.

    Returns:
    - np.ndarray: Array of all chosen numbers across simulations.
    """
    all_picks = []  # Stores all numbers selected during simulations

    for _ in range(mc_sims):
        # Step 1: Select 6 unique numbers based on the adjusted probabilities
        chosen_numbers = np.random.choice(
            np.arange(1, NUM_TOTAL_NUMBERS + 1),  # Numbers 1 to 40
            size=NUM_MAIN_NUMBERS,  # Select 6 numbers
            replace=False,  # No duplicate numbers in a single selection
            p=numbers_prob  # Use the adjusted probabilities
        )
        all_picks.extend(chosen_numbers)  # Add selected numbers to the results

    return np.array(all_picks)  # Return all picks as a NumPy array

def calculate_distribution(picks_array):
    """
    Calculates the frequency distribution of the generated lottery numbers.

    Parameters:
    - picks_array (np.ndarray): Array of all chosen numbers from simulations.

    Returns:
    - np.ndarray: Normalized frequency distribution for the main numbers.
    """
    # Step 1: Count occurrences of each number (1 through 40)
    counts = np.bincount(picks_array - 1, minlength=NUM_TOTAL_NUMBERS)

    # Step 2: Normalize counts to create a probability distribution
    monte_carlo_distribution = counts / counts.sum()

    return monte_carlo_distribution

def monte_carlo_simulation(pipeline):
    """
    Runs a Monte Carlo simulation to estimate the frequency distribution of lottery numbers.
    It leverages historical data, decay factors, and clustering information to generate
    a probability distribution for the 40 main numbers.

    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.

    Returns:
    - None: Adds "monte_carlo" distribution to the pipeline.
    """
    # Step 1: Retrieve historical draw data from the pipeline
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        logging.warning("No historical data available for Monte Carlo simulation.")
        # Assign a uniform distribution if no historical data is present
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL_NUMBERS) / NUM_TOTAL_NUMBERS)
        return

    # Step 2: Determine the number of simulations based on historical data size
    num_draws = len(historical_data)
    mc_sims, _ = get_dynamic_params(num_draws)  # Dynamic Monte Carlo simulation count

    # Step 3: Retrieve supporting data required for the simulation
    decay_factors = pipeline.get_data("decay_factors")
    centroids = pipeline.get_data("centroids")
    clusters = pipeline.get_data("clusters")

    # Step 4: Validate that all necessary supporting data is present
    if decay_factors is None or centroids is None or clusters is None:
        logging.warning("Necessary data missing for Monte Carlo simulation.")
        # Assign a uniform distribution if any supporting data is missing
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL_NUMBERS) / NUM_TOTAL_NUMBERS)
        return

    # Step 5: Adjust number probabilities based on decay factors and cluster centroids
    try:
        numbers_prob = adjust_probabilities(decay_factors, centroids, clusters)
    except AssertionError as e:
        logging.error(e)
        # Assign a uniform distribution if normalization fails
        pipeline.add_data("monte_carlo", np.ones(NUM_TOTAL_NUMBERS) / NUM_TOTAL_NUMBERS)
        return

    # Step 6: Run Monte Carlo simulations to generate lottery numbers
    picks_array = run_simulations(numbers_prob, mc_sims)

    # Step 7: Calculate the frequency distribution from the simulation results
    monte_carlo_distribution = calculate_distribution(picks_array)

    # Step 8: Add the Monte Carlo distribution to the pipeline for downstream use
    pipeline.add_data("monte_carlo", monte_carlo_distribution)

    # Step 9: Log a success message
    logging.info(f"Monte Carlo simulation completed with {mc_sims} simulations.")


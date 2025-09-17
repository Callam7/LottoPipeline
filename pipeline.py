## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Core Data Pipeline and Dynamic Parameter Management
## Description:
## This file defines the `DataPipeline` class, which acts as a shared container for
## data passed between different steps of the pipeline. It includes methods for adding,
## retrieving, and clearing data. Additionally, it includes dynamic parameter calculations
## for simulations and epochs, as well as ticket generation and hit rate analysis.

import os  # For environmental configurations
import logging  # For logging pipeline operations
from typing import Any, Dict, Tuple, List  # For type hinting and clarity
from itertools import combinations  # For generating unique number combinations
import numpy as np  # For numerical computations

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow logging to reduce console clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs from TensorFlow

def get_dynamic_params(num_draws: int) -> Tuple[int, int]:
    """
    Determines dynamic parameters based on the number of historical draws.

    Parameters:
    - num_draws (int): The number of historical lottery draws.

    Returns:
    - Tuple[int, int]: A tuple containing:
        - Monte Carlo simulations count (int): Calculated as min(num_draws * 50, 100,000).
        - Epochs for training (int): Calculated as min(50 + (num_draws // 100), 100).
    """
    mc_sims = min(num_draws * 50, 100_000)
    base_epochs = 50
    dynamic_epochs = min(base_epochs + (num_draws // 100), 100)
    logging.debug(f"Dynamic parameters based on {num_draws} draws: mc_sims={mc_sims}, dynamic_epochs={dynamic_epochs}")
    return mc_sims, dynamic_epochs

class DataPipeline:
    """
    A container for pipeline data.
    Each step can add/retrieve data by key and share it across the pipeline.
    """
    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}
        logging.info("Initialized a new DataPipeline instance.")

    def add_data(self, key: str, value: Any) -> None:
        """
        Adds data to the pipeline with the specified key.

        Parameters:
        - key (str): The key under which the data will be stored.
        - value (Any): The data to store.
        """
        self.data[key] = value
        logging.debug(f"Added data under key '{key}'.")

    def get_data(self, key: str) -> Any:
        """
        Retrieves data from the pipeline by key.

        Parameters:
        - key (str): The key of the data to retrieve.

        Returns:
        - Any: The data associated with the key, or None if the key does not exist.
        """
        data = self.data.get(key)
        if data is not None:
            logging.debug(f"Retrieved data from key '{key}'.")
        else:
            logging.debug(f"No data found for key '{key}'.")
        return data

    def clear_pipeline(self) -> None:
        """
        Clears all data from the pipeline.
        """
        self.data.clear()
        logging.info("Cleared all data from the pipeline.")

def hit_rate_analysis(tickets: List[Dict[str, Any]], historical_data: List[Dict[str, Any]]) -> Tuple[int, Dict[int, int]]:
    """
    Analyzes the performance of generated tickets against historical data.

    Parameters:
    - tickets (List[Dict[str, Any]]): List of generated tickets.
    - historical_data (List[Dict[str, Any]]): List of historical lottery draws.

    Returns:
    - Tuple[int, Dict[int, int]]: Exact matches and partial matches (4/6, 5/6, etc.).
    """
    exact_matches = 0
    partial_matches = {4: 0, 5: 0, 6: 0}

    for draw in historical_data:
        for ticket in tickets:
            main_matches = len(set(ticket["line"]) & set(draw["numbers"]))
            if main_matches >= 4:
                partial_matches[main_matches] += 1
            if main_matches == 6 and ticket["powerball"] == draw["powerball"]:
                exact_matches += 1

    return exact_matches, partial_matches

def generate_ticket(pipeline: DataPipeline, penalty_factor: float = 1.5) -> List[Dict[str, Any]]:
    """
    Combines deep learning predictions and decay factors to generate lottery tickets.
    Utilizes a penalty-based diversity approach to ensure varied ticket lines.

    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.
    - penalty_factor (float): The factor by which to penalize frequently selected numbers.

    Returns:
    - List[Dict[str, Any]]: A list of ticket lines, each containing main numbers and a Powerball number.
    """
    decay_factors = pipeline.get_data("decay_factors")
    predictions = pipeline.get_data("deep_learning_predictions")
    historical_data = pipeline.get_data("historical_data")

    if decay_factors is None or predictions is None:
        logging.error("Decay factors or deep learning predictions are missing in the pipeline.")
        return []

    if historical_data is None:
        logging.error("Historical data is missing in the pipeline.")
        return []

    num_draws = len(historical_data)
    mc_sims, dynamic_epochs = get_dynamic_params(num_draws)
    logging.debug(f"Using {mc_sims} Monte Carlo simulations and {dynamic_epochs} training epochs.")

    # Weighted combination of deep learning predictions and decay factors
    alpha = min(1.0, 0.5 + (num_draws / 10000))  # Example dynamic adjustment
    numbers_prob = alpha * predictions + (1 - alpha) * decay_factors["numbers"]
    numbers_prob = np.clip(numbers_prob, 1e-10, None)
    numbers_prob /= numbers_prob.sum()

    # Powerball probabilities remain based on decay factors
    powerball_prob = decay_factors["powerball"] / decay_factors["powerball"].sum()
    powerball_prob = np.clip(powerball_prob, 1e-10, None)
    powerball_prob /= powerball_prob.sum()

    # Generate ticket lines with penalties for diversity
    ticket: List[Dict[str, Any]] = []
    frequency_penalty = np.zeros(40)
    logging.info("Starting ticket generation process.")

    for _ in range(12):
        adjusted_prob = numbers_prob - penalty_factor * (frequency_penalty / (frequency_penalty.sum() + 1))
        adjusted_prob = np.clip(adjusted_prob, 1e-10, None)
        adjusted_prob /= adjusted_prob.sum()

        # Generate main numbers
        main_numbers = sorted(
            np.random.choice(np.arange(1, 41), size=6, replace=False, p=adjusted_prob)
        )

        # Update penalties for selected numbers
        for num in main_numbers:
            frequency_penalty[num - 1] += 1

        # Generate Powerball number
        powerball = np.random.choice(np.arange(1, 11), p=powerball_prob)
        ticket.append({"line": main_numbers, "powerball": powerball})

    logging.info("Ticket generation completed successfully.")
    return ticket


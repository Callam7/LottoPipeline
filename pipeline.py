# Modified By: Callam
# Project: Lotto Generator
# Purpose: Core Data Pipeline and Dynamic Parameter Management
# Description:
#   - Stores data for all pipeline steps
#   - Provides dynamic epoch scaling ONLY
#   - No Monte Carlo logic exists here anymore

import os  # OS utilities used for environment setup
import logging  # Standard logging module
from typing import Any, Dict, Tuple, List  # Type hinting for better clarity and error checking

# Constants defining the lottery structure
NUM_MAIN_NUMBERS = 40  # Number of main numbers in each draw
NUM_POWERBALL = 10  # Number of possible Powerball values
NUM_TOTAL_NUMBERS = NUM_MAIN_NUMBERS + NUM_POWERBALL  # Total number of output dimensions
TICKET_LINES = 12  # Number of ticket lines to generate
LINE_SIZE = 6  # Number of main numbers per ticket line

# Minimum probability value to prevent numerical issues
MIN_PROB = 1e-12

# Configure logging format and level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress verbose TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_dynamic_params(num_draws: int) -> Tuple[int, int]:
    """
    Dynamic parameter helper.

    NOTE:
    - Monte Carlo no longer uses this function for simulation count.
      Monte Carlo computes its own mc_sims internally.
    - The value is the dynamic epoch count for deep learning.

    Args:
        num_draws (int): Number of historical draws.

    Returns:
        Tuple[None, int]: Placeholder and dynamic epoch count.
    """
    dynamic_epochs = min(50 + (num_draws // 100), 100)  # Increase epochs based on data volume, capped at 100

    logging.debug(f"Dynamic epochs: {dynamic_epochs}")
    return None, dynamic_epochs  # Placeholder for compatibility


class DataPipeline:
    """
    Central data store for all stages of the pipeline.
    Enables shared access to intermediate computations.
    """
    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}

        # Tracks which feature pipes are enabled for ablation testing
        self.active_pipes = {
            "bayesian_fusion_norm": True,
            "monte_carlo": True,
            "redundancy": True,
            "markov_features": True,
            "entropy_features": True,
            "clusters": True,
            "centroids": True
        }

        logging.info("Initialized DataPipeline.")



    def add_data(self, key: str, value: Any) -> None:
        """
        Add a data element to the pipeline.

        Args:
            key (str): Unique key identifier.
            value (Any): Associated data value.
        """
        if key is None:
            raise ValueError("Pipeline key cannot be None.")
        self.data[key] = value  # Store data
        logging.debug(f"Added data under key '{key}'.")

    def get_data(self, key: str) -> Any:
        """
        Retrieve pipeline data.

        If a pipe is disabled during ablation,
        it behaves as though no data exists.
        """

        if key in self.active_pipes and not self.active_pipes[key]:
            logging.debug(f"Pipe '{key}' is disabled.")
            return None

        value = self.data.get(key)

        if value is not None:
            logging.debug(f"Retrieved pipeline data for key '{key}'.")
        else:
            logging.debug(f"No pipeline data for key '{key}'.")

        return value

    def disable_pipe(self, pipe_name: str) -> None:
        """
        Disable a pipe during ablation testing.
        """
        if pipe_name in self.active_pipes:
            self.active_pipes[pipe_name] = False
            logging.info(f"Disabled pipe: {pipe_name}")


    def enable_pipe(self, pipe_name: str) -> None:
        """
        Re-enable a pipe after testing.
        """
        if pipe_name in self.active_pipes:
            self.active_pipes[pipe_name] = True
            logging.info(f"Enabled pipe: {pipe_name}")


    def evaluate(self) -> float:
        """
        Return latest validation AUC.

        Used by Optuna bridge for real ablation scoring.
        """

        val_auc = self.data.get("latest_val_auc")

        if val_auc is None:
            raise ValueError(
                "No validation AUC stored. "
                "deep_learning.py must save latest_val_auc."
            )

        return float(val_auc)


    def clear_pipeline(self) -> None:
        """
        Clears all data from the pipeline.
        """
        self.data.clear()
        logging.info("Pipeline cleared.")


def hit_rate_analysis(
    tickets: List[Dict[str, Any]],
    historical_data: List[Dict[str, Any]]
) -> Tuple[int, Dict[int, int]]:
    """
    Evaluates generated tickets against historical data.

    Args:
        tickets (List[Dict]): Generated tickets to evaluate.
        historical_data (List[Dict]): Past draw results.

    Returns:
        Tuple[int, Dict[int, int]]: Number of exact matches and a breakdown of partial hits (4–6).
    """
    exact_matches = 0  # Count of perfect matches (6 numbers + powerball)
    partial_matches = {4: 0, 5: 0, 6: 0}  # Track partial matches for 4, 5, or 6 numbers

    if not tickets or not historical_data:
        return exact_matches, partial_matches  # No analysis possible

    for draw in historical_data:
        draw_main = set(draw.get("numbers", []))  # Extract main numbers from draw
        draw_powerball = draw.get("powerball")  # Extract powerball

        for ticket in tickets:
            ticket_main = set(ticket["line"])  # Extract main numbers from ticket
            ticket_powerball = ticket["powerball"]  # Extract powerball

            matches = len(ticket_main & draw_main)  # Number of matching main numbers

            if matches >= 4:
                partial_matches[matches] += 1  # Update partial match counter

            if matches == 6 and ticket_powerball == draw_powerball:
                exact_matches += 1  # Count perfect match including powerball

    return exact_matches, partial_matches  # Return full result tuple





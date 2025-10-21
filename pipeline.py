## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Core Data Pipeline and Dynamic Parameter Management
## Description:
##   Manages shared pipeline data, dynamic parameters, ticket generation, and hit rate analysis.
##   Integrates new Bayesian fusion, Monte Carlo, clustering, redundancy, Markov, and entropy features.

import os
import logging
from typing import Any, Dict, Tuple, List
import numpy as np

# ====== Configuration Constants ======
NUM_MAIN_NUMBERS = 40
NUM_POWERBALL = 10
NUM_TOTAL_NUMBERS = NUM_MAIN_NUMBERS + NUM_POWERBALL
TICKET_LINES = 12
LINE_SIZE = 6
MIN_PROB = 1e-12

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ============================================================
# Dynamic Parameter Management
# ============================================================
def get_dynamic_params(num_draws: int) -> Tuple[int, int]:
    """Dynamic Monte Carlo sims and DL epochs based on historical draws."""
    mc_sims = min(num_draws * 50, 100_000)
    dynamic_epochs = min(50 + (num_draws // 100), 100)
    logging.debug(f"Dynamic params: mc_sims={mc_sims}, dynamic_epochs={dynamic_epochs}")
    return mc_sims, dynamic_epochs


# ============================================================
# Core Pipeline Class
# ============================================================
class DataPipeline:
    """Shared container for all pipeline data."""

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}
        logging.info("Initialized DataPipeline.")

    def add_data(self, key: str, value: Any) -> None:
        """Add or update a value in the pipeline."""
        if key is None:
            raise ValueError("Pipeline key cannot be None.")
        self.data[key] = value
        logging.debug(f"Added data under key '{key}'.")

    def get_data(self, key: str) -> Any:
        """Retrieve value from the pipeline by key."""
        value = self.data.get(key)
        if value is not None:
            logging.debug(f"Retrieved data for key '{key}'.")
        else:
            logging.debug(f"No data found for key '{key}'.")
        return value

    def clear_pipeline(self) -> None:
        """Clear all stored data from the pipeline."""
        self.data.clear()
        logging.info("Pipeline cleared.")


# ============================================================
# Hit Rate Analysis
# ============================================================
def hit_rate_analysis(tickets: List[Dict[str, Any]], historical_data: List[Dict[str, Any]]) -> Tuple[int, Dict[int, int]]:
    """Analyze ticket performance versus historical draws."""
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


# ============================================================
# Ticket Generation (Final Step)
# ============================================================
def generate_ticket(pipeline: DataPipeline, penalty_factor: float = 1.5) -> List[Dict[str, Any]]:
    """
    Generate ticket lines using deep learning predictions and Bayesian fusion.
    Combines model predictions with Bayesian fusion weighting and diversity penalties.
    """

    predictions = pipeline.get_data("deep_learning_predictions")
    fusion = pipeline.get_data("bayesian_fusion")
    historical_data = pipeline.get_data("historical_data")

    # Split predictions safely
    if predictions is not None:
        predictions = np.array(predictions, dtype=float)
        if len(predictions) == NUM_TOTAL_NUMBERS:
            predictions_main = predictions[:NUM_MAIN_NUMBERS]
            predictions_powerball = predictions[NUM_MAIN_NUMBERS:]
        elif len(predictions) == NUM_MAIN_NUMBERS:
            predictions_main = predictions
            predictions_powerball = np.ones(NUM_POWERBALL) / NUM_POWERBALL
        else:
            logging.warning(f"Unexpected deep learning prediction length ({len(predictions)}). Using uniform fallback.")
            predictions_main = np.ones(NUM_MAIN_NUMBERS) / NUM_MAIN_NUMBERS
            predictions_powerball = np.ones(NUM_POWERBALL) / NUM_POWERBALL
    else:
        logging.warning("Missing deep learning predictions; using uniform fallback.")
        predictions_main = np.ones(NUM_MAIN_NUMBERS) / NUM_MAIN_NUMBERS
        predictions_powerball = np.ones(NUM_POWERBALL) / NUM_POWERBALL

    # Split fusion safely
    if fusion is not None:
        fusion = np.array(fusion, dtype=float)
        if len(fusion) == NUM_TOTAL_NUMBERS:
            fusion_main = fusion[:NUM_MAIN_NUMBERS]
            fusion_powerball = fusion[NUM_MAIN_NUMBERS:]
        elif len(fusion) == NUM_MAIN_NUMBERS:
            fusion_main = fusion
            fusion_powerball = np.ones(NUM_POWERBALL) / NUM_POWERBALL
        else:
            logging.warning(f"Unexpected Bayesian fusion length ({len(fusion)}). Using uniform fallback.")
            fusion_main = np.ones(NUM_MAIN_NUMBERS) / NUM_MAIN_NUMBERS
            fusion_powerball = np.ones(NUM_POWERBALL) / NUM_POWERBALL
    else:
        logging.warning("Missing Bayesian fusion; using uniform fallback.")
        fusion_main = np.ones(NUM_MAIN_NUMBERS) / NUM_MAIN_NUMBERS
        fusion_powerball = np.ones(NUM_POWERBALL) / NUM_POWERBALL

    ticket: List[Dict[str, Any]] = []
    frequency_penalty = np.zeros(NUM_MAIN_NUMBERS)
    logging.info("Starting ticket generation process.")

    for _ in range(TICKET_LINES):
        # --- MAIN NUMBERS ---
        numbers_prob = fusion_main * predictions_main
        numbers_prob = np.clip(numbers_prob, MIN_PROB, None)
        numbers_prob /= numbers_prob.sum()

        # Apply diversity penalty
        numbers_prob = np.clip(
            numbers_prob - penalty_factor * (frequency_penalty / (frequency_penalty.sum() + 1)),
            MIN_PROB,
            None
        )
        numbers_prob /= numbers_prob.sum()

        main_numbers = sorted(np.random.choice(
            np.arange(1, NUM_MAIN_NUMBERS + 1),
            LINE_SIZE,
            replace=False,
            p=numbers_prob
        ))

        for n in main_numbers:
            frequency_penalty[n - 1] += 1

        # --- POWERBALL ---
        powerball_prob = fusion_powerball * predictions_powerball
        powerball_prob = np.clip(powerball_prob, MIN_PROB, None)
        powerball_prob /= powerball_prob.sum()

        powerball = np.random.choice(np.arange(1, NUM_POWERBALL + 1), p=powerball_prob)

        ticket.append({"line": main_numbers, "powerball": powerball})

    logging.info("Ticket generation completed successfully.")
    pipeline.add_data("generated_tickets", ticket)
    return ticket



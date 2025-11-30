## Modified By: Callam
## Project: Lotto Generator
## Purpose: Core Data Pipeline and Dynamic Parameter Management
## Description:
##   - Stores data for all pipeline steps
##   - Provides dynamic epoch scaling ONLY
##   - No Monte Carlo logic exists here anymore

import os
import logging
from typing import Any, Dict, Tuple, List
import numpy as np

NUM_MAIN_NUMBERS = 40
NUM_POWERBALL = 10
NUM_TOTAL_NUMBERS = NUM_MAIN_NUMBERS + NUM_POWERBALL
TICKET_LINES = 12
LINE_SIZE = 6

MIN_PROB = 1e-12

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def get_dynamic_params(num_draws: int) -> Tuple[int, int]:
    """
    Dynamic parameter helper.

    NOTE:
    - Monte Carlo no longer uses this function for simulation count.
      Monte Carlo computes its own mc_sims internally.
    - The value is the dynamic epoch count for deep learning.

    Returns:
        dynamic_epochs (int): number of Deep Learning epochs to run.
    """
    dynamic_epochs = min(50 + (num_draws // 100), 100)

    logging.debug(f"Dynamic epochs: {dynamic_epochs}")
    return None, dynamic_epochs


class DataPipeline:
    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}
        logging.info("Initialized DataPipeline.")

    def add_data(self, key: str, value: Any) -> None:
        if key is None:
            raise ValueError("Pipeline key cannot be None.")
        self.data[key] = value
        logging.debug(f"Added data under key '{key}'.")

    def get_data(self, key: str) -> Any:
        value = self.data.get(key)
        if value is not None:
            logging.debug(f"Retrieved pipeline data for key '{key}'.")
        else:
            logging.debug(f"No pipeline data for key '{key}'.")
        return value

    def clear_pipeline(self) -> None:
        self.data.clear()
        logging.info("Pipeline cleared.")


def hit_rate_analysis(
    tickets: List[Dict[str, Any]],
    historical_data: List[Dict[str, Any]]
) -> Tuple[int, Dict[int, int]]:
    exact_matches = 0
    partial_matches = {4: 0, 5: 0, 6: 0}

    if not tickets or not historical_data:
        return exact_matches, partial_matches

    for draw in historical_data:
        draw_main = set(draw.get("numbers", []))
        draw_powerball = draw.get("powerball")

        for ticket in tickets:
            ticket_main = set(ticket["line"])
            ticket_powerball = ticket["powerball"]

            matches = len(ticket_main & draw_main)

            if matches >= 4:
                partial_matches[matches] += 1

            if matches == 6 and ticket_powerball == draw_powerball:
                exact_matches += 1

    return exact_matches, partial_matches





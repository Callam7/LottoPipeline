## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: To Analyze and Normalize Lottery Number Frequencies (fixed)
## Description:
## Computes normalized frequency for main (1..40) and Powerball (1..10).
## Handles powerball values that may be stored as ints or lists.
## Stores "number_frequency" (40,), "powerball_frequency" (10,), and
## "number_frequency_combined" (50,) in the pipeline.

import numpy as np
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

NUM_MAIN = 40
NUM_POWERBALL = 10
TOTAL_NUM = NUM_MAIN + NUM_POWERBALL


def analyze_number_frequency(pipeline: Any) -> None:
    """
    Computes correct global frequencies across *all* historical draws.

    Outputs:
        pipeline["number_frequency"]           -> shape (40,)
        pipeline["powerball_frequency"]        -> shape (10,)
        pipeline["number_frequency_combined"]  -> shape (50,)
    """

    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        logging.warning("No historical data available for frequency analysis.")
        pipeline.add_data("number_frequency", np.ones(NUM_MAIN) / NUM_MAIN)
        pipeline.add_data("powerball_frequency", np.ones(NUM_POWERBALL) / NUM_POWERBALL)
        pipeline.add_data("number_frequency_combined", np.ones(TOTAL_NUM) / TOTAL_NUM)
        return

    # ----------------------
    # MAIN NUMBERS
    # ----------------------
    all_main = []
    for draw in historical_data:
        nums = draw.get("numbers") or []
        for n in nums:
            if isinstance(n, int) and 1 <= n <= NUM_MAIN:
                all_main.append(n)

    if all_main:
        counts_main = np.bincount(np.array(all_main, dtype=int) - 1, minlength=NUM_MAIN)
        number_frequency = counts_main / counts_main.sum()
    else:
        number_frequency = np.ones(NUM_MAIN) / NUM_MAIN

    # ----------------------
    # POWERBALL
    # ----------------------
    all_pbs = []
    for draw in historical_data:
        pb = draw.get("powerball")
        if isinstance(pb, int) and 1 <= pb <= NUM_POWERBALL:
            all_pbs.append(pb)
        elif isinstance(pb, list):
            for p in pb:
                if isinstance(p, int) and 1 <= p <= NUM_POWERBALL:
                    all_pbs.append(p)

    if all_pbs:
        counts_pb = np.bincount(np.array(all_pbs, dtype=int) - 1, minlength=NUM_POWERBALL)
        powerball_frequency = counts_pb / counts_pb.sum()
    else:
        powerball_frequency = np.ones(NUM_POWERBALL) / NUM_POWERBALL

    # ----------------------
    # INVALID ENTRY LOGGING
    # ----------------------
    invalid_main = [
        n for draw in historical_data
        for n in (draw.get("numbers") or [])
        if not (isinstance(n, int) and 1 <= n <= NUM_MAIN)
    ]

    invalid_pb = []
    for draw in historical_data:
        pb = draw.get("powerball")
        if pb is None:
            continue
        if isinstance(pb, int):
            if not (1 <= pb <= NUM_POWERBALL):
                invalid_pb.append(pb)
        elif isinstance(pb, list):
            for p in pb:
                if not (isinstance(p, int) and 1 <= p <= NUM_POWERBALL):
                    invalid_pb.append(p)
        else:
            invalid_pb.append(pb)

    if invalid_main:
        logging.warning("Invalid main numbers ignored: %s", sorted(set(invalid_main)))
    if invalid_pb:
        logging.warning("Invalid powerball numbers ignored: %s", sorted(set(invalid_pb)))

    # ----------------------
    # SAVE
    # ----------------------
    pipeline.add_data("number_frequency", number_frequency)
    pipeline.add_data("powerball_frequency", powerball_frequency)
    pipeline.add_data("number_frequency_combined",
                      np.concatenate([number_frequency, powerball_frequency]))

    logging.info("Number frequency analysis completed.")


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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

NUM_MAIN = 40
NUM_POWERBALL = 10
TOTAL_NUM = NUM_MAIN + NUM_POWERBALL

def analyze_number_frequency(pipeline: Any) -> None:
    """
    Analyze historical_data in pipeline and produce:
      - pipeline['number_frequency'] (shape NUM_MAIN)
      - pipeline['powerball_frequency'] (shape NUM_POWERBALL)
      - pipeline['number_frequency_combined'] (shape TOTAL_NUM = NUM_MAIN + NUM_POWERBALL)
    The function tolerates powerball stored as int or list and logs invalid entries.
    """
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        logging.warning("No historical data available for frequency analysis.")
        pipeline.add_data("number_frequency", np.ones(NUM_MAIN) / NUM_MAIN)
        pipeline.add_data("powerball_frequency", np.ones(NUM_POWERBALL) / NUM_POWERBALL)
        pipeline.add_data("number_frequency_combined", np.ones(TOTAL_NUM) / TOTAL_NUM)
        return

    # Initialize frequencies (will hold the final normalized frequencies based on all previous draws up to last)
    main_frequency = np.ones(NUM_MAIN) / NUM_MAIN
    powerball_frequency = np.ones(NUM_POWERBALL) / NUM_POWERBALL

    # Build incremental frequencies up to each draw (final value equals frequencies computed using all previous draws)
    for i, draw in enumerate(historical_data):
        if i == 0:
            continue  # no prior draws to base frequencies on
        # collect all main numbers from previous draws
        prev_main = [
            n for prev_draw in historical_data[:i]
            for n in (prev_draw.get("numbers") or [])
            if isinstance(n, int) and 1 <= n <= NUM_MAIN
        ]
        if prev_main:
            counts_main = np.bincount(np.array(prev_main, dtype=int) - 1, minlength=NUM_MAIN)
            main_frequency = counts_main / counts_main.sum()
        else:
            main_frequency = np.ones(NUM_MAIN) / NUM_MAIN

        # collect all powerball numbers from previous draws (handle int or list)
        prev_pbs = []
        for prev_draw in historical_data[:i]:
            pb = prev_draw.get("powerball")
            if pb is None:
                continue
            if isinstance(pb, list):
                prev_pbs.extend([p for p in pb if isinstance(p, int)])
            elif isinstance(pb, int):
                prev_pbs.append(pb)
            else:
                # ignore other types
                continue

        prev_pbs = [p for p in prev_pbs if 1 <= p <= NUM_POWERBALL]
        if prev_pbs:
            counts_pb = np.bincount(np.array(prev_pbs, dtype=int) - 1, minlength=NUM_POWERBALL)
            powerball_frequency = counts_pb / counts_pb.sum()
        else:
            powerball_frequency = np.ones(NUM_POWERBALL) / NUM_POWERBALL

    # Log invalid numbers found anywhere in historical_data (for diagnostics)
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
        if isinstance(pb, list):
            invalid_pb.extend([p for p in pb if not (isinstance(p, int) and 1 <= p <= NUM_POWERBALL)])
        elif isinstance(pb, int):
            if not (1 <= pb <= NUM_POWERBALL):
                invalid_pb.append(pb)
        else:
            invalid_pb.append(pb)

    if invalid_main:
        logging.warning("Encountered invalid main numbers (ignored): %s", sorted(set(invalid_main)))
    if invalid_pb:
        logging.warning("Encountered invalid powerball numbers (ignored): %s", sorted(set(invalid_pb)))

    # Save separately and also combined for downstream compatibility
    pipeline.add_data("number_frequency", main_frequency)
    pipeline.add_data("powerball_frequency", powerball_frequency)
    pipeline.add_data("number_frequency_combined", np.concatenate([main_frequency, powerball_frequency]))

    logging.info("Number frequency analysis completed.")


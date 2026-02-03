## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: To Analyze and Normalize Lottery Number Frequencies (fixed)
## Description:
## Computes normalized frequency for main (1..40) and Powerball (1..10).
## Handles powerball values that may be stored as ints or lists.
## Stores "number_frequency" (40,), "powerball_frequency" (10,), and
## "number_frequency_combined" (50,) in the pipeline.

import numpy as np                     # Numerical library for fast counting and normalization
import logging                         # Logging for warnings and progress messages
from typing import Any                 # Allows flexible typing for the pipeline object

logging.basicConfig(                   # Configure global logging behavior
    level=logging.INFO,                # Show INFO level and above
    format="%(asctime)s - %(levelname)s - %(message)s"  # Timestamped log format
)

NUM_MAIN = 40                          # Total possible main lottery numbers (1–40)
NUM_POWERBALL = 10                     # Total possible Powerball numbers (1–10)
TOTAL_NUM = NUM_MAIN + NUM_POWERBALL   # Combined output width (50)


def analyze_number_frequency(pipeline: Any) -> None:
    """
    Computes correct global frequencies across *all* historical draws.

    Outputs:
        pipeline["number_frequency"]           -> shape (40,)
        pipeline["powerball_frequency"]        -> shape (10,)
        pipeline["number_frequency_combined"]  -> shape (50,)
    """

    historical_data = pipeline.get_data("historical_data")  # Retrieve stored historical draws

    if not historical_data:  # If no data exists, cannot compute frequencies
        logging.warning("No historical data available for frequency analysis.")
        pipeline.add_data("number_frequency", np.ones(NUM_MAIN) / NUM_MAIN)  # Uniform fallback
        pipeline.add_data("powerball_frequency", np.ones(NUM_POWERBALL) / NUM_POWERBALL)  # Uniform fallback
        pipeline.add_data("number_frequency_combined", np.ones(TOTAL_NUM) / TOTAL_NUM)  # Combined uniform fallback
        return  # Exit early

    # ----------------------
    # MAIN NUMBERS
    # ----------------------
    all_main = []  # Collect every valid main number ever drawn

    for draw in historical_data:               # Loop through every historical draw
        nums = draw.get("numbers") or []       # Get main numbers list safely
        for n in nums:                        # Check each number in the draw
            if isinstance(n, int) and 1 <= n <= NUM_MAIN:  # Validate number range
                all_main.append(n)            # Store valid number

    if all_main:  # Only compute if at least one valid number exists
        counts_main = np.bincount(            # Count occurrences of each number
            np.array(all_main, dtype=int) - 1,  # Convert to 0-based indices
            minlength=NUM_MAIN               # Ensure length is exactly 40
        )
        number_frequency = counts_main / counts_main.sum()  # Normalize to probabilities
    else:
        number_frequency = np.ones(NUM_MAIN) / NUM_MAIN  # Uniform fallback

    # ----------------------
    # POWERBALL
    # ----------------------
    all_pbs = []  # Collect every valid Powerball ever drawn

    for draw in historical_data:               # Loop through draws again
        pb = draw.get("powerball")             # Powerball may be int or list

        if isinstance(pb, int) and 1 <= pb <= NUM_POWERBALL:  # Single PB value
            all_pbs.append(pb)

        elif isinstance(pb, list):             # Some data sources store PB as list
            for p in pb:
                if isinstance(p, int) and 1 <= p <= NUM_POWERBALL:
                    all_pbs.append(p)

    if all_pbs:  # Only compute if at least one valid PB exists
        counts_pb = np.bincount(               # Count PB occurrences
            np.array(all_pbs, dtype=int) - 1,  # Convert to 0-based index
            minlength=NUM_POWERBALL           # Ensure length is exactly 10
        )
        powerball_frequency = counts_pb / counts_pb.sum()  # Normalize to probabilities
    else:
        powerball_frequency = np.ones(NUM_POWERBALL) / NUM_POWERBALL  # Uniform fallback

    # ----------------------
    # INVALID ENTRY LOGGING
    # ----------------------
    invalid_main = [                           # Detect invalid main numbers for logging
        n for draw in historical_data
        for n in (draw.get("numbers") or [])
        if not (isinstance(n, int) and 1 <= n <= NUM_MAIN)
    ]

    invalid_pb = []                            # Detect invalid PB entries

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

    if invalid_main:  # Log once if bad main numbers were found
        logging.warning("Invalid main numbers ignored: %s", sorted(set(invalid_main)))

    if invalid_pb:    # Log once if bad PB numbers were found
        logging.warning("Invalid powerball numbers ignored: %s", sorted(set(invalid_pb)))

    # ----------------------
    # SAVE
    # ----------------------
    pipeline.add_data("number_frequency", number_frequency)  # Store main probabilities (40,)
    pipeline.add_data("powerball_frequency", powerball_frequency)  # Store PB probabilities (10,)
    pipeline.add_data(
        "number_frequency_combined",
        np.concatenate([number_frequency, powerball_frequency])  # Store combined vector (50,)
    )

    logging.info("Number frequency analysis completed.")  # Confirm completion



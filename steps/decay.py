## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: To Calculate Decay-Weighted Lottery Number Frequencies
## Description:
## This file applies a decay factor to historical lottery draw data, assigning more weight to recent draws
## and less weight to older ones. Decay is calculated based on the time elapsed between draws, assuming a
## weekly interval. The normalized decay-weighted frequency distributions are stored in the data pipeline
## for downstream use in prediction models.

import numpy as np  # For numerical and array operations
from datetime import datetime  # To handle and compare draw dates
import logging  # For logging warnings and informational messages

# Optional: install dateutil if needed (pip install python-dateutil)
try:
    from dateutil import parser  # More flexible date parsing
except ImportError:
    parser = None

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _safe_parse_date(date_str):
    """
    Attempts to parse a date string strictly (YYYY-MM-DD). If that fails, tries
    a more flexible parser (dateutil) to handle cases like '2025-1-4' without zero-padding.
    """
    # First, try the strict datetime format
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        # If dateutil is available, use its flexible parsing
        if parser:
            return parser.parse(date_str)
        else:
            # If dateutil isn't installed, re-raise the original error
            raise

def calculate_decay_factors(pipeline, decay_rate=0.98):
    """
    Calculates decay factors for both main lottery numbers and Powerball numbers based on historical draw data.
    Applies a decay to older draws to give more weight to recent results, assuming weekly draw intervals.
    
    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.
    - decay_rate (float): The rate at which older draws' influence decays. Default is 0.98.
    
    Returns:
    - None: Adds "decay_factors" to the pipeline with normalized frequency distributions.
    """

    ## Step 1: Retrieve historical draw data from the pipeline
    ## The "historical_data" key is expected to contain a list of past draws.
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        ## Case: No historical data available
        logging.warning("No historical data available for decay calculations.")
        # Assign uniform distributions for both main numbers and Powerball
        pipeline.add_data("decay_factors", {
            "numbers": np.ones(40) / 40,
            "powerball": np.ones(10) / 10
        })
        return  # Exit the function as there's no data to process

    ## Step 2: Ensure historical data is sorted by draw_date in true chronological order
    try:
        historical_data.sort(key=lambda x: _safe_parse_date(x['draw_date']))
    except KeyError as e:
        logging.error(f"Missing 'draw_date' key in historical data: {e}")
        return
    except ValueError as e:
        logging.error(f"Unable to parse one or more 'draw_date' values: {e}")
        return

    ## Step 3: Initialize arrays for decay-weighted frequency counts
    numbers_frequency = np.zeros(40)  # For main lottery numbers (1 to 40)
    powerball_frequency = np.zeros(10)  # For Powerball numbers (1 to 10)
    last_date = None  # To track the date of the previous draw for decay calculations

    ## Step 4: Iterate over historical draws and calculate decay factors
    for i, draw in enumerate(historical_data):
        try:
            current_date = _safe_parse_date(draw['draw_date'])
        except (ValueError, KeyError) as e:
            # Log a warning and skip draws with invalid or missing dates
            logging.warning(f"Skipping draw at index {i} due to invalid/missing 'draw_date': {e}")
            continue

        if last_date:
            # Calculate the number of days between the current and previous draws
            delta_days = (current_date - last_date).days
            if delta_days < 0:
                # If a date appears out of order, log but still apply minimal or no penalty
                logging.warning(
                    f"Draw at index {i} has an earlier date than the previous draw. "
                    f"last_date={last_date}, current_date={current_date}. "
                    "Using zero days for decay to avoid skipping."
                )
                delta_days = 0  # Optionally use abs(delta_days) if desired
            # Convert days to weeks
            weeks_passed = max(0, delta_days / 7)
            decay_factor = decay_rate ** weeks_passed
        else:
            # First draw in the dataset
            decay_factor = 1.0

        ## Step 5: Update frequency counts for main numbers
        for num in draw.get("numbers", []):
            if 1 <= num <= 40:
                numbers_frequency[num - 1] += decay_factor
            else:
                logging.warning(f"Invalid main number {num} in draw {draw.get('draw_date', 'unknown')}.")

        ## Step 6: Update frequency counts for Powerball
        pball = draw.get("powerball")
        if pball is not None and 1 <= pball <= 10:
            powerball_frequency[pball - 1] += decay_factor
        elif pball is not None:
            logging.warning(f"Invalid Powerball number {pball} in draw {draw.get('draw_date', 'unknown')}.")

        last_date = current_date  # Update for next iteration

    ## Step 7: Normalize frequency counts to create probability distributions

    # Normalize main numbers frequency
    total_numbers = numbers_frequency.sum()
    if total_numbers > 0:
        numbers_frequency /= total_numbers
        logging.info("Normalized main numbers frequency distribution successfully.")
    else:
        numbers_frequency = np.ones(40) / 40
        logging.warning("Total main numbers frequency is zero. Assigned uniform distribution.")

    # Normalize Powerball frequency
    total_powerball = powerball_frequency.sum()
    if total_powerball > 0:
        powerball_frequency /= total_powerball
        logging.info("Normalized Powerball frequency distribution successfully.")
    else:
        powerball_frequency = np.ones(10) / 10
        logging.warning("Total Powerball frequency is zero. Assigned uniform distribution.")

    ## Step 8: Store normalized distributions in the pipeline
    pipeline.add_data("decay_factors", {
        "numbers": numbers_frequency,
        "powerball": powerball_frequency
    })



## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: To Calculate Decay-Weighted Lottery Number Frequencies
## Description:
## This file applies a decay factor to historical lottery draw data, assigning more weight to recent draws 
## and less weight to older ones. Decay is calculated based on the time elapsed between draws, assuming a 
## weekly interval. The normalized decay-weighted frequency distributions are stored in the data pipeline 
## for downstream use in prediction models.

import numpy as np  # For numerical and array operations
from datetime import datetime  # To handle and compare draw dates
import logging  # For logging warnings and informational messages

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    # Step 1: Retrieve historical draw data from the pipeline
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
    
    # Step 2: Initialize arrays for decay-weighted frequency counts
    numbers_frequency = np.zeros(40)  # For main lottery numbers (1 to 40)
    powerball_frequency = np.zeros(10)  # For Powerball numbers (1 to 10)
    
    last_date = None  # To track the date of the previous draw for decay calculations
    
    # Step 3: Iterate over historical draws and calculate decay factors
    for i, draw in enumerate(historical_data):
        try:
            # Parse the draw date from the expected format "YYYY-MM-DD"
            current_date = datetime.strptime(draw['draw_date'], "%Y-%m-%d")
        except (ValueError, KeyError) as e:
            # Log a warning and skip draws with invalid or missing dates
            logging.warning(f"Skipping draw at index {i} due to invalid/missing 'draw_date': {e}")
            continue

        if last_date:
            # Calculate the number of days between the current and previous draws
            delta_days = (current_date - last_date).days
            if delta_days < 0:
                # Case: Current draw date is earlier than the last one
                logging.warning(f"Draw at index {i} has an earlier date than the previous draw. Skipping decay adjustment.")
                decay_factor = decay_rate ** 1  # Apply minimal decay
            else:
                # Apply decay based on the number of weeks passed
                weeks_passed = delta_days / 7  # Convert days to weeks
                decay_factor = decay_rate ** weeks_passed
        else:
            # Case: No previous date available (first draw in the dataset)
            decay_factor = 1.0  # No decay applied to the first draw

        # Step 4: Update frequency counts for main numbers
        for num in draw.get("numbers", []):
            if 1 <= num <= 40:
                numbers_frequency[num - 1] += decay_factor
            else:
                logging.warning(f"Encountered invalid main number {num} in draw dated {draw.get('draw_date', 'unknown')}. Ignoring.")

        # Step 5: Update frequency counts for Powerball numbers
        pball = draw.get("powerball")
        if pball is not None:
            if 1 <= pball <= 10:
                powerball_frequency[pball - 1] += decay_factor
            else:
                logging.warning(f"Encountered invalid Powerball number {pball} in draw dated {draw.get('draw_date', 'unknown')}. Ignoring.")
        else:
            logging.warning(f"Missing 'powerball' number in draw dated {draw.get('draw_date', 'unknown')}. Ignoring.")

        # Update last_date for the next iteration
        last_date = current_date

    # Step 6: Normalize frequency counts to create probability distributions
    ## Normalize main numbers frequency
    total_numbers = numbers_frequency.sum()
    if total_numbers > 0:
        numbers_frequency /= total_numbers
        logging.info("Normalized main numbers frequency distribution successfully.")
    else:
        # Case: Total counts are zero
        numbers_frequency = np.ones(40) / 40
        logging.warning("Total main numbers frequency is zero. Assigned uniform distribution.")

    ## Normalize Powerball frequency
    total_powerball = powerball_frequency.sum()
    if total_powerball > 0:
        powerball_frequency /= total_powerball
        logging.info("Normalized Powerball frequency distribution successfully.")
    else:
        # Case: Total counts are zero
        powerball_frequency = np.ones(10) / 10
        logging.warning("Total Powerball frequency is zero. Assigned uniform distribution.")

    # Step 7: Store normalized distributions in the pipeline
    pipeline.add_data("decay_factors", {
        "numbers": numbers_frequency,
        "powerball": powerball_frequency
    })
    ## The decay-weighted distributions are now ready for use in subsequent steps of the pipeline.

## Modified By: Callam Josef Jackson-Sem
## Project: Lotto Predictor
## Purpose of File: To Analyze and Normalize Lottery Number Frequencies
## Description:
## This file calculates the frequency of each lottery number (1 to 40) from historical draw data. 
## It normalizes the occurrences to create a probability distribution that reflects the likelihood 
## of each number being drawn. The normalized distribution is stored in the data pipeline for 
## use in subsequent prediction steps.

import numpy as np  # For numerical operations and efficient computations
import logging  # For logging warnings and info messages
from typing import Any  # For type hinting of the pipeline object

# Configures logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_number_frequency(pipeline: Any) -> None:
    """
    Analyzes the frequency of each of the 40 main lottery numbers based on historical draw data.
    It computes how often each number has been drawn and normalizes the frequencies to create
    a probability distribution. This distribution is then added to the pipeline for use in subsequent steps.
    
    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.
    
    Returns:
    - None: Adds "number_frequency" to the pipeline. If no valid data is found, assigns a uniform distribution.
    """
    
    # Step 1: Retrieve historical draw data from the pipeline
    ## The "historical_data" key is expected to hold a list of past draws.
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        # Case: No historical data is present in the pipeline
        logging.warning("No historical data available for frequency analysis.")
        # Assign a uniform distribution to the pipeline as a fallback
        pipeline.add_data("number_frequency", np.ones(40) / 40)
        return  # Exit the function as there's no data to process
    
    # Step 2: Extract valid numbers from the historical data
    ## Extract all valid numbers (1 to 40) using list comprehension
    all_numbers = [
        num for draw in historical_data 
        for num in draw.get("numbers", [])  # Safely retrieve "numbers" key
        if 1 <= num <= 40  # Filter out invalid numbers
    ]
    
    # Step 3: Detect and log invalid numbers
    ## Identifys numbers outside the valid range (1 to 40)
    invalid_numbers = [
        num for draw in historical_data 
        for num in draw.get("numbers", []) 
        if not (1 <= num <= 40)
    ]
    if invalid_numbers:
        # Logs the count and list of invalid numbers
        unique_invalid = set(invalid_numbers)
        count_invalid = len(invalid_numbers)
        logging.warning(f"Encountered {count_invalid} invalid number(s): {sorted(unique_invalid)}. Ignoring them in frequency analysis.")
    
    # Step 4: Checks if any valid numbers were found
    if not all_numbers:
        ## Case: No valid numbers are found
        logging.warning("No valid numbers found in historical data. Assigning uniform distribution.")
        # Assigns a uniform distribution to the pipeline
        numbers_frequency = np.ones(40) / 40
    else:
        ## Case: Valid numbers exist
        # Converts the list of numbers to a NumPy array for efficient processing
        all_numbers_array = np.array(all_numbers)
        # Count occurrences of each number using NumPy's bincount function
        ## Subtracts 1 from each number to align with 0-based indexing
        numbers_frequency = np.bincount(all_numbers_array - 1, minlength=40)
        # Calculate the total occurrences for normalization
        total_counts = numbers_frequency.sum()
        
        if total_counts > 0:
            ## Normalizes the frequency counts to create a probability distribution
            numbers_frequency = numbers_frequency / total_counts
            logging.info("Number frequency analysis completed successfully.")
        else:
            # Case: Total counts are zero after filtering
            logging.warning("Total counts are zero after filtering. Assigned uniform distribution to number frequencies.")
            numbers_frequency = np.ones(40) / 40
    
    # Step 5: Store the normalized frequency distribution in the pipeline
    pipeline.add_data("number_frequency", numbers_frequency)
    ## The distribution is now available for use in subsequent steps of the pipeline.
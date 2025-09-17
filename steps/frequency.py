## Modified By: Callam
## Project: Lotto Generator
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
    It computes how often each number has been drawn *up to each draw* and normalizes the frequencies 
    to create a probability distribution. This distribution is then added to the pipeline for use 
    in subsequent steps.
    
    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.
    
    Returns:
    - None: Adds "number_frequency" to the pipeline. If no valid data is found, assigns a uniform distribution.
    """
    
    # Step 1: Retrieve historical draw data from the pipeline
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        # Case: No historical data is present in the pipeline
        logging.warning("No historical data available for frequency analysis.")
        # Assign a uniform distribution to the pipeline as a fallback
        pipeline.add_data("number_frequency", np.ones(40) / 40)
        return  # Exit the function as there's no data to process
    
    # Step 2: Initialize frequency counts for each number
    numbers_frequency = np.zeros(40, dtype=float)
    
    # Step 3: Build frequencies incrementally (up to the last draw)
    for i, draw in enumerate(historical_data):
        # Skip first draw (no prior draws to compute frequency)
        if i == 0:
            continue
        
        # Aggregate all previous draws up to t-1
        prev_numbers = [
            num for prev_draw in historical_data[:i]
            for num in prev_draw.get("numbers", [])
            if 1 <= num <= 40
        ]
        
        # Count occurrences for this point in time
        if prev_numbers:
            counts = np.bincount(np.array(prev_numbers) - 1, minlength=40)
            numbers_frequency = counts / counts.sum()
        else:
            # No valid numbers yet, assign uniform
            numbers_frequency = np.ones(40) / 40
    
    # Step 4: Detect and log invalid numbers globally
    invalid_numbers = [
        num for draw in historical_data 
        for num in draw.get("numbers", []) 
        if not (1 <= num <= 40)
    ]
    if invalid_numbers:
        unique_invalid = set(invalid_numbers)
        count_invalid = len(invalid_numbers)
        logging.warning(f"Encountered {count_invalid} invalid number(s): {sorted(unique_invalid)}. Ignoring them in frequency analysis.")
    
    # Step 5: Store the normalized frequency distribution in the pipeline
    pipeline.add_data("number_frequency", numbers_frequency)


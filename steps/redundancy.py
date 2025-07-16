## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: Generate Sequential / Temporal Features for Lottery Numbers
## Description:
## This file analyzes historical lottery draw sequences to extract features based on transitions,
## recurrences, and number gaps. It calculates how recently each number was drawn and the frequency
## of gaps between appearances. These features enhance the predictive signal for deep learning by
## incorporating temporal context into the dataset.

import numpy as np  # For numerical operations
import logging  # For logging warnings and informational messages

# Configuration parameters
NUM_TOTAL_NUMBERS = 40  # Total unique numbers in the lottery

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_recency_features(historical_data):
    """
    Calculates how many draws ago each number was last drawn.

    Parameters:
    - historical_data (list of dict): Contains past draw results.

    Returns:
    - np.ndarray: Array of recency for each number (1-40).
    """
    recency = np.full(NUM_TOTAL_NUMBERS, len(historical_data), dtype=float)

    for idx, draw in enumerate(reversed(historical_data)):
        for number in draw["numbers"]:
            if 1 <= number <= NUM_TOTAL_NUMBERS:
                if recency[number - 1] == len(historical_data):
                    recency[number - 1] = idx + 1  # How many draws ago this number appeared

    # Normalize recency to between 0 and 1 (recent = 0, old = 1)
    recency /= len(historical_data)
    return recency


def calculate_gap_frequency(historical_data):
    """
    Calculates average gaps between occurrences for each number.

    Parameters:
    - historical_data (list of dict): Contains past draw results.

    Returns:
    - np.ndarray: Array of normalized average gap lengths (1-40).
    """
    gaps = [[] for _ in range(NUM_TOTAL_NUMBERS)]
    last_seen = [-1] * NUM_TOTAL_NUMBERS

    for idx, draw in enumerate(historical_data):
        for number in range(1, NUM_TOTAL_NUMBERS + 1):
            if number in draw["numbers"]:
                if last_seen[number - 1] != -1:
                    gaps[number - 1].append(idx - last_seen[number - 1])
                last_seen[number - 1] = idx

    avg_gaps = np.array([
        np.mean(gap) if gap else len(historical_data)
        for gap in gaps
    ])

    avg_gaps /= len(historical_data)  # Normalize
    return avg_gaps


def sequential_features(pipeline):
    """
    Generates sequential / temporal features from historical draw data.
    These features capture how recently each number appeared and average gaps
    between occurrences to help model sequential lottery behavior.

    Parameters:
    - pipeline (DataPipeline): The pipeline object containing shared data across steps.

    Returns:
    - None: Adds "sequential_features" to the pipeline.
    """
    # Step 1: Retrieve historical draw data
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        logging.warning("No historical data available for sequential feature generation.")
        pipeline.add_data("sequential_features", np.ones(NUM_TOTAL_NUMBERS) / NUM_TOTAL_NUMBERS)
        return

    # Step 2: Calculate how recently each number appeared
    recency_features = calculate_recency_features(historical_data)

    # Step 3: Calculate average gaps between occurrences for each number
    gap_features = calculate_gap_frequency(historical_data)

    # Step 4: Combine features into a single array (shape: 40)
    combined_features = (recency_features + gap_features) / 2

    # Step 5: Flatten the features into a 1D array for use in downstream steps
    flattened_features = combined_features.flatten()

    # Step 6: Add features to pipeline
    pipeline.add_data("redundancy", flattened_features)

    # Step 7: Log a success message
    logging.info("Sequential / Temporal features generated successfully.")

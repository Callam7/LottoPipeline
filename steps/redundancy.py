## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Generate Sequential / Temporal Features for Lottery Numbers (Main + Powerball)
## Description:
## - Calculates temporal recency and gap frequency for both main (1–40) and Powerball (1–10) numbers.
## - Normalizes each domain separately to preserve proportional scaling.
## - Integrates optional clustering centroids to improve feature coherence.
## - Concatenates into a unified shape-(50,) feature vector for downstream modeling.

import numpy as np
import logging

# Config
NUM_MAIN_NUMBERS = 40
NUM_POWERBALL_NUMBERS = 10
NUM_TOTAL_NUMBERS = NUM_MAIN_NUMBERS + NUM_POWERBALL_NUMBERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_recency_features(historical_data, key, num_total):
    """Calculates how many draws ago each number was last drawn."""
    recency = np.full(num_total, len(historical_data), dtype=float)

    for idx, draw in enumerate(reversed(historical_data)):
        draw_nums = draw.get(key, [])
        if isinstance(draw_nums, int):
            draw_nums = [draw_nums]
        for number in draw_nums:
            if 1 <= number <= num_total and recency[number - 1] == len(historical_data):
                recency[number - 1] = idx + 1

    # Normalize (recent = 0, old = 1)
    recency /= len(historical_data)
    return recency


def calculate_gap_frequency(historical_data, key, num_total):
    """Calculates average gaps between occurrences for each number."""
    gaps = [[] for _ in range(num_total)]
    last_seen = [-1] * num_total

    for idx, draw in enumerate(historical_data):
        draw_nums = draw.get(key, [])
        if isinstance(draw_nums, int):
            draw_nums = [draw_nums]
        for number in range(1, num_total + 1):
            if number in draw_nums:
                if last_seen[number - 1] != -1:
                    gaps[number - 1].append(idx - last_seen[number - 1])
                last_seen[number - 1] = idx

    avg_gaps = np.array([np.mean(gap) if gap else len(historical_data) for gap in gaps])
    avg_gaps /= len(historical_data)
    return avg_gaps


def sequential_features(pipeline):
    """
    Generates sequential / temporal features for both main and Powerball numbers.
    Produces a shape-(50,) unified vector (40 main + 10 Powerball).
    """
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        logging.warning("No historical data available for sequential feature generation.")
        pipeline.add_data("redundancy", np.ones(NUM_TOTAL_NUMBERS) / NUM_TOTAL_NUMBERS)
        return

    # --- Main numbers ---
    recency_main = calculate_recency_features(historical_data, "numbers", NUM_MAIN_NUMBERS)
    gap_main = calculate_gap_frequency(historical_data, "numbers", NUM_MAIN_NUMBERS)
    combined_main = (recency_main + gap_main) / 2.0

    # --- Powerball numbers ---
    recency_power = calculate_recency_features(historical_data, "powerball", NUM_POWERBALL_NUMBERS)
    gap_power = calculate_gap_frequency(historical_data, "powerball", NUM_POWERBALL_NUMBERS)
    combined_power = (recency_power + gap_power) / 2.0

    # --- Concatenate into unified vector ---
    combined_features = np.concatenate((combined_main, combined_power))
    combined_features = combined_features.astype(float)

    # Optional integration with clustering centroids
    centroids = pipeline.get_data("centroids")
    if centroids is not None and len(centroids) == NUM_TOTAL_NUMBERS:
        combined_features *= (centroids + 1e-6)  # add small epsilon to preserve scale
        logging.info("Redundancy features modulated with clustering centroids.")
    else:
        logging.warning("Clustering centroids missing or invalid; using raw redundancy values.")

    # Normalize to [0,1]
    combined_features -= np.min(combined_features)
    combined_features /= np.ptp(combined_features) + 1e-9

    # Store result
    pipeline.add_data("redundancy", combined_features)
    logging.info("Sequential / Temporal features generated successfully.")

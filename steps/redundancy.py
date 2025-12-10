## Modified By: Callam
## Project: Lotto Generator
## Purpose: Sequential / Temporal Features (Strict, Cluster-Integrated)
## Description:
## - Correct recency weighting (recent ? stronger).
## - Unbiased gap modeling (initial, internal, final gaps).
## - Variance normalization so recency/gap contribute comparably.
## - Clustering centroids are always applied; if invalid, a neutral vector is used.
## - Output is a unified feature vector of shape (50,).

import numpy as np
import logging

NUM_MAIN_NUMBERS = 40
NUM_POWERBALL_NUMBERS = 10
NUM_TOTAL_NUMBERS = NUM_MAIN_NUMBERS + NUM_POWERBALL_NUMBERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================================================
# RECENCY: corrected (recent = higher score)
# ============================================================
def calculate_recency_features(historical_data, key, num_total):
    """Recency: newer appearances ? larger value in [0,1]."""
    total_draws = len(historical_data)
    recency = np.full(num_total, total_draws, dtype=float)

    # Iterate from newest ? oldest
    for idx, draw in enumerate(reversed(historical_data)):
        draw_nums = draw.get(key, [])
        if isinstance(draw_nums, int):
            draw_nums = [draw_nums]

        for number in draw_nums:
            if 1 <= number <= num_total and recency[number - 1] == total_draws:
                recency[number - 1] = idx + 1

    # Map: oldest ~ 0, newest ~ 1
    recency = 1.0 - (recency / total_draws)
    return recency


# ============================================================
# GAP FREQUENCY: unbiased gap model
# ============================================================
def calculate_gap_frequency(historical_data, key, num_total):
    """
    Average gap length between appearances for each number, including:
    - initial gap (from draw 0 to first appearance)
    - internal gaps (between appearances)
    - final gap (last appearance to final draw)
    """
    total_draws = len(historical_data)
    occurrences = [[] for _ in range(num_total)]

    # Collect occurrence indices
    for idx, draw in enumerate(historical_data):
        draw_nums = draw.get(key, [])
        if isinstance(draw_nums, int):
            draw_nums = [draw_nums]

        draw_set = set(draw_nums)
        for number in draw_set:
            if 1 <= number <= num_total:
                occurrences[number - 1].append(idx)

    avg_gaps = np.zeros(num_total, dtype=float)

    for n in range(num_total):
        occ = occurrences[n]

        if not occ:
            # Never drawn ? maximal gap
            avg_gaps[n] = total_draws
            continue

        gaps = []

        # initial gap
        gaps.append(occ[0])

        # internal gaps
        for i in range(1, len(occ)):
            gaps.append(occ[i] - occ[i - 1])

        # final gap
        gaps.append(total_draws - occ[-1])

        avg_gaps[n] = np.mean(gaps)

    # Normalize to [0,1]
    return avg_gaps / total_draws


# ============================================================
# FULL SEQUENTIAL / TEMPORAL FEATURES
# ============================================================
def sequential_features(pipeline):
    """
    Generates sequential / temporal features for both main and Powerball numbers.
    Output:
        pipeline["redundancy"] -> np.ndarray shape (50,)
    """
    historical_data = pipeline.get_data("historical_data")

    if not historical_data:
        logging.error("No historical data for sequential feature generation.")
        pipeline.add_data("redundancy", np.ones(NUM_TOTAL_NUMBERS) / NUM_TOTAL_NUMBERS)
        return

    # ===================== MAIN (1–40) =====================
    recency_main = calculate_recency_features(historical_data, "numbers", NUM_MAIN_NUMBERS)
    gap_main = calculate_gap_frequency(historical_data, "numbers", NUM_MAIN_NUMBERS)

    # variance normalization to avoid one dominating
    rec_main_std = np.std(recency_main) or 1.0
    gap_main_std = np.std(gap_main) or 1.0

    combined_main = (recency_main / rec_main_std + gap_main / gap_main_std) / 2.0

    # ===================== POWERBALL (1–10) =====================
    recency_power = calculate_recency_features(historical_data, "powerball", NUM_POWERBALL_NUMBERS)
    gap_power = calculate_gap_frequency(historical_data, "powerball", NUM_POWERBALL_NUMBERS)

    rec_power_std = np.std(recency_power) or 1.0
    gap_power_std = np.std(gap_power) or 1.0

    combined_power = (recency_power / rec_power_std + gap_power / gap_power_std) / 2.0

    # ===================== COMBINE (50,) =====================
    combined_features = np.concatenate((combined_main, combined_power)).astype(float)

    # ===================== CLUSTER MODULATION (ALWAYS APPLIED) =====================
    centroids = pipeline.get_data("centroids")
    if centroids is None or len(centroids) != NUM_TOTAL_NUMBERS:
        logging.error(
            f"Centroids missing or invalid (expected length {NUM_TOTAL_NUMBERS}). "
            "Using neutral centroids (all ones)."
        )
        centroids = np.ones(NUM_TOTAL_NUMBERS, dtype=float)
    else:
        centroids = np.asarray(centroids, dtype=float)

    # Always modulate by centroids (or neutral ones)
    combined_features *= (centroids + 1e-6)

    # ===================== FINAL NORMALIZATION =====================
    min_v = np.min(combined_features)
    ptp_v = np.ptp(combined_features) + 1e-9
    combined_features = (combined_features - min_v) / ptp_v

    pipeline.add_data("redundancy", combined_features)
    logging.info("Sequential / Temporal features generated successfully (cluster-modulated).")

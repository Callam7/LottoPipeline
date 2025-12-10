## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Generate Markov Transition Features (Main + Powerball)
## Description:
## Implements a mathematically correct first-order Markov chain based on
## cluster transitions across historical draws.
##
## Key corrections:
## - No longer treats numbers inside a single draw as sequential events.
## - Uses ONE representative cluster per draw (mean cluster or mode cluster).
## - Builds proper cluster→cluster Markov transitions across draws.
## - Redundancy weighting applied AFTER Markov probabilities (safe).
## - Fully normalized shape-(50,) output.

import numpy as np
import logging

NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_markov_matrix(sequence, num_states):
    """Build normalized transition matrix [num_states x num_states]."""
    mat = np.zeros((num_states, num_states), dtype=float)

    for i in range(1, len(sequence)):
        a = sequence[i - 1]
        b = sequence[i]
        if 0 <= a < num_states and 0 <= b < num_states:
            mat[a, b] += 1

    # Normalize rows
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mat = np.divide(mat, row_sums, where=row_sums != 0)

    return mat


def representative_cluster(numbers, mapping, domain_size):
    """
    Compute a representative cluster ID for a draw:
    - takes all numbers within domain
    - maps to their cluster IDs
    - uses the MODE cluster label as the global draw-state
    This produces a proper Markov chain across draws.
    """
    clusters = []
    for n in numbers:
        if 1 <= n <= domain_size:
            clusters.append(mapping[n - 1])

    if len(clusters) == 0:
        return None

    # mode = most common cluster
    values, counts = np.unique(clusters, return_counts=True)
    return int(values[np.argmax(counts)])


def markov_features(pipeline):
    """Produce mathematically correct Markov-based prediction features."""
    historical = pipeline.get_data("historical_data")
    clusters = pipeline.get_data("number_to_cluster")
    redundancy = pipeline.get_data("redundancy")

    if historical is None or clusters is None or redundancy is None:
        logging.warning("Markov inputs missing. Using uniform fallback.")
        pipeline.add_data("markov_features", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    if len(clusters) != NUM_TOTAL or len(redundancy) != NUM_TOTAL:
        logging.warning("Cluster/redundancy dimension mismatch.")
        pipeline.add_data("markov_features", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    # Split cluster + redundancy vectors
    main_map = clusters[:NUM_MAIN]
    power_map = clusters[NUM_MAIN:]
    red_main = redundancy[:NUM_MAIN]
    red_power = redundancy[NUM_MAIN:]

    # =====================================================
    # MAIN MARKOV CHAIN
    # =====================================================
    main_seq = []
    for draw in historical:
        nums = draw.get("numbers") or []
        rep = representative_cluster(nums, main_map, NUM_MAIN)
        if rep is not None:
            main_seq.append(rep)

    if len(main_seq) >= 2:
        num_states_main = int(np.max(main_map)) + 1
        T_main = generate_markov_matrix(main_seq, num_states_main)
        last_state = main_seq[-1]

        scores_main = np.zeros(NUM_MAIN)
        for n in range(NUM_MAIN):
            c = main_map[n]
            if 0 <= c < num_states_main:
                scores_main[n] = T_main[last_state, c]

        # redundancy AFTER Markov
        scores_main *= red_main

        total = scores_main.sum()
        if total > 0:
            scores_main /= total
        else:
            scores_main = np.ones(NUM_MAIN) / NUM_MAIN
    else:
        logging.warning("Not enough main transitions. Using uniform.")
        scores_main = np.ones(NUM_MAIN) / NUM_MAIN

    # =====================================================
    # POWERBALL MARKOV CHAIN
    # =====================================================
    power_seq = []
    for draw in historical:
        pb = draw.get("powerball")
        if isinstance(pb, int) and 1 <= pb <= NUM_POWERBALL:
            rep = representative_cluster([pb], power_map, NUM_POWERBALL)
            if rep is not None:
                power_seq.append(rep)

    if len(power_seq) >= 2:
        num_states_power = int(np.max(power_map)) + 1
        T_power = generate_markov_matrix(power_seq, num_states_power)
        last_state = power_seq[-1]

        scores_power = np.zeros(NUM_POWERBALL)
        for p in range(NUM_POWERBALL):
            c = power_map[p]
            if 0 <= c < num_states_power:
                scores_power[p] = T_power[last_state, c]

        scores_power *= red_power

        total = scores_power.sum()
        if total > 0:
            scores_power /= total
        else:
            scores_power = np.ones(NUM_POWERBALL) / NUM_POWERBALL
    else:
        logging.warning("Not enough Powerball transitions. Using uniform.")
        scores_power = np.ones(NUM_POWERBALL) / NUM_POWERBALL

    # =====================================================
    # COMBINE & NORMALIZE
    # =====================================================
    combined = np.concatenate((scores_main, scores_power))
    combined = np.clip(combined, 0.0, None)
    combined /= combined.sum() or 1.0

    pipeline.add_data("markov_features", combined)
    logging.info("Markov features integrated successfully.")








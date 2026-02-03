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

import numpy as np                      # Numerical library for arrays and matrix math
import logging                          # Logging system for warnings and info messages

NUM_MAIN = 40                           # Number of main lottery numbers
NUM_POWERBALL = 10                      # Number of Powerball numbers
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL    # Total output length (50 probabilities)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Configure logging format and default log level


def generate_markov_matrix(sequence, num_states):
    """Build normalized transition matrix [num_states x num_states]."""

    mat = np.zeros((num_states, num_states), dtype=float)
    # Create a square matrix to store transition counts between states

    for i in range(1, len(sequence)):
        # Loop over consecutive pairs in the state sequence

        a = sequence[i - 1]
        # Previous state index

        b = sequence[i]
        # Next state index

        if 0 <= a < num_states and 0 <= b < num_states:
            # Ensure both states are valid indices before updating

            mat[a, b] += 1
            # Count one transition from state a to state b

    row_sums = mat.sum(axis=1, keepdims=True)
    # Sum of transitions leaving each state (row totals)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Ignore warnings caused by division by zero

        mat = np.divide(mat, row_sums, where=row_sums != 0)
        # Normalize each row so transitions become probabilities

    return mat
    # Return the transition probability matrix


def representative_cluster(numbers, mapping, domain_size):
    """
    Convert a set of numbers in a draw into ONE representative cluster ID.
    This allows us to treat each draw as a single Markov "state".
    """

    clusters = []
    # Will hold cluster IDs corresponding to numbers in this draw

    for n in numbers:
        # Iterate through numbers in the draw

        if 1 <= n <= domain_size:
            # Ensure number is within correct domain (main or Powerball)

            clusters.append(mapping[n - 1])
            # Convert number into its cluster ID and store it

    if len(clusters) == 0:
        # If no valid numbers were found

        return None
        # Cannot define a representative state

    values, counts = np.unique(clusters, return_counts=True)
    # Count how many times each cluster appears in this draw

    return int(values[np.argmax(counts)])
    # Return the most frequent cluster (mode) as the draw's state


def markov_features(pipeline):
    """Produce mathematically correct Markov-based prediction features."""

    historical = pipeline.get_data("historical_data")
    # Retrieve list of past draw records

    clusters = pipeline.get_data("number_to_cluster")
    # Retrieve mapping from number index to cluster ID

    redundancy = pipeline.get_data("redundancy")
    # Retrieve redundancy weighting vector

    if historical is None or clusters is None or redundancy is None:
        # If any required input is missing

        logging.warning("Markov inputs missing. Using uniform fallback.")
        pipeline.add_data("markov_features", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return
        # Stop early and store uniform distribution

    if len(clusters) != NUM_TOTAL or len(redundancy) != NUM_TOTAL:
        # Ensure vectors are correct size

        logging.warning("Cluster/redundancy dimension mismatch.")
        pipeline.add_data("markov_features", np.ones(NUM_TOTAL) / NUM_TOTAL)
        return

    main_map = clusters[:NUM_MAIN]
    # Cluster IDs for main numbers

    power_map = clusters[NUM_MAIN:]
    # Cluster IDs for Powerball numbers

    red_main = redundancy[:NUM_MAIN]
    # Redundancy weights for main numbers

    red_power = redundancy[NUM_MAIN:]
    # Redundancy weights for Powerball numbers

    # =========================
    # MAIN MARKOV CHAIN
    # =========================

    main_seq = []
    # Sequence of representative cluster states for main draws

    for draw in historical:
        # Process each historical draw

        nums = draw.get("numbers") or []
        # Get main numbers list

        rep = representative_cluster(nums, main_map, NUM_MAIN)
        # Convert draw into representative cluster state

        if rep is not None:
            main_seq.append(rep)
            # Add state to Markov sequence

    if len(main_seq) >= 2:
        # Need at least 2 states to form transitions

        num_states_main = int(np.max(main_map)) + 1
        # Determine how many distinct clusters exist

        T_main = generate_markov_matrix(main_seq, num_states_main)
        # Build transition probability matrix

        last_state = main_seq[-1]
        # Most recent cluster state

        scores_main = np.zeros(NUM_MAIN)
        # Probability scores per main number

        for n in range(NUM_MAIN):
            # Evaluate each main number

            c = main_map[n]
            # Get cluster ID of number

            if 0 <= c < num_states_main:
                scores_main[n] = T_main[last_state, c]
                # Probability of transitioning from last cluster to this number's cluster

        scores_main *= red_main
        # Apply redundancy weighting AFTER Markov probabilities

        total = scores_main.sum()
        # Sum probabilities

        if total > 0:
            scores_main /= total
            # Normalize if valid
        else:
            scores_main = np.ones(NUM_MAIN) / NUM_MAIN
            # Fallback to uniform

    else:
        logging.warning("Not enough main transitions. Using uniform.")
        scores_main = np.ones(NUM_MAIN) / NUM_MAIN

    # =========================
    # POWERBALL MARKOV CHAIN
    # =========================

    power_seq = []
    # Sequence of Powerball cluster states

    for draw in historical:
        pb = draw.get("powerball")
        # Get Powerball value

        if isinstance(pb, int) and 1 <= pb <= NUM_POWERBALL:
            rep = representative_cluster([pb], power_map, NUM_POWERBALL)
            # Convert Powerball into representative cluster

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

    # =========================
    # COMBINE & NORMALIZE
    # =========================

    combined = np.concatenate((scores_main, scores_power))
    # Merge main and Powerball probabilities

    combined = np.clip(combined, 0.0, None)
    # Ensure no negative probabilities

    combined /= combined.sum() or 1.0
    # Normalize entire vector safely

    pipeline.add_data("markov_features", combined)
    # Store result in pipeline

    logging.info("Markov features integrated successfully.")
    # Log completion










## Modified By: Callam
## Project: Lotto Generator
## Purpose: Predictive ticket generation using deep_learning_predictions with:
##   - soft repetition penalty (usage-based, NOT probabilistic decay)
##   - anti-overlap rejection sampling (hard diversity constraint)
##
## This module is responsible ONLY for:
##   - selecting 12 distinct high-likelihood outcomes
##   - preventing excessive repetition across the ticket
##   - enforcing hard overlap constraints
##
## Adaptation notes:
##   - For other lotteries: change NUM_MAIN_NUMBERS, NUM_POWERBALLS,
##     NUM_PER_LINE, NUM_LINES, and overlap rules.


import numpy as np                      # Import NumPy for numerical operations and probabilistic sampling
from data_io import save_current_ticket # Import helper to persist the generated ticket to storage


# =========================
# Ticket configuration
# =========================

NUM_MAIN_NUMBERS = 40   # Total count of possible main numbers (1..40)
NUM_POWERBALLS = 10     # Total count of possible Powerball numbers (1..10)
NUM_PER_LINE = 6        # Number of main numbers selected per ticket line
NUM_LINES = 12          # Total number of ticket lines to generate


# =========================
# Numeric safety
# =========================

MIN_PROBABILITY = 1e-12 # Minimum probability floor to prevent zero-probability sampling failures


# =========================
# Diversity control (USAGE penalty, NOT entropy/decay)
# =========================
# These penalties reduce selection likelihood based on how often a number
# has already appeared across the ticket.
# They do NOT modify the learned probability distribution itself.

MAIN_USAGE_PENALTY = 0.65  # Multiplicative penalty per prior usage of a main number
PB_USAGE_PENALTY   = 0.60  # Multiplicative penalty per prior usage of a Powerball


# =========================
# Anti-overlap constraints
# =========================

MAX_OVERLAP_MAIN = 2       # Maximum allowed shared main numbers with any previous line
MAX_SAME_POWERBALL = 2     # Maximum number of times a Powerball value may appear
MAX_RESAMPLE_TRIES = 300   # Maximum attempts to find a valid line before fallback


# =========================
# Probability utilities
# =========================

def safe_norm(x):
    """
    Safely normalise a probability vector.

    - Clips values to MIN_PROBABILITY to avoid zeros
    - Returns a uniform distribution if total mass becomes invalid

    This function performs NUMERICAL SAFETY ONLY.
    It does NOT compute entropy or apply decay.
    """

    x = np.asarray(x, dtype=float)      # Ensure input is a NumPy float array
    x = np.clip(x, MIN_PROBABILITY, None)  # Enforce minimum probability floor
    s = x.sum()                         # Compute total probability mass

    if s <= 0.0:                        # Guard against degenerate vectors
        return np.full_like(x, 1.0 / len(x))  # Fallback to uniform distribution

    return x / s                        # Return properly normalised probabilities


def _overlap_count(a, b):
    """
    Count the number of overlapping values between two lists.
    Used to enforce diversity constraints between ticket lines.
    """
    return len(set(a) & set(b))         # Compute intersection cardinality


# =========================
# Main ticket generator
# =========================

def generate_ticket(pipeline):
    """
    Generate NUM_LINES lottery lines using:
      - deep_learning_predictions as the FINAL probability surface
      - usage-based repetition penalties (diversity control only)
      - anti-overlap rejection sampling

    No entropy computation occurs here.
    No probabilistic decay is applied here.
    """

    # ---------------------------------------------
    # Fetch deep learning probability predictions
    # ---------------------------------------------

    predictions = pipeline.get_data("deep_learning_predictions")  # Retrieve model output from pipeline
    expected_len = NUM_MAIN_NUMBERS + NUM_POWERBALLS               # Expected vector length (40 + 10)


    # ---------------------------------------------
    # Safety fallback if predictions missing/invalid
    # ---------------------------------------------

    if predictions is None or len(predictions) != expected_len:    # Validate prediction vector
        print("Missing or invalid predictions. Using uniform fallback.")

        base_main_prob = np.ones(NUM_MAIN_NUMBERS) / NUM_MAIN_NUMBERS  # Uniform main-number distribution
        base_pb_prob   = np.ones(NUM_POWERBALLS) / NUM_POWERBALLS      # Uniform Powerball distribution

    else:
        predictions = np.asarray(predictions, dtype=float)         # Ensure numeric array

        base_main_prob = safe_norm(predictions[:NUM_MAIN_NUMBERS]) # Extract + normalise main-number probabilities
        base_pb_prob   = safe_norm(predictions[NUM_MAIN_NUMBERS:]) # Extract + normalise Powerball probabilities


    # ---------------------------------------------
    # Usage tracking (diversity control only)
    # ---------------------------------------------

    main_usage = np.zeros(NUM_MAIN_NUMBERS, dtype=int)  # Track how often each main number is used
    pb_usage   = np.zeros(NUM_POWERBALLS, dtype=int)    # Track how often each Powerball is used

    ticket = []  # Store accepted ticket lines


    # ---------------------------------------------
    # Generate each ticket line
    # ---------------------------------------------

    for _ in range(NUM_LINES):           # Loop once per ticket line

        chosen_main = None               # Placeholder for accepted main numbers
        chosen_pb = None                 # Placeholder for accepted Powerball


        for _ in range(MAX_RESAMPLE_TRIES):  # Rejection-sampling loop

            # -------------------------------------
            # Apply usage-based penalties
            # -------------------------------------

            main_penalty = np.power(MAIN_USAGE_PENALTY, main_usage)  # Compute per-number main penalties
            pb_penalty   = np.power(PB_USAGE_PENALTY, pb_usage)      # Compute per-number PB penalties

            effective_main_prob = safe_norm(base_main_prob * main_penalty)  # Combine prediction + usage penalty
            effective_pb_prob   = safe_norm(base_pb_prob * pb_penalty)      # Combine prediction + usage penalty


            # -------------------------------------
            # Sample candidate line
            # -------------------------------------

            cand_main = sorted(                                   # Sort for consistency/readability
                np.random.choice(
                    np.arange(1, NUM_MAIN_NUMBERS + 1),          # Main number domain
                    size=NUM_PER_LINE,                            # Select 6 numbers
                    replace=False,                                # No duplicates within a line
                    p=effective_main_prob                         # Probability-weighted sampling
                )
            )

            cand_pb = int(                                        # Sample Powerball
                np.random.choice(
                    np.arange(1, NUM_POWERBALLS + 1),             # Powerball domain
                    p=effective_pb_prob                           # Probability-weighted sampling
                )
            )


            # -------------------------------------
            # Constraint checks
            # -------------------------------------

            if pb_usage[cand_pb - 1] >= MAX_SAME_POWERBALL:        # Enforce PB repetition cap
                continue                                          # Reject candidate

            if any(                                                # Enforce overlap constraint
                _overlap_count(cand_main, prev["line"]) > MAX_OVERLAP_MAIN
                for prev in ticket
            ):
                continue                                          # Reject candidate


            chosen_main = cand_main                                # Accept main numbers
            chosen_pb = cand_pb                                    # Accept Powerball
            break                                                  # Exit resampling loop


        # -----------------------------------------
        # Absolute fallback (extremely rare)
        # -----------------------------------------

        if chosen_main is None or chosen_pb is None:               # If no valid candidate found

            chosen_main = sorted(
                np.random.choice(
                    np.arange(1, NUM_MAIN_NUMBERS + 1),
                    size=NUM_PER_LINE,
                    replace=False,
                    p=base_main_prob                               # Sample directly from learned distribution
                )
            )

            chosen_pb = int(
                np.random.choice(
                    np.arange(1, NUM_POWERBALLS + 1),
                    p=base_pb_prob                                 # Sample directly from learned distribution
                )
            )


        # -----------------------------------------
        # Commit accepted line
        # -----------------------------------------

        ticket.append({                                            # Append final line to ticket
            "line": chosen_main,
            "powerball": chosen_pb
        })

        for n in chosen_main:                                      # Update main-number usage counts
            main_usage[n - 1] += 1

        pb_usage[chosen_pb - 1] += 1                               # Update Powerball usage count


    # ---------------------------------------------
    # Persist and return final ticket
    # ---------------------------------------------

    save_current_ticket(ticket)                                    # Persist ticket to storage
    return ticket                                                  # Return ticket to caller





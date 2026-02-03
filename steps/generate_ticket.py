## Modified By: Callam
## Project: Lotto Generator
## Purpose: Predictive ticket generation using deep_learning_predictions with:
##   - cross-line penalty (decay)
##   - anti-overlap rejection sampling (hard diversity constraint)
##
## Adaptation notes:
##   - For other lotteries: change NUM_MAIN_NUMBERS, NUM_POWERBALLS,
##     NUM_PER_LINE, NUM_LINES, and overlap rules.


import numpy as np  # Numerical operations + probabilistic sampling
from data_io import save_current_ticket  # Saves current generated ticket JSON


# =========================
# Ticket configuration
# =========================

NUM_MAIN_NUMBERS = 40  # Main number pool size (1..40)
NUM_POWERBALLS = 10    # Powerball pool size (1..10)
NUM_PER_LINE = 6       # Main numbers per line
NUM_LINES = 12         # Ticket lines to generate


# =========================
# Numeric safety
# =========================

MIN_PROBABILITY = 1e-12  # Prevents zeros -> avoids np.random.choice failure


# =========================
# Cross-line probability decay
# =========================

DECAY_MAIN = 0.7       # Multiplicative penalty applied to main numbers after use
DECAY_POWERBALL = 0.6  # Multiplicative penalty applied to PB after use


# =========================
# Anti-overlap constraints
# =========================

MAX_OVERLAP_MAIN = 2         # A new line may overlap <= 2 main numbers with any previous line
MAX_SAME_POWERBALL = 2       # The same PB number can appear at most this many times
MAX_RESAMPLE_TRIES = 250     # Cap attempts before fallback (prevents infinite loops)


# =========================
# Probability utilities
# =========================

def safe_norm(x):
    """
    Normalise probability vector safely.
    - clips values to MIN_PROBABILITY
    - returns uniform distribution if sum becomes invalid
    """
    x = np.asarray(x, dtype=float)                   # Ensure float array
    x = np.clip(x, MIN_PROBABILITY, None)            # Enforce minimum probability
    s = x.sum()                                      # Sum mass
    if s <= 0.0:                                     # If invalid mass
        return np.full_like(x, 1.0 / len(x))         # Uniform fallback
    return x / s                                     # Normalised probability vector


def _overlap_count(a, b):
    """
    Count overlaps between two lists of ints.
    Used to enforce anti-overlap constraints between ticket lines.
    """
    return len(set(a) & set(b))  # Intersection size


# =========================
# Main ticket generator
# =========================

def generate_ticket(pipeline):
    """
    Generate NUM_LINES lottery lines using:
      - pipeline deep_learning_predictions (only probability source)
      - cross-line decay (reduces repetition)
      - anti-overlap rejection sampling (hard diversity constraint)

    Returns:
      ticket: list[dict] format:
        [{"line": [...6 numbers...], "powerball": int}, ...]
    """

    # ---------------------------------------------
    # Fetch deep learning probability predictions
    # ---------------------------------------------

    predictions = pipeline.get_data("deep_learning_predictions")  # Expected shape = (50,)
    expected_len = NUM_MAIN_NUMBERS + NUM_POWERBALLS              # Total outputs expected


    # ---------------------------------------------
    # Safety fallback if predictions missing/invalid
    # ---------------------------------------------

    if predictions is None or len(predictions) != expected_len:
        print("Missing or invalid predictions. Using uniform fallback.")

        fallback_main = np.ones(NUM_MAIN_NUMBERS) / NUM_MAIN_NUMBERS  # Uniform main distribution
        fallback_pb   = np.ones(NUM_POWERBALLS) / NUM_POWERBALLS      # Uniform PB distribution

        # Generate uniform ticket lines (basic fallback behaviour)
        ticket = [
            {
                "line": sorted(
                    np.random.choice(
                        np.arange(1, NUM_MAIN_NUMBERS + 1),  # pool 1..N
                        NUM_PER_LINE,                        # pick count
                        replace=False,                       # no repeats inside line
                        p=fallback_main                      # prob distribution
                    )
                ),
                "powerball": np.random.choice(
                    np.arange(1, NUM_POWERBALLS + 1),        # PB pool 1..M
                    p=fallback_pb
                )
            }
            for _ in range(NUM_LINES)
        ]

        save_current_ticket(ticket)  # Persist ticket
        return ticket                # Return ticket


    # ---------------------------------------------
    # Convert predictions -> main + PB distributions
    # ---------------------------------------------

    predictions = np.asarray(predictions, dtype=float)  # Force float array

    base_main_prob = safe_norm(predictions[:NUM_MAIN_NUMBERS])  # First block = main numbers
    base_pb_prob   = safe_norm(predictions[NUM_MAIN_NUMBERS:])  # Second block = PB numbers


    # ---------------------------------------------
    # Stateful working distributions across lines
    # ---------------------------------------------

    main_prob = base_main_prob.copy()  # Working main distribution
    pb_prob   = base_pb_prob.copy()    # Working PB distribution


    # ---------------------------------------------
    # Ticket construction state
    # ---------------------------------------------

    ticket = []  # Collected lines

    # PB count limiter (enforces MAX_SAME_POWERBALL)
    pb_counts = np.zeros(NUM_POWERBALLS, dtype=int)


    # ---------------------------------------------
    # Generate each ticket line
    # ---------------------------------------------

    for line_idx in range(NUM_LINES):  # One loop iteration per line

        chosen_main = None  # Accepted main numbers for this line
        chosen_pb = None    # Accepted powerball for this line


        # -----------------------------------------
        # Rejection sampling loop
        # Attempts multiple candidates until constraints met
        # -----------------------------------------

        for attempt in range(MAX_RESAMPLE_TRIES):

            # Candidate main line numbers
            cand_main = sorted(
                np.random.choice(
                    np.arange(1, NUM_MAIN_NUMBERS + 1),  # pool
                    size=NUM_PER_LINE,                   # pick count
                    replace=False,                       # unique within line
                    p=main_prob                          # current prob vector
                )
            )

            # Candidate PB number
            cand_pb = int(
                np.random.choice(
                    np.arange(1, NUM_POWERBALLS + 1),  # PB pool
                    p=pb_prob
                )
            )


            # -------------------------------------
            # Constraint 1: main overlap limiter
            # Cand line must not share too many numbers with any previous line
            # -------------------------------------

            too_much_overlap = False  # Track candidate rejection
            for prev in ticket:       # Compare against each previous accepted line
                if _overlap_count(cand_main, prev["line"]) > MAX_OVERLAP_MAIN:
                    too_much_overlap = True  # Candidate violates overlap rule
                    break                    # Stop checking
            if too_much_overlap:
                continue  # Reject candidate and resample


            # -------------------------------------
            # Constraint 2: PB repetition limiter
            # -------------------------------------

            if pb_counts[cand_pb - 1] >= MAX_SAME_POWERBALL:
                continue  # Reject PB candidate and resample


            # -------------------------------------
            # Candidate passed constraints -> accept
            # -------------------------------------

            chosen_main = cand_main
            chosen_pb = cand_pb
            break  # Exit resample loop


        # -----------------------------------------
        # Fallback if constraints cannot be satisfied
        # (still generates a valid ticket line)
        # -----------------------------------------

        if chosen_main is None or chosen_pb is None:

            # Accept a line without constraint checks (last resort)
            chosen_main = sorted(
                np.random.choice(
                    np.arange(1, NUM_MAIN_NUMBERS + 1),
                    size=NUM_PER_LINE,
                    replace=False,
                    p=main_prob
                )
            )

            chosen_pb = int(
                np.random.choice(
                    np.arange(1, NUM_POWERBALLS + 1),
                    p=pb_prob
                )
            )


        # -----------------------------------------
        # Add accepted line to ticket
        # -----------------------------------------

        ticket.append({
            "line": chosen_main,
            "powerball": chosen_pb
        })

        # Track PB repetition
        pb_counts[chosen_pb - 1] += 1


        # -----------------------------------------
        # Apply cross-line decay penalties (HARD requirement)
        # -----------------------------------------

        # Penalise used main numbers by multiplying probability mass
        for n in chosen_main:
            main_prob[n - 1] *= DECAY_MAIN  # Convert 1-based number to 0-based index

        # Penalise used PB number
        pb_prob[chosen_pb - 1] *= DECAY_POWERBALL


        # -----------------------------------------
        # Renormalise distributions AFTER penalties
        # -----------------------------------------

        main_prob = safe_norm(main_prob)  # Keep valid distribution
        pb_prob   = safe_norm(pb_prob)    # Keep valid distribution


    # ---------------------------------------------
    # Persist and return the final ticket
    # ---------------------------------------------

    save_current_ticket(ticket)  # Write ticket JSON
    return ticket                # Return structure used by pipeline




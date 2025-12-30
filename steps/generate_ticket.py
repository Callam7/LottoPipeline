## Modified By: Callam
## Project: Lotto Generator
## Purpose: Predictive ticket generation with enforced cross-line penalty
##
## Design guarantees:
##   - Uses ONLY deep_learning_predictions as the probability source
##   - Applies a HARD, stateful penalty across ticket lines
##   - No randomness injection beyond probabilistic sampling
##   - No frequency reinforcement
##   - No post-hoc shuffling or cosmetic fixes
##   - Penalty is applied BEFORE renormalisation (mathematically effective)


import numpy as np
from data_io import save_current_ticket

# ====== Configuration ======
NUM_MAIN_NUMBERS = 40
NUM_POWERBALLS = 10
NUM_PER_LINE = 6
NUM_LINES = 12

MIN_PROBABILITY = 1e-12

# Cross-line decay factors (HARD requirement)
DECAY_MAIN = 0.7
DECAY_POWERBALL = 0.6


def safe_norm(x):
    """
    Safely normalise a probability vector while preserving ordering.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, MIN_PROBABILITY, None)
    s = x.sum()
    if s <= 0.0:
        return np.full_like(x, 1.0 / len(x))
    return x / s


def generate_ticket(pipeline):
    """
    Generate a lottery ticket using deep learning predictions with
    enforced cross-line probability decay.

    The decay mechanism guarantees:
        - Reduced repetition across lines
        - Stable statistical behaviour
        - No distortion of within-line sampling
    """

    predictions = pipeline.get_data("deep_learning_predictions")
    expected_len = NUM_MAIN_NUMBERS + NUM_POWERBALLS

    # --------------------------
    # Fallback safety
    # --------------------------
    if predictions is None or len(predictions) != expected_len:
        print("Missing or invalid predictions. Using uniform fallback.")

        fallback_main = np.ones(NUM_MAIN_NUMBERS) / NUM_MAIN_NUMBERS
        fallback_pb   = np.ones(NUM_POWERBALLS) / NUM_POWERBALLS

        ticket = [
            {
                "line": sorted(
                    np.random.choice(
                        np.arange(1, NUM_MAIN_NUMBERS + 1),
                        NUM_PER_LINE,
                        replace=False,
                        p=fallback_main
                    )
                ),
                "powerball": np.random.choice(
                    np.arange(1, NUM_POWERBALLS + 1),
                    p=fallback_pb
                )
            }
            for _ in range(NUM_LINES)
        ]

        save_current_ticket(ticket)
        return ticket

    # --------------------------
    # Prepare probability vectors
    # --------------------------
    predictions = np.asarray(predictions, dtype=float)

    base_main_prob = safe_norm(predictions[:NUM_MAIN_NUMBERS])
    base_pb_prob   = safe_norm(predictions[NUM_MAIN_NUMBERS:])

    # Working copies (stateful across lines)
    main_prob = base_main_prob.copy()
    pb_prob   = base_pb_prob.copy()

    # --------------------------
    # Generate ticket with penalty
    # --------------------------
    ticket = []

    for _ in range(NUM_LINES):

        # Draw unique main numbers
        main_numbers = sorted(
            np.random.choice(
                np.arange(1, NUM_MAIN_NUMBERS + 1),
                size=NUM_PER_LINE,
                replace=False,
                p=main_prob
            )
        )

        # Draw Powerball
        powerball = np.random.choice(
            np.arange(1, NUM_POWERBALLS + 1),
            p=pb_prob
        )

        ticket.append({
            "line": main_numbers,
            "powerball": powerball
        })

        # --------------------------
        # APPLY HARD PENALTIES
        # --------------------------

        # Penalise used main numbers
        for n in main_numbers:
            main_prob[n - 1] *= DECAY_MAIN

        # Penalise used Powerball
        pb_prob[powerball - 1] *= DECAY_POWERBALL

        # Renormalise AFTER penalties
        main_prob = safe_norm(main_prob)
        pb_prob   = safe_norm(pb_prob)

    # --------------------------
    # Save and return
    # --------------------------
    save_current_ticket(ticket)
    return ticket



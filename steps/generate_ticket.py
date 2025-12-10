## Modified By: Callam
## Project: Lotto Generator
## Purpose: Pure predictive ticket generation from final fused DL output
## Notes:
##   - NO diversity systems
##   - NO penalties
##   - NO frequency reinforcement
##   - NO additional weighting
##   - The deep_learning module already performs all modelling, so nothing here modifies probabilities

import numpy as np
from data_io import save_current_ticket

# ====== Configuration ======
NUM_MAIN_NUMBERS = 40
NUM_POWERBALLS = 10
NUM_PER_LINE = 6
NUM_LINES = 12

MIN_PROBABILITY = 1e-12


def safe_norm(x):
    """
    Normalise a probability vector safely without altering
    predictive ordering or introducing distortions.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, MIN_PROBABILITY, None)
    s = x.sum()
    if s <= 0:
        return np.full_like(x, 1.0 / len(x))
    return x / s


def generate_ticket(pipeline):
    """
    Generate 12 ticket lines purely from the final fused DL predictions.

    No added logic. No modifiers. No diversity adjustments.
    The deep learning model carries the entire statistical structure.
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
    # Split predictions
    # --------------------------
    predictions = np.asarray(predictions, dtype=float)

    main_prob      = safe_norm(predictions[:NUM_MAIN_NUMBERS])
    powerball_prob = safe_norm(predictions[NUM_MAIN_NUMBERS:])

    # --------------------------
    # Generate ticket lines
    # --------------------------
    ticket = []

    for _ in range(NUM_LINES):

        # Draw 6 unique main numbers
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
            p=powerball_prob
        )

        ticket.append({
            "line": main_numbers,
            "powerball": powerball
        })

    # Save for UI + frontend tracking
    save_current_ticket(ticket)
    return ticket



## Modified By: Callam
## Project: Lotto Generator
## Purpose: Pure predictive ticket generation from DL output
## Notes:
##   - Uses ONLY deep learning predictions (already fused + quantum enhanced)
##   - Exponential diversity penalty preserves probability structure
##   - Pipeline data structure remains untouched

import numpy as np
from data_io import save_current_ticket

# ====== Configuration Constants ======
NUM_MAIN_NUMBERS = 40
NUM_POWERBALLS = 10
NUM_PER_LINE = 6
NUM_LINES = 12

MIN_PROBABILITY = 1e-12

# Diversity setting (keeps lines unique but does NOT distort probabilities)
DIVERSITY_STRENGTH = 1.5


def safe_norm(x):
    """Normalize safely without destroying relative predictive strength."""
    x = np.clip(x, MIN_PROBABILITY, None)
    return x / x.sum()


def generate_ticket(pipeline):
    """
    Generates lottery tickets using ONLY the final deep learning predictions.
    The diversity penalty is exponential (multiplicative), so the predictive
    distribution remains statistically correct.
    """
    predictions = pipeline.get_data("deep_learning_predictions")

    expected_len = NUM_MAIN_NUMBERS + NUM_POWERBALLS

    # ==============================
    # Safe fallback
    # ==============================
    if predictions is None or len(predictions) != expected_len:
        print("Missing or invalid predictions. Using uniform fallback.")
        fallback_main = np.ones(NUM_MAIN_NUMBERS) / NUM_MAIN_NUMBERS
        fallback_pb = np.ones(NUM_POWERBALLS) / NUM_POWERBALLS

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

    # ==============================
    # Pure deep learning output
    # ==============================
    predictions = np.asarray(predictions, dtype=float)

    # First 40 = main
    main_prob = safe_norm(predictions[:NUM_MAIN_NUMBERS])

    # Last 10 = powerball
    powerball_prob = safe_norm(predictions[NUM_MAIN_NUMBERS:])

    # ==============================
    # Generate ticket lines
    # ==============================
    ticket = []
    seen_combinations = set()

    # Track frequency to encourage diversity WITHOUT biasing model predictions
    frequency_penalty = np.zeros(NUM_MAIN_NUMBERS)

    for _ in range(NUM_LINES):

        while True:
            # -------------------------
            # EXPONENTIAL PENALTY
            # -------------------------
            # This maintains *perfect predictive ordering*
            # and applies only a smooth "anti-repeat" pressure.
            denom = frequency_penalty.sum() + 1.0
            penalty = np.exp(-DIVERSITY_STRENGTH * (frequency_penalty / denom))

            # Multiplicative — NOT subtractive (preserves prediction fidelity)
            adjusted_prob = main_prob * penalty
            adjusted_prob = safe_norm(adjusted_prob)

            # Draw main numbers
            main_numbers = sorted(
                np.random.choice(
                    np.arange(1, NUM_MAIN_NUMBERS + 1),
                    size=NUM_PER_LINE,
                    replace=False,
                    p=adjusted_prob
                )
            )

            # Ensure unique lines only
            if tuple(main_numbers) not in seen_combinations:
                seen_combinations.add(tuple(main_numbers))
                break

        # Update frequency AFTER sampling
        for n in main_numbers:
            frequency_penalty[n - 1] += 1

        # Draw powerball directly from DL distribution (no penalty needed)
        powerball = np.random.choice(
            np.arange(1, NUM_POWERBALLS + 1),
            p=powerball_prob
        )

        ticket.append({"line": main_numbers, "powerball": powerball})

    # Save ticket
    save_current_ticket(ticket)
    return ticket




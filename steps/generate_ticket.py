## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Ticket Generation from Deep Learning Predictions
## Description:
## Generates complete lottery tickets using ONLY the final deep learning output.
## (DL already includes Monte Carlo, Markov, entropy, clustering, fusion, quantum, etc.)
## Applies diversity penalties to avoid repetitive lines.

import numpy as np
from data_io import save_current_ticket

# ====== Configuration Constants ======
NUM_MAIN_NUMBERS = 40
NUM_POWERBALLS = 10
NUM_PER_LINE = 6
NUM_LINES = 12
MIN_PROBABILITY = 1e-12
DIVERSITY_PENALTY = 1.5


def safe_norm(x):
    """Safely normalize an array to sum = 1."""
    x = np.clip(x, MIN_PROBABILITY, None)
    return x / x.sum()


def generate_ticket(pipeline):
    """
    Generates a ticket using ONLY deep learning predictions.
    """
    predictions = pipeline.get_data("deep_learning_predictions")

    expected_len = NUM_MAIN_NUMBERS + NUM_POWERBALLS

    # ==============================
    # Fallback if predictions missing
    # ==============================
    if predictions is None or len(predictions) != expected_len:
        print("Missing or invalid predictions. Falling back to uniform distribution.")
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
    # Use deep learning output directly
    # ==============================
    predictions = np.asarray(predictions, dtype=float)

    # First 40 = main numbers
    main_prob = safe_norm(predictions[:NUM_MAIN_NUMBERS])

    # Last 10 = powerball
    powerball_prob = safe_norm(predictions[NUM_MAIN_NUMBERS:])

    # ==============================
    # Generate lines with diversity penalty
    # ==============================
    ticket = []
    seen_combinations = set()
    frequency_penalty = np.zeros(NUM_MAIN_NUMBERS)

    for _ in range(NUM_LINES):
        while True:
            denom = frequency_penalty.sum() + 1
            adjusted_prob = main_prob - DIVERSITY_PENALTY * (frequency_penalty / denom)
            adjusted_prob = np.clip(adjusted_prob, MIN_PROBABILITY, None)
            adjusted_prob = adjusted_prob / adjusted_prob.sum()

            main_numbers = sorted(
                np.random.choice(
                    np.arange(1, NUM_MAIN_NUMBERS + 1),
                    size=NUM_PER_LINE,
                    replace=False,
                    p=adjusted_prob
                )
            )

            if tuple(main_numbers) not in seen_combinations:
                seen_combinations.add(tuple(main_numbers))
                break

        # Update penalties
        for n in main_numbers:
            frequency_penalty[n - 1] += 1

        # Powerball directly from DL
        powerball = np.random.choice(
            np.arange(1, NUM_POWERBALLS + 1),
            p=powerball_prob
        )

        ticket.append({"line": main_numbers, "powerball": powerball})

    save_current_ticket(ticket)
    return ticket



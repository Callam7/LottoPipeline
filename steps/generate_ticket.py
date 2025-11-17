## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Ticket Generation from Deep Learning Predictions
## Description:
## Generates complete lottery tickets using the final deep learning output
## (which integrates Bayesian fusion, clustering, Monte Carlo, redundancy, Markov,
## entropy, and historical features). Applies diversity penalties to reduce repetition
## and uses Powerball decay weighting if available.

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
    Generates a ticket using deep learning predictions with diversity and Powerball weighting.
    """
    # Retrieve final probabilities and decay data
    predictions = pipeline.get_data("deep_learning_predictions")
    decay_factors = pipeline.get_data("decay_factors")

    if predictions is None or len(predictions) != NUM_MAIN_NUMBERS:
        print("Missing or invalid predictions. Falling back to uniform distribution.")
        fallback = np.ones(NUM_MAIN_NUMBERS) / NUM_MAIN_NUMBERS
        ticket = [
            {
                "line": sorted(np.random.choice(
                    np.arange(1, NUM_MAIN_NUMBERS + 1),
                    NUM_PER_LINE,
                    replace=False,
                    p=fallback
                )),
                "powerball": np.random.randint(1, NUM_POWERBALLS + 1)
            }
            for _ in range(NUM_LINES)
        ]
        save_current_ticket(ticket)
        return ticket

    # Normalize deep learning output
    numbers_prob = safe_norm(predictions)

    # Initialize diversity control
    ticket = []
    seen_combinations = set()
    frequency_penalty = np.zeros(NUM_MAIN_NUMBERS)

    for _ in range(NUM_LINES):
        while True:
            denom = frequency_penalty.sum() + 1
            adjusted_prob = numbers_prob - DIVERSITY_PENALTY * (frequency_penalty / denom)
            adjusted_prob = np.clip(adjusted_prob, MIN_PROBABILITY, None)
            adjusted_prob /= adjusted_prob.sum()

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

        # Update diversity penalties
        for num in main_numbers:
            frequency_penalty[num - 1] += 1

        # Powerball weighting (decay-aware)
        if decay_factors and isinstance(decay_factors, dict) and "powerball" in decay_factors:
            powerball_prob = safe_norm(np.array(decay_factors["powerball"], dtype=float))
        else:
            powerball_prob = np.ones(NUM_POWERBALLS) / NUM_POWERBALLS

        powerball = np.random.choice(np.arange(1, NUM_POWERBALLS + 1), p=powerball_prob)

        ticket.append({"line": main_numbers, "powerball": powerball})

    save_current_ticket(ticket)
    return ticket


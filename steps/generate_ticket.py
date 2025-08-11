## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: Ticket Generation from Deep Learning Predictions
## Description:
## This file generates a complete set of lottery ticket lines using only the final
## deep learning predictions from the pipeline. These predictions already incorporate
## all upstream statistical and feature-engineered signals (historical, decay, Monte Carlo,
## clustering, redundancy, Markov, etc.).
##
## The ticket generator applies a diversity penalty to ensure variety across multiple
## generated lines. Powerball selection is weighted using the historical decay factors
## for Powerball if available; otherwise, it defaults to a uniform distribution.
##
## The final ticket is saved to 'current_ticket.json'.

import numpy as np
from data_io import save_current_ticket

def generate_ticket(pipeline):
    """
    Generates a ticket using the final deep learning predictions from the pipeline.
    Diversity penalties are applied to avoid repetitive lines.
    """

    # Retrieve deep learning predictions
    predictions = pipeline.get_data("deep_learning_predictions")
    decay_factors = pipeline.get_data("decay_factors")  # For Powerball weighting

    # Safety fallback if predictions are missing
    if predictions is None:
        print("Missing deep learning predictions. Falling back to uniform.")
        fallback = np.ones(40) / 40
        ticket = [
            {"line": sorted(np.random.choice(np.arange(1, 41), 6, replace=False, p=fallback)),
             "powerball": np.random.randint(1, 11)}
            for _ in range(12)
        ]
        save_current_ticket(ticket)
        return ticket

    # Normalize predictions safely
    def safe_norm(x):
        x = np.clip(x, 1e-12, None)
        return x / x.sum()

    numbers_prob = safe_norm(predictions)

    # Initialize diversity mechanisms
    ticket = []
    seen_combinations = set()
    frequency_penalty = np.zeros(40)

    for _ in range(12):
        while True:
            penalty_factor = 1.5
            denom = frequency_penalty.sum() + 1
            adjusted_prob = numbers_prob - penalty_factor * (frequency_penalty / denom)
            adjusted_prob = np.clip(adjusted_prob, 1e-12, None)
            adjusted_prob /= adjusted_prob.sum()

            main_numbers = sorted(
                np.random.choice(np.arange(1, 41), size=6, replace=False, p=adjusted_prob)
            )

            if tuple(main_numbers) not in seen_combinations:
                seen_combinations.add(tuple(main_numbers))
                break

        # Update penalty tracking
        for num in main_numbers:
            frequency_penalty[num - 1] += 1

        # Powerball selection
        if decay_factors and "powerball" in decay_factors:
            powerball_prob = safe_norm(decay_factors["powerball"])
        else:
            powerball_prob = np.ones(10) / 10

        powerball = np.random.choice(np.arange(1, 11), p=powerball_prob)

        ticket.append({"line": main_numbers, "powerball": powerball})

    # Save and return generated ticket
    save_current_ticket(ticket)
    return ticket


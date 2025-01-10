import numpy as np
from data_io import save_current_ticket

def generate_ticket(pipeline):
    """
    Final step that combines the neural net's predictions with the decay distribution
    (alpha=0.5 each), and then generates 12 lines of picks with a penalty-based
    diversity mechanism. Saves the ticket to 'current_ticket.json' and returns it.
    """
    decay_factors = pipeline.get_data("decay_factors")
    predictions = pipeline.get_data("deep_learning_predictions")

    # Weighted combo
    alpha = 0.5
    numbers_prob = alpha * predictions + (1 - alpha) * decay_factors["numbers"]
    numbers_prob = np.clip(numbers_prob, 1e-12, None)
    numbers_prob /= numbers_prob.sum()

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

        for num in main_numbers:
            frequency_penalty[num - 1] += 1

        powerball_prob = decay_factors["powerball"]
        powerball_prob = np.clip(powerball_prob, 1e-12, None)
        powerball_prob /= powerball_prob.sum()

        powerball = np.random.choice(np.arange(1, 11), p=powerball_prob)
        ticket.append({"line": main_numbers, "powerball": powerball})

    save_current_ticket(ticket)
    return ticket

import numpy as np
from data_io import save_current_ticket

def generate_ticket(pipeline):
    """
    Generates a ticket using all major pipeline outputs:
    - Deep learning predictions (primary weight)
    - Monte Carlo simulation distribution
    - Decay factor probabilities
    - Clustering centroids

    Applies a diversity penalty mechanism to generate 12 unique lines,
    and saves the result to 'current_ticket.json'.
    """
    # Retrieve pipeline outputs
    predictions = pipeline.get_data("deep_learning_predictions")
    monte_carlo = pipeline.get_data("monte_carlo")
    decay_factors = pipeline.get_data("decay_factors")
    clusters = pipeline.get_data("clusters")
    centroids = pipeline.get_data("centroids")

    # Safety fallback
    if any(x is None for x in [predictions, monte_carlo, decay_factors, clusters, centroids]):
        print("Missing data for advanced generation. Falling back to uniform.")
        fallback = np.ones(40) / 40
        ticket = [{"line": sorted(np.random.choice(np.arange(1, 41), 6, replace=False, p=fallback)),
                   "powerball": np.random.randint(1, 11)} for _ in range(12)]
        save_current_ticket(ticket)
        return ticket

    # Normalize components to be safe
    def safe_norm(x):
        x = np.clip(x, 1e-12, None)
        return x / x.sum()

    predictions = safe_norm(predictions)
    monte_carlo = safe_norm(monte_carlo)
    decay = safe_norm(decay_factors["numbers"])
    cluster_adjust = safe_norm(centroids[clusters])

    # Weighted combination
    numbers_prob = (
        0.65 * predictions +
        0.15 * monte_carlo +
        0.10 * decay +
        0.10 * cluster_adjust
    )
    numbers_prob = safe_norm(numbers_prob)

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

        for num in main_numbers:
            frequency_penalty[num - 1] += 1

        # Powerball from decay or fallback
        powerball_prob = decay_factors.get("powerball", np.ones(10) / 10)
        powerball_prob = safe_norm(powerball_prob)
        powerball = np.random.choice(np.arange(1, 11), p=powerball_prob)

        ticket.append({"line": main_numbers, "powerball": powerball})

    save_current_ticket(ticket)
    return ticket


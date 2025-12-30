## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Compute Shannon Entropy Features (Shape 50)
##
## Description:
## Computes Shannon entropy contribution for each lottery number:
##     H_i = -p_i * log2(p_i)
## using the Bayesian-fused unified probability distribution (shape 50).
## Produces a mathematically correct, normalized entropy feature vector.

import numpy as np
import logging

# Number of standard lottery balls
NUM_MAIN = 40

# Number of powerball-style balls
NUM_POWERBALL = 10

# Total symbols in the probability distribution
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL  # = 50

# Configure logging output format and verbosity
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def shannon_entropy_features(pipeline):
    """
    Compute Shannon entropy contributions for each number (1–50).

    Uses the Bayesian fusion probability distribution as input.
    Produces a normalized entropy feature vector of shape (50,).

    NOTE:
    - This computes per-symbol entropy terms, not total entropy.
    - Final normalization makes this suitable for ML feature use.
    """

    # Retrieve the Bayesian-fused probability distribution
    fusion = pipeline.get_data("bayesian_fusion")

    # If fusion data is missing or malformed, fall back to uniform distribution
    if fusion is None or len(fusion) != NUM_TOTAL:
        logging.warning(
            "Fusion distribution missing — fallback to uniform entropy features."
        )

        # Create a uniform probability distribution over 50 symbols
        uniform = np.ones(NUM_TOTAL) / NUM_TOTAL

        # Compute Shannon entropy contribution per symbol:
        # H_i = -p_i * log2(p_i)
        entropy_terms = -uniform * np.log2(uniform)

        # Normalize entropy vector so it sums to 1
        entropy_terms /= entropy_terms.sum()

        # Store result in the pipeline
        pipeline.add_data("entropy_features", entropy_terms)
        return

    # Convert fusion output to a NumPy float array
    p = np.array(fusion, dtype=float)

    # Normalize probabilities to ensure sum(p) == 1
    # (fallback to 1.0 prevents division-by-zero)
    p /= p.sum() or 1.0

    # Compute Shannon entropy contributions safely
    with np.errstate(divide='ignore', invalid='ignore'):
        # clip() prevents log2(0) and keeps values numerically stable
        entropy_terms = -p * np.log2(p.clip(1e-12, 1.0))

    # Remove any negative numerical artifacts
    entropy_terms = np.clip(entropy_terms, 0.0, None)

    # Normalize entropy contributions into a stable feature vector
    entropy_terms /= entropy_terms.sum() or 1.0

    # Store entropy features in the pipeline
    pipeline.add_data("entropy_features", entropy_terms)

    logging.info("Shannon entropy features generated successfully.")




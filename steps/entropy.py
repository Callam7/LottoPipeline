## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Compute Shannon Entropy Features (Shape 50)
## Description:
## Computes Shannon entropy contribution for each lottery number:
##     H_i = -p_i * log2(p_i)
## using the Bayesian-fused unified probability distribution (shape 50).
## Produces a mathematically correct, normalized entropy vector.

import numpy as np
import logging

NUM_MAIN = 40
NUM_POWERBALL = 10
NUM_TOTAL = NUM_MAIN + NUM_POWERBALL

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def shannon_entropy_features(pipeline):
    """
    Compute Shannon entropy contributions for each number (1–50).
    Uses the Bayesian fusion probability distribution as the base.
    Produces a mathematically correct entropy vector of shape (50,).
    """

    fusion = pipeline.get_data("bayesian_fusion")

    if fusion is None or len(fusion) != NUM_TOTAL:
        logging.warning("Fusion distribution missing — fallback to uniform entropy features.")
        uniform = np.ones(NUM_TOTAL) / NUM_TOTAL
        entropy_terms = -uniform * np.log2(uniform)
        entropy_terms /= entropy_terms.sum()
        pipeline.add_data("entropy_features", entropy_terms)
        return

    # Convert to probability vector
    p = np.array(fusion, dtype=float)
    p /= p.sum() or 1.0

    # Shannon entropy contribution per symbol
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy_terms = -p * np.log2(p.clip(1e-12, 1.0))

    # Normalize to shape-50 feature
    entropy_terms = np.clip(entropy_terms, 0.0, None)
    entropy_terms /= entropy_terms.sum() or 1.0

    pipeline.add_data("entropy_features", entropy_terms)
    logging.info("Shannon entropy features generated successfully.")



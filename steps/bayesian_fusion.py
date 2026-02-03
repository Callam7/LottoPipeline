## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Mechanics Estimation + Bayesian Fusion
## Description:
## - Estimate small, data-driven mechanics biases (Dirichlet posterior on observed counts).
## - Run a quick chi-square goodness-of-fit to decide if the mechanics signal is meaningful.
## - Combine frequency, decay, mechanics using log-space fusion (avoids overweighting).
## - Normalize posterior to sum to 1 (probability distribution).
## - Also provide max-normalized version for deep learning feature stacking.
## - No injected bias. If mechanics signal is not significant, collapse mechanics -> uniform.

import numpy as np                     # Numerical operations for probability math
import logging                         # Logging for diagnostics and status messages

logging.basicConfig(                   # Configure logging format and verbosity
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Chi-square critical for df=39, alpha=0.05
_CHI2_CRIT_DF39_0P05 = 55.758          # Threshold to decide if observed bias differs from uniform

# Constants
NUM_MAIN = 40                          # Main number count (1–40)
NUM_POWERBALL = 10                     # Powerball number count (1–10)
TOTAL_NUMBERS = NUM_MAIN + NUM_POWERBALL  # Total probability vector size (50)


def _estimate_mechanics_dirichlet_from_history(historical_data, alpha=1.0):
    """
    Estimate mechanics vector (length 40) from historical draws using a Dirichlet posterior.
    Only for main numbers. Returns:
    - mechanics_vector (shape 40)
    - chi2_stat
    - total_counts
    """
    counts = np.zeros(NUM_MAIN, dtype=float)   # Store observed counts per main number
    total_counts = 0                          # Track total number of main balls observed

    for draw in historical_data:              # Loop over every historical draw
        nums = draw.get("numbers", [])        # Get list of main numbers drawn
        for n in nums:
            if 1 <= n <= NUM_MAIN:            # Validate range
                counts[n - 1] += 1            # Increment count for that number
                total_counts += 1             # Increase total observation count

    denom = total_counts + NUM_MAIN * alpha   # Dirichlet denominator (adds smoothing)
    if denom <= 0:                           # If no data, fallback to uniform
        mechanics = np.ones(NUM_MAIN) / NUM_MAIN
        return mechanics, 0.0, 0

    mechanics = (counts + alpha) / denom     # Dirichlet posterior mean estimate

    expected = total_counts / NUM_MAIN if total_counts > 0 else 1.0  # Expected count under uniform
    with np.errstate(divide='ignore', invalid='ignore'):              # Suppress divide warnings
        chi2 = np.nansum(((counts - expected) ** 2) / (expected + 1e-12))  # Chi-square statistic

    return mechanics, float(chi2), int(total_counts)  # Return posterior, chi2 value, and sample size


def bayesian_fusion_with_mechanics(pipeline, alpha=1.0, chi2_threshold=_CHI2_CRIT_DF39_0P05,
                                   use_mechanics_if_significant=True, verbose=False,
                                   weights=(1.0, 1.0, 1.0)):
    """
    Combine frequency, decay, and mechanics into normalized posterior of shape 50.
    Main numbers: uses mechanics + frequency + decay.
    Powerball: uniform mechanics (or optionally separate mechanics in the future).
    """
    # --- Retrieve pipeline data ---
    freq = pipeline.get_data("number_frequency_combined")  # Get historical frequency probabilities
    if freq is None or len(freq) != TOTAL_NUMBERS:         # Validate shape
        logging.warning("Frequency missing/invalid; using uniform.")
        freq = np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS
    freq = np.array(freq, dtype=float)

    decay = pipeline.get_data("decay_factors")             # Get recency-weighted probabilities
    if decay is None or len(decay) != TOTAL_NUMBERS:
        logging.warning("Decay missing/invalid; using uniform.")
        decay = np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS
    decay = np.array(decay, dtype=float)

    # --- Split main and Powerball ---
    freq_main = freq[:NUM_MAIN]                # First 40 entries = main numbers
    freq_powerball = freq[NUM_MAIN:]           # Last 10 entries = Powerball
    decay_main = decay[:NUM_MAIN]
    decay_powerball = decay[NUM_MAIN:]

    # --- Mechanics estimation for main numbers ---
    historical = pipeline.get_data("historical_data") or []
    mechanics_vec, chi2_stat, total_obs = _estimate_mechanics_dirichlet_from_history(historical, alpha=alpha)

    mechanics_used = mechanics_vec.copy()      # Copy so we can override if insignificant
    mechanics_is_uniform = False

    if use_mechanics_if_significant:           # Only apply mechanics if statistically meaningful
        if total_obs == 0 or chi2_stat < chi2_threshold:
            mechanics_used = np.ones(NUM_MAIN) / NUM_MAIN  # Collapse to uniform if weak signal
            mechanics_is_uniform = True
            if verbose:
                logging.info(f"Mechanics not significant (chi2={chi2_stat:.2f}, n={total_obs}). Using uniform mechanics.")
        else:
            if verbose:
                logging.info(f"Mechanics significant (chi2={chi2_stat:.2f}, n={total_obs}). Using estimated mechanics.")

    # Powerball mechanics: uniform (no physical bias modeling yet)
    mechanics_powerball = np.ones(NUM_POWERBALL) / NUM_POWERBALL

    # --- Clip negatives ---
    freq_main = np.clip(freq_main, 0.0, None)              # Ensure valid probabilities
    freq_powerball = np.clip(freq_powerball, 0.0, None)
    decay_main = np.clip(decay_main, 0.0, None)
    decay_powerball = np.clip(decay_powerball, 0.0, None)
    mechanics_used = np.clip(mechanics_used, 0.0, None)

    # --- Log-space fusion ---
    w_f, w_d, w_m = weights                                # Weighting for each signal source
    eps = 1e-12                                            # Prevent log(0)

    posterior_main = np.exp(                               # Multiply probabilities in log-space
        w_f * np.log(freq_main + eps) +
        w_d * np.log(decay_main + eps) +
        w_m * np.log(mechanics_used + eps)
    )

    posterior_powerball = np.exp(
        w_f * np.log(freq_powerball + eps) +
        w_d * np.log(decay_powerball + eps) +
        w_m * np.log(mechanics_powerball + eps)
    )

    # Concatenate into shape 50
    posterior = np.concatenate([posterior_main, posterior_powerball])

    # Normalize total posterior
    total = posterior.sum()
    if total > 0:
        posterior /= total                                # Convert to probability distribution
    else:
        logging.warning("Posterior sum is zero after fusion. Falling back to uniform distribution.")
        posterior = np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS

    # Max-normalize for DL features
    posterior_norm = posterior / max(posterior.max(), 1e-12)  # Scale max value to 1 for feature input

    # Store results
    pipeline.add_data("bayesian_fusion", posterior)            # True probability distribution
    pipeline.add_data("bayesian_fusion_norm", posterior_norm)  # Scaled feature version
    pipeline.add_data("mechanics_vector", mechanics_vec)       # Raw mechanics estimate
    pipeline.add_data("mechanics_chi2", chi2_stat)             # Chi-square statistic
    pipeline.add_data("mechanics_total_obs", total_obs)        # Number of observed balls
    pipeline.add_data("mechanics_used_is_uniform", bool(mechanics_is_uniform))  # Whether mechanics was suppressed

    logging.info("Bayesian fusion stored successfully")
    return posterior


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

import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Chi-square critical for df=39, alpha=0.05
_CHI2_CRIT_DF39_0P05 = 55.758

# Constants
NUM_MAIN = 40
NUM_POWERBALL = 10
TOTAL_NUMBERS = NUM_MAIN + NUM_POWERBALL  # 50


def _estimate_mechanics_dirichlet_from_history(historical_data, alpha=1.0):
    """
    Estimate mechanics vector (length 40) from historical draws using a Dirichlet posterior.
    Only for main numbers. Returns:
    - mechanics_vector (shape 40)
    - chi2_stat
    - total_counts
    """
    counts = np.zeros(NUM_MAIN, dtype=float)
    total_counts = 0

    for draw in historical_data:
        nums = draw.get("numbers", [])
        for n in nums:
            if 1 <= n <= NUM_MAIN:
                counts[n - 1] += 1
                total_counts += 1

    denom = total_counts + NUM_MAIN * alpha
    if denom <= 0:
        mechanics = np.ones(NUM_MAIN) / NUM_MAIN
        return mechanics, 0.0, 0

    mechanics = (counts + alpha) / denom

    expected = total_counts / NUM_MAIN if total_counts > 0 else 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum(((counts - expected) ** 2) / (expected + 1e-12))

    return mechanics, float(chi2), int(total_counts)


def bayesian_fusion_with_mechanics(pipeline, alpha=1.0, chi2_threshold=_CHI2_CRIT_DF39_0P05,
                                   use_mechanics_if_significant=True, verbose=False,
                                   weights=(1.0, 1.0, 1.0)):
    """
    Combine frequency, decay, and mechanics into normalized posterior of shape 50.
    Main numbers: uses mechanics + frequency + decay.
    Powerball: uniform mechanics (or optionally separate mechanics in the future).
    """
    # --- Retrieve pipeline data ---
    freq = pipeline.get_data("number_frequency_combined")
    if freq is None or len(freq) != TOTAL_NUMBERS:
        logging.warning("Frequency missing/invalid; using uniform.")
        freq = np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS
    freq = np.array(freq, dtype=float)

    decay = pipeline.get_data("decay_factors")
    if decay is None or len(decay) != TOTAL_NUMBERS:
        logging.warning("Decay missing/invalid; using uniform.")
        decay = np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS
    decay = np.array(decay, dtype=float)

    # --- Split main and Powerball ---
    freq_main = freq[:NUM_MAIN]
    freq_powerball = freq[NUM_MAIN:]
    decay_main = decay[:NUM_MAIN]
    decay_powerball = decay[NUM_MAIN:]

    # --- Mechanics estimation for main numbers ---
    historical = pipeline.get_data("historical_data") or []
    mechanics_vec, chi2_stat, total_obs = _estimate_mechanics_dirichlet_from_history(historical, alpha=alpha)

    mechanics_used = mechanics_vec.copy()
    mechanics_is_uniform = False
    if use_mechanics_if_significant:
        if total_obs == 0 or chi2_stat < chi2_threshold:
            mechanics_used = np.ones(NUM_MAIN) / NUM_MAIN
            mechanics_is_uniform = True
            if verbose:
                logging.info(f"Mechanics not significant (chi2={chi2_stat:.2f}, n={total_obs}). Using uniform mechanics.")
        else:
            if verbose:
                logging.info(f"Mechanics significant (chi2={chi2_stat:.2f}, n={total_obs}). Using estimated mechanics.")

    # Powerball mechanics: uniform
    mechanics_powerball = np.ones(NUM_POWERBALL) / NUM_POWERBALL

    # --- Clip negatives ---
    freq_main = np.clip(freq_main, 0.0, None)
    freq_powerball = np.clip(freq_powerball, 0.0, None)
    decay_main = np.clip(decay_main, 0.0, None)
    decay_powerball = np.clip(decay_powerball, 0.0, None)
    mechanics_used = np.clip(mechanics_used, 0.0, None)

    # --- Log-space fusion ---
    w_f, w_d, w_m = weights
    eps = 1e-12

    posterior_main = np.exp(
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
        posterior /= total
    else:
        logging.warning("Posterior sum is zero after fusion. Falling back to uniform distribution.")
        posterior = np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS

    # Max-normalize for DL features
    posterior_norm = posterior / max(posterior.max(), 1e-12)

    # Store results
    pipeline.add_data("bayesian_fusion", posterior)
    pipeline.add_data("bayesian_fusion_norm", posterior_norm)
    pipeline.add_data("mechanics_vector", mechanics_vec)
    pipeline.add_data("mechanics_chi2", chi2_stat)
    pipeline.add_data("mechanics_total_obs", total_obs)
    pipeline.add_data("mechanics_used_is_uniform", bool(mechanics_is_uniform))

    logging.info("Bayesian fusion stored successfully")
    return posterior


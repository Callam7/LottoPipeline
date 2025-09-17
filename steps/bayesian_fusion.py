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
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Chi-square critical for df=39, alpha=0.05 (heuristic test)
_CHI2_CRIT_DF39_0P05 = 55.758  # approximate; used only as a practical threshold


def _estimate_mechanics_dirichlet_from_history(historical_data, alpha=1.0):
    """
    Estimate mechanics vector (length 40) from historical draws using a Dirichlet posterior.
    - Counts how many times each ball (1..40) appeared in the provided draws.
    - Posterior mean = (counts + alpha) / (total_counts + 40*alpha).
    Returns:
    - mechanics_vector (np.array, shape 40) of posterior means (sums to 1)
    - chi2_stat (float): chi-square statistic vs uniform expected counts
    - total_counts (int): total ball observations used (for diagnostics)
    Notes:
    - This is a data-driven estimator. If counts are essentially uniform (chi2 small),
      the calling code may choose to treat mechanics_vector as uniform.
    """
    counts = np.zeros(40, dtype=float)
    total_counts = 0

    # accumulate counts: each draw contributes 6 main numbers
    for draw in historical_data:
        nums = draw.get("numbers", [])
        for n in nums:
            if 1 <= n <= 40:
                counts[n - 1] += 1
                total_counts += 1

    # Posterior mean under Dirichlet(alpha) prior
    denom = total_counts + 40.0 * alpha
    if denom <= 0:
        mechanics = np.ones(40) / 40.0
        return mechanics, 0.0, 0

    mechanics = (counts + alpha) / denom

    # Quick chi-square GOF vs uniform expectation
    expected = total_counts / 40.0 if total_counts > 0 else 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum(((counts - expected) ** 2) / (expected + 1e-12))

    return mechanics, float(chi2), int(total_counts)


def bayesian_fusion_with_mechanics(pipeline, alpha=1.0, chi2_threshold=_CHI2_CRIT_DF39_0P05,
                                   use_mechanics_if_significant=True, verbose=False,
                                   weights=(1.0, 1.0, 1.0)):
    """
    Combine frequency, decay, and mechanics estimate into a normalized posterior (shape 40).
    Fusion is done in log-space to avoid overweighting any single component.
    - frequency: pipeline["number_frequency"] (shape 40, sums to 1 expected)
    - decay: pipeline["decay_factors"]["numbers"] (shape 40, sums to 1 expected)
    - mechanics: estimated from historical data, collapsed to uniform if chi2 not significant
    """
    # Defensive retrieval
    freq = pipeline.get_data("number_frequency")
    if freq is None or len(freq) != 40:
        logging.warning("Frequency missing/invalid; using uniform.")
        freq = np.ones(40) / 40.0
    else:
        freq = np.array(freq, dtype=float)

    decay_container = pipeline.get_data("decay_factors")
    if not decay_container or "numbers" not in decay_container:
        logging.warning("Decay missing/invalid; using uniform.")
        decay = np.ones(40) / 40.0
    else:
        decay = np.array(decay_container["numbers"], dtype=float)

    # Mechanics estimation
    historical = pipeline.get_data("historical_data") or []
    mechanics_vec, chi2_stat, total_obs = _estimate_mechanics_dirichlet_from_history(historical, alpha=alpha)

    mechanics_used = mechanics_vec.copy()
    mechanics_is_uniform = False
    if use_mechanics_if_significant:
        if total_obs == 0 or chi2_stat < chi2_threshold:
            mechanics_used = np.ones(40) / 40.0
            mechanics_is_uniform = True
            if verbose:
                logging.info(f"Mechanics not significant (chi2={chi2_stat:.2f}, n={total_obs}). Using uniform mechanics.")
        else:
            if verbose:
                logging.info(f"Mechanics significant (chi2={chi2_stat:.2f}, n={total_obs}). Using estimated mechanics.")

    # Clip negatives
    freq = np.clip(freq, 0.0, None)
    decay = np.clip(decay, 0.0, None)
    mechanics_used = np.clip(mechanics_used, 0.0, None)

    # --- Log-space fusion ---
    w_f, w_d, w_m = weights
    eps = 1e-12
    log_posterior = (
        w_f * np.log(freq + eps) +
        w_d * np.log(decay + eps) +
        w_m * np.log(mechanics_used + eps)
    )
    posterior = np.exp(log_posterior)

    # Normalize to sum=1
    total = posterior.sum()
    if total > 0:
        posterior /= total
    else:
        logging.warning("Posterior sum is zero after fusion. Falling back to uniform distribution.")
        posterior = np.ones(40) / 40.0

    # Max-normalize for DL features
    posterior_norm = posterior / max(posterior.max(), 1e-12)

    # Store results
    pipeline.add_data("bayesian_fusion", posterior)           # probability distribution
    pipeline.add_data("bayesian_fusion_norm", posterior_norm) # scaled [0,1] for deep learning
    pipeline.add_data("mechanics_vector", mechanics_vec)
    pipeline.add_data("mechanics_chi2", chi2_stat)
    pipeline.add_data("mechanics_total_obs", total_obs)
    pipeline.add_data("mechanics_used_is_uniform", bool(mechanics_is_uniform))
    logging.info(f"Bayesian fusion stored successfully.")

    return posterior



# Modified By: Callam
# Project: Lotto Generator
# Purpose: Dedicated Post-Training Ablation & Pipe Importance Module
# Description:
#   - Performs post-training grouped permutation importance analysis
#   - Measures the contribution of each classical pipeline component
#     by shuffling entire feature blocks and observing the drop in macro AUC
#   - Designed as a clean partner to optuna_bridge.py and RuntimeObserver
#   - Completely decoupled from deep_learning.py training logic
#   - Uses CLASSICAL_FEATURE_BLOCKS as single source of truth for ordering
#   - Future-proof for controlled runtime ablation and richer metrics

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ===================== Single Source of Truth (must stay in sync with deep_learning.py) ===================== #
CLASSICAL_FEATURE_BLOCKS: List[str] = [
    "bayesian_fusion_norm",
    "monte_carlo",
    "redundancy",
    "markov_features",
    "entropy_features",
    "centroids",
    "clusters",
]


def compute_post_training_pipe_importance(
    model: Any,
    Xf_val: np.ndarray,
    Y_val: np.ndarray,
    pipe_feature_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    n_repeats: int = 10,
) -> Dict[str, float]:
    """
    Performs post-training grouped permutation importance analysis.

    For each classical feature block (pipe):
      - The entire block is randomly shuffled multiple times.
      - The resulting drop in macro AUC is measured.
      - Higher positive importance = the model relies more on that pipe.

    This is the standard, model-agnostic method recommended by
    scikit-learn and the broader interpretability literature
    (Breiman 2001, Fisher et al. 2019, and subsequent group variants).

    Args:
        model: Trained Keras model.
        Xf_val: Validation fused feature matrix.
        Y_val: Validation labels (multi-hot).
        pipe_feature_ranges: Optional override of column ranges.
                             If None, ranges are built automatically from
                             CLASSICAL_FEATURE_BLOCKS (each block = 50 columns).
        n_repeats: Number of random shuffles per pipe (default 10).

    Returns:
        Dict[str, float]: {pipe_name: mean_importance}
                          where importance = baseline_auc - mean(shuffled_auc)
    """
    if roc_auc_score is None:
        logging.warning("sklearn not available — skipping pipe importance analysis.")
        return {}

    block_size = 50  # Each classical pipe occupies exactly 50 columns in the fused matrix

    # Build column ranges from the canonical block order if not provided
    if pipe_feature_ranges is None:
        pipe_feature_ranges = {}
        for i, name in enumerate(CLASSICAL_FEATURE_BLOCKS):
            start = i * block_size
            pipe_feature_ranges[name] = (start, start + block_size)

    # Baseline prediction and AUC
    baseline_pred = model.predict(Xf_val, verbose=0)
    try:
        baseline_auc = roc_auc_score(Y_val, baseline_pred, average="macro")
    except Exception:
        baseline_auc = 0.5

    importance: Dict[str, float] = {}

    print("\n" + "=" * 72)
    print("POST-TRAINING GROUPED PERMUTATION IMPORTANCE ANALYSIS")
    print("=" * 72)
    print(f"Baseline Validation AUC: {baseline_auc:.4f}")
    print(f"Repeats per pipe       : {n_repeats}")
    print("-" * 72)
    print(f"{'Pipe Name':<28} {'Importance (drop)':>18}")
    print("-" * 72)

    rng = np.random.default_rng(seed=42)  # Reproducible shuffles

    for pipe_name, (start_col, end_col) in pipe_feature_ranges.items():
        drops = []

        for _ in range(n_repeats):
            X_shuffled = Xf_val.copy()
            # Shuffle the entire block as a unit (grouped permutation)
            block = X_shuffled[:, start_col:end_col].copy()
            rng.shuffle(block, axis=0)
            X_shuffled[:, start_col:end_col] = block

            shuffled_pred = model.predict(X_shuffled, verbose=0)
            try:
                shuffled_auc = roc_auc_score(Y_val, shuffled_pred, average="macro")
            except Exception:
                shuffled_auc = baseline_auc

            drops.append(baseline_auc - shuffled_auc)

        mean_importance = float(np.mean(drops))
        importance[pipe_name] = mean_importance
        print(f"{pipe_name:<28} {mean_importance:>18.4f}")

    if importance:
        most_impactful = max(importance, key=importance.get)
        print("-" * 72)
        print(
            f"Most impactful pipe (largest positive importance): "
            f"{most_impactful} (importance={importance[most_impactful]:.4f})"
        )
    print("=" * 72 + "\n")

    logging.info("Grouped permutation importance analysis completed successfully.")
    return importance


def get_most_impactful_pipe(importance: Dict[str, float]) -> Optional[str]:
    """
    Returns the pipe with the highest positive importance
    (the one the model relies on most).
    """
    if not importance:
        return None
    return max(importance, key=importance.get)
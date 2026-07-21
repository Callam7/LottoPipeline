# Modified By: Callam
# Project: Lotto Generator
# Purpose: Dedicated Post-Training Ablation & Pipe Importance Module
# Description:
#   - Performs post-training feature-block ablation analysis
#   - Measures the contribution of each classical pipeline component
#     on validation performance (primarily macro AUC)
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
) -> Dict[str, float]:
    """
    Performs post-training ablation by zeroing each classical feature block
    one-by-one and measuring the change in macro AUC.

    Negative delta = the pipe contributed positively to performance.
    This function replaces the inline _compute_pipe_importance that previously
    lived inside deep_learning.py.

    Args:
        model: Trained Keras model.
        Xf_val: Validation fused feature matrix.
        Y_val: Validation labels (multi-hot).
        pipe_feature_ranges: Optional override of column ranges.
                             If None, ranges are built automatically from
                             CLASSICAL_FEATURE_BLOCKS (each block = 50 columns).

    Returns:
        Dict[str, float]: {pipe_name: delta_auc}
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
    print("POST-TRAINING PIPE IMPORTANCE ANALYSIS")
    print("=" * 72)
    print(f"Baseline Validation AUC: {baseline_auc:.4f}")
    print("-" * 72)
    print(f"{'Pipe Name':<28} {'Delta (Impact)':>16}")
    print("-" * 72)

    for pipe_name, (start_col, end_col) in pipe_feature_ranges.items():
        X_modified = Xf_val.copy()
        X_modified[:, start_col:end_col] = 0.0

        modified_pred = model.predict(X_modified, verbose=0)
        try:
            modified_auc = roc_auc_score(Y_val, modified_pred, average="macro")
        except Exception:
            modified_auc = baseline_auc

        delta = modified_auc - baseline_auc
        importance[pipe_name] = float(delta)
        print(f"{pipe_name:<28} {delta:>16.4f}")

    if importance:
        most_impactful = min(importance, key=importance.get)
        print("-" * 72)
        print(
            f"Most impactful pipe (largest negative delta): "
            f"{most_impactful} (delta={importance[most_impactful]:.4f})"
        )
    print("=" * 72 + "\n")

    logging.info("Post-training pipe importance analysis completed successfully.")
    return importance


def get_most_impactful_pipe(importance: Dict[str, float]) -> Optional[str]:
    """Returns the pipe name with the most negative impact (weakest performing)."""
    if not importance:
        return None
    return min(importance, key=importance.get)
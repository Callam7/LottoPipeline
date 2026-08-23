# Modified By: Callam
# Project: Lotto Generator
# Purpose: Post-training pipe importance — permutation + full retrain ablation
# Description:
#   - Grouped permutation importance (model fixed, shuffle blocks)
#   - Leave-one-block-out retrain ablation (drop block columns, train fresh model,
#     no writes to epochs / lotto.db)
#   - Both score dicts available for optuna_bridge weak-link decisions
#   - Classical + quantum_features + quantum_kernels

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import keras

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== Constants ===================== #
BLOCK_SIZE = 50
DEFAULT_N_REPEATS = 10
PERMUTATION_SEED = 42
ZERO_THRESHOLD = 1e-6
FALLBACK_AUC = 0.5

QUANTUM_FEATURE_LEN = 16
KERNEL_PROTOTYPES = 24

# Retrain-ablation defaults (isolated — never logged to DB)
ABLATION_EPOCHS = 25
ABLATION_BATCH_SIZE = 32
ABLATION_LR = 1e-3
ABLATION_PATIENCE = 5
ABLATION_SEED = 42

# ===================== Block definitions ===================== #
# Must stay in sync with deep_learning.py classical order
CLASSICAL_FEATURE_BLOCKS: List[str] = [
    "bayesian_fusion_norm",
    "monte_carlo",
    "redundancy",
    "markov_features",
    "entropy_features",
    "centroids",
    "clusters",
]

QUANTUM_FEATURE_BLOCKS: List[str] = [
    "quantum_features",
    "quantum_kernels",
]

ALL_FEATURE_BLOCKS: List[str] = CLASSICAL_FEATURE_BLOCKS + QUANTUM_FEATURE_BLOCKS


def _build_default_ranges() -> Dict[str, Tuple[int, int]]:
    """Column ranges in the fused matrix: classical (350) + quantum (16) + kernels (24)."""
    ranges: Dict[str, Tuple[int, int]] = {}
    for i, name in enumerate(CLASSICAL_FEATURE_BLOCKS):
        start = i * BLOCK_SIZE
        ranges[name] = (start, start + BLOCK_SIZE)

    q_start = len(CLASSICAL_FEATURE_BLOCKS) * BLOCK_SIZE
    ranges["quantum_features"] = (q_start, q_start + QUANTUM_FEATURE_LEN)

    k_start = q_start + QUANTUM_FEATURE_LEN
    ranges["quantum_kernels"] = (k_start, k_start + KERNEL_PROTOTYPES)
    return ranges


def _safe_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_pred, average="macro"))
    except Exception:
        return FALLBACK_AUC


def _print_score(name: str, score: float) -> None:
    if abs(score) < ZERO_THRESHOLD:
        print(f"{name:<28} {'0.0000':>18}")
    else:
        print(f"{name:<28} {score:>+18.4f}")


def get_weak_link(importance: Dict[str, float]) -> Optional[str]:
    """
    Weak link = lowest (most negative) contribution.
    Returns None if every score is noise around zero.
    """
    if not importance:
        return None
    weakest = min(importance, key=importance.get)
    if abs(importance[weakest]) < ZERO_THRESHOLD:
        return None
    return weakest


# =====================================================================
# 1) GROUPED PERMUTATION IMPORTANCE
# =====================================================================

def compute_permutation_importance(
    model: Any,
    Xf_val: np.ndarray,
    Y_val: np.ndarray,
    pipe_feature_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    n_repeats: int = DEFAULT_N_REPEATS,
) -> Dict[str, float]:
    """
    Grouped permutation importance (Breiman / Fisher et al.).

    score = baseline_auc - mean(shuffled_auc)
      Positive → model relies on the block
      Negative / ~0 → weak or unused
    """
    if pipe_feature_ranges is None:
        pipe_feature_ranges = _build_default_ranges()

    baseline_pred = model.predict(Xf_val, verbose=0)
    baseline_auc = _safe_macro_auc(Y_val, baseline_pred)

    importance: Dict[str, float] = {}

    print("\n" + "=" * 72)
    print("GROUPED PERMUTATION IMPORTANCE")
    print("=" * 72)
    print(f"Baseline Validation AUC: {baseline_auc:.4f}")
    print(f"Repeats per pipe       : {n_repeats}")
    print("-" * 72)
    print(f"{'Pipe Name':<28} {'Score (drop)':>18}")
    print("-" * 72)

    rng = np.random.default_rng(seed=PERMUTATION_SEED)

    for pipe_name, (start_col, end_col) in pipe_feature_ranges.items():
        drops = []
        for _ in range(n_repeats):
            X_shuffled = Xf_val.copy()
            block = X_shuffled[:, start_col:end_col].copy()
            rng.shuffle(block, axis=0)
            X_shuffled[:, start_col:end_col] = block

            shuffled_pred = model.predict(X_shuffled, verbose=0)
            shuffled_auc = _safe_macro_auc(Y_val, shuffled_pred)
            drops.append(baseline_auc - shuffled_auc)

        mean_score = float(np.mean(drops))
        importance[pipe_name] = mean_score
        _print_score(pipe_name, mean_score)

    weak_link = get_weak_link(importance)
    print("-" * 72)
    if weak_link is not None:
        print(
            f"Weak link (permutation): {weak_link} "
            f"(score={importance[weak_link]:+.4f})"
        )
    else:
        print("Weak link (permutation): none (all scores ≈ 0)")
    print("=" * 72 + "\n")

    logging.info("Permutation importance completed.")
    return importance


# =====================================================================
# 2) FULL RETRAIN ABLATION (leave-one-block-out, no DB writes)
# =====================================================================

def _drop_column_range(
    X: np.ndarray,
    start_col: int,
    end_col: int,
) -> np.ndarray:
    """Return X with columns [start_col:end_col] removed."""
    return np.concatenate([X[:, :start_col], X[:, end_col:]], axis=1)


def _build_ablation_model(input_dim: int, output_dim: int = 50) -> keras.Model:
    """
    Fresh Keras head for ablation only.
    Not the production model — isolated retrain, never logged to DB.
    """
    keras.utils.set_random_seed(ABLATION_SEED)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(output_dim, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ABLATION_LR),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc", multi_label=True, num_labels=output_dim)],
    )
    return model


def _train_ablation_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
) -> float:
    """
    Train in-memory only. Returns validation macro AUC.
    Must never touch epochs table / lotto.db.
    """
    model = _build_ablation_model(input_dim=X_train.shape[1], output_dim=Y_train.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=ABLATION_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=ABLATION_EPOCHS,
        batch_size=ABLATION_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    pred = model.predict(X_val, verbose=0)
    return _safe_macro_auc(Y_val, pred)


def compute_retrain_ablation_importance(
    Xf_train: np.ndarray,
    Y_train: np.ndarray,
    Xf_val: np.ndarray,
    Y_val: np.ndarray,
    pipe_feature_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    baseline_auc: Optional[float] = None,
) -> Dict[str, float]:
    """
    Leave-one-block-out retrain ablation.

    For each block:
      - Drop those columns from train/val fused features
      - Train a fresh model from scratch (in-memory only, no DB)
      - score = baseline_auc - ablated_auc
        Positive → removing the block hurt → block was useful
        Negative / ~0 → removing did not hurt → weak link candidate

    This is classic ablation on the fused representation. Quantum columns
    are ablated by removal + retrain of the consumer head; classical blocks
    the same. No pipeline/epoch logging occurs here.
    """
    if pipe_feature_ranges is None:
        pipe_feature_ranges = _build_default_ranges()

    if baseline_auc is None:
        # Baseline = retrain on full fused features (fair comparison)
        logging.info("Ablation baseline: training on full fused features...")
        baseline_auc = _train_ablation_model(Xf_train, Y_train, Xf_val, Y_val)

    importance: Dict[str, float] = {}

    print("\n" + "=" * 72)
    print("LEAVE-ONE-BLOCK-OUT RETRAIN ABLATION (no DB writes)")
    print("=" * 72)
    print(f"Baseline Validation AUC: {baseline_auc:.4f}")
    print(f"Ablation epochs         : {ABLATION_EPOCHS}")
    print("-" * 72)
    print(f"{'Pipe Name':<28} {'Score (drop)':>18}")
    print("-" * 72)

    for pipe_name, (start_col, end_col) in pipe_feature_ranges.items():
        logging.info(f"Ablation retrain without '{pipe_name}'...")
        Xtr = _drop_column_range(Xf_train, start_col, end_col)
        Xva = _drop_column_range(Xf_val, start_col, end_col)

        ablated_auc = _train_ablation_model(Xtr, Y_train, Xva, Y_val)
        score = float(baseline_auc - ablated_auc)
        importance[pipe_name] = score
        _print_score(pipe_name, score)

    weak_link = get_weak_link(importance)
    print("-" * 72)
    if weak_link is not None:
        print(
            f"Weak link (retrain ablation): {weak_link} "
            f"(score={importance[weak_link]:+.4f})"
        )
    else:
        print("Weak link (retrain ablation): none (all scores ≈ 0)")
    print("=" * 72 + "\n")

    logging.info("Retrain ablation completed (no DB writes).")
    return importance


# =====================================================================
# 3) COMBINED ENTRY — both methods for optuna_bridge
# =====================================================================

def compute_post_training_pipe_importance(
    model: Any,
    Xf_val: np.ndarray,
    Y_val: np.ndarray,
    Xf_train: Optional[np.ndarray] = None,
    Y_train: Optional[np.ndarray] = None,
    pipe_feature_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    n_repeats: int = DEFAULT_N_REPEATS,
    run_retrain_ablation: bool = True,
) -> Dict[str, Any]:
    """
    Run permutation importance always.
    Run leave-one-block-out retrain ablation when train tensors are provided.

    Returns:
        {
          "permutation": {pipe: score},
          "ablation": {pipe: score} or {},
          "weak_link_permutation": str | None,
          "weak_link_ablation": str | None,
          "weak_link": str | None,   # agreement, else permutation, else ablation
        }
    """
    if pipe_feature_ranges is None:
        pipe_feature_ranges = _build_default_ranges()

    perm_scores = compute_permutation_importance(
        model=model,
        Xf_val=Xf_val,
        Y_val=Y_val,
        pipe_feature_ranges=pipe_feature_ranges,
        n_repeats=n_repeats,
    )

    abl_scores: Dict[str, float] = {}
    if run_retrain_ablation and Xf_train is not None and Y_train is not None:
        abl_scores = compute_retrain_ablation_importance(
            Xf_train=Xf_train,
            Y_train=Y_train,
            Xf_val=Xf_val,
            Y_val=Y_val,
            pipe_feature_ranges=pipe_feature_ranges,
        )
    elif run_retrain_ablation:
        logging.warning(
            "Retrain ablation skipped — Xf_train / Y_train not provided."
        )

    weak_perm = get_weak_link(perm_scores)
    weak_abl = get_weak_link(abl_scores) if abl_scores else None

    # Agreement preferred; otherwise prefer permutation, then ablation
    if weak_perm and weak_abl and weak_perm == weak_abl:
        weak_final = weak_perm
    elif weak_perm:
        weak_final = weak_perm
    else:
        weak_final = weak_abl

    return {
        "permutation": perm_scores,
        "ablation": abl_scores,
        "weak_link_permutation": weak_perm,
        "weak_link_ablation": weak_abl,
        "weak_link": weak_final,
    }

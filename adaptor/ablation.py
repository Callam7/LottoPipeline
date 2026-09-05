# Modified By: Callam
# Project: Lotto Generator
# Purpose: Post-training leave-one-block-out retrain ablation
# Description:
#   - Drop one fused-feature block, train a fresh model in memory
#   - No writes to epochs / lotto.db
#   - Score = baseline_auc - ablated_auc
#   - Positive = block was useful (removing it hurt)
#   - Zero / negative = unused or harmful (weak-link candidate)
#   - Baseline prefers production val AUC from deep_learning
#   - Retrain uses DL loss / lr / dropout / batch when passed
#   - Prefer pipe_feature_ranges from deep_learning.py

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow import keras

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ===================== Constants ===================== #
ZERO_THRESHOLD = 0.002
FALLBACK_AUC = 0.5
NUM_TOTAL = 50

BLOCK_SIZE = 50
QUANTUM_FEATURE_LEN = 16
KERNEL_PROTOTYPES = 24

ABLATION_EPOCHS = 25
ABLATION_BATCH_SIZE = 32
ABLATION_LR = 8e-4
ABLATION_PATIENCE = 5
ABLATION_SEED = 42
ABLATION_DROPOUT = 0.25

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


def _build_default_ranges() -> Dict[str, Tuple[int, int]]:
    """Fallback only. Prefer ranges built in deep_learning.py from the live Xf layout."""
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


def _snap_score(score: float) -> float:
    if abs(score) < ZERO_THRESHOLD:
        return 0.0
    return float(score)


def _print_score(name: str, score: float) -> None:
    if abs(score) < ZERO_THRESHOLD:
        print(f"{name:<28} {'0.0000':>18}")
    else:
        print(f"{name:<28} {score:>+18.4f}")


def get_candidate(importance: Dict[str, float]) -> Optional[str]:
    """
    Candidate = lowest score (most unused / most harmful).
    None only when every score is inside the noise floor.
    A true 0.0 next to large positive scores is a valid unused-pipe candidate.
    """
    if not importance:
        return None
    if all(abs(v) < ZERO_THRESHOLD for v in importance.values()):
        return None
    return min(importance, key=importance.get)


def _drop_column_range(X: np.ndarray, start_col: int, end_col: int) -> np.ndarray:
    return np.concatenate([X[:, :start_col], X[:, end_col:]], axis=1)


def _build_ablation_model(
    input_dim: int,
    output_dim: int = NUM_TOTAL,
    loss_fn: Any = None,
    learning_rate: float = ABLATION_LR,
    dropout_rate: float = ABLATION_DROPOUT,
) -> keras.Model:
    """
    Isolated consumer head. Same depth family as production DL.
    Never attached to EpochLogger / lotto.db.
    """
    keras.utils.set_random_seed(ABLATION_SEED)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(output_dim, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=loss_fn if loss_fn is not None else keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.AUC(
                name="auc",
                multi_label=True,
                num_labels=output_dim,
            )
        ],
    )
    return model


def _train_ablation_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    loss_fn: Any = None,
    learning_rate: float = ABLATION_LR,
    dropout_rate: float = ABLATION_DROPOUT,
    batch_size: int = ABLATION_BATCH_SIZE,
) -> float:
    model = _build_ablation_model(
        input_dim=X_train.shape[1],
        output_dim=Y_train.shape[1],
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
    )
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=ABLATION_EPOCHS,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=ABLATION_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
        ],
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
    production_auc: Optional[float] = None,
    loss_fn: Any = None,
    learning_rate: float = ABLATION_LR,
    dropout_rate: float = ABLATION_DROPOUT,
    batch_size: int = ABLATION_BATCH_SIZE,
) -> Dict[str, float]:
    """
    Leave-one-block-out retrain ablation on the fused matrix.

    Baseline:
      1) explicit baseline_auc
      2) else production_auc from deep_learning
      3) else a full-matrix retrain with the same recipe
    """
    if pipe_feature_ranges is None:
        logging.warning(
            "pipe_feature_ranges not provided — using fallback constants. "
            "Pass ranges from deep_learning.py to avoid column drift."
        )
        pipe_feature_ranges = _build_default_ranges()

    train_kw = dict(
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
    )

    if baseline_auc is None and production_auc is not None:
        baseline_auc = float(production_auc)
        logging.info(f"Ablation baseline: production val AUC = {baseline_auc:.4f}")
    elif baseline_auc is None:
        logging.info("Ablation baseline: training on full fused features...")
        baseline_auc = _train_ablation_model(
            Xf_train, Y_train, Xf_val, Y_val, **train_kw
        )

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
        ablated_auc = _train_ablation_model(Xtr, Y_train, Xva, Y_val, **train_kw)
        score = _snap_score(float(baseline_auc - ablated_auc))
        importance[pipe_name] = score
        _print_score(pipe_name, score)

    candidate = get_candidate(importance)
    print("-" * 72)
    if candidate is not None:
        val = importance[candidate]
        if abs(val) < ZERO_THRESHOLD:
            print(f"Candidate (ablation): {candidate} (score=0.0000)")
        else:
            print(f"Ceak link (ablation): {candidate} (score={val:+.4f})")
    else:
        print("Candidate (ablation): none (all scores ≈ 0)")
    print("=" * 72 + "\n")

    logging.info("Retrain ablation completed (no DB writes).")
    return importance


def compute_post_training_pipe_importance(
    model: Any,
    Xf_val: np.ndarray,
    Y_val: np.ndarray,
    Xf_train: Optional[np.ndarray] = None,
    Y_train: Optional[np.ndarray] = None,
    pipe_feature_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    production_auc: Optional[float] = None,
    loss_fn: Any = None,
    learning_rate: float = ABLATION_LR,
    dropout_rate: float = ABLATION_DROPOUT,
    batch_size: int = ABLATION_BATCH_SIZE,
    **_: Any,
) -> Dict[str, Any]:
    """
    Single method: leave-one-block-out retrain ablation.
    `model` is unused (kept so the deep_learning call site stays stable).
    Extra kwargs are ignored so older call sites do not crash.
    """
    if Xf_train is None or Y_train is None:
        logging.error("Ablation skipped — Xf_train / Y_train not provided.")
        return {
            "ablation": {},
            "candidate": None,
        }

    abl_scores = compute_retrain_ablation_importance(
        Xf_train=Xf_train,
        Y_train=Y_train,
        Xf_val=Xf_val,
        Y_val=Y_val,
        pipe_feature_ranges=pipe_feature_ranges,
        production_auc=production_auc,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
    )

    return {
        "ablation": abl_scores,
        "candidate": get_candidate(abl_scores),
    }
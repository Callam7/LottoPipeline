# Modified By: Callam
# Project: Lotto Generator
# Purpose: Post-training leave-one-block-out retrain ablation
# Description:
#   - Drop one fused-feature block, train a fresh model in memory
#   - No writes to epochs / lotto.db
#   - Score = baseline_auc - ablated_auc
#   - Positive = block helped (removing it hurt val AUC)
#   - Zero / negative = unused, or short retrain beat production
#   - Candidate = lowest score — rewrite inside that file later, do not delete it
#   - Baseline prefers production val AUC from deep_learning
#   - Macro AUC skips labels that are all-0 or all-1 in val (PB 11-14 have no history)

import logging                         # Ablation progress and skip errors
from typing import Any, Dict, List, Optional, Tuple  # Types for ranges and scores
import numpy as np                     # Arrays and finite checks
from sklearn.metrics import roc_auc_score  # Per-label AUC
from tensorflow import keras           # Short retrain head

logging.basicConfig(                   # Same stamp format as the rest of the adaptor
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ===================== Constants ===================== #
ZERO_THRESHOLD = 0.002                 # |score| below this prints and stores as 0
FALLBACK_AUC = 0.5                     # Used if AUC cannot be computed
NUM_TOTAL = 54                         # 40 main + 14 powerball
BLOCK_SIZE = 54                        # Width of one classical block in Xf
QUANTUM_FEATURE_LEN = 16               # Width of quantum_features
KERNEL_PROTOTYPES = 24                 # Width of quantum_kernels
ABLATION_EPOCHS = 25                   # Max epochs for each isolated retrain
ABLATION_BATCH_SIZE = 32               # Default batch if DL does not pass one
ABLATION_LR = 8e-4                     # Default LR if DL does not pass one
ABLATION_PATIENCE = 5                  # Early stop patience on val_auc
ABLATION_SEED = 42                     # Keras seed for the ablation head
ABLATION_DROPOUT = 0.25                # Default dropout if DL does not pass one

CLASSICAL_FEATURE_BLOCKS: List[str] = [  # Order must match DL if using fallback ranges
    "bayesian_fusion_norm",
    "monte_carlo",
    "redundancy",
    "markov_features",
    "entropy_features",
    "centroids",
    "clusters",
]

QUANTUM_FEATURE_BLOCKS: List[str] = [  # Encoder columns after the classical stack
    "quantum_features",
    "quantum_kernels",
]

COMPONENT_CLASS: Dict[str, str] = {    # Name -> pipe | encoder
    name: "pipe" for name in CLASSICAL_FEATURE_BLOCKS
}
COMPONENT_CLASS.update({name: "encoder" for name in QUANTUM_FEATURE_BLOCKS})


def _build_default_ranges() -> Dict[str, Tuple[int, int]]:
    """Fallback only. Prefer live ranges from deep_learning."""
    ranges: Dict[str, Tuple[int, int]] = {}  # name -> (start, end)
    for i, name in enumerate(CLASSICAL_FEATURE_BLOCKS):
        start = i * BLOCK_SIZE         # Each classical block is BLOCK_SIZE wide
        ranges[name] = (start, start + BLOCK_SIZE)
    q_start = len(CLASSICAL_FEATURE_BLOCKS) * BLOCK_SIZE  # First encoder column
    ranges["quantum_features"] = (q_start, q_start + QUANTUM_FEATURE_LEN)
    k_start = q_start + QUANTUM_FEATURE_LEN
    ranges["quantum_kernels"] = (k_start, k_start + KERNEL_PROTOTYPES)
    return ranges


def _safe_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro AUC over labels that have both classes in y_true.
    PB 11-14 are all-zero in val until those numbers exist in history.
    sklearn macro on all 54 columns becomes nan — skip those columns.
    """
    y_true = np.asarray(y_true)        # Labels
    y_pred = np.asarray(y_pred)        # Predictions
    if y_true.ndim != 2 or y_pred.ndim != 2:
        return FALLBACK_AUC            # Need matrices
    n_labels = min(y_true.shape[1], y_pred.shape[1])  # Shared column count
    scores = []                        # One AUC per usable label
    for j in range(n_labels):
        col = y_true[:, j]             # That label in val
        if col.min() == col.max():     # All 0 or all 1 — skip
            continue
        try:
            scores.append(float(roc_auc_score(col, y_pred[:, j])))
        except Exception:
            continue                   # sklearn failed this column
    if not scores:
        return FALLBACK_AUC
    return float(np.mean(scores))      # Mean of usable labels only


def _snap_score(score: float) -> float:
    if not np.isfinite(score):         # nan / inf
        return 0.0
    if abs(score) < ZERO_THRESHOLD:    # Noise band
        return 0.0
    return float(score)


def _print_score(name: str, score: float) -> None:
    if abs(score) < ZERO_THRESHOLD:
        print(f"{name:<28} {'0.0000':>18}")  # No + / - on zero
    else:
        print(f"{name:<28} {score:>+18.4f}")


def get_candidate(
    importance: Dict[str, float],
    class_name: Optional[str] = None,
) -> Optional[str]:
    """
    Lowest score in the given class.
    class_name None = all keys (Stage 1 global pick).
    """
    if class_name is not None:
        importance = {                 # Keep only that class
            k: v for k, v in importance.items()
            if COMPONENT_CLASS.get(k) == class_name
        }
    if not importance:
        return None
    if all(abs(v) < ZERO_THRESHOLD for v in importance.values()):
        return None                    # Everything is noise
    return min(importance, key=importance.get)  # Lowest score wins


def get_candidate_pipe(importance: Dict[str, float]) -> Optional[str]:
    return get_candidate(importance, "pipe")


def get_candidate_encoder(importance: Dict[str, float]) -> Optional[str]:
    return get_candidate(importance, "encoder")


def get_weak_link(importance: Dict[str, float]) -> Optional[str]:
    """Old name. Means pipe candidate only."""
    return get_candidate_pipe(importance)


def _drop_column_range(X: np.ndarray, start_col: int, end_col: int) -> np.ndarray:
    return np.concatenate([X[:, :start_col], X[:, end_col:]], axis=1)  # Drop [start, end)


def _build_ablation_model(
    input_dim: int,                    # Columns left after the drop
    output_dim: int = NUM_TOTAL,       # 54 labels
    loss_fn: Any = None,               # DL loss if passed
    learning_rate: float = ABLATION_LR,
    dropout_rate: float = ABLATION_DROPOUT,
) -> keras.Model:
    keras.utils.set_random_seed(ABLATION_SEED)  # Repeatable init
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
            keras.layers.Dense(output_dim, activation="sigmoid"),  # Multi-label
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
        input_dim=X_train.shape[1],    # Current width after drop
        output_dim=Y_train.shape[1],   # Label width
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
        verbose=0,                     # No epoch spam
    )
    pred = model.predict(X_val, verbose=0)
    return _safe_macro_auc(Y_val, pred)  # Val score for this drop


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
    if pipe_feature_ranges is None:
        logging.warning(
            "pipe_feature_ranges not provided — using fallback constants. "
            "Pass ranges from deep_learning.py to avoid column drift."
        )
        pipe_feature_ranges = _build_default_ranges()

    train_kw = dict(                   # Shared kwargs for every retrain
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        batch_size=batch_size,
    )

    if baseline_auc is None and production_auc is not None and np.isfinite(production_auc):
        baseline_auc = float(production_auc)  # Prefer the live model score
        logging.info(f"Ablation baseline: production val AUC = {baseline_auc:.4f}")
    elif baseline_auc is None:
        logging.info("Ablation baseline: training on full fused features...")
        baseline_auc = _train_ablation_model(
            Xf_train, Y_train, Xf_val, Y_val, **train_kw
        )

    if not np.isfinite(baseline_auc):
        logging.warning("Baseline AUC was not finite — using fallback 0.5")
        baseline_auc = FALLBACK_AUC

    importance: Dict[str, float] = {}  # name -> snapped score

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
        Xtr = _drop_column_range(Xf_train, start_col, end_col)  # Train minus block
        Xva = _drop_column_range(Xf_val, start_col, end_col)    # Val minus block
        ablated_auc = _train_ablation_model(Xtr, Y_train, Xva, Y_val, **train_kw)
        score = _snap_score(float(baseline_auc - ablated_auc))  # Drop vs baseline
        importance[pipe_name] = score
        _print_score(pipe_name, score)

    candidate = get_candidate(importance)  # Lowest of all nine
    print("-" * 72)
    if candidate is not None:
        val = importance[candidate]
        kind = COMPONENT_CLASS.get(candidate, "pipe")
        tag = "0.0000" if abs(val) < ZERO_THRESHOLD else f"{val:+.4f}"
        print(f"Candidate ({kind}): {candidate} (score={tag})")
    else:
        print("Candidate: none (all scores ≈ 0)")
    print("=" * 72 + "\n")
    logging.info("Retrain ablation completed (no DB writes).")
    return importance


def compute_importance(
    model: Any,                        # Unused. Kept so the DL call site stays stable
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
    **_: Any,                          # Ignore extra kwargs from older call sites
) -> Dict[str, Any]:
    if Xf_train is None or Y_train is None:
        logging.error("Ablation skipped — Xf_train / Y_train not provided.")
        return {
            "ablation": {},
            "candidate": None,
            "candidate_class": None,
            "candidate_pipe": None,
            "candidate_encoder": None,
            "weak_link": None,
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
    cand = get_candidate(abl_scores)   # Global lowest
    kind = COMPONENT_CLASS.get(cand) if cand else None
    pipe_cand = cand if kind == "pipe" else None
    enc_cand = cand if kind == "encoder" else None
    return {
        "ablation": abl_scores,
        "candidate": cand,
        "candidate_class": kind,
        "candidate_pipe": pipe_cand,
        "candidate_encoder": enc_cand,
        "weak_link": pipe_cand,        # Old key: pipe only
    }
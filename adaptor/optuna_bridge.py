# Modified By: Callam
# Project: Lotto Generator
# Purpose: Optuna Bridge - Stage 1 (report-only decision context + hyperparameter study)
# Description:
#   - Compares latest training batch to recent history (peak val AUC)
#   - If improved (beats noise-scaled median AND previous best): skip
#   - If not improved:
#       * Run a short Optuna study to produce better hyperparameters
#       * Emit a decision record
#           * pipe candidate → methodology_review
#           * encoder candidate → encoder_review
#           * no candidate → stack_review
#   - Pulls structure from assessment and runtime from the observer
#   - Does not rewrite code. Stores best params for the next training run.
import logging                         # Writes the Stage 1 log lines
import sqlite3                         # Opens lotto.db for the epochs query
import statistics                      # Median and std of previous peak AUCs
from typing import Any, Dict, Optional # Types for pipeline / context dicts

import optuna                          # Hyperparameter search (Stage 1 study)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(                   # One log format for this module
    level=logging.INFO,                # INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Time + level + text
)

DB_PATH = "lotto.db"                   # SQLite file the trainer writes epochs into
PIPE_IMPORTANCE_KEY = "pipe_importance"  # Pipeline key for the ablation score dict
WEAKEST_PIPE_KEY = "weakest_pipe"      # Old pipeline key; used if candidate is missing
CANDIDATE_PIPE_KEY = "candidate_pipe"  # Pipeline key when the winner is a pipe
CANDIDATE_ENCODER_KEY = "candidate_encoder"  # Pipeline key when the winner is an encoder
CANDIDATE_KEY = "candidate"            # Pipeline key for the single winner name
CANDIDATE_CLASS_KEY = "candidate_class"  # Pipeline key: "pipe" or "encoder"
ENCODER_BLOCKS = ("quantum_features", "quantum_kernels")  # Names treated as encoders

PIPE_TO_FILENAME = {                   # Ablation block name -> file basename
    "bayesian_fusion_norm": "bayesian_fusion.py",  # Fusion step
    "monte_carlo": "monte_carlo.py",   # Monte Carlo step
    "redundancy": "redundancy.py",     # Sequential / redundancy step
    "markov_features": "markov.py",    # Markov step
    "entropy_features": "entropy.py",  # Entropy step
    "centroids": "clustering.py",      # Clustering file, centroid vector
    "clusters": "clustering.py",       # Same file, label vector
    "quantum_features": "quantum_features.py",  # Encoder module
    "quantum_kernels": "quantum_kernels.py",    # Encoder module
}

# Optuna study settings (Stage 1 only — short, non-destructive)
OPTUNA_N_TRIALS = 8                    # Max trials when a batch under-performed
OPTUNA_TIMEOUT = 180                   # Hard time limit in seconds


def get_last_six_runs() -> list:       # Newest six date-batches from epochs
    """Return the last 6 date-grouped training batches (peak + avg)."""
    query = """
        WITH BatchStats AS (
            SELECT
                run_date,              -- Draw / train date
                COUNT(*) AS num_epochs, -- Rows that day
                AVG(val_auc) AS avg_val_auc,  -- Mean val AUC that day
                MAX(val_auc) AS peak_val_auc  -- Best val AUC that day
            FROM epochs
            WHERE run_date IS NOT NULL
            GROUP BY run_date          -- One row per date
        )
        SELECT run_date, num_epochs, avg_val_auc, peak_val_auc
        FROM BatchStats
        ORDER BY run_date DESC         -- Newest first
        LIMIT 6                        -- Cap at six batches
    """
    with sqlite3.connect(DB_PATH) as conn:          # Open DB, close after
        rows = conn.cursor().execute(query).fetchall()  # List of tuples
    logging.info("=== Last 6 FULL Training Batches ===")  # Header
    for i, (date, epochs, avg, peak) in enumerate(rows, 1):  # 1-based index
        avg_str = f"{avg:.4f}" if avg is not None else "N/A"    # Format or N/A
        peak_str = f"{peak:.4f}" if peak is not None else "N/A"  # Format or N/A
        logging.info(f"{i}. Date={date} | Epochs={epochs} | Avg={avg_str} | Peak={peak_str}")
    return rows                        # [0] = newest batch


def compare_latest_to_previous(runs: list) -> tuple[bool, float]:
    """
    Skip-gate uses peak val AUC (index 3), not epoch average (index 2).
    should_skip = True only if the latest peak is a genuine improvement
    over recent history (beats both the noise-scaled median and the previous best).
    """
    if len(runs) < 2:                  # Need this run plus at least one prior
        return True, 0.0               # Skip the decision record
    latest_peak = runs[0][3]           # Peak column of newest row
    if latest_peak is None:            # No peak stored
        return False, 0.0              # Do not skip
    previous_peaks = [r[3] for r in runs[1:] if r[3] is not None]  # Older peaks
    if not previous_peaks:             # Nothing to compare against
        return True, 0.0               # Skip

    median_prev = statistics.median(previous_peaks)          # Median of those peaks
    best_prev = max(previous_peaks)                          # Highest previous peak
    delta = latest_peak - median_prev                        # Positive = this batch higher

    # Dynamic bar: scales with recent noise, with a floor so it never becomes tiny
    if len(previous_peaks) >= 2:
        noise = statistics.pstdev(previous_peaks)
    else:
        noise = 0.0
    dynamic_bar = max(0.008, 1.0 * noise)

    # Must clear both conditions
    median_ok = delta > dynamic_bar
    best_ok = latest_peak > best_prev

    if median_ok and best_ok:
        logging.info(
            f"Latest batch IMPROVED "
            f"(delta={delta:+.4f}, bar={dynamic_bar:.4f}, "
            f"best_prev={best_prev:.4f})"
        )
        return True, delta             # Skip report
    logging.info(
        f"Latest batch did NOT improve "
        f"(delta={delta:+.4f}, bar={dynamic_bar:.4f}, "
        f"best_prev={best_prev:.4f})"
    )
    return False, delta                # Build report


def _resolve_runtime_key(
    weakest_pipe: Optional[str],       # Candidate name to look up
    runtime_summary: Dict[str, Any],   # Observer per-key summaries
    structural: Optional[Dict[str, Any]],  # Assessment row or None
) -> Optional[str]:
    """
    Bind the logged tensor to the candidate name.
    Never use structural['outputs'][0].
    """
    if not weakest_pipe:               # No name
        return None
    if weakest_pipe in runtime_summary:  # Observer has that exact key
        return weakest_pipe
    outputs = (structural or {}).get("outputs") or []  # Assessment outputs list
    if weakest_pipe in outputs:        # Name appears as an output
        return weakest_pipe
    return None                        # Cannot bind


def build_decision_context(
    pipeline: Any,                     # Must expose get_data
    assessor: Any = None,              # PipelineAssessment or None
    observer: Any = None,              # RuntimeObserver or None
) -> Optional[Dict[str, Any]]:
    if pipeline is None:               # Nothing to read
        logging.error("Pipeline object not provided.")
        return None
    importance = pipeline.get_data(PIPE_IMPORTANCE_KEY) or {}  # Score dict
    cand = pipeline.get_data(CANDIDATE_KEY) or pipeline.get_data(WEAKEST_PIPE_KEY)  # Winner name
    kind = pipeline.get_data(CANDIDATE_CLASS_KEY)  # "pipe" / "encoder" / None
    if kind is None and cand:          # Class missing — infer from name
        kind = "encoder" if cand in ENCODER_BLOCKS else "pipe"
    pipe_cand = cand if kind == "pipe" else None      # Set only for pipes
    enc_cand = cand if kind == "encoder" else None    # Set only for encoders
    if cand and kind == "pipe":        # Step-file candidate
        action = "methodology_review"
        reason = (
            f"Stage 1 report only. Pipe candidate '{cand}' "
            "is for later search/replacement inside that step file."
        )
    elif cand and kind == "encoder":   # Encoder candidate
        action = "encoder_review"
        reason = (
            f"Stage 1 report only. Encoder candidate '{cand}' "
            "is for Stage 2b spec/snapshot/apply, not a pipe rewrite."
        )
    else:                              # No winner
        action = "stack_review"
        reason = "Stage 1 report only. No candidate (all ablation scores ≈ 0)."
        logging.info("No actionable candidate — stack_review.")
    context: Dict[str, Any] = {        # Record written to the pipeline later
        "candidate": cand,             # Single winner name
        "candidate_class": kind,       # pipe | encoder | None
        "weakest_pipe": pipe_cand,     # Pipe-only copy
        "candidate_pipe": pipe_cand,   # Same
        "candidate_encoder": enc_cand, # Encoder-only copy
        "importance": importance or {},  # Full score table
        "structural": None,            # Filled if pipe + assessor
        "runtime_keys": None,          # Filled if observer
        "action": action,              # methodology_review | encoder_review | stack_review
        "reason": reason,              # One-line why
    }
    if assessor is not None and pipe_cand:  # Structure only for pipes
        try:
            summary = assessor.get_pipe_summary()  # Basename -> file facts
            filename = PIPE_TO_FILENAME.get(pipe_cand)  # Expected .py name
            if filename and filename in summary:  # Exact basename hit
                context["structural"] = summary[filename]
            else:                      # Fallback: name without underscores
                for fname, data in summary.items():
                    if pipe_cand.replace("_", "") in fname.replace("_", ""):
                        context["structural"] = data
                        break
        except Exception as e:
            logging.warning(f"Could not retrieve structural info: {e}")
    if observer is not None:           # Attach this-run summaries
        try:
            run_summary = observer.get_run_summary()  # Read-only
            if run_summary:
                context["runtime_keys"] = run_summary.get("keys", [])
                context["runtime_summary"] = run_summary.get("summaries", {})
                context["runtime_deltas"] = run_summary.get("deltas", {})
        except Exception as e:
            logging.warning(f"Could not retrieve runtime summary: {e}")
    return context                     # Ready to log and store


def _optuna_objective(trial: optuna.Trial, pipeline: Any) -> float:
    """
    Real short training objective for Stage 1.
    Pulls matrices stored by deep_learning.py.
    Does not use EpochLogger (no writes to the live epochs table).
    """
    Xf_train = pipeline.get_data("Xf_train")
    Y_train = pipeline.get_data("Y_train")
    Xf_val = pipeline.get_data("Xf_val")
    Y_val = pipeline.get_data("Y_val")

    if any(x is None for x in (Xf_train, Y_train, Xf_val, Y_val)):
        return 0.5

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.10, 0.40),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_int("epochs", 10, 25),
    }

    import tensorflow as tf
    from tensorflow import keras

    keras.backend.clear_session()

    model = keras.Sequential([
        keras.layers.Input(shape=(Xf_train.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(params["dropout_rate"]),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(params["dropout_rate"]),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(54, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=params["learning_rate"],
            clipnorm=1.0,
        ),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc", multi_label=True)],
    )

    history = model.fit(
        Xf_train,
        Y_train,
        validation_data=(Xf_val, Y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=4,
                restore_best_weights=True,
            )
        ],
    )

    val_aucs = history.history.get("val_auc", [0.5])
    return float(max(val_aucs))


def run_optuna_bridge(
    pipeline: Any = None,              # Optional; needed for a real report
    assessor: Any = None,
    observer: Any = None,
) -> None:
    if observer is not None:           # Touch observer even if we skip later
        try:
            observer.get_run_summary()  # Read; does not freeze previous run
        except Exception as e:
            logging.warning(f"Observer finalize failed: {e}")

    runs = get_last_six_runs()         # Newest six date batches
    if not runs:                       # No epoch rows
        return

    should_skip, overall_delta = compare_latest_to_previous(runs)  # Gate
    logging.info(f"Overall performance delta: {overall_delta:+.4f}")
    if should_skip:                    # Peak improved enough
        return

    # === Optuna study (only when the batch did not improve) ===
    if pipeline is not None:
        logging.info("Starting Stage 1 Optuna study (short, non-destructive)")
        study = optuna.create_study(direction="maximize")  # Higher val AUC is better
        study.optimize(
            lambda trial: _optuna_objective(trial, pipeline),
            n_trials=OPTUNA_N_TRIALS,
            timeout=OPTUNA_TIMEOUT,
            show_progress_bar=False,   # Keep logs clean
        )
        best = study.best_params      # Dict deep_learning.py already expects
        pipeline.add_data("optuna_best_params", best)
        pipeline.add_data("optuna_study_summary", {
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "best_params": best,
        })
        logging.info(f"Optuna best params stored: {best}")
        logging.info(f"Optuna best value: {study.best_value:.4f}")

    # Existing decision-record path (unchanged)
    context = build_decision_context(pipeline, assessor, observer)
    if context is None:                # No pipeline
        return

    logging.info("=== Stage 1 Decision Context ===")
    cand = context.get("candidate")
    kind = context.get("candidate_class")
    logging.info(f"Candidate          : {cand} ({kind})")

    importance = context.get("importance") or {}
    if cand and cand in importance and importance[cand] is not None:
        logging.info(f"Ablation score     : {importance[cand]:+.4f}")

    structural = context.get("structural")
    if structural:
        logging.info(f"Source file        : {structural.get('file_path', 'N/A')}")
        logging.info(f"Functions          : {structural.get('function_count', 0)}")

    runtime_summary = context.get("runtime_summary") or {}
    weak_output_key = _resolve_runtime_key(cand, runtime_summary, structural)
    if weak_output_key and weak_output_key in runtime_summary:
        stats = runtime_summary[weak_output_key]
        logging.info(f"Runtime ({weak_output_key}):")
        for k in ("mean", "std", "entropy", "l1_vs_uniform"):
            if k in stats and isinstance(stats[k], float):
                logging.info(f"  {k:<16} {stats[k]:.4f}")

    logging.info(f"Action             : {context['action']}")
    logging.info(f"Reason             : {context['reason']}")
    logging.info("================================")

    if pipeline is not None:
        pipeline.add_data("adaptor_decision", context)  # Store report. No file edit.


if __name__ == "__main__":             # Direct run (usually no pipeline)
    run_optuna_bridge()
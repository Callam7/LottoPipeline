# Modified By: Callam
# Project: Lotto Generator
# Purpose: Optuna Bridge - Stage 1 (report-only decision context)
# Description:
#   - Compares latest training batch to recent history (peak val AUC)
#   - If improved: skip (no adaptor action)
#   - If not improved: emit a decision record
#       * pipe candidate → methodology_review
#       * encoder candidate → encoder_review
#       * no candidate → stack_review
#   - Pulls structure from assessment and runtime from the observer
#   - Does not rewrite code. Does not search hyperparameters.

import logging                         # Writes the Stage 1 log lines
import sqlite3                         # Opens lotto.db for the epochs query
import statistics                      # Median of previous peak AUCs
from typing import Any, Dict, Optional # Types for pipeline / context dicts

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
MIN_IMPROVEMENT = 0.005                # Skip report if latest peak - median > this

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
    should_skip = True → latest peak beat the recent median by MIN_IMPROVEMENT.
    """
    if len(runs) < 2:                  # Need this run plus at least one prior
        return True, 0.0               # Skip the decision record
    latest_peak = runs[0][3]           # Peak column of newest row
    if latest_peak is None:            # No peak stored
        return False, 0.0              # Do not skip
    previous_peaks = [r[3] for r in runs[1:] if r[3] is not None]  # Older peaks
    if not previous_peaks:             # Nothing to compare against
        return True, 0.0               # Skip
    median_prev = statistics.median(previous_peaks)  # Median of those peaks
    delta = latest_peak - median_prev  # Positive = this batch higher
    if delta > MIN_IMPROVEMENT:        # Beat the bar
        logging.info(f"Latest batch IMPROVED (peak delta = {delta:+.4f})")
        return True, delta             # Skip report
    logging.info(f"Latest batch did NOT improve (peak delta = {delta:+.4f})")
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

    context = build_decision_context(pipeline, assessor, observer)
    if context is None:                # No pipeline
        return

    logging.info("=== Stage 1 Decision Context ===")
    logging.info(f"Candidate          : {context.get('candidate')}")
    logging.info(f"Candidate class    : {context.get('candidate_class')}")

    importance = context.get("importance") or {}  # Score table
    key = context.get("candidate")     # Winner name
    if key and key in importance and importance[key] is not None:
        val = importance[key]          # Ablation score
        if abs(val) < 1e-12:           # Treat tiny as zero print
            logging.info("Ablation score     : 0.0000")
        else:
            logging.info(f"Ablation score     : {val:+.6f}")

    structural = context.get("structural")  # Assessment row or None
    if structural:                     # Pipe path only
        logging.info(f"Source file        : {structural.get('file_path', 'N/A')}")
        logging.info(f"Functions          : {structural.get('function_count', 0)}")
        logging.info(f"Reads              : {structural.get('reads', [])}")
        logging.info(f"Outputs            : {structural.get('outputs', [])}")
        if structural.get("data_movement"):
            logging.info(f"Data movement      : {structural['data_movement']}")

    runtime_summary = context.get("runtime_summary") or {}  # Observer stats
    weak_output_key = _resolve_runtime_key(
        context.get("candidate"),      # Look up the winner only
        runtime_summary,
        structural,
    )
    if weak_output_key and weak_output_key in runtime_summary:
        logging.info(f"Runtime summary for '{weak_output_key}':")
        for k, v in runtime_summary[weak_output_key].items():
            logging.info(f"    {k}: {v}")
    else:
        logging.info("Runtime summary: not available for this block")

    runtime_deltas = context.get("runtime_deltas") or {}  # Vs previous run
    if weak_output_key and weak_output_key in runtime_deltas:
        d = runtime_deltas[weak_output_key]
        if d.get("comparable"):
            if "delta_entropy" in d:
                logging.info(f"Feature delta: entropy {d['delta_entropy']:+.6f}")
            elif "delta_std" in d:
                logging.info(f"Feature delta: std {d['delta_std']:+.6f}")
            elif "delta_mean" in d:
                logging.info(f"Feature delta: mean {d['delta_mean']:+.6f}")
            else:
                logging.info("Feature delta: comparable (no numeric field)")
        else:
            logging.info(
                f"Feature delta: not comparable ({d.get('reason', 'unknown')})"
            )

    logging.info(f"Recommended action: {context['action']}")
    logging.info(f"Reason: {context['reason']}")
    logging.info("================================")

    if pipeline is not None:
        pipeline.add_data("adaptor_decision", context)  # Store report. No file edit.


if __name__ == "__main__":             # Direct run (usually no pipeline)
    run_optuna_bridge()
# Modified By: Callam
# Project: Lotto Generator
# Purpose: Optuna Bridge - Stage 1 (Clean Architecture)
# Description:
# - Consumes the weak link already identified by ablation
#   - Pulls structural information from assessment.py via get_pipe_summary()
#   - Pulls runtime context from the observer via get_snapshot()


import logging
import sqlite3
import statistics
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_PATH = "lotto.db"
PIPE_IMPORTANCE_KEY = "pipe_importance"
WEAKEST_PIPE_KEY = "weakest_pipe"
PIPE_IMPORTANCE_ABLATION_KEY = "pipe_importance_ablation"
MIN_IMPROVEMENT = 0.002

# Minimal, documented mapping from the feature-block names used by ablation
# to the actual source filenames. This is the only place names are translated.
# It exists solely because the classical feature blocks and the .py files
# do not share identical names.
PIPE_TO_FILENAME = {
    "bayesian_fusion_norm": "bayesian_fusion.py",
    "monte_carlo": "monte_carlo.py",
    "redundancy": "redundancy.py",
    "markov_features": "markov.py",
    "entropy_features": "entropy.py",
    "centroids": "clustering.py",
    "clusters": "clustering.py",
    "quantum_features": "quantum_features.py",
    "quantum_kernels": "quantum_kernels.py",
}


def get_last_six_runs() -> list:
    """Return the last 6 full training batches summarised by run_date."""
    query = """
        WITH BatchStats AS (
            SELECT
                run_date,
                COUNT(*) AS num_epochs,
                AVG(val_auc) AS avg_val_auc,
                MAX(val_auc) AS peak_val_auc
            FROM epochs
            WHERE run_date IS NOT NULL
            GROUP BY run_date
        )
        SELECT run_date, num_epochs, avg_val_auc, peak_val_auc
        FROM BatchStats
        ORDER BY run_date DESC
        LIMIT 6
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.cursor().execute(query).fetchall()

    logging.info("=== Last 6 FULL Training Batches ===")
    for i, (date, epochs, avg, peak) in enumerate(rows, 1):
        avg_str = f"{avg:.4f}" if avg is not None else "N/A"
        peak_str = f"{peak:.4f}" if peak is not None else "N/A"
        logging.info(f"{i}. Date={date} | Epochs={epochs} | Avg={avg_str} | Peak={peak_str}")

    return rows


def compare_latest_to_previous(runs: list) -> tuple[bool, float]:
    """
    Returns (should_skip, numeric_delta).
    should_skip = True  → latest run improved → take no action
    numeric_delta       → latest − median of previous (positive = improvement)
    """
    if len(runs) < 2:
        return True, 0.0

    latest = runs[0][2]
    if latest is None:
        return False, 0.0

    previous = [r[2] for r in runs[1:] if r[2] is not None]
    if not previous:
        return True, 0.0

    median_prev = statistics.median(previous)
    delta = latest - median_prev

    if delta > MIN_IMPROVEMENT:
        logging.info(f"Latest batch IMPROVED (delta = {delta:+.4f})")
        return True, delta

    logging.info(f"Latest batch did NOT improve (delta = {delta:+.4f})")
    return False, delta


def build_decision_context(
    pipeline: Any,
    assessor: Any = None,
    observer: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Builds a clean decision context from ablation + assessment + observer.
    This is the single source of truth that later adaptor stages will consume.
    """
    if pipeline is None:
        logging.error("Pipeline object not provided.")
        return None

    # 1. Read what permutation + retrain ablation already decided
    importance = pipeline.get_data(PIPE_IMPORTANCE_KEY) or {}
    importance_ablation = pipeline.get_data(PIPE_IMPORTANCE_ABLATION_KEY) or {}
    weakest_pipe = pipeline.get_data(WEAKEST_PIPE_KEY)

    # weakest_pipe is set by deep_learning from get_weak_link / combined signal.
    # If missing, there is nothing actionable for Stage 1.
    if not weakest_pipe:
        logging.info("No actionable weak link identified.")
        return None

    context: Dict[str, Any] = {
        "weakest_pipe": weakest_pipe,
        "importance": importance or {},
        "importance_ablation": importance_ablation or {},
        "structural": None,
        "runtime_keys": None,
        "action": "methodology_review",
        "reason": "No tunable parameters discovered for this pipe.",
    }

    # 2. Structural information from assessment
    if assessor is not None:
        try:
            summary = assessor.get_pipe_summary()
            filename = PIPE_TO_FILENAME.get(weakest_pipe)
            if filename and filename in summary:
                context["structural"] = summary[filename]
            else:
                # Fallback: try a direct name match
                for fname, data in summary.items():
                    if weakest_pipe.replace("_", "") in fname.replace("_", ""):
                        context["structural"] = data
                        break
        except Exception as e:
            logging.warning(f"Could not retrieve structural info: {e}")

    # 3. Runtime context from observer (richer summary)
    if observer is not None:
        try:
            run_summary = observer.get_run_summary()
            if run_summary:
                context["runtime_keys"] = run_summary.get("keys", [])
                context["runtime_summary"] = run_summary.get("summaries", {})
                context["runtime_deltas"] = run_summary.get("deltas", {})
        except Exception as e:
            logging.warning(f"Could not retrieve runtime summary: {e}")
    return context

def run_optuna_bridge(
    pipeline: Any = None,
    assessor: Any = None,
    observer: Any = None,
) -> None:
    """
    Main entry point. Called after a full pipeline run.
    """
    runs = get_last_six_runs()
    if not runs:
        return

    should_skip, overall_delta = compare_latest_to_previous(runs)
    logging.info(f"Overall performance delta: {overall_delta:+.4f}")

    if should_skip:
        return

    context = build_decision_context(pipeline, assessor, observer)
    if context is None:
        return

    # Clean Stage-1 report
    logging.info("=== Stage 1 Decision Context ===")
    logging.info(f"Weak link          : {context['weakest_pipe']}")

    if context.get("importance"):
        val = context["importance"].get(context["weakest_pipe"])
        if val is not None:
            logging.info(f"Perm score         : {val:+.6f}")

    if context.get("importance_ablation"):
        val = context["importance_ablation"].get(context["weakest_pipe"])
        if val is not None:
            logging.info(f"Ablation score     : {val:+.6f}")

    structural = context.get("structural")
    if structural:
        logging.info(f"Source file        : {structural.get('file_path', 'N/A')}")
        logging.info(f"Functions          : {structural.get('function_count', 0)}")
        logging.info(f"Reads              : {structural.get('reads', [])}")
        logging.info(f"Outputs            : {structural.get('outputs', [])}")
        if structural.get("data_movement"):
            logging.info(f"Data movement      : {structural['data_movement']}")

    # Focused runtime summary for the weak pipe only
    runtime_summary = context.get("runtime_summary") or {}
    weak_output_key = None
    if structural and structural.get("outputs"):
        weak_output_key = structural["outputs"][0]

    if weak_output_key and weak_output_key in runtime_summary:
        logging.info(f"Runtime summary for '{weak_output_key}':")
        for k, v in runtime_summary[weak_output_key].items():
            logging.info(f"    {k}: {v}")
    else:
        logging.info("Runtime summary    : not available for weak pipe output")
    
    runtime_deltas = context.get("runtime_deltas") or {}
    if weak_output_key and weak_output_key in runtime_deltas:
        d = runtime_deltas[weak_output_key]
        if d.get("comparable"):
            logging.info(f"Feature delta      : {d.get('delta_mean', 0.0):+.6f}")
        else:
            logging.info(f"Feature delta      : not comparable ({d.get('reason', 'unknown')})")
    logging.info(f"Recommended action : {context['action']}")
    logging.info(f"Reason             : {context['reason']}")
    logging.info("================================")

    # Store the decision so later adaptor stages can read it
    if pipeline is not None:
        pipeline.add_data("adaptor_decision", context)


if __name__ == "__main__":
    run_optuna_bridge()
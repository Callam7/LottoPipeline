# Modified By: Callam
# Project: Lotto Generator
# Purpose: Optuna Bridge - Stage 1 (report-only decision context)
# Description:
#   - Compares latest training batch to recent history (peak val AUC)
#   - If improved: skip (no adaptor action)
#   - If not improved: emit a decision record
#       * real weak link → methodology_review on that pipe
#       * no weak link   → stack_review
#   - Pulls structure from assessment and runtime from the observer
#   - Does not rewrite code. Does not search hyperparameters.

import logging
import sqlite3
import statistics
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_PATH = "lotto.db"
PIPE_IMPORTANCE_KEY = "pipe_importance"
WEAKEST_PIPE_KEY = "weakest_pipe"
MIN_IMPROVEMENT = 0.005

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
    """Return the last 6 date-grouped training batches (peak + avg)."""
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
    Skip-gate uses peak val AUC (index 3), not epoch average (index 2).
    should_skip = True → latest peak beat the recent median by MIN_IMPROVEMENT.
    """
    if len(runs) < 2:
        return True, 0.0

    latest_peak = runs[0][3]
    if latest_peak is None:
        return False, 0.0

    previous_peaks = [r[3] for r in runs[1:] if r[3] is not None]
    if not previous_peaks:
        return True, 0.0

    median_prev = statistics.median(previous_peaks)
    delta = latest_peak - median_prev

    if delta > MIN_IMPROVEMENT:
        logging.info(f"Latest batch IMPROVED (peak delta = {delta:+.4f})")
        return True, delta

    logging.info(f"Latest batch did NOT improve (peak delta = {delta:+.4f})")
    return False, delta


def _resolve_runtime_key(
    weakest_pipe: Optional[str],
    runtime_summary: Dict[str, Any],
    structural: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Bind the logged tensor to the weak-link *name*.
    Never use structural['outputs'][0] — that is a sibling key
    (centroids would log clusters).
    """
    if not weakest_pipe:
        return None
    if weakest_pipe in runtime_summary:
        return weakest_pipe
    outputs = (structural or {}).get("outputs") or []
    if weakest_pipe in outputs:
        return weakest_pipe
    return None


def build_decision_context(
    pipeline: Any,
    assessor: Any = None,
    observer: Any = None,
) -> Optional[Dict[str, Any]]:
    if pipeline is None:
        logging.error("Pipeline object not provided.")
        return None

    importance = pipeline.get_data(PIPE_IMPORTANCE_KEY) or {}
    weakest_pipe = pipeline.get_data(WEAKEST_PIPE_KEY)

    if weakest_pipe:
        action = "methodology_review"
        reason = (
            f"Stage 1 report only. Weak link '{weakest_pipe}' "
            "is the candidate for later search/replacement."
        )
    else:
        action = "stack_review"
        reason = (
            "Stage 1 report only. No clear pipe weak link "
            "(all ablation scores ≈ 0). Review the stack as a whole."
        )
        logging.info("No actionable weak link identified — stack_review.")

    context: Dict[str, Any] = {
        "weakest_pipe": weakest_pipe,
        "importance": importance or {},
        "structural": None,
        "runtime_keys": None,
        "action": action,
        "reason": reason,
    }

    if assessor is not None and weakest_pipe:
        try:
            summary = assessor.get_pipe_summary()
            filename = PIPE_TO_FILENAME.get(weakest_pipe)
            if filename and filename in summary:
                context["structural"] = summary[filename]
            else:
                for fname, data in summary.items():
                    if weakest_pipe.replace("_", "") in fname.replace("_", ""):
                        context["structural"] = data
                        break
        except Exception as e:
            logging.warning(f"Could not retrieve structural info: {e}")

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
    # Finalize observer on every run, including skipped decisions.
    if observer is not None:
        try:
            observer.get_run_summary()
        except Exception as e:
            logging.warning(f"Observer finalize failed: {e}")

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

    logging.info("=== Stage 1 Decision Context ===")
    logging.info(f"Weak link          : {context['weakest_pipe']}")

    if context.get("weakest_pipe") and context.get("importance"):
        val = context["importance"].get(context["weakest_pipe"])
        if val is None:
            pass
        elif abs(val) < 1e-12:
            logging.info("Ablation score     : 0.0000")
        else:
            logging.info(f"Ablation score     : {val:+.6f}")

    structural = context.get("structural")
    if structural:
        logging.info(f"Source file        : {structural.get('file_path', 'N/A')}")
        logging.info(f"Functions          : {structural.get('function_count', 0)}")
        logging.info(f"Reads              : {structural.get('reads', [])}")
        logging.info(f"Outputs            : {structural.get('outputs', [])}")
        if structural.get("data_movement"):
            logging.info(f"Data movement      : {structural['data_movement']}")

    runtime_summary = context.get("runtime_summary") or {}
    weak_output_key = _resolve_runtime_key(
        context.get("weakest_pipe"),
        runtime_summary,
        structural,
    )

    if weak_output_key and weak_output_key in runtime_summary:
        logging.info(f"Runtime summary for '{weak_output_key}':")
        for k, v in runtime_summary[weak_output_key].items():
            logging.info(f"    {k}: {v}")
    else:
        logging.info("Runtime summary: not available for this block")

    runtime_deltas = context.get("runtime_deltas") or {}
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
        pipeline.add_data("adaptor_decision", context)


if __name__ == "__main__":
    run_optuna_bridge()
"""
Optuna Bridge - Stage 1 Only
Purpose:
- Read pipe_importance from deep learning runs
- Identify the pipe with the largest positive contribution (most negative delta)
- Generate reasonable Optuna suggestions for that pipe
- Nothing else. No rewriting. No self-adaptation yet.
"""

import logging
import sqlite3
import statistics
import optuna

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DB_PATH = "lotto.db"
PIPE_IMPORTANCE_KEY = "pipe_importance"
MIN_IMPROVEMENT = 0.002

# Stage 1: Known parameter spaces for pipes we can currently suggest on.
# Keep this small and honest. Only add pipes when we actually have meaningful parameters to tune.
KNOWN_PARAMETER_SPACES = {
    "entropy_features": {
        "entropy_weight": (0.5, 2.0),
        "entropy_smoothing": (1e-6, 1e-2, "log"),
    },
    "bayesian_fusion_norm": {
        "fusion_weight_freq": (0.5, 1.5),
        "fusion_weight_decay": (0.5, 1.5),
        "fusion_weight_mechanics": (0.0, 1.0),
    },
    "monte_carlo": {
        "mc_weight": (0.5, 2.0),
    },
    "clusters": {
        "n_clusters_main": (3, 8, "int"),
        "n_clusters_powerball": (2, 5, "int"),
    },
    "centroids": {
        "n_clusters_main": (3, 8, "int"),
        "n_clusters_powerball": (2, 5, "int"),
    },
}


def get_last_six_runs():
    query = """
        WITH BatchStats AS (
            SELECT run_date,
                   COUNT(*) as num_epochs,
                   AVG(val_auc) as avg_val_auc,
                   MAX(val_auc) as peak_val_auc
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


def compare_latest_to_previous(runs):
    if len(runs) < 2:
        return True
    latest = runs[0][2]
    if latest is None:
        return False
    previous = [r[2] for r in runs[1:] if r[2] is not None]
    if not previous:
        return True
    median_prev = statistics.median(previous)
    if latest > (median_prev + MIN_IMPROVEMENT):
        logging.info(f"Latest batch IMPROVED ({latest:.4f} > median {median_prev:.4f})")
        return True
    logging.info(f"Latest batch did NOT improve ({latest:.4f}, median {median_prev:.4f})")
    return False


def perform_full_pipe_ablation(pipeline):
    if pipeline is None:
        logging.error("Pipeline object not provided.")
        return None

    importance = pipeline.get_data(PIPE_IMPORTANCE_KEY)
    if not importance or not isinstance(importance, dict):
        logging.warning("No valid pipe_importance found.")
        return None

    most_impactful = min(importance, key=importance.get)
    delta = importance[most_impactful]
    logging.info(f"Most impactful pipe: {most_impactful} (delta={delta:.6f})")
    return most_impactful


def get_optuna_suggestion(pipe_name: str):
    study = optuna.create_study(
        study_name="lotto_pipeline_adaptor",
        storage=f"sqlite:///{DB_PATH}",
        load_if_exists=True,
        direction="maximize"
    )
    trial = study.ask()

    if pipe_name in KNOWN_PARAMETER_SPACES:
        space = KNOWN_PARAMETER_SPACES[pipe_name]
        suggestion = {}
        for param, bounds in space.items():
            if len(bounds) == 3 and bounds[2] == "int":
                suggestion[param] = trial.suggest_int(param, bounds[0], bounds[1])
            elif len(bounds) == 3 and bounds[2] == "log":
                suggestion[param] = trial.suggest_float(param, bounds[0], bounds[1], log=True)
            else:
                suggestion[param] = trial.suggest_float(param, bounds[0], bounds[1])
    else:
        # Honest fallback for pipes we cannot yet suggest meaningful parameters for
        suggestion = {}
        logging.info(f"No defined parameter space for '{pipe_name}'. No specific suggestion generated.")

    logging.info(f"Optuna suggestion for {pipe_name}: {suggestion}")
    return suggestion


def run_optuna_bridge(pipeline=None):
    runs = get_last_six_runs()
    if not runs:
        return

    if compare_latest_to_previous(runs):
        return

    if pipeline is None:
        logging.warning("No pipeline object provided.")
        return

    most_impactful_pipe = perform_full_pipe_ablation(pipeline)
    if most_impactful_pipe is None:
        return

    suggestion = get_optuna_suggestion(most_impactful_pipe)
    if suggestion:
        logging.info(f"=== Stage 1 Suggestion Ready ===")
        logging.info(f"Pipe: {most_impactful_pipe}")
        logging.info(f"Suggested changes: {suggestion}")


if __name__ == "__main__":
    run_optuna_bridge()
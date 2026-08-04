# Modified By: Callam
# Project: Lotto Generator
# Purpose: Runtime Observation Layer for Self-Adaptive Pipeline
# Description:
#   - Captures actual runtime data flowing through the pipeline
#   - Produces compact per-key summaries for the decision layer
#   - Works together with assessment.py (structural layer) and DataPipeline
#   - Non-intrusive: only requires the observer hooks already added to DataPipeline
#   - Designed to survive future code rewriting by the adaptor

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RuntimeObserver:
    """
    Observes and records all data that passes through the DataPipeline at runtime.

    Provides both the raw snapshot and compact per-key summaries so the
    decision layer (optuna_bridge) can work with useful information
    without receiving large arrays.
    """

    def __init__(self, pipeline) -> None:
        """
        Initialise the observer and register it with the DataPipeline.
        """
        self.pipeline = pipeline
        self.current_run_id: Optional[str] = None
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}

        if hasattr(pipeline, "register_observer"):
            pipeline.register_observer(self)
            logging.info("RuntimeObserver successfully registered with DataPipeline.")
        else:
            logging.warning(
                "DataPipeline does not have register_observer(). "
                "Observer will not receive automatic notifications."
            )

    def start_new_run(self, run_id: Optional[str] = None) -> str:
        """Start a new observation session for the current pipeline run."""
        self.current_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.snapshots[self.current_run_id] = {}
        logging.info(f"Started new runtime observation run: {self.current_run_id}")
        return self.current_run_id

    def record_add_data(self, key: str, value: Any) -> None:
        """Called by DataPipeline when add_data() is executed."""
        if self.current_run_id is None:
            logging.debug("No active run. Ignoring record_add_data call.")
            return
        self.snapshots[self.current_run_id][key] = value
        logging.debug(f"Recorded add_data: key='{key}' for run {self.current_run_id}")

    def record_get_data(self, key: str, value: Any) -> None:
        """Optional. Currently unused."""
        pass

    def get_snapshot(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the full raw snapshot for a run."""
        run_id = run_id or self.current_run_id
        if run_id is None:
            logging.warning("No run_id provided and no current run is active.")
            return {}
        return self.snapshots.get(run_id, {})

    def get_value(self, key: str, run_id: Optional[str] = None) -> Any:
        """Return a single value from a run snapshot."""
        snapshot = self.get_snapshot(run_id)
        return snapshot.get(key)

    def _summarise_value(self, value: Any) -> Dict[str, Any]:
        """
        Produce a compact summary of a single recorded value.
        Avoids storing or returning large arrays.
        """
        summary: Dict[str, Any] = {
            "type": type(value).__name__,
        }

        if value is None:
            summary["note"] = "None"
            return summary

        # NumPy arrays and array-like
        if isinstance(value, np.ndarray):
            summary["shape"] = value.shape
            summary["dtype"] = str(value.dtype)

            if value.size > 0 and np.issubdtype(value.dtype, np.number):
                flat = value.astype(float).ravel()
                summary["min"] = float(np.min(flat))
                summary["max"] = float(np.max(flat))
                summary["mean"] = float(np.mean(flat))
                summary["std"] = float(np.std(flat))

                # Simple probability-vector heuristic
                if flat.ndim == 1 and flat.size > 1:
                    total = float(np.sum(flat))
                    if 0.98 <= total <= 1.02:
                        summary["looks_like_probability"] = True
                        # Shannon entropy (natural log)
                        p = flat[flat > 0]
                        if p.size > 0:
                            summary["entropy"] = float(-np.sum(p * np.log(p)))
                    else:
                        summary["looks_like_probability"] = False

            return summary

        # Python lists / tuples of numbers
        if isinstance(value, (list, tuple)) and len(value) > 0:
            try:
                arr = np.asarray(value, dtype=float)
                summary["length"] = len(value)
                summary["min"] = float(np.min(arr))
                summary["max"] = float(np.max(arr))
                summary["mean"] = float(np.mean(arr))
            except (ValueError, TypeError):
                summary["length"] = len(value)
            return summary

        # Simple scalars
        if isinstance(value, (int, float, bool, str)):
            summary["value"] = value
            return summary

        # Fallback
        summary["note"] = "complex object (not summarised)"
        return summary

    def get_run_summary(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Return a compact, useful summary of the entire run.
        This is the primary method the decision layer should call.
        """
        run_id = run_id or self.current_run_id
        snapshot = self.get_snapshot(run_id)

        if not snapshot:
            return {
                "run_id": run_id,
                "keys": [],
                "summaries": {},
            }

        summaries = {}
        for key, value in snapshot.items():
            summaries[key] = self._summarise_value(value)

        return {
            "run_id": run_id,
            "keys": list(snapshot.keys()),
            "summaries": summaries,
        }

    def record_final_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store final training/evaluation metrics for the current run."""
        if self.current_run_id is None:
            logging.warning("Cannot record metrics - no active run.")
            return
        self.metrics[self.current_run_id] = metrics
        logging.info(f"Recorded final metrics for run {self.current_run_id}")

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Return the metrics from the most recent completed run."""
        if not self.metrics:
            return None
        latest_run = max(self.metrics.keys())
        return self.metrics[latest_run]

    def enable_ablation_mode(self, neutralized_keys: List[str]) -> None:
        """
        Placeholder for future runtime ablation support.
        Not used by the current post-training grouped permutation method.
        """
        logging.info(f"Ablation mode requested for keys: {neutralized_keys}")
        # Future implementation reserved
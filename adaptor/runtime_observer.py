# Modified By: Callam
# Project: Lotto Generator
# Purpose: Runtime Observation Layer for Self-Adaptive Pipeline
# Description:
#   - Captures runtime pipeline.add_data values
#   - Compact per-key summaries for optuna_bridge
#   - Previous-run snapshot taken in start_new_run (not in get_run_summary)
#   - Deltas use entropy, then std — never mean of a simplex

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IDENTICAL_TOL = 1e-12


class RuntimeObserver:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline
        self.current_run_id: Optional[str] = None
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._previous_summary: Optional[Dict[str, Any]] = None

        if hasattr(pipeline, "register_observer"):
            pipeline.register_observer(self)
            logging.info("RuntimeObserver successfully registered with DataPipeline.")
        else:
            logging.warning(
                "DataPipeline does not have register_observer(). "
                "Observer will not receive automatic notifications."
            )

    def _freeze_current_as_previous(self) -> None:
        """Store compact stats for the run that is about to be replaced."""
        if self.current_run_id is None:
            return
        snapshot = self.snapshots.get(self.current_run_id) or {}
        if not snapshot:
            return
        summaries = {k: self._summarise_value(v) for k, v in snapshot.items()}
        self._previous_summary = {
            "run_id": self.current_run_id,
            "keys": list(snapshot.keys()),
            "summaries": summaries,
        }

    def start_new_run(self, run_id: Optional[str] = None) -> str:
        """Freeze the last run, then open a new snapshot."""
        self._freeze_current_as_previous()
        self.current_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.snapshots[self.current_run_id] = {}
        logging.info(f"Started new runtime observation run: {self.current_run_id}")
        return self.current_run_id

    def record_add_data(self, key: str, value: Any) -> None:
        if self.current_run_id is None:
            logging.debug("No active run. Ignoring record_add_data call.")
            return
        self.snapshots[self.current_run_id][key] = value
        logging.debug(f"Recorded add_data: key='{key}' for run {self.current_run_id}")

    def record_get_data(self, key: str, value: Any) -> None:
        pass

    def get_snapshot(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        run_id = run_id or self.current_run_id
        if run_id is None:
            logging.warning("No run_id provided and no current run is active.")
            return {}
        return self.snapshots.get(run_id, {})

    def get_value(self, key: str, run_id: Optional[str] = None) -> Any:
        return self.get_snapshot(run_id).get(key)

    def _summarise_value(self, value: Any) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"type": type(value).__name__}

        if value is None:
            summary["note"] = "None"
            return summary

        if isinstance(value, np.ndarray):
            summary["shape"] = value.shape
            summary["dtype"] = str(value.dtype)
            if value.size > 0 and np.issubdtype(value.dtype, np.number):
                flat = value.astype(float).ravel()
                summary["min"] = float(np.min(flat))
                summary["max"] = float(np.max(flat))
                summary["mean"] = float(np.mean(flat))
                summary["std"] = float(np.std(flat))
                if flat.size > 1:
                    total = float(np.sum(flat))
                    if flat.min() >= -1e-12 and 0.98 <= total <= 1.02:
                        summary["looks_like_probability"] = True
                        p = np.clip(flat, 0.0, None)
                        s = float(p.sum())
                        if s > 0:
                            p = p / s
                            p_nz = p[p > 0]
                            summary["entropy"] = float(-np.sum(p_nz * np.log(p_nz)))
                            uniform = 1.0 / flat.size
                            summary["l1_vs_uniform"] = float(np.sum(np.abs(p - uniform)))
                    else:
                        summary["looks_like_probability"] = False
            return summary

        if isinstance(value, (list, tuple)) and len(value) > 0:
            try:
                arr = np.asarray(value, dtype=float).ravel()
                summary["length"] = len(value)
                summary["min"] = float(np.min(arr))
                summary["max"] = float(np.max(arr))
                summary["mean"] = float(np.mean(arr))
                summary["std"] = float(np.std(arr))
            except (ValueError, TypeError):
                summary["length"] = len(value)
            return summary

        if isinstance(value, (int, float, bool, str)):
            summary["value"] = value
            return summary

        summary["note"] = "complex object (not summarised)"
        return summary

    def _compute_delta_from_summaries(
        self, current: Dict[str, Any], previous: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comparable delta order:
          1) entropy (probability vectors)
          2) std
          3) mean only if NOT a probability vector
        """
        try:
            curr_prob = bool(current.get("looks_like_probability"))
            prev_prob = bool(previous.get("looks_like_probability"))

            if "entropy" in current and "entropy" in previous:
                de = float(current["entropy"]) - float(previous["entropy"])
                identical = abs(de) < IDENTICAL_TOL
                if "std" in current and "std" in previous:
                    identical = identical and abs(
                        float(current["std"]) - float(previous["std"])
                    ) < IDENTICAL_TOL
                return {
                    "comparable": True,
                    "identical": identical,
                    "delta_entropy": de,
                    "curr_entropy": float(current["entropy"]),
                    "prev_entropy": float(previous["entropy"]),
                }

            if "std" in current and "std" in previous:
                ds = float(current["std"]) - float(previous["std"])
                return {
                    "comparable": True,
                    "identical": abs(ds) < IDENTICAL_TOL,
                    "delta_std": ds,
                    "curr_std": float(current["std"]),
                    "prev_std": float(previous["std"]),
                }

            if (
                "mean" in current
                and "mean" in previous
                and not curr_prob
                and not prev_prob
            ):
                dm = float(current["mean"]) - float(previous["mean"])
                return {
                    "comparable": True,
                    "identical": abs(dm) < IDENTICAL_TOL,
                    "delta_mean": dm,
                    "curr_mean": float(current["mean"]),
                    "prev_mean": float(previous["mean"]),
                }

            return {"comparable": False, "reason": "no shared comparable statistic"}
        except (TypeError, ValueError, KeyError):
            return {"comparable": False, "reason": "non-numeric summary"}

    def get_run_summary(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Pure read. Does not update previous-run memory."""
        run_id = run_id or self.current_run_id
        snapshot = self.get_snapshot(run_id)

        if not snapshot:
            return {
                "run_id": run_id,
                "keys": [],
                "summaries": {},
                "deltas": {},
                "previous_run_id": (
                    self._previous_summary.get("run_id")
                    if self._previous_summary
                    else None
                ),
            }

        summaries = {key: self._summarise_value(value) for key, value in snapshot.items()}

        deltas: Dict[str, Any] = {}
        previous_run_id = None
        if self._previous_summary is not None:
            previous_run_id = self._previous_summary.get("run_id")
            prev_summaries = self._previous_summary.get("summaries", {})
            for key, curr in summaries.items():
                if key in prev_summaries:
                    deltas[key] = self._compute_delta_from_summaries(
                        curr, prev_summaries[key]
                    )
                else:
                    deltas[key] = {
                        "comparable": False,
                        "reason": "key absent in previous run",
                    }

        return {
            "run_id": run_id,
            "keys": list(snapshot.keys()),
            "summaries": summaries,
            "deltas": deltas,
            "previous_run_id": previous_run_id,
        }

    def record_final_metrics(self, metrics: Dict[str, Any]) -> None:
        if self.current_run_id is None:
            logging.warning("Cannot record metrics - no active run.")
            return
        self.metrics[self.current_run_id] = metrics
        logging.info(f"Recorded final metrics for run {self.current_run_id}")

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        if not self.metrics:
            return None
        latest_run = max(self.metrics.keys())
        return self.metrics[latest_run]

    def enable_ablation_mode(self, neutralized_keys: List[str]) -> None:
        logging.info(f"Ablation mode requested for keys: {neutralized_keys}")
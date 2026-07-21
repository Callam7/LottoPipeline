# Modified By: Callam
# Project: Lotto Generator
# Purpose: Runtime Observation Layer for Self-Adaptive Pipeline
# Description:
#   - Captures actual runtime data flowing through the pipeline
#   - Works together with assessment.py (structural layer) and DataPipeline
#   - Provides per-run snapshots for analysis, ablation testing, and Optuna decisions
#   - Non-intrusive: only requires the observer hooks already added to DataPipeline
#   - Designed to survive future code rewriting by the adaptor

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

# Configure logging format and level (matches pipeline.py style)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RuntimeObserver:
    """
    Observes and records all data that passes through the DataPipeline at runtime.

    This class works as an observer that is automatically notified
    whenever add_data() is called on the pipeline. It stores per-run
    snapshots so the adaptor (Optuna bridge + ablation) can later query
    exactly what values were produced and how they affected performance.

    The observer is designed to be:
      - Non-intrusive (does not require changes inside step files)
      - Robust (handles missing runs gracefully)
      - Future-proof (easy to extend for ablation and persistence)
    """

    def __init__(self, pipeline) -> None:
        """
        Initialise the observer and register it with the DataPipeline.

        Args:
            pipeline: The DataPipeline instance (must have register_observer method).
        """
        self.pipeline = pipeline
        self.current_run_id: Optional[str] = None
        self.snapshots: Dict[str, Dict[str, Any]] = {}      # run_id -> {key: value}
        self.metrics: Dict[str, Dict[str, Any]] = {}        # run_id -> final metrics

        # Register with the pipeline (the hooks were added in pipeline.py)
        if hasattr(pipeline, "register_observer"):
            pipeline.register_observer(self)
            logging.info("RuntimeObserver successfully registered with DataPipeline.")
        else:
            logging.warning(
                "DataPipeline does not have register_observer(). "
                "Observer will not receive automatic notifications."
            )

    def start_new_run(self, run_id: Optional[str] = None) -> str:
        """
        Start a new observation session for the current pipeline run.

        Args:
            run_id (Optional[str]): Custom run identifier. If None, a timestamp is generated.

        Returns:
            str: The run_id that was started.
        """
        self.current_run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.snapshots[self.current_run_id] = {}
        logging.info(f"Started new runtime observation run: {self.current_run_id}")
        return self.current_run_id

    def record_add_data(self, key: str, value: Any) -> None:
        """
        Automatically called by DataPipeline when add_data() is executed.

        Stores the key-value pair in the current run's snapshot.
        """
        if self.current_run_id is None:
            logging.debug("No active run. Ignoring record_add_data call.")
            return

        self.snapshots[self.current_run_id][key] = value
        logging.debug(f"Recorded add_data: key='{key}' for run {self.current_run_id}")

    def record_get_data(self, key: str, value: Any) -> None:
        """
        Optional method to record data reads (get_data calls).

        Currently left as pass-through for future debugging if needed.
        """
        pass

    def get_snapshot(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve the full snapshot of data recorded during a specific run.

        Args:
            run_id (Optional[str]): The run to retrieve. Defaults to current run.

        Returns:
            Dict[str, Any]: Dictionary of all recorded key-value pairs.
        """
        run_id = run_id or self.current_run_id
        if run_id is None:
            logging.warning("No run_id provided and no current run is active.")
            return {}
        return self.snapshots.get(run_id, {})

    def get_value(self, key: str, run_id: Optional[str] = None) -> Any:
        """
        Retrieve a single value from a run snapshot.

        Args:
            key (str): The data key to look up.
            run_id (Optional[str]): The run to search. Defaults to current run.

        Returns:
            Any: The stored value, or None if not found.
        """
        snapshot = self.get_snapshot(run_id)
        return snapshot.get(key)

    def record_final_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Store the final training/evaluation metrics for the current run.

        This is typically called at the end of deep_learning_prediction.

        Args:
            metrics (Dict[str, Any]): Dictionary containing val_auc, loss, etc.
        """
        if self.current_run_id is None:
            logging.warning("Cannot record metrics - no active run.")
            return

        self.metrics[self.current_run_id] = metrics
        logging.info(f"Recorded final metrics for run {self.current_run_id}")

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Return the metrics from the most recent completed run.

        Returns:
            Optional[Dict[str, Any]]: Metrics dict or None if no runs exist.
        """
        if not self.metrics:
            return None

        latest_run = max(self.metrics.keys())
        return self.metrics[latest_run]

    def enable_ablation_mode(self, neutralized_keys: List[str]) -> None:
        """
        Placeholder for future ablation support.

        When implemented, this method will allow temporarily overriding
        specific pipeline values during a run for controlled testing.
        """
        logging.info(f"Ablation mode requested for keys: {neutralized_keys}")
        # Future implementation will go here (used by ablation.py)
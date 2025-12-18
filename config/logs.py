## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Epoch Logging Utility
## Description:
## Provides a custom Keras callback that logs deep learning training metrics
## into the SQLite 'epochs' table. Each training run is grouped by a unique run_date.
## Adds pacing between inserts to prevent DB contention.

from datetime import datetime
import time
import tensorflow as tf
from database import insert_epoch_metrics  # Uses your existing DB manager


class EpochLogger(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to log epoch metrics into the 'epochs' table.
    Runs after each epoch completes, ensuring metrics are stored consistently.
    """

    def __init__(self, delay=0.1):
        """
        Parameters:
        - delay (float): Seconds to pause after each insert. Helps SQLite process steadily.
        """
        super().__init__()
        self.run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.delay = delay

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

    # ---- Metric normalization layer (CRITICAL FIX) ----

        def _get(*keys, default=0.0):
            for k in keys:
                if k in logs:
                    return float(logs[k])
            return float(default)

        try:
            insert_epoch_metrics(
            run_date=self.run_date,
            epoch=epoch + 1,

            loss=_get("loss"),
            val_loss=_get("val_loss"),

            # Accept BOTH legacy and current metric names
            binary_accuracy=_get("binary_accuracy", "bin_acc"),
            val_binary_accuracy=_get("val_binary_accuracy", "val_bin_acc"),

            auc=_get("auc"),
            val_auc=_get("val_auc"),

            mae=_get("mae"),
            val_mae=_get("val_mae"),
        )
            time.sleep(self.delay)

        except Exception as e:
            print(f"[EpochLogger] Error inserting metrics: {e}")



def get_run_date():
    """
    Utility to fetch the most recent run_date string (for grouping results).
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

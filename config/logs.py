## Modified By: Callam
## Project: Lotto Predictor
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
        """
        Called automatically by Keras after each epoch.
        Inserts metrics into the 'epochs' table, then pauses briefly.
        """
        if logs is None:
            return

        try:
            insert_epoch_metrics(
                run_date=self.run_date,
                epoch=epoch + 1,  # store epochs as 1-based instead of 0-based
                loss=float(logs.get("loss", 0.0)),
                val_loss=float(logs.get("val_loss", 0.0)),
                binary_accuracy=float(logs.get("binary_accuracy", 0.0)),
                val_binary_accuracy=float(logs.get("val_binary_accuracy", 0.0)),
                auc=float(logs.get("auc", 0.0)),
                val_auc=float(logs.get("val_auc", 0.0)),
                mae=float(logs.get("mae", 0.0)),
                val_mae=float(logs.get("val_mae", 0.0)),
            )
            time.sleep(self.delay)  # pacing
        except Exception as e:
            print(f"[EpochLogger] Error inserting metrics: {e}")


def get_run_date():
    """
    Utility to fetch the most recent run_date string (for grouping results).
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

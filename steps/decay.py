## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: To Calculate Decay-Weighted Lottery Number Frequencies
## Description:
## This file applies a decay factor to historical lottery draw data, assigning more weight to recent draws
## and less weight to older ones. Decay is calculated based on the time difference between each draw and the
## most recent draw. The normalized decay-weighted frequency distributions are stored in the data pipeline
## for downstream use in prediction models.

import numpy as np
from datetime import datetime
import logging

try:
    from dateutil import parser
except ImportError:
    parser = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
NUM_MAIN = 40
NUM_POWERBALL = 10
TOTAL_NUMBERS = NUM_MAIN + NUM_POWERBALL  # 50


def _safe_parse_date(date_value):
    """
    Robustly parse a date into a datetime.

    Accepts:
        - datetime objects (returned as-is)
        - strings in "%Y-%m-%d" format
        - other string formats if python-dateutil is available

    Raises:
        ValueError if parsing fails.
    """
    if isinstance(date_value, datetime):
        return date_value

    if date_value is None:
        raise ValueError("Empty/None date value")

    # Normalise to string
    date_str = str(date_value)

    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        if parser is not None:
            # Let dateutil have a go; will raise if it cannot parse
            return parser.parse(date_str)
        raise ValueError(f"Unparseable date: {date_str!r}")


def calculate_decay_factors(pipeline, decay_rate: float = 0.98):
    """
    Calculates decay-weighted frequency distributions for main numbers (1-40)
    and Powerball numbers (1-10), normalizes separately, and concatenates into
    a single shape-(50,) array stored in the pipeline as 'decay_factors'.

    More recent draws receive higher weight (decay_factor close to 1),
    older draws receive exponentially smaller weights.
    """
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        logging.warning("No historical data available for decay calculations.")
        pipeline.add_data("decay_factors", np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS)
        return

    # ---- Step 1: parse dates safely and filter out unusable draws ----
    dated_draws = []
    for i, draw in enumerate(historical_data):
        d_str = draw.get("draw_date") or draw.get("date")
        if not d_str:
            logging.warning(
                f"Skipping draw at index {i}: missing 'draw_date'/'date' field."
            )
            continue
        try:
            dt = _safe_parse_date(d_str)
        except Exception as e:
            logging.warning(
                f"Skipping draw at index {i} due to unparseable date {d_str!r}: {e}"
            )
            continue
        dated_draws.append((draw, dt))

    if not dated_draws:
        logging.warning(
            "All historical draws had invalid or missing dates; using uniform decay_factors."
        )
        pipeline.add_data("decay_factors", np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS)
        return

    # ---- Step 2: determine reference (most recent) date ----
    latest_date = max(dt for _, dt in dated_draws)

    main_frequency = np.zeros(NUM_MAIN, dtype=float)
    powerball_frequency = np.zeros(NUM_POWERBALL, dtype=float)

    # ---- Step 3: accumulate decay-weighted counts ----
    for draw, dt in dated_draws:
        delta_days = (latest_date - dt).days
        if delta_days < 0:
            # Should not really happen once latest_date is max, but guard anyway.
            logging.warning(
                f"Encountered draw dated after latest_date ({dt} > {latest_date}); "
                f"using zero age for decay."
            )
            delta_days = 0

        weeks_passed = delta_days / 7.0
        decay_factor = decay_rate ** weeks_passed

        # Main numbers
        for num in draw.get("numbers", []) or []:
            if isinstance(num, int) and 1 <= num <= NUM_MAIN:
                main_frequency[num - 1] += decay_factor
            else:
                logging.warning(
                    f"Invalid main number {num!r} in draw dated "
                    f"{draw.get('draw_date') or draw.get('date')!r}; ignored."
                )

        # Powerball(s)
        pb = draw.get("powerball")
        if pb is None:
            continue

        if isinstance(pb, list):
            pb_iter = pb
        else:
            pb_iter = [pb]

        for p in pb_iter:
            if isinstance(p, int) and 1 <= p <= NUM_POWERBALL:
                powerball_frequency[p - 1] += decay_factor
            else:
                logging.warning(
                    f"Invalid Powerball value {p!r} in draw dated "
                    f"{draw.get('draw_date') or draw.get('date')!r}; ignored."
                )

    # ---- Step 4: normalise separately and concatenate ----
    main_sum = main_frequency.sum()
    if main_sum > 0.0:
        main_frequency /= main_sum
    else:
        logging.warning(
            "Main frequency sum is zero after decay; assigning uniform main distribution."
        )
        main_frequency = np.ones(NUM_MAIN, dtype=float) / NUM_MAIN

    pb_sum = powerball_frequency.sum()
    if pb_sum > 0.0:
        powerball_frequency /= pb_sum
    else:
        logging.warning(
            "Powerball frequency sum is zero after decay; assigning uniform PB distribution."
        )
        powerball_frequency = np.ones(NUM_POWERBALL, dtype=float) / NUM_POWERBALL

    combined_frequency = np.concatenate([main_frequency, powerball_frequency])
    pipeline.add_data("decay_factors", combined_frequency)
    logging.info("Decay-weighted frequency calculation completed.")



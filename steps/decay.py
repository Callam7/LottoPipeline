## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: To Calculate Decay-Weighted Lottery Number Frequencies
## Description:
## This file applies a decay factor to historical lottery draw data, assigning more weight to recent draws
## and less weight to older ones. Decay is calculated based on the time elapsed between draws, assuming a
## weekly interval. The normalized decay-weighted frequency distributions are stored in the data pipeline
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

def _safe_parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        if parser:
            return parser.parse(date_str)
        else:
            raise

def calculate_decay_factors(pipeline, decay_rate=0.98):
    """
    Calculates decay-weighted frequency distributions for main numbers (1-40)
    and Powerball numbers (1-10), normalizes separately, and concatenates into
    a single shape 50 array stored in the pipeline as 'decay_factors'.
    """
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        logging.warning("No historical data available for decay calculations.")
        pipeline.add_data("decay_factors", np.ones(TOTAL_NUMBERS) / TOTAL_NUMBERS)
        return

    try:
        historical_data.sort(key=lambda x: _safe_parse_date(x['draw_date']))
    except (KeyError, ValueError) as e:
        logging.error(f"Error sorting historical data by 'draw_date': {e}")
        return

    main_frequency = np.zeros(NUM_MAIN)
    powerball_frequency = np.zeros(NUM_POWERBALL)
    last_date = None

    for i, draw in enumerate(historical_data):
        try:
            current_date = _safe_parse_date(draw['draw_date'])
        except (ValueError, KeyError) as e:
            logging.warning(f"Skipping draw at index {i} due to invalid/missing 'draw_date': {e}")
            continue

        if last_date:
            delta_days = (current_date - last_date).days
            if delta_days < 0:
                logging.warning(
                    f"Draw at index {i} has earlier date than previous draw. "
                    f"Using zero days for decay."
                )
                delta_days = 0
            weeks_passed = max(0, delta_days / 7)
            decay_factor = decay_rate ** weeks_passed
        else:
            decay_factor = 1.0

        # Main numbers
        for num in draw.get("numbers", []):
            if 1 <= num <= NUM_MAIN:
                main_frequency[num - 1] += decay_factor
            else:
                logging.warning(f"Invalid main number {num} in draw {draw.get('draw_date', 'unknown')}.")

        # Powerball
        pball = draw.get("powerball")
        if pball is not None and 1 <= pball <= NUM_POWERBALL:
            powerball_frequency[pball - 1] += decay_factor
        elif pball is not None:
            logging.warning(f"Invalid Powerball number {pball} in draw {draw.get('draw_date', 'unknown')}.")

        last_date = current_date

    # Normalize separately
    if main_frequency.sum() > 0:
        main_frequency /= main_frequency.sum()
    else:
        main_frequency = np.ones(NUM_MAIN) / NUM_MAIN
        logging.warning("Main frequency sum zero; assigned uniform distribution.")

    if powerball_frequency.sum() > 0:
        powerball_frequency /= powerball_frequency.sum()
    else:
        powerball_frequency = np.ones(NUM_POWERBALL) / NUM_POWERBALL
        logging.warning("Powerball frequency sum zero; assigned uniform distribution.")

    # Concatenate into shape 50
    combined_frequency = np.concatenate([main_frequency, powerball_frequency])
    pipeline.add_data("decay_factors", combined_frequency)



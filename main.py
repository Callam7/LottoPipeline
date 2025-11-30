## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Main Program Execution (Updated for Correct Pipeline Order)
## Description:
## Entry point for the Lotto Predictor program. Handles database, pipeline execution,
## ticket generation, and user interface. Displays main + Powerball frequency in option 4.

# -*- coding: utf-8 -*-

import os
import numpy as np
from datetime import datetime

# Database functions
from database import (
    initialize_database,
    fetch_recent_draws,
    fetch_all_draws,
    insert_draw
)

# Data I/O
from data_io import load_current_ticket, save_current_ticket

# Pipeline structure
from pipeline import DataPipeline
from steps.generate_ticket import generate_ticket


# Pipeline steps
from steps.historical import process_historical_data
from steps.frequency import analyze_number_frequency
from steps.decay import calculate_decay_factors
from steps.clustering import kmeans_clustering_and_correlation
from steps.monte_carlo import monte_carlo_simulation
from steps.redundancy import sequential_features
from steps.markov import markov_features
from steps.entropy import shannon_entropy_features
from steps.bayesian_fusion import bayesian_fusion_with_mechanics
from steps.deep_learning import deep_learning_prediction

# Constants
NUM_PICK_MAIN = 6
MAX_MAIN_NUMBER = 40
NUM_POWERBALL = 10
TICKET_LINES = 12


# ============================================================
# Utility Functions
# ============================================================
def verify_draw_order():
    """Verify that draw_id reflects chronological order."""
    all_draws = fetch_all_draws()
    if not all_draws:
        return
    dates = [draw['draw_date'] for draw in all_draws]
    if dates == sorted(dates):
        print("Verification Passed: draw_id correctly reflects chronological order.")
    else:
        print("Verification Failed: draw_id does NOT correctly reflect chronological order.")


def get_latest_draw_date():
    """Return the most recent draw date."""
    all_draws = fetch_all_draws()
    if not all_draws:
        return None
    latest_draw = all_draws[-1]
    try:
        return datetime.strptime(latest_draw['draw_date'], "%Y-%m-%d")
    except ValueError:
        return None


def view_number_stats(pipeline):
    """Display frequency stats for main numbers and Powerball."""
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        print("No historical data in pipeline. Fetching from database...")
        all_draws = fetch_all_draws()
        if not all_draws:
            print("No draws in database.")
            return
        pipeline.add_data("historical_data", all_draws)
        historical_data = all_draws

    number_frequency = pipeline.get_data("number_frequency")
    powerball_frequency = pipeline.get_data("powerball_frequency")

    if number_frequency is None or powerball_frequency is None:
        print("Frequency data missing. Running analysis...")
        analyze_number_frequency(pipeline)
        number_frequency = pipeline.get_data("number_frequency")
        powerball_frequency = pipeline.get_data("powerball_frequency")

    print("\n--- Main Numbers Frequency (1..40) ---")
    print("Number | Occurrences | % of main picks")
    total_main_picks = len(historical_data) * NUM_PICK_MAIN
    for i in range(MAX_MAIN_NUMBER):
        count = int(number_frequency[i] * total_main_picks)
        percent = number_frequency[i] * 100
        print(f"{i+1:2d}     | {count:10d}   | {percent:6.2f}%")

    print("\n--- Powerball Frequency (1..10) ---")
    print("Number | Occurrences | % of Powerball picks")
    total_powerball_picks = len(historical_data)
    for i in range(NUM_POWERBALL):
        count = int(powerball_frequency[i] * total_powerball_picks)
        percent = powerball_frequency[i] * 100
        print(f"{i+1:2d}     | {count:10d}   | {percent:6.2f}%")


# ============================================================
# Safe Execution Wrapper
# ============================================================
def safe_run(step_fn, pipeline, name):
    """Safely execute pipeline stage with error handling."""
    try:
        step_fn(pipeline)
        print(f"[OK] {name} completed.")
    except Exception as e:
        print(f"[ERROR] {name} failed: {e}")


# ============================================================
# Main Program Loop
# ============================================================
def main():
    initialize_database()
    verify_draw_order()
    pipeline = DataPipeline()

    while True:
        print("\n--- Lotto Predictor Menu ---")
        print("1. Display Current Ticket")
        print("2. List Last 10 Results (from DB)")
        print("3. Insert New Draw & Generate Ticket")
        print("4. Number Stats")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            current_ticket_data = load_current_ticket()
            current_ticket = current_ticket_data.get("current_ticket", [])
            if not current_ticket:
                print("No current ticket. Generate one first.")
            else:
                print("\n--- Current Ticket ---")
                for idx, line in enumerate(current_ticket, 1):
                    print(f"Line {idx}: {line['line']} | Powerball: {line['powerball']}")

        elif choice == "2":
            last_draws = fetch_recent_draws(10)
            if not last_draws:
                print("No historical draws.")
            else:
                print("\n--- Last 10 Draws ---")
                for draw in last_draws:
                    print(f"Date: {draw['draw_date']} | Numbers: {draw['numbers']} | "
                          f"Bonus: {draw['bonus']} | Powerball: {draw['powerball']}")

        elif choice == "3":
            draw_date = input("Enter draw date (YYYY-MM-DD) or press Enter for today: ")
            if not draw_date:
                draw_date = datetime.now().strftime("%Y-%m-%d")
            else:
                try:
                    datetime.strptime(draw_date, "%Y-%m-%d")
                except ValueError:
                    print("Invalid date format.")
                    continue

            latest_date = get_latest_draw_date()
            if latest_date and datetime.strptime(draw_date, "%Y-%m-%d") <= latest_date:
                print(f"Error: New draw date must be after {latest_date.strftime('%Y-%m-%d')}.")
                continue

            try:
                numbers = list(map(int, input(f"Enter {NUM_PICK_MAIN} main numbers (1-{MAX_MAIN_NUMBER}): ").split()))
                if len(numbers) != NUM_PICK_MAIN or len(set(numbers)) != NUM_PICK_MAIN:
                    raise ValueError(f"Exactly {NUM_PICK_MAIN} distinct numbers required.")
                if any(n < 1 or n > MAX_MAIN_NUMBER for n in numbers):
                    raise ValueError(f"Numbers must be between 1 and {MAX_MAIN_NUMBER}.")
                bonus = int(input(f"Enter bonus number (1-{MAX_MAIN_NUMBER}): "))
                if not (1 <= bonus <= MAX_MAIN_NUMBER):
                    raise ValueError(f"Bonus must be between 1 and {MAX_MAIN_NUMBER}.")
                powerball = int(input(f"Enter Powerball (1-{NUM_POWERBALL}): "))
                if not (1 <= powerball <= NUM_POWERBALL):
                    raise ValueError(f"Powerball must be between 1 and {NUM_POWERBALL}.")
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue

            new_id = insert_draw(draw_date, numbers, bonus, powerball)
            if not new_id:
                print("Error inserting draw.")
                continue
            print(f"New draw inserted with draw_id = {new_id}")

            # --- Full Pipeline Execution ---
            all_draws = fetch_all_draws()
            pipeline.clear_pipeline()
            pipeline.add_data("historical_data", all_draws)

            safe_run(lambda p: process_historical_data({"past_results": all_draws}, p), pipeline, "Historical Processing")
            safe_run(analyze_number_frequency, pipeline, "Frequency Analysis")
            safe_run(calculate_decay_factors, pipeline, "Decay Calculation")
            safe_run(bayesian_fusion_with_mechanics, pipeline, "Bayesian Fusion")
            safe_run(kmeans_clustering_and_correlation, pipeline, "Clustering")
            safe_run(monte_carlo_simulation, pipeline, "Monte Carlo Simulation")
            safe_run(sequential_features, pipeline, "Sequential/Redundancy")  
            safe_run(markov_features, pipeline, "Markov Features")
            safe_run(shannon_entropy_features, pipeline, "Entropy Features")
            safe_run(deep_learning_prediction, pipeline, "Deep Learning Prediction")
            # --- Generate Ticket ---
            new_ticket = generate_ticket(pipeline)
            print("\nNew ticket generated:")
            for idx, line in enumerate(new_ticket, 1):
                print(f"Line {idx}: {line['line']} | Powerball: {line['powerball']}")

        elif choice == "4":
            view_number_stats(pipeline)

        elif choice == "5":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Select 1-5.")


if __name__ == "__main__":
    main()


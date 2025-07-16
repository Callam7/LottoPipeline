## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: Main Program Execution
## Description:
## This file serves as the entry point for the Lotto Predictor program.
## It handles the database initialization, user menu, and step-by-step execution of the pipeline to generate and analyze lottery tickets.
## The program ensures data integrity and provides a user-friendly interface for managing draws and generating predictions.

# -*- coding: utf-8 -*-

import os  # For operating system-related tasks
import numpy as np  # For numerical computations
from datetime import datetime  # For date and time handling

# Database functions
from database import (
    initialize_database,
    fetch_recent_draws,
    fetch_all_draws,
    insert_draw,
    fetch_draw_by_date
)

# Data I/O functions
from data_io import load_current_ticket, save_current_ticket

# Pipeline structure and generation
from pipeline import DataPipeline, generate_ticket

# Pipeline steps
from steps.historical import process_historical_data
from steps.frequency import analyze_number_frequency
from steps.decay import calculate_decay_factors
from steps.clustering import kmeans_clustering_and_correlation
from steps.monte_carlo import monte_carlo_simulation
from steps.redundancy import sequential_features  # Newly added redundancy import

from steps.deep_learning import deep_learning_prediction

def verify_draw_order():
    """
    Verifies that the draws in the database are ordered chronologically by draw_date.
    Ensures that draw_id corresponds to the correct order.
    """
    all_draws = fetch_all_draws()
    if not all_draws:
        return
    dates = [draw['draw_date'] for draw in all_draws]
    sorted_dates = sorted(dates)
    if dates == sorted_dates:
        print("Verification Passed: draw_id correctly reflects chronological order.")
    else:
        print("Verification Failed: draw_id does NOT correctly reflect chronological order.")

def get_latest_draw_date():
    """
    Retrieves the latest draw_date from the database.
    Returns:
    - datetime: The latest draw date as a datetime object, or None if no draws exist.
    """
    all_draws = fetch_all_draws()
    if not all_draws:
        return None
    latest_draw = all_draws[-1]  # Assumes fetch_all_draws is ordered ascending
    try:
        latest_date = datetime.strptime(latest_draw['draw_date'], "%Y-%m-%d")
        return latest_date
    except ValueError:
        return None

def view_number_stats(pipeline):
    """
    Computes and displays the frequency analysis of lottery numbers based on historical data.
    Ensures frequency data is present in the pipeline and fetches it if necessary.
    """
    # Check for historical data
    historical_data = pipeline.get_data("historical_data")
    if not historical_data:
        print("No historical data available in the pipeline. Fetching from database...")
        all_draws = fetch_all_draws()
        if not all_draws:
            print("No draws in the database. Please insert some first.")
            return
        pipeline.add_data("historical_data", all_draws)
        historical_data = all_draws

    # Compute frequency data if not already present
    number_frequency = pipeline.get_data("number_frequency")
    if number_frequency is None:
        print("Frequency data is not available. Running frequency analysis...")
        analyze_number_frequency(pipeline)
        number_frequency = pipeline.get_data("number_frequency")

    # Display frequency statistics
    if isinstance(number_frequency, np.ndarray) and len(number_frequency) == 40:
        total_main_picks = len(historical_data) * 6
        print("\nNumber Frequency (1..40)")
        print("-------------------------------------------------")
        print("Number | Occurrences |   % of all main picks")
        print("-------------------------------------------------")
        for i in range(40):
            count = number_frequency[i] * total_main_picks  # Denormalize frequency
            percent = number_frequency[i] * 100
            print(f"{i + 1:2d}     | {int(count):10d}   | {percent:8.2f}%")
    else:
        print("Frequency data is invalid. Please ensure the pipeline is functioning correctly.")

def main():
    """
    Main function to execute the Lotto Predictor program.
    Handles the user menu and coordinates database operations, pipeline execution, and ticket generation.
    """
    # Initialize the database
    initialize_database()

    # Verify that draw order is correct
    verify_draw_order()

    # Create a new data pipeline
    pipeline = DataPipeline()

    while True:
        # Display the main menu
        print("\n--- Lotto Predictor Menu ---")
        print("1. Display Current Ticket")
        print("2. List Last 10 Results (from DB)")
        print("3. Insert New Draw & Generate Ticket (DB -> pipeline)")
        print("4. Number Stats")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            # Display the current ticket
            current_ticket_data = load_current_ticket()
            current_ticket = current_ticket_data.get("current_ticket", [])
            if not current_ticket:
                print("No current ticket found. Generate a new ticket first.")
            else:
                print("\n--- Current Ticket ---")
                for idx, line in enumerate(current_ticket, 1):
                    print(f"Line {idx}: {line['line']} | Powerball: {line['powerball']}")

        elif choice == "2":
            # List the last 10 draws from the database
            last_draws = fetch_recent_draws(10)
            if not last_draws:
                print("No historical draws available.")
            else:
                print("\n--- Last 10 Draws ---")
                for draw in reversed(last_draws):
                    print(f"Date: {draw['draw_date']} | "
                          f"Numbers: {draw['numbers']} | "
                          f"Bonus: {draw['bonus']} | "
                          f"Powerball: {draw['powerball']}")

        elif choice == "3":
            # Insert a new draw and generate a ticket
            draw_date = input("Enter draw date (YYYY-MM-DD) or press Enter for today's date: ")
            if not draw_date:
                draw_date = datetime.now().strftime("%Y-%m-%d")
            else:
                try:
                    datetime.strptime(draw_date, "%Y-%m-%d")
                except ValueError:
                    print("Invalid date format. Please use YYYY-MM-DD.")
                    continue

            # Validate the new draw date
            latest_date = get_latest_draw_date()
            if latest_date and datetime.strptime(draw_date, "%Y-%m-%d") <= latest_date:
                print(f"Error: New draw date must be after {latest_date.strftime('%Y-%m-%d')}.")
                continue

            try:
                # Collect draw details from the user
                numbers = list(map(int, input("Enter the 6 main numbers separated by spaces: ").split()))
                if len(numbers) != 6 or len(set(numbers)) != 6:
                    raise ValueError("Exactly 6 distinct numbers are required.")
                if any(n < 1 or n > 40 for n in numbers):
                    raise ValueError("Main numbers must be between 1 and 40.")
                bonus = int(input("Enter the bonus number (1-40): "))
                if not (1 <= bonus <= 40):
                    raise ValueError("Bonus number must be between 1 and 40.")
                powerball = int(input("Enter the Powerball number (1-10): "))
                if not (1 <= powerball <= 10):
                    raise ValueError("Powerball number must be between 1 and 10.")
            except ValueError as e:
                print(f"Invalid input: {e}")
                continue

            # Insert the new draw
            new_id = insert_draw(draw_date, numbers, bonus, powerball)
            if new_id:
                print(f"New draw inserted with draw_id = {new_id}")
            else:
                print("Error inserting draw. It might already exist or data is invalid.")
                continue

            # Run the pipeline
            all_draws = fetch_all_draws()
            pipeline.clear_pipeline()
            process_historical_data({"past_results": all_draws}, pipeline)
            analyze_number_frequency(pipeline)
            calculate_decay_factors(pipeline)
            kmeans_clustering_and_correlation(pipeline)
            monte_carlo_simulation(pipeline)
            # Add redundancy step before Monte Carlo
            sequential_features(pipeline)
            deep_learning_prediction(pipeline)

            # Generate and save the ticket
            new_ticket = generate_ticket(pipeline)
            save_current_ticket(new_ticket)
            print("\nNew ticket generated and saved to current_ticket.json:")
            for idx, line in enumerate(new_ticket, 1):
                print(f"Line {idx}: {line['line']} | Powerball: {line['powerball']}")

        elif choice == "4":
            # Display number statistics
            view_number_stats(pipeline)

        elif choice == "5":
            # Exit the program
            print("Exiting program.")
            break

        else:
            print("Invalid choice. Please select a number between 1 and 5.")

if __name__ == "__main__":
    main()

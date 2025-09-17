## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Manage Input/Output for Generated Ticket Data
## Description:
## This file provides functions to handle the saving and loading of generated lottery ticket data
## to and from the `current_ticket.json` file. The file ensures data integrity, handles errors gracefully,
## and logs relevant information for debugging.

import json  # For JSON read/write operations
import os  # For checking file existence
import logging  # For logging events and errors
from typing import Any, Dict, List  # For type annotations

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for file paths
CURRENT_TICKET_FILE = "current_ticket.json"  # The file storing the current ticket data

def load_current_ticket() -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads the currently generated ticket lines from `current_ticket.json`.

    Expected JSON structure:
    {
        "current_ticket": [
            {"line": [int, int, ...], "powerball": int},
            ...
        ]
    }

    Returns:
    - Dict[str, List[Dict[str, Any]]]: The loaded ticket data. If the file does not exist or contains
      invalid JSON, returns an empty ticket structure.
    """
    # Check if the file exists
    if not os.path.exists(CURRENT_TICKET_FILE):
        logging.warning(f"'{CURRENT_TICKET_FILE}' not found. Returning empty ticket structure.")
        return {"current_ticket": []}

    try:
        # Open and load JSON data from the file
        with open(CURRENT_TICKET_FILE, "r") as f:
            data = json.load(f)
            # Validate that the file contains the expected structure
            if "current_ticket" not in data or not isinstance(data["current_ticket"], list):
                logging.error(f"Invalid structure in '{CURRENT_TICKET_FILE}'. Expected 'current_ticket' as a list.")
                return {"current_ticket": []}
            logging.info(f"Successfully loaded ticket data from '{CURRENT_TICKET_FILE}'.")
            return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from '{CURRENT_TICKET_FILE}': {e}. Returning empty ticket structure.")
        return {"current_ticket": []}
    except Exception as e:
        logging.error(f"Unexpected error loading '{CURRENT_TICKET_FILE}': {e}. Returning empty ticket structure.")
        return {"current_ticket": []}

def save_current_ticket(ticket: List[Dict[str, Any]]) -> None:
    """
    Saves the generated ticket lines to `current_ticket.json`.

    Each ticket entry should be a dictionary with:
        - "line": List[int] - A list of main lottery numbers.
        - "powerball": int - The Powerball number.

    Args:
    - ticket (List[Dict[str, Any]]): A list of ticket entries to be saved.

    Raises:
    - ValueError: If the ticket data is not in the expected format.
    """
    # Validate that the ticket data is a list
    if not isinstance(ticket, list):
        logging.error("Ticket data must be a list of dictionaries.")
        raise ValueError("Invalid ticket format: Expected a list of dictionaries.")

    normalized_ticket = []  # To store validated and cleaned ticket entries
    for idx, line_dict in enumerate(ticket):
        # Validate that each entry is a dictionary
        if not isinstance(line_dict, dict):
            logging.warning(f"Skipping invalid ticket entry at index {idx}: Expected a dictionary.")
            continue
        # Ensure each entry contains the required keys
        if "line" not in line_dict or "powerball" not in line_dict:
            logging.warning(f"Skipping ticket entry at index {idx}: Missing 'line' or 'powerball' key.")
            continue
        # Validate that all numbers are integers
        try:
            main_line = [int(num) for num in line_dict["line"]]
            pball = int(line_dict["powerball"])
        except (ValueError, TypeError) as e:
            logging.warning(f"Skipping ticket entry at index {idx} due to invalid number types: {e}.")
            continue
        # Add the cleaned entry to the normalized ticket list
        normalized_ticket.append({"line": main_line, "powerball": pball})

    # Prepare the data to save
    data_to_save = {"current_ticket": normalized_ticket}

    try:
        # Write the validated data to the file
        with open(CURRENT_TICKET_FILE, "w") as f:
            json.dump(data_to_save, f, indent=2)
        logging.info(f"Successfully saved {len(normalized_ticket)} ticket line(s) to '{CURRENT_TICKET_FILE}'.")
    except Exception as e:
        logging.error(f"Failed to save ticket data to '{CURRENT_TICKET_FILE}': {e}.")

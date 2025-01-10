## Modified By: Callam
## Project: Lotto Predictor
## Purpose of File: Database Initialization and Management
## Description:
## This file defines functions for initializing and interacting with the SQLite database `lotto.db`.
## It handles the creation of the 'draws' table and provides utility functions for inserting,
## fetching, and managing lottery draw data.

import sqlite3  # For SQLite database operations
from sqlite3 import Error  # To handle SQLite-specific exceptions
from datetime import datetime  # For handling date and time operations

DB_FILENAME = "lotto.db"  # The name of the SQLite database file

def get_connection():
    """
    Creates (or opens) the lotto.db file and returns a connection.

    Returns:
    - sqlite3.Connection: The database connection object, or None if an error occurs.
    """
    try:
        conn = sqlite3.connect(DB_FILENAME)
        return conn
    except Error as e:
        print("Error connecting to SQLite:", e)
        return None

def initialize_database():
    """
    Ensures the 'draws' table exists in the database.
    Creates the table if it does not already exist.
    """
    conn = get_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()

        # Create or ensure the 'draws' table exists
        create_draws_table_sql = """
        CREATE TABLE IF NOT EXISTS draws (
            draw_id INTEGER PRIMARY KEY,  -- Auto-generated unique ID
            draw_date TEXT NOT NULL UNIQUE,  -- Unique date of the draw
            numbers TEXT NOT NULL,          -- Comma-separated main numbers
            bonus INTEGER NOT NULL CHECK (bonus BETWEEN 1 AND 40),
            powerball INTEGER NOT NULL CHECK (powerball BETWEEN 1 AND 10)
        );
        """
        cursor.execute(create_draws_table_sql)

        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print("Error during database initialization:", e)

def insert_draw(draw_date, numbers, bonus, powerball):
    """
    Inserts a single draw record into the 'draws' table.

    Parameters:
    - draw_date (str): The date of the draw in "YYYY-MM-DD" format.
    - numbers (list of int): List of 6 distinct integers representing main numbers.
    - bonus (int): The bonus number.
    - powerball (int): The Powerball number.

    Returns:
    - int: The newly inserted draw_id, or None if an error occurs.
    """
    conn = get_connection()
    if not conn:
        return None

    numbers_str = ",".join(map(str, numbers))  # Convert list to comma-separated string

    try:
        cursor = conn.cursor()
        # Retrieve the current maximum draw_id
        cursor.execute("SELECT MAX(draw_id) FROM draws")
        result = cursor.fetchone()
        max_draw_id = result[0] if result[0] is not None else 0
        new_draw_id = max_draw_id + 1

        sql = """
        INSERT INTO draws (draw_id, draw_date, numbers, bonus, powerball)
        VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(sql, (new_draw_id, draw_date, numbers_str, bonus, powerball))
        conn.commit()
        cursor.close()
        conn.close()
        return new_draw_id
    except sqlite3.IntegrityError as e:
        print(f"IntegrityError inserting draw on {draw_date}: {e}")
        return None
    except Error as e:
        print("Error inserting draw:", e)
        return None

def fetch_all_draws():
    """
    Fetches all draw records from the database.

    Returns:
    - list of dict: Each dictionary represents a draw with keys:
      - "draw_date": str
      - "numbers": list of int
      - "bonus": int
      - "powerball": int
    """
    conn = get_connection()
    if not conn:
        return []

    draws_list = []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT draw_date, numbers, bonus, powerball FROM draws ORDER BY draw_date ASC")
        rows = cursor.fetchall()
        for (draw_date, nums_str, bonus, powerball) in rows:
            try:
                num_list = list(map(int, nums_str.split(",")))
            except ValueError:
                continue  # Skip invalid number formats

            draw_dict = {
                "draw_date": draw_date,
                "numbers": num_list,
                "bonus": bonus,
                "powerball": powerball
            }
            draws_list.append(draw_dict)
        cursor.close()
        conn.close()
    except Error as e:
        print("Error fetching draws:", e)

    return draws_list

def fetch_recent_draws(limit=10):
    """
    Fetches the most recent 'limit' draw records from the database.

    Parameters:
    - limit (int): The maximum number of recent draws to fetch.

    Returns:
    - list of dict: Each dictionary represents a draw, similar to fetch_all_draws.
    """
    conn = get_connection()
    if not conn:
        return []

    draws_list = []
    try:
        cursor = conn.cursor()
        sql = """
        SELECT draw_date, numbers, bonus, powerball
        FROM draws
        ORDER BY draw_date DESC
        LIMIT ?
        """
        cursor.execute(sql, (limit,))
        rows = cursor.fetchall()
        for (draw_date, nums_str, bonus, powerball) in rows:
            try:
                num_list = list(map(int, nums_str.split(",")))
            except ValueError:
                continue  # Skip invalid number formats

            draw_dict = {
                "draw_date": draw_date,
                "numbers": num_list,
                "bonus": bonus,
                "powerball": powerball
            }
            draws_list.append(draw_dict)
        cursor.close()
        conn.close()
    except Error as e:
        print("Error fetching recent draws:", e)
    return draws_list

def fetch_draw_by_date(draw_date):
    """
    Fetches a single draw record by its draw_date.

    Parameters:
    - draw_date (str): The date of the draw to fetch.

    Returns:
    - dict: The draw record, or None if not found.
    """
    conn = get_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor()
        sql = """
        SELECT draw_date, numbers, bonus, powerball
        FROM draws
        WHERE draw_date = ?
        """
        cursor.execute(sql, (draw_date,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row:
            try:
                num_list = list(map(int, row[1].split(",")))
            except ValueError:
                return None
            return {
                "draw_date": row[0],
                "numbers": num_list,
                "bonus": row[2],
                "powerball": row[3]
            }
        else:
            return None
    except Error as e:
        print("Error fetching draw by date:", e)
        return None

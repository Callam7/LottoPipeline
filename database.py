## Modified By: Callam
## Project: Lotto Generator
## Purpose of File: Database Initialization and Management
## Description:
## This file defines functions for initializing and interacting with the SQLite database lotto.db.
## It handles the creation of the 'draws' table and the new 'epochs' table for deep learning epoch metrics.
## Provides utility functions for inserting, fetching, and managing lottery draw and epoch data.

import sqlite3
from sqlite3 import Error
from datetime import datetime

DB_FILENAME = "lotto.db"  # The SQLite database filename


def get_connection():
    """
    Creates (or opens) the lotto.db file and returns a connection.
    Returns:
    - sqlite3.Connection object or None on error.
    """
    try:
        conn = sqlite3.connect(DB_FILENAME)
        return conn
    except Error as e:
        print("Error connecting to SQLite:", e)
        return None


def initialize_database():
    """
    Ensures the 'draws' and 'epochs' tables exist in the database.
    Creates them if they do not already exist.
    """
    conn = get_connection()
    if not conn:
        return
    try:
        cursor = conn.cursor()

        # Draws table
        create_draws_table_sql = """
        CREATE TABLE IF NOT EXISTS draws (
            draw_id INTEGER PRIMARY KEY,       -- Stable unique ID
            draw_date TEXT NOT NULL UNIQUE,    -- Unique date of the draw
            numbers TEXT NOT NULL,             -- Comma-separated main numbers
            bonus INTEGER NOT NULL CHECK (bonus BETWEEN 1 AND 40),
            powerball INTEGER NOT NULL CHECK (powerball BETWEEN 1 AND 10)
        );
        """
        cursor.execute(create_draws_table_sql)

        # Epochs table
        create_epochs_table_sql = """
        CREATE TABLE IF NOT EXISTS epochs (
            id INTEGER PRIMARY KEY,
            run_date TEXT NOT NULL,             -- Date/time grouping this training run
            epoch INTEGER NOT NULL,             -- Epoch number within this run
            loss REAL NOT NULL,
            val_loss REAL NOT NULL,
            binary_accuracy REAL NOT NULL,
            val_binary_accuracy REAL NOT NULL,
            auc REAL NOT NULL,
            val_auc REAL NOT NULL,
            mae REAL NOT NULL,
            val_mae REAL NOT NULL
        );
        """
        cursor.execute(create_epochs_table_sql)

        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print("Error during database initialization:", e)


def insert_draw(draw_date, numbers, bonus, powerball):
    """
    Inserts a single draw record into the 'draws' table.
    Keeps stable draw_id rather than autoincrement.
    """
    conn = get_connection()
    if not conn:
        return None
    numbers_str = ",".join(map(str, numbers))
    try:
        cursor = conn.cursor()
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
    """
    conn = get_connection()
    if not conn:
        return []
    draws_list = []
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT draw_date, numbers, bonus, powerball FROM draws ORDER BY draw_date ASC"
        )
        rows = cursor.fetchall()
        for (draw_date, nums_str, bonus, powerball) in rows:
            try:
                num_list = list(map(int, nums_str.split(",")))
            except ValueError:
                continue
            draws_list.append({
                "draw_date": draw_date,
                "numbers": num_list,
                "bonus": bonus,
                "powerball": powerball
            })
        cursor.close()
        conn.close()
        return draws_list
    except Error as e:
        print("Error fetching draws:", e)
        return []


def fetch_recent_draws(limit=10):
    """
    Fetches the most recent 'limit' draw records.
    """
    conn = get_connection()
    if not conn:
        return []
    draws_list = []
    try:
        cursor = conn.cursor()
        sql = """
        SELECT draw_date, numbers, bonus, powerball
        FROM draws ORDER BY date(draw_date) DESC LIMIT ?
        """
        cursor.execute(sql, (limit,))
        rows = cursor.fetchall()
        for (draw_date, nums_str, bonus, powerball) in rows:
            try:
                num_list = list(map(int, nums_str.split(",")))
            except ValueError:
                continue
            draws_list.append({
                "draw_date": draw_date,
                "numbers": num_list,
                "bonus": bonus,
                "powerball": powerball
            })
        cursor.close()
        conn.close()
        return draws_list
    except Error as e:
        print("Error fetching recent draws:", e)
        return []


def fetch_draw_by_date(draw_date):
    """
    Fetches a draw record by date.
    """
    conn = get_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        sql = """
        SELECT draw_date, numbers, bonus, powerball
        FROM draws WHERE draw_date = ?
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


def insert_epoch_metrics(run_date, epoch, loss, val_loss,
                         binary_accuracy, val_binary_accuracy,
                         auc, val_auc, mae, val_mae):
    """
    Inserts a single epoch's metrics into the 'epochs' table.
    """
    conn = get_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        sql = """
        INSERT INTO epochs (
            run_date, epoch, loss, val_loss,
            binary_accuracy, val_binary_accuracy,
            auc, val_auc, mae, val_mae
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(sql, (
            run_date, epoch, loss, val_loss,
            binary_accuracy, val_binary_accuracy,
            auc, val_auc, mae, val_mae
        ))
        conn.commit()
        row_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return row_id
    except Error as e:
        print("Error inserting epoch metrics:", e)
        return None


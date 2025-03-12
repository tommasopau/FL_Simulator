from db import get_engine, create_tables, Base
import sqlite3
import os

def clear_database():
    engine = get_engine()
    Base.metadata.drop_all(engine)
    create_tables(engine)
    print("simulation.db has been cleared and tables recreated.")

def get_columns(db_path: str = None, table_name: str = 'simulation_results') -> list:
    """
    Retrieves the column names of a specified table from the SQLite database.

    Args:
        db_path (str, optional): Path to the SQLite database file. Defaults to None.
        table_name (str, optional): Name of the table to retrieve columns from. Defaults to 'simulation_results'.

    Returns:
        list: A list of column names.
    """
    try:
        engine = get_engine(db_path)
        db_file = engine.url.database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        conn.close()
        return columns
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    clear_database()
    columns = get_columns()
    print("Columns in the 'simulation_results' table:", columns)
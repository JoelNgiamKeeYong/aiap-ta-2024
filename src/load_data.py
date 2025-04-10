# load_data.py
# This script loads the dataset from a SQLite database and returns it as a pandas DataFrame.

import sqlite3
import pandas as pd
from pathlib import Path

def load_data(db_path, table_name):
    """
    Load the dataset from the specified SQLite database path and table name.
    
    Args:
        db_path (str): Path to the SQLite database file.
        table_name (str): Name of the table to load data from.
    
    Returns:
        pandas.DataFrame: The dataset loaded into a DataFrame.
    
    Raises:
        FileNotFoundError: If the database file is not found.
        ValueError: If the specified table does not exist in the database.
        sqlite3.Error: If there's an error connecting to or querying the database.
    """
    print("📥 Starting data loading process...")

    # Ensure the path is a Path object
    db_path = Path(db_path)
    
    # Check if the database file exists
    if not db_path.exists():
        raise FileNotFoundError(f"❌ Database file not found at {db_path}. Please ensure the dataset is in the correct location.")
    
    try:
        print(f"   └── Connecting to SQLite database at path: {db_path}...")
        
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        
        # Check if the table exists in the database
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        print(f"   └── Checking if table '{table_name}' exists in the database...")
        if table_name not in tables:
            conn.close()
            raise ValueError(f"❌ Table '{table_name}' not found in the database. Available tables: {tables}")
        
        # Query to load all data from the specified table
        query = f"SELECT * FROM {table_name}"
        print(f"   └── Loading data from table '{table_name}' into a pandas DataFrame...")
        data = pd.read_sql_query(query, conn)
        
        # Close the connection
        conn.close()
        
        print(f"✅ Successfully loaded {len(data):,} records from the database!")
        return data
    
    except sqlite3.Error as e:
        raise sqlite3.Error(f"❌ Database error: {e}") from e
    except Exception as e:
        raise Exception(f"❌ Error loading data: {e}") from e

if __name__ == "__main__":
    data = load_data()
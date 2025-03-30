import sqlite3
import pandas as pd
from pathlib import Path

def load_data(db_path="/root/data/noshow.db"):
    """
    Load the dataset from the specified SQLite database path.
    
    Args:
        db_path (str): Path to the SQLite database file (default: /root/data/noshow.db).
    
    Returns:
        pandas.DataFrame: The dataset loaded into a DataFrame.
    
    Raises:
        FileNotFoundError: If the database file is not found.
        sqlite3.Error: If there's an error connecting to or querying the database.
    """
    db_path = Path(db_path)
    
    # Check if the database file exists
    if not db_path.exists():
        raise FileNotFoundError(f"❌ Database file not found at {db_path}. Please ensure the dataset is in the correct location. 📂")
    
    try:
        print(f"📡 Connecting to database at {db_path}...")
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        
        # Query to load all data (assuming the table name is 'bookings' - adjust if needed)
        query = "SELECT * FROM noshow"
        print(f"📊 Loading data from table 'noshow'...")
        data = pd.read_sql_query(query, conn)
        
        # Close the connection
        conn.close()
        
        print(f"✅ Successfully loaded {len(data)} records! 🎉")
        return data
    
    except sqlite3.Error as e:
        raise sqlite3.Error(f"❌ Database error: {e} 😞") from e
    except Exception as e:
        raise Exception(f"❌ Error loading data: {e} 😞") from e

if __name__ == "__main__":
    try:
        # Test the load_data function
        data = load_data()
        print("First few rows of the dataset:")
        print(data.head())
    except Exception as e:
        print(f"❌ Error: {e}")
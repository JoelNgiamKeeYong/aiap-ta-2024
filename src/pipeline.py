import sqlite3
import pandas as pd
from preprocess import preprocess_data
from model import train_and_evaluate  # Adjust import based on your file structure
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the no-show prediction pipeline")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logistic_regression", "xgboost"])
    parser.add_argument("--preprocess", default="standard", choices=["standard", "advanced"])
    args = parser.parse_args()

    print("📡 Connecting to database at data\\noshow.db...")
    conn = sqlite3.connect("data/noshow.db")
    print("📊 Loading data from table 'noshow'...")
    data = pd.read_sql_query("SELECT * FROM noshow", conn)
    print(f"✅ Successfully loaded {len(data)} records! 🎉")
    conn.close()

    # Unpack the tuple returned by preprocess_data
    X, y = preprocess_data(data, method=args.preprocess)
    results = train_and_evaluate(X, y, model_type=args.model)  # Pass X and y separately

if __name__ == "__main__":
    main()
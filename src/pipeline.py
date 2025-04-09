# pipeline.py
# This script orchestrates the entire pipeline for the project.

import argparse
from load_data import load_data 
from preprocess_data import preprocess_data
from train_model import train_and_evaluate  

db_path="data/noshow.db"

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the no-show prediction pipeline")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logistic_regression", "xgboost"])
    args = parser.parse_args()

    # Step 1: Load the dataset
    data = load_data(db_path=db_path)

    # Step 2: Preprocess the dataset, including cleaning and feature engineering
    X, y = preprocess_data(data)



    # Preprocess the data
    

    # Train and evaluate the model
    print(f"🤖 Training and evaluating model ({args.model})...")
    results = train_and_evaluate(X, y, model_type=args.model)  # Pass X and y separately
    print("✅ Model training and evaluation completed!")

if __name__ == "__main__":
    main()
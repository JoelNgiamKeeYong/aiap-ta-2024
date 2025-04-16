# src/pipeline.py
# This script orchestrates the entire pipeline for the project.

import os
import time
import argparse
import matplotlib
import yaml
from load_data import load_data 
from clean_data import clean_data
from preprocess_data import preprocess_data
from train_and_evaluate_model import train_and_evaluate_models  
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def main():
    # Start timer
    start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the no-show prediction pipeline")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logistic_regression", "xgboost"])
    args = parser.parse_args()

    # Load configuration from YAML file
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set environment variables for parallel processing
    os.environ['LOKY_MAX_CPU_COUNT'] = config["LOKY_MAX_CPU_COUNT"]

    # Set the backend for matplotlib to avoid display issues
    matplotlib.use('Agg') 

    # Step 1: Load the dataset
    df = load_data(config["db_path"], config["db_table_name"])  

    # Step 2: Clean the dataset
    df_cleaned = clean_data(df)

    # Step 3: Perform feature engineering



    # Step 3: Preprocess the dataset, including cleaning and feature engineering
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df_cleaned, config["test_size"], config["random_state"])

    # Step 4: Define the models and their respective hyperparameter grids
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=config["random_state"]),
            "params": config["model_configuration"]['Logistic Regression']['params']
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', random_state=config["random_state"]),
            "params": config["model_configuration"]['XGBoost']['params']
        },
        "LightGBM": {  
            "model": LGBMClassifier(verbose=-1, force_row_wise=True, random_state=config["random_state"]),
            "params": config["model_configuration"]['LightGBM']['params'] 
        }
    }    


    # Step 4: Traing and evaluate models
    train_and_evaluate_models(
        models,
        X_train, X_test, y_train, y_test,
        config["n_jobs"], config["cv_folds"], config["scoring_metric"], feature_names
    )

    # Record time taken for pipeline execution
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\n✅ Completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
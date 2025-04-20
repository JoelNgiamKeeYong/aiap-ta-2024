# src/pipeline.py
# This script orchestrates the entire pipeline for the project.

import os
import time
import yaml
import argparse
import matplotlib
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Importing custom modules
from load_data import load_data 
from clean_data import clean_data
from build_models import build_models
from evaluate_models import evaluate_models

def main():
    # Start timer
    start_time = time.time()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the ML prediction pipeline")
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "logistic_regression", "xgboost"])
    args = parser.parse_args()
    # To add more configurations here
    ###
    ###

    # Load configuration from YAML file
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration values
    LOKY_MAX_CPU_COUNT = config["LOKY_MAX_CPU_COUNT"]
    DB_PATH = config["db_path"]
    DB_TABLE_NAME = config["db_table_name"]
    TARGET = config["target"]
    TEST_SIZE = config["test_size"]
    RANDOM_STATE = config["random_state"]
    N_JOBS = config["n_jobs"]
    CV_FOLDS = config["cv_folds"]
    SCORING_METRIC = config["scoring_metric"]
    LR_PARAMS = config["model_configuration"]["Logistic Regression"]["params"]
    XG_PARAMS = config["model_configuration"]["XGBoost"]["params"]
    LGBM_PARAMS = config["model_configuration"]["LightGBM"]["params"]

    # Set environment variables for parallel processing and matplotlib backend
    os.environ['LOKY_MAX_CPU_COUNT'] = LOKY_MAX_CPU_COUNT 
    matplotlib.use('Agg')  # Non-interactive backend for plotting

    # Step 1: Load the dataset
    df = load_data(db_path=DB_PATH, db_table_name=DB_TABLE_NAME)  

    # Step 2: Clean the dataset
    df_cleaned = clean_data(df=df)

    # Step 3: Perform feature engineering
    ###
    ###

    # Step 4: Define the models and their respective hyperparameter grids
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "params": LR_PARAMS
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE),
            "params": XG_PARAMS
        },
        "LightGBM": {  
            "model": LGBMClassifier(verbose=-1, force_row_wise=True, random_state=RANDOM_STATE),
            "params": LGBM_PARAMS
        }
    }    

    # Step 5: Build the models
    trained_models, X_train, X_test, y_train, y_test = build_models(
        df_cleaned=df_cleaned,
        target=TARGET,
        models=models,
        n_jobs=N_JOBS, 
        cv_folds=CV_FOLDS,
        scoring_metric=SCORING_METRIC,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Step 6: Evaluate the models
    evaluate_models(
        trained_models=trained_models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    # Print a summary table of training times and model sizes
    print("\n📊 Summary Table of Training Times and Model Sizes:")
    table_data = [
        [model_name, training_time, model_size_kb]
        for model_name, best_model, training_time, model_size_kb in trained_models
    ]
    headers = ["Model Name", "Training Time (s)", "Model Size (KB)"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Record time taken for pipeline execution
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\n✅ Completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
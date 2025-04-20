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

    # Define constants from the configuration
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

    # Set environment variables 
    os.environ['LOKY_MAX_CPU_COUNT'] = LOKY_MAX_CPU_COUNT  # Set environment variables for parallel processing
    matplotlib.use('Agg')  # Set the backend for matplotlib to avoid display issues

    # Step 1: Load the dataset
    df = load_data(db_path=DB_PATH, db_table_name=DB_TABLE_NAME)  

    # Step 2: Clean the dataset
    df_cleaned = clean_data(df=df)

    # Step 3: Perform feature engineering


    # Step 3: Preprocess the dataset, including cleaning and feature engineering
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
        df_cleaned=df_cleaned,
        target=TARGET,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

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

    # Step 4: Traing and evaluate models
    train_and_evaluate_models(
        models,
        X_train, X_test, y_train, y_test,
        n_jobs=N_JOBS, 
        cv_folds=CV_FOLDS,
        scoring_metric=SCORING_METRIC,
        feature_names=feature_names
    )

    # Record time taken for pipeline execution
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\n✅ Completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
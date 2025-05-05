# src/train_models.py

import joblib
import os
import time
from scipy.stats import uniform, randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def train_models(
    models,
    X_train, y_train,
    use_randomized_cv,
    n_jobs, cv_folds, scoring_metric, random_state
):
    """
    Preprocess the cleaned dataset and train multiple machine learning models.

    Parameters:
        df_cleaned (pd.DataFrame): The cleaned dataset.
        target (str): The name of the target variable.
        models (dict): Dictionary of models and their hyperparameter grids.
        n_jobs (int): Number of parallel jobs for GridSearchCV.
        cv_folds (int): Number of cross-validation folds.
        scoring_metric (str): Scoring metric for GridSearchCV.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - trained_models (list): List of tuples containing model name, best model, training time, and space required.
            - X_train (pd.DataFrame): Transformed training feature matrix.
            - X_test (pd.DataFrame): Transformed testing feature matrix.
            - y_train (pd.Series): Training target variable.
            - y_test (pd.Series): Testing target variable.
    """
    try:
        print("\n🤖 Training the candidate models...")
        start_time = time.time()

        # Ensure the models and output directory exists
        os.makedirs("models", exist_ok=True)
        os.makedirs("output", exist_ok=True)

        trained_models = []

        # Loop through each model and perform training and evaluation
        for model_name, model_info in models.items():
            print(f"\n   ⛏️  Training {model_name} model...")
            start_time = time.time()

            # Perform hyperparameter tuning using GridSearchCV
            print(f"      └── Performing hyperparameter tuning...")
            if use_randomized_cv == True:
                print(f"      └── Utilising randomized search cross-validation...")
                # Convert params to distributions for RandomizedSearchCV
                param_distributions = parse_hyperparameters(model_info["params_rscv"])
                # Use params for RandomizedSearchCV
                search = RandomizedSearchCV(
                    estimator=model_info["model"],
                    param_distributions=param_distributions,
                    n_iter=10,
                    scoring=scoring_metric,
                    cv=cv_folds,
                    n_jobs=n_jobs,
                    random_state=random_state
                )
            else:
                print(f"      └── Utilising grid search cross-validation...")
                # Use params for GridSearchCV
                search = GridSearchCV(
                    estimator=model_info["model"],
                    param_grid=model_info["params_gscv"],
                    scoring=scoring_metric,
                    cv=cv_folds,
                    n_jobs=n_jobs,
                )

            # Fit training data
            search.fit(X_train, y_train)

            # Measure training time
            end_time = time.time()
            training_time = end_time - start_time
            print(f"      └── Model trained successfully in {training_time:.2f} seconds.")

            # Extract the best model and parameters
            best_model = search.best_estimator_
            best_params = {k: float(round(v, 1)) if isinstance(v, float) else v for k, v in search.best_params_.items()}
            print(f"      └── Best parameters: {best_params}")

            # Save the trained model permanently
            model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
            print(f"      └── Saving trained model to {model_path}...")
            joblib.dump(best_model, model_path)
            model_size_kb = round(os.path.getsize(model_path) / 1024, 2)  # Size in KB
            print(f"      └── Model size: {model_size_kb} KB")
            
            # Store the trained model details in a list for later use
            trained_models.append([model_name, best_model, training_time, model_size_kb])


        return trained_models

    except Exception as e:
        print(f"❌ An error occurred during model training: {e}")
        raise RuntimeError("Model training process failed.") from e
    

def parse_hyperparameters(params):
    """
    Parse hyperparameters from YAML configuration into a format suitable for RandomizedSearchCV.
    """
    parsed_params = {}
    for param_name, param_config in params.items():
        if isinstance(param_config, list):  # Categorical parameters
            parsed_params[param_name] = param_config
        elif isinstance(param_config, dict):  # Continuous or discrete parameters
            param_type = param_config["type"]
            if param_type == "uniform":
                parsed_params[param_name] = uniform(loc=param_config["low"], scale=param_config["high"] - param_config["low"])
            elif param_type == "randint":
                parsed_params[param_name] = randint(param_config["low"], param_config["high"])
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")
    return parsed_params
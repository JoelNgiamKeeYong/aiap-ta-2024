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
    Trains multiple machine learning models with hyperparameter tuning and evaluates their performance.

    This function automates the training of multiple machine learning models using either Grid Search Cross-Validation (GridSearchCV) or Randomized Search Cross-Validation (RandomizedSearchCV) for hyperparameter tuning. It ensures reproducibility by utilizing a fixed random seed and parallelizes the computation using the specified number of jobs. The trained models are saved to disk, and their details (e.g., training time, model size) are recorded for further analysis.

    Parameters:
        models (dict): 
            A dictionary where each key is the name of a model, and the value is another dictionary containing:
            - "model": The instantiated machine learning model.
            - "params_gscv": Hyperparameter grid for GridSearchCV.
            - "params_rscv": Hyperparameter distributions for RandomizedSearchCV.
        X_train (pd.DataFrame or np.ndarray): 
            The training feature matrix.
        y_train (pd.Series or np.ndarray): 
            The training target variable.
        use_randomized_cv (bool): 
            Whether to use RandomizedSearchCV (True) or GridSearchCV (False) for hyperparameter tuning.
        n_jobs (int): 
            Number of parallel jobs to run during cross-validation. Use -1 to utilize all available CPU cores.
        cv_folds (int): 
            Number of cross-validation folds for hyperparameter tuning.
        scoring_metric (str): 
            The scoring metric to optimize during hyperparameter tuning (e.g., "accuracy", "f1", "roc_auc").
        random_state (int): 
            Random seed for reproducibility, used in RandomizedSearchCV and other stochastic processes.

    Returns:
        list: 
            A list of tuples, where each tuple contains:
            - model_name (str): The name of the trained model.
            - best_model (object): The best estimator after hyperparameter tuning.
            - training_time (float): Time taken to train the model (in seconds).
            - model_size_kb (float): Size of the saved model file (in KB).

    Raises:
        RuntimeError: 
            If an error occurs during model training, a RuntimeError is raised with details about the failure.
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
    
#################################################################################################################################
#################################################################################################################################
# HELPER FUNCTIONS

#################################################################################################################################
#################################################################################################################################
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
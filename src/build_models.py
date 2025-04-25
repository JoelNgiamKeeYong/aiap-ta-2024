# src/build_models.py

import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def build_models(
    df_cleaned, target, models,
    n_jobs, cv_folds, scoring_metric,
    test_size=0.1, random_state=42
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
        print("\n🔧 Preprocessing the dataset...")
        start_time = time.time()

        # Separate features and target
        print("   └── Separating the features and the target...")
        y = df_cleaned[target]  # Target variable
        X = df_cleaned.drop(columns=[target])  # Feature matrix

        # Split the data into training and testing sets
        print("   └── Splitting the data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            stratify=y,                  # Ensures the same class distribution in train and test sets, important for imbalanced datasets
            test_size=test_size,         # Set in the config.yaml
            random_state=random_state    # To ensure reproducibility as per EDA
        )

        # Define numerical and categorical features
        print("   └── Defining numerical and categorical features...")
        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Define and fit the preprocessing pipeline
        print("   └── Defining preprocessing pipeline...")
        print("       └── Standard scaler for numerical features...")
        print("       └── One hot encoder for categorical features...")
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),  # Scale numerical features
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # Encode categorical features
            ]
        )

        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"\n✅ Data preprocessing completed in {elapsed_time:.2f} seconds!")

        # Ensure the models and output directory exists
        os.makedirs("models", exist_ok=True)
        os.makedirs("output", exist_ok=True)

        trained_models = []

        # Loop through each model and perform training and evaluation
        for model_name, model_info in models.items():
            print(f"\n🛠️  Training {model_name} model...")
            start_time = time.time()

            # Define the pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model_info["model"])
            ]).set_output(transform="pandas") 

            # Prefix all hyperparameters in the parameter grid with 'model__'
            prefixed_params = {f"model__{key}": value for key, value in model_info["params"].items()}

            # Perform hyperparameter tuning using GridSearchCV
            print(f"   └── Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=prefixed_params,
                scoring=scoring_metric,
                cv=cv_folds,
                n_jobs=n_jobs
            )
            grid_search.fit(X_train, y_train)

            # Extract the best model and parameters
            best_model = grid_search.best_estimator_
            print(f"   └── Best parameters: {grid_search.best_params_}")

            # Measure training time
            end_time = time.time()
            training_time = round(end_time - start_time, 2)

            # Save the trained model permanently
            model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
            joblib.dump(best_model, model_path)
            model_size_kb = round(os.path.getsize(model_path) / 1024, 2)  # Size in KB
            print(f"   └── Saving trained model to {model_path}...")

            # Store the trained model details in a list for later use
            trained_models.append([model_name, best_model, training_time, model_size_kb])

            print(f"\n✅ {model_name} model trained successfully! Training time: {training_time:.2f} seconds. Model size: {model_size_kb} KB.")

        return trained_models, X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"❌ An error occurred during model building: {e}")
        raise RuntimeError("Model building process failed.") from e
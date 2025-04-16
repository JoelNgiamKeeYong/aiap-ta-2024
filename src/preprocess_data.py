# src/preprocess_data.py

import joblib
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df_cleaned, test_size=0.1, random_state=42):
    """
    Preprocess the cleaned dataset to prepare it for machine learning modeling.

    Parameters:
        df_cleaned (pd.DataFrame): The cleaned dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Transformed training and testing data (X_train, X_test, y_train, y_test, feature_names).
    """
    try:
        print("\n🔧 Preprocessing the dataset...")
        start_time = time.time()

        # Separate features and target
        X, y = separate_features_and_target(df_cleaned)

        # Define numerical and categorical features
        numerical_features, categorical_features = define_feature_types(X)

        # Define and fit the preprocessing pipeline
        preprocessor = create_preprocessor(numerical_features, categorical_features)
        X_train_unprocessed, X_test_unprocessed, y_train, y_test = split_data(X, y, test_size, random_state)
        preprocessor.fit(X_train_unprocessed)

        # Transform the data
        X_train, X_test, feature_names = transform_data(preprocessor, X_train_unprocessed, X_test_unprocessed)

        # Save the fitted preprocessor
        save_preprocessor(preprocessor)

        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"✅ Completed in {elapsed_time:.2f} seconds. Data preprocessing completed successfully!")
        return X_train, X_test, y_train, y_test, feature_names

    except Exception as e:
        print(f"❌ An error occurred during data preprocessing: {e}")
        raise RuntimeError("Data preprocessing process failed.") from e


def separate_features_and_target(df_cleaned):
    """
    Separate the dataset into features (X) and target (y).

    Args:
        df_cleaned (pd.DataFrame): The cleaned dataset.

    Returns:
        tuple: Features (X) and target (y).
    """
    print("   └── Separating the features and the target...")
    y = df_cleaned['no_show']  # Target variable
    X = df_cleaned.drop(columns=['no_show'])  # Feature matrix
    return X, y


def define_feature_types(X):
    """
    Define numerical and categorical features.

    Args:
        X (pd.DataFrame): Feature matrix.

    Returns:
        tuple: Numerical and categorical feature lists.
    """
    print("   └── Defining numerical and categorical features...")
    numerical_features = ['arrival_day', 'checkout_day', 'price']
    categorical_features = [col for col in X.columns if col not in numerical_features]
    return numerical_features, categorical_features


def create_preprocessor(numerical_features, categorical_features):
    """
    Create a preprocessing pipeline for numerical and categorical features.

    Args:
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.

    Returns:
        ColumnTransformer: Fitted preprocessor.
    """
    print("   └── Defining preprocessing pipeline...")
    print("       └── Standard scaler for numerical features...")
    print("       └── One hot encoder for categorical features...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),  # Scale numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical features
        ]
    )
    return preprocessor


def split_data(X, y, test_size, random_state):
    """
    Split the dataset into training and testing sets.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Training and testing splits (X_train, X_test, y_train, y_test).
    """
    print("\n   └── Splitting the data into training and testing sets...")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def transform_data(preprocessor, X_train_unprocessed, X_test_unprocessed):
    """
    Transform the training and testing data using the preprocessor.

    Args:
        preprocessor (ColumnTransformer): Fitted preprocessor.
        X_train_unprocessed (pd.DataFrame): Unprocessed training feature matrix.
        X_test_unprocessed (pd.DataFrame): Unprocessed testing feature matrix.

    Returns:
        tuple: Transformed training and testing data (X_train, X_test, feature_names).
    """
    print("   └── Transforming the data...")
    X_train = preprocessor.transform(X_train_unprocessed)
    X_test = preprocessor.transform(X_test_unprocessed)

    print("   └── Extracting feature names...")
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    return X_train, X_test, feature_names


def save_preprocessor(preprocessor):
    """
    Save the fitted preprocessor to a file.

    Args:
        preprocessor (ColumnTransformer): Fitted preprocessor.
    """
    os.makedirs("models", exist_ok=True)  # Ensure the `models` directory exists
    preprocessor_path = "models/preprocessor.joblib"
    print(f"\n💾 Saving preprocessor to {preprocessor_path}...")
    joblib.dump(preprocessor, preprocessor_path)
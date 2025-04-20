# src/preprocess_data.py

import joblib
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df_cleaned, target, test_size=0.1, random_state=42):
    """
    Preprocess the cleaned dataset to prepare it for machine learning modeling.

    Parameters:
        df_cleaned (pd.DataFrame): The cleaned dataset.
        target (str): The name of the target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Transformed training and testing data (X_train, X_test, y_train, y_test, feature_names).
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
        X_train_unprocessed, X_test_unprocessed, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

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
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Encode categorical features
            ]
        )
        preprocessor.fit(X_train_unprocessed)

        # Extract feature names
        print("\n   └── Extracting feature names...")
        feature_names = preprocessor.get_feature_names_out()

        # Transform the data
        print("   └── Transforming the data...")
        X_train = preprocessor.transform(X_train_unprocessed)
        X_test = preprocessor.transform(X_test_unprocessed)

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        if hasattr(X_test, "toarray"):
            X_test = X_test.toarray()

        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)

        # Save the fitted preprocessor
        print("   └── Saving the fitted preprocessor...")
        os.makedirs("models", exist_ok=True) 
        preprocessor_path = "models/preprocessor.joblib"
        print(f"\n💾 Saving preprocessor to {preprocessor_path}...")
        joblib.dump(preprocessor, preprocessor_path)

        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"✅ Completed in {elapsed_time:.2f} seconds. Data preprocessing completed successfully!")
        return X_train, X_test, y_train, y_test, feature_names

    except Exception as e:
        print(f"❌ An error occurred during data preprocessing: {e}")
        raise RuntimeError("Data preprocessing process failed.") from e

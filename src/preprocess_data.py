# preprocess_data.py
# This script preprocesses the dataset for training. This will include mainly items from the EDA (eda.ipynb) which include data cleaning and feature engineering

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from helpers import clean_arrival_month_column, clean_checkout_day_column, clean_price_column

def preprocess_data(df):
    """
    Preprocess the dataset for training.
    
    Args:
        data (pandas.DataFrame): Raw dataset from the database.
        method (str): Preprocessing method ('standard' or 'advanced').
    
    Returns:
        tuple: (X, y) where X is the preprocessed features and y is the target.
    """
    print(f"🛠️ Preprocessing data with method...")



    # Separate features and target
    # if "no_show" not in df_cleaned.columns:
    #     raise ValueError("❌ Target column 'no_show' not found in the dataset.")
    # y = df_cleaned["no_show"]
    # X = df_cleaned.drop(columns=["no_show"])
    
    # # Define numerical and categorical columns based on the dataset attributes
    # numerical_cols = ["price", "num_adults", "num_children", "arrival_day", "checkout_day"]
    # categorical_cols = ["branch", "booking_month", "arrival_month", "checkout_month", 
    #                    "country", "first_time", "room", "platform"]
    
    # # Remove booking_id if present (not useful for prediction)
    # if "booking_id" in X.columns:
    #     X = X.drop(columns=["booking_id"])
    
    # # Clean the 'price' column: handle both SGD$ and USD$, and None values
    # if "price" in X.columns:
    #     def convert_price(price):
    #         if pd.isna(price) or price is None:
    #             return 0.0
    #         price = price.strip()
    #         if 'SGD$' in price:
    #             return float(price.replace('SGD$', '').strip())
    #         elif 'USD$' in price:
    #             return float(price.replace('USD$', '').strip())
    #         try:
    #             return float(price)
    #         except ValueError:
    #             return 0.0
        
    #     try:
    #         X["price"] = X["price"].apply(convert_price)
    #     except Exception as e:
    #         print("❌ Unique values in 'price' column:", X["price"].unique()[:10])
    #         raise ValueError(f"❌ Failed to convert 'price' to float: {e}")
    
    # # Function to convert text numbers (e.g., "one") to numeric values
    # def convert_text_to_number(value):
    #     if pd.isna(value) or value is None:
    #         return 0.0
    #     value = str(value).strip().lower()
    #     text_to_num = {
    #         "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    #         "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    #     }
    #     if value in text_to_num:
    #         return float(text_to_num[value])
    #     try:
    #         return float(value)
    #     except ValueError:
    #         return 0.0
    
    # # Ensure other numerical columns are in the correct format
    # for col in numerical_cols:
    #     if col in X.columns and col != "price":  # Already handled price
    #         try:
    #             X[col] = X[col].apply(convert_text_to_number)
    #             X[col] = pd.to_numeric(X[col], errors="raise")
    #         except Exception as e:
    #             print(f"❌ Unique values in '{col}' column:", X[col].unique()[:10])
    #             print(f"❌ Rows with non-numeric values in '{col}':")
    #             print(X[X[col].apply(lambda x: not str(x).replace('.', '').isdigit())][[col]].head())
    #             raise ValueError(f"❌ Column '{col}' contains non-numeric values: {e}")
    
    # # Define preprocessing steps with sparse_threshold=0 to force dense output
    # preprocessor = ColumnTransformer([
    #     ("num", StandardScaler(), numerical_cols),
    #     ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    # ], sparse_threshold=0)  # Force dense output
    
    # # Apply preprocessing and convert to dense array if sparse
    # try:
    #     X_preprocessed = preprocessor.fit_transform(X)
    #     if hasattr(X_preprocessed, "toarray"):  # Check if sparse matrix
    #         X_preprocessed = X_preprocessed.toarray()
    # except Exception as e:
    #     raise ValueError(f"❌ Preprocessing failed: {e}")
    
    # # Get feature names after transformation
    # num_features = numerical_cols
    # cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    # feature_names = num_features + list(cat_features)
    
    # # Debugging: Check shapes
    # print(f"Debug: X_preprocessed shape: {X_preprocessed.shape}")
    # print(f"Debug: Number of feature names: {len(feature_names)}")
    
    # # Convert to DataFrame
    # try:
    #     X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)
    # except Exception as e:
    #     raise ValueError(f"❌ Failed to create DataFrame: {e}")
    
    # print(f"✅ Preprocessing complete! Features: {X_preprocessed.shape[1]} 🎉")
    # return X_preprocessed, y
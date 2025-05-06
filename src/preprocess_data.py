# src/preprocess_data.py

import os
import time
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df_cleaned, target, test_size=0.2, random_state=42):
    """
    Preprocesses the cleaned dataset to prepare it for machine learning modeling.

    This function performs a series of preprocessing steps as per the 🛠️ indicators identified during Exploratory Data Analysis (EDA). The steps include splitting the data into training and testing sets, encoding categorical variables, scaling numerical features, and saving the processed splits for reuse. If preprocessed files already exist in the specified output path, the function skips the preprocessing steps and loads the existing splits.

    Parameters:
        df_cleaned (pd.DataFrame): 
            The cleaned dataset to be preprocessed. This should be a pandas DataFrame containing all relevant features and the target variable.
        target (str): 
            The name of the target variable in the dataset.
        test_size (float, optional): 
            Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): 
            Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: 
            A tuple containing the training and testing splits: (X_train, X_test, y_train, y_test), along with the full preprocessed DataFrame and feature names.

    Raises:
        RuntimeError: 
            If an error occurs during preprocessing, a RuntimeError is raised with details about the failure.

    Example Usage:
        >>> cleaned_data = pd.read_csv("data/cleaned_data.csv")
        >>> X_train, X_test, y_train, y_test, df_preprocessed, feature_names = preprocess_data(cleaned_data, target="target_column")
        >>> print(X_train.head())
    """
    try:
        # Define output paths
        X_train_path = "./data/X_train.csv"
        y_train_path = "./data/y_train.csv"
        X_test_path = "./data/X_test.csv"
        y_test_path = "./data/y_test.csv"
        df_preprocessed_path = "./data/preprocessed_data.csv"

        # Check if the preprocessed data files already exists
        if (os.path.exists(X_train_path) and 
            os.path.exists(y_train_path) and 
            os.path.exists(X_test_path) and 
            os.path.exists(y_test_path) and
            os.path.exists(df_preprocessed_path)):
            
            print("\n✅ Found existing preprocessed data. Skipping preprocessing...")
            
            # Load the preprocessed splits
            X_train = pd.read_csv(X_train_path)
            y_train = pd.read_csv(y_train_path).squeeze()  # Convert to Series
            X_test = pd.read_csv(X_test_path)
            y_test = pd.read_csv(y_test_path).squeeze()    # Convert to Series
            df_preprocessed = pd.read_csv(df_preprocessed_path)
            
            # Extract feature names from X_train (columns of the feature matrix)
            feature_names = X_train.columns.tolist()
            
            # Return the loaded splits and feature names
            return X_train, X_test, y_train, y_test, df_preprocessed, feature_names

        # Carry out preprocessing id preprocessed data files do not exist
        print("\n🔧 Preprocessing the dataset...")
        start_time = time.time()

        # Split the data into training and testing sets
        print("\n   ⚙️  Splitting the data into training and testing sets...")
        df_train, df_test, X_train, X_test, y_train, y_test = split_data(df_cleaned=df_cleaned, target=target, test_size=test_size, random_state=random_state)

        # Carry out additional cleaning
        print("\n   ⚙️  Carrying out additional dataset cleaning...")
        [df_train, df_test] = impute_missing_values_price_in_sgd(list_of_dfs=[df_train, df_test])
        [df_train, df_test] = remove_outliers_price_in_sgd(list_of_dfs=[df_train, df_test])
        [df_train, df_test] = remove_missing_values_room(list_of_dfs=[df_train, df_test])

        # Create new features
        print("\n   ⚙️  Creating new features via feature engineering...")
        [df_train, df_test] = create_months_to_arrival_feature(list_of_dfs=[df_train, df_test])
        [df_train, df_test] = create_stay_duration_days_feature(list_of_dfs=[df_train, df_test])
        [df_train, df_test] = create_stay_category_feature(list_of_dfs=[df_train, df_test])
        [df_train, df_test] = create_has_children_feature(list_of_dfs=[df_train, df_test])
        [df_train, df_test] = create_total_pax_feature(list_of_dfs=[df_train, df_test])

        # Drop irrelevant features
        print("\n   ⚙️  Dropping irrelevant features...")
        [df_train, df_test] = drop_irrelevant_features(list_of_dfs=[df_train, df_test])

        # Checking data integrity after partial preprocessing
        print("\n   ⚙️  Checking data integrity after preprocessing...")
        print(f"      └── Training set shape: {df_train.shape} ...")
        print(f"      └── Test set shape: {df_test.shape} ...")
        total_rows = df_train.shape[0] + df_test.shape[0]
        test_ratio = (df_test.shape[0] / total_rows) * 100
        print(f"      └── Test set constitutes {test_ratio:.2f}% of the total dataset. Original split: {(test_size*100):.2f}%")

        # Separate features (X) and target (y) after partial preprocessing
        print("\n   ⚙️  Separating features (X) and target (y) after partial preprocessing...")
        X_train = df_train.drop(columns=[target])
        y_train = df_train[target] 
        X_test = df_test.drop(columns=[target])
        y_test = df_test[target] 

        # Transform features by apply feature scaling and encoding
        print("   ⚙️  Transforming features by applying scaling and encoding...")
        [X_train, X_test] = transform_features(list_of_dfs=[X_train, X_test])

        # Removing features based on feature selection findings in EDA
        print("\n   ⚙️  Removing features based on feature selection findings in EDA...")
        [X_train, X_test], feature_names = remove_features_based_on_feature_selection(list_of_dfs=[X_train, X_test])

        # Combine features and target into a single DataFrame
        X_combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        y_combined = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
        df_preprocessed = pd.concat([X_combined, y_combined], axis=1)

        # Save the preprocessed data to CSV files
        print("\n💾 Saving preprocessed data to /data folder...")

        # Save consoldiated preprocessed file
        df_preprocessed.to_csv(df_preprocessed_path, index=False)

        # Save each split to its respective file as concatenated df's do not represent the initial train test splits
        X_train.to_csv(X_train_path, index=False)
        X_test.to_csv(X_test_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)

        # Record time taken
        end_time = time.time() 
        elapsed_time = end_time - start_time
        print(f"\n✅ Data preprocessing completed in {elapsed_time:.2f} seconds!")

        # Return data
        return X_train, X_test, y_train, y_test, df_preprocessed, feature_names

    except Exception as e:
        print(f"❌ An error occurred during data preprocessing: {e}")
        raise RuntimeError("Data preprocessing process failed.") from e
    

#################################################################################################################################
#################################################################################################################################
# HELPER FUNCTIONS

#################################################################################################################################
#################################################################################################################################
def split_data(df_cleaned, target, test_size=0.2, random_state=42):
    print("      └── Separating the features and the target...")
    y = df_cleaned[target]  # Target variable
    X = df_cleaned.drop(columns=[target])  # Feature matrix

    print("      └── Splitting the data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,                  # Ensures the same class distribution in train and test sets
        test_size=test_size,         # Proportion of the dataset for the test split
        random_state=random_state    # For reproducibility
    )

    # Combine features and target into single DataFrames for training and testing
    print("      └── Combining features and target into single DataFrames...")
    df_train = pd.concat([X_train, y_train], axis=1)  # Combine X_train and y_train
    df_test = pd.concat([X_test, y_test], axis=1)     # Combine X_test and y_test

    # Print shapes for verification
    print(f"      └── Training set shape: {df_train.shape}")
    print(f"      └── Test set shape: {df_test.shape}")

    return df_train, df_test, X_train, X_test, y_train, y_test

#################################################################################################################################
#################################################################################################################################
def impute_missing_values_price_in_sgd(list_of_dfs):
    print("\n      └── Imputing missing values in `price_in_sgd` feature with SGD median...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Replace 0 values with NaN and calculate the SGD median from the first DataFrame
    list_of_dfs[0]['price_in_sgd'] = list_of_dfs[0]['price_in_sgd'].replace(0, np.nan)
    sgd_median = list_of_dfs[0][list_of_dfs[0]['currency_type'] == 'SGD']['price_in_sgd'].median()

    # Fallback to overall median if SGD median is NaN
    if pd.isna(sgd_median):
        print("      └── No SGD rows found. Falling back to overall median...")
        sgd_median = list_of_dfs[0]['price_in_sgd'].median()

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the imputation logic
    for i, df in enumerate(list_of_dfs, start=1):
        # Replace 0 values with NaN
        df['price_in_sgd'] = df['price_in_sgd'].replace(0, np.nan)

        # Impute missing values with the calculated median
        df['price_in_sgd'] = df['price_in_sgd'].fillna(sgd_median)

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    # Print the median value for reference
    print(f"      └── Missing values imputed with median: {sgd_median:.2f}")

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def remove_outliers_price_in_sgd(list_of_dfs):
    print("\n      └── Removing outliers in `price_in_sgd` feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")
    
    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the outlier removal logic
    for i, df in enumerate(list_of_dfs, start=1):
        # Identify rows where the room is "President Suite" and price_in_sgd is less than 1500
        is_president_suite = df['room'] == 'President Suite'
        is_low_price = df['price_in_sgd'] < 1500
        is_outlier = is_president_suite & is_low_price
        is_not_outlier = ~is_outlier

        # Count the number of outliers
        num_outliers = is_outlier.sum()
        print(f"      └── Identified {num_outliers} outliers in the 'price_in_sgd' feature in DataFrame {i}.")

        # Filter out the outliers
        df = df[is_not_outlier]

        # Count the number of rows remaining after removing outliers
        num_remaining = len(df)
        print(f"      └── Removed outliers. {num_remaining} rows remain in DataFrame {i}.")

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def remove_missing_values_room(list_of_dfs):
    print("\n      └── Removing missing values in room feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the removal logic
    for i, df in enumerate(list_of_dfs, start=1):
        # Count the initial number of rows
        initial_rows = len(df)

        # Filter rows where 'room' is NOT either 'Missing' or np.nan
        df = df[~df['room'].isin(['Missing', np.nan])]

        # Calculate the number of rows dropped
        final_rows = len(df)
        rows_dropped = initial_rows - final_rows

        # Print the number of rows dropped and the number of rows left
        print(f"      └── Removed {rows_dropped} rows with missing values in 'room' for DataFrame {i}.")
        print(f"      └── {final_rows} rows remain in the dataset for DataFrame {i}.")

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
# Define a mapping from month names to numeric values
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

#################################################################################################################################
#################################################################################################################################
def create_months_to_arrival_feature(list_of_dfs):
    print("      └── Creating months_to_arrival feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the feature creation logic
    for i, df in enumerate(list_of_dfs, start=1):

        # Map the month names to numeric values
        df['booking_month_numeric'] = df['booking_month'].map(month_mapping).astype('int64')
        df['arrival_month_numeric'] = df['arrival_month'].map(month_mapping).astype('int64')

        # Calculate the difference in months (accounting for year wrap-around)
        df['months_to_arrival'] = (df['arrival_month_numeric'] - df['booking_month_numeric']) % 12

        # Drop intermediate columns if not needed
        df.drop(columns=['booking_month_numeric', 'arrival_month_numeric'], inplace=True)

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def create_stay_duration_days_feature(list_of_dfs):
    print("      └── Creating stay_duration_days feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the feature creation logic
    for i, df in enumerate(list_of_dfs, start=1):

        # Map month names to numeric values for arrival and checkout months
        df['arrival_month_numeric'] = df['arrival_month'].map(month_mapping)
        df['checkout_month_numeric'] = df['checkout_month'].map(month_mapping)

        # Function to validate dates
        def validate_date(row, month_col, day_col):
            try:
                datetime.datetime(year=2023, month=row[month_col], day=row[day_col])
                return True
            except ValueError:
                return False

        # Validate arrival and checkout dates
        df['is_valid_arrival'] = df.apply(validate_date, axis=1, month_col='arrival_month_numeric', day_col='arrival_day')
        df['is_valid_checkout'] = df.apply(validate_date, axis=1, month_col='checkout_month_numeric', day_col='checkout_day')

        # Filter out rows with invalid dates
        valid_df = df[df['is_valid_arrival'] & df['is_valid_checkout']].copy()

        # Create datetime objects for arrival and checkout dates
        valid_df['arrival_date'] = pd.to_datetime(
            valid_df[['arrival_month_numeric', 'arrival_day']].astype(int).apply(
                lambda x: f"{x.iloc[0]:02d}-{x.iloc[1]:02d}-2023", axis=1
            ), format='%m-%d-%Y'
        )

        # Handle year change for checkout dates
        valid_df['checkout_date'] = valid_df.apply(
            lambda row: pd.to_datetime(f"{row['checkout_month_numeric']:02d}-{row['checkout_day']:02d}-2023", format='%m-%d-%Y')
            if row['checkout_month_numeric'] >= row['arrival_month_numeric']
            else pd.to_datetime(f"{row['checkout_month_numeric']:02d}-{row['checkout_day']:02d}-2024", format='%m-%d-%Y'),
            axis=1
        )

        # Calculate the difference in days
        valid_df['stay_duration_days'] = (valid_df['checkout_date'] - valid_df['arrival_date']).dt.days.astype('int64')

        # Drop intermediate columns if not needed
        intermediate_columns = [
            'arrival_month_numeric', 'checkout_month_numeric',
            'arrival_date', 'checkout_date',
            'is_valid_arrival', 'is_valid_checkout'
        ]
        valid_df.drop(columns=intermediate_columns, inplace=True)

        # Append the modified DataFrame to the results list
        modified_dfs.append(valid_df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def create_stay_category_feature(list_of_dfs):
    print("      └── Creating stay_category feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")

    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the feature creation logic
    for i, df in enumerate(list_of_dfs, start=1):

        # Define bins for short-term, mid-term, and long-term stays
        bins = [0, 3, 14, df['stay_duration_days'].max()]
        labels = ['Short-term', 'Mid-term', 'Long-term']

        # Create a new column for stay categories
        df['stay_category'] = pd.cut(
            df['stay_duration_days'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs

#################################################################################################################################
#################################################################################################################################
def create_has_children_feature(list_of_dfs):
    print("      └── Creating has_children feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")
    
    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the feature creation logic
    for i, df in enumerate(list_of_dfs, start=1):

        # Create the new binary column
        df['has_children'] = df['num_children'].apply(lambda x: 1 if x > 0 else 0)

        # Convert the new column to categorical type for memory efficiency
        df['has_children'] = df['has_children'].astype('category')

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs    

#################################################################################################################################
#################################################################################################################################
def create_total_pax_feature(list_of_dfs):
    print("      └── Creating total_pax feature...")

    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")
    
    # Initialize a list to store the modified DataFrames
    modified_dfs = []

    # Iterate over each DataFrame and apply the feature creation logic
    for i, df in enumerate(list_of_dfs, start=1):

        # Create the new feature by summing `num_adults` and `num_children`
        df['total_pax'] = df['num_adults'].astype('int') + df['num_children'].astype('int')

        # Optionally convert the new column to integer type for consistency
        df['total_pax'] = df['total_pax'].astype('category')

        # Append the modified DataFrame to the results list
        modified_dfs.append(df)

    return modified_dfs    

#################################################################################################################################
#################################################################################################################################
def drop_irrelevant_features(list_of_dfs):
    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")
    
    # Define features to remove
    features_to_drop = ["arrival_day", "checkout_day", "stay_duration_days","num_children"]

    # Initialize a list to store the modified DataFrames
    list_of_modified_dfs = []

    # Loop through each DataFrame and apply the logic
    for i, df in enumerate(list_of_dfs, start=1):        
        # Drop the specified features
        modified_df = df.drop(columns=features_to_drop, errors='ignore')
        
        # Append the modified DataFrame to the results list
        list_of_modified_dfs.append(modified_df)

    # Print confirmation
    print(f"      └── Dropped features: {features_to_drop}")
    
    # Return the list of modified DataFrames
    return list_of_modified_dfs

#################################################################################################################################
#################################################################################################################################
def transform_features(list_of_dfs):
    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")
    
    # Define numerical and categorical features based on the first DataFrame
    print("      └── Defining numerical and categorical features...")
    numerical_features = list_of_dfs[0].select_dtypes(include=['number']).columns.tolist()
    categorical_features = list_of_dfs[0].select_dtypes(include=['object', 'category']).columns.tolist()

    # Define the feature transformer pipeline
    print("      └── Defining the feature transformer pipeline...")
    print("      └── Standard scaler for numerical features...")
    print("      └── One hot encoder for categorical features...")
    print("      └── Transforming the features...")
    transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),  # Scale numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # Encode categorical features
        ]
    )

    # Initialize a list to store the preprocessed DataFrames
    list_of_modified_dfs = []

    # Fit-transform the first DataFrame and transform the rest
    for i, df in enumerate(list_of_dfs):
        if i == 0:
            # Fit-transform the first DataFrame
            transformed_data = transformer.fit_transform(df)
        else:
            # Transform the remaining DataFrames
            transformed_data = transformer.transform(df)

        # Extract feature names and create a DataFrame
        feature_names = [name.split('__', 1)[-1] for name in transformer.get_feature_names_out()]
        preprocessed_df = pd.DataFrame(transformed_data, columns=feature_names)
        list_of_modified_dfs.append(preprocessed_df)

    return list_of_modified_dfs

#################################################################################################################################
#################################################################################################################################
def remove_features_based_on_feature_selection(list_of_dfs):
    if not list_of_dfs:
        raise ValueError("The input list of DataFrames cannot be empty.")
    
    # Define features to remove
    features_to_remove = [
        "checkout_month_January", "checkout_month_February", "checkout_month_March",
        "checkout_month_April", "checkout_month_May", "checkout_month_June",
        "checkout_month_July", "checkout_month_August", "checkout_month_September",
        "checkout_month_October", "checkout_month_November", "checkout_month_December",  
    ]

    # Initialize a list to store the modified DataFrames
    list_of_modified_dfs = []

    # Loop through each DataFrame and apply the logic
    for i, df in enumerate(list_of_dfs, start=1):        
        # Drop the specified features
        modified_df = df.drop(columns=features_to_remove, errors='ignore')
        
        # Append the modified DataFrame to the results list
        list_of_modified_dfs.append(modified_df)
        
    # Update feature_names to reflect the remaining columns
    feature_names = list_of_modified_dfs[0].columns.tolist()  

    # Print confirmation
    print(f"      └── Removed features: {features_to_remove}")
    
    # Return the list of modified DataFrames
    return list_of_modified_dfs, feature_names
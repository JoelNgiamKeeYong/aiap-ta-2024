# src/clean_data.py

import os
import time
import pandas as pd

def clean_data(df):
    """
    This function performs a series of cleaning and preprocessing steps to ensure the dataset is ready for downstream tasks such as feature engineering and model training. The steps include removing irrelevant features, handling missing values, standardizing categorical and numerical columns, and saving the cleaned dataset for reuse.

    Cleaning functions are applied sequentially as per the 🧼 indicators identified during Exploratory Data Analysis (EDA). If a cleaned dataset already exists in the specified output path, the function skips the cleaning process and loads the existing file.

    Parameters:
        df (pd.DataFrame): 
            The raw dataset to be cleaned. This should be a pandas DataFrame containing all relevant features and target variables.

    Returns:
        pd.DataFrame: 
            A cleaned and preprocessed DataFrame ready for further analysis or modeling.

    Raises:
        RuntimeError: 
            If an error occurs during the cleaning process, a RuntimeError is raised with details about the failure.

    Example Usage:
        >>> raw_data = pd.read_csv("data/raw_data.csv")
        >>> cleaned_data = clean_data(raw_data)
        >>> print(cleaned_data.head())
    """
    try:
        # Define output path
        output_path = "./data/cleaned_data.csv"

        # Check if the cleaned data file already exists
        if os.path.exists(output_path):
            print(f"\n✅ Found existing cleaned data. Skipping cleaning process...")
            return pd.read_csv(output_path)

        # Carry out cleaning id cleaned data file does not exist
        print("\n🧼 Cleaning the dataset...")
        start_time = time.time()

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()

        # Remove irrelevant features
        print("   └── Dropping irrelevant columns...")
        df_cleaned = drop_booking_id_function(df_cleaned)

        # Apply cleaning steps to specified columns
        print("\n   └── Cleaning specified columns...")
        df_cleaned = clean_no_show_function(df_cleaned)
        df_cleaned = clean_branch_function(df_cleaned)
        df_cleaned = clean_booking_month_function(df_cleaned)
        df_cleaned = clean_arrival_month_function(df_cleaned)
        df_cleaned = clean_arrival_day_function(df_cleaned)
        df_cleaned = clean_checkout_month_function(df_cleaned)
        df_cleaned = clean_checkout_day_function(df_cleaned)
        df_cleaned = clean_country_function(df_cleaned)
        df_cleaned = clean_first_time_function(df_cleaned)
        df_cleaned = clean_room_function(df_cleaned)
        df_cleaned = clean_price_function(df_cleaned)
        df_cleaned = clean_platform_function(df_cleaned)
        df_cleaned = clean_num_adults_function(df_cleaned)
        df_cleaned = clean_num_children_function(df_cleaned)

        # Save the cleaned data to a CSV file
        print(f"\n💾 Saving cleaned data to {output_path}...")
        df_cleaned.to_csv(output_path, index=False)

        # Record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n✅ Data cleaning completed in {elapsed_time:.2f} seconds!")

        # Return cleaned dataset
        return df_cleaned

    except Exception as e:
        print(f"\n❌ An error occurred during data cleaning: {e}")
        raise RuntimeError("Data cleaning process failed.") from e


#################################################################################################################################
#################################################################################################################################
# HELPER FUNCTIONS

#################################################################################################################################
#################################################################################################################################
def drop_booking_id_function(df_cleaned):
    df_cleaned = df_cleaned.copy()
    
    # Drop column
    print("       └── Dropping 'booking_id' column...")
    df_cleaned = df_cleaned.drop(['booking_id'], axis=1)
    
    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_no_show_function(df_cleaned):
    print("\n   🫧  Cleaning 'no_show' column...")
    df_cleaned = df_cleaned.copy()
    
    # Remove rows with missing values in the specified column
    print("      └── Removing row(s) with missing values in 'no_show' column...")
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=['no_show']).copy()
    removed_rows = initial_rows - len(df_cleaned)
    print(f"      └── Removed {removed_rows} rows with missing values.")
    
    # Convert the column to integer type and then to categorical type
    print (f"      └── Converting 'no_show' column to categorical type...")
    df_cleaned['no_show'] = df_cleaned['no_show'].astype('int64').astype('category')
    
    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_branch_function(df_cleaned):
    print("\n   🫧  Cleaning 'branch' column...")
    df_cleaned = df_cleaned.copy()

    # Convert the column to categorical type
    print(f"      └── Converting 'branch' column to categorical type...")
    df_cleaned['branch'] = df_cleaned['branch'].astype('category')
    
    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_booking_month_function(df_cleaned):
    print("\n   🫧  Cleaning 'booking_month' column...")
    df_cleaned = df_cleaned.copy()

    # Convert the column to categorical type
    print(f"      └── Converting 'booking_month' column to categorical type...")
    df_cleaned['booking_month'] = df_cleaned['booking_month'].astype('category')
    
    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_arrival_month_function(df_cleaned):
    print("\n   🫧  Cleaning 'arrival_month' column...")
    df_cleaned = df_cleaned.copy()

    # Define a mapping from lowercase month names to their proper forms
    month_mapping = {
        'january': 'January',
        'february': 'February',
        'march': 'March',
        'april': 'April',
        'may': 'May',
        'june': 'June',
        'july': 'July',
        'august': 'August',
        'september': 'September',
        'october': 'October',
        'november': 'November',
        'december': 'December'
    }

    # Convert the column values to lowercase for case-insensitive comparison
    print(f"      └── Converting 'arrival_month' values to lowercase...")
    df_cleaned['arrival_month'] = df_cleaned['arrival_month'].str.lower()

    # Map the lowercase values to their proper forms using the dictionary
    print(f"      └── Standardizing 'arrival_month' values...")
    df_cleaned['arrival_month'] = df_cleaned['arrival_month'].map(month_mapping).fillna(df_cleaned['arrival_month'])

    # Convert to categorical type for better memory efficiency and performance
    print(f"      └── Converting 'arrival_month' column to categorical type...")
    df_cleaned['arrival_month'] = df_cleaned['arrival_month'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_arrival_day_function(df_cleaned):
    print("\n   🫧  Cleaning 'arrival_day' column...")
    df_cleaned = df_cleaned.copy()

    # Convert the column to integer type
    print(f"      └── Converting 'arrival_day' column to integer type...")
    df_cleaned['arrival_day'] = df_cleaned['arrival_day'].astype('int64')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_checkout_month_function(df_cleaned):
    print("\n   🫧  Cleaning 'checkout_month' column...")
    df_cleaned = df_cleaned.copy()

    # Convert the column to categorical type
    print("      └── Converting 'checkout_month' column to categorical type...")
    df_cleaned['checkout_month'] = df_cleaned['checkout_month'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_checkout_day_function(df_cleaned):
    print("\n   🫧  Cleaning 'checkout_day' column...")
    df_cleaned = df_cleaned.copy()

    # Convert negative values to positive using abs()
    print("      └── Converting negative values in 'checkout_day' to positive...")
    df_cleaned['checkout_day'] = df_cleaned['checkout_day'].apply(abs)

    # Convert the column to integer type
    print("      └── Converting 'checkout_day' column to integer type...")
    df_cleaned['checkout_day'] = df_cleaned['checkout_day'].astype('int64')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_country_function(df_cleaned):
    print("\n   🫧  Cleaning 'country' column...")
    df_cleaned = df_cleaned.copy()

    # Convert the column to categorical type
    print("      └── Converting 'country' column to categorical type...")
    df_cleaned['country'] = df_cleaned['country'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_first_time_function(df_cleaned):
    print("\n   🫧  Cleaning 'first_time' column...")
    df_cleaned = df_cleaned.copy()

    # Map "Yes" to 1 and "No" to 0
    print("      └── Mapping 'Yes' to 1 and 'No' to 0 in 'first_time' column...")
    df_cleaned['first_time'] = df_cleaned['first_time'].map({'Yes': 1, 'No': 0})

    # Convert the column to integer type
    print("      └── Converting 'first_time' column to integer type...")
    df_cleaned['first_time'] = df_cleaned['first_time'].astype('int64')

    # Convert the column to categorical type for better memory efficiency
    print("      └── Converting 'first_time' column to categorical type...")
    df_cleaned['first_time'] = df_cleaned['first_time'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_room_function(df_cleaned):
    print("\n   🫧  Cleaning 'room' column...")
    df_cleaned = df_cleaned.copy()

    # Impute missing values with "Missing"
    print("      └── Imputing missing values in 'room' column with 'Missing'...")
    initial_missing = df_cleaned['room'].isnull().sum()
    df_cleaned['room'] = df_cleaned['room'].fillna("Missing")
    print(f"      └── Imputed {initial_missing} missing values with 'Missing'.")

    # Convert the column to categorical type
    print("      └── Converting 'room' column to categorical type...")
    df_cleaned['room'] = df_cleaned['room'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_price_function(df_cleaned):
    print("\n   🫧  Cleaning 'price' column...")
    df_cleaned = df_cleaned.copy()
    exchange_rate = 1.35  # Exchange rate for USD to SGD

    # Impute missing values with 0
    print("      └── Imputing missing values in 'price' column with 0...")
    initial_missing = df_cleaned['price'].isnull().sum()
    df_cleaned['price'] = df_cleaned['price'].fillna(0)
    remaining_missing = df_cleaned['price'].isnull().sum()
    print(f"      └── Imputed {initial_missing} missing values with 0. Remaining missing values: {remaining_missing}")

    # Add a new column to track the original currency type
    print("      └── Extracting currency type from 'price' column...")
    print("      └── Creating new currency_type column...")
    df_cleaned['currency_type'] = df_cleaned['price'].str.extract(r'^(\w+)\$', expand=False).fillna('SGD').astype('category')

    # Remove currency prefixes (e.g., "SGD$" or "USD$")
    print("      └── Removing currency prefixes (e.g., 'SGD$', 'USD$')...")
    df_cleaned['price'] = df_cleaned['price'].str.replace(r'^\w+\$', '', regex=True)

    # Convert the cleaned price column to numeric
    print("      └── Converting 'price' column to numeric...")
    df_cleaned['price'] = pd.to_numeric(df_cleaned['price'], errors='coerce')

    # Convert USD to SGD
    print(f"      └── Converting USD prices to SGD using an exchange rate of {exchange_rate}...")
    df_cleaned['price_in_sgd'] = df_cleaned.apply(
        lambda row: row['price'] * exchange_rate if row['currency_type'] == 'USD' else row['price'],
        axis=1
    )

    # Handle missing values in the new column (e.g., replace with 0)
    print("      └── Replacing missing values in price_in_sgd column with 0...")
    df_cleaned['price_in_sgd'] = df_cleaned['price_in_sgd'].fillna(0)

    # Drop the original 'price' column
    print(f"      └── Dropping the original 'price' column...")
    df_cleaned = df_cleaned.drop(columns=['price'])

    # Reorder columns to place 'currency_type' before 'price_in_sgd'
    print(f"      └── Reordering columns to place 'currency_type' before `price_in_sgd`...")
    cols = list(df_cleaned.columns)
    price_in_sgd_index = cols.index('price_in_sgd')
    cols.insert(price_in_sgd_index, 'currency_type')  # Insert 'currency_type' before 'price_in_sgd'

    # Remove duplicate columns (if any) while preserving order
    cols = list(dict.fromkeys(cols))
    df_cleaned = df_cleaned[cols]

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_platform_function(df_cleaned):
    print("\n   🫧  Cleaning 'platform' column...")
    df_cleaned = df_cleaned.copy()

    # Convert the column to categorical type
    print("      └── Converting 'platform' column to categorical type...")
    df_cleaned['platform'] = df_cleaned['platform'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_num_adults_function(df_cleaned):
    print("\n   🫧  Cleaning 'num_adults' column...")
    df_cleaned = df_cleaned.copy()

    # Define the mapping for textual representations to integers
    mapping = {"one": 1, "two": 2}

    # Map textual representations to integers
    print("      └── Mapping textual values in 'num_adults' to integers...")
    df_cleaned['num_adults'] = df_cleaned['num_adults'].replace(mapping)

    # Convert the column to integer type
    print("      └── Converting 'num_adults' column to integer type...")
    df_cleaned['num_adults'] = df_cleaned['num_adults'].astype('int64')

    # Convert the column to categorical type for better memory efficiency
    print("      └── Converting 'num_adults' column to categorical type...")
    df_cleaned['num_adults'] = df_cleaned['num_adults'].astype('category')

    return df_cleaned

#################################################################################################################################
#################################################################################################################################
def clean_num_children_function(df_cleaned):
    print("\n   🫧  Cleaning 'num_children' column...")
    df_cleaned = df_cleaned.copy()

    # Convert the column to integer type
    print("      └── Converting 'num_children' column to integer type...")
    df_cleaned['num_children'] = df_cleaned['num_children'].astype('int64')

    # Convert the column to categorical type for better memory efficiency
    print("      └── Converting 'num_children' column to categorical type...")
    df_cleaned['num_children'] = df_cleaned['num_children'].astype('category')

    return df_cleaned
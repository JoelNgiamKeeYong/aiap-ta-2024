# src/clean_data.py

from utils import compare_dataframes 
from helpers import clean_checkout_month_column, clean_arrival_day_column, clean_arrival_month_column, clean_booking_month_column, clean_branch_column, clean_first_time_column, clean_room_column, clean_country_column, clean_checkout_day_column, clean_price_column, clean_no_show_column, clean_platform_column, clean_num_adults_column, clean_num_children_column

def clean_data(df):
    """
    Clean and preprocess the raw dataset to prepare it for machine learning modeling.

    This function performs the following steps:
    1. Removes irrelevant features (e.g., `booking_id`).
    2. Cleans individual columns based on their specific requirements:
       - Handles missing values in critical columns (`no_show`, `room`, `price`).
       - Converts categorical columns to the `category` dtype for memory efficiency.
       - Standardizes date-related columns (`arrival_month`, `arrival_day`, `checkout_month`, `checkout_day`).
       - Maps ordinal or string-based categories to numerical representations where applicable.
    3. Ensures all columns are in the correct data types for downstream processing.
    4. Saves the cleaned dataset to a CSV file for further use.

    Parameters:
        df (pd.DataFrame): The raw dataset loaded from the database.

    Returns:
        pd.DataFrame: A cleaned and preprocessed DataFrame ready for feature engineering and modeling.
    """
    print("\n🧼 Cleaning the dataset...")

    try:
        # Step 1: Remove irrelevant features
        irrelevant_columns = ['booking_id']
        print(f"   └── Removing irrelevant columns: {', '.join(irrelevant_columns)}...")
        df_cleaned = df.drop(columns=irrelevant_columns, errors='ignore')

        # Step 2: Define the cleaning sequence
        cleaning_steps = [
            ("no_show", clean_no_show_column),
            ("branch", clean_branch_column),
            ("booking_month", clean_booking_month_column),
            ("arrival_month", clean_arrival_month_column),
            ("arrival_day", clean_arrival_day_column),
            ("checkout_month", clean_checkout_month_column),
            ("checkout_day", clean_checkout_day_column),
            ("country", clean_country_column),
            ("first_time", clean_first_time_column),
            ("room", clean_room_column),
            ("price", clean_price_column),
            ("platform", clean_platform_column),
            ("num_adults", clean_num_adults_column),
            ("num_children", clean_num_children_column),
        ]

        # Step 3: Apply each cleaning step sequentially
        for column_name, cleaning_function in cleaning_steps:
            df_cleaned = cleaning_function(df_cleaned)

        # Step 4: Compare the original and cleaned DataFrames
        compare_dataframes(df, df_cleaned)

        # Step 5: Save the cleaned data to a CSV file
        output_path = "./data/cleaned_data.csv"
        print(f"\n💾 Saving cleaned data to {output_path}...")
        df_cleaned.to_csv(output_path, index=False)

        print("✅ Data cleaning completed successfully!")
        return df_cleaned

    except Exception as e:
        print(f"❌ An error occurred during data cleaning: {e}")
        raise RuntimeError("Data cleaning process failed.") from e
# src/helpers/clean_price_column.py

import pandas as pd

import pandas as pd

def clean_price_column(df):
    """
    Clean the 'price' column by:
    1. Dropping rows with missing values.
    2. Removing currency prefixes (e.g., "SGD$" or "USD$").
    3. Converting USD to SGD using the provided exchange rate.
    4. Adding a new column indicating the original currency type.
    5. Handling missing values after conversion (e.g., replace with 0).
    6. Reordering columns to place 'currency_type' before 'price'.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the cleaned price column in SGD and a 'currency_type' column.
    """
    # Define the column name to be cleaned
    column_name = 'price'
    exchange_rate = 1.35  # Exchange rate for USD to SGD

    # Print the cleaning process
    print(f"\n   🫧  Cleaning {column_name} column...")

    try:
        # Ensure the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()

        # Step 1: Drop rows with missing values in the specified column
        initial_rows = len(df_cleaned)
        print(f"      └── Removing rows with missing values in {column_name} column...")
        df_cleaned = df_cleaned.dropna(subset=[column_name])
        removed_rows = initial_rows - len(df_cleaned)
        print(f"      └── Removed {removed_rows} rows with missing values.")

        # Step 2: Add a new column to track the original currency type
        print(f"      └── Extracting currency type from {column_name} column...")
        print(f"      └── Creating new currency_type column...")
        df_cleaned['currency_type'] = df_cleaned[column_name].str.extract(r'^(\w+)\$', expand=False).fillna('SGD').astype('category')

        # Step 3: Remove currency prefixes (e.g., "SGD$" or "USD$")
        print(f"      └── Removing currency prefixes (e.g., 'SGD$', 'USD$')...")
        df_cleaned[column_name] = df_cleaned[column_name].str.replace(r'^\w+\$', '', regex=True)

        # Step 4: Convert the cleaned price column to numeric
        print(f"      └── Converting {column_name} column to numeric...")
        df_cleaned[column_name] = pd.to_numeric(df_cleaned[column_name], errors='coerce')

        # Step 5: Convert USD to SGD
        print(f"      └── Converting USD prices to SGD using an exchange rate of {exchange_rate}...")
        df_cleaned[column_name] = df_cleaned.apply(
            lambda row: row[column_name] * exchange_rate if row['currency_type'] == 'USD' else row[column_name],
            axis=1
        )

        # Step 6: Handle missing values (e.g., replace with 0)
        print(f"      └── Replacing missing values in {column_name} column with 0...")
        df_cleaned[column_name] = df_cleaned[column_name].fillna(0)

        # Step 7: Reorder columns to place 'currency_type' before 'price'
        print(f"      └── Reordering columns to place 'currency_type' before {column_name}...")
        cols = list(df_cleaned.columns)
        column_index = cols.index(column_name)
        cols.insert(column_index, 'currency_type')  # Insert 'currency_type' before 'price'

        # Remove duplicate columns (if any) while preserving order
        cols = list(dict.fromkeys(cols))
        df_cleaned = df_cleaned[cols]

        return df_cleaned

    except Exception as e:
        print(f"❌ An error occurred while cleaning the '{column_name}' column: {e}")
        raise RuntimeError(f"An error occurred while cleaning the '{column_name}' column: {e}") from e
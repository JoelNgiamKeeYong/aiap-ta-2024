# src/helpers/clean_checkout_month_column.py

def clean_checkout_month_column(df):
    """
    Clean the 'checkout_month' column by converting it to a categorical data type.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the 'checkout_month' column converted to a categorical type.
    """
    # Define the column name to be cleaned
    column_name = 'checkout_month'

    # Print the cleaning process
    print(f"\n   🫧  Cleaning {column_name} column...")

    try:
        # Ensure the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()

        # Convert the column to categorical type
        print(f"      └── Converting {column_name} column to categorical type...")
        df_cleaned[column_name] = df_cleaned[column_name].astype('category')

        return df_cleaned

    except Exception as e:
        print(f"❌ An error occurred while cleaning the '{column_name}' column: {e}")
        raise RuntimeError(f"An error occurred while cleaning the '{column_name}' column: {e}") from e
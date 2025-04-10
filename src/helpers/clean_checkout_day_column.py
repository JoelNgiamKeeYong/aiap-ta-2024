# src/helpers/clean_checkout_day_column.py


def clean_checkout_day_column(df):
    """
    Clean the 'checkout_day' column by converting negative values to positive and ensuring the column is of integer type.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the 'checkout_day' column cleaned.
    """
    # Define the column name to be cleaned
    column_name = 'checkout_day'

    # Print the cleaning process
    print(f"\n   🫧  Cleaning {column_name} column...")

    try:
        # Ensure the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()

        # Convert negative values to positive using abs()
        print(f"      └── Converting negative values in {column_name} to positive...")
        df_cleaned[column_name] = df_cleaned[column_name].apply(abs)

        # Convert the column to integer type
        print(f"      └── Converting {column_name} column to integer type...")
        df_cleaned[column_name] = df_cleaned[column_name].astype('int64')

        return df_cleaned

    except Exception as e:
        print(f"❌ An error occurred while cleaning the '{column_name}' column: {e}")
        raise RuntimeError(f"An error occurred while cleaning the '{column_name}' column: {e}") from e
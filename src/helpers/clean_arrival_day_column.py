# src/helpers/clean_arrival_day_column.py

def clean_arrival_day_column(df):
    """
    Clean the 'arrival_day' column by converting it to an integer type.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the 'arrival_day' column converted to an integer type.
    """
    # Define the column name to be cleaned
    column_name = 'arrival_day'

    # Print the cleaning process
    print(f"\n   🫧  Cleaning {column_name} column...")

    try:
        # Ensure the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()

        # Convert the column to integer type
        print(f"      └── Converting {column_name} column to integer type...")
        df_cleaned[column_name] = df_cleaned[column_name].astype('int64')

        return df_cleaned

    except Exception as e:
        print(f"❌ An error occurred while cleaning the '{column_name}' column: {e}")
        raise RuntimeError(f"An error occurred while cleaning the '{column_name}' column: {e}") from e
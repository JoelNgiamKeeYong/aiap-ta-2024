# src/helpers/clean_room_column.py


def clean_room_column(df):
    """
    Clean the 'room' column by removing rows with missing values and converting it to a categorical data type.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the 'room' column cleaned.
    """
    # Define the column name to be cleaned
    column_name = 'room'

    # Print the cleaning process
    print(f"\n   🫧  Cleaning {column_name} column...")

    try:
        # Ensure the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()

        # Step 1: Remove rows with missing values in the specified column
        initial_rows = len(df_cleaned)
        print(f"      └── Removing rows with missing values in {column_name} column...")
        df_cleaned = df_cleaned.dropna(subset=[column_name])
        removed_rows = initial_rows - len(df_cleaned)
        print(f"      └── Removed {removed_rows} rows with missing values.")

        # Step 2: Convert the column to categorical type
        print(f"      └── Converting {column_name} column to categorical type...")
        df_cleaned[column_name] = df_cleaned[column_name].astype('category')

        return df_cleaned

    except Exception as e:
        print(f"❌ An error occurred while cleaning the '{column_name}' column: {e}")
        raise RuntimeError(f"An error occurred while cleaning the '{column_name}' column: {e}") from e
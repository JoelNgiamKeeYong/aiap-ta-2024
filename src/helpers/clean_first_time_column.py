# src/helpers/clean_first_time_column.py


def clean_first_time_column(df):
    """
    Clean the 'first_time' column by mapping "Yes" to 1 and "No" to 0, and converting it to a categorical type.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the 'first_time' column cleaned.
    """
    # Define the column name to be cleaned
    column_name = 'first_time'

    # Print the cleaning process
    print(f"\n   🫧  Cleaning {column_name} column...")

    try:
        # Ensure the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()

        # Map "Yes" to 1 and "No" to 0
        print(f"      └── Mapping 'Yes' to 1 and 'No' to 0 in {column_name} column...")
        df_cleaned[column_name] = df_cleaned[column_name].map({'Yes': 1, 'No': 0})

        # Convert the column to integer type
        print(f"      └── Converting {column_name} column to integer type...")
        df_cleaned[column_name] = df_cleaned[column_name].astype('int64')

        # Convert the column to categorical type for better memory efficiency
        print(f"      └── Converting {column_name} column to categorical type...")
        df_cleaned[column_name] = df_cleaned[column_name].astype('category')

        return df_cleaned

    except Exception as e:
        print(f"❌ An error occurred while cleaning the '{column_name}' column: {e}")
        raise RuntimeError(f"An error occurred while cleaning the '{column_name}' column: {e}") from e
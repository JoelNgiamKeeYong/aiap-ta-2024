def clean_arrival_month_column(df):
    """
    Standardize month names in the 'arrival_month' column by converting all variations to their proper form.
    Handles case insensitivity and matches all permutations.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with standardized month names in the 'arrival_month' column.
    """
    # Define the column name to be cleaned
    column_name = 'arrival_month'

    # Print the cleaning process
    print(f"\n   🫧  Cleaning {column_name} column...")

    try:
        # Ensure the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        df_cleaned = df.copy()

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
        print(f"      └── Converting {column_name} values to lowercase...")
        df_cleaned[column_name] = df_cleaned[column_name].str.lower()

        # Map the lowercase values to their proper forms using the dictionary
        print(f"      └── Standardizing {column_name} values...")
        df_cleaned[column_name] = df_cleaned[column_name].map(month_mapping).fillna(df_cleaned[column_name])

        # Convert to categorical type for better memory efficiency and performance
        print(f"      └── Converting {column_name} column to categorical type...")
        df_cleaned[column_name] = df_cleaned[column_name].astype('category')

        return df_cleaned

    except Exception as e:
        print(f"❌ An error occurred while cleaning the '{column_name}' column: {e}")
        raise RuntimeError(f"An error occurred while cleaning the '{column_name}' column: {e}") from e
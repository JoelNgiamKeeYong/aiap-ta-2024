# src/helpers/clean_no_show_column.py

def clean_no_show_column(df):
    """
    Clean the 'no_show' column by removing rows with missing values and converting the column to categorical type.
    
    Steps:
    1. Remove rows with missing values in the specified column.
    2. Convert the column to integer type and then to categorical type.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: A DataFrame with the cleaned 'no_show' column.
    """
    # Define the column name to be cleaned
    column_name = 'no_show'  

    # Print the cleaning process
    print(f"\n   🫧  Cleaning {column_name} column...")

    try:
        # Ensure the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        
        # Step 1: Remove rows with missing values in the specified column
        initial_rows = len(df)
        print(f"      └── Removing row(s) with missing values in {column_name} column...")
        df_cleaned = df.dropna(subset=[column_name]).copy()
        removed_rows = initial_rows - len(df_cleaned)
        print(f"      └── Removed {removed_rows} rows with missing values.")
        
        # Step 2: Convert the column to integer type and then to categorical type
        print (f"      └── Converting {column_name} column to categorical type...")
        df_cleaned[column_name] = df_cleaned[column_name].astype('int64').astype('category')
        
        return df_cleaned

    except Exception as e:
        raise RuntimeError(f"An error occurred while cleaning the '{column_name}' column: {e}") from e
# remove_missing_rows.py

def remove_missing_rows(df, column_name):
    """
    Remove rows with missing values in the specified column and log details about the operation.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column to check for missing values.

    Returns:
        None
    """
    # Log the initial shape
    print(f"📊 Shape Before Removal: {df.shape}")

    # Remove rows with missing values in the specified column
    df_new = df.dropna(subset=[column_name])

    # Log the final shape
    print(f"📊 Shape After Removal: {df_new.shape}")

    # Calculate the number of rows removed
    rows_removed = df.shape[0] - df_new.shape[0]
    print(f"🗑️ Rows Removed: {rows_removed}")

    # Confirm no missing values remain
    missing_after = df_new[column_name].isnull().sum()
    if missing_after == 0:
        print(f"🎉 No Missing Values Remain in Column '{column_name}'.")
    else:
        print(f"⚠️ Warning: Missing Values Still Exist in Column '{column_name}'.")

    return df_new
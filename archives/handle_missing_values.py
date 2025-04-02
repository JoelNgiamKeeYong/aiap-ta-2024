def handle_missing_values(df, threshold=0.05):
    """
    Handle missing values in the DataFrame:
    - Remove rows for columns below the threshold.
    - Inform the user about columns above the threshold without taking action.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        threshold (float): The percentage threshold (default is 5%) above which missing values 
                           will trigger a warning instead of removal.
    
    Returns:
        pd.DataFrame: A cleaned DataFrame with missing values handled.

    Time Complexity:
        - O(n * m), where n is the number of rows and m is the number of columns.
        - Iterating through each column to calculate missing values takes O(m).
        - For each column, calculating missing values involves scanning all rows, taking O(n).
        - Dropping rows with missing values in below-threshold columns also takes O(n).

    Space Complexity:
        - O(m), where m is the number of columns.
        - The `missing_info` list stores information about missing values for each column, requiring O(m) space.
        - The `below_threshold` and `above_threshold` lists store subsets of column names, requiring O(m) space.

    Assumptions and Edge Cases:
        - Assumes the dataset contains numeric or categorical data types.
        - Handles edge cases such as columns with no missing values or datasets with no rows.
    """
    print("⚙️ Handling missing values in the dataset...")
    initial_shape = df.shape

    # Calculate total rows
    total_rows = len(df)

    # Identify all columns with missing values
    missing_info = []
    for col in df.columns:
        missing_values = df[col].isnull().sum()
        if missing_values > 0:
            missing_percentage = (missing_values / total_rows) * 100
            missing_info.append((col, missing_values, missing_percentage))

    # Separate columns into below-threshold and above-threshold groups
    below_threshold = [col for col, _, pct in missing_info if pct <= threshold]
    above_threshold = [col for col, _, pct in missing_info if pct > threshold]

    # Print summary for columns below the threshold
    if below_threshold:
        print(f"\n✅ Removing rows with missing values in the following columns (below {threshold * 100}% threshold):")
        for col, missing_values, missing_pct in missing_info:
            if col in below_threshold:
                print(f"  - {col}: {missing_values} missing value(s) ({missing_pct:.2f}% of total rows)")
        # Remove rows with missing values in below-threshold columns
        df = df.dropna(subset=below_threshold)

    # Print summary for columns above the threshold
    if above_threshold:
        print(f"\n❌ No action taken for the following columns (above {threshold * 100}% threshold). Please manually inspect:")
        for col, missing_values, missing_pct in missing_info:
            if col in above_threshold:
                print(f"  - {col}: {missing_values} missing value(s) ({missing_pct:.2f}% of total rows)")

    final_shape = df.shape
    print("\n✅ Missing values handling completed.")
    print(f"Initial shape: {initial_shape}")
    print(f"Final shape: {final_shape}")

    return df
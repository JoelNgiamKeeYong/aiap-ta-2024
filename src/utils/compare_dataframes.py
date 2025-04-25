# src/utils/compare_dataframes.py

def compare_dataframes(df_original, df_cleaned):
    """
    Compare two DataFrames to determine the number of rows/columns dropped/added and changes in data size.

    Parameters:
        df_original (pd.DataFrame): The original DataFrame.
        df_cleaned (pd.DataFrame): The cleaned DataFrame.

    Returns:
        None
    """
    # Step 1: Compare Row Counts
    rows_before = df_original.shape[0]
    rows_after = df_cleaned.shape[0]
    rows_dropped = rows_before - rows_after

    # Step 2: Compare Column Counts
    cols_before = df_original.shape[1]
    cols_after = df_cleaned.shape[1]
    dropped_columns = set(df_original.columns) - set(df_cleaned.columns)
    added_columns = set(df_cleaned.columns) - set(df_original.columns)

    # Step 3: Compare Data Size
    size_before = df_original.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert to MB
    size_after = df_cleaned.memory_usage(deep=True).sum() / (1024 * 1024)    # Convert to MB
    size_reduction = size_before - size_after

    # Step 4: Display Results
    print("\n📚 Comparing between original and new DataFrames:\n")
    print(f"    └── 📘 Rows Before: {rows_before:,}")
    print(f"    └── 📘 Rows After: {rows_after:,}")
    print(f"    └── 📘 Dropped Rows: {rows_dropped:,} ({rows_dropped / rows_before:.2%} of total rows)\n")

    print(f"    └── 📙 Columns Before: {cols_before}")
    print(f"    └── 📙 Columns After: {cols_after}")
    if dropped_columns:
        print(f"    └── 📙 Dropped Columns: {', '.join(dropped_columns)}")
    else:
        print("    └── 📙 No columns were dropped.")
    if added_columns:
        print(f"    └── 📙 Added Columns: {', '.join(added_columns)}")
    else:
        print("    └── 📙 No columns were added.")
    print()

    print(f"    └── 📗 Data Size Before: {size_before:.2f} MB")
    print(f"    └── 📗 Data Size After: {size_after:.2f} MB")
    print(f"    └── 📗 Size Reduction: {size_reduction:.2f} MB ({size_reduction / size_before:.2%} reduction)")
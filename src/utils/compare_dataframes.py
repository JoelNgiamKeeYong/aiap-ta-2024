# src/utils/compare_dataframes.py

def compare_dataframes(df_original, df_new, original_name_string="original", new_name_string="new", show_verbose=True):
    """
    Compare two DataFrames to determine the number of rows/columns dropped/added and changes in data size.

    Parameters:
        df_original (pd.DataFrame): The original DataFrame.
        df_new (pd.DataFrame): The cleaned DataFrame.

    Returns:
        None
    """
    # Step 1: Compare Row Counts
    rows_before = df_original.shape[0]
    rows_after = df_new.shape[0]
    rows_dropped = rows_before - rows_after

    # Step 2: Compare Column Counts
    cols_before = df_original.shape[1]
    cols_after = df_new.shape[1]
    dropped_columns = set(df_original.columns) - set(df_new.columns)
    added_columns = set(df_new.columns) - set(df_original.columns)

    # Step 3: Compare Data Size
    size_before = df_original.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert to MB
    size_after = df_new.memory_usage(deep=True).sum() / (1024 * 1024)  # Convert to MB
    size_change = size_after - size_before

    # Step 4: Determine Size Change Type
    if size_change > 0:
        size_label = "Size Increase"
        size_percentage = (size_change / size_before) * 100
    else:
        size_label = "Size Reduction"
        size_percentage = (abs(size_change) / size_before) * 100

    # Step 5: Display Results
    print(f"\n📚 Comparing between {original_name_string} and {new_name_string} DataFrames:\n")
    print(f"    └── 📘 Rows Before: {rows_before:,}")
    print(f"    └── 📘 Rows After: {rows_after:,}")
    print(f"    └── 📘 Dropped Rows: {rows_dropped:,} ({abs(rows_dropped) / rows_before:.2%} of total rows)")

    print(f"\n    └── 📙 Columns Before: {cols_before}")
    print(f"    └── 📙 Columns After: {cols_after}")
    if show_verbose:
        if dropped_columns:
            print(f"    └── 📙 Dropped Columns: {', '.join(dropped_columns)}")
        else:
            print("    └── 📙 No columns were dropped.")
        if added_columns:
            print(f"    └── 📙 Added Columns: {', '.join(added_columns)}")
        else:
            print("    └── 📙 No columns were added.")

    print(f"\n    └── 📗 Data Size Before: {size_before:.2f} MB")
    print(f"    └── 📗 Data Size After: {size_after:.2f} MB")
    print(f"    └── 📗 {size_label}: {size_change:.2f} MB ({size_percentage:.2f}% {size_label})")
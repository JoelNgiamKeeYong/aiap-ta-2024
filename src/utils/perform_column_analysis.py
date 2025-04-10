# src/utils/perform_column_analysis.py

from IPython.display import display

def perform_column_analysis(df, column_name, show_distribution=True):
    """
    Perform analysis on a specific column in the DataFrame:
    - Check data type.
    - Count the number of unique values.
    - Display distribution (absolute counts and proportions).
    - Identify and display rows with missing values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to analyze.
        show_distribution (bool): Whether to display the distribution of values. Default is True.

    Returns:
        None
    """
    # Check the data type of the column
    print("🔢 Data Type:", df[column_name].dtype)

    # Count Unique Values
    unique_values_count = df[column_name].nunique()
    print(f"💎 Number of Unique Values: {unique_values_count}")

    # Show an example of the column's values
    print("📋 Sample Values:")
    display(df[[column_name]].head())  # Display the first 5 rows of the column

    # Check the distribution of the column
    distribution = (
        df[column_name]
        .value_counts()
        .rename_axis('Value')
        .reset_index(name='Count')
    )
    distribution['Proportion'] = distribution['Count'] / distribution['Count'].sum()
    if show_distribution:
        print("📊 Value Distribution:")
        display(distribution.style.format({
            'Count': '{:,}',  # Add thousand separators
            'Proportion': '{:.2%}'  # Format as percentage with 2 decimal places
        }))

    # Check for missing values
    missing_count = df[column_name].isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100  # Calculate percentage of missing rows
    if missing_count == 0:
        print("✅ No missing values found.")
    else:
        print(f"⚠️ Rows with missing values (Total: {missing_count}, {missing_percentage:.2f}% of total rows):")
        missing_rows = df[df[column_name].isnull()]
        display(missing_rows)

    return None
# standardize_month_names.py

def standardize_month_names(df, column_name):
    """
    Standardize month names in the specified column by converting all variations to their proper form.
    Handles case insensitivity and matches all permutations.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing month names.

    Returns:
        pd.DataFrame: A DataFrame with standardized month names.
    """
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
    df[column_name] = df[column_name].str.lower()

    # Map the lowercase values to their proper forms using the dictionary
    df[column_name] = df[column_name].map(month_mapping).fillna(df[column_name])

    return df
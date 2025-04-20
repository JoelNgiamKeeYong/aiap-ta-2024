# src/utils/remove_irrelevant_features.py

def remove_irrelevant_features(df, columns):
    """
    Remove irrelevant features from the dataset.

    Args:
        df (pd.DataFrame): The raw dataset.

    Returns:
        pd.DataFrame: Dataset with irrelevant features removed.
    """
    irrelevant_columns = columns
    print(f"   └── Removing irrelevant columns: {', '.join(irrelevant_columns)}...")
    return df.drop(columns=irrelevant_columns, errors='ignore')
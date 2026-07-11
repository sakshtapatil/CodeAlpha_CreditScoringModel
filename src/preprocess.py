import pandas as pd

def preprocess_data(df):
    """
    Preprocess the dataset:
    1. Separate features (X) and target (y)
    2. Perform One-Hot Encoding on categorical columns

    Parameters:
        df (DataFrame): Input dataset

    Returns:
        X (DataFrame): Processed features
        y (Series): Target variable
    """

    # Separate features and target
    X = df.drop("credit_risk", axis=1)
    y = df["credit_risk"]

    # Find categorical columns
    categorical_columns = X.select_dtypes(include=["object"]).columns

    # Apply One-Hot Encoding
    X = pd.get_dummies(X, columns=categorical_columns)

    return X, y
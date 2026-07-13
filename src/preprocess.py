import pandas as pd

def preprocess_data(df):
  

    
    X = df.drop("credit_risk", axis=1)
    y = df["credit_risk"]

    
    categorical_columns = X.select_dtypes(include=["object"]).columns

    
    X = pd.get_dummies(X, columns=categorical_columns)

    return X, y
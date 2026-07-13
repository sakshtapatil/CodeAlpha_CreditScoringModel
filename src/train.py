import os

print("Current Working Directory:", os.getcwd())
import pandas as pd
df=pd.read_csv("data/GermanCredit.csv")
print(df["purpose"].unique())
print(df["employment_duration"].unique())
print(df["property"].unique())
print(df["job"].unique())

print(df.head(5))
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.isnull().sum())
from preprocess import preprocess_data

X, y = preprocess_data(df)
print(y.value_counts())
print(y.unique())
feature_columns = X.columns
print(X.shape)
print(X.dtypes)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)


y_pred = lr_model.predict(X_test)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)



accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


cm = confusion_matrix(y_test, y_pred)
print(cm)


print(classification_report(y_test, y_pred))


roc = roc_auc_score(y_test, y_pred)
print("ROC-AUC:", roc)

from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)


dt_model.fit(X_train, y_train)


dt_pred = dt_model.predict(X_test)


print("\n----- Decision Tree -----")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
print("ROC-AUC:", roc_auc_score(y_test, dt_pred))


from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


rf_model.fit(X_train, y_train)


rf_pred = rf_model.predict(X_test)




print("\n----- Random Forest -----")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_pred))


import joblib


joblib.dump(rf_model, "models/random_forest.pkl")


joblib.dump(scaler, "models/scaler.pkl")


joblib.dump(feature_columns, "models/feature_columns.pkl")

print("✅ Random Forest model saved successfully!")
print("✅ Scaler saved successfully!")
print("✅ Feature columns saved successfully!")
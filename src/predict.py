import joblib
import pandas as pd

# Load saved files
model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# New customer
new_customer = {
    "status": "... < 100 DM",
    "duration": 12,
    "credit_history": "critical account/other credits existing",
    "purpose": "car (new)",
    "amount": 3000,
    "savings": "... < 100 DM",
    "employment_duration": "1 <= ... < 4 years",
    "installment_rate": 4,
    "personal_status_sex": "male : single",
    "other_debtors": "none",
    "present_residence": 2,
    "property": "real estate",
    "age": 35,
    "other_installment_plans": "none",
    "housing": "own",
    "number_credits": 1,
    "job": "skilled employee / official",
    "people_liable": 1,
    "telephone": "yes",
    "foreign_worker": "yes"
}

# Convert to DataFrame
new_df = pd.DataFrame([new_customer])

# One-Hot Encoding
new_df = pd.get_dummies(new_df)

# Match training columns
new_df = new_df.reindex(columns=feature_columns, fill_value=0)

# Scale
new_df = scaler.transform(new_df)

# Predict
prediction = model.predict(new_df)

# Result
if prediction[0] == 1:
    print("✅ Good Credit Risk")
else:
    print("❌ Bad Credit Risk")
import streamlit as st
import pandas as pd
import joblib



model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")



st.title("🏦 Credit Scoring Prediction System")
st.write("Enter the customer details below.")



col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Loan Duration (Months)", min_value=1, value=12)
    amount = st.number_input("Loan Amount", min_value=100, value=3000)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    installment_rate = st.number_input("Installment Rate", min_value=1, max_value=4, value=2)

with col2:
    present_residence = st.number_input("Present Residence", min_value=1, max_value=4, value=2)
    number_credits = st.number_input("Number of Existing Credits", min_value=1, value=1)
    people_liable = st.number_input("People Liable", min_value=1, max_value=2, value=1)



status = st.selectbox(
    "Status",
    [
        "... < 100 DM",
        "0 <= ... < 200 DM",
        ">= 200 DM / salary assignments for at least 1 year",
        "no checking account"
    ]
)

credit_history = st.selectbox(
    "Credit History",
    [
        "critical account/other credits existing",
        "existing credits paid back duly till now",
        "delay in paying off in the past",
        "all credits at this bank paid back duly",
        "no credits taken/all credits paid back duly"
    ]
)

purpose = st.selectbox(
    "Purpose",
    [
        "domestic appliances",
        "retraining",
        "radio/television",
        "car (new)",
        "car (used)",
        "others",
        "repairs",
        "education",
        "furniture/equipment",
        "business"
    ]
)

savings = st.selectbox(
    "Savings",
    [
        "... < 100 DM",
        "100 <= ... < 500 DM",
        "500 <= ... < 1000 DM",
        ">= 1000 DM",
        "unknown/no savings account"
    ]
)

employment_duration = st.selectbox(
    "Employment Duration",
    [
        "... >= 7 years",
        "1 <= ... < 4 years",
        "4 <= ... < 7 years",
        "unemployed",
        "... < 1 year"
    ]
)

personal_status_sex = st.selectbox(
    "Personal Status / Sex",
    [
        "male : single",
        "female : divorced/separated/married",
        "male : married/widowed",
        "male : divorced/separated"
    ]
)

other_debtors = st.selectbox(
    "Other Debtors",
    [
        "none",
        "co-applicant",
        "guarantor"
    ]
)

property = st.selectbox(
    "Property",
    [
        "real estate",
        "building society savings agreement/life insurance",
        "unknown/no property",
        "car or other"
    ]
)

other_installment_plans = st.selectbox(
    "Other Installment Plans",
    [
        "bank",
        "stores",
        "none"
    ]
)

housing = st.selectbox(
    "Housing",
    [
        "own",
        "rent",
        "free"
    ]
)

job = st.selectbox(
    "Job",
    [
        "skilled employee/official",
        "unskilled - resident",
        "management/self-employed/highly qualified employee/officer",
        "unemployed/unskilled - non-resident"
    ]
)

telephone = st.selectbox(
    "Telephone",
    [
        "yes",
        "no"
    ]
)

foreign_worker = st.selectbox(
    "Foreign Worker",
    [
        "yes",
        "no"
    ]
)



if st.button("Predict Credit Risk"):

    customer = {
        "status": status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "amount": amount,
        "savings": savings,
        "employment_duration": employment_duration,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "present_residence": present_residence,
        "property": property,
        "age": age,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "number_credits": number_credits,
        "job": job,
        "people_liable": people_liable,
        "telephone": telephone,
        "foreign_worker": foreign_worker
    }

    
    new_df = pd.DataFrame([customer])

    
    new_df = pd.get_dummies(new_df)

    
    new_df = new_df.reindex(columns=feature_columns, fill_value=0)

    
    new_df = scaler.transform(new_df)

    

    
    prediction = model.predict(new_df)

    if prediction[0] == 1:
        st.success("✅ Good Credit Risk")
    else:
        st.error("❌ Bad Credit Risk")
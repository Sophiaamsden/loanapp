import streamlit as st
import pandas as pd
import pickle

# Load model
with open("my_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature columns
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.title("Loan Approval Predictor")

st.write("Enter the applicant details below:")

# ---- User Input Fields ----
user_input = {}

# Numerical fields
user_input["Granted_Loan_Amount"] = st.number_input("Granted Loan Amount", min_value=0)
user_input["FICO_score"] = st.number_input("FICO Score", min_value=0)
user_input["Monthly_Gross_Income"] = st.number_input("Monthly Gross Income", min_value=0)
user_input["Monthly_Housing_Payment"] = st.number_input("Monthly Housing Payment", min_value=0)

# Categorical fields
user_input["Reason"] = st.selectbox("Loan Reason", ["debt_consolidation", "credit_card", "home_improvement", "other"])
user_input["Employment_Status"] = st.selectbox("Employment Status", ["full_time", "part_time", "self_employed", "unemployed"])
user_input["Employment_Sector"] = st.selectbox("Employment Sector", ["information_technology", "finance", "healthcare", "Unknown"])
user_input["Lender"] = st.selectbox("Lender", ["A", "B", "C"])

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# One-hot encode to match training
input_encoded = pd.get_dummies(input_df)

# Reindex to match model columns
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Predict
if st.button("Predict Approval"):
    prediction = model.predict(input_encoded)[0]
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Denied")

# =============================================
# src/predict.py
# =============================================

import joblib
import pandas as pd


def predict(input_data):
    model = joblib.load('model.pkl')
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]

# =============================================
# app.py (Streamlit)
# =============================================

import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')

st.title("Customer Churn Prediction App")

# Example inputs
tenure = st.slider("Tenure", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0)

input_df = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

if st.button("Predict"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")

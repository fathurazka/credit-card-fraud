import streamlit as st
import pandas as pd
from joblib import dump, load
import requests
import os

# API URL - uses environment variable in production, localhost for development
MODEL_API_URL = os.environ.get("MODEL_API_URL", "http://127.0.0.1:5005/invocations")

st.title("Credit Card Fraud Detection", text_alignment="center")

with st.form("fraud_detection_form"):
    
    # adding columns
    col1, col2 = st.columns(2)
    
    with col1:
        distance_from_home = st.number_input("Distance from Home (in KM)", step=0.1)
        distance_from_last_transaction = st.number_input("Distance from Last Transaction (in KM)", step=0.1)
        ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", step=0.1)
        
    with col2:
        repeat_retailer = st.selectbox("Repeat Retailer", options=["No", "Yes"])
        used_chip = st.selectbox("Used Chip", options=["No", "Yes"])
        used_pin_number = st.selectbox("Used PIN Number", options=["No", "Yes"])

    online_order = st.selectbox("Online Order", options=["No", "Yes"])

    submit_button = st.form_submit_button("Predict Fraud", use_container_width=True)

if submit_button:
    columns = pd.read_csv("MLproject/data_columns.csv").columns.tolist()
    
    # Create input DataFrame with proper column order
    input_data = pd.DataFrame([[
        distance_from_home,
        distance_from_last_transaction,
        ratio_to_median_purchase_price,
        1 if repeat_retailer == "Yes" else 0,
        1 if used_chip == "Yes" else 0,
        1 if used_pin_number == "Yes" else 0,
        1 if online_order == "Yes" else 0
    ]], columns=columns)
    
    # Preprocess the data
    preprocessor = load("MLproject/preprocessor.joblib")
    processed_data = preprocessor.transform(input_data)
    
    payload = {
        "inputs": processed_data.tolist()
    }
    
    # Sending request to the MLflow model API
    response = requests.post(MODEL_API_URL, json=payload)
    if response.status_code == 200:
        prediction = response.json()
        res = prediction['predictions'][0]
        if res == 1:
            st.error("The transaction is predicted to be FRAUDULENT.")
        else:
            st.success("The transaction is predicted to be LEGITIMATE.")
    else:
        st.error(f"Error in prediction: {response.status_code} - {response.text}")
    
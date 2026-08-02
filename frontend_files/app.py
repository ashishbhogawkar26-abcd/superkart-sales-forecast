
import streamlit as st
import requests
import pandas as pd
import json

# --- Configuration ---
# IMPORTANT: Replace with your actual backend API URL. This URL is exposed by your GitHub Codespace when the backend container is running and its port is forwarded.
# Example: 'https://your-codespace-name-7860.app.github.dev'
BACKEND_API_URL = "http://backend:5000" # 'backend' is the service name in Docker Compose/network
PREDICT_URL = f"{BACKEND_API_URL}/v1/predict"
BATCH_PREDICT_URL = f"{BACKEND_API_URL}/v1/predictbatch"

# --- Streamlit App Layout ---
st.set_page_config(page_title="SuperKart Sales Predictor", layout="centered")

st.title("🛒 SuperKart Sales Predictor")
st.markdown("Welcome to the SuperKart Sales Prediction application! Enter the product and store details below to get an estimated sales value.")

# --- Input Form ---
st.header("Product and Store Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        product_weight = st.number_input("Product Weight (grams)", min_value=1.0, max_value=25.0, value=10.0, step=0.1)
        product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'Regular', 'No Sugar', 'reg'])
        product_allocated_area = st.number_input("Product Allocated Area (ratio)", min_value=0.0, max_value=0.2, value=0.07, step=0.001, format="%.3f")
        product_mrp = st.number_input("Product MRP (Maximum Retail Price)", min_value=50.0, max_value=300.0, value=150.0, step=0.1)

    with col2:
        store_size = st.selectbox("Store Size", ['Small', 'Medium', 'High'])
        store_location_city_type = st.selectbox("Store Location City Type", ['Tier 1', 'Tier 2', 'Tier 3'])
        store_type = st.selectbox("Store Type", ['Supermarket Type1', 'Supermarket Type2', 'Departmental Store', 'Food Mart'])
        store_age_years = st.number_input("Store Age (Years)", min_value=1, max_value=50, value=10, step=1)
        product_type = st.selectbox("Product Type", [
            'Snack Foods', 'Dairy', 'Soft Drinks', 'Household', 'Health and Hygiene',
            'Fruits and Vegetables', 'Baking Goods', 'Frozen Foods', 'Meat',
            'Canned', 'Breads', 'Hard Drinks', 'Others', 'Starchy Foods',
            'Breakfast', 'Seafood'
        ])

    # Note: Product_Id and Store_Id are typically excluded from direct user input for prediction
    # or handled internally if needed. For this demo, we assume the model pipeline handles features directly.

    submitted = st.form_submit_button("Get Sales Prediction")

    if submitted:
        # Prepare payload for the API request
        payload = {
            

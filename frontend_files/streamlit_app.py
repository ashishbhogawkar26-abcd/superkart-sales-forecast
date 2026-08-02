import streamlit as st
import requests
import pandas as pd
import json

# Set page title and layout
st.set_page_config(page_title="SuperKart Sales Forecast", layout="wide")

st.title("🛒 SuperKart Sales Forecasting")
st.markdown("Predict total sales revenue for products across different store outlets.")

# Backend URL (Internal Docker network address)
BACKEND_URL = "http://backend:7860"

# Sidebar for navigation
app_mode = st.sidebar.selectbox("Choose Prediction Mode", ["Single Prediction", "Batch Prediction"])

if app_mode == "Single Prediction":
    st.header("Single Product Sales Prediction")

    col1, col2 = st.columns(2)

    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.0)
        sugar_content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        allocated_area = st.slider("Allocated Area Ratio", 0.0, 0.3, 0.05)
        mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
        product_id_char = st.selectbox("Product Category ID", ["FD", "DR", "NC"])

    with col2:
        store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
        city_type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
        store_age = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=15)
        type_cat = st.selectbox("Perishability", ["Non Perishables", "Perishables"])

    if st.button("Predict Sales"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar_content,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category": type_cat
        }

        try:
            response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)
            result = response.json()
            if "prediction" in result:
                st.success(f"### Estimated Total Sales: ${result['prediction']:,.2f}")
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")

else:
    st.header("Batch Sales Prediction")
    st.markdown("Upload a CSV file with product and store attributes to get bulk forecasts.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        if st.button("Process Batch"):
            files = {'file': uploaded_file.getvalue()}
            try:
                response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files)
                predictions = response.json()

                # Display as a dataframe
                df_results = pd.DataFrame(list(predictions.items()), columns=['Row Index', 'Predicted Sales'])
                st.write(df_results)

                # Download button
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", csv, "forecasts.csv", "text/csv")
            except Exception as e:
                st.error(f"Batch processing failed: {e}")

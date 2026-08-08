import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SuperKart Sales Forecast", layout="wide")

st.title("SuperKart Sales Forecasting")
st.markdown("Predict total sales revenue for products across different store outlets.")

# 'backend' is the docker-compose service name on the shared network
BACKEND_URL = "http://backend:7860"

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
        # These strings must match the fitted OneHotEncoder categories exactly.
        store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
        city_type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
        store_age = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=15)
        type_cat = st.selectbox("Perishability", ["Non Perishables", "Perishables"])

    if st.button("Predict Sales"):
        # Store_Id is deliberately absent: it is dropped during training.
        payload = {
            "Product_Weight": product_weight,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": mrp,
            "Store_Age_Years": store_age,
            "Product_Sugar_Content": sugar_content,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Product_Type_Category": type_cat,
        }

        try:
            response = requests.post(BACKEND_URL + "/v1/predict", json=payload, timeout=60)
            try:
                result = response.json()
            except ValueError:
                st.error("Backend returned status " + str(response.status_code) + ": " + response.text[:300])
                result = None

            if result is not None:
                if "prediction" in result:
                    st.success("### Estimated Total Sales: $" + format(result["prediction"], ",.2f"))
                else:
                    st.error("Backend error " + str(response.status_code) + ": " + str(result.get("error", result)))
        except Exception as e:
            st.error("Could not connect to backend: " + str(e))

else:
    st.header("Batch Sales Prediction")

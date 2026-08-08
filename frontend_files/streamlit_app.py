import io
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="SuperKart Sales Forecast", layout="wide")
st.title("SuperKart Sales Forecasting")
st.markdown("Predict total sales revenue for products across different store outlets.")

# 'backend' is the docker-compose service name on the shared network
BACKEND_URL = "http://backend:7860"

FALLBACK_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Product_Id_char",
    "Product_Type_Category",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Store_Age_Years",
]

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
            response = requests.post(BACKEND_URL + "/v1/predict", json=payload, timeout=30)
            try:
                result = response.json()
            except ValueError:
                st.error("Backend returned status " + str(response.status_code) + ": " + response.text[:300])
                result = None
            if result is not None:
                if response.status_code == 200 and "prediction" in result:
                    st.success("### Estimated Total Sales: $" + format(result["prediction"], ",.2f"))
                else:
                    st.error("Backend error " + str(response.status_code) + ": " + str(result.get("error", result)))
        except Exception as e:
            st.error("Could not connect to backend: " + str(e))

else:
    st.header("Batch Sales Prediction")

    # Ask the backend which columns it expects, so the UI can never drift from the model.
    expected = list(FALLBACK_COLUMNS)
    try:
        info = requests.get(BACKEND_URL + "/", timeout=10).json()
        if isinstance(info, dict) and info.get("expected_columns"):
            expected = list(info["expected_columns"])
    except Exception:
        pass

    st.markdown("Upload a CSV with one row per product/store combination.")
    st.markdown("Required columns: " + ", ".join(expected))

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        try:
            preview_df = pd.read_csv(io.BytesIO(raw_bytes))
        except Exception as e:
            preview_df = None
            st.error("Could not read the uploaded file as CSV: " + str(e))

        if preview_df is not None:
            st.write("Preview of uploaded data (" + str(len(preview_df)) + " rows):")
            st.dataframe(preview_df.head(20))

            missing = [c for c in expected if c not in preview_df.columns]
            if missing:
                st.error("Uploaded file is missing required column(s): " + str(missing))
            elif st.button("Predict Batch Sales"):
                resp = None
                with st.spinner("Scoring..."):
                    try:
                        resp = requests.post(
                            BACKEND_URL + "/v1/predictbatch",
                            files={"file": ("batch.csv", raw_bytes, "text/csv")},
                            timeout=120,
                        )
                    except Exception as e:
                        st.error("Could not connect to backend: " + str(e))

                if resp is not None and resp.status_code == 200:
                    preds = resp.json()
                    out = preview_df.copy()
                    out["Predicted_Sales"] = [preds.get(str(i)) for i in range(len(out))]
                    st.success("Scored all " + str(len(out)) + " rows.")
                    st.dataframe(out)
                    st.download_button(
                        "Download predictions as CSV",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="superkart_batch_predictions.csv",
                        mime="text/csv",
                    )

                elif resp is not None:
                    try:
                        message = resp.json().get("error", resp.text[:300])
                    except ValueError:
                        message = resp.text[:300]
                    st.error("Backend rejected the file (status " + str(resp.status_code) + "): " + str(message))

                    if len(preview_df) <= 200:
                        st.info("The backend validates the file as a whole. Re-scoring row by row so valid rows are still forecast and invalid rows are flagged.")
                        values = []
                        reasons = []
                        progress = st.progress(0.0)
                        for i in range(len(preview_df)):
                            row_csv = preview_df.iloc[[i]].to_csv(index=False).encode("utf-8")
                            try:
                                r = requests.post(
                                    BACKEND_URL + "/v1/predictbatch",
                                    files={"file": ("row.csv", row_csv, "text/csv")},
                                    timeout=60,
                                )
                                if r.status_code == 200:
                                    values.append(float(r.json()["0"]))
                                    reasons.append("")
                                else:
                                    values.append(None)
                                    try:
                                        reasons.append(str(r.json().get("error", r.text[:200])))
                                    except ValueError:
                                        reasons.append(r.text[:200])
                            except Exception as e:
                                values.append(None)
                                reasons.append(str(e))
                            progress.progress((i + 1) / len(preview_df))
                        progress.empty()

                        out = preview_df.copy()
                        out["Predicted_Sales"] = values
                        out["Rejection_Reason"] = reasons
                        n_ok = sum(1 for v in values if v is not None)
                        st.warning(
                            "Scored " + str(n_ok) + " of " + str(len(out)) + " rows. "
                            + str(len(out) - n_ok) + " row(s) could not be scored and are flagged below."
                        )
                        st.dataframe(out)
                        st.download_button(
                            "Download results as CSV",
                            out.to_csv(index=False).encode("utf-8"),
                            file_name="superkart_batch_predictions.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("File has more than 200 rows. Please correct the invalid values and re-upload.")

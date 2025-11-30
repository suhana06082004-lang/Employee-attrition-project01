import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("✨ Employee Attrition Prediction App")

st.write("""
Upload a CSV file (same structure as your processed test data)  
and the model will predict whether each employee is likely to leave.
""")

# Load model
model = joblib.load("models/logistic_model_improved.joblib")

uploaded_file = st.file_uploader("Upload Employee Data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.write(df.head())

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]

    df["Attrition_Pred"] = predictions
    df["Attrition_Probability"] = probabilities

    st.write("### Predictions:")
    st.write(df[["Attrition_Pred", "Attrition_Probability"]].head())

    # download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

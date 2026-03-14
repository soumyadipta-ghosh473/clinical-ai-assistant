import streamlit as st
import pandas as pd
import joblib
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from io import BytesIO

# ---- PROMPT VERSIONING ----
def load_prompt():
    with open("prompts/clinical_prompt_v1.txt", "r") as f:
        prompt = f.read()
    return prompt

st.set_page_config(page_title="AI Clinical Assistant", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.main-title {
font-size:42px;
font-weight:700;
color:#2E8B57;
}
.box {
background-color:#F0F8FF;
padding:20px;
border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">AI Clinical Assistant</p>', unsafe_allow_html=True)

model = joblib.load("models/xgboost_model.pkl")

st.sidebar.title("Model Information")

st.sidebar.write("**Model:** XGBoost Classifier")
st.sidebar.write("**Type:** Gradient Boosting Ensemble")
st.sidebar.write("**Architecture:** 200 Decision Trees")
st.sidebar.write("**Explainability:** SHAP")

# ---------- PATIENT INPUT FORM ----------

st.subheader("Enter Patient Clinical Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input("Temperature", value=37.0)
    heart_rate = st.number_input("Heart Rate", value=80)
    wbc = st.number_input("WBC", value=7.0)
    hemoglobin = st.number_input("Hemoglobin", value=14.0)

with col2:
    bp_sys = st.number_input("Systolic BP", value=120)
    bp_dia = st.number_input("Diastolic BP", value=80)
    glucose = st.number_input("Blood Glucose", value=100)
    spo2 = st.number_input("SpO2", value=98)

with col3:
    creatinine = st.number_input("Creatinine", value=1.0)
    resp_rate = st.number_input("Respiratory Rate", value=16)
    icu_stay = st.number_input("ICU Length of Stay", value=3)
    lab_tests = st.number_input("Number of Lab Tests", value=5)

if st.button("Predict Risk"):

    patient_data = pd.DataFrame({
        "Temperature":[temperature],
        "Blood_Pressure_Systolic":[bp_sys],
        "Blood_Pressure_Diastolic":[bp_dia],
        "Creatinine":[creatinine],
        "Hemoglobin":[hemoglobin],
        "WBC":[wbc],
        "Heart_Rate":[heart_rate],
        "Blood_Glucose":[glucose],
        "SpO2":[spo2],
        "Respiratory_Rate":[resp_rate],
        "ICU_Length_of_Stay":[icu_stay],
        "Number_of_Lab_Tests":[lab_tests]
    })

    prediction = model.predict_proba(patient_data)[:,1]
    risk = prediction[0]

    st.metric("Readmission Risk Score", round(risk,3))

    if risk > 0.7:
        st.error("High Risk")
    else:
        st.success("Low Risk")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)

    shap_df = pd.DataFrame({
        "Feature": patient_data.columns,
        "Importance": abs(shap_values[0])
    })

    top_features = shap_df.sort_values("Importance", ascending=False).head(5)

    st.subheader("Top Risk Factors")
    st.table(top_features)

# -------- Clinical Explanation using Prompt --------

prompt_template = load_prompt()

features = ", ".join(top_features["Feature"].values)

clinical_explanation = prompt_template.replace("{features}", features)

st.subheader("Clinical Interpretation")
st.write(clinical_explanation)

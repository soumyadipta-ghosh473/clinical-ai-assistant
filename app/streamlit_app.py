import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq
import os
import numpy as np
import pathlib
import sys

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from io import BytesIO

APP_VERSION = "1.5.0"

# ---------- PATH FIX ----------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

model_path = BASE_DIR / "models" / "xgboost_model.pkl"
feature_path = BASE_DIR / "models" / "feature_names.pkl"

from src.multimodal import multimodal_fusion

st.set_page_config(page_title="AI Clinical Assistant", page_icon="🩺", layout="wide")

# ---------- STYLING ----------
st.markdown("""
<style>
.block-container{padding-top:1rem;}
.section{
background:white;
padding:20px;
border-radius:12px;
box-shadow:0px 3px 10px rgba(0,0,0,0.08);
margin-bottom:25px;
}
.footer{font-size:13px;color:gray;text-align:center;}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center; color:#1E88E5; font-size:42px;'>
🩺 AI Clinical Assistant Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown(f"<p class='footer'>Version {APP_VERSION}</p>", unsafe_allow_html=True)

# ---------- LOAD ----------
model = joblib.load(model_path)

try:
    feature_names = joblib.load(feature_path)
except:
    feature_names = None

# ---------- SIDEBAR ----------
st.sidebar.title("Model Information")
st.sidebar.write("**Model:** XGBoost")
st.sidebar.write("**Architecture:** Gradient Boosting Trees")
st.sidebar.write("**Explainability:** SHAP")

# ---------- CHATBOT (FIXED) ----------
st.sidebar.markdown("## 👨‍⚕️ Doctor AI Assistant")

question = st.sidebar.text_input("Ask a medical question")

if question:
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":"Only answer medical questions."},
                {"role":"user","content":question}
            ]
        )

        st.sidebar.write(r.choices[0].message.content)

    except:
        st.sidebar.warning("LLM not working")

# ---------- ICU MONITOR ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("ICU Live Monitoring Panel")

time = np.arange(0, 20)
hr = np.random.normal(80, 3, 20)
temp = np.random.normal(37, 0.2, 20)
spo2 = np.random.normal(98, 1, 20)
bp = np.random.normal(120, 5, 20)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(px.line(x=time, y=hr), use_container_width=True)
    st.plotly_chart(px.line(x=time, y=temp), use_container_width=True)

with col2:
    st.plotly_chart(px.line(x=time, y=spo2), use_container_width=True)
    st.plotly_chart(px.line(x=time, y=bp), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------- INPUT ----------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.subheader("Patient Clinical Parameters")

c1, c2, c3 = st.columns(3)

with c1:
    temperature = st.number_input("Temperature", 37.0)
    heart_rate = st.number_input("Heart Rate", 80)
    wbc = st.number_input("WBC", 7.0)
    hemoglobin = st.number_input("Hemoglobin", 14.0)

with c2:
    bp_sys = st.number_input("Systolic BP", 120)
    bp_dia = st.number_input("Diastolic BP", 80)
    glucose = st.number_input("Blood Glucose", 100)
    spo2 = st.number_input("SpO2", 98)

with c3:
    creatinine = st.number_input("Creatinine", 1.0)
    resp = st.number_input("Respiratory Rate", 16)
    icu = st.number_input("ICU Length of Stay", 3)
    labs = st.number_input("Number of Lab Tests", 5)
    meds = st.number_input("Number of Medications", 2)

clinical_note = st.text_area("Clinical Notes (Doctor Input)")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICT ----------
if st.button("Predict Risk"):

    data = pd.DataFrame({
        "ICU_Length_of_Stay":[icu],
        "Blood_Glucose":[glucose],
        "Creatinine":[creatinine],
        "Hemoglobin":[hemoglobin],
        "WBC":[wbc],
        "Heart_Rate":[heart_rate],
        "Blood_Pressure_Systolic":[bp_sys],
        "Blood_Pressure_Diastolic":[bp_dia],
        "SpO2":[spo2],
        "Respiratory_Rate":[resp],
        "Temperature":[temperature],
        "Number_of_Lab_Tests":[labs],
        "Number_of_Medications":[meds]
    })

    if feature_names:
        for col in feature_names:
            if col not in data.columns:
                data[col] = 0
        data = data[feature_names]

    pred = model.predict_proba(data)[:,1]
    risk = pred[0]

    st.metric("Readmission Risk Score", round(risk,3))

    # ---------- GAUGE ----------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text':"Readmission Risk"},
        gauge={'axis':{'range':[0,1]},
        'steps':[{'range':[0,0.4],'color':'green'},
                 {'range':[0.4,0.7],'color':'yellow'},
                 {'range':[0.7,1],'color':'red'}]}
    ))
    st.plotly_chart(fig)

    st.markdown("---")

    # ---------- SHAP ----------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    shap_df = pd.DataFrame({
        "Feature":data.columns,
        "Importance":abs(shap_values[0])
    })

    top = shap_df.sort_values("Importance",ascending=False).head(5)

    st.subheader("Top Risk Factors")
    st.table(top)

    fig, ax = plt.subplots(figsize=(8,4))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=data.iloc[0],
            feature_names=data.columns
        ),
        show=False
    )
    st.pyplot(fig)

    st.markdown("---")

    # ---------- MULTIMODAL ----------
    features = ", ".join(top["Feature"].values)

    fusion_output = multimodal_fusion(
        risk,
        features,
        clinical_note
    )

    st.subheader("Multimodal Clinical Decision")
    st.write(fusion_output)

    st.markdown("---")

    # ---------- REPORT ----------
    reasoning = f"""
Risk Score: {round(risk,3)}

Primary Risk Factors:
{features}
"""

    st.subheader("AI Clinical Assessment")
    st.write(reasoning)

    buffer = BytesIO()
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("AI Clinical Risk Report",styles['Title']))
    elements.append(Spacer(1,20))
    elements.append(Paragraph(reasoning,styles['Normal']))

    table_data=[list(data.columns)]+data.values.tolist()
    elements.append(Table(table_data))

    doc = SimpleDocTemplate(buffer,pagesize=letter)
    doc.build(elements)

    st.download_button("Download PDF Report",buffer.getvalue(),"clinical_report.pdf")
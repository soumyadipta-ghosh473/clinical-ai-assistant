\# 🩺 AI Clinical Assistant



An AI-powered \*\*Clinical Decision Support System\*\* that predicts ICU readmission risk using machine learning and explainable AI.



The system integrates \*\*XGBoost prediction, SHAP explainability, Streamlit dashboards, prompt versioning, and CI/CD automation\*\*.



\---



\# 🚀 Features



• ICU Readmission Risk Prediction

• SHAP Explainable AI

• Interactive Clinical Dashboard

• Risk Gauge Visualization

• AI Doctor Chatbot

• Prompt Versioning

• Clinical Interpretation Layer

• PDF Clinical Report Generation

• CI/CD Pipeline using GitHub Actions



\---



\# 🧠 Model



Model Used: \*\*XGBoost Classifier\*\*



Architecture:



\- Gradient Boosting Ensemble

\- 200 Decision Trees

\- Max Depth: 6

\- Learning Rate: 0.05



Explainability:



\- SHAP (SHapley Additive Explanations)



Evaluation Metrics:



\- AUROC

\- AUPRC

\- Calibration Curve



\---



\# 🏗 System Architecture



Clinical Dataset

↓

Data Preprocessing (KNN + Percentile Clipping)

↓

XGBoost Prediction Model

↓

SHAP Explainability

↓

Clinical Interpretation Layer

↓

Streamlit Clinical Dashboard

↓

PDF Medical Report



\---



\# 🛠 Tech Stack



Python

XGBoost

Streamlit

SHAP

Plotly

Scikit-Learn

Pandas

ReportLab

GitHub Actions



\---



\# 📊 Dashboard Capabilities



The dashboard allows clinicians to:



• Enter patient parameters

• Predict ICU readmission risk

• Visualize risk using a gauge chart

• View SHAP feature importance plots

• Generate downloadable clinical reports

• Ask questions using the AI Doctor Assistant



\---



\# 🔄 CI/CD Pipeline



GitHub Actions automatically performs:



• Dependency installation

• Pipeline validation

• Model environment testing



\---



\## Ethical Compliance



\- MIMIC-IV dataset requires CITI training for access

\- Data is de-identified under HIPAA Safe Harbor standards

\- This system uses only de-identified patient data

\- No real patient-identifiable information is used



\## Prompt Robustness Testing



The system evaluates LLM performance on incomplete and noisy clinical inputs such as:



\- "Patient fever"

\- "HR high maybe infection ???"

\- "tachycardia but stable??"



This ensures robustness in real-world clinical scenarios.

## Ethical Compliance



\- MIMIC dataset requires CITI training certification

\- Data is de-identified under HIPAA Safe Harbor

\- This project complies with PhysioNet data usage agreement

\- No personal patient data is exposed



\## Project Architecture

clinical-ai-assistant

│

├── app

│   └── streamlit\_app.py

│

├── src

│   ├── preprocessing.py

│   ├── train\_model.py

│   ├── mimic\_pipeline.py

│   ├── calibration.py

│   ├── bootstrap\_eval.py

│   ├── bias\_analysis.py

│   └── prompt\_testing.py

│

├── data

├── models

├── prompts



\# 👨‍⚕️ Author



Soumyadipta Ghosh


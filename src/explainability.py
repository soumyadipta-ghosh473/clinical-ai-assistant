import shap
import joblib
import matplotlib.pyplot as plt
from src.preprocessing import preprocess_pipeline

# Load trained model
model = joblib.load("models/xgboost_model.pkl")

# Load processed dataset
df = preprocess_pipeline("data/mimic_iii_data.csv")

# Remove non-ML columns
X = df.drop(columns=[
    "Readmission_Flag",
    "Diagnoses",
    "Medications",
    "Patient_ID",
    "ICU_Admission_ID"
])

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# Create SHAP summary plot
shap.summary_plot(shap_values, X)

# Force plot window to display
plt.show()
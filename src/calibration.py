import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.preprocessing import preprocess_pipeline

# 🔥 NEW IMPORTS (ADDED)
from src.bootstrap_eval import bootstrap_auc
from src.bias_analysis import evaluate_bias


# Load dataset
df = preprocess_pipeline("data/mimic_iii_data.csv")

# Target
y = df["Readmission_Flag"]

# Features
X = df.drop(columns=[
    "Readmission_Flag",
    "Diagnoses",
    "Medications",
    "Patient_ID",
    "ICU_Admission_ID"
], errors="ignore")


# Train test split (UNCHANGED)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Load trained model
model = joblib.load("models/xgboost_model.pkl")


# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]


# ---------- CALIBRATION (UNCHANGED) ----------
prob_true, prob_pred = calibration_curve(
    y_test,
    probs,
    n_bins=10
)

plt.figure()

plt.plot(prob_pred, prob_true, marker='o', label="Model")
plt.plot([0, 1], [0, 1], linestyle='--', label="Perfect")

plt.xlabel("Predicted Probability")
plt.ylabel("Actual Probability")
plt.title("Calibration Curve")

plt.legend()
plt.grid()

plt.show()


# ---------- NEW: BOOTSTRAP CONFIDENCE INTERVAL ----------
ci = bootstrap_auc(y_test, probs)

print("\nBootstrap AUC Confidence Interval (95%):")
print(f"Lower: {round(ci[0],3)} | Upper: {round(ci[1],3)}")


# ---------- NEW: BIAS ANALYSIS ----------
# Create temporary dataframe for bias evaluation

bias_df = X_test.copy()

bias_df["y"] = y_test.values
bias_df["pred"] = probs

# Only runs if Gender exists (safe)
evaluate_bias(bias_df)
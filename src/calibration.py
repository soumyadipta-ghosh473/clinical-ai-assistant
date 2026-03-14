import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_pipeline


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
])

# Train test split
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

# Compute calibration
prob_true, prob_pred = calibration_curve(
    y_test,
    probs,
    n_bins=10
)

# Plot calibration curve
plt.figure()

plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("Predicted Probability")
plt.ylabel("Actual Probability")
plt.title("Calibration Curve")

plt.show()
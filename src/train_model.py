import joblib
import xgboost as xgb
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from src.preprocessing import preprocess_pipeline
from src.split import subject_level_split

# Load data
df = preprocess_pipeline("data/mimic_iii_data.csv", use_mice=True)

# ---------- SUBJECT-LEVEL SPLIT ----------
train_df, test_df = subject_level_split(df)

# Target
y_train = train_df["Readmission_Flag"]
y_test = test_df["Readmission_Flag"]

# Features
X_train = train_df.drop(columns=[
    "Readmission_Flag",
    "Diagnoses",
    "Medications",
    "Patient_ID",
    "ICU_Admission_ID"
], errors="ignore")

X_test = test_df.drop(columns=[
    "Readmission_Flag",
    "Diagnoses",
    "Medications",
    "Patient_ID",
    "ICU_Admission_ID"
], errors="ignore")

# ---------- MODEL ----------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ---------- PREDICTIONS ----------
pred = model.predict_proba(X_test)[:, 1]

# ---------- METRICS ----------
roc = roc_auc_score(y_test, pred)
pr = average_precision_score(y_test, pred)

print("AUROC:", roc)
print("AUPRC:", pr)

# ---------- SAVE MODEL ----------
joblib.dump(model, "models/xgboost_model.pkl")

print("Model saved in models/xgboost_model.pkl")
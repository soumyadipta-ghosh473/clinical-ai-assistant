import joblib
import xgboost as xgb
import numpy as np
import os  # ✅ NEW

from sklearn.metrics import roc_auc_score, average_precision_score

from src.preprocessing import preprocess_pipeline
from src.split import subject_level_split
from src.temporal_features import create_temporal_features


# ---------- ENSURE MODELS DIR EXISTS ----------
os.makedirs("models", exist_ok=True)


# ---------- LOAD DATA ----------
df = preprocess_pipeline("data/mimic_iii_data.csv", use_mice=True)


# ---------- TEMPORAL FEATURE ENGINEERING ----------
df_temporal = create_temporal_features(df)


# ---------- MERGE TARGET ----------
target_df = df[["Patient_ID", "Readmission_Flag"]].drop_duplicates()
df = df_temporal.merge(target_df, on="Patient_ID")


# ---------- SUBJECT-LEVEL SPLIT ----------
train_df, test_df = subject_level_split(df)


# ---------- TARGET ----------
y_train = train_df["Readmission_Flag"]
y_test = test_df["Readmission_Flag"]


# ---------- FEATURES ----------
X_train = train_df.drop(columns=[
    "Readmission_Flag",
    "Patient_ID"
], errors="ignore")

X_test = test_df.drop(columns=[
    "Readmission_Flag",
    "Patient_ID"
], errors="ignore")


# ---------- MODEL ----------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric="logloss"
)


# ---------- TRAIN ----------
model.fit(X_train, y_train)


# ---------- PREDICT ----------
pred = model.predict_proba(X_test)[:, 1]


# ---------- METRICS ----------
roc = roc_auc_score(y_test, pred)
pr = average_precision_score(y_test, pred)

print("AUROC:", roc)
print("AUPRC:", pr)


# ---------- SAVE MODEL ----------
joblib.dump(model, "models/xgboost_model.pkl")
print("Model saved in models/xgboost_model.pkl")


# ---------- SAVE FEATURE NAMES ----------
joblib.dump(X_train.columns.tolist(), "models/feature_names.pkl")
print("Feature names saved in models/feature_names.pkl")
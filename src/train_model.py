import pandas as pd
import joblib
import xgboost as xgb

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from src.preprocessing import preprocess_pipeline


# ---------- LOAD DATA ----------
df = preprocess_pipeline("data/mimic_iii_data.csv")


# ---------- FEATURES & TARGET ----------
y = df["Readmission_Flag"]

X = df.drop(columns=[
    "Readmission_Flag",
    "Diagnoses",
    "Medications",
    "Patient_ID",
    "ICU_Admission_ID"
], errors="ignore")


# ---------- PATIENT-LEVEL SPLIT ----------
if "Patient_ID" in df.columns:

    groups = df["Patient_ID"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("Using Patient-Level Split (subject_id)")

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("Using Standard Train-Test Split")


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
print("Model saved in models/xgboost_model.pkl")
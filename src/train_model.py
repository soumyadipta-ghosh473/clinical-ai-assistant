import joblib
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_pipeline


# Load processed dataset
df = preprocess_pipeline("data/mimic_iii_data.csv")

# Define target variable
y = df["Readmission_Flag"]

# Define features (remove non-ML columns)
X = df.drop(columns=[
    "Readmission_Flag",
    "Diagnoses",
    "Medications",
    "Patient_ID",
    "ICU_Admission_ID"
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create XGBoost model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric="logloss"
)

# Train model
model.fit(X_train, y_train)

# Predict probabilities
pred = model.predict_proba(X_test)[:, 1]

# Evaluate model
roc = roc_auc_score(y_test, pred)
pr = average_precision_score(y_test, pred)

print("AUROC:", roc)
print("AUPRC:", pr)

# Save trained model
joblib.dump(model, "models/xgboost_model.pkl")

print("Model saved in models/xgboost_model.pkl")
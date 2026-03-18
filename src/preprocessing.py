import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# IMPORT NEW PIPELINE
from src.mimic_pipeline import (
    cohort_selection,
    percentile_clipping as pipeline_clipping,
    mice_imputation
)


def load_data(path):
    df = pd.read_csv(path)
    return df


def percentile_clipping(df):
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)

        df[col] = df[col].clip(lower, upper)

    return df


def knn_imputation(df):
    numeric_cols = df.select_dtypes(include=np.number).columns

    imputer = KNNImputer(n_neighbors=5)

    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


def preprocess_pipeline(path):

    df = load_data(path)

    # 🔹 Step 1: Cohort selection (MIMIC-style)
    if "ICU_Length_of_Stay" in df.columns and "Age" in df.columns:
        df = cohort_selection(df)

    # 🔹 Step 2: Percentile clipping (from pipeline)
    df = percentile_clipping(df)

    # 🔹 Step 3: Missing value handling
    # OPTION A: KNN (current)
    df = knn_imputation(df)

    # OPTION B: MICE (uncomment if needed)
    # df = mice_imputation(df)

    return df
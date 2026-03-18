import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def load_data(path):
    df = pd.read_csv(path)
    return df


# ---------- PERCENTILE CLIPPING ----------
def percentile_clipping(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    return df


# ---------- KNN IMPUTATION ----------
def knn_imputation(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    imputer = KNNImputer(n_neighbors=5)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


# ---------- MICE IMPUTATION ----------
def mice_imputation(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    imputer = IterativeImputer(max_iter=10, random_state=42)
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


# ---------- MAIN PIPELINE ----------
def preprocess_pipeline(path, use_mice=True):

    df = load_data(path)

    df = percentile_clipping(df)

    df = add_missing_indicators(df)  # NEW

    if use_mice:
        df = mice_imputation(df)
    else:
        df = knn_imputation(df)

    return df

# ---------- MISSINGNESS INDICATORS ----------
def add_missing_indicators(df):

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    return df
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def load_mimic_data(path):
    df = pd.read_csv(path)
    return df


def cohort_selection(df):

    df = df[df["ICU_Length_of_Stay"] >= 1]
    df = df[df["Age"] >= 18]

    return df


def percentile_clipping(df):

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    return df


def mice_imputation(df):

    imputer = IterativeImputer(max_iter=10)

    numeric_cols = df.select_dtypes(include=np.number).columns

    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


def temporal_features(df):

    agg = df.groupby("ICU_Admission_ID").agg({
        "Heart_Rate": ["mean","max","min","std"],
        "Temperature": ["mean","max"],
        "SpO2": ["mean","min"]
    })

    agg.columns = ["_".join(c) for c in agg.columns]

    agg.reset_index(inplace=True)

    return agg
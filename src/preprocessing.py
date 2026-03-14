import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


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

    df = percentile_clipping(df)

    df = knn_imputation(df)

    return df
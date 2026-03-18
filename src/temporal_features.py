import pandas as pd
import numpy as np


def create_temporal_features(df):

    # ---------- ASSUMPTION ----------
    # If timestamp exists → use real window
    # else → simulate grouping by Patient_ID

    group_col = "Patient_ID"

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Remove target & IDs
    drop_cols = ["Readmission_Flag", "Patient_ID", "ICU_Admission_ID"]
    numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    agg_df = df.groupby(group_col)[numeric_cols].agg([
        "mean",
        "min",
        "max",
        "std"
    ])

    # Flatten column names
    agg_df.columns = ["_".join(col) for col in agg_df.columns]

    agg_df = agg_df.reset_index()

    return agg_df
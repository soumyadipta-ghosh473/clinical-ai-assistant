import pandas as pd


def create_temporal_features(df, group_col="Patient_ID"):

    numeric_cols = df.select_dtypes(include="number").columns

    # Remove target
    numeric_cols = [c for c in numeric_cols if c != "Readmission_Flag"]

    agg_df = df.groupby(group_col)[numeric_cols].agg([
        "mean", "min", "max", "std"
    ])

    # Flatten column names
    agg_df.columns = ["_".join(col) for col in agg_df.columns]

    agg_df = agg_df.reset_index()

    return agg_df
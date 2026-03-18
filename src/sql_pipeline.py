import pandas as pd


def build_cohort(patients, admissions, diagnoses):

    # ---------- JOIN TABLES ----------
    df = patients.merge(admissions, on="subject_id", how="inner")
    df = df.merge(diagnoses, on="hadm_id", how="inner")

    # ---------- CLEAN ----------
    df = df.drop_duplicates()

    # ---------- INCLUSION ----------
    if "age" in df.columns:
        df = df[df["age"] >= 18]

    if "icd_code" in df.columns:
        df = df[df["icd_code"].str.startswith("I50", na=False)]

    # ---------- EXCLUSION ----------
    if "los" in df.columns:
        df = df[df["los"] >= 1]

    return df
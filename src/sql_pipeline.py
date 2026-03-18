import pandas as pd


def build_cohort(patients, admissions, diagnoses):

    # Join tables
    df = patients.merge(admissions, on="subject_id")
    df = df.merge(diagnoses, on="hadm_id")

    # ---------- INCLUSION ----------
    df = df[df["age"] >= 18]

    # ICD filter (Heart Failure example)
    df = df[df["icd_code"].str.startswith("I50")]

    # ---------- EXCLUSION ----------
    df = df[df["los"] >= 1]  # at least 24 hrs

    return df
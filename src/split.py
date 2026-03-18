import pandas as pd
from sklearn.model_selection import train_test_split


def subject_level_split(df, subject_col="Patient_ID", test_size=0.2, random_state=42):
    """
    Split dataset at patient level to avoid data leakage.
    """

    # Unique patients
    unique_subjects = df[subject_col].unique()

    # Split subjects
    train_subj, test_subj = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_state
    )

    # Create splits
    train_df = df[df[subject_col].isin(train_subj)]
    test_df = df[df[subject_col].isin(test_subj)]

    return train_df, test_df
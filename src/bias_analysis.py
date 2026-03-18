from sklearn.metrics import roc_auc_score


def evaluate_bias(df, y_true_col="y", pred_col="pred", group_col="Gender"):

    # Check if column exists
    if group_col not in df.columns:
        print("Bias analysis skipped: column not found")
        return

    groups = df[group_col].dropna().unique()

    print("\nBias Analysis Results:")

    for g in groups:

        subset = df[df[group_col] == g]

        # Skip small or invalid groups
        if len(subset) == 0:
            continue

        if len(subset[y_true_col].unique()) < 2:
            print(f"{group_col}: {g} → Not enough class variation")
            continue

        auc = roc_auc_score(subset[y_true_col], subset[pred_col])

        print(f"{group_col}: {g} → AUC: {round(auc,3)}")
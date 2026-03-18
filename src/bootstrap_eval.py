import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc(y_true, pred, n_iterations=1000, random_state=42):
    """
    Compute bootstrapped 95% confidence interval for AUC.

    Parameters:
    y_true : array-like
        True labels
    pred : array-like
        Predicted probabilities
    n_iterations : int
        Number of bootstrap samples
    random_state : int
        Random seed for reproducibility

    Returns:
    ci : tuple
        (lower_bound, upper_bound)
    """

    np.random.seed(random_state)

    # Convert to numpy arrays (important for safe indexing)
    y_true = np.array(y_true)
    pred = np.array(pred)

    scores = []

    for i in range(n_iterations):

        # Sample with replacement
        idx = np.random.choice(len(y_true), len(y_true), replace=True)

        y_sample = y_true[idx]
        pred_sample = pred[idx]

        # Skip invalid samples (only one class present)
        if len(np.unique(y_sample)) < 2:
            continue

        score = roc_auc_score(y_sample, pred_sample)

        scores.append(score)

    # Handle edge case (no valid scores)
    if len(scores) == 0:
        print("Warning: No valid bootstrap samples.")
        return (0.0, 0.0)

    # Compute 95% CI
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return (lower, upper)
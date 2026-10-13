"""Classification evaluation for exported distracted-driving predictions."""

from sklearn.metrics import accuracy_score, classification_report


def evaluate_predictions(y_true, y_pred):
    """Return accuracy and a per-class text report.

    The labels can be strings or integers. Both sequences must contain the
    same positive number of items.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Expected non-empty ground truth and predictions of equal length.")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        ),
    }

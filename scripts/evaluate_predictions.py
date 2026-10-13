"""CLI for evaluating class-label predictions saved as CSV."""

import argparse
import csv
from pathlib import Path

from distracted_driving.metrics import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a CSV containing y_true and y_pred columns."
    )
    parser.add_argument("csv_path", type=Path, help="CSV containing ground truth and predictions")
    args = parser.parse_args()

    with args.csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"y_true", "y_pred"}.issubset(reader.fieldnames or []):
            parser.error("CSV must have y_true and y_pred columns")
        rows = list(reader)

    result = evaluate_predictions(
        [row["y_true"] for row in rows],
        [row["y_pred"] for row in rows],
    )
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(result["classification_report"])


if __name__ == "__main__":
    main()

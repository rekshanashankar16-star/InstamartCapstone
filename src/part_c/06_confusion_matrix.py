
"""06_confusion_matrix.py
Evaluate few-shot ticket classification using a confusion matrix.
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "outputs" / "classification_comparison.csv"
MATRIX_PATH = BASE_DIR / "outputs" / "confusion_matrix.csv"
REPORT_PATH = BASE_DIR / "outputs" / "classification_report.txt"

CATEGORIES = ["billing", "delivery", "fraud", "technical"]


def main():
    df = pd.read_csv(INPUT_PATH, keep_default_na=False)

    # Evaluate only successfully classified tickets
    evaluated = df[
        df["few_shot_status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ].copy()

    # Keep valid category predictions
    evaluated = evaluated[
        evaluated["true_category"].isin(CATEGORIES)
        & evaluated["few_shot_category"].isin(CATEGORIES)
    ]

    y_true = evaluated["true_category"]
    y_pred = evaluated["few_shot_category"]

    # Generate confusion matrix
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CATEGORIES
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CATEGORIES,
        columns=CATEGORIES
    )

    matrix_df.index.name = "Actual Category"
    matrix_df.columns.name = "Predicted Category"

    matrix_df.to_csv(MATRIX_PATH)

    # Generate classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=CATEGORIES,
        zero_division=0
    )

    accuracy = (y_true == y_pred).mean()

    print("\nFEW-SHOT CLASSIFICATION EVALUATION")
    print("Evaluated tickets:", len(evaluated))
    print(f"Accuracy: {accuracy:.2%}")

    print("\nCONFUSION MATRIX")
    print(matrix_df)

    print("\nCLASSIFICATION REPORT")
    print(report)

    REPORT_PATH.write_text(
        "Few-Shot Classification Evaluation\n\n"
        f"Evaluated tickets: {len(evaluated)}\n"
        f"Accuracy: {accuracy:.2%}\n\n"
        "Confusion Matrix:\n"
        f"{matrix_df.to_string()}\n\n"
        "Classification Report:\n"
        f"{report}",
        encoding="utf-8"
    )

    print("\nSaved confusion matrix:", MATRIX_PATH)
    print("Saved classification report:", REPORT_PATH)


if __name__ == "__main__":
    main()

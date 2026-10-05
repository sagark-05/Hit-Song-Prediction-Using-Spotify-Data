from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "final_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "confusion_matrix.png"

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def main() -> None:
    final_df = pd.read_csv(DATA_PATH)
    X = final_df[AUDIO_FEATURES]
    y = final_df["hit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    final_model = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )

    final_model.fit(X_train, y_train)
    final_predictions = final_model.predict(X_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        final_predictions,
        labels=[0, 1],
        display_labels=["NON-HIT (0)", "HIT (1)"],
        cmap="Blues",
        values_format="d",
    )
    plt.title("Final Tuned Decision Tree - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print("Confusion matrix saved successfully!")
    print(f"Output path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

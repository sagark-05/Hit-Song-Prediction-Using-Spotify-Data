from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "final_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "decision_tree_final.pkl"

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

    X_train, _, y_train, _ = train_test_split(
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    print("Final Decision Tree model saved successfully.")
    print(f"Saved file: {MODEL_PATH}")


if __name__ == "__main__":
    main()

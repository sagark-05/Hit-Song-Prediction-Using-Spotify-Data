from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


TUNED_RESULTS = pd.DataFrame(
    [
        ["Logistic Regression", 0.6445, 0.5574, 0.5400, 0.5486],
        ["Decision Tree", 0.6195, 0.5157, 0.8013, 0.6275],
        ["SVM", 0.6390, 0.5423, 0.6250, 0.5807],
        ["KNN", 0.6190, 0.5239, 0.5213, 0.5226],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-score"],
)


BASELINE_VS_TUNED = pd.DataFrame(
    [
        ["Logistic Regression", 0.5486, 0.5486],
        ["Decision Tree", 0.5059, 0.6275],
        ["SVM", 0.5564, 0.5807],
        ["KNN", 0.5484, 0.5226],
    ],
    columns=["Model", "Baseline", "Tuned"],
)


def export_model_comparison() -> Path:
    output_path = RESULTS_DIR / "model_comparison.csv"
    TUNED_RESULTS.to_csv(output_path, index=False)
    return output_path


def export_f1_comparison_chart() -> Path:
    output_path = RESULTS_DIR / "baseline_vs_tuned_f1.png"
    positions = np.arange(len(BASELINE_VS_TUNED))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    baseline_bars = ax.bar(
        positions - bar_width / 2,
        BASELINE_VS_TUNED["Baseline"],
        bar_width,
        label="Baseline",
        color="steelblue",
    )
    tuned_bars = ax.bar(
        positions + bar_width / 2,
        BASELINE_VS_TUNED["Tuned"],
        bar_width,
        label="Tuned",
        color="darkorange",
    )

    ax.set_title("Baseline vs Tuned F1-score")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1-score")
    ax.set_xticks(positions)
    ax.set_xticklabels(BASELINE_VS_TUNED["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 0.75)
    ax.legend()

    for bars in (baseline_bars, tuned_bars):
        ax.bar_label(bars, fmt="%.4f", padding=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def report_confusion_matrix_unavailable() -> None:
    print(
        "Confusion matrix was not created: the existing final test predictions "
        "from model_training.ipynb are not available to this standalone script. "
        "No values were invented."
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    created_paths = [
        export_model_comparison(),
        export_f1_comparison_chart(),
    ]

    for path in created_paths:
        print(f"Created: {path}")

    report_confusion_matrix_unavailable()


if __name__ == "__main__":
    main()

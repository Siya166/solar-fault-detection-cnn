"""Report test-set performance from the predictions saved by training.

Usage:  python -m solarfault.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)

from . import CLASSES, NO_ANOMALY

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def binary_view(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Collapse to "anomaly or not". For a maintenance crew, the first question is which
    modules need a visit at all, so recall on anomalies (faults found) matters most."""
    t, p = y_true != NO_ANOMALY, y_pred != NO_ANOMALY
    return {
        "anomaly_recall": recall_score(t, p),
        "anomaly_precision": precision_score(t, p),
        "anomaly_f1": f1_score(t, p),
        "accuracy": accuracy_score(t, p),
    }


def bootstrap_ci(y_true, y_pred, metric, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric(y_true[idx], y_pred[idx]))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def main() -> None:
    data = np.load(RESULTS / "test_predictions.npz")
    y, pred = data["y_true"], data["y_pred"]
    labels = list(range(len(CLASSES)))

    report = classification_report(y, pred, labels=labels, target_names=CLASSES, digits=3)
    macro = lambda a, b: f1_score(a, b, average="macro", labels=labels, zero_division=0)  # noqa: E731
    acc_ci = bootstrap_ci(y, pred, accuracy_score)
    f1_ci = bootstrap_ci(y, pred, macro)
    binary = binary_view(y, pred)
    summary = {
        "test_images": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "accuracy_ci_95": acc_ci,
        "macro_f1": macro(y, pred),
        "macro_f1_ci_95": f1_ci,
        "anomaly_vs_normal": binary,
    }
    (RESULTS / "test_metrics.json").write_text(json.dumps(summary, indent=2))
    (RESULTS / "classification_report.txt").write_text(report)
    print(report)
    print(json.dumps(summary, indent=2))

    figures = RESULTS / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=labels, display_labels=CLASSES,
                                            normalize="true", values_format=".2f", cmap="Blues",
                                            colorbar=False, ax=ax, xticks_rotation=45)
    ax.set_title("Test set confusion matrix (each row sums to 1 = recall per class)")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    history = json.loads((RESULTS / "training_history.json").read_text())
    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    epochs = [h["epoch"] for h in history]
    ax1.plot(epochs, [h["train_loss"] for h in history], color="#2f6fb3", label="Training loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="#2f6fb3")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_macro_f1"] for h in history], color="#d9822b", label="Validation macro F1")
    ax2.set_ylabel("Validation macro F1", color="#d9822b")
    ax1.set_title("Training curve")
    fig.tight_layout()
    fig.savefig(figures / "training_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()

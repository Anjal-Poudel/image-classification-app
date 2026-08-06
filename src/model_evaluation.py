"""
Stage 5: Model Evaluation
============================
Loads the trained model and the held-out test split, computes the metrics
requested in params.yaml -> model_evaluation.metrics, and optionally
renders a confusion matrix.

Outputs:
  - metrics/eval_metrics.json      DVC metric (accuracy/precision/recall/f1)
  - reports/confusion_matrix.png   if model_evaluation.confusion_matrix: true
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    me_params = load_params()["model_evaluation"]
    requested_metrics = me_params["metrics"]
    want_confusion_matrix = me_params["confusion_matrix"]

    bundle = joblib.load("models/model.joblib")
    model = bundle["model"]

    data = np.load("data/features/features.npz")
    X_test, y_test = data["X_test"], data["y_test"]

    y_pred = model.predict(X_test)

    available = {
        "accuracy": lambda: accuracy_score(y_test, y_pred),
        "precision": lambda: precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": lambda: recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": lambda: f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    results = {}
    for metric_name in requested_metrics:
        if metric_name not in available:
            raise ValueError(f"Unknown metric '{metric_name}' requested in params.yaml")
        results[metric_name] = float(available[metric_name]())

    results["n_test_samples"] = int(len(y_test))

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[model_evaluation] test metrics:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    if want_confusion_matrix:
        os.makedirs("reports", exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, "reports/confusion_matrix.png")
        print("[model_evaluation] confusion matrix saved to reports/confusion_matrix.png")

    print("[model_evaluation] metrics saved to metrics/eval_metrics.json")


if __name__ == "__main__":
    main()

"""
Stage 4: Model Training
==========================
Trains a multi-layer perceptron (see mlp.py) on the selected features
using the hyperparameters in params.yaml -> model_training.

Outputs:
  - models/model.joblib            the trained model
  - metrics/train_history.json     per-epoch loss/accuracy (a DVC metric)
  - reports/training_curves.png    loss & accuracy curves
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from mlp import NumpyMLP


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def plot_history(history: dict, out_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    mt_params = load_params()["model_training"]

    hidden_layers = mt_params["hidden_layers"]
    learning_rate = mt_params["learning_rate"]
    batch_size = mt_params["batch_size"]
    epochs = mt_params["epochs"]
    optimizer = mt_params["optimizer"]
    dropout_rate = mt_params["dropout_rate"]
    early_stopping_patience = mt_params["early_stopping_patience"]
    random_seed = mt_params["random_seed"]

    data = np.load("data/features/features.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    num_classes = int(max(y_train.max(), y_val.max()) + 1)
    input_dim = X_train.shape[1]

    print(f"[model_training] input_dim={input_dim} num_classes={num_classes} "
          f"hidden_layers={hidden_layers} optimizer={optimizer} lr={learning_rate}")

    model = NumpyMLP(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        optimizer=optimizer,
        learning_rate=learning_rate,
        random_seed=random_seed,
    )

    history = model.fit(
        X_train, y_train, X_val, y_val,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "num_classes": num_classes, "input_dim": input_dim}, "models/model.joblib")

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    os.makedirs("reports", exist_ok=True)
    plot_history(history, "reports/training_curves.png")

    print("[model_training] model saved to models/model.joblib")
    print("[model_training] history saved to metrics/train_history.json")
    print("[model_training] training curves saved to reports/training_curves.png")


if __name__ == "__main__":
    main()

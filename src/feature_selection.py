"""
Stage 3: Feature Selection
=============================
Flattens the processed images into feature vectors and reduces their
dimensionality before they reach the model. The selector is fit on the
training split ONLY (to avoid data leakage) and then applied to the
validation and test splits.

Supported methods (params.yaml -> feature_selection.method):
  - select_k_best:       ANOVA F-test, keep the top `num_features` pixels
  - pca:                 PCA, project down to `num_features` components
  - variance_threshold:  drop near-constant pixels below `variance_threshold`

Output: data/features/features.npz (+ metadata.json) and the fitted
selector saved to models/feature_selector.joblib so it can be reapplied
at inference time.
"""
import json
import os

import joblib
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def build_selector(method: str, num_features: int, variance_threshold: float):
    if method == "select_k_best":
        return SelectKBest(score_func=f_classif, k=num_features)
    if method == "pca":
        return PCA(n_components=num_features, random_state=42)
    if method == "variance_threshold":
        return VarianceThreshold(threshold=variance_threshold)
    raise ValueError(f"Unknown feature_selection.method='{method}'")


def main():
    fs_params = load_params()["feature_selection"]
    method = fs_params["method"]
    num_features = fs_params["num_features"]
    variance_threshold = fs_params["variance_threshold"]

    data = np.load("data/processed/dataset.npz")
    X_train, y_train = flatten(data["X_train"]), data["y_train"]
    X_val, y_val = flatten(data["X_val"]), data["y_val"]
    X_test, y_test = flatten(data["X_test"]), data["y_test"]

    print(f"[feature_selection] method='{method}' input_dim={X_train.shape[1]}")

    # PCA can request at most min(n_samples, n_features) components.
    if method == "pca":
        num_features = min(num_features, X_train.shape[0], X_train.shape[1])

    selector = build_selector(method, num_features, variance_threshold)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)

    print(f"[feature_selection] output_dim={X_train_sel.shape[1]}")

    os.makedirs("data/features", exist_ok=True)
    np.savez_compressed(
        "data/features/features.npz",
        X_train=X_train_sel.astype(np.float32), y_train=y_train,
        X_val=X_val_sel.astype(np.float32), y_val=y_val,
        X_test=X_test_sel.astype(np.float32), y_test=y_test,
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(selector, "models/feature_selector.joblib")

    metadata = {
        "method": method,
        "input_dim": int(X_train.shape[1]),
        "output_dim": int(X_train_sel.shape[1]),
    }
    with open("data/features/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("[feature_selection] saved to data/features/features.npz")
    print("[feature_selection] selector saved to models/feature_selector.joblib")


if __name__ == "__main__":
    main()

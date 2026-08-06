"""
Stage 2: Data Processing
==========================
Reads the raw image folder produced by data_collection.py, resizes and
normalizes every image, optionally augments the training split, and
produces a stratified train / validation / test split.

Output: data/processed/dataset.npz containing X_train, y_train, X_val,
y_val, X_test, y_test, plus data/processed/metadata.json with a summary.
"""
import glob
import json
import os

import numpy as np
import yaml
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_raw_images(raw_data_dir: str, image_size: int, grayscale: bool):
    """Walk data/raw/<label>/*.png, resize + convert every image."""
    mode = "L" if grayscale else "RGB"
    images, labels = [], []

    class_dirs = sorted(
        d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))
    )
    for class_label in class_dirs:
        for path in sorted(glob.glob(os.path.join(raw_data_dir, class_label, "*.png"))):
            img = Image.open(path).convert(mode).resize((image_size, image_size), Image.BILINEAR)
            images.append(np.asarray(img, dtype=np.float32))
            labels.append(int(class_label))

    return np.array(images), np.array(labels)


def augment_image(img_array: np.ndarray, rotation_range: float, horizontal_flip: bool,
                   zoom_range: float, rng: np.random.Generator) -> np.ndarray:
    """Apply one random augmentation (rotation + optional flip + zoom) to a
    single image, returning a new array of the same shape."""
    mode = "L" if img_array.ndim == 2 else "RGB"
    img = Image.fromarray(img_array.astype(np.uint8), mode=mode)

    angle = rng.uniform(-rotation_range, rotation_range)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    if horizontal_flip and rng.random() < 0.5:
        img = ImageOps.mirror(img)

    if zoom_range > 0:
        scale = 1.0 + rng.uniform(-zoom_range, zoom_range)
        w, h = img.size
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        # center-crop / pad back to the original size
        canvas = Image.new(mode, (w, h), color=0)
        paste_x, paste_y = (w - new_w) // 2, (h - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas

    return np.asarray(img, dtype=np.float32)


def augment_split(X: np.ndarray, y: np.ndarray, aug_cfg: dict, random_seed: int):
    """Return X, y extended with one augmented copy of every training image."""
    rng = np.random.default_rng(random_seed)
    augmented = np.array([
        augment_image(img, aug_cfg["rotation_range"], aug_cfg["horizontal_flip"],
                       aug_cfg["zoom_range"], rng)
        for img in X
    ])
    X_out = np.concatenate([X, augmented], axis=0)
    y_out = np.concatenate([y, y], axis=0)
    return X_out, y_out


def main():
    all_params = load_params()
    dc_params = all_params["data_collection"]
    dp_params = all_params["data_processing"]

    raw_data_dir = dc_params["raw_data_dir"]
    random_seed = dc_params["random_seed"]

    image_size = dp_params["image_size"]
    grayscale = dp_params["grayscale"]
    normalize = dp_params["normalize"]
    test_split = dp_params["test_split"]
    val_split = dp_params["val_split"]
    aug_cfg = dp_params["augmentation"]

    print(f"[data_processing] loading raw images from '{raw_data_dir}' ...")
    X, y = load_raw_images(raw_data_dir, image_size, grayscale)
    print(f"[data_processing] loaded {len(X)} images, shape={X.shape[1:]}")

    # train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=random_seed
    )
    # train vs val (val_split is a fraction of the remaining train+val pool)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_split, stratify=y_trainval, random_state=random_seed
    )

    if aug_cfg.get("enabled", False):
        before = len(X_train)
        X_train, y_train = augment_split(X_train, y_train, aug_cfg, random_seed)
        print(f"[data_processing] augmentation enabled: train set {before} -> {len(X_train)} images")

    if normalize:
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed(
        "data/processed/dataset.npz",
        X_train=X_train.astype(np.float32), y_train=y_train,
        X_val=X_val.astype(np.float32), y_val=y_val,
        X_test=X_test.astype(np.float32), y_test=y_test,
    )

    metadata = {
        "image_size": image_size,
        "grayscale": grayscale,
        "normalized": normalize,
        "augmentation_enabled": aug_cfg.get("enabled", False),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "n_classes": int(len(np.unique(y))),
    }
    with open("data/processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[data_processing] train={metadata['n_train']} val={metadata['n_val']} test={metadata['n_test']}")
    print("[data_processing] saved to data/processed/dataset.npz")


if __name__ == "__main__":
    main()

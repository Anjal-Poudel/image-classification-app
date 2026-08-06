"""
Stage 1: Data Collection
=========================
Pulls the source image dataset and materializes it on disk as a standard
image-classification folder layout:

    data/raw/<class_label>/img_0000.png
    data/raw/<class_label>/img_0001.png
    ...

Using scikit-learn's bundled "digits" dataset (1,797 8x8 grayscale images of
handwritten digits 0-9) means this stage runs fully offline, in seconds, on
a CPU-only laptop -- no large download, no GPU. Because the output is a real
folder-of-images layout, every downstream stage (processing, feature
selection, training, evaluation) works exactly the way it would against any
other image dataset. Swapping in a different source dataset later only
requires editing this one file.
"""
import os
import shutil

import numpy as np
import yaml
from PIL import Image
from sklearn.datasets import load_digits


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collect_sklearn_digits(raw_data_dir: str, random_seed: int) -> None:
    """Load sklearn's digits dataset and dump each sample as a PNG file,
    organized into one sub-folder per class label."""
    np.random.seed(random_seed)

    digits = load_digits()
    images, labels = digits.images, digits.target  # images: (N, 8, 8), values 0-16

    if os.path.exists(raw_data_dir):
        shutil.rmtree(raw_data_dir)
    os.makedirs(raw_data_dir, exist_ok=True)

    counts = {}
    for idx, (img, label) in enumerate(zip(images, labels)):
        class_dir = os.path.join(raw_data_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        # Scale the 0-16 pixel range up to standard 0-255 8-bit images.
        img_uint8 = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode="L")

        counts[label] = counts.get(label, 0)
        pil_img.save(os.path.join(class_dir, f"img_{counts[label]:04d}.png"))
        counts[label] += 1

    total = sum(counts.values())
    print(f"[data_collection] dataset='sklearn_digits' -> {total} images across {len(counts)} classes")
    for label in sorted(counts):
        print(f"  class {label}: {counts[label]} images")
    print(f"[data_collection] saved to '{raw_data_dir}'")


def main():
    params = load_params()["data_collection"]

    dataset = params["dataset"]
    raw_data_dir = params["raw_data_dir"]
    random_seed = params["random_seed"]

    if dataset == "sklearn_digits":
        collect_sklearn_digits(raw_data_dir, random_seed)
    else:
        raise ValueError(
            f"Unknown data_collection.dataset='{dataset}'. "
            "Add a new collector function in data_collection.py to support it."
        )


if __name__ == "__main__":
    main()

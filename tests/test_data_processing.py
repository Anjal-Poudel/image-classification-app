from pathlib import Path

import numpy as np
from PIL import Image

from data_processing import augment_image, augment_split, load_raw_images


def _save_gray_png(path: Path, value: int):
    arr = np.full((8, 8), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_load_raw_images_resizes_and_reads_labels(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "0").mkdir(parents=True)
    (raw / "1").mkdir(parents=True)

    _save_gray_png(raw / "0" / "a.png", 20)
    _save_gray_png(raw / "0" / "b.png", 40)
    _save_gray_png(raw / "1" / "c.png", 180)

    X, y = load_raw_images(str(raw), image_size=16, grayscale=True)

    assert X.shape == (3, 16, 16)
    assert y.shape == (3,)
    assert set(np.unique(y)) == {0, 1}


def test_augment_image_preserves_shape_and_returns_float32():
    rng = np.random.default_rng(7)
    img = np.random.default_rng(1).integers(0, 256, size=(32, 32), dtype=np.uint8)

    aug = augment_image(
        img,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.2,
        rng=rng,
    )

    assert aug.shape == img.shape
    assert aug.dtype == np.float32


def test_augment_split_doubles_samples_and_labels():
    X = np.random.default_rng(0).integers(0, 256, size=(5, 16, 16), dtype=np.uint8).astype(np.float32)
    y = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    aug_cfg = {
        "rotation_range": 10,
        "horizontal_flip": True,
        "zoom_range": 0.1,
    }

    X_out, y_out = augment_split(X, y, aug_cfg, random_seed=123)

    assert X_out.shape[0] == 10
    assert y_out.shape[0] == 10
    np.testing.assert_array_equal(y_out[: len(y)], y)
    np.testing.assert_array_equal(y_out[len(y):], y)
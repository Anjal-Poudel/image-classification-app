from pathlib import Path
import sys

import numpy as np
import pytest


# Ensure tests can import modules from src/ directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def tiny_classification_data():
    """Small, deterministic dataset for fast MLP unit tests."""
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.0],
            [0.0, 0.9],
            [0.9, 0.1],
            [1.0, 0.8],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 1, 1, 1, 0, 1, 1, 1], dtype=np.int64)

    X_val = np.array(
        [
            [0.0, 0.2],
            [0.2, 0.0],
            [0.8, 0.9],
            [0.9, 0.2],
        ],
        dtype=np.float32,
    )
    y_val = np.array([0, 0, 1, 1], dtype=np.int64)

    return X_train, y_train, X_val, y_val
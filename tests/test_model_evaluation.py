from pathlib import Path

import numpy as np
import yaml

from model_evaluation import load_params, plot_confusion_matrix


def test_load_params_reads_yaml(tmp_path: Path):
    cfg = {
        "model_evaluation": {
            "metrics": ["accuracy", "f1"],
            "confusion_matrix": True,
        }
    }
    cfg_path = tmp_path / "params.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    loaded = load_params(str(cfg_path))
    assert loaded["model_evaluation"]["metrics"] == ["accuracy", "f1"]


def test_plot_confusion_matrix_creates_png(tmp_path: Path):
    cm = np.array([[8, 1], [2, 7]], dtype=np.int64)
    out = tmp_path / "confusion_matrix.png"

    plot_confusion_matrix(cm, str(out))

    assert out.exists()
    assert out.stat().st_size > 0
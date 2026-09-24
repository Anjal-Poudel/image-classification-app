from pathlib import Path

import yaml

from model_training import load_params, plot_history


def test_load_params_reads_yaml(tmp_path: Path):
    cfg = {
        "model_training": {
            "hidden_layers": [16, 8],
            "learning_rate": 0.001,
        }
    }
    cfg_path = tmp_path / "params.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    loaded = load_params(str(cfg_path))
    assert loaded["model_training"]["hidden_layers"] == [16, 8]


def test_plot_history_creates_png_file(tmp_path: Path):
    history = {
        "train_loss": [1.2, 1.0, 0.8],
        "train_acc": [0.4, 0.6, 0.75],
        "val_loss": [1.3, 1.1, 0.9],
        "val_acc": [0.35, 0.55, 0.7],
    }
    out = tmp_path / "training_curves.png"

    plot_history(history, str(out))

    assert out.exists()
    assert out.stat().st_size > 0
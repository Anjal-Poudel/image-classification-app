from pathlib import Path

import yaml

from data_collection import collect_sklearn_digits, load_params


def test_load_params_reads_yaml(tmp_path: Path):
    cfg = {
        "data_collection": {
            "dataset": "sklearn_digits",
            "raw_data_dir": "data/raw",
            "random_seed": 123,
        }
    }
    cfg_path = tmp_path / "params.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    loaded = load_params(str(cfg_path))
    assert loaded["data_collection"]["dataset"] == "sklearn_digits"
    assert loaded["data_collection"]["random_seed"] == 123


def test_collect_sklearn_digits_creates_class_folders_and_pngs(tmp_path: Path):
    raw_dir = tmp_path / "raw"

    collect_sklearn_digits(str(raw_dir), random_seed=42)

    class_dirs = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    assert class_dirs == [str(i) for i in range(10)]

    png_count = len(list(raw_dir.rglob("*.png")))
    assert png_count > 1000

import pandas as pd
import numpy as np
from src.preprocess import clean_dataset


def test_preprocessing_removes_unnecessary_columns(tmp_path):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    df = pd.DataFrame({
        "id": [1, 2],
        "attack_cat": ["Normal", "Fuzzers"],
        "dur": [0.5, 1.2],
        "proto": ["tcp", "udp"],
        "label": [0, 1]
    })

    df.to_csv(input_file, index=False)

    clean_dataset(str(input_file), str(output_file))

    result = pd.read_csv(output_file)

    assert "id" not in result.columns
    assert "attack_cat" not in result.columns
    assert "label" in result.columns


def test_preprocessing_preserves_labels(tmp_path):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    df = pd.DataFrame({
        "dur": [0.5, 1.2],
        "proto": ["tcp", "udp"],
        "label": [0, 1]
    })

    df.to_csv(input_file, index=False)

    clean_dataset(str(input_file), str(output_file))

    result = pd.read_csv(output_file)

    assert result["label"].tolist() == [0, 1]


def test_preprocessing_replaces_infinity(tmp_path):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    df = pd.DataFrame({
        "dur": [0.5, np.inf],
        "proto": ["tcp", "udp"],
        "label": [0, 1]
    })

    df.to_csv(input_file, index=False)

    clean_dataset(str(input_file), str(output_file))

    result = pd.read_csv(output_file)

    assert result["dur"].isna().sum() == 1

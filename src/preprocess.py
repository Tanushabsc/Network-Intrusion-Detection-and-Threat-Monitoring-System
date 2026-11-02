
import os
import numpy as np
import pandas as pd

TRAIN_FILE = "data/raw/UNSW_NB15_training-set.csv"
TEST_FILE = "data/raw/UNSW_NB15_testing-set.csv"

OUTPUT_DIR = "data/processed"


def clean_dataset(input_file, output_file):
    print(f"Reading: {input_file}")

    df = pd.read_csv(input_file)

    # Remove columns that should not be used as model features
    columns_to_drop = ["id", "attack_cat"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Replace infinite values with missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Check that the target column exists
    if "label" not in df.columns:
        raise ValueError("The dataset does not contain the label column.")

    # Convert target to integer
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    # Validate target labels
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected values found in label column.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Rows: {len(df)}")
    print(f"Benign records: {(df['label'] == 0).sum()}")
    print(f"Attack records: {(df['label'] == 1).sum()}")


if __name__ == "__main__":
    clean_dataset(
        TRAIN_FILE,
        f"{OUTPUT_DIR}/train.csv"
    )

    clean_dataset(
        TEST_FILE,
        f"{OUTPUT_DIR}/test.csv"
    )

    print("\nPreprocessing completed successfully!")

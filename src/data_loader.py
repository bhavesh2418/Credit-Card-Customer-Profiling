"""
data_loader.py — Load and validate the raw credit card dataset.
"""

import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_FILE, ID_COLUMN


def load_raw_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load raw CSV and run basic validation checks."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            "Run:  python scripts/download_data.py"
        )
    df = pd.read_csv(filepath)
    _validate(df)
    return df


def _validate(df: pd.DataFrame):
    assert len(df) > 0, "Dataset is empty"
    missing = df.isnull().sum()
    print(f"Loaded  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing : {missing[missing > 0].to_dict()}")
    print(f"Dupes   : {df.duplicated().sum()}")

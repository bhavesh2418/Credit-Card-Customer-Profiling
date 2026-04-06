"""
download_data.py — Download Credit Card dataset from Kaggle API.
Dataset: https://www.kaggle.com/datasets/arjunbhasin2013/ccdata
"""

import os, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"


def download():
    username = os.getenv("KAGGLE_USERNAME")
    key      = os.getenv("KAGGLE_KEY")
    if not username or not key:
        print("ERROR: Set KAGGLE_USERNAME and KAGGLE_KEY in .env")
        sys.exit(1)
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"]      = key
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    import kaggle
    kaggle.api.authenticate()
    print("Downloading dataset from Kaggle...")
    kaggle.api.dataset_download_files("arjunbhasin2013/ccdata", path=str(DATA_RAW), unzip=True)

    for f in DATA_RAW.glob("*.csv"):
        print(f"Downloaded: {f.name}  ({f.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    download()
